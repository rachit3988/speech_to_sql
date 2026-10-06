"""Static validation of model-generated SQL before it reaches the database.

This is one of three independent layers (with the read-only DB role and the
read-only, time-limited transaction). It parses the statement with sqlglot and
rejects anything that is not a single, side-effect-free SELECT over tables in the
catalog, then executes the *re-generated* SQL so what runs is exactly what was checked.

Error messages are written for the agent: they say what was wrong and how to fix it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from speech_to_sql.db.catalog import SchemaCatalog, TableInfo
from speech_to_sql.errors import UnsafeSQLError

logging.getLogger("sqlglot").setLevel(logging.ERROR)


def _node_types(*names: str) -> tuple[type[exp.Expr], ...]:
    # Resolve by name so the guard keeps working if sqlglot renames/removes a class.
    return tuple(getattr(exp, n) for n in names if isinstance(getattr(exp, n, None), type))


_WRITE_NODES = _node_types(
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Create",
    "Drop",
    "Alter",
    "TruncateTable",
    "Copy",
    "Grant",
    "Revoke",
    "Command",
    "Set",
    "Use",
    "Transaction",
    "Commit",
    "Rollback",
    "LoadData",
    "Pragma",
    "Attach",
    "Detach",
    "Analyze",
    "Refresh",
    "Kill",
    "Install",
    "Execute",
)
_INTO_NODES = _node_types("Into")
_LOCK_NODES = _node_types("Lock")

# Functions with side effects, filesystem/network access, or that execute nested SQL.
_DENIED_FUNCTIONS = frozenset(
    {
        "set_config",
        "current_setting",
        "nextval",
        "setval",
        "currval",
        "lastval",
        "query_to_xml",
        "query_to_xmlschema",
        "query_to_xml_and_xmlschema",
        "cursor_to_xml",
        "cursor_to_xmlschema",
        "table_to_xml",
        "table_to_xmlschema",
        "table_to_xml_and_xmlschema",
        "schema_to_xml",
        "database_to_xml",
        "txid_current",
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
    }
)
_DENIED_PREFIXES = ("pg_", "lo_", "dblink", "xpath")
_MAX_SQL_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class ValidatedSQL:
    sql: str
    tables: tuple[str, ...]
    limit_applied: bool


def _function_name(func: exp.Func) -> str:
    if isinstance(func, exp.Anonymous):
        return str(func.name).lower()
    return func.sql_name().lower()


class SQLGuard:
    def __init__(self, dialect: str, max_rows: int) -> None:
        self.dialect = dialect
        # One extra row lets the executor detect truncation.
        self.hard_limit = max_rows + 1

    def validate(self, sql: str, catalog: SchemaCatalog) -> ValidatedSQL:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise UnsafeSQLError("Empty SQL statement.", "empty")
        if len(sql) > _MAX_SQL_CHARS:
            raise UnsafeSQLError(f"SQL is longer than {_MAX_SQL_CHARS} characters.", "too_long")

        try:
            statements = [s for s in sqlglot.parse(sql, dialect=self.dialect) if s is not None]
        except (ParseError, TokenError) as exc:
            raise UnsafeSQLError(f"Could not parse SQL ({self.dialect}): {str(exc)[:300]}", "parse_error") from exc
        if len(statements) != 1:
            raise UnsafeSQLError("Submit exactly one SQL statement.", "multiple_statements")
        root = statements[0]
        if not isinstance(root, exp.Query):
            raise UnsafeSQLError(
                "Only read-only SELECT queries (optionally with WITH/UNION) are allowed.",
                "not_select",
            )

        self._check_nodes(root)
        self._check_functions(root)
        tables = self._check_tables(root, catalog)
        self._check_hidden_columns(root, tables, catalog)
        root, limit_applied = self._enforce_limit(root)

        return ValidatedSQL(
            sql=root.sql(dialect=self.dialect),
            tables=tuple(sorted({t.qualified_name for t in tables.values()})),
            limit_applied=limit_applied,
        )

    # --- checks ----------------------------------------------------------------

    def _check_nodes(self, root: exp.Expr) -> None:
        for node in root.walk():
            if isinstance(node, _WRITE_NODES):
                raise UnsafeSQLError(
                    f"Statement contains a forbidden {node.key.upper()} operation; "
                    "only read-only SELECT queries are allowed.",
                    "write_operation",
                )
            if isinstance(node, _INTO_NODES):
                raise UnsafeSQLError("SELECT ... INTO is not allowed.", "select_into")
            if isinstance(node, _LOCK_NODES):
                raise UnsafeSQLError("Row locking clauses (FOR UPDATE/SHARE) are not allowed.", "locking")

    def _check_functions(self, root: exp.Expr) -> None:
        for func in root.find_all(exp.Func):
            name = _function_name(func)
            if name in _DENIED_FUNCTIONS or name.startswith(_DENIED_PREFIXES):
                raise UnsafeSQLError(f"Function {name}() is not allowed.", "forbidden_function")
            if isinstance(func.parent, exp.Dot) and func.parent.expression is func:
                raise UnsafeSQLError(
                    f"Schema-qualified function calls are not allowed ({func.parent.sql()}).",
                    "qualified_function",
                )

    def _check_tables(self, root: exp.Expr, catalog: SchemaCatalog) -> dict[str, TableInfo]:
        """Resolve every table reference; return alias/name (lower-case) -> table."""
        cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
        referenced: dict[str, TableInfo] = {}
        for table in root.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                continue  # table-valued function, e.g. generate_series(); vetted as a function
            if table.catalog:
                raise UnsafeSQLError("Cross-database references are not allowed.", "cross_database")
            name, schema = table.name, table.db or None
            if schema is None and name.lower() in cte_names:
                continue
            info = catalog.resolve(name, schema)
            if info is None:
                shown = ", ".join(catalog.table_names()[:50])
                raise UnsafeSQLError(
                    f"Unknown or inaccessible table {table.sql(dialect=self.dialect)}. Available tables: {shown}",
                    "unknown_table",
                )
            referenced[name.lower()] = info
            if table.alias:
                referenced[table.alias.lower()] = info
        return referenced

    def _check_hidden_columns(self, root: exp.Expr, tables: dict[str, TableInfo], catalog: SchemaCatalog) -> None:
        restricted = {k: catalog.hidden_columns_for(t) for k, t in tables.items()}
        restricted = {k: v for k, v in restricted.items() if v}
        if not restricted:
            return
        hidden_any = frozenset().union(*restricted.values())
        visible_any = {c.name.lower() for t in tables.values() for c in t.columns}

        for star in root.find_all(exp.Star):
            parent = star.parent
            if isinstance(parent, exp.Column) and parent.table:
                if parent.table.lower() in restricted:
                    raise UnsafeSQLError(
                        f"{parent.table}.* would expose restricted columns; list columns explicitly.",
                        "restricted_column",
                    )
            elif isinstance(parent, exp.Select):
                raise UnsafeSQLError(
                    "SELECT * would expose restricted columns; list the columns you need.",
                    "restricted_column",
                )

        for column in root.find_all(exp.Column):
            if isinstance(column.this, exp.Star):
                continue
            name = column.name.lower()
            qualifier = column.table.lower() if column.table else None
            if qualifier is not None:
                if name in restricted.get(qualifier, frozenset()):
                    raise UnsafeSQLError(f"Column {column.sql()} is restricted.", "restricted_column")
            elif name in hidden_any:
                raise UnsafeSQLError(f"Column {name} is restricted.", "restricted_column")
            elif name in restricted and name not in visible_any:
                # Whole-row reference such as `SELECT c FROM customers c` or row_to_json(c).
                raise UnsafeSQLError(
                    f"Whole-row reference to {name} would expose restricted columns.",
                    "restricted_column",
                )

    def _enforce_limit(self, root: exp.Query) -> tuple[exp.Query, bool]:
        limit = root.args.get("limit")
        value = None
        if isinstance(limit, exp.Limit):
            value = limit.expression
        elif isinstance(limit, exp.Fetch):  # FETCH FIRST n ROWS ONLY
            value = limit.args.get("count")
        if isinstance(value, exp.Literal) and value.is_int and int(value.name) <= self.hard_limit:
            return root, False
        if isinstance(root, (exp.Select, exp.SetOperation)):
            return root.limit(self.hard_limit), True
        wrapped = exp.select("*").from_(root.subquery("_q")).limit(self.hard_limit)
        return wrapped, True
