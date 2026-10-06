"""PostgreSQL adapter (asyncpg).

Safety at this layer, independent of the SQL guard:
  * every connection defaults to ``default_transaction_read_only = on``;
  * every query runs in an explicit READ ONLY transaction with ``SET LOCAL statement_timeout``;
  * introspection only exposes tables/columns the connected role can SELECT, so
    column-level GRANTs on the database are reflected in what the agent sees.
Connect with a dedicated read-only role (see deploy/postgres/readonly_role.sql).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

import asyncpg
import structlog

from speech_to_sql.db.base import QueryResult, rank_matches, search_words, to_jsonable
from speech_to_sql.db.catalog import ColumnInfo, ForeignKey, SchemaCatalog, TableInfo
from speech_to_sql.errors import DatabaseUnavailableError, QueryExecutionError, QueryTimeoutError

log = structlog.get_logger(__name__)

_TABLES_SQL = """
SELECT c.oid, n.nspname AS schema, c.relname AS name,
       obj_description(c.oid, 'pg_class') AS comment,
       c.reltuples::bigint AS row_estimate
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'v', 'm', 'p', 'f')
  AND NOT c.relispartition
  AND n.nspname = ANY($1::text[])
  AND has_any_column_privilege(c.oid, 'SELECT')
ORDER BY n.nspname, c.relname
"""

_COLUMNS_SQL = """
SELECT a.attrelid, a.attname, format_type(a.atttypid, a.atttypmod) AS data_type,
       NOT a.attnotnull AS nullable, col_description(a.attrelid, a.attnum) AS comment
FROM pg_attribute a
WHERE a.attrelid = ANY($1::oid[]) AND a.attnum > 0 AND NOT a.attisdropped
  AND has_column_privilege(a.attrelid, a.attnum, 'SELECT')
ORDER BY a.attrelid, a.attnum
"""

_CONSTRAINTS_SQL = """
SELECT con.conrelid, con.contype::text AS contype,
       ARRAY(SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum
             ORDER BY k.ord) AS cols,
       rn.nspname AS ref_schema, rc.relname AS ref_table,
       ARRAY(SELECT a.attname FROM unnest(con.confkey) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.attnum
             ORDER BY k.ord) AS ref_cols
FROM pg_constraint con
LEFT JOIN pg_class rc ON rc.oid = con.confrelid
LEFT JOIN pg_namespace rn ON rn.oid = rc.relnamespace
WHERE con.conrelid = ANY($1::oid[]) AND con.contype IN ('p', 'f')
ORDER BY con.conrelid, con.conname
"""


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class PostgresDatabase:
    dialect = "postgres"

    def __init__(
        self,
        dsn: str,
        *,
        schemas: list[str],
        hidden_tables: list[str],
        hidden_columns: list[str],
        min_size: int = 1,
        max_size: int = 10,
        default_timeout_s: float = 10.0,
        timezone: str = "UTC",
    ) -> None:
        self._dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._schemas = schemas or ["public"]
        self._hidden_tables = hidden_tables
        self._hidden_columns = hidden_columns
        self._min_size = min_size
        self._max_size = max_size
        self._default_timeout_ms = int(default_timeout_s * 1000)
        # Pin the session time zone so date_trunc/now() results don't depend on server config.
        self._timezone = timezone
        self._pool: asyncpg.Pool | None = None
        self._has_trgm = False

    async def connect(self) -> None:
        try:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                command_timeout=self._default_timeout_ms / 1000 + 5,
                server_settings={
                    "application_name": "speech-to-sql",
                    "default_transaction_read_only": "on",
                    "statement_timeout": str(self._default_timeout_ms),
                    "idle_in_transaction_session_timeout": "30000",
                    "timezone": self._timezone,
                },
            )
            async with self._pool.acquire() as conn:
                self._has_trgm = bool(await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'"))
        except (OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
            raise DatabaseUnavailableError(f"cannot connect to Postgres: {exc}") from exc
        log.info("postgres_connected", pg_trgm=self._has_trgm)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def _p(self) -> asyncpg.Pool:
        if self._pool is None:
            raise DatabaseUnavailableError("database pool is not initialised")
        return self._pool

    async def ping(self) -> None:
        try:
            async with self._p.acquire(timeout=5) as conn:
                await conn.fetchval("SELECT 1")
        except (TimeoutError, OSError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
            raise DatabaseUnavailableError(f"database ping failed: {exc}") from exc

    async def introspect(self) -> SchemaCatalog:
        async with self._p.acquire() as conn:
            table_rows = await conn.fetch(_TABLES_SQL, self._schemas)
            oids = [r["oid"] for r in table_rows]
            column_rows = await conn.fetch(_COLUMNS_SQL, oids)
            constraint_rows = await conn.fetch(_CONSTRAINTS_SQL, oids)

        columns: dict[int, list[Any]] = defaultdict(list)
        for r in column_rows:
            columns[r["attrelid"]].append(r)
        pks: dict[int, set[str]] = defaultdict(set)
        fks: dict[int, list[ForeignKey]] = defaultdict(list)
        for r in constraint_rows:
            if r["contype"] == "p":
                pks[r["conrelid"]].update(r["cols"])
            elif r["ref_table"]:
                fks[r["conrelid"]].append(
                    ForeignKey(
                        columns=tuple(r["cols"]),
                        ref_table=f"{r['ref_schema']}.{r['ref_table']}",
                        ref_columns=tuple(r["ref_cols"]),
                    )
                )

        tables = [
            TableInfo(
                schema=r["schema"],
                name=r["name"],
                comment=r["comment"],
                row_estimate=r["row_estimate"] if r["row_estimate"] >= 0 else None,
                columns=tuple(
                    ColumnInfo(
                        name=c["attname"],
                        data_type=c["data_type"],
                        nullable=c["nullable"],
                        is_primary_key=c["attname"] in pks[r["oid"]],
                        comment=c["comment"],
                    )
                    for c in columns[r["oid"]]
                ),
                foreign_keys=tuple(fks[r["oid"]]),
            )
            for r in table_rows
        ]
        return SchemaCatalog.build(
            tables,
            default_schema=self._schemas[0],
            hidden_tables=self._hidden_tables,
            hidden_columns=self._hidden_columns,
        )

    async def execute(self, sql: str, *, max_rows: int, timeout_s: float) -> QueryResult:
        started = time.perf_counter()
        try:
            # Client-side deadline as a backstop for the server-side statement_timeout.
            async with (
                asyncio.timeout(timeout_s + 2),
                self._p.acquire() as conn,
                conn.transaction(readonly=True, isolation="repeatable_read"),
            ):
                await conn.execute(f"SET LOCAL statement_timeout = {int(timeout_s * 1000)}")
                stmt = await conn.prepare(sql)
                columns = [a.name for a in stmt.get_attributes()]
                cursor = await stmt.cursor()
                records = await cursor.fetch(max_rows + 1)
        except asyncpg.QueryCanceledError as exc:
            raise QueryTimeoutError(f"query exceeded the {timeout_s:g}s time limit") from exc
        except TimeoutError as exc:
            raise QueryTimeoutError(f"query exceeded the {timeout_s:g}s time limit") from exc
        except asyncpg.PostgresError as exc:
            detail = f"{exc.__class__.__name__}: {exc}"
            hint = getattr(exc, "hint", None)
            raise QueryExecutionError(f"{detail} (hint: {hint})" if hint else detail) from exc
        except (OSError, asyncpg.InterfaceError) as exc:
            raise DatabaseUnavailableError(f"database error: {exc}") from exc

        truncated = len(records) > max_rows
        rows = [[to_jsonable(v) for v in rec.values()] for rec in records[:max_rows]]
        return QueryResult(
            columns=columns,
            rows=rows,
            truncated=truncated,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    async def search_values(
        self, table: TableInfo, column: str, term: str, *, limit: int, timeout_s: float
    ) -> list[str]:
        col = table.column(column)
        if col is None:
            raise QueryExecutionError(f"unknown column {column!r} on {table.qualified_name}")
        target = f"{quote_ident(table.schema or 'public')}.{quote_ident(table.name)}"
        expr = f"{quote_ident(col.name)}::text"
        select = f"SELECT DISTINCT {expr} AS v FROM {target} WHERE {expr} ILIKE $1 LIMIT $2"
        values: list[str] = []
        try:
            async with (
                asyncio.timeout(timeout_s + 2),
                self._p.acquire() as conn,
                conn.transaction(readonly=True),
            ):
                await conn.execute(f"SET LOCAL statement_timeout = {int(timeout_s * 1000)}")
                rows = await conn.fetch(select, f"%{_escape_like(term)}%", limit * 4)
                values = [r["v"] for r in rows]
                if not values and self._has_trgm:
                    # Misheard or misspelled values: rank by trigram similarity.
                    rows = await conn.fetch(
                        f"SELECT v FROM (SELECT DISTINCT {expr} AS v FROM {target}) s "
                        f"WHERE v IS NOT NULL AND similarity(v, $1) > 0.2 "
                        f"ORDER BY similarity(v, $1) DESC, v LIMIT $2",
                        term,
                        limit,
                    )
                    if rows:
                        return [r["v"] for r in rows]
                for word in search_words(term) if not values else []:
                    rows = await conn.fetch(select, f"%{_escape_like(word)}%", limit * 4)
                    values.extend(r["v"] for r in rows)
        except (asyncpg.QueryCanceledError, TimeoutError) as exc:
            raise QueryTimeoutError("value search timed out") from exc
        except asyncpg.PostgresError as exc:
            raise QueryExecutionError(f"{exc.__class__.__name__}: {exc}") from exc
        return rank_matches([v for v in values if v is not None], term, limit)
