from __future__ import annotations

import pytest

from speech_to_sql.db.catalog import ColumnInfo, SchemaCatalog, TableInfo
from speech_to_sql.errors import UnsafeSQLError
from speech_to_sql.sql.guard import SQLGuard


def _table(name: str, *cols: str) -> TableInfo:
    return TableInfo(schema="public", name=name, columns=tuple(ColumnInfo(c, "text") for c in cols))


@pytest.fixture
def catalog() -> SchemaCatalog:
    return SchemaCatalog.build(
        [
            _table("customers", "id", "full_name", "email", "city"),
            _table("orders", "id", "customer_id", "order_date", "status"),
            _table("secrets", "id", "value"),
        ],
        default_schema="public",
        hidden_tables=["secrets"],
        hidden_columns=["customers.email"],
    )


@pytest.fixture
def guard() -> SQLGuard:
    return SQLGuard("postgres", max_rows=100)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, full_name FROM customers",
        "select count(*) from orders where status = 'delivered'",
        "WITH recent AS (SELECT * FROM orders WHERE order_date > '2025-01-01') SELECT count(*) FROM recent",
        "SELECT city FROM customers UNION SELECT status FROM orders",
        "SELECT c.full_name, count(o.id) AS n FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY 1",
        "SELECT public.orders.id FROM public.orders",
        "SELECT g FROM generate_series(1, 3) AS g",
        "SELECT date_trunc('month', order_date::date) AS m, count(*) FROM orders GROUP BY 1 ORDER BY 1;",
        "SELECT id FROM orders FETCH FIRST 5 ROWS ONLY",
    ],
)
def test_allows_read_only_queries(guard: SQLGuard, catalog: SchemaCatalog, sql: str) -> None:
    result = guard.validate(sql, catalog)
    assert result.sql


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("", "empty"),
        ("DELETE FROM orders", "not_select"),
        ("UPDATE orders SET status = 'x'", "not_select"),
        ("INSERT INTO orders (id) VALUES (1)", "not_select"),
        ("DROP TABLE orders", "not_select"),
        ("TRUNCATE orders", "not_select"),
        ("COPY orders TO '/tmp/x'", "not_select"),
        ("GRANT SELECT ON orders TO public", "not_select"),
        ("SET statement_timeout = 0", "not_select"),
        ("EXPLAIN ANALYZE DELETE FROM orders", "not_select"),
        ("SELECT 1; DROP TABLE orders", "multiple_statements"),
        ("WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d", "write_operation"),
        ("SELECT * INTO backup FROM orders", "select_into"),
        ("SELECT * FROM orders FOR UPDATE", "locking"),
        ("SELECT pg_sleep(30)", "forbidden_function"),
        ("SELECT pg_catalog.pg_sleep(30)", "forbidden_function"),
        ("SELECT pg_read_file('/etc/passwd')", "forbidden_function"),
        ("SELECT query_to_xml('delete from orders', true, true, '')", "forbidden_function"),
        ("SELECT nextval('orders_id_seq')", "forbidden_function"),
        ("SELECT set_config('statement_timeout', '0', false)", "forbidden_function"),
        ("SELECT dblink_exec('dbname=x', 'drop table y')", "forbidden_function"),
        ("SELECT public.my_func()", "qualified_function"),
        ("SELECT * FROM pg_catalog.pg_authid", "unknown_table"),
        ("SELECT * FROM information_schema.tables", "unknown_table"),
        ("SELECT * FROM secrets", "unknown_table"),
        ("SELECT * FROM nope", "unknown_table"),
        ("SELECT * FROM otherdb.public.orders", "cross_database"),
        ("SELECT email FROM customers", "restricted_column"),
        ("SELECT c.email FROM customers c", "restricted_column"),
        ("SELECT * FROM customers", "restricted_column"),
        ("SELECT c.* FROM customers c", "restricted_column"),
        ("SELECT row_to_json(c) FROM customers c", "restricted_column"),
        ("SELECT id FROM customers WHERE email LIKE '%@x.com'", "restricted_column"),
        ("SELECT FROM WHERE", "parse_error"),
    ],
)
def test_rejects_unsafe_queries(guard: SQLGuard, catalog: SchemaCatalog, sql: str, reason: str) -> None:
    with pytest.raises(UnsafeSQLError) as exc_info:
        guard.validate(sql, catalog)
    assert exc_info.value.reason == reason


def test_star_allowed_on_unrestricted_table(guard: SQLGuard, catalog: SchemaCatalog) -> None:
    assert guard.validate("SELECT * FROM orders", catalog).sql.startswith("SELECT *")


def test_injects_limit_when_missing(guard: SQLGuard, catalog: SchemaCatalog) -> None:
    result = guard.validate("SELECT id FROM orders", catalog)
    assert result.sql.endswith("LIMIT 101")
    assert result.limit_applied


def test_keeps_smaller_limit(guard: SQLGuard, catalog: SchemaCatalog) -> None:
    result = guard.validate("SELECT id FROM orders ORDER BY id DESC LIMIT 5", catalog)
    assert result.sql.endswith("LIMIT 5")
    assert not result.limit_applied


def test_clamps_larger_limit(guard: SQLGuard, catalog: SchemaCatalog) -> None:
    assert guard.validate("SELECT id FROM orders LIMIT 100000", catalog).sql.endswith("LIMIT 101")


def test_limits_union_as_a_whole(guard: SQLGuard, catalog: SchemaCatalog) -> None:
    sql = guard.validate("SELECT city FROM customers UNION SELECT status FROM orders", catalog).sql
    assert sql.endswith("LIMIT 101")


def test_reports_referenced_tables(guard: SQLGuard, catalog: SchemaCatalog) -> None:
    result = guard.validate(
        "WITH x AS (SELECT customer_id FROM orders) SELECT c.id FROM customers c JOIN x ON x.customer_id = c.id",
        catalog,
    )
    assert result.tables == ("public.customers", "public.orders")


def test_unknown_table_error_lists_available_tables(guard: SQLGuard, catalog: SchemaCatalog) -> None:
    with pytest.raises(UnsafeSQLError) as exc_info:
        guard.validate("SELECT * FROM order", catalog)
    assert "public.orders" in exc_info.value.message
    assert "secrets" not in exc_info.value.message


def test_sqlite_dialect() -> None:
    catalog = SchemaCatalog.build([TableInfo(None, "t", (ColumnInfo("a", "int"),))], default_schema=None)
    guard = SQLGuard("sqlite", max_rows=10)
    assert guard.validate("select a from t", catalog).sql == "SELECT a FROM t LIMIT 11"
    with pytest.raises(UnsafeSQLError):
        guard.validate("SELECT load_extension('evil')", catalog)
    with pytest.raises(UnsafeSQLError):
        guard.validate("ATTACH DATABASE 'x.db' AS x", catalog)
