"""Integration tests against a live Postgres.

Set S2S_TEST_PG_DSN to a *disposable* database (the demo tables are dropped and recreated):
    createdb s2s_test && S2S_TEST_PG_DSN=postgresql://localhost/s2s_test uv run pytest -m postgres
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from speech_to_sql.db.postgres import PostgresDatabase
from speech_to_sql.demo.seed import generate, seed_postgres
from speech_to_sql.errors import QueryExecutionError, QueryTimeoutError
from speech_to_sql.sql.guard import SQLGuard

DSN = os.environ.get("S2S_TEST_PG_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DSN, reason="S2S_TEST_PG_DSN not set"),
]


@pytest.fixture(scope="module")
def seeded() -> str:
    import asyncio

    assert DSN
    asyncio.run(seed_postgres(DSN, generate(seed=7), reset=True))
    return DSN


@pytest.fixture
async def pg(seeded: str) -> AsyncIterator[PostgresDatabase]:
    db = PostgresDatabase(seeded, schemas=["public"], hidden_tables=[], hidden_columns=["customers.email"], max_size=2)
    await db.connect()
    yield db
    await db.close()


async def test_introspection(pg: PostgresDatabase) -> None:
    catalog = await pg.introspect()
    orders = catalog.resolve("orders")
    assert orders is not None and orders.qualified_name == "public.orders"
    assert orders.column("id") is not None and orders.column("id").is_primary_key  # type: ignore[union-attr]
    assert any(fk.ref_table == "public.customers" for fk in orders.foreign_keys)
    assert orders.comment == "One row per order"
    customers = catalog.resolve("customers")
    assert customers is not None and customers.column("email") is None


async def test_execute_and_truncation(pg: PostgresDatabase) -> None:
    catalog = await pg.introspect()
    sql = SQLGuard("postgres", max_rows=3).validate("SELECT id, order_date FROM orders ORDER BY id", catalog).sql
    result = await pg.execute(sql, max_rows=3, timeout_s=5)
    assert result.columns == ["id", "order_date"]
    assert [r[0] for r in result.rows] == [1, 2, 3]
    assert result.truncated
    assert isinstance(result.rows[0][1], str)  # dates serialised as ISO strings


async def test_transactions_are_read_only_even_without_the_guard(pg: PostgresDatabase) -> None:
    with pytest.raises(QueryExecutionError, match="read-only"):
        await pg.execute("DELETE FROM order_items", max_rows=1, timeout_s=5)
    with pytest.raises(QueryExecutionError):
        await pg.execute("CREATE TABLE pwned (a int)", max_rows=1, timeout_s=5)
    rows = await pg.execute("SELECT count(*) FROM order_items", max_rows=1, timeout_s=5)
    assert rows.rows[0][0] > 0


async def test_statement_timeout(pg: PostgresDatabase) -> None:
    with pytest.raises(QueryTimeoutError):
        await pg.execute("SELECT pg_sleep(3)", max_rows=1, timeout_s=0.5)
    await pg.ping()  # the pool recovers


async def test_value_search(pg: PostgresDatabase) -> None:
    catalog = await pg.introspect()
    customers = catalog.resolve("customers")
    products = catalog.resolve("products")
    assert customers is not None and products is not None
    assert (await pg.search_values(customers, "company", "acme", limit=3, timeout_s=5))[0] == "Acme Corporation"
    assert "EchoBuds Pro" in await pg.search_values(products, "name", "echo buds pro", limit=3, timeout_s=5)


async def test_session_settings_are_pinned(pg: PostgresDatabase) -> None:
    result = await pg.execute(
        "SELECT current_setting('TimeZone'), current_setting('default_transaction_read_only')",
        max_rows=1,
        timeout_s=5,
    )
    assert result.rows == [["UTC", "on"]]
