from __future__ import annotations

import pytest

from speech_to_sql.agent.prompts import HistoryTurn, build_system_prompt, build_user_message
from speech_to_sql.db.catalog import CatalogProvider, ColumnInfo, ForeignKey, SchemaCatalog, TableInfo


def make_catalog(**kw: object) -> SchemaCatalog:
    tables = [
        TableInfo("public", "orders", (ColumnInfo("id", "int", False, True), ColumnInfo("customer_id", "int"))),
        TableInfo(
            "public",
            "customers",
            (ColumnInfo("id", "int", False, True), ColumnInfo("email", "text"), ColumnInfo("name", "text")),
            comment="People",
        ),
        TableInfo("sales", "orders", (ColumnInfo("id", "int"),)),
    ]
    tables[0] = TableInfo(
        "public", "orders", tables[0].columns, foreign_keys=(ForeignKey(("customer_id",), "public.customers", ("id",)),)
    )
    return SchemaCatalog.build(tables, default_schema="public", **kw)  # type: ignore[arg-type]


def test_resolve_prefers_default_schema_and_is_case_insensitive() -> None:
    catalog = make_catalog()
    assert catalog.resolve("ORDERS").qualified_name == "public.orders"  # type: ignore[union-attr]
    assert catalog.resolve("orders", "sales").qualified_name == "sales.orders"  # type: ignore[union-attr]
    assert catalog.resolve("sales.orders") is not None
    assert catalog.resolve("missing") is None


def test_hidden_columns_and_tables_are_removed() -> None:
    catalog = make_catalog(hidden_tables=["sales.orders"], hidden_columns=["customers.email"])
    customers = catalog.resolve("customers")
    assert customers is not None and customers.column("email") is None
    assert catalog.hidden_columns_for(customers) == {"email"}
    assert catalog.resolve("orders", "sales") is None
    text, complete = catalog.render(max_chars=10_000)
    assert complete and "email" not in text and "sales.orders" not in text
    assert "FK (customer_id) -> public.customers(id)" in text


def test_render_falls_back_to_table_list_for_large_schemas() -> None:
    catalog = make_catalog()
    full, complete = catalog.render(max_chars=10_000)
    brief, brief_complete = catalog.render(max_chars=80)
    assert complete and not brief_complete
    assert "customer_id" in full and "customer_id" not in brief
    prompt = build_system_prompt(dialect="postgres", schema_text=brief, schema_complete=False, max_rows=10)
    assert "call describe_tables" in prompt


def test_user_message_includes_history_and_date() -> None:
    msg = build_user_message(
        "and by month?", today="2026-10-05", history=[HistoryTurn("orders in 2025", "SELECT 1", "1200")]
    )
    assert msg.startswith("Today's date: 2026-10-05")
    assert "Q1: orders in 2025" in msg and "SQL1: SELECT 1" in msg
    assert msg.endswith("Question (transcribed from speech): and by month?")


async def test_catalog_provider_caches_and_serves_stale_on_failure() -> None:
    calls = 0
    fail = False

    async def loader() -> SchemaCatalog:
        nonlocal calls
        calls += 1
        if fail:
            raise RuntimeError("db down")
        return make_catalog()

    provider = CatalogProvider(loader, ttl_s=60)
    first = await provider.get()
    assert await provider.get() is first and calls == 1

    provider.invalidate()
    fail = True
    assert await provider.get() is first  # stale but available
    assert calls == 2

    empty = CatalogProvider(loader, ttl_s=60)
    with pytest.raises(RuntimeError):
        await empty.get()


def test_anthropic_key_from_dotenv_reaches_the_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from speech_to_sql.config import Settings
    from speech_to_sql.runtime import create_llm_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-dotenv\nS2S_LLM_EFFORT=low\n")
    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]
    assert settings.llm_effort == "low"
    assert create_llm_client(settings).api_key == "sk-ant-from-dotenv"
