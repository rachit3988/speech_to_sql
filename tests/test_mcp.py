from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from speech_to_sql.api import create_app
from speech_to_sql.config import Settings
from speech_to_sql.db.sqlite import SQLiteDatabase
from speech_to_sql.mcp_server import build_mcp_server
from speech_to_sql.runtime import open_services
from tests.conftest import ScriptedClaude, final, tool_call, with_overrides

SQL = "SELECT status, count(*) AS n FROM orders GROUP BY status ORDER BY n DESC"
EXPECTED_TOOLS = {"ask_database", "describe_database", "find_values", "run_sql"}


@asynccontextmanager
async def mcp_client(
    settings: Settings, demo_db_path: Path, script: ScriptedClaude | None = None, **overrides: Any
) -> AsyncIterator[Client]:
    s = with_overrides(settings, **overrides)
    db = SQLiteDatabase(str(demo_db_path), hidden_tables=[], hidden_columns=s.hidden_columns, pool_size=2)
    llm = (script or ScriptedClaude([])).client()
    async with (
        open_services(s, db=db, llm=llm, with_speech=False) as svc,
        Client(build_mcp_server(lambda: svc, s)) as client,
    ):
        yield client


async def test_tools_are_discoverable_and_read_only(settings: Settings, demo_db_path: Path) -> None:
    desc = "Acme e-commerce data: customers, orders, products"
    async with mcp_client(settings, demo_db_path, mcp_data_description=desc) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == EXPECTED_TOOLS
        assert all(t.annotations and t.annotations.read_only_hint for t in tools.values())
        assert all(t.annotations and t.annotations.destructive_hint is False for t in tools.values())
        assert desc in (tools["ask_database"].description or "")
        assert tools["ask_database"].output_schema is not None
        assert "ctx" not in tools["ask_database"].input_schema["properties"]
        assert desc in (client.instructions or "")


async def test_ask_database_returns_structured_answer_with_progress(settings: Settings, demo_db_path: Path) -> None:
    script = ScriptedClaude([tool_call("run_sql", {"sql": SQL}), final(answer="Mostly delivered.", sql=SQL)])
    progress: list[str | None] = []

    async def on_progress(_progress: float, _total: float | None, message: str | None) -> None:
        progress.append(message)

    async with mcp_client(settings, demo_db_path, script) as client:
        result = await client.call_tool(
            "ask_database",
            {"question": "orders by status", "context": "We are reviewing fulfilment quality."},
            progress_callback=on_progress,
        )
    assert not result.is_error, result.content
    data = result.structured_content
    assert data["status"] == "answered" and data["answer"] == "Mostly delivered."
    assert data["columns"] == ["status", "n"] and data["row_count"] == 5
    assert data["rows"][0][0] == "delivered" and data["truncated"] is False
    assert data["sql"].startswith("SELECT status")
    assert progress == ["Running SQL", "Query returned 5 row(s)"]
    prompt = script.requests[0]["messages"][0]["content"]
    assert "We are reviewing fulfilment quality." in prompt
    assert "transcribed from speech" not in prompt


async def test_ask_database_caps_rows(settings: Settings, demo_db_path: Path) -> None:
    script = ScriptedClaude([final(sql=SQL)])
    async with mcp_client(settings, demo_db_path, script) as client:
        result = await client.call_tool("ask_database", {"question": "orders by status", "max_rows": 2})
    data = result.structured_content
    assert data["row_count"] == 2 and len(data["rows"]) == 2 and data["truncated"] is True


async def test_ask_database_rejects_oversized_max_rows(settings: Settings, demo_db_path: Path) -> None:
    async with mcp_client(settings, demo_db_path, mcp_max_rows=10) as client:
        result = await client.call_tool("ask_database", {"question": "x", "max_rows": 11})
    assert result.is_error


async def test_agent_failures_are_tool_errors(settings: Settings, demo_db_path: Path) -> None:
    script = ScriptedClaude([final(sql="DROP TABLE orders")])
    async with mcp_client(settings, demo_db_path, script) as client:
        result = await client.call_tool("ask_database", {"question": "drop it"})
    assert result.is_error
    assert "agent_failed" in result.content[0].text  # type: ignore[union-attr]


async def test_run_sql_is_guarded(settings: Settings, demo_db_path: Path) -> None:
    async with mcp_client(settings, demo_db_path) as client:
        ok = await client.call_tool("run_sql", {"sql": "SELECT count(*) AS n FROM customers"})
        assert not ok.is_error and ok.structured_content["rows"] == [[400]]
        assert ok.structured_content["sql"].endswith("LIMIT 501")

        for bad in ("DELETE FROM orders", "SELECT email FROM customers", "SELECT * FROM sqlite_master"):
            res = await client.call_tool("run_sql", {"sql": bad})
            assert res.is_error, bad
            assert "unsafe_sql" in res.content[0].text  # type: ignore[union-attr]

        capped = await client.call_tool("run_sql", {"sql": "SELECT id FROM orders", "max_rows": 3})
        assert capped.structured_content["row_count"] == 3 and capped.structured_content["truncated"]


async def test_describe_and_find_values(settings: Settings, demo_db_path: Path) -> None:
    async with mcp_client(settings, demo_db_path) as client:
        schema = (await client.call_tool("describe_database", {})).content[0].text  # type: ignore[union-attr]
        assert schema.startswith("Dialect: SQLite") and "TABLE orders" in schema and "email" not in schema
        one = (await client.call_tool("describe_database", {"tables": ["products", "nope"]})).content[0].text  # type: ignore[union-attr]
        assert "TABLE products" in one and "TABLE orders" not in one and "Unknown tables: nope" in one

        found = await client.call_tool(
            "find_values", {"table": "customers", "column": "company", "search_term": "acme"}
        )
        assert found.structured_content["result"][0] == "Acme Corporation"
        missing = await client.call_tool("find_values", {"table": "customers", "column": "colour", "search_term": "x"})
        assert missing.is_error


async def test_raw_sql_tools_can_be_disabled(settings: Settings, demo_db_path: Path) -> None:
    async with mcp_client(settings, demo_db_path, mcp_allow_raw_sql=False) as client:
        assert {t.name for t in (await client.list_tools()).tools} == {"ask_database", "describe_database"}
        assert "run_sql" not in (client.instructions or "")


async def test_ask_database_reports_missing_llm(settings: Settings, demo_db_path: Path) -> None:
    script = ScriptedClaude([])
    original = script.handler

    def no_model(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.startswith("/v1/models/"):
            return httpx2.Response(
                401, json={"type": "error", "error": {"type": "authentication_error", "message": "x"}}
            )
        return original(request)

    script.handler = no_model  # type: ignore[method-assign]
    async with mcp_client(settings, demo_db_path, script) as client:
        result = await client.call_tool("ask_database", {"question": "x"})
    assert result.is_error and "llm_unavailable" in result.content[0].text  # type: ignore[union-attr]


# --- over HTTP, through the real /mcp endpoint -------------------------------------------


@asynccontextmanager
async def running_app(
    settings: Settings, script: ScriptedClaude, **overrides: Any
) -> AsyncIterator[httpx2.ASGITransport]:
    app = create_app(with_overrides(settings, **overrides), llm=script.client())
    async with app.router.lifespan_context(app):
        yield httpx2.ASGITransport(app=app)


async def test_http_endpoint_with_bearer_auth(settings: Settings) -> None:
    count_sql = "SELECT count(*) FROM customers"
    script = ScriptedClaude([tool_call("run_sql", {"sql": count_sql}), final(answer="400 customers.", sql=count_sql)])
    progress: list[str | None] = []

    async def on_progress(_p: float, _t: float | None, message: str | None) -> None:
        progress.append(message)

    async with (
        running_app(settings, script, api_keys=["agent-key"]) as transport,
        httpx2.AsyncClient(transport=transport, headers={"Authorization": "Bearer agent-key"}) as http,
        Client(streamable_http_client("http://testserver/mcp", http_client=http)) as client,
    ):
        assert {t.name for t in (await client.list_tools()).tools} == EXPECTED_TOOLS
        result = await client.call_tool(
            "ask_database", {"question": "how many customers"}, progress_callback=on_progress
        )
        assert result.structured_content["rows"] == [[400]]
        assert progress == ["Running SQL", "Query returned 1 row(s)"]


async def test_http_endpoint_requires_api_key(settings: Settings) -> None:
    async with (
        running_app(settings, ScriptedClaude([]), api_keys=["agent-key"]) as transport,
        httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http,
    ):
        denied = await http.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"].startswith("Bearer")
    assert denied.headers["x-request-id"]


async def test_http_endpoint_rejects_foreign_hosts_without_auth(settings: Settings) -> None:
    headers = {"Accept": "application/json, text/event-stream"}
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    async with (
        running_app(settings, ScriptedClaude([])) as transport,
        httpx2.AsyncClient(transport=transport) as http,
    ):
        evil = await http.post("http://evil.example/mcp", json=body, headers=headers)
        assert evil.status_code in (400, 403, 421)
        async with Client(streamable_http_client("http://localhost:8000/mcp", http_client=http)) as client:
            assert len((await client.list_tools()).tools) == 4


async def test_mcp_can_be_disabled(settings: Settings) -> None:
    async with (
        running_app(settings, ScriptedClaude([]), mcp_enabled=False) as transport,
        httpx2.AsyncClient(transport=transport, base_url="http://localhost") as http,
    ):
        res = await http.post("/mcp", json={}, headers={"Accept": "application/json"})
    assert res.status_code in (404, 405)


# --- over stdio, as a local agent would launch it ------------------------------------------


async def test_stdio_server_end_to_end(demo_db_path: Path, tmp_path: Path) -> None:
    import os
    import sys

    from mcp import StdioServerParameters

    env = {k: v for k, v in os.environ.items() if not k.startswith(("S2S_", "ANTHROPIC_"))}
    env |= {"S2S_DATABASE_URL": f"sqlite:///{demo_db_path}", "S2S_STT_PROVIDER": "none", "S2S_LOG_LEVEL": "WARNING"}
    # Isolate from any saved `ant auth login` profile so no real API call can happen.
    env |= {"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path)}
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "speech_to_sql", "mcp"], env=env, cwd=str(demo_db_path.parent)
    )
    async with Client(params) as client:
        assert {t.name for t in (await client.list_tools()).tools} == EXPECTED_TOOLS
        result = await client.call_tool("run_sql", {"sql": "SELECT count(*) AS n FROM products"})
        assert not result.is_error and result.structured_content["rows"] == [[53]]
        # No Anthropic credentials in this environment: the agent tool says so instead of crashing.
        ask = await client.call_tool("ask_database", {"question": "how many products"})
        assert ask.is_error and "llm_unavailable" in ask.content[0].text  # type: ignore[union-attr]
