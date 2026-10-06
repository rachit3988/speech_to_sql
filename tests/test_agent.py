from __future__ import annotations

from typing import Any

import pytest

from speech_to_sql.agent import AgentEvent, AgentOutcome, HistoryTurn, SQLAgent
from speech_to_sql.config import Settings
from speech_to_sql.db.catalog import CatalogProvider
from speech_to_sql.db.sqlite import SQLiteDatabase
from speech_to_sql.errors import AgentError, DatabaseUnavailableError
from tests.conftest import ScriptedClaude, final, message, tool_call, with_overrides

DELIVERED_SQL = "SELECT count(*) AS delivered FROM orders WHERE status = 'delivered'"


async def run_agent(
    agent: SQLAgent, question: str = "how many orders were delivered", **kw: Any
) -> tuple[list[AgentEvent], AgentOutcome]:
    events = [e async for e in agent.run(question, **kw)]
    assert events[-1].type == "answer"
    return events, events[-1].data["outcome"]


@pytest.fixture
def make_agent(sqlite_db: SQLiteDatabase, catalogs: CatalogProvider, settings: Settings):  # type: ignore[no-untyped-def]
    def _make(script: ScriptedClaude, **overrides: Any) -> SQLAgent:
        s = with_overrides(settings, **overrides)
        return SQLAgent(script.client(), sqlite_db, catalogs, s)

    return _make


async def test_happy_path_runs_sql_and_returns_rows(make_agent, sqlite_db: SQLiteDatabase) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude(
        [
            tool_call(
                "search_column_values",
                {"table": "orders", "column": "status", "search_term": "delivered"},
                thinking="Check how statuses are stored.",
            ),
            tool_call("run_sql", {"sql": DELIVERED_SQL}, call_id="toolu_2"),
            final(
                answer="There were many delivered orders.",
                sql=DELIVERED_SQL,
                assumptions=["counted all time"],
            ),
        ]
    )
    events, outcome = await run_agent(make_agent(script))

    expected = await sqlite_db.execute(
        "SELECT count(*) FROM orders WHERE status = 'delivered'", max_rows=1, timeout_s=5
    )
    assert outcome.final.status == "answered"
    assert outcome.result is not None and outcome.result.rows == expected.rows
    assert outcome.result.columns == ["delivered"]
    assert outcome.sql is not None and outcome.sql.endswith("LIMIT 501")
    assert outcome.tool_calls == 2
    assert outcome.usage.llm_calls == 3 and outcome.usage.cache_read_tokens == 4500

    types = [e.type for e in events]
    assert types == ["thinking", "tool_call", "tool_result", "tool_call", "tool_result", "answer"]
    assert "'delivered'" in script.tool_results(1)[0]["content"]
    assert script.tool_results(2)[0]["content"].startswith("OK: 1 row(s)")


async def test_request_uses_cacheable_prompt_and_production_params(make_agent) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude([final(status="cannot_answer", answer="No such data.")])
    await run_agent(make_agent(script))
    body, headers = script.requests[0], script.headers[0]

    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers["anthropic-beta"]
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "TABLE orders" in body["system"][0]["text"]
    assert "email" not in body["system"][0]["text"]  # hidden column never reaches the model
    assert "Today's date:" not in body["system"][0]["text"]  # volatile data stays out of the cached prefix
    assert {t["name"] for t in body["tools"]} == {"describe_tables", "search_column_values", "run_sql"}
    assert all(t["strict"] for t in body["tools"])
    assert "tool_choice" not in body


async def test_system_prompt_is_byte_stable_across_requests(make_agent) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude([final(answer="a"), final(answer="b")])
    agent = make_agent(script)
    await run_agent(agent, "first question")
    await run_agent(agent, "second question", history=[HistoryTurn("first question", None, "a")])
    assert script.requests[0]["system"] == script.requests[1]["system"]
    assert script.requests[0]["tools"] == script.requests[1]["tools"]
    assert "Q1: first question" in script.requests[1]["messages"][0]["content"]


async def test_rejected_sql_is_returned_to_model_as_error(make_agent) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude(
        [
            tool_call("run_sql", {"sql": "DELETE FROM orders"}),
            tool_call("run_sql", {"sql": "SELECT email FROM customers"}, call_id="toolu_2"),
            tool_call("run_sql", {"sql": DELIVERED_SQL}, call_id="toolu_3"),
            final(sql=DELIVERED_SQL),
        ]
    )
    _, outcome = await run_agent(make_agent(script))
    first, second = script.tool_results(1)[0], script.tool_results(2)[0]
    assert first["is_error"] and "Rejected" in str(first["content"])
    assert second["is_error"] and "restricted" in str(second["content"])
    assert outcome.final.status == "answered"


async def test_database_errors_are_returned_to_model(make_agent) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude(
        [
            tool_call("run_sql", {"sql": "SELECT no_such_column FROM orders"}),
            final(status="cannot_answer", answer="Could not find that column."),
        ]
    )
    _, outcome = await run_agent(make_agent(script))
    result = script.tool_results(1)[0]
    assert result["is_error"] and "Database error" in str(result["content"])
    assert outcome.result is None


async def test_unknown_tool_arguments_get_helpful_errors(make_agent) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude(
        [
            tool_call("search_column_values", {"table": "orders", "column": "colour", "search_term": "red"}),
            tool_call("describe_tables", {"table_names": ["orders", "nope"]}, call_id="toolu_2"),
            final(status="cannot_answer", answer="no colour data"),
        ]
    )
    await run_agent(make_agent(script))
    search = script.tool_results(1)[0]
    assert search["is_error"] and "Columns: id, customer_id" in str(search["content"])
    describe = script.tool_results(2)[0]["content"]
    assert "TABLE orders" in describe and "Unknown tables: nope" in describe


async def test_final_sql_not_previously_run_is_executed(make_agent) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude([final(sql="SELECT count(*) AS n FROM customers")])
    _, outcome = await run_agent(make_agent(script))
    assert outcome.result is not None and outcome.result.rows == [[400]]


async def test_unsafe_final_sql_fails_the_run(make_agent) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude([final(sql="DROP TABLE orders")])
    with pytest.raises(AgentError, match="final SQL failed validation"):
        await run_agent(make_agent(script))


async def test_refusal_becomes_cannot_answer(make_agent) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude([message([], "refusal")])
    _, outcome = await run_agent(make_agent(script))
    assert outcome.final.status == "cannot_answer"
    assert outcome.result is None


async def test_step_budget_is_enforced(make_agent) -> None:  # type: ignore[no-untyped-def]
    loop = [tool_call("run_sql", {"sql": DELIVERED_SQL}, call_id=f"toolu_{i}") for i in range(10)]
    script = ScriptedClaude(loop)
    with pytest.raises(AgentError, match="budget of 3 steps"):
        await run_agent(make_agent(script, agent_max_iterations=3))
    assert len(script.requests) == 3


async def test_max_tokens_truncation_is_an_error_and_tools_do_not_run(make_agent) -> None:  # type: ignore[no-untyped-def]
    truncated = message(
        [{"type": "tool_use", "id": "toolu_1", "name": "run_sql", "input": {"sql": "SEL"}}], "max_tokens"
    )
    script = ScriptedClaude([truncated])
    with pytest.raises(AgentError, match="ran out of output tokens"):
        await run_agent(make_agent(script))
    assert len(script.requests) == 1


async def test_api_errors_are_mapped(make_agent) -> None:  # type: ignore[no-untyped-def]
    overloaded = {
        "_status": 529,
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    }
    script = ScriptedClaude([overloaded])
    with pytest.raises(AgentError, match="HTTP 529"):
        await run_agent(make_agent(script))


async def test_database_outage_aborts_instead_of_looping(
    make_agent, sqlite_db: SQLiteDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    async def down(*_a: Any, **_k: Any) -> Any:
        raise DatabaseUnavailableError("connection refused")

    monkeypatch.setattr(sqlite_db, "execute", down)
    script = ScriptedClaude([tool_call("run_sql", {"sql": DELIVERED_SQL}), final()])
    with pytest.raises(DatabaseUnavailableError):
        await run_agent(make_agent(script))
    assert len(script.requests) == 1


async def test_fallback_disabled_omits_beta(make_agent) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude([final(answer="x")])
    await run_agent(make_agent(script, llm_refusal_fallback=False, llm_effort="low"))
    assert "fallbacks" not in script.requests[0]
    assert "server-side-fallback" not in script.headers[0].get("anthropic-beta", "")
    assert script.requests[0]["output_config"]["effort"] == "low"
