from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from speech_to_sql.api import create_app
from speech_to_sql.config import Settings
from tests.conftest import FakeTranscriber, ScriptedClaude, final, tool_call, with_overrides

SQL = "SELECT status, count(*) AS n FROM orders GROUP BY status ORDER BY n DESC"


def scripted_answer() -> ScriptedClaude:
    return ScriptedClaude([tool_call("run_sql", {"sql": SQL}), final(answer="Most orders were delivered.", sql=SQL)])


@pytest.fixture
def make_client(settings: Settings):  # type: ignore[no-untyped-def]
    clients: list[TestClient] = []

    def _make(script: ScriptedClaude | None = None, transcriber: Any = None, **overrides: Any) -> TestClient:
        s = with_overrides(settings, **overrides)
        app = create_app(s, llm=(script or ScriptedClaude([])).client(), transcriber=transcriber)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def parse_sse(text: str) -> Iterator[tuple[str, dict[str, Any]]]:
    for chunk in text.strip().split("\n\n"):
        lines = chunk.splitlines()
        if not lines or lines[0].startswith(":"):
            continue
        event = lines[0].removeprefix("event: ")
        yield event, json.loads(lines[1].removeprefix("data: "))


def test_health_ready_meta_and_ui(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client()
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["ready"] is True
    assert client.get("/v1/meta").json()["voice_enabled"] is False
    assert "s2s_requests_total" in client.get("/metrics").text
    assert "Speech to SQL" in client.get("/").text
    assert client.get("/healthz").headers["x-request-id"]


def test_query_returns_rows_and_usage(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(scripted_answer())
    res = client.post("/v1/query", json={"question": "orders by status"}, headers={"X-Request-ID": "req-123"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["request_id"] == "req-123"
    assert body["status"] == "answered"
    assert body["columns"] == ["status", "n"]
    assert body["rows"][0][0] == "delivered"
    assert body["row_count"] == 5 and body["truncated"] is False
    assert body["usage"]["llm_calls"] == 2
    assert body["tool_calls"] == 1
    assert body["timings_ms"]["total"] >= body["timings_ms"]["agent"]


def test_query_stream_emits_progress_then_result(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(scripted_answer())
    res = client.post("/v1/query/stream", json={"question": "orders by status"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    events = list(parse_sse(res.text))
    assert [e for e, _ in events] == ["tool_call", "tool_result", "result"]
    assert events[1][1]["row_count"] == 5
    assert events[-1][1]["answer"] == "Most orders were delivered."


def test_query_stream_reports_errors_as_events(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(ScriptedClaude([final(sql="DELETE FROM orders")]))
    events = list(parse_sse(client.post("/v1/query/stream", json={"question": "x"}).text))
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "agent_failed"


def test_agent_failure_maps_to_502(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(ScriptedClaude([final(sql="DELETE FROM orders")]))
    res = client.post("/v1/query", json={"question": "x"})
    assert res.status_code == 502
    assert res.json()["error"]["code"] == "agent_failed"


def test_history_is_forwarded_to_the_model(make_client) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude([final(answer="ok")])
    client = make_client(script)
    history = [{"question": "orders by status", "sql": SQL, "answer": "Most were delivered."}]
    assert client.post("/v1/query", json={"question": "only for 2025", "history": history}).status_code == 200
    prompt = script.requests[0]["messages"][0]["content"]
    assert "Q1: orders by status" in prompt and "only for 2025" in prompt


def test_validation_errors(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client()
    res = client.post("/v1/query", json={"question": ""})
    assert res.status_code == 422 and res.json()["error"]["code"] == "invalid_request"


def test_voice_query_transcribes_then_answers(make_client) -> None:  # type: ignore[no-untyped-def]
    stt = FakeTranscriber("orders by status")
    client = make_client(scripted_answer(), transcriber=stt)
    res = client.post(
        "/v1/voice-query",
        files={"audio": ("speech.webm", b"\x1a\x45\xdf\xa3" * 300, "audio/webm;codecs=opus")},
        data={"history": json.dumps([{"question": "hi", "answer": "hello"}])},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["transcript"]["text"] == "orders by status"
    assert body["question"] == "orders by status"
    assert "stt" in body["timings_ms"]
    assert stt.calls == [(1200, "audio/webm;codecs=opus")]


def test_transcribe_endpoint_and_upload_limits(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(transcriber=FakeTranscriber("hello"), max_audio_bytes=1000)
    ok = client.post("/v1/transcribe", files={"audio": ("a.wav", b"x" * 500, "audio/wav")})
    assert ok.status_code == 200 and ok.json()["text"] == "hello"
    big = client.post("/v1/transcribe", files={"audio": ("a.wav", b"x" * 1500, "audio/wav")})
    assert big.status_code == 413
    bad_type = client.post("/v1/transcribe", files={"audio": ("a.txt", b"x" * 10, "text/plain")})
    assert bad_type.status_code == 422
    empty = client.post("/v1/transcribe", files={"audio": ("a.wav", b"", "audio/wav")})
    assert empty.status_code == 422
    bad_history = client.post(
        "/v1/voice-query", files={"audio": ("a.wav", b"x" * 10, "audio/wav")}, data={"history": "not json"}
    )
    assert bad_history.status_code == 422


def test_voice_disabled_returns_501(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client()
    res = client.post("/v1/transcribe", files={"audio": ("a.wav", b"x" * 10, "audio/wav")})
    assert res.status_code == 501 and res.json()["error"]["code"] == "speech_disabled"


def test_api_key_auth(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(ScriptedClaude([final(answer="ok")]), api_keys=["secret-1", "secret-2"])
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/meta").json()["auth_required"] is True
    assert client.post("/v1/query", json={"question": "x"}).status_code == 401
    assert client.post("/v1/query", json={"question": "x"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/v1/query", json={"question": "x"}, headers={"X-API-Key": "secret-2"}).status_code == 200


def test_rate_limit(make_client) -> None:  # type: ignore[no-untyped-def]
    client = make_client(ScriptedClaude([final(answer="ok")] * 3), rate_limit_per_minute=2)
    codes = [client.post("/v1/query", json={"question": "x"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    limited = client.post("/v1/query", json={"question": "x"})
    assert int(limited.headers["retry-after"]) >= 1


async def test_heartbeat_stream_yields_pings_and_detects_dead_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from speech_to_sql.api import routes
    from speech_to_sql.errors import AgentError

    monkeypatch.setattr(routes, "_HEARTBEAT_S", 0.05)

    async def slow() -> Any:
        await asyncio.sleep(0.12)
        yield "item"

    items = [i async for i in routes._with_heartbeat(slow())]
    assert items[-1] == "item" and None in items

    async def cancelled() -> Any:
        raise asyncio.CancelledError
        yield  # pragma: no cover

    with pytest.raises(AgentError, match="stopped unexpectedly"):
        async for _ in routes._with_heartbeat(cancelled()):
            pass


def test_missing_llm_credentials_fail_readiness_and_queries(settings: Settings) -> None:
    import anthropic

    from tests.conftest import with_overrides

    s = with_overrides(settings)
    no_creds = anthropic.AsyncAnthropic(api_key=None, auth_token=None)
    no_creds.api_key = None  # in case the environment provides one
    with TestClient(create_app(s, llm=no_creds)) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert "no Anthropic credentials" in ready.json()["checks"]["llm"]
        res = client.post("/v1/query", json={"question": "x"})
        assert res.status_code == 503 and res.json()["error"]["code"] == "llm_unavailable"


def test_unknown_model_is_reported(make_client) -> None:  # type: ignore[no-untyped-def]
    script = ScriptedClaude([])
    original = script.handler

    def handler(request: Any) -> Any:
        import httpx2

        if request.url.path.startswith("/v1/models/"):
            return httpx2.Response(404, json={"type": "error", "error": {"type": "not_found_error", "message": "nope"}})
        return original(request)

    script.handler = handler  # type: ignore[method-assign]
    client = make_client(script)
    assert "not available" in client.get("/readyz").json()["checks"]["llm"]
