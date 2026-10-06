from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from speech_to_sql.config import Settings
from speech_to_sql.db.catalog import CatalogProvider
from speech_to_sql.db.sqlite import SQLiteDatabase
from speech_to_sql.demo.seed import generate, seed_sqlite
from speech_to_sql.stt import Transcript

# --- demo database ---------------------------------------------------------------


@pytest.fixture(scope="session")
def demo_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("db") / "demo.db"
    seed_sqlite(str(path), generate(seed=7), reset=True)
    return path


@pytest.fixture
async def sqlite_db(demo_db_path: Path) -> AsyncIterator[SQLiteDatabase]:
    db = SQLiteDatabase(str(demo_db_path), hidden_tables=[], hidden_columns=["customers.email"], pool_size=2)
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
def settings(demo_db_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=f"sqlite:///{demo_db_path}",
        hidden_columns=["customers.email"],
        stt_provider="none",
        rate_limit_per_minute=0,
        agent_max_iterations=6,
        agent_timeout_s=30,
        log_level="WARNING",
    )


def with_overrides(settings: Settings, **overrides: Any) -> Settings:
    """Copy settings with validation (``model_copy`` would skip coercion, e.g. str -> SecretStr)."""
    return Settings(_env_file=None, **{**settings.model_dump(), **overrides})  # type: ignore[call-arg]


@pytest.fixture
def catalogs(sqlite_db: SQLiteDatabase) -> CatalogProvider:
    return CatalogProvider(sqlite_db.introspect, ttl_s=600)


# --- scripted Claude ---------------------------------------------------------------

Turn = Callable[[dict[str, Any]], dict[str, Any]] | dict[str, Any]


def message(content: list[dict[str, Any]], stop_reason: str, *, model: str = "claude-opus-5-5") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 40,
            "cache_read_input_tokens": 1500,
            "cache_creation_input_tokens": 0,
        },
    }


def tool_call(
    name: str, args: dict[str, Any], *, call_id: str = "toolu_1", thinking: str | None = None
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if thinking:
        content.append({"type": "thinking", "thinking": thinking, "signature": "sig"})
    content.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
    return message(content, "tool_use")


def final(status: str = "answered", answer: str = "done", sql: str | None = None, **extra: Any) -> dict[str, Any]:
    payload = {
        "status": status,
        "answer": answer,
        "sql": sql,
        "assumptions": extra.get("assumptions", []),
        "clarification_question": extra.get("clarification_question"),
    }
    return message([{"type": "text", "text": json.dumps(payload)}], "end_turn")


class ScriptedClaude:
    """Serves canned Messages API responses in order and records each request body."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET" and request.url.path.startswith("/v1/models/"):
            model_id = request.url.path.rsplit("/", 1)[-1]
            return httpx2.Response(
                200,
                json={"id": model_id, "type": "model", "display_name": model_id, "created_at": "2026-01-01T00:00:00Z"},
            )
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        if not self.turns:
            return httpx2.Response(
                500, json={"type": "error", "error": {"type": "api_error", "message": "script exhausted"}}
            )
        turn = self.turns.pop(0)
        payload = turn(body) if callable(turn) else turn
        status = payload.pop("_status", 200)
        return httpx2.Response(status, json=payload)

    def client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self.handler)),
        )

    def tool_results(self, index: int) -> list[dict[str, Any]]:
        """tool_result blocks sent in request ``index``."""
        last = self.requests[index]["messages"][-1]
        return [b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


class FakeTranscriber:
    name = "fake"

    def __init__(self, text: str = "how many orders were delivered") -> None:
        self.text = text
        self.calls: list[tuple[int, str]] = []

    async def transcribe(self, audio: bytes, content_type: str) -> Transcript:
        self.calls.append((len(audio), content_type))
        return Transcript(text=self.text, confidence=0.93, duration_s=2.1, provider=self.name)

    async def close(self) -> None:
        pass
