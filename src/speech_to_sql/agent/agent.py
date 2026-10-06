"""The text-to-SQL agent: Claude + schema-aware tools + a guarded, read-only executor.

The SDK's tool runner drives the loop. Each turn is inspected before and after its
tools run so progress can be streamed to the client, and the loop is bounded by an
iteration cap and a wall-clock deadline. The final answer is a structured output
(``FinalAnswer``); its SQL is re-validated and its result served from the per-run cache.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from zoneinfo import ZoneInfo

import anthropic
import structlog
from anthropic.types.beta import BetaUsage
from pydantic import BaseModel, Field

from speech_to_sql import metrics
from speech_to_sql.agent.prompts import HistoryTurn, build_system_prompt, build_user_message
from speech_to_sql.agent.tools import AgentEvent, RunContext, build_tools
from speech_to_sql.config import Settings
from speech_to_sql.db.base import Database, QueryResult
from speech_to_sql.db.catalog import CatalogProvider, SchemaCatalog
from speech_to_sql.errors import AgentError, AgentTimeoutError, UnsafeSQLError
from speech_to_sql.sql.guard import SQLGuard

log = structlog.get_logger(__name__)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class FinalAnswer(BaseModel):
    """The agent's structured final answer."""

    status: Literal["answered", "needs_clarification", "cannot_answer"]
    answer: str = Field(description="Plain-language answer that can be read aloud.")
    sql: str | None = Field(description="The executed query that answers the question.")
    assumptions: list[str] = Field(description="Interpretations made to answer the question.")
    clarification_question: str | None = Field(description="Set only for needs_clarification.")


@dataclass(slots=True)
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage: BetaUsage) -> None:
        self.llm_calls += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.cache_read_tokens += usage.cache_read_input_tokens or 0
        self.cache_write_tokens += usage.cache_creation_input_tokens or 0
        metrics.LLM_TOKENS.labels("input").inc(usage.input_tokens or 0)
        metrics.LLM_TOKENS.labels("output").inc(usage.output_tokens or 0)
        metrics.LLM_TOKENS.labels("cache_read").inc(usage.cache_read_input_tokens or 0)
        metrics.LLM_TOKENS.labels("cache_write").inc(usage.cache_creation_input_tokens or 0)


@dataclass(frozen=True, slots=True)
class LLMStatus:
    usable: bool
    detail: str


async def verify_llm(client: anthropic.AsyncAnthropic, model: str) -> LLMStatus:
    """Check credentials and the model ID once at startup (the Models API is free to call)."""
    try:
        await client.models.retrieve(model)
    except TypeError:
        # Raised by the SDK before sending when no credentials can be resolved.
        return LLMStatus(False, "no Anthropic credentials configured (set ANTHROPIC_API_KEY)")
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        return LLMStatus(False, "Anthropic credentials were rejected")
    except anthropic.NotFoundError:
        return LLMStatus(False, f"model {model!r} is not available to these credentials")
    except anthropic.APIError as exc:
        # Transient (network, 5xx, 429): don't block startup; requests will retry.
        return LLMStatus(True, f"unverified ({exc.__class__.__name__})")
    return LLMStatus(True, "ok")


@dataclass(slots=True)
class AgentOutcome:
    final: FinalAnswer
    sql: str | None  # the guard-normalized SQL that produced ``result``
    result: QueryResult | None
    usage: Usage
    model: str
    tool_calls: int = 0
    queries_run: list[str] = field(default_factory=list)


class SQLAgent:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        db: Database,
        catalogs: CatalogProvider,
        settings: Settings,
    ) -> None:
        self._client = client
        self._db = db
        self._catalogs = catalogs
        self._s = settings
        self._guard = SQLGuard(db.dialect, settings.max_result_rows)
        self._system_cache: tuple[int, str] | None = None

    def _system_prompt(self, catalog: SchemaCatalog) -> str:
        # Rebuilt only when the catalog object changes, keeping the cached prefix stable.
        if self._system_cache is None or self._system_cache[0] != id(catalog):
            text, complete = catalog.render(max_chars=self._s.max_schema_prompt_chars)
            prompt = build_system_prompt(
                dialect=self._db.dialect,
                schema_text=text,
                schema_complete=complete,
                max_rows=self._s.max_result_rows,
            )
            self._system_cache = (id(catalog), prompt)
        return self._system_cache[1]

    def _request_params(self, catalog: SchemaCatalog, user_message: str, ctx: RunContext) -> dict[str, Any]:
        s = self._s
        params: dict[str, Any] = {
            "model": s.llm_model,
            "max_tokens": s.llm_max_tokens,
            "system": [{"type": "text", "text": self._system_prompt(catalog), "cache_control": {"type": "ephemeral"}}],
            # Automatic caching of the growing conversation tail across loop iterations.
            "cache_control": {"type": "ephemeral"},
            "messages": [{"role": "user", "content": user_message}],
            "tools": build_tools(ctx),
            "output_format": FinalAnswer,
            "output_config": {"effort": s.llm_effort},
            "thinking": {"type": "adaptive", "display": s.llm_thinking_display},
            "max_iterations": s.agent_max_iterations,
        }
        if s.llm_refusal_fallback:
            params["fallbacks"] = "default"
            params["betas"] = [_FALLBACK_BETA]
        return params

    async def run(
        self,
        question: str,
        *,
        history: Sequence[HistoryTurn] = (),
        from_speech: bool = True,
        context: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent, yielding progress events; the last event is ``answer`` (or an error is raised)."""
        catalog = await self._catalogs.get()
        ctx = RunContext(
            db=self._db,
            catalog=catalog,
            guard=self._guard,
            max_rows=self._s.max_result_rows,
            preview_rows=self._s.agent_preview_rows,
            statement_timeout_s=self._s.statement_timeout_s,
        )
        today = dt.datetime.now(ZoneInfo(self._s.timezone)).date().isoformat()
        user_message = build_user_message(
            question,
            today=today,
            history=list(history)[-self._s.max_history_turns :],
            from_speech=from_speech,
            context=context,
        )
        runner = self._client.beta.messages.tool_runner(**self._request_params(catalog, user_message, ctx))
        usage = Usage()
        last = None
        tool_calls = 0

        try:
            async with asyncio.timeout(self._s.agent_timeout_s):
                async for message in runner:
                    last = message
                    usage.add(message.usage)
                    metrics.LLM_CALLS.labels(message.stop_reason or "none").inc()
                    for event in _message_events(message):
                        if event.type == "tool_call":
                            tool_calls += 1
                        yield event
                    if message.stop_reason == "tool_use":
                        # Run this turn's tools now (the runner caches the result) so their
                        # progress reaches the client before the next model call.
                        await runner.generate_tool_call_response()
                        for event in ctx.drain():
                            yield event
                        if ctx.fatal is not None:
                            raise ctx.fatal
        except TimeoutError as exc:
            raise AgentTimeoutError(f"agent did not finish within {self._s.agent_timeout_s:g}s") from exc
        except anthropic.APITimeoutError as exc:
            raise AgentTimeoutError("the language model timed out") from exc
        except anthropic.RateLimitError as exc:
            raise AgentError("the language model is rate limited; retry shortly") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            log.error("llm_auth_failed", error=str(exc))
            raise AgentError("the language model credentials are invalid") from exc
        except anthropic.BadRequestError as exc:
            log.error("llm_bad_request", error=exc.message, request_id=exc.request_id)
            raise AgentError(f"the language model rejected the request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise AgentError(f"language model error (HTTP {exc.status_code})") from exc
        except anthropic.APIConnectionError as exc:
            raise AgentError("could not reach the language model") from exc

        outcome = await self._finish(last, ctx, catalog, usage, tool_calls)
        metrics.AGENT_OUTCOMES.labels(outcome.final.status).inc()
        yield AgentEvent("answer", {"outcome": outcome})

    async def _finish(
        self, last: Any, ctx: RunContext, catalog: SchemaCatalog, usage: Usage, tool_calls: int
    ) -> AgentOutcome:
        if last is None:
            raise AgentError("the language model returned no response")
        served_by = _served_by(last)
        if last.stop_reason == "refusal":
            details = getattr(last, "stop_details", None)
            log.warning("llm_refusal", category=getattr(details, "category", None))
            final = FinalAnswer(
                status="cannot_answer",
                answer="I can't help with that request.",
                sql=None,
                assumptions=[],
                clarification_question=None,
            )
            return AgentOutcome(final, None, None, usage, served_by, tool_calls, list(ctx.executed))
        if last.stop_reason == "tool_use":
            raise AgentError(f"agent exceeded its budget of {self._s.agent_max_iterations} steps")
        if last.stop_reason == "max_tokens":
            raise AgentError("agent ran out of output tokens before answering")

        parsed = getattr(last, "parsed_output", None)
        if not isinstance(parsed, FinalAnswer):
            raise AgentError("agent finished without a structured answer")
        final = parsed

        sql, result = None, None
        if final.status == "answered" and final.sql:
            try:
                validated = self._guard.validate(final.sql, catalog)
            except UnsafeSQLError as exc:
                metrics.SQL_REJECTED.labels(exc.reason).inc()
                raise AgentError(f"final SQL failed validation: {exc.message}") from exc
            sql = validated.sql
            result = ctx.executed.get(sql)
            if result is None:  # the model answered with SQL it did not run verbatim
                result = await self._db.execute(
                    sql, max_rows=self._s.max_result_rows, timeout_s=self._s.statement_timeout_s
                )
        return AgentOutcome(final, sql, result, usage, served_by, tool_calls, list(ctx.executed))


def _message_events(message: Any) -> list[AgentEvent]:
    events = []
    for block in message.content:
        if block.type == "thinking" and getattr(block, "thinking", ""):
            events.append(AgentEvent("thinking", {"text": block.thinking}))
        elif block.type == "text" and message.stop_reason == "tool_use" and block.text.strip():
            events.append(AgentEvent("thinking", {"text": block.text}))
        elif block.type == "tool_use":
            events.append(AgentEvent("tool_call", {"tool": block.name, "input": block.input}))
        elif block.type == "fallback":
            events.append(AgentEvent("fallback", {"to_model": getattr(block.to, "model", None)}))
    return events


def _served_by(message: Any) -> str:
    """The model that produced the final turn (differs from the requested model after a fallback)."""
    return str(getattr(message, "model", "") or "")
