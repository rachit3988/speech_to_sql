from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import structlog
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import TypeAdapter, ValidationError

from speech_to_sql import metrics
from speech_to_sql.agent import AgentOutcome, HistoryTurn
from speech_to_sql.api.schemas import (
    ErrorBody,
    ErrorResponse,
    HistoryItem,
    QueryRequest,
    QueryResponse,
    TranscriptOut,
)
from speech_to_sql.errors import (
    AgentError,
    InvalidInputError,
    PayloadTooLargeError,
    SpeechDisabledError,
    SpeechToSQLError,
)
from speech_to_sql.runtime import Services
from speech_to_sql.security import Principal
from speech_to_sql.stt import Transcript

log = structlog.get_logger(__name__)

_AUDIO_TYPES = ("audio/", "video/webm", "video/mp4", "application/octet-stream")
_HEARTBEAT_S = 15.0
_history_adapter = TypeAdapter(list[HistoryItem])


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


def authorize(request: Request, svc: Services = Depends(get_services)) -> Principal:
    principal = svc.authenticator.authenticate(request.headers, request.client.host if request.client else None)
    svc.limiter.enforce(principal.id)
    return principal


public = APIRouter()
api = APIRouter(prefix="/v1", dependencies=[Depends(authorize)])


# --- public endpoints -------------------------------------------------------------


@public.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@public.get("/readyz", include_in_schema=False)
async def readyz(svc: Services = Depends(get_services)) -> JSONResponse:
    checks: dict[str, str] = {}
    try:
        await svc.db.ping()
        checks["database"] = "ok"
    except SpeechToSQLError as exc:
        checks["database"] = exc.message
    try:
        catalog = await svc.catalogs.get()
        checks["catalog"] = f"ok ({len(catalog.tables)} tables)"
    except Exception as exc:  # noqa: BLE001 - surfaced in the readiness payload
        checks["catalog"] = f"error: {exc.__class__.__name__}"
    checks["llm"] = svc.llm_status.detail
    ready = checks["database"] == "ok" and checks["catalog"].startswith("ok") and svc.llm_status.usable
    return JSONResponse({"ready": ready, "checks": checks}, status_code=200 if ready else 503)


@public.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@public.get("/v1/meta")
async def meta(svc: Services = Depends(get_services)) -> dict[str, Any]:
    """Non-secret client configuration for the web UI."""
    return {
        "auth_required": svc.authenticator.enabled,
        "voice_enabled": svc.transcriber is not None,
        "max_audio_bytes": svc.settings.max_audio_bytes,
        "max_history_turns": svc.settings.max_history_turns,
    }


# --- API --------------------------------------------------------------------------


@api.post("/transcribe", response_model=TranscriptOut, responses={422: {"model": ErrorResponse}})
async def transcribe(
    request: Request, audio: UploadFile = File(...), svc: Services = Depends(get_services)
) -> TranscriptOut:
    transcript = await _transcribe(svc, request, audio)
    return TranscriptOut.from_transcript(transcript)


@api.post("/query", response_model=QueryResponse, responses={422: {"model": ErrorResponse}})
async def query(body: QueryRequest, request: Request, svc: Services = Depends(get_services)) -> QueryResponse:
    svc.require_llm()
    return await _answer(
        svc,
        request,
        body.question,
        [h.to_turn() for h in body.history],
        from_speech=body.from_speech,
    )


@api.post("/voice-query", response_model=QueryResponse, responses={422: {"model": ErrorResponse}})
async def voice_query(
    request: Request,
    audio: UploadFile = File(...),
    history: str | None = Form(default=None, description="JSON list of {question, sql, answer}"),
    svc: Services = Depends(get_services),
) -> QueryResponse:
    svc.require_llm()
    turns = _parse_history(history)
    started = time.perf_counter()
    transcript = await _transcribe(svc, request, audio)
    stt_ms = (time.perf_counter() - started) * 1000
    return await _answer(
        svc,
        request,
        transcript.text,
        turns,
        from_speech=True,
        transcript=transcript,
        extra_timings={"stt": stt_ms},
    )


@api.post(
    "/query/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Server-sent events"}},
)
async def query_stream(
    body: QueryRequest, request: Request, svc: Services = Depends(get_services)
) -> StreamingResponse:
    """Stream agent progress as server-sent events.

    Events: ``thinking``, ``tool_call``, ``tool_result``, ``fallback``, then ``result``
    (a QueryResponse) or ``error`` (an ErrorBody).
    """
    svc.require_llm()
    request_id = request.state.request_id
    turns = [h.to_turn() for h in body.history]

    async def events() -> AsyncIterator[str]:
        started = time.perf_counter()
        stream = svc.agent.run(body.question, history=turns, from_speech=body.from_speech)
        try:
            async for event in _with_heartbeat(stream):
                if event is None:
                    yield ": ping\n\n"
                elif event.type == "answer":
                    outcome: AgentOutcome = event.data["outcome"]
                    total_ms = (time.perf_counter() - started) * 1000
                    response = QueryResponse.build(
                        request_id=request_id,
                        question=body.question,
                        outcome=outcome,
                        timings_ms={"agent": total_ms, "total": total_ms},
                    )
                    _record(svc, "query_stream", response, total_ms / 1000)
                    yield _sse("result", response.model_dump(mode="json"))
                else:
                    yield _sse(event.type, event.data)
        except SpeechToSQLError as exc:
            metrics.REQUESTS.labels("query_stream", exc.code).inc()
            log.warning("query_failed", code=exc.code, error=exc.message)
            yield _sse("error", ErrorBody(code=exc.code, message=exc.message, request_id=request_id).model_dump())
        except Exception:
            metrics.REQUESTS.labels("query_stream", "internal_error").inc()
            log.exception("query_stream_crashed")
            body_ = ErrorBody(code="internal_error", message="internal error", request_id=request_id)
            yield _sse("error", body_.model_dump())

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- helpers ----------------------------------------------------------------------


async def _transcribe(svc: Services, request: Request, audio: UploadFile) -> Transcript:
    if svc.transcriber is None:
        raise SpeechDisabledError("speech input is disabled on this server (S2S_STT_PROVIDER=none)")
    content_type = (audio.content_type or "application/octet-stream").split(";")[0].strip().lower()
    if not content_type.startswith(_AUDIO_TYPES):
        raise InvalidInputError(f"unsupported audio content type {content_type!r}")
    data = await _read_limited(audio, svc.settings.max_audio_bytes)
    if not data:
        raise InvalidInputError("audio file is empty")
    started = time.perf_counter()
    try:
        transcript = await svc.transcriber.transcribe(data, audio.content_type or content_type)
    finally:
        metrics.STAGE_SECONDS.labels("stt").observe(time.perf_counter() - started)
    if svc.settings.log_payloads:
        log.info("transcribed", text=transcript.text, confidence=transcript.confidence)
    else:
        log.info("transcribed", chars=len(transcript.text), confidence=transcript.confidence)
    return transcript


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks, total = [], 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(f"audio exceeds the {limit // (1024 * 1024)} MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


async def _answer(
    svc: Services,
    request: Request,
    question: str,
    history: Sequence[HistoryTurn],
    *,
    from_speech: bool,
    transcript: Transcript | None = None,
    extra_timings: dict[str, float] | None = None,
) -> QueryResponse:
    route = "voice_query" if transcript else "query"
    started = time.perf_counter()
    outcome: AgentOutcome | None = None
    try:
        async for event in svc.agent.run(question, history=history, from_speech=from_speech):
            if event.type == "answer":
                outcome = event.data["outcome"]
    except SpeechToSQLError as exc:
        metrics.REQUESTS.labels(route, exc.code).inc()
        raise
    if outcome is None:  # pragma: no cover - run() always ends with an answer or raises
        raise SpeechToSQLError("agent produced no answer")
    agent_s = time.perf_counter() - started
    timings = {**(extra_timings or {}), "agent": agent_s * 1000}
    timings["total"] = sum(timings.values())
    response = QueryResponse.build(
        request_id=request.state.request_id,
        question=question,
        outcome=outcome,
        timings_ms=timings,
        transcript=transcript,
    )
    _record(svc, route, response, agent_s)
    return response


def _record(svc: Services, route: str, response: QueryResponse, agent_s: float) -> None:
    metrics.STAGE_SECONDS.labels("agent").observe(agent_s)
    metrics.REQUESTS.labels(route, response.status).inc()
    fields: dict[str, Any] = {
        "status": response.status,
        "model": response.model,
        "tool_calls": response.tool_calls,
        "rows": response.row_count,
        "llm_calls": response.usage.llm_calls,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_tokens": response.usage.cache_read_tokens,
        "agent_ms": round(agent_s * 1000, 1),
        "sql": response.sql,
    }
    if svc.settings.log_payloads:
        fields["question"] = response.question
    log.info("query_completed", **fields)


def _parse_history(raw: str | None) -> list[HistoryTurn]:
    if not raw:
        return []
    try:
        items = _history_adapter.validate_json(raw)
    except ValidationError as exc:
        raise InvalidInputError(f"invalid history: {exc.errors()[0]['msg']}") from exc
    return [item.to_turn() for item in items[:20]]


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _with_heartbeat(stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Re-yield ``stream``, inserting ``None`` whenever it is silent for a while.

    The stream is consumed by a single background task (so task-bound constructs inside
    it, like ``asyncio.timeout``, keep working); heartbeats keep proxies from closing an
    idle connection while the model is thinking.
    """
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for item in stream:
                await queue.put(("item", item))
        except Exception as exc:  # noqa: BLE001 - re-raised in the consumer below
            await queue.put(("error", exc))
        else:
            await queue.put(("done", None))
        finally:
            await stream.aclose()  # type: ignore[attr-defined]

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_S)
            except TimeoutError:
                if task.done() and queue.empty():  # producer died without reporting (e.g. cancelled)
                    raise AgentError("the agent stopped unexpectedly") from None
                yield None
                continue
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
