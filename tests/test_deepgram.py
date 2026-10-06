from __future__ import annotations

from typing import Any

import httpx
import pytest

from speech_to_sql.errors import EmptyTranscriptError, TranscriptionError
from speech_to_sql.stt.deepgram import DeepgramTranscriber


def dg_response(text: str, confidence: float = 0.97) -> dict[str, Any]:
    return {
        "metadata": {"duration": 3.2},
        "results": {"channels": [{"alternatives": [{"transcript": text, "confidence": confidence}]}]},
    }


def make(handler: Any, **kw: Any) -> DeepgramTranscriber:
    return DeepgramTranscriber("dg-key", transport=httpx.MockTransport(handler), keyterms=["EchoBuds", "Globex"], **kw)


async def test_transcribes_with_keyterms_and_auth() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=dg_response("top five customers by revenue"))

    stt = make(handler)
    result = await stt.transcribe(b"audio-bytes", "audio/webm")
    await stt.close()

    assert result.text == "top five customers by revenue"
    assert result.confidence == pytest.approx(0.97)
    assert result.duration_s == pytest.approx(3.2)
    req = seen[0]
    assert req.url.path == "/v1/listen"
    assert req.headers["authorization"] == "Token dg-key"
    assert req.headers["content-type"] == "audio/webm"
    assert req.url.params.get_list("keyterm") == ["EchoBuds", "Globex"]
    assert req.url.params["model"] == "nova-3"
    assert req.content == b"audio-bytes"


async def test_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("speech_to_sql.stt.deepgram.asyncio.sleep", _no_sleep)
    responses = [
        httpx.Response(503),
        httpx.Response(429, headers={"retry-after": "1"}),
        httpx.Response(200, json=dg_response("ok")),
    ]
    stt = make(lambda _r: responses.pop(0), max_retries=2)
    assert (await stt.transcribe(b"a", "audio/wav")).text == "ok"


async def test_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("speech_to_sql.stt.deepgram.asyncio.sleep", _no_sleep)
    stt = make(lambda _r: httpx.Response(503), max_retries=1)
    with pytest.raises(TranscriptionError, match="HTTP 503"):
        await stt.transcribe(b"a", "audio/wav")


async def test_does_not_retry_client_errors() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, json={"err_msg": "Invalid credentials."})

    with pytest.raises(TranscriptionError, match="Invalid credentials"):
        await make(handler).transcribe(b"a", "audio/wav")
    assert len(calls) == 1


async def test_empty_transcript() -> None:
    with pytest.raises(EmptyTranscriptError):
        await make(lambda _r: httpx.Response(200, json=dg_response("  "))).transcribe(b"a", "audio/wav")


async def test_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("speech_to_sql.stt.deepgram.asyncio.sleep", _no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(TranscriptionError, match="unreachable"):
        await make(handler, max_retries=1).transcribe(b"a", "audio/wav")


async def _no_sleep(_s: float) -> None:
    return None
