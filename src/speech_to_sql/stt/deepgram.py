"""Deepgram pre-recorded transcription over its REST API.

Uses the REST endpoint directly (rather than the SDK) to keep the dependency surface
small and stable. Domain vocabulary is boosted with Nova-3 keyterm prompting.
"""

from __future__ import annotations

import asyncio
import random

import httpx
import structlog

from speech_to_sql.errors import EmptyTranscriptError, TranscriptionError
from speech_to_sql.stt.base import Transcript

log = structlog.get_logger(__name__)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class DeepgramTranscriber:
    name = "deepgram"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-3",
        language: str = "en",
        keyterms: list[str] | None = None,
        base_url: str = "https://api.deepgram.com",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._params: list[tuple[str, str | int | float | bool | None]] = [
            ("model", model),
            ("language", language),
            ("smart_format", "true"),
            ("punctuate", "true"),
        ]
        # Keyterm prompting: repeat the parameter per term (Nova-3 / Flux only).
        self._params += [("keyterm", term) for term in (keyterms or [])[:100]]
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Token {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def transcribe(self, audio: bytes, content_type: str) -> Transcript:
        response = await self._post_with_retries(audio, content_type)
        try:
            payload = response.json()
            alt = payload["results"]["channels"][0]["alternatives"][0]
            text = str(alt.get("transcript", "")).strip()
            confidence = alt.get("confidence")
            duration = payload.get("metadata", {}).get("duration")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise TranscriptionError("unexpected response from Deepgram") from exc
        if not text:
            raise EmptyTranscriptError("no speech was detected in the audio")
        return Transcript(
            text=text,
            confidence=float(confidence) if confidence is not None else None,
            duration_s=float(duration) if duration is not None else None,
            provider=self.name,
        )

    async def _post_with_retries(self, audio: bytes, content_type: str) -> httpx.Response:
        attempt = 0
        while True:
            try:
                response = await self._client.post(
                    "/v1/listen",
                    params=self._params,
                    content=audio,
                    headers={"Content-Type": content_type or "application/octet-stream"},
                )
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    raise TranscriptionError(f"Deepgram unreachable: {exc.__class__.__name__}") from exc
                delay = self._backoff(attempt, None)
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in _RETRYABLE_STATUS or attempt >= self._max_retries:
                    raise TranscriptionError(
                        f"Deepgram returned HTTP {response.status_code}: {_error_detail(response)}"
                    )
                delay = self._backoff(attempt, response.headers.get("retry-after"))
            attempt += 1
            log.warning("deepgram_retry", attempt=attempt, delay_s=round(delay, 2))
            await asyncio.sleep(delay)

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 10.0)
            except ValueError:
                pass
        return min(0.5 * 2.0**attempt, 8.0) + random.uniform(0, 0.25)  # noqa: S311 - jitter, not crypto


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        return str(body.get("err_msg") or body.get("message") or body)[:300]
    except ValueError:
        return response.text[:300]
