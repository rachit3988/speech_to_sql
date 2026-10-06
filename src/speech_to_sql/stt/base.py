from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    confidence: float | None = None
    duration_s: float | None = None
    provider: str = ""


class Transcriber(Protocol):
    """Speech-to-text backend. Implementations must be safe for concurrent use."""

    name: str

    async def transcribe(self, audio: bytes, content_type: str) -> Transcript: ...

    async def close(self) -> None: ...
