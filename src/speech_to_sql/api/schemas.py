from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from speech_to_sql.agent import AgentOutcome, HistoryTurn
from speech_to_sql.stt import Transcript


class HistoryItem(BaseModel):
    question: str = Field(max_length=2000)
    sql: str | None = Field(default=None, max_length=10_000)
    answer: str | None = Field(default=None, max_length=4000)

    def to_turn(self) -> HistoryTurn:
        return HistoryTurn(question=self.question, sql=self.sql, answer=self.answer)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)
    from_speech: bool = Field(
        default=False, description="True when the question is a (possibly edited) speech transcript."
    )


class TranscriptOut(BaseModel):
    text: str
    confidence: float | None
    duration_s: float | None
    provider: str

    @classmethod
    def from_transcript(cls, t: Transcript) -> TranscriptOut:
        return cls(text=t.text, confidence=t.confidence, duration_s=t.duration_s, provider=t.provider)


class UsageOut(BaseModel):
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


class QueryResponse(BaseModel):
    request_id: str
    question: str
    status: Literal["answered", "needs_clarification", "cannot_answer"]
    answer: str
    clarification_question: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    model: str
    tool_calls: int
    usage: UsageOut
    timings_ms: dict[str, float] = Field(default_factory=dict)
    transcript: TranscriptOut | None = None

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        question: str,
        outcome: AgentOutcome,
        timings_ms: dict[str, float],
        transcript: Transcript | None = None,
    ) -> QueryResponse:
        final, result, u = outcome.final, outcome.result, outcome.usage
        return cls(
            request_id=request_id,
            question=question,
            status=final.status,
            answer=final.answer,
            clarification_question=final.clarification_question,
            assumptions=final.assumptions,
            sql=outcome.sql,
            columns=result.columns if result else [],
            rows=result.rows if result else [],
            row_count=result.row_count if result else 0,
            truncated=result.truncated if result else False,
            model=outcome.model,
            tool_calls=outcome.tool_calls,
            usage=UsageOut(
                llm_calls=u.llm_calls,
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_tokens=u.cache_read_tokens,
                cache_write_tokens=u.cache_write_tokens,
            ),
            timings_ms={k: round(v, 1) for k, v in timings_ms.items()},
            transcript=TranscriptOut.from_transcript(transcript) if transcript else None,
        )


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
