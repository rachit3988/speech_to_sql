"""Domain exceptions. The API layer maps these to HTTP status codes."""

from __future__ import annotations

import math


class SpeechToSQLError(Exception):
    """Base class for all expected (non-bug) failures."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    @property
    def headers(self) -> dict[str, str] | None:
        """Extra HTTP response headers for this error."""
        return None


class AuthenticationError(SpeechToSQLError):
    status_code = 401
    code = "unauthorized"

    @property
    def headers(self) -> dict[str, str] | None:
        return {"WWW-Authenticate": 'Bearer realm="speech-to-sql"'}


class RateLimitedError(SpeechToSQLError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"rate limit exceeded; retry in {math.ceil(retry_after_s)}s")
        self.retry_after_s = retry_after_s

    @property
    def headers(self) -> dict[str, str] | None:
        return {"Retry-After": str(math.ceil(self.retry_after_s))}


class InvalidInputError(SpeechToSQLError):
    status_code = 422
    code = "invalid_input"


class PayloadTooLargeError(SpeechToSQLError):
    status_code = 413
    code = "payload_too_large"


class TranscriptionError(SpeechToSQLError):
    status_code = 502
    code = "transcription_failed"


class EmptyTranscriptError(SpeechToSQLError):
    status_code = 422
    code = "empty_transcript"


class SpeechDisabledError(SpeechToSQLError):
    status_code = 501
    code = "speech_disabled"


class AgentError(SpeechToSQLError):
    status_code = 502
    code = "agent_failed"


class LLMUnavailableError(SpeechToSQLError):
    """The language model is not usable (missing/invalid credentials or unknown model)."""

    status_code = 503
    code = "llm_unavailable"


class AgentTimeoutError(AgentError):
    status_code = 504
    code = "agent_timeout"


class UnsafeSQLError(SpeechToSQLError):
    """The SQL guard rejected a statement."""

    status_code = 422
    code = "unsafe_sql"

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class QueryExecutionError(SpeechToSQLError):
    """The database rejected or failed a (guard-approved) statement."""

    status_code = 422
    code = "query_failed"


class QueryTimeoutError(QueryExecutionError):
    status_code = 504
    code = "query_timeout"


class DatabaseUnavailableError(SpeechToSQLError):
    status_code = 503
    code = "database_unavailable"
