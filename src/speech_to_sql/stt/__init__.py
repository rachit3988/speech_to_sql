import structlog

from speech_to_sql.config import Settings
from speech_to_sql.stt.base import Transcriber, Transcript

log = structlog.get_logger(__name__)

__all__ = ["Transcriber", "Transcript", "create_transcriber"]


def create_transcriber(settings: Settings) -> Transcriber | None:
    """Build the configured speech-to-text backend (``None`` = voice input disabled)."""
    if settings.stt_provider == "none":
        return None
    if settings.stt_provider == "deepgram":
        if settings.deepgram_api_key is None:
            if settings.environment == "dev":
                log.warning("voice_disabled", reason="DEEPGRAM_API_KEY is not set")
                return None
            raise ValueError("DEEPGRAM_API_KEY is required when S2S_STT_PROVIDER=deepgram")
        from speech_to_sql.stt.deepgram import DeepgramTranscriber

        return DeepgramTranscriber(
            settings.deepgram_api_key.get_secret_value(),
            model=settings.deepgram_model,
            language=settings.deepgram_language,
            keyterms=settings.stt_keyterms,
            base_url=settings.deepgram_base_url,
            timeout_s=settings.stt_timeout_s,
            max_retries=settings.stt_max_retries,
        )
    raise ValueError(f"unknown STT provider {settings.stt_provider!r}")
