"""Long-lived service objects, shared by every front end (REST API, MCP over HTTP, MCP over stdio)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anthropic
import structlog

from speech_to_sql.agent import LLMStatus, SQLAgent, verify_llm
from speech_to_sql.config import Settings
from speech_to_sql.db import CatalogProvider, Database, create_database
from speech_to_sql.errors import LLMUnavailableError
from speech_to_sql.security import Authenticator, RateLimiter
from speech_to_sql.sql.guard import SQLGuard
from speech_to_sql.stt import Transcriber, create_transcriber

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class Services:
    settings: Settings
    db: Database
    catalogs: CatalogProvider
    agent: SQLAgent
    guard: SQLGuard
    transcriber: Transcriber | None
    llm: anthropic.AsyncAnthropic
    llm_status: LLMStatus
    authenticator: Authenticator
    limiter: RateLimiter

    def require_llm(self) -> None:
        if not self.llm_status.usable:
            raise LLMUnavailableError(f"the language model is not configured: {self.llm_status.detail}")


def create_llm_client(settings: Settings) -> anthropic.AsyncAnthropic:
    # Pass the key explicitly when it came from .env (pydantic-settings does not export .env
    # into os.environ, where the SDK looks); otherwise let the SDK resolve credentials itself.
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    if key:
        return anthropic.AsyncAnthropic(
            api_key=key, timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries
        )
    return anthropic.AsyncAnthropic(timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries)


@asynccontextmanager
async def open_services(
    settings: Settings,
    *,
    db: Database | None = None,
    transcriber: Transcriber | None = None,
    llm: anthropic.AsyncAnthropic | None = None,
    with_speech: bool = True,
) -> AsyncIterator[Services]:
    """Connect to the database, load the catalog, check the LLM, and build the agent.

    ``db``/``transcriber``/``llm`` overrides exist for tests and embedding.
    """
    database = db or create_database(settings)
    await database.connect()
    stt: Transcriber | None = None
    client: anthropic.AsyncAnthropic | None = None
    try:
        catalogs = CatalogProvider(database.introspect, ttl_s=settings.catalog_ttl_s)
        try:
            catalog = await catalogs.get()
            log.info("catalog_loaded", tables=len(catalog.tables))
        except Exception:
            # Start anyway; readiness reports not-ready until introspection succeeds.
            log.exception("catalog_load_failed")
        client = llm or create_llm_client(settings)
        llm_status = await verify_llm(client, settings.llm_model)
        if not llm_status.usable:
            log.error("llm_unavailable", detail=llm_status.detail)
        if transcriber is not None:
            stt = transcriber
        elif with_speech:
            stt = create_transcriber(settings)
        services = Services(
            settings=settings,
            db=database,
            catalogs=catalogs,
            agent=SQLAgent(client, database, catalogs, settings),
            guard=SQLGuard(database.dialect, settings.max_result_rows),
            transcriber=stt,
            llm=client,
            llm_status=llm_status,
            authenticator=Authenticator(settings),
            limiter=RateLimiter(settings.rate_limit_per_minute),
        )
        log.info(
            "startup_complete",
            model=settings.llm_model,
            effort=settings.llm_effort,
            llm=llm_status.detail,
            dialect=database.dialect,
            voice=stt is not None,
            auth=settings.auth_enabled,
        )
        yield services
    finally:
        if stt is not None:
            await stt.close()
        if client is not None:
            await client.close()
        await database.close()
        log.info("shutdown_complete")
