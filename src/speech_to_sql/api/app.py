"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.mcpserver import MCPServer
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Route

from speech_to_sql.api.middleware import RequestContextMiddleware
from speech_to_sql.api.routes import api, public
from speech_to_sql.api.schemas import ErrorBody, ErrorResponse
from speech_to_sql.config import Settings, get_settings
from speech_to_sql.db import Database
from speech_to_sql.errors import SpeechToSQLError
from speech_to_sql.logging_setup import configure_logging
from speech_to_sql.mcp_server import build_mcp_server, mount_http
from speech_to_sql.runtime import Services, open_services
from speech_to_sql.stt import Transcriber

log = structlog.get_logger(__name__)

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _error(
    request: Request, status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, request_id=getattr(request.state, "request_id", None))
    )
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


def create_app(
    settings: Settings | None = None,
    *,
    db: Database | None = None,
    transcriber: Transcriber | None = None,
    llm: anthropic.AsyncAnthropic | None = None,
) -> FastAPI:
    """Build the app. ``db``/``transcriber``/``llm`` overrides exist for tests and embedding."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.environment == "prod")
    if settings.environment == "prod" and not settings.auth_enabled:
        log.warning("auth_disabled_in_prod", hint="set S2S_API_KEYS")

    mcp_server: MCPServer | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with open_services(settings, db=db, transcriber=transcriber, llm=llm) as services:
            app.state.services = services
            if mcp_server is None:
                yield
            else:
                async with mcp_server.session_manager.run():
                    yield

    app = FastAPI(
        title="Speech-to-SQL Agent",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment == "dev" else None,
        redoc_url=None,
    )

    @app.exception_handler(SpeechToSQLError)
    async def _domain_error(request: Request, exc: SpeechToSQLError) -> JSONResponse:
        log.warning("request_failed", code=exc.code, error=exc.message)
        return _error(request, exc.status_code, exc.code, exc.message, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(request, exc.status_code, "http_error", str(exc.detail), dict(exc.headers or {}))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(request, 422, "invalid_request", f"{loc}: {first.get('msg', 'invalid request')}")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error")
        return _error(request, 500, "internal_error", "internal error")

    app.include_router(public)
    app.include_router(api)

    def services() -> Services:
        current: Services = app.state.services
        return current

    if settings.mcp_enabled:
        # Other agents connect here (Streamable HTTP, stateless). Registered before the UI mount at "/".
        mcp_server = build_mcp_server(services, settings)
        app.router.routes.append(Route("/mcp", endpoint=mount_http(mcp_server, settings, services)))

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )
    app.add_middleware(RequestContextMiddleware)

    if settings.serve_web_ui and _WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
    return app
