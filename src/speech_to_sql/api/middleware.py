"""Pure-ASGI middleware (safe with streaming responses): request IDs and access logs."""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = structlog.get_logger("speech_to_sql.access")

_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.perf_counter()
        status_code = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            path = scope.get("path", "")
            if path not in ("/healthz", "/readyz", "/metrics"):
                log.info(
                    "http_request",
                    method=scope.get("method"),
                    path=path,
                    status=status_code,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            structlog.contextvars.clear_contextvars()
