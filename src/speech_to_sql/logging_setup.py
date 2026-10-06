"""Structured logging: JSON lines in prod, readable console output in dev."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

import structlog


def configure_logging(level: str = "INFO", *, json_logs: bool = True, stream: TextIO | None = None) -> None:
    """Configure structlog + stdlib logging. Pass ``stream=sys.stderr`` when stdout carries a protocol (MCP stdio)."""
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info if json_logs else (lambda _l, _m, e: e),
            renderer,
        ],
    )
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Route uvicorn's loggers through the same handler; our middleware logs access lines.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "httpcore", "httpx2", "httpcore2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
