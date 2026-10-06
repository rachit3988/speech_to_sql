"""Command-line entry point (configuration via S2S_* environment variables / .env).

speech-to-sql            run the HTTP service: REST API, web UI and MCP endpoint at /mcp
speech-to-sql serve      same as above
speech-to-sql mcp        run as an MCP server over stdio, for a local agent that launches it
"""

from __future__ import annotations

import argparse
import asyncio
import os

import uvicorn


def serve() -> None:
    uvicorn.run(
        "speech_to_sql.api:create_app",
        factory=True,
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104 - container default
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        timeout_graceful_shutdown=30,
        log_config=None,  # logging is configured by the app (structlog)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="speech-to-sql", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the HTTP service (default)")
    sub.add_parser("mcp", help="run as an MCP server over stdio")
    args = parser.parse_args()

    if args.command == "mcp":
        from speech_to_sql.mcp_server import run_stdio

        asyncio.run(run_stdio())
    else:
        serve()


if __name__ == "__main__":
    main()
