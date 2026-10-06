"""MCP server: lets other agents use this agent (and its guarded SQL tools) as tools.

Served over Streamable HTTP at ``/mcp`` next to the REST API, and over stdio with
``speech-to-sql mcp`` for local agents (Claude Code, Claude Desktop, IDEs). Every tool is
read-only and goes through the same SQL guard and read-only database role as the REST API.

Note: no ``from __future__ import annotations`` here. The MCP SDK resolves tool signatures
(including the injected ``Context``) from real annotation objects.
"""

import sys
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal

import structlog
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from speech_to_sql import metrics
from speech_to_sql.agent import AgentEvent, AgentOutcome
from speech_to_sql.config import Settings, get_settings
from speech_to_sql.errors import (
    AuthenticationError,
    QueryExecutionError,
    SpeechToSQLError,
    UnsafeSQLError,
)
from speech_to_sql.logging_setup import configure_logging
from speech_to_sql.runtime import Services, open_services
from speech_to_sql.security import Principal

log = structlog.get_logger(__name__)

_INSTRUCTIONS = """\
Read-only access to {data}.

Use ask_database whenever a task needs facts, numbers, lists or trends from this data. It works out \
the SQL, runs it, checks the result and returns a plain-language answer with the rows and the SQL it \
used. Ask one focused question per call, with any filters, time ranges and units spelled out.
{raw_sql}
Nothing here can modify data."""

_RAW_SQL_HINT = """
To write SQL yourself instead, call describe_database for the schema, find_values to see how a value \
is stored, and run_sql to execute one read-only SELECT ({dialect}).
"""


class AskResult(BaseModel):
    status: Literal["answered", "needs_clarification", "cannot_answer"]
    answer: str = Field(description="Plain-language answer.")
    clarification_question: str | None = Field(description="Set when status is needs_clarification.")
    assumptions: list[str] = Field(description="Interpretations made to answer the question.")
    sql: str | None = Field(description="The query that produced the rows.")
    columns: list[str]
    rows: list[list[Any]]
    row_count: int = Field(description="Number of rows included here.")
    truncated: bool = Field(description="True if more rows exist than are included.")


class SQLResult(BaseModel):
    sql: str = Field(description="The statement as executed (after validation and row capping).")
    columns: list[str]
    rows: list[list[Any]]
    row_count: int = Field(description="Number of rows included here.")
    truncated: bool = Field(description="True if more rows exist than are included.")
    elapsed_ms: float


def _principal(svc: Services, ctx: Context) -> Principal:
    request = ctx.request_context.request
    if request is None:  # stdio: the local process that launched this server
        return Principal(id="stdio")
    client = getattr(request, "client", None)
    return svc.authenticator.authenticate(ctx.headers or {}, getattr(client, "host", None))


def _progress_message(event: AgentEvent) -> str | None:
    data = event.data
    if event.type == "tool_call":
        tool, args = data.get("tool"), data.get("input") or {}
        if tool == "run_sql":
            return "Running SQL"
        if tool == "search_column_values":
            return f"Looking up {args.get('search_term')!r} in {args.get('table')}.{args.get('column')}"
        if tool == "describe_tables":
            return f"Inspecting {', '.join(args.get('table_names') or [])}"
    if event.type == "tool_result" and data.get("tool") == "run_sql":
        return f"Query returned {data['row_count']} row(s)" if data.get("ok") else "Query failed; revising"
    return None


def build_mcp_server(services: Callable[[], Services], settings: Settings) -> MCPServer:
    """Build the MCP server. ``services`` is called per request (it is ready once startup completes)."""
    dialect = "SQLite" if settings.database_url.get_secret_value().startswith("sqlite") else "PostgreSQL"
    data = settings.mcp_data_description or "a SQL database (call describe_database to see what it contains)"
    server = MCPServer(
        name="speech-to-sql",
        title="SQL data analyst",
        description=f"Answers questions about {data} with read-only SQL.",
        version="0.1.0",
        instructions=_INSTRUCTIONS.format(
            data=data,
            raw_sql=_RAW_SQL_HINT.format(dialect=dialect) if settings.mcp_allow_raw_sql else "",
        ),
    )
    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
    max_rows_cap = settings.mcp_max_rows

    async def guarded(tool: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        """Run a tool body, turning expected failures into messages the calling agent can act on."""
        started = time.perf_counter()
        try:
            result = await fn()
        except SpeechToSQLError as exc:
            metrics.REQUESTS.labels(f"mcp_{tool}", exc.code).inc()
            log.info("mcp_tool_failed", tool=tool, code=exc.code, error=exc.message)
            raise ToolError(f"{exc.code}: {exc.message}") from exc
        metrics.REQUESTS.labels(f"mcp_{tool}", "ok").inc()
        log.info("mcp_tool_call", tool=tool, ms=round((time.perf_counter() - started) * 1000, 1))
        return result

    @server.tool(
        title="Ask the database",
        annotations=read_only,
        description=(
            f"Answer a question using live data from {data}. Call this whenever you need facts, numbers, "
            "lists, comparisons or trends from this data instead of guessing or asking the user. Pass one "
            "focused question in plain language. The tool writes, runs and checks the SQL itself and "
            "returns an answer, the rows, the SQL used and any assumptions; if it asks for clarification, "
            "resolve it and call again. Read-only; typically takes 5-60 seconds."
        ),
    )
    async def ask_database(
        ctx: Context,
        question: Annotated[
            str,
            Field(min_length=1, max_length=2000, description="The question, e.g. 'Top 5 products by revenue in 2025'."),
        ],
        context: Annotated[
            str | None,
            Field(
                max_length=4000,
                description="Optional background needed to interpret the question, e.g. earlier findings, "
                "definitions or the user's goal.",
            ),
        ] = None,
        max_rows: Annotated[int, Field(ge=1, le=max_rows_cap, description="Maximum rows to return.")] = 50,
    ) -> AskResult:
        async def body() -> AskResult:
            svc = services()
            svc.require_llm()
            svc.limiter.enforce(_principal(svc, ctx).id)
            outcome: AgentOutcome | None = None
            step = 0
            started = time.perf_counter()
            async for event in svc.agent.run(question, from_speech=False, context=context):
                if event.type == "answer":
                    outcome = event.data["outcome"]
                elif message := _progress_message(event):
                    step += 1
                    await ctx.report_progress(step, None, message)
            metrics.STAGE_SECONDS.labels("agent").observe(time.perf_counter() - started)
            if outcome is None:  # pragma: no cover - run() always ends with an answer or raises
                raise SpeechToSQLError("agent produced no answer")
            final, result = outcome.final, outcome.result
            rows = result.rows if result else []
            return AskResult(
                status=final.status,
                answer=final.answer,
                clarification_question=final.clarification_question,
                assumptions=final.assumptions,
                sql=outcome.sql,
                columns=result.columns if result else [],
                rows=rows[:max_rows],
                row_count=min(len(rows), max_rows),
                truncated=len(rows) > max_rows or bool(result and result.truncated),
            )

        result: AskResult = await guarded("ask_database", body)
        return result

    @server.tool(
        title="Describe the database",
        annotations=read_only,
        description=(
            f"Show the schema of {data}: tables, columns, types, keys and comments. Call this to check "
            "whether the data you need exists before asking, or before writing SQL with run_sql. Pass "
            "table names to see just those tables."
        ),
    )
    async def describe_database(
        tables: Annotated[
            list[str] | None, Field(description="Optional table names; omit to describe everything.")
        ] = None,
    ) -> str:
        async def body() -> str:
            catalog = await services().catalogs.get()
            header = f"Dialect: {dialect}\n\n"
            if not tables:
                text, complete = catalog.render(max_chars=settings.max_schema_prompt_chars)
                if not complete:
                    text += "\n\n(Large schema: only table names are listed. Pass table names for their columns.)"
                return header + text
            parts, unknown = [], []
            for name in tables[:50]:
                info = catalog.resolve(name)
                if info is None:
                    unknown.append(name)
                else:
                    parts.append(catalog.render_table(info))
            if unknown:
                parts.append(f"Unknown tables: {', '.join(unknown)}. Available: {', '.join(catalog.table_names())}")
            return header + "\n\n".join(parts)

        result: str = await guarded("describe_database", body)
        return result

    if not settings.mcp_allow_raw_sql:
        return server

    @server.tool(
        title="Find stored values",
        annotations=read_only,
        description=(
            "Find how a value is actually stored in a text column (case-insensitive, tolerant of partial "
            "or misspelled input), e.g. 'acme' -> 'Acme Corporation'. Use before filtering on names, "
            "places, products or statuses in run_sql."
        ),
    )
    async def find_values(
        table: Annotated[str, Field(description="Table name.")],
        column: Annotated[str, Field(description="Text column to search.")],
        search_term: Annotated[str, Field(min_length=1, max_length=200, description="The value to look for.")],
    ) -> list[str]:
        async def body() -> list[str]:
            svc = services()
            catalog = await svc.catalogs.get()
            info = catalog.resolve(table)
            if info is None:
                raise QueryExecutionError(f"unknown table {table!r}; available: {', '.join(catalog.table_names())}")
            if info.column(column) is None:
                cols = ", ".join(c.name for c in info.columns)
                raise QueryExecutionError(f"unknown column {column!r} on {info.qualified_name}; columns: {cols}")
            return await svc.db.search_values(
                info, column, search_term.strip(), limit=10, timeout_s=settings.statement_timeout_s
            )

        result: list[str] = await guarded("find_values", body)
        return result

    @server.tool(
        title="Run read-only SQL",
        annotations=ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        ),
        description=(
            f"Execute one read-only {dialect} SELECT (WITH/UNION allowed) against {data} and return the "
            "rows. Writes, DDL, system catalogs and unsafe functions are rejected with an explanation. "
            "Prefer ask_database unless you already know the schema and the exact query you need."
        ),
    )
    async def run_sql(
        ctx: Context,
        sql: Annotated[str, Field(min_length=1, max_length=20_000, description="A single SELECT statement.")],
        max_rows: Annotated[int, Field(ge=1, le=max_rows_cap, description="Maximum rows to return.")] = 50,
    ) -> SQLResult:
        async def body() -> SQLResult:
            svc = services()
            svc.limiter.enforce(_principal(svc, ctx).id)
            catalog = await svc.catalogs.get()
            try:
                validated = svc.guard.validate(sql, catalog)
            except UnsafeSQLError as exc:
                metrics.SQL_REJECTED.labels(exc.reason).inc()
                raise
            result = await svc.db.execute(validated.sql, max_rows=max_rows, timeout_s=settings.statement_timeout_s)
            return SQLResult(
                sql=validated.sql,
                columns=result.columns,
                rows=result.rows,
                row_count=result.row_count,
                truncated=result.truncated,
                elapsed_ms=round(result.elapsed_ms, 1),
            )

        result: SQLResult = await guarded("run_sql", body)
        return result

    return server


# --- HTTP integration ---------------------------------------------------------------


def transport_security(settings: Settings) -> TransportSecuritySettings:
    """DNS-rebinding protection for /mcp.

    Without API keys the endpoint only answers requests addressed to localhost (plus configured
    hosts), so a web page cannot reach a local dev server through a rebinding attack. With API
    keys, the host check applies only if hosts are configured.
    """
    hosts = settings.mcp_allowed_hosts
    if "*" in hosts or (settings.auth_enabled and not hosts):
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    local = ["127.0.0.1", "localhost", "[::1]"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[*local, *(f"{h}:*" for h in local), *hosts],
        allowed_origins=[
            *(f"http://{h}:*" for h in local),
            *(f"https://{h}" for h in hosts),
            *(f"http://{h}" for h in hosts),
        ],
    )


class MCPAuthGate:
    """Authenticates /mcp requests (X-API-Key or Bearer) before they reach the MCP transport."""

    def __init__(self, app: ASGIApp, services: Callable[[], Services]) -> None:
        self._app = app
        self._services = services

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            request = Request(scope)
            try:
                self._services().authenticator.authenticate(
                    request.headers, request.client.host if request.client else None
                )
            except AuthenticationError as exc:
                body = {"error": {"code": exc.code, "message": exc.message}}
                await JSONResponse(body, status_code=exc.status_code, headers=exc.headers)(scope, receive, send)
                return
        await self._app(scope, receive, send)


def mount_http(server: MCPServer, settings: Settings, services: Callable[[], Services]) -> ASGIApp:
    """Create the stateless Streamable HTTP endpoint (needs ``server.session_manager.run()`` at startup)."""
    server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,  # no session affinity needed: any replica can serve any request
        transport_security=transport_security(settings),
    )
    return MCPAuthGate(StreamableHTTPASGIApp(server.session_manager), services)


# --- stdio ----------------------------------------------------------------------------


async def run_stdio(settings: Settings | None = None) -> None:
    """Serve MCP over stdin/stdout for a local agent that launches this process."""
    settings = settings or get_settings()
    # stdout carries the protocol: every log line must go to stderr.
    configure_logging(settings.log_level, json_logs=True, stream=sys.stderr)
    async with open_services(settings, with_speech=False) as svc:
        await build_mcp_server(lambda: svc, settings).run_stdio_async()
