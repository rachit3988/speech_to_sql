"""Tools the agent can call. Each request gets its own tool instances bound to a RunContext."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from anthropic import beta_async_tool
from anthropic.lib.tools import BetaAsyncFunctionTool, ToolError

from speech_to_sql import metrics
from speech_to_sql.db.base import Database, QueryResult
from speech_to_sql.db.catalog import SchemaCatalog
from speech_to_sql.errors import DatabaseUnavailableError, QueryExecutionError, UnsafeSQLError
from speech_to_sql.sql.guard import SQLGuard

log = structlog.get_logger(__name__)

_MAX_CELL_CHARS = 80
_MAX_DESCRIBE_TABLES = 20


@dataclass(frozen=True, slots=True)
class AgentEvent:
    type: str  # thinking | tool_call | tool_result | answer | error
    data: dict[str, Any]


@dataclass(slots=True)
class RunContext:
    db: Database
    catalog: SchemaCatalog
    guard: SQLGuard
    max_rows: int
    preview_rows: int
    statement_timeout_s: float
    # Validated SQL -> full result, so the final answer's query is not run twice.
    executed: dict[str, QueryResult] = field(default_factory=dict)
    events: list[AgentEvent] = field(default_factory=list)
    # Set when a tool hits an infrastructure failure; the agent aborts instead of retrying.
    fatal: Exception | None = None

    def emit(self, type_: str, **data: Any) -> None:
        self.events.append(AgentEvent(type_, data))

    def drain(self) -> list[AgentEvent]:
        events, self.events = self.events, []
        return events


def format_result(result: QueryResult, *, preview_rows: int, sql: str) -> str:
    shown = result.rows[:preview_rows]
    lines = [
        f"OK: {result.row_count} row(s){' (truncated at the row cap)' if result.truncated else ''}"
        f" in {result.elapsed_ms:.0f} ms. Executed SQL: {sql}",
        " | ".join(result.columns),
    ]
    for row in shown:
        lines.append(" | ".join(_cell(v) for v in row))
    if result.row_count > len(shown):
        lines.append(f"... {result.row_count - len(shown)} more row(s) not shown")
    if result.row_count == 0:
        lines.append("(no rows)")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= _MAX_CELL_CHARS else text[: _MAX_CELL_CHARS - 1] + "…"


def build_tools(ctx: RunContext) -> list[BetaAsyncFunctionTool[Any]]:
    @beta_async_tool(strict=True)
    async def describe_tables(table_names: list[str]) -> str:
        """Get the columns, types, keys and comments of one or more tables.

        Call this before writing SQL against a table whose columns are not listed in the schema.

        Args:
            table_names: Table names, optionally schema-qualified (e.g. "orders" or "public.orders").
        """
        if not table_names:
            raise ToolError("Provide at least one table name.")
        parts, unknown = [], []
        for name in table_names[:_MAX_DESCRIBE_TABLES]:
            table = ctx.catalog.resolve(name)
            if table is None:
                unknown.append(name)
            else:
                parts.append(ctx.catalog.render_table(table))
        if unknown:
            parts.append(
                f"Unknown tables: {', '.join(unknown)}. Available: {', '.join(ctx.catalog.table_names()[:100])}"
            )
        metrics.TOOL_CALLS.labels("describe_tables", "ok").inc()
        ctx.emit("tool_result", tool="describe_tables", summary=f"described {len(table_names)} table(s)")
        return "\n\n".join(parts)

    @beta_async_tool(strict=True)
    async def search_column_values(table: str, column: str, search_term: str) -> str:
        """Find how a value mentioned in the question is actually stored in a column.

        Call this before filtering on names, companies, products, cities, statuses or other
        free-text values, because the spoken form or transcription may differ from the stored
        form. Matching is case-insensitive and tolerant of partial or misheard words.

        Args:
            table: Table name, optionally schema-qualified.
            column: Column to search (a text-like column).
            search_term: The value as heard in the question, e.g. "acme corp".
        """
        info = ctx.catalog.resolve(table)
        if info is None:
            raise ToolError(f"Unknown table {table!r}. Available: {', '.join(ctx.catalog.table_names()[:100])}")
        if info.column(column) is None:
            cols = ", ".join(c.name for c in info.columns)
            raise ToolError(f"Unknown column {column!r} on {info.qualified_name}. Columns: {cols}")
        term = search_term.strip()
        if not term:
            raise ToolError("search_term must not be empty.")
        try:
            values = await ctx.db.search_values(info, column, term, limit=10, timeout_s=ctx.statement_timeout_s)
        except QueryExecutionError as exc:
            metrics.TOOL_CALLS.labels("search_column_values", "error").inc()
            raise ToolError(exc.message) from exc
        except DatabaseUnavailableError as exc:
            ctx.fatal = exc
            raise ToolError("Database unavailable.") from exc
        metrics.TOOL_CALLS.labels("search_column_values", "ok").inc()
        ctx.emit(
            "tool_result",
            tool="search_column_values",
            summary=f"{len(values)} match(es) for {term!r} in {info.name}.{column}",
        )
        if not values:
            return f"No values in {info.qualified_name}.{column} resemble {term!r}."
        return f"Closest stored values in {info.qualified_name}.{column}:\n" + "\n".join(repr(v) for v in values)

    @beta_async_tool(strict=True)
    async def run_sql(sql: str) -> str:
        """Execute one read-only SELECT statement and return a preview of the result.

        Use this to test every query, and always run the query you give as the final answer.
        A row cap is applied automatically. On error, the message explains what to fix.

        Args:
            sql: A single SELECT statement (WITH and UNION are allowed).
        """
        try:
            validated = ctx.guard.validate(sql, ctx.catalog)
        except UnsafeSQLError as exc:
            metrics.SQL_REJECTED.labels(exc.reason).inc()
            metrics.TOOL_CALLS.labels("run_sql", "rejected").inc()
            ctx.emit("tool_result", tool="run_sql", ok=False, error=exc.message)
            raise ToolError(f"Rejected: {exc.message}") from exc

        result = ctx.executed.get(validated.sql)
        if result is None:
            try:
                result = await ctx.db.execute(validated.sql, max_rows=ctx.max_rows, timeout_s=ctx.statement_timeout_s)
            except QueryExecutionError as exc:
                metrics.TOOL_CALLS.labels("run_sql", "error").inc()
                ctx.emit("tool_result", tool="run_sql", ok=False, sql=validated.sql, error=exc.message)
                raise ToolError(f"Database error: {exc.message}") from exc
            except DatabaseUnavailableError as exc:
                ctx.fatal = exc
                raise ToolError("Database unavailable.") from exc
            ctx.executed[validated.sql] = result

        metrics.TOOL_CALLS.labels("run_sql", "ok").inc()
        ctx.emit(
            "tool_result",
            tool="run_sql",
            ok=True,
            sql=validated.sql,
            row_count=result.row_count,
            elapsed_ms=round(result.elapsed_ms, 1),
        )
        return format_result(result, preview_rows=ctx.preview_rows, sql=validated.sql)

    return [describe_tables, search_column_values, run_sql]
