"""Prometheus metrics (exposed at /metrics)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUESTS = Counter("s2s_requests_total", "API requests by route and outcome", ["route", "outcome"])
STAGE_SECONDS = Histogram(
    "s2s_stage_seconds",
    "Latency of pipeline stages",
    ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120),
)
LLM_TOKENS = Counter(
    "s2s_llm_tokens_total",
    "Claude tokens by kind (input, output, cache_read, cache_write)",
    ["kind"],
)
LLM_CALLS = Counter("s2s_llm_calls_total", "Claude Messages API calls", ["stop_reason"])
TOOL_CALLS = Counter("s2s_tool_calls_total", "Agent tool invocations", ["tool", "outcome"])
SQL_REJECTED = Counter("s2s_sql_rejected_total", "Statements rejected by the SQL guard", ["reason"])
AGENT_OUTCOMES = Counter("s2s_agent_outcomes_total", "Final agent status", ["status"])
