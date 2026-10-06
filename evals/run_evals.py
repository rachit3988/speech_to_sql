"""Execution-accuracy evals for the text-to-SQL agent (bypasses speech-to-text).

Each case is a spoken-style question plus either a gold SQL query (the agent passes when
its result matches the gold result) or the statuses it must return (e.g. refusing a write).
Results are compared column-by-column as multisets with a small numeric tolerance, so
column order, extra columns and formatting of numbers don't cause false failures.

This calls the Claude API once per agent step (typically 2-5 calls per case) and costs money.

    # against the Postgres demo DB (see README):
    S2S_DATABASE_URL=postgresql://localhost/speech_demo uv run python evals/run_evals.py
    uv run python evals/run_evals.py --only top5_customers,acme_orders --effort low
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from speech_to_sql.agent import SQLAgent
from speech_to_sql.config import Settings
from speech_to_sql.db import CatalogProvider, QueryResult, create_database
from speech_to_sql.errors import SpeechToSQLError
from speech_to_sql.logging_setup import configure_logging
from speech_to_sql.runtime import create_llm_client

# USD per million tokens for the default model (claude-opus-5-5); used for the cost estimate only.
PRICE = {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00}
HERE = Path(__file__).resolve().parent


@dataclass
class CaseResult:
    id: str
    question: str
    passed: bool
    reason: str
    status: str | None = None
    sql: str | None = None
    latency_s: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: dict[str, int] = field(default_factory=dict)


# --- result comparison ----------------------------------------------------------------


def _norm(value: Any) -> tuple[int, Any]:
    if value is None:
        return (2, "")
    if isinstance(value, (bool, int, float)):
        return (0, float(value))
    text = str(value).strip()
    try:
        return (0, float(text))
    except ValueError:
        return (1, text.lower())


def _close(a: tuple[int, Any], b: tuple[int, Any]) -> bool:
    if a[0] == b[0] == 0:
        return math.isclose(a[1], b[1], rel_tol=0.01, abs_tol=0.01)
    return a == b


def _columns_equal(gold: list[Any], pred: list[Any]) -> bool:
    g, p = sorted(map(_norm, gold)), sorted(map(_norm, pred))
    return len(g) == len(p) and all(_close(x, y) for x, y in zip(g, p, strict=True))


def compare(gold: QueryResult, pred: QueryResult | None, match_columns: list[str] | None) -> tuple[bool, str]:
    if pred is None:
        return False, "agent returned no result set"
    if gold.row_count != pred.row_count:
        return False, f"row count {pred.row_count} != expected {gold.row_count}"
    gold_cols = [list(c) for c in zip(*gold.rows, strict=True)] if gold.rows else [[] for _ in gold.columns]
    pred_cols = [list(c) for c in zip(*pred.rows, strict=True)] if pred.rows else [[] for _ in pred.columns]
    wanted = [gold.columns.index(c) for c in match_columns] if match_columns else range(len(gold.columns))
    used: set[int] = set()
    for gi in wanted:
        match = next(
            (pi for pi, col in enumerate(pred_cols) if pi not in used and _columns_equal(gold_cols[gi], col)),
            None,
        )
        if match is None:
            return False, f"no column matches expected column {gold.columns[gi]!r}"
        used.add(match)
    return True, "result matches"


# --- runner ---------------------------------------------------------------------------


async def run_case(agent: SQLAgent, db: Any, case: dict[str, Any], sem: asyncio.Semaphore) -> CaseResult:
    async with sem:
        started = time.perf_counter()
        result = CaseResult(id=case["id"], question=case["question"], passed=False, reason="")
        try:
            outcome = None
            async for event in agent.run(case["question"], from_speech=True):
                if event.type == "answer":
                    outcome = event.data["outcome"]
            assert outcome is not None
        except SpeechToSQLError as exc:
            result.reason = f"{exc.code}: {exc.message}"
            result.latency_s = time.perf_counter() - started
            return result
        result.latency_s = time.perf_counter() - started
        result.status = outcome.final.status
        result.sql = outcome.sql
        result.llm_calls = outcome.usage.llm_calls
        result.tool_calls = outcome.tool_calls
        result.tokens = {
            "input": outcome.usage.input_tokens,
            "output": outcome.usage.output_tokens,
            "cache_read": outcome.usage.cache_read_tokens,
            "cache_write": outcome.usage.cache_write_tokens,
        }
        if "expect_status" in case:
            result.passed = outcome.final.status in case["expect_status"]
            result.reason = f"status {outcome.final.status}"
        elif outcome.final.status != "answered":
            result.reason = f"status {outcome.final.status}: {outcome.final.answer[:120]}"
        else:
            gold = await db.execute(case["gold_sql"], max_rows=10_000, timeout_s=30)
            result.passed, result.reason = compare(gold, outcome.result, case.get("match_columns"))
        return result


def cost_usd(tokens: dict[str, int]) -> float:
    return sum(tokens.get(k, 0) * PRICE[k] for k in PRICE) / 1_000_000


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.jsonl")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path, help="write JSON results here (default: evals/results/<ts>.json)")
    args = parser.parse_args()

    overrides: dict[str, Any] = {"log_level": "WARNING"}
    if args.effort:
        overrides["llm_effort"] = args.effort
    if args.model:
        overrides["llm_model"] = args.model
    settings = Settings(**overrides)
    # The privacy case expects customers.email to be unavailable, as with the read-only role.
    settings.hidden_columns = sorted({*settings.hidden_columns, "customers.email"})
    configure_logging(settings.log_level, json_logs=False)

    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    db = create_database(settings)
    await db.connect()
    client = create_llm_client(settings)
    agent = SQLAgent(client, db, CatalogProvider(db.introspect, ttl_s=3600), settings)
    sem = asyncio.Semaphore(args.concurrency)
    try:
        results = await asyncio.gather(*(run_case(agent, db, c, sem) for c in cases))
    finally:
        await client.close()
        await db.close()

    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {r.id:<26} {r.latency_s:5.1f}s  calls={r.llm_calls}  {r.reason}")
    total_tokens = {k: sum(r.tokens.get(k, 0) for r in results) for k in PRICE}
    latencies = sorted(r.latency_s for r in results)
    passed = sum(r.passed for r in results)
    summary = {
        "model": settings.llm_model,
        "effort": settings.llm_effort,
        "cases": len(results),
        "passed": passed,
        "accuracy": round(passed / max(len(results), 1), 3),
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_s": round(latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)], 2) if latencies else None,
        "tokens": total_tokens,
        "est_cost_usd": round(cost_usd(total_tokens), 4),
    }
    print(json.dumps(summary, indent=2))

    out = args.out or HERE / "results" / f"{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2))
    print(f"wrote {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
