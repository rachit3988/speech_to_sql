from __future__ import annotations

from evals.run_evals import compare
from speech_to_sql.db.base import QueryResult


def qr(columns: list[str], rows: list[list[object]]) -> QueryResult:
    return QueryResult(columns=columns, rows=rows, truncated=False, elapsed_ms=0)


def test_matches_with_reordered_and_extra_columns() -> None:
    gold = qr(["name", "revenue"], [["a", 10.0], ["b", 5.0]])
    pred = qr(["id", "revenue_usd", "customer"], [[2, 5.004, "B"], [1, 10, "A"]])
    assert compare(gold, pred, None) == (True, "result matches")


def test_match_columns_ignores_formatting_of_other_columns() -> None:
    gold = qr(["month", "orders"], [["2025-01-01T00:00:00", 3], ["2025-02-01T00:00:00", 4]])
    pred = qr(["month", "order_count"], [["2025-01", 3], ["2025-02", 4]])
    assert compare(gold, pred, None)[0] is False
    assert compare(gold, pred, ["orders"])[0] is True


def test_detects_wrong_values_and_row_counts() -> None:
    gold = qr(["n"], [[42]])
    assert compare(gold, qr(["n"], [[41]]), None)[0] is False
    assert compare(gold, qr(["n"], [[42], [1]]), None)[0] is False
    assert compare(gold, None, None)[0] is False
