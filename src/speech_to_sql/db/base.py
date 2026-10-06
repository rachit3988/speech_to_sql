"""Database abstraction shared by the Postgres and SQLite adapters."""

from __future__ import annotations

import base64
import datetime as dt
import decimal
import difflib
import ipaddress
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from speech_to_sql.db.catalog import SchemaCatalog, TableInfo


@dataclass(frozen=True, slots=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]  # JSON-safe values
    truncated: bool
    elapsed_ms: float

    @property
    def row_count(self) -> int:
        return len(self.rows)


class Database(Protocol):
    """A read-only query backend.

    Implementations must enforce read-only execution and a statement timeout at the
    database level; the SQL guard is a second, independent layer of defense.
    """

    #: sqlglot dialect name used to parse and validate SQL for this backend.
    dialect: str

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def ping(self) -> None: ...

    async def introspect(self) -> SchemaCatalog: ...

    async def execute(self, sql: str, *, max_rows: int, timeout_s: float) -> QueryResult:
        """Run a guard-approved SELECT, returning at most ``max_rows`` rows."""
        ...

    async def search_values(
        self, table: TableInfo, column: str, term: str, *, limit: int, timeout_s: float
    ) -> list[str]:
        """Distinct values of ``table.column`` that resemble ``term`` (best first)."""
        ...


def to_jsonable(value: Any) -> Any:
    """Convert driver values into JSON-serializable equivalents."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else str(value)
    if isinstance(value, decimal.Decimal):
        # Keep integers exact; fractional decimals become floats for charting/JSON.
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if isinstance(value, (uuid.UUID, ipaddress.IPv4Address, ipaddress.IPv6Address)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "base64:" + base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    return str(value)


def rank_matches(values: list[str], term: str, limit: int) -> list[str]:
    """Order candidate values by closeness to ``term`` (exact > prefix > substring > fuzzy)."""
    t = term.lower()

    def score(v: str) -> tuple[int, float, str]:
        lv = v.lower()
        if lv == t:
            tier = 0
        elif lv.startswith(t):
            tier = 1
        elif t in lv:
            tier = 2
        else:
            tier = 3
        return tier, -difflib.SequenceMatcher(None, lv, t).ratio(), lv

    return sorted(dict.fromkeys(values), key=score)[:limit]


def search_words(term: str) -> list[str]:
    """Split a spoken search term into words worth matching individually."""
    return [w for w in term.replace("%", " ").replace("_", " ").split() if len(w) >= 3][:5]
