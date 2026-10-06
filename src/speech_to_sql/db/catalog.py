"""Schema catalog: what the agent is allowed to see and query.

The catalog is introspected from the live database, filtered by the configured
hidden tables/columns, and rendered deterministically (sorted) so the system
prompt built from it is byte-stable and prompt-cacheable.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    comment: str | None = None


@dataclass(frozen=True, slots=True)
class ForeignKey:
    columns: tuple[str, ...]
    ref_table: str  # qualified name
    ref_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableInfo:
    schema: str | None
    name: str
    columns: tuple[ColumnInfo, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    comment: str | None = None
    row_estimate: int | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name

    def column(self, name: str) -> ColumnInfo | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


@dataclass(slots=True)
class SchemaCatalog:
    tables: dict[str, TableInfo]  # keyed by lower-cased qualified name
    default_schema: str | None = None
    # Lower-cased qualified table name -> lower-cased hidden column names.
    hidden_columns: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        tables: Iterable[TableInfo],
        *,
        default_schema: str | None,
        hidden_tables: Iterable[str] = (),
        hidden_columns: Iterable[str] = (),
    ) -> SchemaCatalog:
        hidden_t = {_qualify(t, default_schema) for t in hidden_tables}
        hidden_c: dict[str, set[str]] = {}
        for spec in hidden_columns:
            table_name, _, column = spec.lower().rpartition(".")
            if table_name and column:
                hidden_c.setdefault(_qualify(table_name, default_schema), set()).add(column)

        kept: dict[str, TableInfo] = {}
        for table in tables:
            key = table.qualified_name.lower()
            if key in hidden_t:
                continue
            hidden_here = hidden_c.get(key, set())
            columns = tuple(c for c in table.columns if c.name.lower() not in hidden_here)
            fks = tuple(
                fk
                for fk in table.foreign_keys
                if fk.ref_table.lower() not in hidden_t and not hidden_here.intersection(c.lower() for c in fk.columns)
            )
            kept[key] = TableInfo(
                schema=table.schema,
                name=table.name,
                columns=columns,
                foreign_keys=fks,
                comment=table.comment,
                row_estimate=table.row_estimate,
            )
        return cls(
            tables=dict(sorted(kept.items())),
            default_schema=default_schema,
            hidden_columns={k: frozenset(v) for k, v in hidden_c.items() if k in kept},
        )

    def resolve(self, name: str, schema: str | None = None) -> TableInfo | None:
        """Find a table by (optionally schema-qualified) name, case-insensitively."""
        name = name.strip().strip('"').lower()
        if schema:
            return self.tables.get(f"{schema.strip(chr(34)).lower()}.{name}")
        if "." in name:
            return self.tables.get(name)
        if self.default_schema:
            found = self.tables.get(f"{self.default_schema.lower()}.{name}")
            if found:
                return found
        matches = [t for t in self.tables.values() if t.name.lower() == name]
        return matches[0] if len(matches) == 1 else None

    def hidden_columns_for(self, table: TableInfo) -> frozenset[str]:
        return self.hidden_columns.get(table.qualified_name.lower(), frozenset())

    # --- Rendering -------------------------------------------------------------

    def render_table(self, table: TableInfo) -> str:
        lines = [f"TABLE {table.qualified_name}"]
        meta = []
        if table.row_estimate is not None:
            meta.append(f"~{table.row_estimate} rows")
        if table.comment:
            meta.append(table.comment)
        if meta:
            lines[0] += f"  -- {'; '.join(meta)}"
        for col in table.columns:
            parts = [f"  {col.name} {col.data_type}"]
            if col.is_primary_key:
                parts.append("PK")
            if not col.nullable and not col.is_primary_key:
                parts.append("NOT NULL")
            line = " ".join(parts)
            if col.comment:
                line += f"  -- {col.comment}"
            lines.append(line)
        for fk in table.foreign_keys:
            lines.append(f"  FK ({', '.join(fk.columns)}) -> {fk.ref_table}({', '.join(fk.ref_columns)})")
        return "\n".join(lines)

    def render(self, *, max_chars: int) -> tuple[str, bool]:
        """Render the catalog for the system prompt.

        Returns ``(text, complete)``. When the full rendering exceeds ``max_chars``
        only table names (and comments) are listed and ``complete`` is False, so the
        agent must call ``describe_tables`` before writing SQL.
        """
        full = "\n\n".join(self.render_table(t) for t in self.tables.values())
        if len(full) <= max_chars:
            return full, True
        brief = "\n".join(
            f"{t.qualified_name}" + (f"  -- {t.comment}" if t.comment else "") for t in self.tables.values()
        )
        return brief[:max_chars], False

    def table_names(self) -> list[str]:
        return [t.qualified_name for t in self.tables.values()]


def _qualify(name: str, default_schema: str | None) -> str:
    name = name.lower()
    if "." in name or not default_schema:
        return name
    return f"{default_schema.lower()}.{name}"


class CatalogProvider:
    """Caches the catalog in-process and refreshes it after ``ttl_s``.

    Concurrent callers share a single in-flight refresh. If a refresh fails while a
    previous catalog exists, the stale catalog keeps serving (and the error is
    re-raised only when there is nothing to serve).
    """

    def __init__(self, loader: Callable[[], Awaitable[SchemaCatalog]], ttl_s: float) -> None:
        self._loader = loader
        self._ttl_s = ttl_s
        self._catalog: SchemaCatalog | None = None
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._catalog is not None

    async def get(self) -> SchemaCatalog:
        if self._catalog is not None and time.monotonic() - self._loaded_at < self._ttl_s:
            return self._catalog
        async with self._lock:
            if self._catalog is not None and time.monotonic() - self._loaded_at < self._ttl_s:
                return self._catalog
            try:
                self._catalog = await self._loader()
                self._loaded_at = time.monotonic()
            except Exception:
                if self._catalog is None:
                    raise
                log.exception("catalog_refresh_failed_serving_stale")
                # Serve stale; retry after a short back-off rather than on every call.
                self._loaded_at = time.monotonic() - self._ttl_s + min(30.0, self._ttl_s)
            return self._catalog

    def invalidate(self) -> None:
        self._loaded_at = 0.0
