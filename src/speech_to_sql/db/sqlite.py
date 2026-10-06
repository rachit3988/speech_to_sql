"""SQLite adapter (aiosqlite) for local development, demos and tests.

The file is opened read-only (``mode=ro``) with ``PRAGMA query_only``; long queries
are interrupted when the statement timeout elapses.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import aiosqlite

from speech_to_sql.db.base import QueryResult, rank_matches, search_words, to_jsonable
from speech_to_sql.db.catalog import ColumnInfo, ForeignKey, SchemaCatalog, TableInfo
from speech_to_sql.errors import DatabaseUnavailableError, QueryExecutionError, QueryTimeoutError


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class SQLiteDatabase:
    dialect = "sqlite"

    def __init__(
        self,
        path: str,
        *,
        hidden_tables: list[str],
        hidden_columns: list[str],
        pool_size: int = 4,
    ) -> None:
        self._path = Path(path)
        self._hidden_tables = hidden_tables
        self._hidden_columns = hidden_columns
        self._pool_size = max(1, pool_size)
        self._pool: asyncio.Queue[aiosqlite.Connection] | None = None

    async def connect(self) -> None:
        if not self._path.exists():
            raise DatabaseUnavailableError(
                f"SQLite database {self._path} not found (run `make seed` to create the demo DB)"
            )
        pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue()
        for _ in range(self._pool_size):
            conn = await aiosqlite.connect(f"file:{self._path}?mode=ro", uri=True)
            await conn.execute("PRAGMA query_only = ON")
            pool.put_nowait(conn)
        self._pool = pool

    async def close(self) -> None:
        if self._pool is None:
            return
        while not self._pool.empty():
            await self._pool.get_nowait().close()
        self._pool = None

    async def _acquire(self) -> aiosqlite.Connection:
        if self._pool is None:
            raise DatabaseUnavailableError("database is not connected")
        return await self._pool.get()

    def _release(self, conn: aiosqlite.Connection) -> None:
        if self._pool is not None:
            self._pool.put_nowait(conn)

    async def ping(self) -> None:
        conn = await self._acquire()
        try:
            await conn.execute("SELECT 1")
        except sqlite3.Error as exc:
            raise DatabaseUnavailableError(f"database ping failed: {exc}") from exc
        finally:
            self._release(conn)

    async def introspect(self) -> SchemaCatalog:
        conn = await self._acquire()
        try:
            cur = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            names = [r[0] for r in await cur.fetchall()]
            tables = []
            for name in names:
                cols = await (await conn.execute(f"PRAGMA table_info({quote_ident(name)})")).fetchall()
                fk_rows = await (await conn.execute(f"PRAGMA foreign_key_list({quote_ident(name)})")).fetchall()
                count = await (await conn.execute(f"SELECT count(*) FROM {quote_ident(name)}")).fetchone()
                fks: dict[int, list[tuple[str, str, str]]] = {}
                for fk in fk_rows:  # id, seq, table, from, to, ...
                    fks.setdefault(fk[0], []).append((fk[3], fk[2], fk[4]))
                tables.append(
                    TableInfo(
                        schema=None,
                        name=name,
                        row_estimate=count[0] if count else None,
                        columns=tuple(
                            ColumnInfo(
                                name=c[1],
                                data_type=(c[2] or "ANY").lower(),
                                nullable=not c[3],
                                is_primary_key=bool(c[5]),
                            )
                            for c in cols
                        ),
                        foreign_keys=tuple(
                            ForeignKey(
                                columns=tuple(p[0] for p in parts),
                                ref_table=parts[0][1],
                                ref_columns=tuple(p[2] for p in parts),
                            )
                            for parts in fks.values()
                        ),
                    )
                )
        finally:
            self._release(conn)
        return SchemaCatalog.build(
            tables,
            default_schema=None,
            hidden_tables=self._hidden_tables,
            hidden_columns=self._hidden_columns,
        )

    async def _run(
        self, sql: str, params: tuple[object, ...], *, max_rows: int, timeout_s: float
    ) -> tuple[list[str], list[tuple[object, ...]], bool]:
        conn = await self._acquire()
        try:
            async with asyncio.timeout(timeout_s):
                cur = await conn.execute(sql, params)
                rows = [tuple(r) for r in await cur.fetchmany(max_rows + 1)]
                columns = [d[0] for d in cur.description or ()]
                await cur.close()
            return columns, rows[:max_rows], len(rows) > max_rows
        except TimeoutError as exc:
            await conn.interrupt()
            raise QueryTimeoutError(f"query exceeded the {timeout_s:g}s time limit") from exc
        except sqlite3.Error as exc:
            raise QueryExecutionError(f"{exc.__class__.__name__}: {exc}") from exc
        finally:
            self._release(conn)

    async def execute(self, sql: str, *, max_rows: int, timeout_s: float) -> QueryResult:
        started = time.perf_counter()
        columns, rows, truncated = await self._run(sql, (), max_rows=max_rows, timeout_s=timeout_s)
        return QueryResult(
            columns=columns,
            rows=[[to_jsonable(v) for v in row] for row in rows],
            truncated=truncated,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    async def search_values(
        self, table: TableInfo, column: str, term: str, *, limit: int, timeout_s: float
    ) -> list[str]:
        col = table.column(column)
        if col is None:
            raise QueryExecutionError(f"unknown column {column!r} on {table.qualified_name}")
        expr = f"CAST({quote_ident(col.name)} AS TEXT)"
        values: list[str] = []
        for word in dict.fromkeys([term, *search_words(term)]):
            escaped = word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            _, rows, _ = await self._run(
                f"SELECT DISTINCT {expr} FROM {quote_ident(table.name)} WHERE {expr} LIKE ? ESCAPE '\\' LIMIT ?",
                (f"%{escaped}%", limit * 4),
                max_rows=limit * 4,
                timeout_s=timeout_s,
            )
            values.extend(str(r[0]) for r in rows if r[0] is not None)
            if len(values) >= limit:
                break
        return rank_matches(values, term, limit)
