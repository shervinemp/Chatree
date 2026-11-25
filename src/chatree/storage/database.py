from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class Database:
    """Async SQLite connection manager with WAL mode, write serialization, FTS5, and sqlite-vec."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = Path(db_path) if str(db_path) != ":memory:" else ":memory:"
        self._conn: aiosqlite.Connection | None = None
        self.has_sqlite_vec: bool = False
        self.write_lock: asyncio.Lock = asyncio.Lock()

    async def connect(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON;")
        await conn.execute("PRAGMA busy_timeout = 5000;")
        if isinstance(self.db_path, Path):
            await conn.execute("PRAGMA journal_mode = WAL;")
            await conn.execute("PRAGMA synchronous = NORMAL;")

        # Load sqlite-vec C extension automatically when installed (`chatree[vec]`)
        await self._try_load_sqlite_vec(conn)

        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        await conn.executescript(schema_sql)
        await self._run_additive_migrations(conn)
        await conn.commit()
        self._conn = conn
        return conn

    @asynccontextmanager
    async def write_transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Serialize concurrent async writers on the shared connection with atomic commit/rollback."""
        async with self.write_lock:
            conn = self.conn
            try:
                yield conn
                await conn.commit()
            except Exception:
                try:
                    await conn.rollback()
                except Exception:
                    pass
                raise

    async def _try_load_sqlite_vec(self, conn: aiosqlite.Connection) -> None:
        try:
            import sqlite_vec  # type: ignore[import-untyped]
        except ImportError:
            self.has_sqlite_vec = False
            return

        try:
            await conn.enable_load_extension(True)
            ext_path = sqlite_vec.loadable_path()
            await conn.load_extension(ext_path)
            await conn.enable_load_extension(False)
            self.has_sqlite_vec = True
        except Exception:
            self.has_sqlite_vec = False

    async def _run_additive_migrations(self, conn: aiosqlite.Connection) -> None:
        cur = await conn.execute("PRAGMA table_info(nodes)")
        cols = {row["name"] for row in await cur.fetchall()}
        if "tool_permission" not in cols:
            await conn.execute(
                "ALTER TABLE nodes ADD COLUMN tool_permission TEXT NOT NULL DEFAULT 'read_only'"
            )
        if "node_kind" not in cols:
            await conn.execute(
                "ALTER TABLE nodes ADD COLUMN node_kind TEXT NOT NULL DEFAULT 'chat'"
            )
        if "rollup_summary" not in cols:
            await conn.execute("ALTER TABLE nodes ADD COLUMN rollup_summary TEXT")
        if "summary_minor_version" not in cols:
            await conn.execute(
                "ALTER TABLE nodes ADD COLUMN summary_minor_version INTEGER NOT NULL DEFAULT 0"
            )
        if "last_change_severity" not in cols:
            await conn.execute(
                "ALTER TABLE nodes ADD COLUMN last_change_severity TEXT NOT NULL DEFAULT 'material'"
            )
        if "last_change_delta" not in cols:
            await conn.execute(
                "ALTER TABLE nodes ADD COLUMN last_change_delta TEXT NOT NULL DEFAULT ''"
            )

        dep_cur = await conn.execute("PRAGMA table_info(node_dependencies)")
        dep_cols = {row["name"] for row in await dep_cur.fetchall()}
        if "artifact_slug" not in dep_cols:
            await conn.execute("ALTER TABLE node_dependencies ADD COLUMN artifact_slug TEXT")

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS branch_facts (
                id TEXT PRIMARY KEY,
                tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
                node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                fact_text TEXT NOT NULL,
                entities_json TEXT NOT NULL DEFAULT '[]',
                resolved_date TEXT NOT NULL DEFAULT '',
                source_msg_seq INTEGER,
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'superseded')),
                created_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_branch_facts_tree_node ON branch_facts(tree_id, node_id, status)"
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tree_sessions (
                tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
                id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'Main Session',
                kind TEXT NOT NULL DEFAULT 'human' CHECK(kind IN ('human', 'agent')),
                head_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
                reflog_cursor INTEGER NOT NULL DEFAULT 0,
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (tree_id, id)
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tree_sessions_tree ON tree_sessions(tree_id, updated_at DESC)"
        )
        rf_cur = await conn.execute("PRAGMA table_info(reflog)")
        rf_cols = {row["name"] for row in await rf_cur.fetchall()}
        if "session_id" not in rf_cols:
            await conn.execute(
                "ALTER TABLE reflog ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default'"
            )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reflog_tree_session_seq ON reflog(tree_id, session_id, seq DESC)"
        )
        # Backfill primary 'default' visitor session for any pre-existing trees
        await conn.execute(
            """
            INSERT OR IGNORE INTO tree_sessions (
                tree_id, id, title, kind, head_node_id, reflog_cursor, is_default, created_at, updated_at
            )
            SELECT id, 'default', 'Main Session', 'human', head_node_id, reflog_cursor, 1, created_at, updated_at
            FROM trees
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tree_invariants (
                tree_id TEXT NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
                source_summary_version INTEGER NOT NULL DEFAULT 0,
                source_msg_seq INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (tree_id, key)
            )
            """
        )
        inv_cur = await conn.execute("PRAGMA table_info(tree_invariants)")
        inv_cols = {row["name"] for row in await inv_cur.fetchall()}
        if "source_summary_version" not in inv_cols:
            await conn.execute(
                "ALTER TABLE tree_invariants ADD COLUMN source_summary_version INTEGER NOT NULL DEFAULT 0"
            )
        if "source_msg_seq" not in inv_cols:
            await conn.execute(
                "ALTER TABLE tree_invariants ADD COLUMN source_msg_seq INTEGER NOT NULL DEFAULT 0"
            )

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected. Call await db.connect() first.")
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
