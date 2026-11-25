from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
import struct
import uuid
from typing import Any

import aiosqlite
try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:
    _np = None

from chatree.config import ToolPermission
from chatree.storage.database import Database
from chatree.types import (
    ArtifactInput,
    ArtifactModel,
    ArtifactRef,
    BackgroundJobModel,
    BranchFactModel,
    ChangeSeverity,
    ContextMode,
    EmbeddingRecord,
    IpcEventModel,
    IpcEventType,
    MailboxItemModel,
    MailboxStatus,
    MessageModel,
    NodeModel,
    NodeStatus,
    RagGranularity,
    ReflogAction,
    ReflogEntryModel,
    SearchHit,
    SearchScope,
    StaleDependencyInfo,
    TreeInvariantModel,
    TreeModel,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> tuple[float, ...]:
    n = len(blob) // 4
    return struct.unpack(f"<{n}f", blob)


def cosine_similarity(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 1e-12:
        return 0.0
    return dot / denom


def normalize_bm25(raw: float) -> float:
    """Strictly monotone mapping of SQLite FTS5 bm25() (more negative = stronger match) into (0, 1)."""
    clamped = max(-40.0, min(40.0, raw + 2.0))
    return 1.0 / (1.0 + math.exp(0.35 * clamped))


def sanitize_fts_query(raw: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", raw.strip())
    if not tokens:
        return ""
    seen: set[str] = set()
    parts: list[str] = []
    for tok in tokens[:14]:
        low = tok.lower()
        if low not in seen:
            seen.add(low)
            parts.append(f'"{tok}"*')
    return " OR ".join(parts)


def _row_to_tree(row: aiosqlite.Row) -> TreeModel:
    return TreeModel(
        id=row["id"],
        title=row["title"],
        root_node_id=row["root_node_id"],
        head_node_id=row["head_node_id"],
        default_model=row["default_model"],
        shadow_model=row["shadow_model"],
        summary_model=row["summary_model"],
        auto_commit_handoffs=bool(row["auto_commit_handoffs"]),
        reflog_cursor=row["reflog_cursor"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_node(
    row: aiosqlite.Row,
    stale_dependencies: list[StaleDependencyInfo] | None = None,
    child_count: int = 0,
    pending_handoffs_count: int = 0,
) -> NodeModel:
    return NodeModel(
        id=row["id"],
        tree_id=row["tree_id"],
        parent_id=row["parent_id"],
        path=row["path"],
        depth=row["depth"],
        title=row["title"],
        goal=row["goal"],
        summary_brief=row["summary_brief"],
        context_mode=row["context_mode"],
        spawn_msg_seq=row["spawn_msg_seq"],
        slice_turn_count=row["slice_turn_count"],
        status=row["status"],
        tool_permission=row["tool_permission"],
        model_override=row["model_override"],
        summary=row["summary"],
        summary_version=row["summary_version"],
        summary_minor_version=row["summary_minor_version"],
        last_change_severity=row["last_change_severity"],
        last_change_delta=row["last_change_delta"],
        rollup_summary=row["rollup_summary"],
        is_dirty=bool(row["is_dirty"]),
        last_msg_seq=row["last_msg_seq"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        cache_read_tokens=row["cache_read_tokens"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        stale_dependencies=stale_dependencies or [],
        child_count=child_count,
        pending_handoffs_count=pending_handoffs_count,
    )


def _row_to_message(row: aiosqlite.Row) -> MessageModel:
    return MessageModel(
        id=row["id"],
        node_id=row["node_id"],
        tree_id=row["tree_id"],
        seq=row["seq"],
        role=row["role"],
        content=row["content"],
        tool_calls=json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else None,
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
        model_used=row["model_used"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        cache_read_tokens=row["cache_read_tokens"],
        latency_ms=row["latency_ms"],
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        created_at=row["created_at"],
    )


def _row_to_ipc(row: aiosqlite.Row) -> IpcEventModel:
    return IpcEventModel(
        id=row["id"],
        tree_id=row["tree_id"],
        trace_id=row["trace_id"],
        hop_depth=row["hop_depth"],
        from_node_id=row["from_node_id"],
        to_node_id=row["to_node_id"],
        event_type=row["event_type"],
        payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
        response=json.loads(row["response_json"]) if row["response_json"] else {},
        status=row["status"],
        target_summary_version=row["target_summary_version"],
        target_msg_seq=row["target_msg_seq"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        latency_ms=row["latency_ms"],
        created_at=row["created_at"],
    )


def _row_to_artifact(row: aiosqlite.Row) -> ArtifactModel:
    return ArtifactModel(
        id=row["id"],
        tree_id=row["tree_id"],
        slug=row["slug"],
        version=row["version"],
        kind=row["kind"],
        title=row["title"],
        content=row["content"],
        summary=row["summary"],
        created_by_node_id=row["created_by_node_id"],
        created_at=row["created_at"],
    )


def _row_to_mailbox(row: aiosqlite.Row) -> MailboxItemModel:
    raw_refs = json.loads(row["artifact_refs_json"]) if row["artifact_refs_json"] else []
    refs = [ArtifactRef(**r) for r in raw_refs]
    return MailboxItemModel(
        id=row["id"],
        tree_id=row["tree_id"],
        to_node_id=row["to_node_id"],
        from_node_id=row["from_node_id"],
        ipc_event_id=row["ipc_event_id"],
        kind=row["kind"],
        status=row["status"],
        summary=row["summary"],
        artifact_refs=refs,
        payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
        created_at=row["created_at"],
        delivered_at=row["delivered_at"],
    )


class SQLiteRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def conn(self) -> aiosqlite.Connection:
        return self.db.conn

    # ------------------------------------------------------------------
    # Trees
    # ------------------------------------------------------------------

    async def create_tree(
        self,
        title: str,
        goal: str,
        default_model: str,
        shadow_model: str,
        summary_model: str,
        auto_commit_handoffs: bool,
    ) -> tuple[TreeModel, NodeModel, ReflogEntryModel]:
        now = utc_now()
        tree_id = new_id("t")
        root_id = new_id("n")
        root_path = f"/{root_id}/"

        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO trees (
                    id, title, root_node_id, head_node_id,
                    default_model, shadow_model, summary_model,
                    auto_commit_handoffs, reflog_cursor, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    tree_id,
                    title,
                    root_id,
                    root_id,
                    default_model,
                    shadow_model,
                    summary_model,
                    1 if auto_commit_handoffs else 0,
                    now,
                    now,
                ),
            )

            await self.conn.execute(
                """
                INSERT INTO nodes (
                    id, tree_id, parent_id, path, depth, title, goal, summary_brief,
                    context_mode, spawn_msg_seq, slice_turn_count, status, tool_permission,
                    model_override, summary, summary_version, rollup_summary, is_dirty,
                    last_msg_seq, tokens_in, tokens_out, cache_read_tokens, created_at, updated_at
                ) VALUES (?, ?, NULL, ?, 0, ?, ?, '', 'root', 0, 6, 'ACTIVE', 'read_write',
                          NULL, NULL, 0, NULL, 0, 0, 0, 0, 0, ?, ?)
                """,
                (root_id, tree_id, root_path, title, goal, now, now),
            )
            await self._upsert_node_fts(root_id, tree_id, title, goal, "", "")

            rf_id = new_id("rf")
            await self.conn.execute(
                """
                INSERT INTO reflog (
                    id, tree_id, seq, from_node_id, to_node_id, action,
                    note, orientation_cue, cue_consumed, created_at
                ) VALUES (?, ?, 1, NULL, ?, 'init', ?, '', 1, ?)
                """,
                (rf_id, tree_id, root_id, f"Created tree '{title}'", now),
            )

        tree = await self.get_tree(tree_id)
        root = await self.get_node(root_id)
        reflog_entries = await self.list_reflog(tree_id, limit=1)
        assert tree is not None and root is not None
        return tree, root, reflog_entries[0]

    async def get_tree(self, tree_id: str) -> TreeModel | None:
        cur = await self.conn.execute("SELECT * FROM trees WHERE id = ?", (tree_id,))
        row = await cur.fetchone()
        return _row_to_tree(row) if row else None

    async def list_trees(self) -> list[TreeModel]:
        cur = await self.conn.execute("SELECT * FROM trees ORDER BY updated_at DESC")
        return [_row_to_tree(r) for r in await cur.fetchall()]

    async def update_tree(self, tree_id: str, **fields: Any) -> TreeModel | None:
        allowed = {
            "title",
            "default_model",
            "shadow_model",
            "summary_model",
            "auto_commit_handoffs",
            "head_node_id",
            "root_node_id",
            "reflog_cursor",
        }
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return await self.get_tree(tree_id)
        if "auto_commit_handoffs" in updates:
            updates["auto_commit_handoffs"] = 1 if updates["auto_commit_handoffs"] else 0
        updates["updated_at"] = utc_now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        await self.conn.execute(
            f"UPDATE trees SET {set_clause} WHERE id = ?",
            list(updates.values()) + [tree_id],
        )
        await self.conn.commit()
        return await self.get_tree(tree_id)

    async def delete_tree(self, tree_id: str) -> bool:
        async with self.db.write_transaction():
            await self.conn.execute("DELETE FROM messages_fts WHERE tree_id = ?", (tree_id,))
            await self.conn.execute("DELETE FROM nodes_fts WHERE tree_id = ?", (tree_id,))
            await self.conn.execute("DELETE FROM embeddings_fts WHERE tree_id = ?", (tree_id,))
            cur = await self.conn.execute("DELETE FROM trees WHERE id = ?", (tree_id,))
            deleted = cur.rowcount > 0
        return deleted

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def create_child_node(
        self,
        parent: NodeModel,
        title: str,
        goal: str,
        context_mode: ContextMode,
        summary_brief: str,
        spawn_msg_seq: int,
        slice_turn_count: int = 6,
        model_override: str | None = None,
        tool_permission: ToolPermission = "read_only",
    ) -> NodeModel:
        now = utc_now()
        node_id = new_id("n")
        path = f"{parent.path}{node_id}/"
        depth = parent.depth + 1

        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO nodes (
                    id, tree_id, parent_id, path, depth, title, goal, summary_brief,
                    context_mode, spawn_msg_seq, slice_turn_count, status, tool_permission,
                    model_override, summary, summary_version, rollup_summary, is_dirty,
                    last_msg_seq, tokens_in, tokens_out, cache_read_tokens, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?,
                          ?, NULL, 0, NULL, 0, 0, 0, 0, 0, ?, ?)
                """,
                (
                    node_id,
                    parent.tree_id,
                    parent.id,
                    path,
                    depth,
                    title,
                    goal,
                    summary_brief,
                    context_mode,
                    spawn_msg_seq,
                    slice_turn_count,
                    tool_permission,
                    model_override,
                    now,
                    now,
                ),
            )
            await self._upsert_node_fts(node_id, parent.tree_id, title, goal, summary_brief, "")
            await self.conn.execute(
                "UPDATE trees SET updated_at = ? WHERE id = ?", (now, parent.tree_id)
            )
        node = await self.get_node(node_id)
        assert node is not None
        return node

    async def get_node(self, node_id: str) -> NodeModel | None:
        cur = await self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = await cur.fetchone()
        if not row:
            return None
        stale_map = await self.compute_stale_dependencies_map(row["tree_id"])
        child_cur = await self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM nodes WHERE parent_id = ?", (node_id,)
        )
        child_cnt = (await child_cur.fetchone())["cnt"]
        mb_cur = await self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM mailbox WHERE to_node_id = ? AND status = 'pending_review'",
            (node_id,),
        )
        mb_cnt = (await mb_cur.fetchone())["cnt"]
        return _row_to_node(
            row,
            stale_dependencies=stale_map.get(node_id, []),
            child_count=child_cnt,
            pending_handoffs_count=mb_cnt,
        )

    async def list_nodes(self, tree_id: str) -> list[NodeModel]:
        cur = await self.conn.execute(
            "SELECT * FROM nodes WHERE tree_id = ? ORDER BY depth ASC, created_at ASC",
            (tree_id,),
        )
        rows = await cur.fetchall()
        stale_map = await self.compute_stale_dependencies_map(tree_id)
        child_counts: dict[str, int] = {}
        for r in rows:
            pid = r["parent_id"]
            if pid:
                child_counts[pid] = child_counts.get(pid, 0) + 1
        mb_cur = await self.conn.execute(
            """
            SELECT to_node_id, COUNT(*) AS cnt
            FROM mailbox
            WHERE tree_id = ? AND status = 'pending_review'
            GROUP BY to_node_id
            """,
            (tree_id,),
        )
        mb_counts = {r["to_node_id"]: r["cnt"] for r in await mb_cur.fetchall()}
        return [
            _row_to_node(
                r,
                stale_dependencies=stale_map.get(r["id"], []),
                child_count=child_counts.get(r["id"], 0),
                pending_handoffs_count=mb_counts.get(r["id"], 0),
            )
            for r in rows
        ]

    async def get_children(self, node_id: str) -> list[NodeModel]:
        cur = await self.conn.execute(
            "SELECT * FROM nodes WHERE parent_id = ? ORDER BY created_at ASC", (node_id,)
        )
        return [_row_to_node(r) for r in await cur.fetchall()]

    async def count_active_children(self, parent_id: str) -> int:
        cur = await self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM nodes WHERE parent_id = ? AND status = 'ACTIVE'",
            (parent_id,),
        )
        return int((await cur.fetchone())["cnt"])

    async def get_ancestors(self, node: NodeModel) -> list[NodeModel]:
        ids = [seg for seg in node.path.strip("/").split("/") if seg and seg != node.id]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cur = await self.conn.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})", ids
        )
        by_id = {r["id"]: _row_to_node(r) for r in await cur.fetchall()}
        return [by_id[i] for i in ids if i in by_id]

    async def update_node(self, node_id: str, **fields: Any) -> NodeModel | None:
        allowed = {
            "title",
            "goal",
            "summary_brief",
            "status",
            "tool_permission",
            "model_override",
            "summary",
            "summary_version",
            "rollup_summary",
            "is_dirty",
            "parent_id",
            "path",
            "depth",
            "context_mode",
            "spawn_msg_seq",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return await self.get_node(node_id)
        if "is_dirty" in updates:
            updates["is_dirty"] = 1 if updates["is_dirty"] else 0
        updates["updated_at"] = utc_now()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        async with self.db.write_transaction():
            await self.conn.execute(
                f"UPDATE nodes SET {set_clause} WHERE id = ?",
                list(updates.values()) + [node_id],
            )
            cur = await self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            row = await cur.fetchone()
            if row is not None:
                await self._upsert_node_fts(
                    row["id"],
                    row["tree_id"],
                    row["title"],
                    row["goal"],
                    row["summary"] or row["summary_brief"],
                    row["rollup_summary"] or "",
                )
        return await self.get_node(node_id)

    async def commit_node_summary(
        self,
        node_id: str,
        summary: str,
        status: NodeStatus | None = None,
        *,
        severity: ChangeSeverity = "material",
        change_delta: str = "",
    ) -> NodeModel:
        now = utc_now()
        async with self.db.write_transaction():
            cur = await self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            row = await cur.fetchone()
            if row is None:
                raise ValueError(f"Node not found: {node_id}")
            cur_major = int(row["summary_version"])
            cur_minor = int(row["summary_minor_version"])
            # First summary is always major v1.0; subsequent updates respect materiality severity
            if cur_major == 0 or severity == "material":
                next_major = cur_major + 1
                next_minor = 0
                eff_severity: ChangeSeverity = "material"
            else:
                next_major = cur_major
                next_minor = cur_minor + 1
                eff_severity = "cosmetic"
            new_status = status or row["status"]
            await self.conn.execute(
                """
                UPDATE nodes
                SET summary = ?,
                    summary_version = ?,
                    summary_minor_version = ?,
                    last_change_severity = ?,
                    last_change_delta = ?,
                    is_dirty = 0,
                    status = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    summary,
                    next_major,
                    next_minor,
                    eff_severity,
                    change_delta,
                    new_status,
                    now,
                    node_id,
                ),
            )
            await self._upsert_node_fts(
                node_id,
                row["tree_id"],
                row["title"],
                row["goal"],
                summary,
                row["rollup_summary"] or "",
            )
        updated = await self.get_node(node_id)
        assert updated is not None
        return updated

    async def increment_node_usage(
        self,
        node_id: str,
        tokens_in: int,
        tokens_out: int,
        cache_read_tokens: int = 0,
    ) -> None:
        now = utc_now()
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                UPDATE nodes
                SET tokens_in = tokens_in + ?,
                    tokens_out = tokens_out + ?,
                    cache_read_tokens = cache_read_tokens + ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (tokens_in, tokens_out, cache_read_tokens, now, node_id),
            )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def append_message(
        self,
        node_id: str,
        tree_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        model_used: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cache_read_tokens: int = 0,
        latency_ms: int = 0,
        metadata: dict[str, Any] | None = None,
        mark_dirty: bool = True,
    ) -> MessageModel:
        msg_id = new_id("m")
        now = utc_now()
        dirty_expr = "1" if mark_dirty else "is_dirty"

        async with self.db.write_transaction():
            cur = await self.conn.execute("SELECT last_msg_seq FROM nodes WHERE id = ?", (node_id,))
            row = await cur.fetchone()
            if not row:
                raise ValueError(f"Node not found: {node_id}")
            next_seq = row["last_msg_seq"] + 1

            await self.conn.execute(
                """
                INSERT INTO messages (
                    id, node_id, tree_id, seq, role, content,
                    tool_calls_json, tool_call_id, tool_name, model_used,
                    tokens_in, tokens_out, cache_read_tokens, latency_ms,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    node_id,
                    tree_id,
                    next_seq,
                    role,
                    content,
                    json.dumps(tool_calls) if tool_calls is not None else None,
                    tool_call_id,
                    tool_name,
                    model_used,
                    tokens_in,
                    tokens_out,
                    cache_read_tokens,
                    latency_ms,
                    json.dumps(metadata) if metadata else None,
                    now,
                ),
            )
            await self.conn.execute(
                f"""
                UPDATE nodes
                SET last_msg_seq = ?,
                    is_dirty = {dirty_expr},
                    tokens_in = tokens_in + ?,
                    tokens_out = tokens_out + ?,
                    cache_read_tokens = cache_read_tokens + ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (next_seq, tokens_in, tokens_out, cache_read_tokens, now, node_id),
            )
            await self.conn.execute(
                "UPDATE trees SET updated_at = ? WHERE id = ?", (now, tree_id)
            )
            if content.strip():
                await self.conn.execute(
                    """
                    INSERT INTO messages_fts (message_id, tree_id, node_id, seq, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (msg_id, tree_id, node_id, str(next_seq), content),
                )

        cur_msg = await self.conn.execute("SELECT * FROM messages WHERE id = ?", (msg_id,))
        msg_row = await cur_msg.fetchone()
        assert msg_row is not None
        return _row_to_message(msg_row)

    async def list_messages(self, node_id: str, up_to_seq: int | None = None) -> list[MessageModel]:
        if up_to_seq is None:
            cur = await self.conn.execute(
                "SELECT * FROM messages WHERE node_id = ? ORDER BY seq ASC", (node_id,)
            )
        else:
            cur = await self.conn.execute(
                "SELECT * FROM messages WHERE node_id = ? AND seq <= ? ORDER BY seq ASC",
                (node_id, up_to_seq),
            )
        return [_row_to_message(r) for r in await cur.fetchall()]

    async def replace_node_messages_for_extraction(
        self,
        node_id: str,
        tree_id: str,
        remaining_messages: list[MessageModel],
    ) -> None:
        """
        Re-compact a node's messages during explicit retroactive tangent extraction
        (`extract_to_child`), preserving original monotonic `seq` IDs so pre-existing
        forked children (`spawn_msg_seq`) and `node_dependencies.seen_msg_seq` remain valid,
        and invalidating `shadow_cache` for this node.
        """
        async with self.db.write_transaction():
            cur_last = await self.conn.execute(
                "SELECT last_msg_seq FROM nodes WHERE id = ?", (node_id,)
            )
            last_row = await cur_last.fetchone()
            prev_last_seq = int(last_row["last_msg_seq"]) if last_row else 0

            await self.conn.execute("DELETE FROM messages_fts WHERE node_id = ?", (node_id,))
            await self.conn.execute("DELETE FROM messages WHERE node_id = ?", (node_id,))
            await self.conn.execute(
                "DELETE FROM shadow_cache WHERE target_node_id = ?", (node_id,)
            )

            max_seq = prev_last_seq
            for m in remaining_messages:
                preserved_seq = m.seq
                if preserved_seq > max_seq:
                    max_seq = preserved_seq
                await self.conn.execute(
                    """
                    INSERT INTO messages (
                        id, node_id, tree_id, seq, role, content,
                        tool_calls_json, tool_call_id, tool_name, model_used,
                        tokens_in, tokens_out, cache_read_tokens, latency_ms,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m.id,
                        node_id,
                        tree_id,
                        preserved_seq,
                        m.role,
                        m.content,
                        json.dumps(m.tool_calls) if m.tool_calls is not None else None,
                        m.tool_call_id,
                        m.tool_name,
                        m.model_used,
                        m.tokens_in,
                        m.tokens_out,
                        m.cache_read_tokens,
                        m.latency_ms,
                        json.dumps(m.metadata) if m.metadata else None,
                        m.created_at,
                    ),
                )
                if m.content.strip():
                    await self.conn.execute(
                        """
                        INSERT INTO messages_fts (message_id, tree_id, node_id, seq, content)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (m.id, tree_id, node_id, str(preserved_seq), m.content),
                    )
            await self.conn.execute(
                "UPDATE nodes SET last_msg_seq = ?, is_dirty = 1, updated_at = ? WHERE id = ?",
                (max_seq, utc_now(), node_id),
            )

    async def get_last_safe_fork_seq(self, node_id: str) -> int:
        msgs = await self.list_messages(node_id)
        open_tool_ids: set[str] = set()
        last_safe = 0
        for m in msgs:
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.get("id"):
                        open_tool_ids.add(str(tc["id"]))
            elif m.role == "tool" and m.tool_call_id:
                open_tool_ids.discard(str(m.tool_call_id))
            if not open_tool_ids:
                last_safe = m.seq
        return last_safe

    # ------------------------------------------------------------------
    # IPC Events
    # ------------------------------------------------------------------

    async def create_ipc_event(
        self,
        tree_id: str,
        trace_id: str,
        hop_depth: int,
        from_node_id: str,
        to_node_id: str | None,
        event_type: IpcEventType,
        payload: dict[str, Any],
        response: dict[str, Any],
        status: str = "ok",
        target_summary_version: int | None = None,
        target_msg_seq: int | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: int = 0,
    ) -> IpcEventModel:
        event_id = new_id("ipc")
        now = utc_now()
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO ipc_events (
                    id, tree_id, trace_id, hop_depth, from_node_id, to_node_id,
                    event_type, payload_json, response_json, status,
                    target_summary_version, target_msg_seq,
                    tokens_in, tokens_out, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    tree_id,
                    trace_id,
                    hop_depth,
                    from_node_id,
                    to_node_id,
                    event_type,
                    json.dumps(payload),
                    json.dumps(response),
                    status,
                    target_summary_version,
                    target_msg_seq,
                    tokens_in,
                    tokens_out,
                    latency_ms,
                    now,
                ),
            )
        cur = await self.conn.execute("SELECT * FROM ipc_events WHERE id = ?", (event_id,))
        row = await cur.fetchone()
        assert row is not None
        return _row_to_ipc(row)

    async def list_ipc_events(self, tree_id: str, limit: int = 200) -> list[IpcEventModel]:
        cur = await self.conn.execute(
            "SELECT * FROM ipc_events WHERE tree_id = ? ORDER BY created_at DESC LIMIT ?",
            (tree_id, limit),
        )
        return [_row_to_ipc(r) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # Dependencies & Staleness
    # ------------------------------------------------------------------

    async def upsert_dependency(
        self,
        tree_id: str,
        consumer_node_id: str,
        source_node_id: str,
        seen_summary_version: int,
        seen_msg_seq: int,
        last_ipc_event_id: str | None = None,
        artifact_slug: str | None = None,
    ) -> None:
        if consumer_node_id == source_node_id:
            return
        now = utc_now()
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO node_dependencies (
                    consumer_node_id, source_node_id, tree_id,
                    seen_summary_version, seen_msg_seq, artifact_slug, last_ipc_event_id,
                    dismissed_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(consumer_node_id, source_node_id) DO UPDATE SET
                    seen_summary_version = excluded.seen_summary_version,
                    seen_msg_seq = excluded.seen_msg_seq,
                    artifact_slug = COALESCE(excluded.artifact_slug, node_dependencies.artifact_slug),
                    last_ipc_event_id = excluded.last_ipc_event_id,
                    dismissed_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    consumer_node_id,
                    source_node_id,
                    tree_id,
                    seen_summary_version,
                    seen_msg_seq,
                    artifact_slug,
                    last_ipc_event_id,
                    now,
                ),
            )

    async def list_dependencies(self, tree_id: str) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT * FROM node_dependencies WHERE tree_id = ?", (tree_id,)
        )
        return [dict(r) for r in await cur.fetchall()]

    async def dismiss_dependency(self, consumer_node_id: str, source_node_id: str) -> None:
        now = utc_now()
        async with self.db.write_transaction():
            cur = await self.conn.execute(
                "SELECT summary_version, last_msg_seq FROM nodes WHERE id = ?",
                (source_node_id,),
            )
            src = await cur.fetchone()
            if not src:
                return
            await self.conn.execute(
                """
                UPDATE node_dependencies
                SET seen_summary_version = ?, seen_msg_seq = ?, dismissed_at = ?, updated_at = ?
                WHERE consumer_node_id = ? AND source_node_id = ?
                """,
                (
                    src["summary_version"],
                    src["last_msg_seq"],
                    now,
                    now,
                    consumer_node_id,
                    source_node_id,
                ),
            )

    async def compute_stale_dependencies_map(
        self,
        tree_id: str,
        *,
        transitive: bool = True,
        max_hops: int = 3,
    ) -> dict[str, list[StaleDependencyInfo]]:
        cur = await self.conn.execute(
            """
            SELECT
                d.consumer_node_id, d.source_node_id,
                d.seen_summary_version, d.seen_msg_seq, d.artifact_slug,
                s.title AS source_title,
                s.summary_version AS current_summary_version,
                s.summary_minor_version AS current_minor_version,
                s.last_change_severity AS source_severity,
                s.last_change_delta AS source_delta,
                s.last_msg_seq AS current_msg_seq,
                s.is_dirty AS source_is_dirty,
                c.title AS consumer_title
            FROM node_dependencies d
            JOIN nodes s ON s.id = d.source_node_id
            JOIN nodes c ON c.id = d.consumer_node_id
            WHERE d.tree_id = ?
            """,
            (tree_id,),
        )
        rows = await cur.fetchall()
        stale_map: dict[str, list[StaleDependencyInfo]] = {}
        consumers_by_source: dict[str, list[str]] = {}
        titles: dict[str, str] = {}
        fresh_direct_edges: set[tuple[str, str]] = set()
        direct_stale_pairs: set[tuple[str, str]] = set()

        for r in rows:
            consumer_id = r["consumer_node_id"]
            source_id = r["source_node_id"]
            titles[source_id] = r["source_title"]
            titles[consumer_id] = r["consumer_title"]
            consumers_by_source.setdefault(source_id, []).append(consumer_id)

            ver_advanced = r["current_summary_version"] > r["seen_summary_version"]
            seq_advanced = r["current_msg_seq"] > r["seen_msg_seq"]
            severity: ChangeSeverity = r["source_severity"] or "material"
            delta_txt = r["source_delta"] or ""

            if ver_advanced or seq_advanced:
                # If a branch was cleanly re-summarized with ONLY a cosmetic minor edit
                # (major version unchanged, not dirty, severity == 'cosmetic'), keep severity='cosmetic'
                # so downstream transitive consumers are NOT spammed!
                effective_severity: ChangeSeverity = (
                    "cosmetic"
                    if (not ver_advanced and not bool(r["source_is_dirty"]) and severity == "cosmetic")
                    else "material"
                )
                if ver_advanced:
                    reason = (
                        f"Summary updated (v{r['seen_summary_version']} -> "
                        f"v{r['current_summary_version']})"
                    )
                    if delta_txt:
                        reason = f"{reason} — {delta_txt}"
                elif effective_severity == "cosmetic":
                    reason = (
                        f"Cosmetic edit (v{r['current_summary_version']}.{r['current_minor_version']})"
                        + (f" — {delta_txt}" if delta_txt else "")
                    )
                else:
                    delta = r["current_msg_seq"] - r["seen_msg_seq"]
                    reason = f"{delta} new message(s) added since last read (summary dirty)"
                direct_stale_pairs.add((consumer_id, source_id))
                stale_map.setdefault(consumer_id, []).append(
                    StaleDependencyInfo(
                        source_node_id=source_id,
                        source_title=r["source_title"],
                        seen_summary_version=r["seen_summary_version"],
                        current_summary_version=r["current_summary_version"],
                        seen_msg_seq=r["seen_msg_seq"],
                        current_msg_seq=r["current_msg_seq"],
                        source_is_dirty=bool(r["source_is_dirty"]),
                        reason=reason,
                        hops=1,
                        via_node_ids=[],
                        severity=effective_severity,
                        change_delta=delta_txt,
                        artifact_slug=r["artifact_slug"],
                    )
                )
            else:
                fresh_direct_edges.add((consumer_id, source_id))

        if not transitive or max_hops <= 1:
            return stale_map

        # Cycle-safe BFS propagating ONLY material upstream changes (prevents alert fatigue!)
        from collections import deque

        queue: deque[tuple[str, StaleDependencyInfo]] = deque()
        seen_consumer_source: set[tuple[str, str]] = set(direct_stale_pairs)
        for consumer_id, infos in stale_map.items():
            for info in infos:
                if info.severity == "material":
                    queue.append((consumer_id, info))

        while queue:
            mid_node_id, info = queue.popleft()
            if info.hops >= max_hops:
                continue
            for downstream_id in consumers_by_source.get(mid_node_id, []):
                if downstream_id == info.source_node_id:
                    continue
                if downstream_id in info.via_node_ids:
                    continue
                if (downstream_id, info.source_node_id) in fresh_direct_edges:
                    continue
                pair = (downstream_id, info.source_node_id)
                if pair in seen_consumer_source:
                    continue
                seen_consumer_source.add(pair)
                via_chain = [*info.via_node_ids, mid_node_id]
                via_labels = " -> ".join(titles.get(vid, vid) for vid in via_chain)
                trans_info = StaleDependencyInfo(
                    source_node_id=info.source_node_id,
                    source_title=info.source_title,
                    seen_summary_version=info.seen_summary_version,
                    current_summary_version=info.current_summary_version,
                    seen_msg_seq=info.seen_msg_seq,
                    current_msg_seq=info.current_msg_seq,
                    source_is_dirty=info.source_is_dirty,
                    reason=f"Transitive upstream change via {via_labels}: {info.reason}",
                    hops=info.hops + 1,
                    via_node_ids=via_chain,
                    severity="material",
                    change_delta=info.change_delta,
                    artifact_slug=info.artifact_slug,
                )
                stale_map.setdefault(downstream_id, []).append(trans_info)
                queue.append((downstream_id, trans_info))

        return stale_map

    # ------------------------------------------------------------------
    # Shadow-Turn Memoization Cache
    # ------------------------------------------------------------------

    @staticmethod
    def hash_shadow_prompt(prompt: str) -> str:
        normalized = " ".join(prompt.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def get_cached_shadow_answer(
        self,
        target_node_id: str,
        target_msg_seq: int,
        model: str,
        prompt: str,
    ) -> dict[str, Any] | None:
        p_hash = self.hash_shadow_prompt(prompt)
        cur = await self.conn.execute(
            """
            SELECT id, answer, summary_version, hit_count
            FROM shadow_cache
            WHERE target_node_id = ? AND target_msg_seq = ? AND model = ? AND prompt_hash = ?
            """,
            (target_node_id, target_msg_seq, model, p_hash),
        )
        row = await cur.fetchone()
        if not row:
            return None
        async with self.db.write_transaction():
            await self.conn.execute(
                "UPDATE shadow_cache SET hit_count = hit_count + 1 WHERE id = ?",
                (row["id"],),
            )
        return {
            "cache_id": row["id"],
            "answer": row["answer"],
            "summary_version": row["summary_version"],
            "hit_count": int(row["hit_count"]) + 1,
        }

    async def list_cached_shadow_answers_for_node(
        self,
        target_node_id: str,
        target_msg_seq: int,
        model: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            """
            SELECT id, prompt_text, answer, summary_version, hit_count
            FROM shadow_cache
            WHERE target_node_id = ? AND target_msg_seq = ? AND model = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (target_node_id, target_msg_seq, model, limit),
        )
        return [
            {
                "id": r["id"],
                "prompt_text": r["prompt_text"],
                "answer": r["answer"],
                "summary_version": r["summary_version"],
                "hit_count": r["hit_count"],
            }
            for r in await cur.fetchall()
        ]

    async def increment_shadow_cache_hit(self, cache_id: str) -> int:
        async with self.db.write_transaction():
            await self.conn.execute(
                "UPDATE shadow_cache SET hit_count = hit_count + 1 WHERE id = ?",
                (cache_id,),
            )
            cur = await self.conn.execute(
                "SELECT hit_count FROM shadow_cache WHERE id = ?", (cache_id,)
            )
            row = await cur.fetchone()
        return int(row["hit_count"]) if row else 1

    async def invalidate_shadow_cache(self, target_node_id: str) -> None:
        async with self.db.write_transaction():
            await self.conn.execute(
                "DELETE FROM shadow_cache WHERE target_node_id = ?", (target_node_id,)
            )

    async def put_cached_shadow_answer(
        self,
        tree_id: str,
        target_node_id: str,
        target_msg_seq: int,
        model: str,
        prompt: str,
        answer: str,
        summary_version: int,
        *,
        max_entries: int = 500,
    ) -> None:
        p_hash = self.hash_shadow_prompt(prompt)
        cache_id = new_id("sc")
        now = utc_now()
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO shadow_cache (
                    id, tree_id, target_node_id, target_msg_seq, model,
                    prompt_hash, prompt_text, answer, summary_version, hit_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(target_node_id, target_msg_seq, model, prompt_hash) DO UPDATE SET
                    answer = excluded.answer,
                    summary_version = excluded.summary_version
                """,
                (
                    cache_id,
                    tree_id,
                    target_node_id,
                    target_msg_seq,
                    model,
                    p_hash,
                    prompt,
                    answer,
                    summary_version,
                    now,
                ),
            )
            # Evict stale older msg_seq cache rows for this node & cap per-tree shadow_cache size
            await self.conn.execute(
                "DELETE FROM shadow_cache WHERE target_node_id = ? AND target_msg_seq < ?",
                (target_node_id, target_msg_seq),
            )
            await self.conn.execute(
                """
                DELETE FROM shadow_cache
                WHERE tree_id = ? AND id NOT IN (
                    SELECT id FROM shadow_cache
                    WHERE tree_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (tree_id, tree_id, max_entries),
            )

    async def list_consumers_of_source(
        self,
        source_node_id: str,
        *,
        transitive: bool = True,
        max_hops: int = 3,
    ) -> list[str]:
        cur = await self.conn.execute(
            "SELECT tree_id FROM nodes WHERE id = ?", (source_node_id,)
        )
        row = await cur.fetchone()
        if not row:
            return []
        cur_edges = await self.conn.execute(
            "SELECT consumer_node_id, source_node_id FROM node_dependencies WHERE tree_id = ?",
            (row["tree_id"],),
        )
        consumers_by_source: dict[str, list[str]] = {}
        for r in await cur_edges.fetchall():
            consumers_by_source.setdefault(r["source_node_id"], []).append(r["consumer_node_id"])

        if not transitive or max_hops <= 1:
            return list(dict.fromkeys(consumers_by_source.get(source_node_id, [])))

        from collections import deque

        visited: set[str] = {source_node_id}
        ordered: list[str] = []
        q: deque[tuple[str, int]] = deque([(source_node_id, 0)])
        while q:
            curr_id, depth = q.popleft()
            if depth >= max_hops:
                continue
            for nxt in consumers_by_source.get(curr_id, []):
                if nxt not in visited:
                    visited.add(nxt)
                    ordered.append(nxt)
                    q.append((nxt, depth + 1))
        return ordered

    # ------------------------------------------------------------------
    # Reflog
    # ------------------------------------------------------------------

    async def append_reflog(
        self,
        tree_id: str,
        from_node_id: str | None,
        to_node_id: str,
        action: ReflogAction,
        note: str = "",
        orientation_cue: str = "",
        update_head: bool = True,
    ) -> ReflogEntryModel:
        rf_id = new_id("rf")
        now = utc_now()
        cue_consumed = 0 if orientation_cue else 1

        async with self.db.write_transaction():
            cur = await self.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM reflog WHERE tree_id = ?",
                (tree_id,),
            )
            next_seq = (await cur.fetchone())["max_seq"] + 1
            await self.conn.execute(
                """
                INSERT INTO reflog (
                    id, tree_id, seq, from_node_id, to_node_id, action,
                    note, orientation_cue, cue_consumed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rf_id,
                    tree_id,
                    next_seq,
                    from_node_id,
                    to_node_id,
                    action,
                    note,
                    orientation_cue,
                    cue_consumed,
                    now,
                ),
            )
            if update_head:
                await self.conn.execute(
                    "UPDATE trees SET head_node_id = ?, reflog_cursor = 0, updated_at = ? WHERE id = ?",
                    (to_node_id, now, tree_id),
                )
        entries = await self.list_reflog(tree_id, limit=1)
        return entries[0]

    async def list_reflog(self, tree_id: str, limit: int = 100) -> list[ReflogEntryModel]:
        cur = await self.conn.execute(
            "SELECT * FROM reflog WHERE tree_id = ? ORDER BY seq DESC LIMIT ?",
            (tree_id, limit),
        )
        rows = await cur.fetchall()
        return [
            ReflogEntryModel(
                id=r["id"],
                tree_id=r["tree_id"],
                seq=r["seq"],
                head_ref=f"HEAD@{{{idx}}}",
                from_node_id=r["from_node_id"],
                to_node_id=r["to_node_id"],
                action=r["action"],
                note=r["note"],
                orientation_cue=r["orientation_cue"],
                cue_consumed=bool(r["cue_consumed"]),
                created_at=r["created_at"],
            )
            for idx, r in enumerate(rows)
        ]

    async def get_unconsumed_orientation_cues(
        self, tree_id: str, node_id: str
    ) -> list[ReflogEntryModel]:
        cur = await self.conn.execute(
            """
            SELECT * FROM reflog
            WHERE tree_id = ? AND to_node_id = ? AND cue_consumed = 0 AND orientation_cue != ''
            ORDER BY seq DESC
            LIMIT 3
            """,
            (tree_id, node_id),
        )
        return [
            ReflogEntryModel(
                id=r["id"],
                tree_id=r["tree_id"],
                seq=r["seq"],
                head_ref="HEAD@{0}",
                from_node_id=r["from_node_id"],
                to_node_id=r["to_node_id"],
                action=r["action"],
                note=r["note"],
                orientation_cue=r["orientation_cue"],
                cue_consumed=False,
                created_at=r["created_at"],
            )
            for r in await cur.fetchall()
        ]

    async def mark_orientation_cues_consumed(self, reflog_ids: list[str]) -> None:
        if not reflog_ids:
            return
        placeholders = ",".join("?" for _ in reflog_ids)
        async with self.db.write_transaction():
            await self.conn.execute(
                f"UPDATE reflog SET cue_consumed = 1 WHERE id IN ({placeholders})", reflog_ids
            )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def create_artifact(
        self,
        tree_id: str,
        art: ArtifactInput,
        created_by_node_id: str | None = None,
    ) -> ArtifactModel:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", art.slug.strip()).strip("-") or "artifact"
        art_id = new_id("art")
        now = utc_now()
        async with self.db.write_transaction():
            cur = await self.conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_v FROM artifacts WHERE tree_id = ? AND slug = ?",
                (tree_id, slug),
            )
            next_v = (await cur.fetchone())["max_v"] + 1
            await self.conn.execute(
                """
                INSERT INTO artifacts (
                    id, tree_id, slug, version, kind, title, content, summary,
                    created_by_node_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    art_id,
                    tree_id,
                    slug,
                    next_v,
                    art.kind,
                    art.title,
                    art.content,
                    art.summary,
                    created_by_node_id,
                    now,
                ),
            )
        cur_row = await self.conn.execute("SELECT * FROM artifacts WHERE id = ?", (art_id,))
        row = await cur_row.fetchone()
        assert row is not None
        return _row_to_artifact(row)

    async def list_artifacts(self, tree_id: str) -> list[ArtifactModel]:
        cur = await self.conn.execute(
            "SELECT * FROM artifacts WHERE tree_id = ? ORDER BY created_at DESC",
            (tree_id,),
        )
        return [_row_to_artifact(r) for r in await cur.fetchall()]

    async def list_artifacts_by_node(self, node_id: str) -> list[ArtifactModel]:
        cur = await self.conn.execute(
            "SELECT * FROM artifacts WHERE created_by_node_id = ? ORDER BY created_at DESC",
            (node_id,),
        )
        return [_row_to_artifact(r) for r in await cur.fetchall()]

    async def get_artifact(self, artifact_id: str) -> ArtifactModel | None:
        cur = await self.conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        row = await cur.fetchone()
        return _row_to_artifact(row) if row else None

    # ------------------------------------------------------------------
    # Mailbox
    # ------------------------------------------------------------------

    async def create_mailbox_item(
        self,
        tree_id: str,
        to_node_id: str,
        from_node_id: str,
        kind: str,
        status: MailboxStatus,
        summary: str,
        artifact_refs: list[ArtifactRef],
        payload: dict[str, Any] | None = None,
        ipc_event_id: str | None = None,
    ) -> MailboxItemModel:
        mb_id = new_id("mb")
        now = utc_now()
        delivered_at = now if status == "delivered" else None
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO mailbox (
                    id, tree_id, to_node_id, from_node_id, ipc_event_id,
                    kind, status, summary, artifact_refs_json, payload_json,
                    created_at, delivered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mb_id,
                    tree_id,
                    to_node_id,
                    from_node_id,
                    ipc_event_id,
                    kind,
                    status,
                    summary,
                    json.dumps([r.model_dump() for r in artifact_refs]),
                    json.dumps(payload or {}),
                    now,
                    delivered_at,
                ),
            )
        item = await self.get_mailbox_item(mb_id)
        assert item is not None
        return item

    async def get_mailbox_item(self, mailbox_id: str) -> MailboxItemModel | None:
        cur = await self.conn.execute("SELECT * FROM mailbox WHERE id = ?", (mailbox_id,))
        row = await cur.fetchone()
        return _row_to_mailbox(row) if row else None

    async def list_mailbox_items(
        self,
        tree_id: str,
        to_node_id: str | None = None,
        status: MailboxStatus | None = None,
    ) -> list[MailboxItemModel]:
        clauses = ["tree_id = ?"]
        params: list[Any] = [tree_id]
        if to_node_id is not None:
            clauses.append("to_node_id = ?")
            params.append(to_node_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        cur = await self.conn.execute(
            f"SELECT * FROM mailbox WHERE {' AND '.join(clauses)} ORDER BY created_at DESC",
            params,
        )
        return [_row_to_mailbox(r) for r in await cur.fetchall()]

    async def update_mailbox_item(
        self,
        mailbox_id: str,
        status: MailboxStatus | None = None,
        summary: str | None = None,
        artifact_refs: list[ArtifactRef] | None = None,
    ) -> MailboxItemModel | None:
        updates: dict[str, Any] = {}
        if status is not None:
            updates["status"] = status
            if status == "delivered":
                updates["delivered_at"] = utc_now()
        if summary is not None:
            updates["summary"] = summary
        if artifact_refs is not None:
            updates["artifact_refs_json"] = json.dumps([r.model_dump() for r in artifact_refs])
        if not updates:
            return await self.get_mailbox_item(mailbox_id)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        async with self.db.write_transaction():
            await self.conn.execute(
                f"UPDATE mailbox SET {set_clause} WHERE id = ?",
                list(updates.values()) + [mailbox_id],
            )
        return await self.get_mailbox_item(mailbox_id)

    # ------------------------------------------------------------------
    # 3-Tier Embeddings Storage (chunk / summary / rollup) + FTS5
    # ------------------------------------------------------------------

    async def list_node_chunk_embeddings(
        self, node_id: str
    ) -> list[tuple[int, int | None, int | None, str, bytes]]:
        """Return existing chunk embeddings for incremental tail re-indexing."""
        cur = await self.conn.execute(
            """
            SELECT chunk_index, seq_start, seq_end, content, vector_blob
            FROM embeddings
            WHERE node_id = ? AND granularity = 'chunk'
            ORDER BY chunk_index ASC
            """,
            (node_id,),
        )
        return [
            (
                int(r["chunk_index"]),
                r["seq_start"],
                r["seq_end"],
                str(r["content"]),
                bytes(r["vector_blob"]),
            )
            for r in await cur.fetchall()
        ]

    async def replace_node_embeddings(
        self,
        tree_id: str,
        node_id: str,
        path: str,
        granularity: RagGranularity,
        items: list[tuple[int, int | None, int | None, str, list[float], int]],
    ) -> int:
        """
        Atomically replace embeddings for `(node_id, granularity)`.
        Each item is `(chunk_index, seq_start, seq_end, content, vector, summary_version)`.
        """
        now = utc_now()
        async with self.db.write_transaction():
            await self.conn.execute(
                "DELETE FROM embeddings_fts WHERE node_id = ? AND granularity = ?",
                (node_id, granularity),
            )
            await self.conn.execute(
                "DELETE FROM embeddings WHERE node_id = ? AND granularity = ?",
                (node_id, granularity),
            )
            for chunk_idx, seq_start, seq_end, content, vector, summary_ver in items:
                emb_id = new_id("emb")
                blob = pack_vector(vector)
                await self.conn.execute(
                    """
                    INSERT INTO embeddings (
                        id, tree_id, node_id, path, granularity, chunk_index,
                        seq_start, seq_end, content, vector_blob, vector_dim,
                        summary_version, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        emb_id,
                        tree_id,
                        node_id,
                        path,
                        granularity,
                        chunk_idx,
                        seq_start,
                        seq_end,
                        content,
                        blob,
                        len(vector),
                        summary_ver,
                        now,
                    ),
                )
                await self.conn.execute(
                    """
                    INSERT INTO embeddings_fts (embedding_id, tree_id, node_id, granularity, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (emb_id, tree_id, node_id, granularity, content),
                )
        return len(items)

    async def list_embeddings_with_vectors(
        self,
        tree_id: str,
        eligible_node_ids: set[str],
        granularities: set[RagGranularity] | None = None,
    ) -> list[tuple[EmbeddingRecord, bytes]]:
        if not eligible_node_ids:
            return []
        params: list[Any] = [tree_id]
        # Avoid hitting SQLite's bind-parameter limit on large trees (>400 eligible nodes)
        use_in_clause = len(eligible_node_ids) <= 400
        node_clause = ""
        if use_in_clause:
            placeholders = ",".join("?" for _ in eligible_node_ids)
            node_clause = f" AND node_id IN ({placeholders})"
            params.extend(eligible_node_ids)
        gran_clause = ""
        if granularities:
            g_ph = ",".join("?" for _ in granularities)
            gran_clause = f" AND granularity IN ({g_ph})"
            params.extend(granularities)

        cur = await self.conn.execute(
            f"""
            SELECT id, tree_id, node_id, path, granularity, chunk_index,
                   seq_start, seq_end, content, vector_blob, summary_version, updated_at
            FROM embeddings
            WHERE tree_id = ?{node_clause}{gran_clause}
            """,
            params,
        )
        results: list[tuple[EmbeddingRecord, bytes]] = []
        for r in await cur.fetchall():
            if not use_in_clause and r["node_id"] not in eligible_node_ids:
                continue
            rec = EmbeddingRecord(
                id=r["id"],
                tree_id=r["tree_id"],
                node_id=r["node_id"],
                path=r["path"],
                granularity=r["granularity"],
                chunk_index=r["chunk_index"],
                seq_start=r["seq_start"],
                seq_end=r["seq_end"],
                content=r["content"],
                summary_version=r["summary_version"],
                updated_at=r["updated_at"],
            )
            results.append((rec, bytes(r["vector_blob"])))
        return results

    async def query_embeddings_vector_sql(
        self,
        tree_id: str,
        eligible_node_ids: set[str],
        query_vector: list[float],
        limit: int = 30,
        granularities: set[RagGranularity] | None = None,
    ) -> list[tuple[EmbeddingRecord, float]]:
        """
        Uses `sqlite-vec`'s `vec_distance_cosine` in SQLite C code when available;
        transparently falls back to exact float32 cosine similarity in Python otherwise.
        """
        if not eligible_node_ids or not query_vector:
            return []

        q_dim = len(query_vector)
        q_blob = pack_vector(query_vector)

        if self.db.has_sqlite_vec:
            use_in_clause = len(eligible_node_ids) <= 400
            params: list[Any] = [q_blob, tree_id, q_dim]
            node_clause = ""
            if use_in_clause:
                placeholders = ",".join("?" for _ in eligible_node_ids)
                node_clause = f" AND node_id IN ({placeholders})"
                params.extend(eligible_node_ids)
            gran_clause = ""
            if granularities:
                g_ph = ",".join("?" for _ in granularities)
                gran_clause = f" AND granularity IN ({g_ph})"
                params.extend(granularities)
            sql_limit = limit if use_in_clause else max(limit * 4, 200)
            params.append(sql_limit)
            try:
                cur = await self.conn.execute(
                    f"""
                    SELECT id, tree_id, node_id, path, granularity, chunk_index,
                           seq_start, seq_end, content, summary_version, updated_at,
                           (1.0 - vec_distance_cosine(vector_blob, ?)) AS sim
                    FROM embeddings
                    WHERE tree_id = ? AND vector_dim = ?{node_clause}{gran_clause}
                    ORDER BY sim DESC
                    LIMIT ?
                    """,
                    params,
                )
                out: list[tuple[EmbeddingRecord, float]] = []
                for r in await cur.fetchall():
                    if not use_in_clause and r["node_id"] not in eligible_node_ids:
                        continue
                    rec = EmbeddingRecord(
                        id=r["id"],
                        tree_id=r["tree_id"],
                        node_id=r["node_id"],
                        path=r["path"],
                        granularity=r["granularity"],
                        chunk_index=r["chunk_index"],
                        seq_start=r["seq_start"],
                        seq_end=r["seq_end"],
                        content=r["content"],
                        summary_version=r["summary_version"],
                        updated_at=r["updated_at"],
                    )
                    out.append((rec, float(r["sim"] or 0.0)))
                    if len(out) >= limit:
                        break
                return out
            except Exception:
                # Fall through cleanly to pure-Python cosine if extension call fails
                pass

        # Portable float32 cosine fallback (vectorized via NumPy BLAS when available -> 20-50x faster!)
        rows = await self.list_embeddings_with_vectors(tree_id, eligible_node_ids, granularities)
        valid_rows = [(rec, blob) for rec, blob in rows if len(blob) == q_dim * 4]
        if not valid_rows:
            return []
        if _np is not None:
            q_arr = _np.asarray(query_vector, dtype=_np.float32)
            q_norm = float(_np.linalg.norm(q_arr))
            if q_norm <= 1e-12:
                return [(rec, 0.0) for rec, _ in valid_rows[:limit]]
            mat = _np.frombuffer(
                b"".join(blob for _, blob in valid_rows), dtype=_np.float32
            ).reshape(len(valid_rows), q_dim)
            m_norms = _np.linalg.norm(mat, axis=1)
            denom = _np.maximum(m_norms * q_norm, 1e-12)
            sims = (mat @ q_arr) / denom
            top_n = min(limit, len(valid_rows))
            if top_n < len(valid_rows):
                top_idx = _np.argpartition(-sims, top_n - 1)[:top_n]
                top_idx = top_idx[_np.argsort(-sims[top_idx])]
            else:
                top_idx = _np.argsort(-sims)
            return [(valid_rows[int(i)][0], float(sims[int(i)])) for i in top_idx]

        scored: list[tuple[EmbeddingRecord, float]] = []
        for rec, blob in valid_rows:
            sim = cosine_similarity(query_vector, unpack_vector(blob))
            scored.append((rec, sim))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    async def query_embeddings_bm25(
        self,
        tree_id: str,
        eligible_node_ids: set[str],
        query: str,
        limit: int = 30,
        granularities: set[RagGranularity] | None = None,
    ) -> list[tuple[EmbeddingRecord, float]]:
        fts_q = sanitize_fts_query(query)
        if not fts_q or not eligible_node_ids:
            return []
        cur = await self.conn.execute(
            """
            SELECT embedding_id, bm25(embeddings_fts) AS rank
            FROM embeddings_fts
            WHERE tree_id = ? AND embeddings_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (tree_id, fts_q, limit * 3),
        )
        matched = await cur.fetchall()
        if not matched:
            return []
        rank_by_id = {r["embedding_id"]: float(r["rank"]) for r in matched}
        e_ids = list(rank_by_id.keys())
        ph = ",".join("?" for _ in e_ids)
        cur_rec = await self.conn.execute(
            f"""
            SELECT id, tree_id, node_id, path, granularity, chunk_index,
                   seq_start, seq_end, content, summary_version, updated_at
            FROM embeddings
            WHERE id IN ({ph})
            """,
            e_ids,
        )
        out: list[tuple[EmbeddingRecord, float]] = []
        for r in await cur_rec.fetchall():
            if r["node_id"] not in eligible_node_ids:
                continue
            if granularities and r["granularity"] not in granularities:
                continue
            rec = EmbeddingRecord(
                id=r["id"],
                tree_id=r["tree_id"],
                node_id=r["node_id"],
                path=r["path"],
                granularity=r["granularity"],
                chunk_index=r["chunk_index"],
                seq_start=r["seq_start"],
                seq_end=r["seq_end"],
                content=r["content"],
                summary_version=r["summary_version"],
                updated_at=r["updated_at"],
            )
            bm25_raw = rank_by_id.get(rec.id, 0.0)
            bm25_norm = normalize_bm25(bm25_raw)
            out.append((rec, bm25_norm))
        out.sort(key=lambda item: item[1], reverse=True)
        return out[:limit]

    async def count_embeddings(self, tree_id: str) -> int:
        cur = await self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM embeddings WHERE tree_id = ?", (tree_id,)
        )
        return int((await cur.fetchone())["cnt"])

    async def count_tree_messages(self, tree_id: str) -> int:
        cur = await self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM messages WHERE tree_id = ?", (tree_id,)
        )
        return int((await cur.fetchone())["cnt"])

    # ------------------------------------------------------------------
    # Direct FTS5 Search (fallback / lexical component)
    # ------------------------------------------------------------------

    async def _upsert_node_fts(
        self,
        node_id: str,
        tree_id: str,
        title: str,
        goal: str,
        summary: str,
        rollup_summary: str,
    ) -> None:
        await self.conn.execute("DELETE FROM nodes_fts WHERE node_id = ?", (node_id,))
        await self.conn.execute(
            """
            INSERT INTO nodes_fts (node_id, tree_id, title, goal, summary, rollup_summary)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (node_id, tree_id, title, goal, summary, rollup_summary),
        )

    async def search_tree_fts(
        self,
        tree_id: str,
        query: str,
        scope: SearchScope = "all",
        current_node: NodeModel | None = None,
        top_k: int = 5,
    ) -> list[SearchHit]:
        fts_query = sanitize_fts_query(query)
        if not fts_query:
            return []

        all_nodes = await self.list_nodes(tree_id)
        node_by_id = {n.id: n for n in all_nodes}
        eligible_ids = self.filter_node_ids_by_scope(all_nodes, scope, current_node)
        if not eligible_ids:
            return []

        hits: list[SearchHit] = []
        cur_nodes = await self.conn.execute(
            """
            SELECT node_id,
                   bm25(nodes_fts, 3.0, 2.0, 1.5, 1.2) AS rank,
                   snippet(nodes_fts, 4, '[', ']', '...', 24) AS snip
            FROM nodes_fts
            WHERE tree_id = ? AND nodes_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (tree_id, fts_query, max(top_k * 3, 15)),
        )
        for r in await cur_nodes.fetchall():
            nid = r["node_id"]
            if nid not in eligible_ids or nid not in node_by_id:
                continue
            n = node_by_id[nid]
            bm25_s = round(normalize_bm25(float(r["rank"])), 4)
            snippet = r["snip"] or n.summary or n.rollup_summary or n.goal or n.title
            hits.append(
                SearchHit(
                    node_id=n.id,
                    node_title=n.title,
                    node_status=n.status,
                    path=n.path,
                    summary_version=n.summary_version,
                    summary=n.summary,
                    snippet=snippet[:360],
                    score=round(bm25_s + 0.15, 4),
                    bm25_score=bm25_s,
                    granularity="summary" if n.summary else "node",
                    source="bm25",
                )
            )

        cur_msgs = await self.conn.execute(
            """
            SELECT node_id, seq,
                   bm25(messages_fts) AS rank,
                   snippet(messages_fts, 4, '[', ']', '...', 28) AS snip
            FROM messages_fts
            WHERE tree_id = ? AND messages_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (tree_id, fts_query, max(top_k * 4, 20)),
        )
        for r in await cur_msgs.fetchall():
            nid = r["node_id"]
            if nid not in eligible_node_ids_safe(eligible_ids) or nid not in node_by_id:
                continue
            n = node_by_id[nid]
            bm25_s = round(normalize_bm25(float(r["rank"])), 4)
            seq_val = int(r["seq"])
            hits.append(
                SearchHit(
                    node_id=n.id,
                    node_title=n.title,
                    node_status=n.status,
                    path=n.path,
                    summary_version=n.summary_version,
                    summary=n.summary,
                    snippet=(r["snip"] or "")[:360],
                    score=bm25_s,
                    bm25_score=bm25_s,
                    granularity="message",
                    matched_seq=seq_val,
                    seq_range=(seq_val, seq_val),
                    source="bm25",
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        deduped: list[SearchHit] = []
        seen: set[tuple[str, int | None]] = set()
        for h in hits:
            k = (h.node_id, h.matched_seq)
            if k not in seen:
                seen.add(k)
                deduped.append(h)
            if len(deduped) >= top_k:
                break
        return deduped

    @staticmethod
    def filter_node_ids_by_scope(
        all_nodes: list[NodeModel],
        scope: SearchScope,
        current_node: NodeModel | None,
    ) -> set[str]:
        if scope == "all" or current_node is None:
            return {n.id for n in all_nodes}
        if scope == "ancestors":
            return {
                seg for seg in current_node.path.strip("/").split("/") if seg and seg != current_node.id
            }
        if scope == "subtree":
            return {n.id for n in all_nodes if n.path.startswith(current_node.path)}
        if scope == "siblings":
            return {
                n.id
                for n in all_nodes
                if n.parent_id == current_node.parent_id and n.id != current_node.id
            }
        return {n.id for n in all_nodes}

    # ------------------------------------------------------------------
    # Runtime Settings
    # ------------------------------------------------------------------

    async def get_all_settings(self) -> dict[str, Any]:
        cur = await self.conn.execute("SELECT key, value_json FROM app_settings")
        return {r["key"]: json.loads(r["value_json"]) for r in await cur.fetchall()}

    async def set_settings(self, values: dict[str, Any]) -> None:
        now = utc_now()
        async with self.db.write_transaction():
            for k, v in values.items():
                await self.conn.execute(
                    """
                    INSERT INTO app_settings (key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                    """,
                    (k, json.dumps(v), now),
                )

    # ------------------------------------------------------------------
    # Durable Background Jobs
    # ------------------------------------------------------------------

    async def upsert_background_job(self, job: BackgroundJobModel) -> None:
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO background_jobs (
                    job_id, tree_id, parent_node_id, child_node_id, title, goal,
                    status, turns_executed, tokens_used, summary, error,
                    mailbox_id, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    turns_executed = excluded.turns_executed,
                    tokens_used = excluded.tokens_used,
                    summary = excluded.summary,
                    error = excluded.error,
                    mailbox_id = excluded.mailbox_id,
                    completed_at = excluded.completed_at
                """,
                (
                    job.job_id,
                    job.tree_id,
                    job.parent_node_id,
                    job.child_node_id,
                    job.title,
                    job.goal,
                    job.status,
                    job.turns_executed,
                    job.tokens_used,
                    job.summary,
                    job.error,
                    job.mailbox_id,
                    job.created_at,
                    job.completed_at,
                ),
            )

    async def get_background_job(self, job_id: str) -> BackgroundJobModel | None:
        cur = await self.conn.execute(
            "SELECT * FROM background_jobs WHERE job_id = ?", (job_id,)
        )
        r = await cur.fetchone()
        return BackgroundJobModel(**dict(r)) if r else None

    async def list_background_jobs(self, tree_id: str | None = None) -> list[BackgroundJobModel]:
        if tree_id is not None:
            cur = await self.conn.execute(
                "SELECT * FROM background_jobs WHERE tree_id = ? ORDER BY created_at DESC",
                (tree_id,),
            )
        else:
            cur = await self.conn.execute(
                "SELECT * FROM background_jobs ORDER BY created_at DESC"
            )
        return [BackgroundJobModel(**dict(r)) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # Durable Sandbox Workspaces
    # ------------------------------------------------------------------

    async def upsert_sandbox_workspace(
        self, node_id: str, path: str, branch_name: str, base_ref: str
    ) -> None:
        now = utc_now()
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO sandbox_workspaces (node_id, path, branch_name, base_ref, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    path = excluded.path,
                    branch_name = excluded.branch_name,
                    base_ref = excluded.base_ref
                """,
                (node_id, path, branch_name, base_ref, now),
            )

    async def delete_sandbox_workspace(self, node_id: str) -> None:
        async with self.db.write_transaction():
            await self.conn.execute(
                "DELETE FROM sandbox_workspaces WHERE node_id = ?", (node_id,)
            )

    async def list_sandbox_workspaces(self) -> list[dict[str, str]]:
        cur = await self.conn.execute("SELECT * FROM sandbox_workspaces")
        return [dict(r) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # Tier-2.5 Branch-Scoped Atomic Temporal Facts
    # ------------------------------------------------------------------

    async def replace_branch_facts(
        self,
        tree_id: str,
        node_id: str,
        facts: list[dict[str, Any]],
        *,
        source_msg_seq: int | None = None,
    ) -> list[BranchFactModel]:
        """
        Supersede prior active facts for THIS `node_id` ONLY (never touching other branches!)
        and insert the newly extracted atomic temporal facts for `node_id`.
        """
        now = utc_now()
        created: list[BranchFactModel] = []
        async with self.db.write_transaction():
            await self.conn.execute(
                "UPDATE branch_facts SET status = 'superseded' WHERE node_id = ? AND status = 'active'",
                (node_id,),
            )
            for item in facts:
                fact_text = str(item.get("fact") or item.get("fact_text") or "").strip()
                if not fact_text:
                    continue
                raw_ents = item.get("entities") or []
                if isinstance(raw_ents, str):
                    raw_ents = [raw_ents]
                entities = [
                    str(e).strip()
                    for e in raw_ents
                    if str(e).strip()
                ][:12]
                resolved_date = str(item.get("resolved_date") or "").strip()[:40]
                msg_seq = item.get("source_msg_seq", source_msg_seq)
                fid = new_id("fact")
                await self.conn.execute(
                    """
                    INSERT INTO branch_facts (
                        id, tree_id, node_id, fact_text, entities_json,
                        resolved_date, source_msg_seq, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (
                        fid,
                        tree_id,
                        node_id,
                        fact_text[:600],
                        json.dumps(entities),
                        resolved_date,
                        msg_seq,
                        now,
                    ),
                )
                created.append(
                    BranchFactModel(
                        id=fid,
                        tree_id=tree_id,
                        node_id=node_id,
                        fact_text=fact_text[:600],
                        entities=entities,
                        resolved_date=resolved_date,
                        source_msg_seq=msg_seq,
                        status="active",
                        created_at=now,
                    )
                )
        return created

    async def list_branch_facts(
        self,
        tree_id: str,
        *,
        node_id: str | None = None,
        active_only: bool = True,
        exclude_archived: bool = True,
    ) -> list[BranchFactModel]:
        clauses = ["f.tree_id = ?"]
        params: list[Any] = [tree_id]
        if node_id is not None:
            clauses.append("f.node_id = ?")
            params.append(node_id)
        if active_only:
            clauses.append("f.status = 'active'")
        if exclude_archived:
            clauses.append("n.status != 'ARCHIVED'")
        cur = await self.conn.execute(
            f"""
            SELECT f.*
            FROM branch_facts f
            JOIN nodes n ON n.id = f.node_id
            WHERE {' AND '.join(clauses)}
            ORDER BY f.created_at ASC
            """,
            params,
        )
        return [
            BranchFactModel(
                id=r["id"],
                tree_id=r["tree_id"],
                node_id=r["node_id"],
                fact_text=r["fact_text"],
                entities=json.loads(r["entities_json"]) if r["entities_json"] else [],
                resolved_date=r["resolved_date"] or "",
                source_msg_seq=r["source_msg_seq"],
                status=r["status"],
                created_at=r["created_at"],
            )
            for r in await cur.fetchall()
        ]

    # ------------------------------------------------------------------
    # Tree-Level PINNED Invariants (Bounded Core Memory)
    # ------------------------------------------------------------------

    async def upsert_tree_invariant(
        self,
        tree_id: str,
        key: str,
        value: str,
        *,
        source_node_id: str | None = None,
        max_items: int = 8,
        max_value_chars: int = 120,
    ) -> TreeInvariantModel:
        clean_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip()).strip("_")[:48]
        if not clean_key:
            raise ValueError("Invariant key must contain at least one alphanumeric character.")
        clean_val = " ".join(value.strip().split())[:max_value_chars]
        if not clean_val:
            raise ValueError("Invariant value cannot be empty.")
        now = utc_now()
        src_sv = 0
        src_seq = 0
        if source_node_id:
            scur = await self.conn.execute(
                "SELECT summary_version, last_msg_seq FROM nodes WHERE id = ?",
                (source_node_id,),
            )
            srow = await scur.fetchone()
            if srow is not None:
                src_sv = int(srow["summary_version"])
                src_seq = int(srow["last_msg_seq"])
        async with self.db.write_transaction():
            cur = await self.conn.execute(
                "SELECT key FROM tree_invariants WHERE tree_id = ?", (tree_id,)
            )
            existing_keys = {r["key"] for r in await cur.fetchall()}
            if clean_key not in existing_keys and len(existing_keys) >= max_items:
                raise ValueError(
                    f"Pinned invariant limit reached ({max_items}). Unpin or update an existing key first."
                )
            await self.conn.execute(
                """
                INSERT INTO tree_invariants (
                    tree_id, key, value, source_node_id, source_summary_version, source_msg_seq, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tree_id, key) DO UPDATE SET
                    value = excluded.value,
                    source_node_id = excluded.source_node_id,
                    source_summary_version = excluded.source_summary_version,
                    source_msg_seq = excluded.source_msg_seq,
                    updated_at = excluded.updated_at
                """,
                (tree_id, clean_key, clean_val, source_node_id, src_sv, src_seq, now),
            )
        return TreeInvariantModel(
            tree_id=tree_id,
            key=clean_key,
            value=clean_val,
            source_node_id=source_node_id,
            source_summary_version=src_sv,
            source_msg_seq=src_seq,
            updated_at=now,
        )

    async def delete_tree_invariant(self, tree_id: str, key: str) -> bool:
        clean_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip()).strip("_")[:48]
        async with self.db.write_transaction():
            cur = await self.conn.execute(
                "DELETE FROM tree_invariants WHERE tree_id = ? AND key = ?",
                (tree_id, clean_key),
            )
            return (cur.rowcount or 0) > 0

    async def list_tree_invariants(self, tree_id: str) -> list[TreeInvariantModel]:
        cur = await self.conn.execute(
            "SELECT * FROM tree_invariants WHERE tree_id = ? ORDER BY key ASC",
            (tree_id,),
        )
        return [
            TreeInvariantModel(
                tree_id=r["tree_id"],
                key=r["key"],
                value=r["value"],
                source_node_id=r["source_node_id"],
                source_summary_version=int(r["source_summary_version"]) if "source_summary_version" in r.keys() else 0,
                source_msg_seq=int(r["source_msg_seq"]) if "source_msg_seq" in r.keys() else 0,
                updated_at=r["updated_at"],
            )
            for r in await cur.fetchall()
        ]


def eligible_node_ids_safe(ids: set[str]) -> set[str]:
    return ids
