from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from chatree.storage.repos.base import (
    _row_to_message,
    _row_to_node,
    _row_to_tree,
    new_id,
    utc_now,
)
from chatree.types import (
    ChangeSeverity,
    ContextMode,
    MessageModel,
    NodeKind,
    NodeModel,
    NodeStatus,
    ReflogEntryModel,
    StaleDependencyInfo,
    TitleSource,
    ToolPermission,
    TreeModel,
    is_placeholder_title,
)


class TreesNodesMessagesMixin:
    db: Any
    conn: Any
    if TYPE_CHECKING:
        async def _upsert_node_fts(
            self,
            node_id: str,
            tree_id: str,
            title: str,
            goal: str,
            summary: str,
            rollup_summary: str,
        ) -> None: ...
        async def list_reflog(
            self,
            tree_id: str,
            limit: int = 100,
            *,
            session_id: str | None = "default",
        ) -> list[ReflogEntryModel]: ...
        async def compute_stale_dependencies_map(
            self,
            tree_id: str,
            *,
            transitive: bool = True,
            max_hops: int = 3,
        ) -> dict[str, list[StaleDependencyInfo]]: ...

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
        *,
        title_source: TitleSource | None = None,
    ) -> tuple[TreeModel, NodeModel, ReflogEntryModel]:
        now = utc_now()
        tree_id = new_id("t")
        root_id = new_id("n")
        root_path = f"/{root_id}/"
        eff_title_source: TitleSource = (
            title_source
            if title_source is not None
            else ("placeholder" if is_placeholder_title(title) else "user")
        )

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
                    id, tree_id, parent_id, path, depth, title, title_source, brief_title, goal, summary_brief,
                    context_mode, spawn_msg_seq, slice_turn_count, status, tool_permission,
                    model_override, summary, summary_version, rollup_summary, is_dirty,
                    last_msg_seq, tokens_in, tokens_out, cache_read_tokens, created_at, updated_at
                ) VALUES (?, ?, NULL, ?, 0, ?, ?, ?, ?, '', 'root', 0, 6, 'ACTIVE', 'read_write',
                          NULL, NULL, 0, NULL, 0, 0, 0, 0, 0, ?, ?)
                """,
                (root_id, tree_id, root_path, title, eff_title_source, title, goal, now, now),
            )
            await self._upsert_node_fts(root_id, tree_id, title, goal, "", "")

            await self.conn.execute(
                """
                INSERT INTO tree_sessions (
                    tree_id, id, title, kind, head_node_id, reflog_cursor,
                    is_default, created_at, updated_at
                ) VALUES (?, 'default', 'Main Session', 'human', ?, 0, 1, ?, ?)
                """,
                (tree_id, root_id, now, now),
            )

            rf_id = new_id("rf")
            await self.conn.execute(
                """
                INSERT INTO reflog (
                    id, tree_id, session_id, seq, from_node_id, to_node_id, action,
                    note, orientation_cue, cue_consumed, created_at
                ) VALUES (?, ?, 'default', 1, NULL, ?, 'init', ?, '', 1, ?)
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
            deleted = bool((cur.rowcount or 0) > 0)
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
        node_kind: NodeKind = "chat",
        title_source: TitleSource | None = None,
    ) -> NodeModel:
        now = utc_now()
        node_id = new_id("n")
        path = f"{parent.path}{node_id}/"
        depth = parent.depth + 1
        eff_title_source: TitleSource = (
            title_source
            if title_source is not None
            else ("placeholder" if is_placeholder_title(title) else "user")
        )

        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO nodes (
                    id, tree_id, parent_id, path, depth, title, title_source, brief_title, goal, summary_brief,
                    context_mode, spawn_msg_seq, slice_turn_count, status, node_kind, tool_permission,
                    model_override, summary, summary_version, rollup_summary, is_dirty,
                    last_msg_seq, tokens_in, tokens_out, cache_read_tokens, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?,
                          ?, NULL, 0, NULL, 0, 0, 0, 0, 0, ?, ?)
                """,
                (
                    node_id,
                    parent.tree_id,
                    parent.id,
                    path,
                    depth,
                    title,
                    eff_title_source,
                    title,
                    goal,
                    summary_brief,
                    context_mode,
                    spawn_msg_seq,
                    slice_turn_count,
                    node_kind,
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

    async def count_active_children(
        self, parent_id: str, *, node_kind: NodeKind | None = None
    ) -> int:
        if node_kind is None:
            cur = await self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM nodes WHERE parent_id = ? AND status = 'ACTIVE'",
                (parent_id,),
            )
        else:
            cur = await self.conn.execute(
                "SELECT COUNT(*) AS cnt FROM nodes WHERE parent_id = ? AND status = 'ACTIVE' AND node_kind = ?",
                (parent_id, node_kind),
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
            "title_source",
            "brief_title",
            "goal",
            "summary_brief",
            "status",
            "node_kind",
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

    async def list_messages_from_seq(self, node_id: str, min_seq: int) -> list[MessageModel]:
        cur = await self.conn.execute(
            "SELECT * FROM messages WHERE node_id = ? AND seq >= ? ORDER BY seq ASC",
            (node_id, min_seq),
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
            await self.conn.execute(
                "DELETE FROM embeddings_fts WHERE node_id = ? AND granularity = 'chunk'",
                (node_id,),
            )
            await self.conn.execute(
                "DELETE FROM embeddings WHERE node_id = ? AND granularity = 'chunk'",
                (node_id,),
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
        if hasattr(self, "invalidate_vector_cache"):
            self.invalidate_vector_cache(tree_id)

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
