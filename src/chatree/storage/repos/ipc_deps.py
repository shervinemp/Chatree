from __future__ import annotations

import hashlib
import json
from typing import Any

from chatree.storage.repos.base import (
    _row_to_ipc,
    new_id,
    utc_now,
)
from chatree.types import (
    ChangeSeverity,
    IpcEventModel,
    IpcEventType,
    StaleDependencyInfo,
)


class IpcDepsShadowMixin:
    db: Any
    conn: Any

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
