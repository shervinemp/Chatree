from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from chatree.storage.repos.base import (
    _np,
    _row_to_artifact,
    _row_to_ipc,
    _row_to_mailbox,
    _row_to_message,
    _row_to_node,
    _row_to_tree,
    cosine_similarity,
    eligible_node_ids_safe,
    new_id,
    normalize_bm25,
    pack_vector,
    sanitize_fts_query,
    unpack_vector,
    utc_now,
)
from chatree.types import (
    ArtifactInput,
    ArtifactModel,
    ArtifactRef,
    BackgroundJobModel,
    BranchFactModel,
    ChangeSeverity,
    ChildContextMode,
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
    ToolPermission,
    TreeInvariantModel,
    TreeModel,
)


class SearchVectorsMixin:
    db: Any
    conn: Any
    _vector_cache: dict[str, dict[int, tuple[list[EmbeddingRecord], Any]]]
    _vector_cache_hits: int
    _vector_cache_misses: int

    def invalidate_vector_cache(self, tree_id: str | None = None) -> None:
        """Invalidate cached pre-normalized NumPy vector matrices for `tree_id` (or all trees)."""
        if tree_id is None:
            self._vector_cache.clear()
        else:
            self._vector_cache.pop(tree_id, None)

    def vector_cache_stats(self) -> dict[str, int]:
        return {
            "cached_trees": len(self._vector_cache),
            "hits": self._vector_cache_hits,
            "misses": self._vector_cache_misses,
        }

    async def _get_tree_vector_cache(
        self, tree_id: str
    ) -> dict[int, tuple[list[EmbeddingRecord], Any]]:
        cached = self._vector_cache.get(tree_id)
        if cached is not None:
            self._vector_cache_hits += 1
            return cached
        self._vector_cache_misses += 1
        cur = await self.conn.execute(
            """
            SELECT id, tree_id, node_id, path, granularity, chunk_index,
                   seq_start, seq_end, content, vector_blob, vector_dim,
                   summary_version, updated_at
            FROM embeddings
            WHERE tree_id = ?
            """,
            (tree_id,),
        )
        by_dim_recs: dict[int, list[EmbeddingRecord]] = {}
        by_dim_blobs: dict[int, list[bytes]] = {}
        for r in await cur.fetchall():
            blob = bytes(r["vector_blob"])
            dim = int(r["vector_dim"])
            if dim <= 0 or len(blob) != dim * 4:
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
            by_dim_recs.setdefault(dim, []).append(rec)
            by_dim_blobs.setdefault(dim, []).append(blob)

        built: dict[int, tuple[list[EmbeddingRecord], Any]] = {}
        for dim, recs in by_dim_recs.items():
            blobs = by_dim_blobs[dim]
            if _np is not None:
                mat = _np.frombuffer(b"".join(blobs), dtype=_np.float32).reshape(len(recs), dim)
                norms = _np.linalg.norm(mat, axis=1, keepdims=True)
                unit_mat = mat / _np.maximum(norms, 1e-12)
                built[dim] = (recs, unit_mat)
            else:
                built[dim] = (recs, blobs)
        self._vector_cache[tree_id] = built
        return built

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
        self.invalidate_vector_cache(tree_id)
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

        # Cached per-tree pre-normalized NumPy matrix search (zero SQLite BLOB I/O on repeat queries!)
        tree_mats = await self._get_tree_vector_cache(tree_id)
        dim_entry = tree_mats.get(q_dim)
        if not dim_entry:
            return []
        all_recs, mat_or_blobs = dim_entry
        keep_indices = [
            i
            for i, rec in enumerate(all_recs)
            if rec.node_id in eligible_node_ids
            and (not granularities or rec.granularity in granularities)
        ]
        if not keep_indices:
            return []

        if _np is not None:
            q_arr = _np.asarray(query_vector, dtype=_np.float32)
            q_norm = float(_np.linalg.norm(q_arr))
            if q_norm <= 1e-12:
                return [(all_recs[i], 0.0) for i in keep_indices[:limit]]
            q_unit = q_arr / q_norm
            sub_mat = mat_or_blobs if len(keep_indices) == len(all_recs) else mat_or_blobs[keep_indices]
            sims = sub_mat @ q_unit
            top_n = min(limit, len(keep_indices))
            if top_n < len(keep_indices):
                top_local = _np.argpartition(-sims, top_n - 1)[:top_n]
                top_local = top_local[_np.argsort(-sims[top_local])]
            else:
                top_local = _np.argsort(-sims)
            return [(all_recs[keep_indices[int(j)]], float(sims[int(j)])) for j in top_local]

        scored: list[tuple[EmbeddingRecord, float]] = []
        for idx in keep_indices:
            sim = cosine_similarity(query_vector, unpack_vector(mat_or_blobs[idx]))
            scored.append((all_recs[idx], sim))
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

