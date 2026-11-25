from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from chatree.config import AblationConfig, RagConfig
from chatree.protocols import Embedder, Reranker
from chatree.providers.mock import estimate_tokens
from chatree.storage.repository import SQLiteRepository, unpack_vector
from chatree.types import (
    EmbeddingRecord,
    MessageModel,
    NodeModel,
    NodeStatus,
    RagGranularity,
    SearchHit,
    SearchScope,
)


def extract_key_identifiers(text: str, limit: int = 16) -> list[str]:
    """
    Extract salient technical identifiers, spec constants, standalone numeric thresholds/units,
    artifact refs, and decision tokens (e.g. `ED25519_JWKS_24H`, `HNSW_EF_64`, `schema.sql@v2`,
    `15m`, `60`, `0.05`, `$500`, `99.9%`) for structured Tier-3 rollups and materiality diffing.
    """
    if not text:
        return []
    candidates = re.findall(
        r"\b[A-Za-z0-9._-]+@v\d+\b"
        r"|\b[A-Z][A-Z0-9_]{2,}\b"
        r"|\b[A-Za-z]+(?:[0-9]+[A-Za-z0-9_]*)\b"
        r"|\$?\b\d+(?:\.\d+)?(?:%|ms|s|m|h|d|kb|mb|gb|tb|rps|qps|k|x)?\b",
        text,
        flags=re.IGNORECASE,
    )
    seen: set[str] = set()
    out: list[str] = []
    for tok in candidates:
        up = tok.upper()
        if up in {"ACTIVE", "COMPLETED", "ARCHIVED", "NONE", "HEAD", "IPC", "DEMO", "MODE"}:
            continue
        # Only keep uppercase constants for the pure-letter uppercase branch (ignore regular words)
        if tok.isalpha() and tok != tok.upper():
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
        if len(out) >= limit:
            break
    return out


def _chunk_messages(
    messages: list[MessageModel],
    target_tokens: int = 320,
    overlap_tokens: int = 60,
) -> list[tuple[int, int, int, str]]:
    """
    Sliding-window turn chunker that preserves turn IDs (`seq_start`, `seq_end`) for exact hits.
    Returns list of `(chunk_index, seq_start, seq_end, chunk_text)`.
    """
    turn_blocks: list[tuple[int, str, int]] = []
    for m in messages:
        if m.role not in ("user", "assistant") or not m.content.strip():
            continue
        # Handoff & extracted-tangent stubs are child pointers already indexed via Tier-2/3;
        # excluding them from Tier-1 chunks ensures chunk search routes to the authoritative branch.
        if m.metadata.get("ipc_handoff") or m.metadata.get("ipc_extracted_stub"):
            continue
        line = f"[{m.role} #{m.seq}] {m.content.strip()}"
        turn_blocks.append((m.seq, line, estimate_tokens(line)))

    if not turn_blocks:
        return []

    chunks: list[tuple[int, int, int, str]] = []
    idx = 0
    chunk_num = 0

    while idx < len(turn_blocks):
        cur_lines: list[str] = []
        cur_tokens = 0
        seq_start = turn_blocks[idx][0]
        seq_end = seq_start
        end_idx = idx

        while end_idx < len(turn_blocks):
            seq, line, tok = turn_blocks[end_idx]
            if cur_lines and cur_tokens + tok > target_tokens:
                break
            cur_lines.append(line)
            cur_tokens += tok
            seq_end = seq
            end_idx += 1

        chunks.append((chunk_num, seq_start, seq_end, "\n".join(cur_lines)))
        chunk_num += 1

        if end_idx >= len(turn_blocks):
            break

        # Rewind by overlap_tokens for context continuity across chunk boundaries
        back_tokens = 0
        next_idx = end_idx
        while next_idx > idx + 1 and back_tokens < overlap_tokens:
            next_idx -= 1
            back_tokens += turn_blocks[next_idx][2]
        idx = max(idx + 1, next_idx)

    return chunks


class TreeRAGService:
    """
    3-Tier Hybrid Tree RAG:
      Tier 1 (`chunk`): Sliding-window blocks (~300-500 tokens) with turn seq ranges.
      Tier 2 (`summary`): Lazy node-level summary + title/goal embedding.
      Tier 3 (`rollup`): Subtree rollup on parents folding in descendant summaries so
                         high-level queries locate the right branch even 3+ levels deep.

    Retrieval fuses SQLite FTS5 BM25 + vector cosine similarity (`sqlite-vec` or exact
    float32 fallback) via weighted Reciprocal Rank Fusion (RRF) with tree scope, status,
    granularity, and recency filtering.
    """

    def __init__(
        self,
        repo: SQLiteRepository,
        embedder: Embedder,
        *,
        rag_config: RagConfig | None = None,
        ablations: AblationConfig | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.cfg = rag_config or RagConfig()
        self.ablations = ablations or AblationConfig()
        self.reranker = reranker

    # ------------------------------------------------------------------
    # Indexing (Tiers 1, 2, 3)
    # ------------------------------------------------------------------

    async def _build_branch_header(self, node: NodeModel) -> str:
        ancestors = await self.repo.get_ancestors(node)
        trail = " > ".join([*(a.title for a in ancestors), node.title])
        goal_part = f" | Goal: {node.goal}" if node.goal else ""
        return f"[Path: {trail}{goal_part}]"

    async def index_node_chunks(self, node_id: str) -> int:
        if not self.cfg.enabled:
            return 0
        node = await self.repo.get_node(node_id)
        if node is None:
            return 0
        messages = await self.repo.list_messages(node_id)
        chunks = _chunk_messages(
            messages,
            target_tokens=self.cfg.chunk_target_tokens,
            overlap_tokens=self.cfg.chunk_overlap_tokens,
        )
        if not chunks:
            return 0
        # Incremental tail chunk indexing: reuse unchanged chunk vectors from SQLite (O(1) per turn!)
        header = await self._build_branch_header(node)
        contextualized = [f"{header}\n{c[3]}" for c in chunks]
        existing_rows = await self.repo.list_node_chunk_embeddings(node.id)
        existing_by_key: dict[tuple[int, int | None, int | None, str], bytes] = {
            (c_idx, s_start, s_end, content): blob
            for c_idx, s_start, s_end, content, blob in existing_rows
        }

        vectors_by_pos: list[list[float] | None] = [None] * len(chunks)
        missing_positions: list[int] = []
        missing_texts: list[str] = []
        for pos, ((c_idx, s_start, s_end, _raw), ctx_text) in enumerate(zip(chunks, contextualized)):
            cached_blob = existing_by_key.get((c_idx, s_start, s_end, ctx_text))
            if cached_blob is not None:
                vectors_by_pos[pos] = list(unpack_vector(cached_blob))
            else:
                missing_positions.append(pos)
                missing_texts.append(ctx_text)

        if not missing_positions and len(existing_rows) == len(chunks):
            return len(chunks)

        if missing_texts:
            new_vecs = await self.embedder.embed(missing_texts)
            for pos, vec in zip(missing_positions, new_vecs):
                vectors_by_pos[pos] = vec

        payload = [
            (c_idx, s_start, s_end, ctx_text, vec or [], node.summary_version)
            for (c_idx, s_start, s_end, _raw), ctx_text, vec in zip(
                chunks, contextualized, vectors_by_pos
            )
        ]
        return await self.repo.replace_node_embeddings(
            tree_id=node.tree_id,
            node_id=node.id,
            path=node.path,
            granularity="chunk",
            items=payload,
        )

    async def index_node_summary(self, node_id: str) -> int:
        if not self.cfg.enabled:
            return 0
        node = await self.repo.get_node(node_id)
        if node is None:
            return 0
        summary_body = node.summary or node.summary_brief or node.goal
        if not summary_body.strip():
            return 0
        header = await self._build_branch_header(node)
        text = f"{header}\nSummary: {summary_body}".strip()
        vectors = await self.embedder.embed([text])
        return await self.repo.replace_node_embeddings(
            tree_id=node.tree_id,
            node_id=node.id,
            path=node.path,
            granularity="summary",
            items=[(0, 1 if node.last_msg_seq > 0 else None, node.last_msg_seq or None, text, vectors[0], node.summary_version)],
        )

    async def index_node_facts(self, node_id: str) -> int:
        """Index Tier-2.5 provenance-linked atomic temporal facts for `node_id`."""
        if not self.cfg.enabled or not self.cfg.enable_atomic_facts or not self.ablations.enable_atomic_facts:
            return 0
        node = await self.repo.get_node(node_id)
        if node is None:
            return 0
        facts = await self.repo.list_branch_facts(
            node.tree_id, node_id=node.id, active_only=True, exclude_archived=False
        )
        if not facts:
            return await self.repo.replace_node_embeddings(
                tree_id=node.tree_id,
                node_id=node.id,
                path=node.path,
                granularity="fact",
                items=[],
            )
        header = await self._build_branch_header(node)
        texts: list[str] = []
        for f in facts:
            date_tag = f" [Date: {f.resolved_date}]" if f.resolved_date else ""
            ent_tag = f" [Entities: {', '.join(f.entities)}]" if f.entities else ""
            texts.append(f"{header}{date_tag}{ent_tag}\nFact: {f.fact_text}")
        vectors = await self.embedder.embed(texts)
        payload = [
            (idx, f.source_msg_seq, f.source_msg_seq, txt, vec, node.summary_version)
            for idx, (f, txt, vec) in enumerate(zip(facts, texts, vectors))
        ]
        return await self.repo.replace_node_embeddings(
            tree_id=node.tree_id,
            node_id=node.id,
            path=node.path,
            granularity="fact",
            items=payload,
        )

    async def refresh_subtree_rollup(self, parent_id: str) -> int:
        """Fold direct children's summaries, rollups, and structured key identifiers into `parent_id`."""
        if not self.cfg.enabled or not self.ablations.enable_subtree_rollups:
            return 0
        parent = await self.repo.get_node(parent_id)
        if parent is None:
            return 0
        children = await self.repo.get_children(parent_id)
        if not children:
            return 0

        lines: list[str] = [f"Subtree rollup under '{parent.title}' ({parent.id}):"]
        if parent.summary:
            lines.append(f"- Self summary: {parent.summary[:280]}")
        key_facts: list[str] = []
        for ch in children:
            if ch.status == "ARCHIVED":
                continue
            snippet = ch.summary or ch.rollup_summary or ch.goal or ch.summary_brief
            if snippet:
                lines.append(
                    f"- Child [{ch.title} | {ch.id} | {ch.status}]: {' '.join(snippet.split())[:320]}"
                )
                key_facts.extend(extract_key_identifiers(snippet, limit=8))
            elif ch.rollup_summary:
                lines.append(f"- Subtree [{ch.title}]: {ch.rollup_summary[:260]}")
                key_facts.extend(extract_key_identifiers(ch.rollup_summary, limit=8))

            # Also pull exact identifiers from the child's recent turns & artifacts so deep specs surface
            ch_arts = await self.repo.list_artifacts_by_node(ch.id)
            for a in ch_arts[:5]:
                key_facts.append(f"{a.slug}@v{a.version}")
            ch_msgs = await self.repo.list_messages(ch.id)
            for m in ch_msgs[-4:]:
                if m.role in ("user", "assistant"):
                    key_facts.extend(extract_key_identifiers(m.content, limit=6))

        dedup_facts = list(dict.fromkeys(key_facts))[:24]
        if dedup_facts:
            lines.append("Key Decisions & Identifiers: " + ", ".join(dedup_facts))

        rollup_text = "\n".join(lines)
        await self.repo.update_node(parent_id, rollup_summary=rollup_text)
        vectors = await self.embedder.embed([rollup_text])
        return await self.repo.replace_node_embeddings(
            tree_id=parent.tree_id,
            node_id=parent.id,
            path=parent.path,
            granularity="rollup",
            items=[(0, None, None, rollup_text, vectors[0], parent.summary_version)],
        )

    async def propagate_rollups_upward(self, node_id: str) -> int:
        """
        Propagate Tier-3 subtree rollups bottom-up from `node_id`'s parent, bounded by
        `cfg.max_rollup_depth` so deep auto-compaction chains stay O(1) instead of O(D).
        """
        if not self.cfg.enabled or not self.ablations.enable_subtree_rollups:
            return 0
        node = await self.repo.get_node(node_id)
        if node is None:
            return 0
        ancestors = await self.repo.get_ancestors(node)
        updated = 0
        # Walk bottom-up: direct parent first, capped at max_rollup_depth
        for anc in list(reversed(ancestors))[: self.cfg.max_rollup_depth]:
            updated += await self.refresh_subtree_rollup(anc.id)
        return updated

    async def index_tree(self, tree_id: str) -> dict[str, int]:
        """Full re-index of all tiers (chunks, summaries, atomic facts, subtree rollups)."""
        nodes = await self.repo.list_nodes(tree_id)
        counts = {"chunk": 0, "summary": 0, "fact": 0, "rollup": 0}
        for n in nodes:
            counts["chunk"] += await self.index_node_chunks(n.id)
            counts["summary"] += await self.index_node_summary(n.id)
            counts["fact"] += await self.index_node_facts(n.id)
        # Rollups bottom-up by descending depth
        for n in sorted(nodes, key=lambda x: x.depth, reverse=True):
            if n.child_count > 0:
                counts["rollup"] += await self.refresh_subtree_rollup(n.id)
        return counts

    # ------------------------------------------------------------------
    # Hybrid Retrieval (BM25 + Vector RRF + Scope/Status/Recency Filters)
    # ------------------------------------------------------------------

    async def search(
        self,
        tree_id: str,
        query: str,
        *,
        scope: SearchScope = "all",
        current_node: NodeModel | None = None,
        top_k: int = 5,
        status_filter: set[NodeStatus] | None = None,
        granularities: set[RagGranularity] | None = None,
        max_per_node: int | None = None,
        use_reranker: bool = True,
    ) -> list[SearchHit]:
        if not query.strip():
            return []

        if not self.cfg.enabled or not self.ablations.enable_hybrid_rag:
            return await self.repo.search_tree_fts(
                tree_id=tree_id,
                query=query,
                scope=scope,
                current_node=current_node,
                top_k=top_k,
            )

        all_nodes = await self.repo.list_nodes(tree_id)
        node_by_id = {n.id: n for n in all_nodes}
        eligible_ids = self.repo.filter_node_ids_by_scope(all_nodes, scope, current_node)
        # Dead-end quarantine: exclude ARCHIVED branches across all RAG tiers unless explicitly requested
        effective_statuses: set[NodeStatus] = (
            set(status_filter) if status_filter is not None else {"ACTIVE", "COMPLETED"}
        )
        eligible_ids = {
            nid for nid in eligible_ids if nid in node_by_id and node_by_id[nid].status in effective_statuses
        }
        if not eligible_ids:
            return []

        default_grans: set[RagGranularity] = {"chunk", "summary", "rollup"}
        if self.cfg.enable_atomic_facts and self.ablations.enable_atomic_facts:
            default_grans.add("fact")
        active_granularities: set[RagGranularity] = set(granularities or default_grans)
        if not self.ablations.enable_subtree_rollups:
            active_granularities.discard("rollup")
        if not self.cfg.enable_atomic_facts or not self.ablations.enable_atomic_facts:
            active_granularities.discard("fact")

        # Lazy bootstrap: if no embeddings exist for eligible nodes yet, index chunks/summaries on demand
        if await self.repo.count_embeddings(tree_id) == 0:
            await self.index_tree(tree_id)

        query_vec = (await self.embedder.embed([query]))[0]
        candidate_limit = max(top_k * 6, 48) if max_per_node else max(top_k * 4, 24)

        vec_ranked = await self.repo.query_embeddings_vector_sql(
            tree_id=tree_id,
            eligible_node_ids=eligible_ids,
            query_vector=query_vec,
            limit=candidate_limit,
            granularities=active_granularities,
        )
        bm25_ranked = await self.repo.query_embeddings_bm25(
            tree_id=tree_id,
            eligible_node_ids=eligible_ids,
            query=query,
            limit=candidate_limit,
            granularities=active_granularities,
        )

        # Reciprocal Rank Fusion over embedding records
        rrf_k = self.cfg.rrf_k
        fused: dict[str, dict[str, Any]] = {}

        for rank_idx, (rec, sim) in enumerate(vec_ranked, start=1):
            slot = fused.setdefault(
                rec.id, {"rec": rec, "rrf": 0.0, "vec": 0.0, "bm25": 0.0}
            )
            slot["vec"] = max(0.0, sim)
            # Only reward positive similarity so noise vectors don't dominate exact BM25 hits
            if sim > 0.0:
                slot["rrf"] += self.cfg.vector_weight * (1.0 / (rrf_k + rank_idx)) + (
                    self.cfg.vector_weight * 0.15 * sim
                )

        for rank_idx, (rec, bm25_s) in enumerate(bm25_ranked, start=1):
            slot = fused.setdefault(
                rec.id, {"rec": rec, "rrf": 0.0, "vec": 0.0, "bm25": 0.0}
            )
            slot["bm25"] = bm25_s
            slot["rrf"] += self.cfg.bm25_weight * (1.0 / (rrf_k + rank_idx)) + (
                self.cfg.bm25_weight * 0.15 * bm25_s
            )

        if not fused:
            return await self.repo.search_tree_fts(
                tree_id=tree_id,
                query=query,
                scope=scope,
                current_node=current_node,
                top_k=top_k,
            )

        now_ts = datetime.now(timezone.utc).timestamp()
        q_tokens = {w.lower() for w in re.findall(r"[A-Za-z0-9_]{3,}", query)}
        allow_archived_facts = bool(status_filter and "ARCHIVED" in status_filter)
        hits: list[SearchHit] = []
        for entry in fused.values():
            emb_rec: EmbeddingRecord = entry["rec"]
            node = node_by_id.get(emb_rec.node_id)
            if node is None:
                continue
            # Dead-end quarantine: facts from ARCHIVED branches never surface unless explicitly requested
            if emb_rec.granularity == "fact" and node.status == "ARCHIVED" and not allow_archived_facts:
                continue
            recency_bonus = 0.0
            try:
                age_hours = max(
                    0.0, (now_ts - datetime.fromisoformat(emb_rec.updated_at).timestamp()) / 3600.0
                )
                recency_bonus = self.cfg.recency_boost * (1.0 / (1.0 + age_hours / 24.0)) * 0.02
            except Exception:
                pass
            entity_bonus = 0.0
            if emb_rec.granularity == "fact" and q_tokens:
                ent_m = re.search(r"\[Entities:\s*([^\]]+)\]", emb_rec.content)
                if ent_m:
                    ent_words = {
                        w.lower() for w in re.findall(r"[A-Za-z0-9_]{3,}", ent_m.group(1))
                    }
                    matched_ents = len(q_tokens & ent_words)
                    if matched_ents > 0:
                        entity_bonus = min(0.05, 0.02 * matched_ents)
            score = round(entry["rrf"] + recency_bonus + entity_bonus, 6)
            src: Literal["hybrid", "vector", "bm25"] = (
                "hybrid"
                if (entry["vec"] > 0 and entry["bm25"] > 0)
                else ("vector" if entry["vec"] > 0 else "bm25")
            )
            seq_range = (
                (emb_rec.seq_start, emb_rec.seq_end)
                if emb_rec.seq_start is not None and emb_rec.seq_end is not None
                else None
            )
            hits.append(
                SearchHit(
                    node_id=node.id,
                    node_title=node.title,
                    node_status=node.status,
                    node_kind=node.node_kind,
                    path=node.path,
                    summary_version=node.summary_version,
                    summary=node.summary,
                    snippet=emb_rec.content[:380],
                    score=score,
                    bm25_score=round(entry["bm25"], 4),
                    vector_score=round(entry["vec"], 4),
                    granularity=emb_rec.granularity,
                    matched_seq=emb_rec.seq_start,
                    seq_range=seq_range,
                    source=src,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        if max_per_node is not None and max_per_node > 0:
            per_node_counts: dict[str, int] = {}
            diversified: list[SearchHit] = []
            for h in hits:
                cnt = per_node_counts.get(h.node_id, 0)
                if cnt < max_per_node:
                    per_node_counts[h.node_id] = cnt + 1
                    diversified.append(h)
            hits = diversified
        if use_reranker and self.reranker is not None and hits:
            return await self.reranker.rerank(query, hits[: max(top_k * 3, 24)], top_k)
        return hits[:top_k]
