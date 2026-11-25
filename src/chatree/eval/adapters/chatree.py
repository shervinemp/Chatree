from __future__ import annotations

import asyncio
import time
from typing import Any, Sequence

from chatree.config import AblationConfig, ChatreeConfig
from chatree.eval.adapters.base import AdapterAnswer
from chatree.protocols import Embedder, LLMProvider, Reranker
from chatree.providers.mock import MockLLMProvider
from chatree.sdk.client import Chatree, DialogueTree


class ChatreeSystemAdapter:
    """
    Adapter evaluating Chatree with configurable `AblationConfig`, `oracle_routing=False`
    (non-oracle 3-tier hybrid search + causal freshness + shadow turns), and real/mock LLMs.
    """

    def __init__(
        self,
        *,
        name: str = "chatree_full",
        model: str = "mock/chatree-demo",
        shadow_model: str = "inherit",
        summary_model: str = "inherit",
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        ablations: AblationConfig | None = None,
        max_ask_targets: int = 4,
        generate_session_summaries: bool = False,
        use_provided_summaries: bool = False,
        include_branch_summary_map: bool = False,
        concise_synthesis: bool = True,
        enable_recursive_hop: bool = True,
        ingest_concurrency: int = 4,
    ) -> None:
        self._name = name
        self.model = model
        self.shadow_model = shadow_model
        self.summary_model = summary_model
        self.llm = llm or (MockLLMProvider(stream_delay_s=0.0) if model.startswith("mock/") else None)
        self.embedder = embedder
        self.reranker = reranker
        self.ablations = ablations or AblationConfig()
        self.max_ask_targets = max_ask_targets
        self.generate_session_summaries = generate_session_summaries
        self.use_provided_summaries = use_provided_summaries
        self.include_branch_summary_map = include_branch_summary_map
        self.concise_synthesis = concise_synthesis
        self.enable_recursive_hop = enable_recursive_hop
        self.ingest_concurrency = max(1, ingest_concurrency)
        self._ct: Chatree | None = None
        self._tree: DialogueTree | None = None
        self._session_nodes: dict[str, str] = {}
        self._node_dates: dict[str, str] = {}
        self._node_order: dict[str, int] = {}
        self._node_summaries: dict[str, str] = {}
        self._node_titles: dict[str, str] = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0
        self._root_id: str = ""

    def clone(self) -> ChatreeSystemAdapter:
        """Create an isolated fresh adapter sharing immutable provider/embedder/reranker instances."""
        return ChatreeSystemAdapter(
            name=self._name,
            model=self.model,
            shadow_model=self.shadow_model,
            summary_model=self.summary_model,
            llm=self.llm,
            embedder=self.embedder,
            reranker=self.reranker,
            ablations=self.ablations.model_copy(),
            max_ask_targets=self.max_ask_targets,
            generate_session_summaries=self.generate_session_summaries,
            use_provided_summaries=self.use_provided_summaries,
            include_branch_summary_map=self.include_branch_summary_map,
            concise_synthesis=self.concise_synthesis,
            enable_recursive_hop=self.enable_recursive_hop,
            ingest_concurrency=self.ingest_concurrency,
        )

    def get_ingest_usage(self) -> tuple[int, int]:
        return self._ingest_tokens_in, self._ingest_tokens_out

    async def archive_session(self, session_id: str, *, reason: str = "") -> None:
        """Quarantine a scrapped/dead-end session using Chatree's native `archive_branch` primitive."""
        assert self._tree is not None
        nid = self._session_nodes.get(session_id)
        if not nid or nid == self._root_id:
            return
        await self._tree.archive(nid, reason=reason, switch_to_parent=False)
        self._node_summaries.pop(nid, None)

    @property
    def name(self) -> str:
        return self._name

    async def reset(self, conversation_id: str) -> None:
        await self.close()
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            shadow_model=self.shadow_model,
            summary_model=self.summary_model,
            auto_commit_handoffs=True,
            ablations=self.ablations,
        )
        self._ct = Chatree.in_memory(
            config=cfg, llm=self.llm, embedder=self.embedder, reranker=self.reranker
        )
        await self._ct.startup()
        self._tree = await self._ct.create_tree(
            conversation_id,
            default_model=self.model,
            shadow_model=self.shadow_model,
            summary_model=self.summary_model,
            auto_commit_handoffs=True,
        )
        root = await self._tree.root()
        self._root_id = root.id
        self._session_nodes = {"root": root.id}
        self._node_dates = {}
        self._node_order = {}
        self._node_summaries = {}
        self._node_titles = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0

    async def start_session(
        self,
        session_id: str,
        *,
        title: str = "",
        date_time: str = "",
        parent_session_id: str | None = None,
    ) -> None:
        assert self._tree is not None and self._ct is not None
        parent_id = self._session_nodes.get(parent_session_id or "root", self._root_id)
        label = title or session_id
        goal = f"{label} | Date: {date_time}" if date_time else label
        child = await self._tree.spawn(
            goal=goal,
            parent_id=parent_id,
            title=label,
            mode="clean",
            summary_brief=f"Session date: {date_time}" if date_time else "",
            switch_head=False,
        )
        self._session_nodes[session_id] = child.id
        self._node_dates[child.id] = date_time
        self._node_order[child.id] = len(self._node_order)
        self._node_titles[child.id] = label

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        assert self._tree is not None and self._ct is not None
        if session_id not in self._session_nodes:
            await self.start_session(session_id, title=session_id, date_time=date_time)
        nid = self._session_nodes[session_id]
        stamped = f"[{date_time}] {content}" if date_time else content
        norm_role = "user" if role.lower() in ("user", "human", "speaker_a") else "assistant"
        await self._ct.repo.append_message(
            node_id=nid, tree_id=self._tree.id, role=norm_role, content=stamped, mark_dirty=True
        )

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        assert self._tree is not None and self._ct is not None
        nid = self._session_nodes.get(session_id)
        if not nid or nid == self._root_id:
            return
        if self.use_provided_summaries and summary and summary.strip():
            await self._ct.rag.index_node_chunks(nid)
            sum_text = summary.strip()
            await self._tree.node(nid).finish(
                summary=sum_text, auto_commit=True, return_to_parent=False
            )
            self._node_summaries[nid] = sum_text
        elif self.generate_session_summaries:
            tree_model = await self._ct.repo.get_tree(self._tree.id)
            node_model = await self._ct.repo.get_node(nid)
            if tree_model is not None and node_model is not None:
                sess_date = self._node_dates.get(nid, "")
                updated, _regen, tin, tout, _lat = await self._ct.staleness.ensure_fresh_summary(
                    tree_model,
                    node_model,
                    force=True,
                    new_status="COMPLETED",
                    session_date=sess_date,
                )
                self._ingest_tokens_in += tin
                self._ingest_tokens_out += tout
                self._node_titles[nid] = updated.title
                self._node_summaries[nid] = updated.summary or session_id
            else:
                self._node_summaries[nid] = session_id
            # Index Tier-1 chunks once AFTER exit-titling so chunk headers carry the semantic title
            await self._ct.rag.index_node_chunks(nid)
        else:
            await self._ct.rag.index_node_chunks(nid)
            # Index lightweight local header + first/last snippet without burning an extra LLM call
            msgs = await self._ct.repo.list_messages(nid)
            excerpt = " | ".join(m.content[:140] for m in msgs[:2] + msgs[-2:]) if msgs else session_id
            await self._ct.repo.commit_node_summary(nid, excerpt[:420], status="COMPLETED")
            await self._ct.rag.index_node_summary(nid)
            await self._ct.rag.propagate_rollups_upward(nid)
            self._node_summaries[nid] = excerpt[:260]

    async def ingest_conversation(self, sessions: Sequence[Any]) -> None:
        """
        High-throughput batch session ingest: creates all session nodes & appends verbatim turns
        in deterministic chronological order, then runs `finish_session()` (LLM summary/TITLE/FACTS_JSON
        + RAG indexing) concurrently across up to `self.ingest_concurrency` sessions, followed by
        deterministic dead-end archiving. Cuts multi-session ingest wall-clock ~4-5x with zero
        accuracy change.
        """
        for sess in sessions:
            await self.start_session(
                sess.session_id, title=sess.title, date_time=sess.date_time
            )
            for turn in sess.turns:
                await self.ingest_turn(
                    sess.session_id,
                    turn.role,
                    turn.content,
                    date_time=turn.date_time or sess.date_time,
                )

        if self.ingest_concurrency > 1 and len(sessions) > 1 and self.generate_session_summaries:
            sem = asyncio.Semaphore(self.ingest_concurrency)

            async def _finish_one(s: Any) -> None:
                async with sem:
                    await self.finish_session(s.session_id, summary=s.summary)

            await asyncio.gather(*(_finish_one(s) for s in sessions))
        else:
            for sess in sessions:
                await self.finish_session(sess.session_id, summary=sess.summary)

        for sess in sessions:
            if getattr(sess, "archived", False):
                await self.archive_session(
                    sess.session_id, reason=getattr(sess, "archive_reason", "")
                )

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        """
        Delegate directly to Chatree's core `TreeQueryEngine.deep_ask()` so external benchmarks
        exercise the EXACT same multi-hop shadow + Tier-2.5 atomic facts + synthesis code path
        available to live SDK/agent users (`tree.ask(..., deep=True)`).
        """
        assert self._tree is not None and self._ct is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question

        metrics_before = await self._tree.metrics()
        tree_m = await self._tree.model()
        root_m = await (await self._tree.root()).model()
        deep_out = await self._ct.query_engine.deep_ask(
            tree=tree_m,
            caller=root_m,
            query=prompt_q,
            scope="all",
            max_targets=self.max_ask_targets,
            enable_recursive_hop=self.enable_recursive_hop,
            include_branch_summary_map=self.include_branch_summary_map,
            concise_synthesis=self.concise_synthesis,
            node_dates=self._node_dates,
            node_order=self._node_order,
            node_titles=self._node_titles,
            node_summaries=self._node_summaries,
        )
        queried_ids: set[str] = set(deep_out.get("targets_queried") or [])
        final_text = str(deep_out.get("answer") or "")
        synth_tokens_in = int(deep_out.get("synth_tokens_in") or 0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics_after = await self._tree.metrics()
        root_tok = await self._ct.context.estimate_node_context_tokens(root_m)
        max_branch_tok = 0
        for tid in queried_ids:
            nd = await self._ct.repo.get_node(tid)
            if nd is not None:
                bt = await self._ct.context.estimate_node_context_tokens(nd)
                if bt > max_branch_tok:
                    max_branch_tok = bt
        peak_ctx = max(root_tok, synth_tokens_in, max_branch_tok)
        return AdapterAnswer(
            answer=final_text,
            tokens_in=max(0, metrics_after.total_tokens_in - metrics_before.total_tokens_in),
            tokens_out=max(0, metrics_after.total_tokens_out - metrics_before.total_tokens_out),
            cache_read_tokens=max(
                0, metrics_after.total_cache_read_tokens - metrics_before.total_cache_read_tokens
            ),
            peak_context_tokens=peak_ctx,
            latency_ms=round(elapsed_ms, 2),
            metadata={
                "targets_queried": sorted(queried_ids),
                "facts_used": int(deep_out.get("facts_used") or 0),
            },
        )

    async def close(self) -> None:
        if self._ct is not None:
            await self._ct.close()
            self._ct = None
            self._tree = None
            self._session_nodes.clear()
