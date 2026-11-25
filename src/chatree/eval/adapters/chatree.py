from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Protocol, runtime_checkable

_RLM_LOCAL_REPL_LOCK = threading.Lock()

from chatree.config import AblationConfig, ChatreeConfig
from chatree.protocols import Embedder, LLMProvider, Reranker
from chatree.providers.mock import MockLLMProvider
from chatree.providers.openai_compat import resolve_provider_endpoint
from chatree.sdk.client import Chatree, DialogueTree


from chatree.eval.adapters.base import AdapterAnswer


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
        )

    def get_ingest_usage(self) -> tuple[int, int]:
        return self._ingest_tokens_in, self._ingest_tokens_out

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
        await self._ct.rag.index_node_chunks(nid)
        if self.use_provided_summaries and summary and summary.strip():
            sum_text = summary.strip()
            await self._tree.node(nid).finish(
                summary=sum_text, auto_commit=True, return_to_parent=False
            )
            self._node_summaries[nid] = sum_text
        elif self.generate_session_summaries and not self.model.startswith("mock/"):
            import inspect
            from chatree.engine.staleness import parse_summary_and_facts

            msgs = await self._ct.repo.list_messages(nid)
            sess_date = self._node_dates.get(nid, "")
            date_ctx = f"Session date: {sess_date}\n" if sess_date else ""
            transcript = "\n".join(f"{m.role}: {m.content}" for m in msgs)[:7000]
            sum_kwargs: dict[str, Any] = {}
            if "max_tokens" in inspect.signature(self._ct.llm.complete).parameters:
                sum_kwargs["max_tokens"] = 900
            sresp = await self._ct.llm.complete(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write a crisp 2-3 sentence summary of this conversation session preserving "
                            "key people, events, dates, plans, preferences, and concrete facts.\n"
                            "Then on a new final line output `FACTS_JSON:` followed by a compact JSON array of "
                            "up to 8 atomic facts: "
                            '[{"fact": "short 1-sentence fact naming exact speaker/person and resolved absolute date if applicable", '
                            '"entities": ["PersonOrTopic"], "resolved_date": "YYYY-MM-DD or empty"}]. '
                            "CRITICAL: resolve relative time words ('yesterday', 'last Saturday', 'two weeks ago', "
                            "'next month') into explicit absolute calendar dates using the Session date!"
                        ),
                    },
                    {"role": "user", "content": f"{date_ctx}{transcript}"},
                ],
                tools=None,
                temperature=0.0,
                **sum_kwargs,
            )
            self._ingest_tokens_in += sresp.tokens_in
            self._ingest_tokens_out += sresp.tokens_out
            sum_text, extracted_facts = parse_summary_and_facts(sresp.content)
            sum_text = sum_text.strip() or sresp.content.strip()
            await self._ct.repo.commit_node_summary(nid, sum_text, status="COMPLETED")
            await self._ct.rag.index_node_summary(nid)
            if extracted_facts and self.ablations.enable_atomic_facts:
                await self._ct.repo.replace_branch_facts(
                    self._tree.id, nid, extracted_facts, source_msg_seq=len(msgs) or None
                )
                await self._ct.rag.index_node_facts(nid)
            await self._ct.rag.propagate_rollups_upward(nid)
            self._node_summaries[nid] = sum_text
        elif self.generate_session_summaries:
            await self._tree.node(nid).finish(
                summary=None, auto_commit=True, return_to_parent=False
            )
            nm = await self._ct.repo.get_node(nid)
            self._node_summaries[nid] = (nm.summary if nm else "") or session_id
        else:
            # Index lightweight local header + first/last snippet without burning an extra LLM call
            msgs = await self._ct.repo.list_messages(nid)
            excerpt = " | ".join(m.content[:140] for m in msgs[:2] + msgs[-2:]) if msgs else session_id
            await self._ct.repo.commit_node_summary(nid, excerpt[:420], status="COMPLETED")
            await self._ct.rag.index_node_summary(nid)
            await self._ct.rag.propagate_rollups_upward(nid)
            self._node_summaries[nid] = excerpt[:260]

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
