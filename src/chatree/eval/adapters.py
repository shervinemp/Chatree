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
from chatree.sdk.client import Chatree, DialogueTree


@dataclass
class AdapterAnswer:
    answer: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    peak_context_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemorySystemAdapter(Protocol):
    """
    Unified interface for head-to-head benchmarking of Chatree against official upstream
    memory/context frameworks (`mem0ai`, `letta`, `graphiti-core`, Linear, Flat-RAG) on
    identical multi-session datasets (LongMemEval, LoCoMo) and branching scenarios.
    """

    @property
    def name(self) -> str: ...

    async def reset(self, conversation_id: str) -> None: ...

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None: ...

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None: ...

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None: ...

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer: ...

    async def close(self) -> None: ...


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
        assert self._tree is not None and self._ct is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question

        metrics_before = await self._tree.metrics()
        n_sessions = max(0, len(self._session_nodes) - 1)
        eff_targets = min(self.max_ask_targets, max(1, n_sessions)) if n_sessions else self.max_ask_targets
        ask_out = await self._tree.ask(
            prompt_q,
            scope="all",
            max_targets=eff_targets,
            caller_id=self._root_id,
        )
        queried_ids: set[str] = set(ask_out.get("targets_queried") or [])

        def _extract_found(out_dict: dict[str, Any]) -> list[dict[str, Any]]:
            raw = out_dict.get("found_answers")
            if raw is not None:
                return list(raw)
            return [
                a
                for a in (out_dict.get("answers") or [])
                if a.get("found_in_branch", True)
                and not str(a.get("answer") or "").strip().upper().startswith("NOT_FOUND_IN_BRANCH")
            ]

        found = _extract_found(ask_out)

        # Build compact chronological Tier-2 Branch Summary Map (Root's committed child summaries)
        ordered_nids = sorted(
            [nid for sid, nid in self._session_nodes.items() if sid != "root"],
            key=lambda nid: (self._node_dates.get(nid, ""), self._node_order.get(nid, 0)),
        )
        summary_map_lines = [
            f"- {nid} ({self._node_titles.get(nid, nid)}"
            + (f" | {self._node_dates[nid]}" if self._node_dates.get(nid) else "")
            + f"): {self._node_summaries.get(nid, '')[:600]}"
            for nid in ordered_nids
            if self._node_summaries.get(nid)
        ]
        summary_map_text = "\n".join(summary_map_lines)

        # Zero-LLM-cost Tier-2.5 atomic dated facts retrieval (NumPy cosine + BM25 + cross-encoder):
        # surfaces exact resolved-date atomic facts with source branch IDs for both Hop-2 routing
        # and chronological final synthesis without adding any extra LLM calls!
        retrieved_facts_text = ""
        if self.ablations.enable_atomic_facts and n_sessions > 0:
            fact_hits = await self._ct.rag.search(
                self._tree.id,
                prompt_q,
                top_k=10,
                granularities=("fact",),
            )
            if fact_hits:
                fact_hits_sorted = sorted(
                    fact_hits,
                    key=lambda h: (
                        self._node_dates.get(h.node_id, ""),
                        self._node_order.get(h.node_id, 0),
                    ),
                )
                fact_lines = []
                for fh in fact_hits_sorted:
                    snippet = fh.snippet.strip()
                    if "Fact:" in snippet:
                        snippet = snippet.split("Fact:", 1)[1].strip()
                    sdate = self._node_dates.get(fh.node_id, "")
                    date_sfx = f" | Session: {sdate}" if sdate else ""
                    fact_lines.append(
                        f"- [{fh.node_id} | {self._node_titles.get(fh.node_id, fh.node_id)}{date_sfx}] {snippet[:260]}"
                    )
                retrieved_facts_text = "\n".join(fact_lines)

        # Tier-2 Guided + RLM-inspired Hop-2 Shadow Drill-Down (real LLMs only):
        # Using the compact Branch Summary Map + Hop-1 evidence, Root can select up to 4 specific
        # unqueried branch IDs (or a refined search query) and run parallel read-only shadow turns
        # using the EXACT original dated `prompt_q`!
        q_low = question.lower()
        needs_broad = any(
            kw in q_low
            for kw in (
                "how many", "how much", "total", "all ", "recommend", "suggest", "would ",
                "likely", "prefer", "before", "after", "recent", "latest", "now", "currently",
                "how long", "when did", "how often",
            )
        )
        if (
            self.enable_recursive_hop
            and not self.model.startswith("mock/")
            and n_sessions > len(queried_ids)
            and (len(found) < 2 or needs_broad or self.include_branch_summary_map)
        ):
            hop1_preview = "\n".join(
                f"- {a.get('target_id')} ({a.get('target_title', a.get('target_id'))}): "
                f"{str(a.get('answer') or '')[:220]}"
                for a in found
            ) or "(none found in initial branches)"
            unqueried_map = "\n".join(
                ln for nid, ln in zip(ordered_nids, summary_map_lines) if nid not in queried_ids
            )
            plan_resp = await self._ct.llm.complete(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the root coordinator of a dialogue tree deciding whether to run a "
                            "second parallel read-only shadow drill-down on unqueried child branches. "
                            "If Hop-1 evidence already completely answers the question, return "
                            '{"need_more": false, "target_ids": [], "followup_query": ""}. '
                            "Otherwise pick up to 4 relevant unqueried branch IDs from the Branch Summary Map "
                            "AND/OR a focused follow-up search query: "
                            '{"need_more": true, "target_ids": ["n_..."], "followup_query": "..."}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {prompt_q}\n\n"
                            f"Hop-1 Verified Shadow Evidence:\n{hop1_preview}\n\n"
                            + (
                                f"Retrieved Tier-2.5 Atomic Dated Facts:\n{retrieved_facts_text}\n\n"
                                if retrieved_facts_text
                                else ""
                            )
                            + (
                                f"Unqueried Branch Summary Map:\n{unqueried_map[:16000]}"
                                if self.include_branch_summary_map and unqueried_map
                                else f"({n_sessions - len(queried_ids)} unqueried branches remain)"
                            )
                        ),
                    },
                ],
                tools=None,
                temperature=0.0,
            )
            await self._ct.repo.increment_node_usage(
                self._root_id, plan_resp.tokens_in, plan_resp.tokens_out, plan_resp.cache_read_tokens
            )
            m_plan = re.search(r"\{.*\}", plan_resp.content, re.DOTALL)
            if m_plan:
                try:
                    plan = json.loads(m_plan.group(0))
                    if plan.get("need_more"):
                        valid_unqueried = set(ordered_nids) - queried_ids
                        raw_tids = plan.get("target_ids") or []
                        chosen_tids = [
                            str(t) for t in raw_tids if str(t) in valid_unqueried
                        ][:4]
                        seen_tids = {str(a.get("target_id")) for a in found}
                        if chosen_tids:
                            # Direct parallel read-only shadow requests to Root-selected branches!
                            hop2_res = await asyncio.gather(
                                *(
                                    self._tree.request(tid, prompt=prompt_q, caller_id=self._root_id)
                                    for tid in chosen_tids
                                )
                            )
                            queried_ids.update(chosen_tids)
                            for out_d in hop2_res:
                                if (
                                    out_d.get("found_in_branch", True)
                                    and not str(out_d.get("answer") or "")
                                    .strip()
                                    .upper()
                                    .startswith("NOT_FOUND_IN_BRANCH")
                                ):
                                    tid = str(out_d.get("target_id") or "")
                                    if tid and tid not in seen_tids:
                                        seen_tids.add(tid)
                                        found.append(out_d)
                        else:
                            fq = str(plan.get("followup_query") or "").strip()
                            if fq:
                                hop2_out = await self._tree.ask(
                                    fq,
                                    scope="all",
                                    max_targets=min(4, max(1, n_sessions - len(queried_ids))),
                                    caller_id=self._root_id,
                                    exclude_node_ids=queried_ids,
                                    shadow_prompt=prompt_q,
                                )
                                queried_ids.update(hop2_out.get("targets_queried") or [])
                                for a in _extract_found(hop2_out):
                                    tid = str(a.get("target_id") or "")
                                    if tid and tid not in seen_tids:
                                        seen_tids.add(tid)
                                        found.append(a)
                except Exception:
                    pass

        # Sort verified branch answers chronologically (oldest -> newest) so temporal & update
        # questions see the exact timeline before concise final synthesis!
        found_sorted = sorted(
            found,
            key=lambda a: (
                self._node_dates.get(str(a.get("target_id") or ""), ""),
                self._node_order.get(str(a.get("target_id") or ""), 0),
            ),
        )

        synth_tokens_in = 0
        if not found_sorted and not (self.include_branch_summary_map and summary_map_text and not self.model.startswith("mock/")):
            final_text = "I do not know — this information was not mentioned in any prior session."
        elif len(found_sorted) == 1 and (self.model.startswith("mock/") or not self.concise_synthesis):
            final_text = str(found_sorted[0].get("answer") or found_sorted[0].get("summary") or "")
        else:
            # Chronologically ordered evidence synthesis (verbatim shadow evidence + optional Tier-2 summary map)
            evidence = "\n\n".join(
                f"[Branch: {a.get('target_title', a.get('target_id'))}"
                + (
                    f" | Date: {self._node_dates[str(a.get('target_id'))]}"
                    if self._node_dates.get(str(a.get("target_id") or ""))
                    else ""
                )
                + f"]\n{a.get('answer') or a.get('summary') or ''}"
                for a in found_sorted
            ) or "(No direct verbatim shadow hits — check Branch Summary Map carefully; abstain if unmentioned.)"
            facts_section = (
                f"\n\n[Retrieved Atomic Dated Facts (Tier-2.5, Provenance-Tagged)]\n{retrieved_facts_text}"
                if (retrieved_facts_text and not self.model.startswith("mock/"))
                else ""
            )
            summary_section = (
                f"\n\n[Branch Summary Map (Chronological Overview)]\n{summary_map_text[:16000]}"
                if (self.include_branch_summary_map and summary_map_text and not self.model.startswith("mock/"))
                else ""
            )
            synth = await self._ct.llm.complete(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Synthesize one direct, concise, accurate answer from the chronologically "
                            "ordered verified branch shadow answers (highest authority), provenance-tagged "
                            "atomic dated facts, and branch overview (oldest to newest). "
                            "- If facts changed over time, use the latest dated update. "
                            "- Use resolved calendar dates from atomic facts & session headers to compute exact dates/durations. "
                            "- For counting/aggregation questions, combine distinct items across all branches. "
                            "- For recommendation, preference, or hypothetical/inference questions ('Would X likely...', "
                            "'What fields...', 'What kind of...'), NEVER reply 'I do not know' — always commit to a "
                            "concrete grounded inference (e.g., likely yes/no + specific traits/clues from the branches). "
                            "- For concrete factual questions whose specific event, entity, or person NEVER occurred "
                            "in any branch (e.g. wrong person or fabricated event), verify names strictly and state clearly "
                            "that you do not know."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"[Verified Shadow Evidence]\n{evidence}{facts_section}{summary_section}\n\nQuestion: {prompt_q}",
                    },
                ],
                tools=None,
                temperature=0.0,
            )
            await self._ct.repo.increment_node_usage(
                self._root_id, synth.tokens_in, synth.tokens_out, synth.cache_read_tokens
            )
            synth_tokens_in = synth.tokens_in
            final_text = synth.content

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics_after = await self._tree.metrics()
        root_tok = await self._ct.context.estimate_node_context_tokens(await (await self._tree.root()).model())
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
            metadata={"targets_queried": sorted(queried_ids)},
        )

    async def close(self) -> None:
        if self._ct is not None:
            await self._ct.close()
            self._ct = None
            self._tree = None
            self._session_nodes.clear()


class FullContextLinearAdapter:
    """Fair Full-Context Linear baseline: puts all prior session turns in one flat prompt."""

    def __init__(
        self,
        *,
        name: str = "full_context_linear",
        model: str = "mock/chatree-demo",
        llm: LLMProvider | None = None,
        max_context_tokens: int = 120_000,
    ) -> None:
        self._name = name
        self.model = model
        self.llm = llm or MockLLMProvider(stream_delay_s=0.0)
        self.max_context_tokens = max_context_tokens
        self._turns: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def clone(self) -> FullContextLinearAdapter:
        return FullContextLinearAdapter(
            name=self._name,
            model=self.model,
            llm=self.llm,
            max_context_tokens=self.max_context_tokens,
        )

    async def reset(self, conversation_id: str) -> None:
        del conversation_id
        self._turns = []

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del parent_session_id
        header = f"--- Session {title or session_id}" + (f" ({date_time})" if date_time else "") + " ---"
        self._turns.append({"role": "user", "content": header})

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        del session_id
        norm_role = "user" if role.lower() in ("user", "human", "speaker_a") else "assistant"
        stamped = f"[{date_time}] {content}" if date_time else content
        self._turns.append({"role": norm_role, "content": stamped})

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del session_id, summary

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        # Guard against exceeding provider context window (e.g. DeepSeek 128k on LongMemEval)
        budget_chars = max(4000, (self.max_context_tokens - 600) * 4)
        kept_turns: list[dict[str, str]] = []
        used_chars = 0
        truncated = False
        for t in reversed(self._turns):
            tlen = len(t.get("content", "")) + 16
            if used_chars + tlen > budget_chars and kept_turns:
                truncated = True
                break
            kept_turns.append(t)
            used_chars += tlen
        kept_turns.reverse()
        msgs: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question concisely based on the full conversation history. "
                    "If information changed over time, use the latest update. If unknown, say so."
                ),
            },
            *kept_turns,
            {"role": "user", "content": prompt_q},
        ]
        resp = await self.llm.complete(model=self.model, messages=msgs, tools=None, temperature=0.0)
        return AdapterAnswer(
            answer=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cache_read_tokens=resp.cache_read_tokens,
            peak_context_tokens=resp.tokens_in,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            metadata={"truncated": truncated},
        )

    async def close(self) -> None:
        self._turns = []


class FlatHybridRAGAdapter:
    """Fair Flat Hybrid RAG baseline (BM25 + dense vector chunks -> LLM synthesis, no shadow turns)."""

    def __init__(
        self,
        *,
        name: str = "flat_hybrid_rag",
        model: str = "mock/chatree-demo",
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        top_k: int = 5,
    ) -> None:
        self._name = name
        self.model = model
        self.llm = llm or (MockLLMProvider(stream_delay_s=0.0) if model.startswith("mock/") else None)
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self._ct: Chatree | None = None
        self._tree: DialogueTree | None = None
        self._root_id: str = ""
        self._dirty_index: bool = False

    def clone(self) -> FlatHybridRAGAdapter:
        return FlatHybridRAGAdapter(
            name=self._name,
            model=self.model,
            llm=self.llm,
            embedder=self.embedder,
            reranker=self.reranker,
            top_k=self.top_k,
        )

    @property
    def name(self) -> str:
        return self._name

    async def reset(self, conversation_id: str) -> None:
        await self.close()
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            ablations=AblationConfig(enable_subtree_rollups=False, enable_shadow_turns=False),
        )
        self._ct = Chatree.in_memory(
            config=cfg, llm=self.llm, embedder=self.embedder, reranker=self.reranker
        )
        await self._ct.startup()
        self._tree = await self._ct.create_tree(conversation_id, default_model=self.model)
        self._root_id = (await self._tree.root()).id

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del parent_session_id
        assert self._ct is not None and self._tree is not None
        header = f"[Session {title or session_id}" + (f" | {date_time}" if date_time else "") + "]"
        await self._ct.repo.append_message(self._root_id, self._tree.id, "user", header)

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        del session_id
        assert self._ct is not None and self._tree is not None
        norm_role = "user" if role.lower() in ("user", "human", "speaker_a") else "assistant"
        stamped = f"[{date_time}] {content}" if date_time else content
        await self._ct.repo.append_message(self._root_id, self._tree.id, norm_role, stamped)

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del session_id, summary
        self._dirty_index = True

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._ct is not None and self._tree is not None
        if self._dirty_index:
            await self._ct.rag.index_node_chunks(self._root_id)
            self._dirty_index = False
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        hits = await self._tree.search(
            prompt_q, scope="all", top_k=self.top_k, granularities={"chunk"}
        )
        chunks_txt = "\n\n".join(f"- {h.snippet}" for h in hits)
        resp = await self._ct.llm.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question concisely using only the retrieved conversation snippets. "
                        "If facts changed over time, prefer the latest dated update. If absent, say so."
                    ),
                },
                {"role": "user", "content": f"[RAG Context]\n{chunks_txt}\n\nQuestion: {prompt_q}"},
            ],
            tools=None,
            temperature=0.0,
        )
        return AdapterAnswer(
            answer=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cache_read_tokens=resp.cache_read_tokens,
            peak_context_tokens=resp.tokens_in,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            metadata={"hits_count": len(hits)},
        )

    async def close(self) -> None:
        if self._ct is not None:
            await self._ct.close()
            self._ct = None
            self._tree = None


class Mem0OfficialAdapter:
    """
    Thin adapter wrapping the OFFICIAL upstream `mem0ai` Python SDK (`from mem0 import Memory`)
    for apples-to-apples head-to-head comparison—zero toy reimplementation.
    Requires `pip install mem0ai`.
    """

    def __init__(
        self,
        *,
        name: str = "mem0_official",
        model: str = "deepseek/deepseek-chat",
        llm: LLMProvider | None = None,
        mem0_config: dict[str, Any] | None = None,
        top_k: int = 8,
    ) -> None:
        self._name = name
        self.model = model
        self.llm = llm
        self.mem0_config = mem0_config
        self.top_k = top_k
        self._mem: Any = None
        self._user_id = "eval_user"
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._session_buf: dict[str, list[dict[str, str]]] = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0

    @property
    def name(self) -> str:
        return self._name

    def get_ingest_usage(self) -> tuple[int, int]:
        return self._ingest_tokens_in, self._ingest_tokens_out

    async def reset(self, conversation_id: str) -> None:
        await self.close()
        try:
            from mem0 import Memory  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Official Mem0 adapter requires the upstream package: `uv pip install mem0ai`."
            ) from exc
        self._tmpdir = tempfile.TemporaryDirectory(prefix="chatree-mem0-")
        self._user_id = f"chatree_{conversation_id}"
        self._session_buf = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0
        cfg = dict(self.mem0_config or {})
        if not cfg and "deepseek" in self.model.lower():
            ds_model = self.model.split("/", 1)[-1] if "/" in self.model else self.model
            cfg = {
                "llm": {"provider": "deepseek", "config": {"model": ds_model, "temperature": 0.0}},
                "embedder": {
                    "provider": "fastembed",
                    "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": str(Path(self._tmpdir.name) / "qdrant"),
                        "embedding_model_dims": 384,
                    },
                },
            }
        cfg.setdefault("history_db_path", str(Path(self._tmpdir.name) / "mem0_history.db"))
        self._mem = Memory.from_config(cfg) if cfg else Memory()

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del title, date_time, parent_session_id
        self._session_buf[session_id] = []

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        norm_role = "user" if role.lower() in ("user", "human", "speaker_a") else "assistant"
        stamped = f"[{date_time}] {content}" if date_time else content
        self._session_buf.setdefault(session_id, []).append({"role": norm_role, "content": stamped})

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del summary
        msgs = self._session_buf.pop(session_id, [])
        if msgs and self._mem is not None:
            # Estimate write-path extraction tokens (Mem0 performs LLM extraction + dedup per add)
            chars = sum(len(m["content"]) for m in msgs)
            self._ingest_tokens_in += max(200, chars // 4 + 350)
            res = await asyncio.to_thread(self._mem.add, msgs, user_id=self._user_id)
            out_chars = len(json.dumps(res, default=str))
            self._ingest_tokens_out += max(40, out_chars // 4)

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._mem is not None and self.llm is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question

        def _do_search() -> Any:
            try:
                return self._mem.search(
                    prompt_q, filters={"user_id": self._user_id}, limit=self.top_k
                )
            except (TypeError, ValueError):
                return self._mem.search(prompt_q, user_id=self._user_id, limit=self.top_k)

        raw = await asyncio.to_thread(_do_search)
        items = raw.get("results", raw) if isinstance(raw, dict) else raw
        mem_lines = [
            f"- {it.get('memory', it) if isinstance(it, dict) else it}" for it in (items or [])
        ]
        resp = await self.llm.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the user's question concisely using the retrieved Mem0 memories.",
                },
                {
                    "role": "user",
                    "content": f"[Mem0 Memories]\n" + "\n".join(mem_lines) + f"\n\nQuestion: {prompt_q}",
                },
            ],
            tools=None,
            temperature=0.0,
        )
        return AdapterAnswer(
            answer=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cache_read_tokens=resp.cache_read_tokens,
            peak_context_tokens=resp.tokens_in,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            metadata={"memories_retrieved": len(mem_lines)},
        )

    async def close(self) -> None:
        if self._mem is not None:
            for attr in ("vector_store", "_telemetry_vector_store"):
                try:
                    vs = getattr(self._mem, attr, None)
                    client = getattr(vs, "client", None)
                    if client is not None and hasattr(client, "close"):
                        client.close()
                except Exception:
                    pass
            try:
                db = getattr(self._mem, "db", None)
                conn = getattr(db, "connection", None)
                if conn is not None and hasattr(conn, "close"):
                    conn.close()
            except Exception:
                pass
        self._mem = None
        self._session_buf.clear()
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None


class LettaOfficialAdapter:
    """
    Thin adapter wrapping the OFFICIAL upstream `letta_client` / `letta` SDK for apples-to-apples
    head-to-head comparison against MemGPT/Letta—zero toy reimplementation.
    Requires `pip install letta-client` (pointing at a local Letta server).
    """

    def __init__(
        self,
        *,
        name: str = "letta_official",
        base_url: str = "http://localhost:8283",
        model: str = "openai/gpt-4o-mini",
        embedding: str = "openai/text-embedding-3-small",
    ) -> None:
        self._name = name
        self.base_url = base_url
        self.model = model
        self.embedding = embedding
        self._client: Any = None
        self._agent_id: str | None = None

    @property
    def name(self) -> str:
        return self._name

    async def reset(self, conversation_id: str) -> None:
        await self.close()
        try:
            from letta_client import Letta  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Official Letta adapter requires the upstream package: `uv pip install letta-client`."
            ) from exc
        self._client = Letta(base_url=self.base_url)
        agent = await asyncio.to_thread(
            self._client.agents.create,
            name=f"chatree_eval_{conversation_id}",
            model=self.model,
            embedding=self.embedding,
        )
        self._agent_id = agent.id

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del session_id, title, date_time, parent_session_id

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        assert self._client is not None and self._agent_id is not None
        stamped = f"[{session_id} | {date_time}] {role}: {content}"
        await asyncio.to_thread(
            self._client.agents.passages.create, agent_id=self._agent_id, text=stamped
        )

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del session_id, summary

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._client is not None and self._agent_id is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        resp = await asyncio.to_thread(
            self._client.agents.messages.create,
            agent_id=self._agent_id,
            messages=[{"role": "user", "content": prompt_q}],
        )
        texts = [
            getattr(m, "content", "")
            for m in getattr(resp, "messages", [])
            if getattr(m, "message_type", "") == "assistant_message"
        ]
        usage = getattr(resp, "usage", None)
        return AdapterAnswer(
            answer="\n".join(str(t) for t in texts if t),
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    async def close(self) -> None:
        if self._client is not None and self._agent_id is not None:
            try:
                await asyncio.to_thread(self._client.agents.delete, agent_id=self._agent_id)
            except Exception:
                pass
        self._client = None
        self._agent_id = None


class GraphitiOfficialAdapter:
    """
    Thin adapter wrapping the OFFICIAL upstream `graphiti-core` (Zep bi-temporal KG engine).
    Requires `pip install graphiti-core` and a configured graph driver instance.
    """

    def __init__(
        self,
        *,
        name: str = "graphiti_official",
        graphiti_instance: Any = None,
        model: str = "deepseek/deepseek-chat",
        llm: LLMProvider | None = None,
        top_k: int = 8,
    ) -> None:
        self._name = name
        self._graphiti = graphiti_instance
        self.model = model
        self.llm = llm
        self.top_k = top_k
        self._group_id = "eval"
        self._session_buf: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return self._name

    async def reset(self, conversation_id: str) -> None:
        if self._graphiti is None:
            try:
                import graphiti_core  # type: ignore[import-not-found]  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Official Graphiti/Zep adapter requires `uv pip install graphiti-core` "
                    "plus a connected Graphiti instance passed to `GraphitiOfficialAdapter(graphiti_instance=...)`."
                ) from exc
            raise ValueError("Pass an initialized `graphiti_instance` to GraphitiOfficialAdapter.")
        self._group_id = f"chatree_{conversation_id}"
        self._session_buf = {}

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del title, date_time, parent_session_id
        self._session_buf[session_id] = []

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        self._session_buf.setdefault(session_id, []).append(f"[{date_time}] {role}: {content}")

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del summary
        from datetime import datetime, timezone

        lines = self._session_buf.pop(session_id, [])
        if lines and self._graphiti is not None:
            await self._graphiti.add_episode(
                name=f"{self._group_id}_{session_id}",
                episode_body="\n".join(lines),
                source_description="conversation session",
                reference_time=datetime.now(timezone.utc),
                group_id=self._group_id,
            )

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._graphiti is not None and self.llm is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        edges = await self._graphiti.search(prompt_q, group_ids=[self._group_id], num_results=self.top_k)
        facts = [f"- {getattr(e, 'fact', str(e))}" for e in (edges or [])]
        resp = await self.llm.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer concisely using the retrieved bi-temporal knowledge graph facts.",
                },
                {
                    "role": "user",
                    "content": "[Graphiti Facts]\n" + "\n".join(facts) + f"\n\nQuestion: {prompt_q}",
                },
            ],
            tools=None,
            temperature=0.0,
        )
        return AdapterAnswer(
            answer=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cache_read_tokens=resp.cache_read_tokens,
            peak_context_tokens=resp.tokens_in,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            metadata={"facts_retrieved": len(facts)},
        )

    async def close(self) -> None:
        self._session_buf.clear()


class TreeSummaryOnlyAdapter:
    """
    Context-Folding / Hierarchical Summary-Only baseline (ReSum / FoldGRPO-style folding WITHOUT
    live read-only shadow turns). Summarizes each session at `finish_session` (tracking write-path
    LLM tokens!) and answers queries solely from retrieved session summaries.
    """

    def __init__(
        self,
        *,
        name: str = "tree_summary_only",
        model: str = "mock/chatree-demo",
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        top_k: int = 6,
    ) -> None:
        self._name = name
        self.model = model
        self.llm = llm or (MockLLMProvider(stream_delay_s=0.0) if model.startswith("mock/") else None)
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self._ct: Chatree | None = None
        self._tree: DialogueTree | None = None
        self._root_id: str = ""
        self._session_nodes: dict[str, str] = {}
        self._node_dates: dict[str, str] = {}
        self._node_order: dict[str, int] = {}
        self._node_titles: dict[str, str] = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0

    @property
    def name(self) -> str:
        return self._name

    def clone(self) -> TreeSummaryOnlyAdapter:
        return TreeSummaryOnlyAdapter(
            name=self._name,
            model=self.model,
            llm=self.llm,
            embedder=self.embedder,
            reranker=self.reranker,
            top_k=self.top_k,
        )

    def get_ingest_usage(self) -> tuple[int, int]:
        return self._ingest_tokens_in, self._ingest_tokens_out

    async def reset(self, conversation_id: str) -> None:
        await self.close()
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            ablations=AblationConfig(enable_shadow_turns=False),
        )
        self._ct = Chatree.in_memory(
            config=cfg, llm=self.llm, embedder=self.embedder, reranker=self.reranker
        )
        await self._ct.startup()
        self._tree = await self._ct.create_tree(conversation_id, default_model=self.model)
        self._root_id = (await self._tree.root()).id
        self._session_nodes = {"root": self._root_id}
        self._node_dates = {}
        self._node_order = {}
        self._node_titles = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        assert self._tree is not None
        parent_id = self._session_nodes.get(parent_session_id or "root", self._root_id)
        label = title or session_id
        child = await self._tree.spawn(
            goal=f"{label} | Date: {date_time}" if date_time else label,
            parent_id=parent_id,
            title=label,
            mode="clean",
            switch_head=False,
        )
        self._session_nodes[session_id] = child.id
        self._node_dates[child.id] = date_time
        self._node_order[child.id] = len(self._node_order)
        self._node_titles[child.id] = label

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        assert self._ct is not None and self._tree is not None
        if session_id not in self._session_nodes:
            await self.start_session(session_id, title=session_id, date_time=date_time)
        nid = self._session_nodes[session_id]
        stamped = f"[{date_time}] {content}" if date_time else content
        norm_role = "user" if role.lower() in ("user", "human", "speaker_a") else "assistant"
        await self._ct.repo.append_message(nid, self._tree.id, norm_role, stamped, mark_dirty=True)

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        assert self._ct is not None and self._tree is not None
        nid = self._session_nodes.get(session_id)
        if not nid or nid == self._root_id:
            return
        if summary and summary.strip():
            sum_text = summary.strip()
        else:
            msgs = await self._ct.repo.list_messages(nid)
            transcript = "\n".join(f"{m.role}: {m.content}" for m in msgs)[:6000]
            resp = await self._ct.llm.complete(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize this conversation session concisely in 3-5 bullet points, "
                            "preserving key facts, names, dates, preferences, and decisions."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ],
                tools=None,
                temperature=0.0,
            )
            self._ingest_tokens_in += resp.tokens_in
            self._ingest_tokens_out += resp.tokens_out
            sum_text = resp.content
        await self._ct.repo.commit_node_summary(nid, sum_text, status="COMPLETED")
        await self._ct.rag.index_node_summary(nid)

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._ct is not None and self._tree is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        hits = await self._tree.search(
            prompt_q, scope="all", top_k=self.top_k, granularities={"summary"}
        )
        ordered = sorted(
            hits,
            key=lambda h: (
                self._node_dates.get(h.node_id, ""),
                self._node_order.get(h.node_id, 0),
            ),
        )
        summaries_txt = "\n\n".join(
            f"[Session: {h.node_title}"
            + (f" | Date: {self._node_dates[h.node_id]}" if self._node_dates.get(h.node_id) else "")
            + f"]\n{h.summary or h.snippet}"
            for h in ordered
        )
        resp = await self._ct.llm.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer concisely using ONLY the retrieved folded session summaries below. "
                        "If facts changed over time, prefer the latest date. If unknown, say so."
                    ),
                },
                {"role": "user", "content": f"[Folded Summaries]\n{summaries_txt}\n\nQuestion: {prompt_q}"},
            ],
            tools=None,
            temperature=0.0,
        )
        return AdapterAnswer(
            answer=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cache_read_tokens=resp.cache_read_tokens,
            peak_context_tokens=resp.tokens_in,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            metadata={"summaries_retrieved": len(ordered)},
        )

    async def close(self) -> None:
        if self._ct is not None:
            await self._ct.close()
            self._ct = None
            self._tree = None
            self._session_nodes.clear()


class RLMOfficialAdapter:
    """
    Thin adapter wrapping MIT's OFFICIAL upstream Recursive Language Models library
    (`pip install rlms` -> `from rlm import RLM`, Zhang, Kraska & Khattab, arXiv:2512.24601)
    for apples-to-apples comparison on identical multi-session datasets.
    """

    def __init__(
        self,
        *,
        name: str = "rlm_official",
        model: str = "deepseek/deepseek-chat",
        backend: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        max_depth: int = 1,
        max_iterations: int = 10,
        max_concurrent_subcalls: int = 4,
        max_timeout: float = 90.0,
    ) -> None:
        self._name = name
        self.model = model
        self.backend = backend
        self.base_url = base_url
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_concurrent_subcalls = max_concurrent_subcalls
        self.max_timeout = max_timeout
        self._sessions: list[str] = []
        self._cur_lines: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    # Note: intentionally no `clone()` because upstream `rlms` LocalREPL mutates
    # process-global `os.chdir()` and `sys.stdout/sys.stderr` during execution.

    async def reset(self, conversation_id: str) -> None:
        del conversation_id
        self._sessions = []
        self._cur_lines = []

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del parent_session_id
        hdr = f"=== Session {title or session_id}" + (f" ({date_time})" if date_time else "") + " ==="
        self._cur_lines = [hdr]

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        del session_id
        stamped = f"[{date_time}] {role}: {content}" if date_time else f"{role}: {content}"
        self._cur_lines.append(stamped)

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del session_id, summary
        if self._cur_lines:
            self._sessions.append("\n".join(self._cur_lines))
            self._cur_lines = []

    def _build_rlm(self) -> Any:
        try:
            from rlm import RLM  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Official RLM adapter requires the upstream package: `uv pip install rlms`."
            ) from exc

        model_name = self.model.split("/", 1)[-1] if "/" in self.model else self.model
        b_kwargs: dict[str, Any] = {"model_name": model_name}
        low = self.model.lower()
        if "deepseek" in low:
            b_kwargs["base_url"] = self.base_url or os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
            key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
            if key:
                b_kwargs["api_key"] = key
        else:
            if self.base_url:
                b_kwargs["base_url"] = self.base_url
            if self.api_key:
                b_kwargs["api_key"] = self.api_key

        return RLM(
            backend=self.backend,  # type: ignore[arg-type]
            backend_kwargs=b_kwargs,
            environment="local",
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_concurrent_subcalls=self.max_concurrent_subcalls,
            max_timeout=self.max_timeout,
            verbose=False,
        )

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        t0 = time.perf_counter()
        if self._cur_lines:
            self._sessions.append("\n".join(self._cur_lines))
            self._cur_lines = []
        full_context = "\n\n".join(self._sessions)
        prompt_q = (
            f"{f'[Current date: {date_time}] ' if date_time else ''}"
            f"Answer concisely based on the conversation history stored in `context`. "
            f"If facts changed over time, use the latest dated update. If unmentioned, say you do not know.\n"
            f"Question: {question}"
        )

        def _run_sync() -> Any:
            with _RLM_LOCAL_REPL_LOCK:
                saved_cwd = os.getcwd()
                inst = self._build_rlm()
                try:
                    return inst.completion(prompt=full_context, root_prompt=prompt_q)
                except SystemExit as sexc:
                    raise RuntimeError(f"RLM REPL invoked SystemExit({sexc.code})") from None
                finally:
                    try:
                        os.chdir(saved_cwd)
                    except OSError:
                        pass

        try:
            comp = await asyncio.to_thread(_run_sync)
            answer_txt = str(getattr(comp, "response", "") or "")
            t_in = 0
            t_out = 0
            calls = 0
            usage = getattr(comp, "usage_summary", None)
            if usage is not None:
                for msum in (getattr(usage, "model_usage_summaries", None) or {}).values():
                    t_in += int(getattr(msum, "total_input_tokens", 0) or 0)
                    t_out += int(getattr(msum, "total_output_tokens", 0) or 0)
                    calls += int(getattr(msum, "total_calls", 0) or 0)
            peak_tok = t_in // max(1, calls) if calls else t_in
            return AdapterAnswer(
                answer=answer_txt,
                tokens_in=t_in,
                tokens_out=t_out,
                cache_read_tokens=0,
                peak_context_tokens=peak_tok,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                metadata={"rlm_calls": calls},
            )
        except (Exception, SystemExit) as exc:
            return AdapterAnswer(
                answer=f"I do not know (RLM error: {type(exc).__name__}).",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                metadata={"error": str(exc)},
            )

    async def close(self) -> None:
        self._sessions.clear()
        self._cur_lines.clear()
