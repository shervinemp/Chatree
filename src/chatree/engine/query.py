from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from chatree.config import AblationConfig
from chatree.engine.guardrails import TraceContext
from chatree.engine.rag import TreeRAGService
from chatree.protocols import LLMProvider, ModelRouter
from chatree.storage.repository import SQLiteRepository
from chatree.types import NodeModel, SearchScope, TreeModel


class TreeQueryEngine:
    """
    Unified multi-hop tree retrieval & chronological synthesis engine shared by BOTH:
      1. Live production callers (`tree.ask(..., deep=True)` & `dialogue_tree_search_and_query(deep=True)`), and
      2. External publishable benchmarks (`ChatreeSystemAdapter`).

    Eliminates benchmark-vs-runtime divergence by executing one single code path:
      Hop 1: Parallel read-only shadow Q&A over top RAG-retrieved branches (`ipc.ask_tree`)
      Tier-2.5: Zero-LLM-cost provenance-tagged atomic dated facts retrieval (`rag.search(fact)`)
      Hop 2 (optional): Summary-map + atomic-facts guided shadow drill-down on unqueried branches
      Synthesis: Chronologically ordered evidence fusion (verbatim shadow answers > dated facts > summaries)
    """

    def __init__(
        self,
        repo: SQLiteRepository,
        ipc: Any,
        rag: TreeRAGService,
        llm: LLMProvider,
        router: ModelRouter,
        *,
        ablations: AblationConfig | None = None,
    ) -> None:
        self.repo = repo
        self.ipc = ipc
        self.rag = rag
        self.llm = llm
        self.router = router
        self.ablations = ablations or AblationConfig()

    @staticmethod
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

    async def deep_ask(
        self,
        tree: TreeModel,
        caller: NodeModel,
        query: str,
        *,
        scope: SearchScope = "all",
        max_targets: int = 4,
        enable_recursive_hop: bool = True,
        include_branch_summary_map: bool = True,
        concise_synthesis: bool = True,
        node_dates: dict[str, str] | None = None,
        node_order: dict[str, int] | None = None,
        node_titles: dict[str, str] | None = None,
        node_summaries: dict[str, str] | None = None,
        trace_ctx: TraceContext | None = None,
    ) -> dict[str, Any]:
        all_nodes = await self.repo.list_nodes(tree.id)
        active_branches = [
            n for n in all_nodes if n.id != tree.root_node_id and n.status != "ARCHIVED"
        ]
        n_branches = len(active_branches)
        eff_targets = min(max_targets, max(1, n_branches)) if n_branches else max_targets

        # Build metadata maps directly from persisted nodes when caller doesn't pass overrides
        dates = dict(node_dates or {})
        order = dict(node_order or {})
        titles = dict(node_titles or {})
        summaries = dict(node_summaries or {})
        for idx, n in enumerate(active_branches):
            titles.setdefault(n.id, n.title)
            order.setdefault(n.id, idx)
            if n.summary and n.id not in summaries:
                summaries[n.id] = n.summary
            if n.id not in dates:
                m_dt = re.search(r"Date:\s*(.+)$", n.goal or "")
                dates[n.id] = m_dt.group(1).strip() if m_dt else n.created_at[:10]

        # Hop 1: Parallel read-only shadow Q&A across top matching branches
        hop1_exec = await self.ipc.ask_tree(
            tree=tree,
            caller=caller,
            query=query,
            scope=scope,
            max_targets=eff_targets,
            trace_ctx=trace_ctx,
        )
        ask_out = hop1_exec.tool_output
        queried_ids: set[str] = set(ask_out.get("targets_queried") or [])
        found = self._extract_found(ask_out)

        ordered_nids = sorted(
            [n.id for n in active_branches],
            key=lambda nid: (dates.get(nid, ""), order.get(nid, 0)),
        )
        summary_map_lines = [
            f"- {nid} ({titles.get(nid, nid)}"
            + (f" | {dates[nid]}" if dates.get(nid) else "")
            + f"): {summaries.get(nid, '')[:600]}"
            for nid in ordered_nids
            if summaries.get(nid)
        ]
        summary_map_text = "\n".join(summary_map_lines)

        # Tier-2.5 zero-LLM-cost atomic dated facts retrieval
        retrieved_facts_text = ""
        facts_used = 0
        if self.ablations.enable_atomic_facts and n_branches > 0:
            fact_hits = await self.rag.search(
                tree.id,
                query,
                scope=scope,
                current_node=caller,
                top_k=10,
                granularities={"fact"},
            )
            if fact_hits:
                fact_hits_sorted = sorted(
                    fact_hits,
                    key=lambda h: (dates.get(h.node_id, ""), order.get(h.node_id, 0)),
                )
                fact_lines: list[str] = []
                for fh in fact_hits_sorted:
                    snippet = fh.snippet.strip()
                    if "Fact:" in snippet:
                        snippet = snippet.split("Fact:", 1)[1].strip()
                    sdate = dates.get(fh.node_id, "")
                    date_sfx = f" | Session: {sdate}" if sdate else ""
                    fact_lines.append(
                        f"- [{fh.node_id} | {titles.get(fh.node_id, fh.node_title)}{date_sfx}] {snippet[:260]}"
                    )
                retrieved_facts_text = "\n".join(fact_lines)
                facts_used = len(fact_lines)

        model = self.router.resolve_model(tree, caller, "chat")
        is_mock = model.startswith("mock/")

        # Hop 2: Optional summary-map + atomic-facts guided shadow drill-down
        q_low = query.lower()
        needs_broad = any(
            kw in q_low
            for kw in (
                "how many", "how much", "total", "all ", "recommend", "suggest", "would ",
                "likely", "prefer", "before", "after", "recent", "latest", "now", "currently",
                "how long", "when did", "how often",
            )
        )
        if (
            enable_recursive_hop
            and not is_mock
            and n_branches > len(queried_ids)
            and (len(found) < 2 or needs_broad or include_branch_summary_map)
        ):
            hop1_preview = "\n".join(
                f"- {a.get('target_id')} ({a.get('target_title', a.get('target_id'))}): "
                f"{str(a.get('answer') or '')[:220]}"
                for a in found
            ) or "(none found in initial branches)"
            unqueried_map = "\n".join(
                ln for nid, ln in zip(ordered_nids, summary_map_lines) if nid not in queried_ids
            )
            plan_resp = await self.llm.complete(
                model=model,
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
                            f"Question: {query}\n\n"
                            f"Hop-1 Verified Shadow Evidence:\n{hop1_preview}\n\n"
                            + (
                                f"Retrieved Tier-2.5 Atomic Dated Facts:\n{retrieved_facts_text}\n\n"
                                if retrieved_facts_text
                                else ""
                            )
                            + (
                                f"Unqueried Branch Summary Map:\n{unqueried_map[:16000]}"
                                if include_branch_summary_map and unqueried_map
                                else f"({n_branches - len(queried_ids)} unqueried branches remain)"
                            )
                        ),
                    },
                ],
                tools=None,
                temperature=0.0,
            )
            await self.repo.increment_node_usage(
                caller.id, plan_resp.tokens_in, plan_resp.tokens_out, plan_resp.cache_read_tokens
            )
            m_plan = re.search(r"\{.*\}", plan_resp.content, re.DOTALL)
            if m_plan:
                try:
                    plan = json.loads(m_plan.group(0))
                    if plan.get("need_more"):
                        valid_unqueried = set(ordered_nids) - queried_ids
                        raw_tids = plan.get("target_ids") or []
                        chosen_tids = [str(t) for t in raw_tids if str(t) in valid_unqueried][:4]
                        seen_tids = {str(a.get("target_id")) for a in found}
                        if chosen_tids:
                            hop2_res = await asyncio.gather(
                                *(
                                    self.ipc.request(
                                        tree=tree,
                                        caller=caller,
                                        target_id=tid,
                                        prompt=query,
                                        mutate=False,
                                        trace_ctx=trace_ctx,
                                    )
                                    for tid in chosen_tids
                                )
                            )
                            queried_ids.update(chosen_tids)
                            for exec_r in hop2_res:
                                out_d = exec_r.tool_output
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
                                hop2_exec = await self.ipc.ask_tree(
                                    tree=tree,
                                    caller=caller,
                                    query=fq,
                                    scope=scope,
                                    max_targets=min(4, max(1, n_branches - len(queried_ids))),
                                    exclude_node_ids=queried_ids,
                                    shadow_prompt=query,
                                    trace_ctx=trace_ctx,
                                )
                                hop2_out = hop2_exec.tool_output
                                queried_ids.update(hop2_out.get("targets_queried") or [])
                                for a in self._extract_found(hop2_out):
                                    tid = str(a.get("target_id") or "")
                                    if tid and tid not in seen_tids:
                                        seen_tids.add(tid)
                                        found.append(a)
                except Exception:
                    pass

        found_sorted = sorted(
            found,
            key=lambda a: (
                dates.get(str(a.get("target_id") or ""), ""),
                order.get(str(a.get("target_id") or ""), 0),
            ),
        )

        synth_tokens_in = 0
        has_fallback_evidence = not is_mock and bool(
            (include_branch_summary_map and summary_map_text) or retrieved_facts_text
        )
        if not found_sorted and not has_fallback_evidence:
            final_text = "I do not know — this information was not mentioned in any prior session."
        elif len(found_sorted) == 1 and (is_mock or not concise_synthesis):
            final_text = str(found_sorted[0].get("answer") or found_sorted[0].get("summary") or "")
        else:
            evidence = "\n\n".join(
                f"[Branch: {a.get('target_title', a.get('target_id'))}"
                + (
                    f" | Date: {dates[str(a.get('target_id'))]}"
                    if dates.get(str(a.get("target_id") or ""))
                    else ""
                )
                + f"]\n{a.get('answer') or a.get('summary') or ''}"
                for a in found_sorted
            ) or "(No direct verbatim shadow hits — verify against atomic facts & Branch Summary Map; abstain if unmentioned.)"
            facts_section = (
                f"\n\n[Retrieved Atomic Dated Facts (Tier-2.5, Provenance-Tagged)]\n{retrieved_facts_text}"
                if (retrieved_facts_text and not is_mock)
                else ""
            )
            summary_section = (
                f"\n\n[Branch Summary Map (Chronological Overview)]\n{summary_map_text[:16000]}"
                if (include_branch_summary_map and summary_map_text and not is_mock)
                else ""
            )
            synth = await self.llm.complete(
                model=model,
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
                            "concrete grounded inference grounded in branch clues. "
                            "- For concrete factual questions whose asked person, entity, or event NEVER occurred "
                            "in any branch (e.g. wrong speaker or fabricated event), verify names strictly and either "
                            "state you do not know or explicitly correct the false premise."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"[Verified Shadow Evidence]\n{evidence}{facts_section}{summary_section}\n\nQuestion: {query}",
                    },
                ],
                tools=None,
                temperature=0.0,
            )
            await self.repo.increment_node_usage(
                caller.id, synth.tokens_in, synth.tokens_out, synth.cache_read_tokens
            )
            synth_tokens_in = synth.tokens_in
            final_text = synth.content

        return {
            "query": query,
            "scope": scope,
            "deep": True,
            "answer": final_text,
            "targets_queried": sorted(queried_ids),
            "answers": found_sorted,
            "found_answers": found_sorted,
            "facts_used": facts_used,
            "synth_tokens_in": synth_tokens_in,
        }
