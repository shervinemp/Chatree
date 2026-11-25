from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from chatree.config import AblationConfig
from chatree.engine.guardrails import TraceContext
from chatree.engine.rag import TreeRAGService
from chatree.engine.staleness import enrich_date_with_weekday
from chatree.protocols import LLMProvider, ModelRouter
from chatree.storage.repository import SQLiteRepository
from chatree.types import NodeModel, SearchScope, TreeModel

_CURRENT_DATE_PREFIX_RE = re.compile(r"^(\[Current date:\s*)([^\]]+)(\]\s*)", flags=re.IGNORECASE)
_BETWEEN_COMPARE_RE = re.compile(
    r"\b(?:between|compare|comparing|both)\s+(.{3,90}?)\s+and\s+(.{3,90}?)(?:[?.!,]|$)",
    flags=re.IGNORECASE,
)
_FROM_TO_COMPARE_RE = re.compile(
    r"\bfrom\s+(?:when\s+)?(.{4,80}?)\s+to\s+(?:when\s+)?(.{4,80}?)(?:[?.!,]|$)",
    flags=re.IGNORECASE,
)
_COUNTING_KWS = (
    "how many",
    "how much",
    "total",
    "all ",
    "every ",
    "list ",
    "count",
    "times did",
    "items",
    "altogether",
)
_TEMPORAL_COMPARE_KWS = (
    "between ",
    "how many days",
    "how many weeks",
    "how many months",
    "how many years",
    "how long",
    "before ",
    "after ",
    "since ",
    "earliest",
    "first ",
    "order ",
    "when did",
    "how often",
    "compare",
    "both ",
)
_UPDATE_KWS = (
    "currently",
    "current ",
    "now",
    "latest",
    "most recent",
    "recent ",
    "relocation",
    "moved",
    "switched",
    "changed",
    "updated",
    "still ",
    "anymore",
)
_PREF_OPEN_KWS = (
    "recommend",
    "suggest",
    "would ",
    "likely",
    "prefer",
    "might ",
    "what kind",
    "what fields",
)


def classify_query_intent(query: str) -> dict[str, bool]:
    """Classify query intent to adapt retrieval fan-out, fact depth, and recency ordering."""
    q_low = (query or "").lower()
    return {
        "is_counting": any(kw in q_low for kw in _COUNTING_KWS),
        "is_temporal_compare": any(kw in q_low for kw in _TEMPORAL_COMPARE_KWS),
        "is_update": any(kw in q_low for kw in _UPDATE_KWS),
        "is_pref_open": any(kw in q_low for kw in _PREF_OPEN_KWS),
    }


def extract_comparison_subqueries(query: str) -> list[str]:
    """
    Deterministically decompose explicit two-event comparison queries (`between X and Y`,
    `compare X and Y`, `both X and Y`, `from X to Y`) into separate sub-queries so event X
    cannot starve event Y in single-vector top-k retrieval. Avoids splitting casual phrases.
    """
    cleaned = _CURRENT_DATE_PREFIX_RE.sub("", (query or "").strip()).strip()
    if not cleaned:
        return []
    for pat in (_BETWEEN_COMPARE_RE, _FROM_TO_COMPARE_RE):
        m = pat.search(cleaned)
        if not m:
            continue
        left = m.group(1).strip(" \"'.,;:?!")
        right = m.group(2).strip(" \"'.,;:?!")
        # Avoid trivial 1-word conjunctions like "research and development"
        if len(left.split()) >= 2 and len(right.split()) >= 2:
            return [left, right]
        if (len(left) >= 5 and len(right) >= 5) and ("between" in pat.pattern or "compare" in cleaned.lower()):
            return [left, right]
    return []


def _enrich_query_current_date(query: str) -> str:
    """Enrich leading `[Current date: ...]` header with exact calendar day-of-week."""
    m = _CURRENT_DATE_PREFIX_RE.match(query or "")
    if not m:
        return query
    enriched = enrich_date_with_weekday(m.group(2))
    return f"{m.group(1)}{enriched}{m.group(3)}{query[m.end():]}"


def parse_escalation_directive(
    raw_text: str, valid_unqueried: set[str]
) -> tuple[bool, list[str], str]:
    """
    Parse fused Answer-or-Escalate output. Returns `(should_escalate, target_ids, followup_query)`.
    Supports both lightweight line directives (`NEED_MORE_BRANCHES: [...]` / `FOLLOWUP_QUERY: ...`)
    and legacy JSON (`{"need_more": true, "target_ids": [...], "followup_query": "..."}`).
    """
    txt = (raw_text or "").strip()
    if not txt:
        return False, [], ""

    # 1) Lightweight directive format (only triggers if explicitly emitted)
    need_m = re.search(r"^\s*NEED_MORE_BRANCHES\s*:\s*(\[[^\]]*\])", txt, flags=re.IGNORECASE | re.MULTILINE)
    fq_m = re.search(r"^\s*FOLLOWUP_QUERY\s*:\s*(.+)$", txt, flags=re.IGNORECASE | re.MULTILINE)
    if need_m or fq_m:
        chosen: list[str] = []
        if need_m:
            try:
                parsed = json.loads(need_m.group(1))
                if isinstance(parsed, list):
                    chosen = [str(t) for t in parsed if str(t) in valid_unqueried][:4]
            except Exception:
                # Regex salvage for unquoted node IDs like [n_abc123, n_def456]
                chosen = [
                    tid
                    for tid in re.findall(r"\bn_[A-Za-z0-9_-]+\b", need_m.group(1))
                    if tid in valid_unqueried
                ][:4]
        fq = fq_m.group(1).strip().strip("\"'") if fq_m else ""
        if chosen or fq:
            return True, chosen, fq
        return False, [], ""

    # 2) JSON fallback if the model replies with a JSON escalation object
    if txt.startswith("{") or '"need_more"' in txt[:120]:
        m_json = re.search(r"\{.*\}", txt, re.DOTALL)
        if m_json:
            try:
                plan = json.loads(m_json.group(0))
                if isinstance(plan, dict) and plan.get("need_more"):
                    raw_tids = plan.get("target_ids") or []
                    chosen = [str(t) for t in raw_tids if str(t) in valid_unqueried][:4]
                    fq = str(plan.get("followup_query") or "").strip()
                    if chosen or fq:
                        return True, chosen, fq
            except Exception:
                pass
    return False, [], ""


class TreeQueryEngine:
    """
    Unified multi-hop tree retrieval & chronological synthesis engine shared by BOTH:
      1. Live production callers (`tree.ask(..., deep=True)` & `dialogue_tree_search_and_query(deep=True)`), and
      2. External publishable benchmarks (`ChatreeSystemAdapter`).

    Execution pipeline:
      1. Intent & Date Anchoring: weekday-enriched dates, counting/comparison detection, sub-query splitting
      2. Hop 1 + Tier-2.5 Facts: parallel read-only shadow Q&A (adaptive targets for counting) + dated atomic facts
      3. Hybrid Branch Summary Map: full chronological map for <=25 branches, or top-16 RAG-relevant full summaries
         + 1-line title/date directory for remaining branches at larger scales
      4. Fused Answer-or-Escalate Synthesis: answers in 1 LLM call when Hop-1 + facts suffice, or escalates to
         targeted Hop-2 parallel shadows followed by final synthesis when missing evidence is detected.
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

    async def _build_hybrid_summary_map(
        self,
        *,
        tree_id: str,
        caller: NodeModel,
        query: str,
        scope: SearchScope,
        ordered_nids: list[str],
        queried_ids: set[str],
        titles: dict[str, str],
        dates: dict[str, str],
        summaries: dict[str, str],
        is_update: bool,
        full_threshold: int = 64,
        top_unqueried: int = 24,
    ) -> str:
        """
        Build a chronological branch summary map:
          - For <= `full_threshold` branches: emit full summaries (zero recall loss).
          - Above `full_threshold`: emit full summaries for queried + top-`top_unqueried`
            RAG-relevant unqueried branches (with recency boost for update queries), plus a
            compact 1-line `id | title | date` directory for all remaining branches so none
            are hidden even at 500+ sessions.
        """
        summarized_nids = [nid for nid in ordered_nids if summaries.get(nid)]
        if not summarized_nids:
            return ""

        if len(summarized_nids) <= full_threshold:
            return "\n".join(
                f"- {nid} ({titles.get(nid, nid)}"
                + (f" | {dates[nid]}" if dates.get(nid) else "")
                + f"): {summaries[nid][:600]}"
                for nid in summarized_nids
            )

        # Rank unqueried branches by local 0-LLM-cost RAG similarity (with newest-first tie-break for updates)
        unqueried_nids = [nid for nid in summarized_nids if nid not in queried_ids]
        rag_hits = await self.rag.search(
            tree_id,
            query,
            scope=scope,
            current_node=caller,
            top_k=max(24, top_unqueried * 2),
            granularities={"summary", "fact", "chunk"},
            max_per_node=1,
            use_reranker=False,
        )
        rag_rank = {h.node_id: idx for idx, h in enumerate(rag_hits)}
        selected_unqueried: list[str] = []
        for h in rag_hits:
            if h.node_id in unqueried_nids and h.node_id not in selected_unqueried:
                selected_unqueried.append(h.node_id)
            if len(selected_unqueried) >= top_unqueried:
                break
        if is_update and len(selected_unqueried) < top_unqueried:
            # Always include the most recent sessions for knowledge-update queries
            for nid in reversed(unqueried_nids):
                if nid not in selected_unqueried:
                    selected_unqueried.append(nid)
                if len(selected_unqueried) >= top_unqueried:
                    break
        elif is_update and unqueried_nids:
            # Ensure top-4 newest unqueried sessions are never crowded out on update queries
            newest_tail = [nid for nid in reversed(unqueried_nids) if nid not in selected_unqueried][:4]
            if newest_tail:
                selected_unqueried = selected_unqueried[: max(4, top_unqueried - len(newest_tail))] + newest_tail

        detailed_set = set(queried_ids) | set(selected_unqueried)
        # Preserve chronological ordering across detailed lines
        detailed_lines = [
            f"- {nid} ({titles.get(nid, nid)}"
            + (f" | {dates[nid]}" if dates.get(nid) else "")
            + f"): {summaries[nid][:520]}"
            for nid in summarized_nids
            if nid in detailed_set
        ]
        other_lines = [
            f"- {nid} | {titles.get(nid, nid)}"
            + (f" | {dates[nid]}" if dates.get(nid) else "")
            for nid in summarized_nids
            if nid not in detailed_set
        ]
        # Keep deterministic rank hint unused-free
        _ = rag_rank
        sections = ["\n".join(detailed_lines)]
        if other_lines:
            sections.append(
                "[Other Branches Directory (title & date only — escalate ID if relevant)]\n"
                + "\n".join(other_lines)
            )
        return "\n\n".join(sections)

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

        enriched_query = _enrich_query_current_date(query)
        intent = classify_query_intent(enriched_query)
        sub_queries = extract_comparison_subqueries(enriched_query)

        # Dynamically expand Hop-1 fan-out for counting/aggregation queries so 5-8 session
        # enumerations are not truncated at 4 branches.
        target_cap = max_targets
        if intent["is_counting"] and n_branches > max_targets:
            target_cap = min(max(max_targets, 6), 8)
        eff_targets = min(target_cap, max(1, n_branches)) if n_branches else target_cap

        # Build metadata maps directly from persisted nodes, enriching dates with weekdays
        raw_dates = dict(node_dates or {})
        order = dict(node_order or {})
        titles = dict(node_titles or {})
        summaries = dict(node_summaries or {})
        dates: dict[str, str] = {}
        sort_dates: dict[str, str] = {}
        for idx, n in enumerate(active_branches):
            titles.setdefault(n.id, n.title)
            order.setdefault(n.id, idx)
            if n.summary and n.id not in summaries:
                summaries[n.id] = n.summary
            raw_d = raw_dates.get(n.id)
            if not raw_d:
                m_dt = re.search(r"Date:\s*(.+)$", n.goal or "")
                raw_d = m_dt.group(1).strip() if m_dt else n.created_at[:10]
            sort_dates[n.id] = raw_d
            dates[n.id] = enrich_date_with_weekday(raw_d) if raw_d else ""

        # Hop 1: Parallel read-only shadow Q&A across top matching branches (with sub-query union)
        hop1_exec = await self.ipc.ask_tree(
            tree=tree,
            caller=caller,
            query=enriched_query,
            scope=scope,
            max_targets=eff_targets,
            trace_ctx=trace_ctx,
            extra_queries=sub_queries or None,
        )
        ask_out = hop1_exec.tool_output
        queried_ids: set[str] = set(ask_out.get("targets_queried") or [])
        found = self._extract_found(ask_out)

        ordered_nids = sorted(
            [n.id for n in active_branches],
            key=lambda nid: (sort_dates.get(nid, ""), order.get(nid, 0)),
        )

        # Tier-2.5 zero-LLM-cost atomic dated facts retrieval (expanded for counting queries)
        retrieved_facts_text = ""
        facts_used = 0
        if self.ablations.enable_atomic_facts and n_branches > 0:
            fact_k = 18 if intent["is_counting"] else 10
            fact_hits = await self.rag.search(
                tree.id,
                enriched_query,
                scope=scope,
                current_node=caller,
                top_k=fact_k,
                granularities={"fact"},
            )
            if sub_queries:
                seen_fids = {(h.node_id, h.snippet) for h in fact_hits}
                for sq in sub_queries[:2]:
                    for sh in await self.rag.search(
                        tree.id,
                        sq,
                        scope=scope,
                        current_node=caller,
                        top_k=6,
                        granularities={"fact"},
                    ):
                        key = (sh.node_id, sh.snippet)
                        if key not in seen_fids:
                            seen_fids.add(key)
                            fact_hits.append(sh)
            if fact_hits:
                fact_hits_sorted = sorted(
                    fact_hits,
                    key=lambda h: (sort_dates.get(h.node_id, ""), order.get(h.node_id, 0)),
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

        summary_map_text = ""
        if include_branch_summary_map and not is_mock:
            summary_map_text = await self._build_hybrid_summary_map(
                tree_id=tree.id,
                caller=caller,
                query=enriched_query,
                scope=scope,
                ordered_nids=ordered_nids,
                queried_ids=queried_ids,
                titles=titles,
                dates=dates,
                summaries=summaries,
                is_update=intent["is_update"],
            )

        def _format_evidence(items: list[dict[str, Any]]) -> str:
            sorted_items = sorted(
                items,
                key=lambda a: (
                    sort_dates.get(str(a.get("target_id") or ""), ""),
                    order.get(str(a.get("target_id") or ""), 0),
                ),
            )
            return "\n\n".join(
                f"[Branch: {a.get('target_title', a.get('target_id'))}"
                + (
                    f" | Date: {dates[str(a.get('target_id'))]}"
                    if dates.get(str(a.get("target_id") or ""))
                    else ""
                )
                + f"]\n{a.get('answer') or a.get('summary') or ''}"
                for a in sorted_items
            ) or "(No direct verbatim shadow hits — verify against atomic facts & Branch Summary Map; abstain if unmentioned.)"

        base_synthesis_rules = (
            "Synthesize one direct, concise, accurate answer from the chronologically "
            "ordered verified branch shadow answers (highest authority), provenance-tagged "
            "atomic dated facts, and branch overview (oldest to newest). "
            "- If facts changed over time, ALWAYS use the latest dated update. "
            "- Use resolved calendar dates AND exact day-of-week anchors in session headers to compute exact dates/durations. "
            "- For counting/aggregation questions, carefully combine ALL distinct items across all branches without double-counting. "
            "- Verify exact speaker/subject names strictly — never attribute one person's job, event, or action to another speaker. "
            "- For recommendation, preference, or hypothetical/inference questions ('Would X likely...', "
            "'What fields...', 'What kind of...'), NEVER reply 'I do not know' — always commit to a "
            "concrete grounded inference grounded in branch clues. "
            "- For concrete factual questions whose asked person, entity, or event NEVER occurred "
            "in any branch (e.g. wrong speaker or fabricated event), verify names strictly and either "
            "state you do not know or explicitly correct the false premise."
        )

        # Dedicated Hop-2 Planner ONLY when needed (low Hop-1 hits, counting, temporal comparison,
        # knowledge-update, or preference/inference) — skips the extra planner LLM call on clean
        # single-hop queries where Hop-1 already found strong evidence (>= 2 hits).
        needs_hop2_planner = (
            enable_recursive_hop
            and not is_mock
            and n_branches > len(queried_ids)
            and (
                len(found) < 2
                or intent["is_counting"]
                or intent["is_temporal_compare"]
                or intent["is_update"]
                or intent["is_pref_open"]
            )
        )
        if needs_hop2_planner:
            hop1_preview = "\n".join(
                f"- {a.get('target_id')} ({a.get('target_title', a.get('target_id'))}): "
                f"{str(a.get('answer') or '')[:220]}"
                for a in found
            ) or "(none found in initial branches)"
            unqueried_nids_for_plan = [nid for nid in ordered_nids if nid not in queried_ids]
            unqueried_map = await self._build_hybrid_summary_map(
                tree_id=tree.id,
                caller=caller,
                query=enriched_query,
                scope=scope,
                ordered_nids=unqueried_nids_for_plan,
                queried_ids=set(),
                titles=titles,
                dates=dates,
                summaries=summaries,
                is_update=intent["is_update"],
            )
            plan_resp = await self.llm.complete(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the root coordinator of a dialogue tree deciding whether to run a "
                            "second parallel read-only shadow drill-down on unqueried child branches. "
                            "Check the Unqueried Branch Summary Map carefully for missing items (counts), "
                            "a second compared event/date, user preferences/background clues, or a newer update. "
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
                            f"Question: {enriched_query}\n\n"
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
            valid_unqueried = set(ordered_nids) - queried_ids
            should_escalate, chosen_tids, followup_q = parse_escalation_directive(
                plan_resp.content, valid_unqueried
            )
            if should_escalate:
                seen_tids = {str(a.get("target_id")) for a in found}
                if chosen_tids:
                    hop2_res = await asyncio.gather(
                        *(
                            self.ipc.request(
                                tree=tree,
                                caller=caller,
                                target_id=tid,
                                prompt=enriched_query,
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
                elif followup_q:
                    hop2_exec = await self.ipc.ask_tree(
                        tree=tree,
                        caller=caller,
                        query=followup_q,
                        scope=scope,
                        max_targets=min(4, max(1, n_branches - len(queried_ids))),
                        exclude_node_ids=queried_ids,
                        shadow_prompt=enriched_query,
                        trace_ctx=trace_ctx,
                    )
                    hop2_out = hop2_exec.tool_output
                    queried_ids.update(hop2_out.get("targets_queried") or [])
                    for a in self._extract_found(hop2_out):
                        tid = str(a.get("target_id") or "")
                        if tid and tid not in seen_tids:
                            seen_tids.add(tid)
                            found.append(a)

        synth_tokens_in = 0
        has_fallback_evidence = not is_mock and bool(summary_map_text or retrieved_facts_text)
        if not found and not has_fallback_evidence:
            final_text = "I do not know — this information was not mentioned in any prior session."
        elif len(found) == 1 and (is_mock or not concise_synthesis):
            final_text = str(found[0].get("answer") or found[0].get("summary") or "")
        else:
            evidence = _format_evidence(found)
            facts_section = (
                f"\n\n[Retrieved Atomic Dated Facts (Tier-2.5, Provenance-Tagged)]\n{retrieved_facts_text}"
                if (retrieved_facts_text and not is_mock)
                else ""
            )
            summary_section = (
                f"\n\n[Branch Summary Map (Chronological Overview)]\n{summary_map_text[:16000]}"
                if (summary_map_text and not is_mock)
                else ""
            )
            synth = await self.llm.complete(
                model=model,
                messages=[
                    {"role": "system", "content": base_synthesis_rules},
                    {
                        "role": "user",
                        "content": (
                            f"[Verified Shadow Evidence]\n{evidence}"
                            f"{facts_section}{summary_section}\n\n"
                            f"Question: {enriched_query}"
                        ),
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

        found_sorted = sorted(
            found,
            key=lambda a: (
                sort_dates.get(str(a.get("target_id") or ""), ""),
                order.get(str(a.get("target_id") or ""), 0),
            ),
        )
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
