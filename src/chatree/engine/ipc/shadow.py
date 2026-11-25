from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any

from chatree.engine.guardrails import TraceContext
from chatree.engine.ipc.types import IpcExecutionResult
from chatree.engine.rag import extract_key_identifiers
from chatree.providers.mock import estimate_messages_tokens
from chatree.storage.repository import cosine_similarity
from chatree.tools import (
    CORE_TOOL_NAMES,
    SHADOW_READ_ONLY_CORE_TOOLS,
)
from chatree.types import (
    NodeModel,
    SearchScope,
    TreeModel,
)


class IpcShadowMixin:
    repo: Any
    context: Any
    llm: Any
    router: Any
    reflog: Any
    staleness: Any
    rag: Any
    tools: Any
    hooks: Any
    guardrails: Any
    get_node_cache_age_seconds: Any
    _resolve_model: Any
    mark_node_llm_warm: Any
    execute_tool: Any
    ablations: Any
    sandbox: Any

    # ------------------------------------------------------------------
    # 3. request (cached summary OR read-only shadow turn)
    # ------------------------------------------------------------------

    async def request(
        self,
        tree: TreeModel,
        caller: NodeModel,
        target_id: str,
        prompt: str | None = None,
        mutate: bool = False,
        trace_ctx: TraceContext | None = None,
    ) -> IpcExecutionResult:
        ctx = trace_ctx or TraceContext.start(
            caller.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()

        target = await self.repo.get_node(target_id)
        if target is None or target.tree_id != tree.id:
            return IpcExecutionResult(
                tool_output={
                    "error": "TARGET_NOT_FOUND",
                    "detail": f"Node {target_id} was not found in tree {tree.id}.",
                }
            )

        if mutate and not self.guardrails.allow_mutating_requests:
            return IpcExecutionResult(
                tool_output={
                    "error": "MUTATION_DISABLED",
                    "detail": "GuardrailPolicy has allow_mutating_requests=False.",
                }
            )

        target_arts = await self.repo.list_artifacts_by_node(target.id)
        art_labels = [f"{a.slug}@v{a.version} ({a.title})" for a in target_arts[:10]]

        # Ablation hook: if shadow turns are disabled, degrade prompt requests to summary-only!
        effective_prompt = prompt if self.ablations.enable_shadow_turns else None

        if effective_prompt is None:
            needs_llm = target.summary is None or target.is_dirty
            if needs_llm:
                decision = ctx.check_ipc_call(target.id, costs_hop=False, costs_tokens=True)
                if not decision.allowed:
                    ev = await self.repo.create_ipc_event(
                        tree_id=tree.id,
                        trace_id=ctx.trace_id,
                        hop_depth=ctx.hop_depth,
                        from_node_id=caller.id,
                        to_node_id=target.id,
                        event_type="request_summary",
                        payload={"prompt": None},
                        response={"error": decision.error_code, "detail": decision.detail},
                        status=decision.status,
                    )
                    return IpcExecutionResult(
                        tool_output={"error": decision.error_code, "detail": decision.detail},
                        ipc_event=ev,
                    )

            refreshed_target, regenerated, t_in, t_out, llm_ms = (
                await self.staleness.ensure_fresh_summary(tree, target, trace_ctx=ctx)
            )
            latency_ms = max(llm_ms, int((time.perf_counter() - t0) * 1000))
            out = {
                "target_id": refreshed_target.id,
                "target_title": refreshed_target.title,
                "mode": "refreshed_summary" if regenerated else "cached_summary",
                "cache_hit": not regenerated,
                "summary": refreshed_target.summary or "",
                "summary_version": refreshed_target.summary_version,
                "target_msg_seq": refreshed_target.last_msg_seq,
                "artifacts": art_labels,
                "ablation_shadow_disabled": prompt is not None and not self.ablations.enable_shadow_turns,
            }
            ev = await self.repo.create_ipc_event(
                tree_id=tree.id,
                trace_id=ctx.trace_id,
                hop_depth=ctx.hop_depth,
                from_node_id=caller.id,
                to_node_id=refreshed_target.id,
                event_type="request_summary",
                payload={"prompt": None, "cache_hit": not regenerated},
                response=out,
                status="ok",
                target_summary_version=refreshed_target.summary_version,
                target_msg_seq=refreshed_target.last_msg_seq,
                tokens_in=t_in,
                tokens_out=t_out,
                latency_ms=latency_ms,
            )
            await self.repo.upsert_dependency(
                tree_id=tree.id,
                consumer_node_id=caller.id,
                source_node_id=refreshed_target.id,
                seen_summary_version=refreshed_target.summary_version,
                seen_msg_seq=refreshed_target.last_msg_seq,
                last_ipc_event_id=ev.id,
            )
            return IpcExecutionResult(tool_output=out, ipc_event=ev)

        # Read-only Shadow Turn (or explicit mutate=True)
        decision = ctx.check_ipc_call(target.id, costs_hop=True, costs_tokens=True)
        if not decision.allowed:
            ev = await self.repo.create_ipc_event(
                tree_id=tree.id,
                trace_id=ctx.trace_id,
                hop_depth=ctx.hop_depth,
                from_node_id=caller.id,
                to_node_id=target.id,
                event_type="request_mutate" if mutate else "request_shadow",
                payload={"prompt": effective_prompt, "mutate": mutate},
                response={"error": decision.error_code, "detail": decision.detail},
                status=decision.status,
            )
            return IpcExecutionResult(
                tool_output={"error": decision.error_code, "detail": decision.detail},
                ipc_event=ev,
            )

        next_trace = ctx.descend(target.id)
        live_ctx_tokens = await self.context.estimate_node_context_tokens(target)
        cache_age_s = self.get_node_cache_age_seconds(target)
        shadow_model = self._resolve_model(
            tree,
            target,
            "shadow",
            context_tokens=live_ctx_tokens,
            cache_age_seconds=cache_age_s,
        )

        # Shadow-turn memoization cache: exact O(1) match + optional high-precision semantic match!
        if not mutate and self.ablations.enable_shadow_cache:
            cached = await self.repo.get_cached_shadow_answer(
                target_node_id=target.id,
                target_msg_seq=target.last_msg_seq,
                model=shadow_model,
                prompt=effective_prompt,
            )
            semantic_hit = False
            if (
                cached is None
                and self.guardrails.semantic_shadow_cache_threshold < 1.0
                and self.rag is not None
                and self.rag.embedder is not None
            ):
                candidates = await self.repo.list_cached_shadow_answers_for_node(
                    target_node_id=target.id,
                    target_msg_seq=target.last_msg_seq,
                    model=shadow_model,
                    limit=16,
                )
                if candidates:
                    q_ids = {k.lower() for k in extract_key_identifiers(effective_prompt, limit=12)}
                    compatible = [
                        c
                        for c in candidates
                        if not q_ids
                        or q_ids.issubset(
                            {k.lower() for k in extract_key_identifiers(c["prompt_text"], limit=12)}
                        )
                    ]
                    if compatible:
                        vecs = await self.rag.embedder.embed(
                            [effective_prompt, *(c["prompt_text"] for c in compatible)]
                        )
                        if len(vecs) == len(compatible) + 1:
                            q_vec = vecs[0]
                            best_sim = -1.0
                            best_cand: dict[str, Any] | None = None
                            for cand, c_vec in zip(compatible, vecs[1:]):
                                sim = cosine_similarity(q_vec, c_vec)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_cand = cand
                            if (
                                best_cand is not None
                                and best_sim >= self.guardrails.semantic_shadow_cache_threshold
                            ):
                                new_hits = await self.repo.increment_shadow_cache_hit(
                                    best_cand["id"]
                                )
                                cached = {
                                    "answer": best_cand["answer"],
                                    "summary_version": best_cand["summary_version"],
                                    "hit_count": new_hits,
                                }
                                semantic_hit = True
            if cached is not None:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                cached_found = not str(cached["answer"]).strip().upper().startswith(
                    "NOT_FOUND_IN_BRANCH"
                )
                out = {
                    "target_id": target.id,
                    "target_title": target.title,
                    "mode": "cached_shadow_turn",
                    "cache_hit": True,
                    "semantic_cache_hit": semantic_hit,
                    "found_in_branch": cached_found,
                    "shadow_cache_hits": cached["hit_count"],
                    "answer": cached["answer"],
                    "mutated": False,
                    "summary_version": target.summary_version,
                    "target_msg_seq": target.last_msg_seq,
                    "artifacts": art_labels,
                    "model_used": shadow_model,
                }
                ev = await self.repo.create_ipc_event(
                    tree_id=tree.id,
                    trace_id=ctx.trace_id,
                    hop_depth=next_trace.hop_depth,
                    from_node_id=caller.id,
                    to_node_id=target.id,
                    event_type="request_shadow",
                    payload={"prompt": effective_prompt, "mutate": False, "model": shadow_model, "cache_hit": True},
                    response=out,
                    status="ok",
                    target_summary_version=target.summary_version,
                    target_msg_seq=target.last_msg_seq,
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=latency_ms,
                )
                await self.repo.upsert_dependency(
                    tree_id=tree.id,
                    consumer_node_id=caller.id,
                    source_node_id=target.id,
                    seen_summary_version=target.summary_version,
                    seen_msg_seq=target.last_msg_seq,
                    last_ipc_event_id=ev.id,
                )
                return IpcExecutionResult(tool_output=out, ipc_event=ev)

        shadow_messages = await self.context.assemble_for_shadow_query(
            target_node=target, caller_node=caller, prompt=effective_prompt
        )
        # Preserve identical canonical `tools` array when `preserve_kv_cache_tool_schemas=True`
        # so DeepSeek / OpenAI / Anthropic do NOT invalidate KV cache at byte 0; read-only
        # shadow safety is still enforced deterministically at execution time below.
        shadow_tools = (
            self.tools.schemas_for_node(
                target.tool_permission,
                include_core=True,
                include_compound_ask=self.guardrails.enable_compound_ask_tool,
                is_shadow=True,
                preserve_kv_cache_schemas=self.guardrails.preserve_kv_cache_tool_schemas,
            )
            if next_trace.hop_depth <= next_trace.max_hop_depth
            else None
        )

        total_in = 0
        total_out = 0
        total_cache_read = 0
        final_answer = ""
        nested_guardrail_error: dict[str, Any] | None = None

        fork_bp = await self.context.get_fork_breakpoint_index(target)
        complete_params = inspect.signature(self.llm.complete).parameters
        for _round in range(3):
            est_prompt = max(self.guardrails.min_turn_reserve, estimate_messages_tokens(shadow_messages))
            reserved = ctx.ledger.try_reserve(
                est_prompt, min_reserve=self.guardrails.min_turn_reserve
            )
            if reserved is None:
                ev = await self.repo.create_ipc_event(
                    tree_id=tree.id,
                    trace_id=ctx.trace_id,
                    hop_depth=next_trace.hop_depth,
                    from_node_id=caller.id,
                    to_node_id=target.id,
                    event_type="request_mutate" if mutate else "request_shadow",
                    payload={"prompt": effective_prompt, "mutate": mutate, "model": shadow_model},
                    response={
                        "error": "TOKEN_BUDGET_EXCEEDED",
                        "detail": (
                            f"Trace token budget exhausted before shadow turn "
                            f"({ctx.ledger.tokens_used}+{ctx.ledger.tokens_reserved}/"
                            f"{ctx.ledger.token_budget}, needed~{est_prompt})."
                        ),
                    },
                    status="rejected_budget",
                )
                return IpcExecutionResult(
                    tool_output={
                        "error": "TOKEN_BUDGET_EXCEEDED",
                        "detail": ev.response["detail"],
                    },
                    ipc_event=ev,
                )
            try:
                extra_kw: dict[str, Any] = {}
                if not mutate and self.guardrails.shadow_max_tokens is not None and "max_tokens" in complete_params:
                    extra_kw["max_tokens"] = self.guardrails.shadow_max_tokens
                if fork_bp is not None and "fork_breakpoint_index" in complete_params:
                    extra_kw["fork_breakpoint_index"] = fork_bp
                resp = await self.llm.complete(
                    model=shadow_model,
                    messages=shadow_messages,
                    tools=shadow_tools,
                    temperature=0.1,
                    **extra_kw,
                )
            except Exception:
                ctx.ledger.release(reserved)
                raise
            self.mark_node_llm_warm(target.id)
            ctx.ledger.settle(
                reserved,
                resp.tokens_in,
                resp.tokens_out,
                cache_read_tokens=resp.cache_read_tokens,
                cache_read_weight=self.guardrails.cache_read_budget_weight,
            )
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            total_cache_read += resp.cache_read_tokens

            if not resp.tool_calls:
                final_answer = resp.content
                break

            shadow_messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [tc.to_openai_dict() for tc in resp.tool_calls],
                }
            )
            for tc in resp.tool_calls:
                if tc.name in CORE_TOOL_NAMES and tc.name not in SHADOW_READ_ONLY_CORE_TOOLS:
                    tool_res: dict[str, Any] = {
                        "error": "READ_ONLY_SHADOW",
                        "detail": f"Tool '{tc.name}' is disabled during a read-only shadow turn.",
                    }
                else:
                    sub = await self.execute_tool(
                        tree=tree,
                        caller=target,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        trace_ctx=next_trace,
                        is_shadow=True,
                    )
                    tool_res = sub.tool_output
                    if tool_res.get("error"):
                        nested_guardrail_error = tool_res
                shadow_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(tool_res),
                    }
                )
        else:
            final_answer = "(Shadow turn reached maximum internal tool rounds.)"

        await self.repo.increment_node_usage(target.id, total_in, total_out, total_cache_read)

        stale_node_ids: list[str] = []
        if mutate:
            await self.repo.append_message(
                node_id=target.id,
                tree_id=tree.id,
                role="user",
                content=f"[Mutating IPC Query from `{caller.id}`]: {effective_prompt}",
            )
            await self.repo.append_message(
                node_id=target.id,
                tree_id=tree.id,
                role="assistant",
                content=final_answer,
                model_used=shadow_model,
            )
            await self.rag.index_node_chunks(target.id)
            refreshed = await self.repo.get_node(target.id)
            assert refreshed is not None
            target = refreshed
            stale_node_ids = [
                cid
                for cid in await self.staleness.get_affected_consumers(target.id)
                if cid != caller.id
            ]

        if not mutate and self.ablations.enable_shadow_cache and not nested_guardrail_error and final_answer.strip():
            await self.repo.put_cached_shadow_answer(
                tree_id=tree.id,
                target_node_id=target.id,
                target_msg_seq=target.last_msg_seq,
                model=shadow_model,
                prompt=effective_prompt,
                answer=final_answer,
                summary_version=target.summary_version,
                max_entries=self.guardrails.max_shadow_cache_entries,
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        found_in_branch = not final_answer.strip().upper().startswith("NOT_FOUND_IN_BRANCH")
        out = {
            "target_id": target.id,
            "target_title": target.title,
            "mode": "mutating_turn" if mutate else "shadow_turn",
            "cache_hit": False,
            "found_in_branch": found_in_branch,
            "answer": final_answer,
            "mutated": mutate,
            "summary_version": target.summary_version,
            "target_msg_seq": target.last_msg_seq,
            "artifacts": art_labels,
            "model_used": shadow_model,
        }
        if nested_guardrail_error:
            out["nested_guardrail_warning"] = nested_guardrail_error

        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=ctx.trace_id,
            hop_depth=next_trace.hop_depth,
            from_node_id=caller.id,
            to_node_id=target.id,
            event_type="request_mutate" if mutate else "request_shadow",
            payload={"prompt": effective_prompt, "mutate": mutate, "model": shadow_model},
            response=out,
            status="ok",
            target_summary_version=target.summary_version,
            target_msg_seq=target.last_msg_seq,
            tokens_in=total_in,
            tokens_out=total_out,
            latency_ms=latency_ms,
        )
        await self.repo.upsert_dependency(
            tree_id=tree.id,
            consumer_node_id=caller.id,
            source_node_id=target.id,
            seen_summary_version=target.summary_version,
            seen_msg_seq=target.last_msg_seq,
            last_ipc_event_id=ev.id,
        )
        return IpcExecutionResult(
            tool_output=out, ipc_event=ev, stale_node_ids=stale_node_ids
        )

    # ------------------------------------------------------------------
    # 4. search_tree (3-tier Hybrid Tree RAG)
    # ------------------------------------------------------------------

    async def search_tree(
        self,
        tree: TreeModel,
        caller: NodeModel,
        query: str,
        scope: SearchScope = "all",
        top_k: int = 5,
        trace_ctx: TraceContext | None = None,
    ) -> IpcExecutionResult:
        ctx = trace_ctx or TraceContext.start(
            caller.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()
        if scope not in ("all", "ancestors", "subtree", "siblings"):
            scope = "all"
        hits = await self.rag.search(
            tree_id=tree.id,
            query=query,
            scope=scope,
            current_node=caller,
            top_k=max(1, min(top_k, 20)),
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        out = {
            "query": query,
            "scope": scope,
            "count": len(hits),
            "hits": [h.model_dump() for h in hits],
        }
        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=ctx.trace_id,
            hop_depth=ctx.hop_depth,
            from_node_id=caller.id,
            to_node_id=None,
            event_type="search",
            payload={"query": query, "scope": scope, "top_k": top_k},
            response={"count": len(hits), "node_ids": [h.node_id for h in hits]},
            status="ok",
            latency_ms=latency_ms,
        )
        return IpcExecutionResult(tool_output=out, ipc_event=ev)

    # ------------------------------------------------------------------
    # 4b. ask_tree (Single-Hop Compound Retrieve-then-Ask)
    # ------------------------------------------------------------------

    async def ask_tree(
        self,
        tree: TreeModel,
        caller: NodeModel,
        query: str,
        scope: SearchScope = "all",
        max_targets: int = 2,
        trace_ctx: TraceContext | None = None,
        exclude_node_ids: set[str] | None = None,
        shadow_prompt: str | None = None,
        extra_queries: list[str] | None = None,
    ) -> IpcExecutionResult:
        """
        Fuses `search_tree` + parallel `asyncio.gather` shadow `request()` calls into one
        single round trip, avoiding the serial `search_tree -> parent LLM -> request` latency tax.
        Optional `extra_queries` (e.g., decomposed multi-event sub-queries) are interleaved
        round-robin with the primary query so one entity never starves another in top-k selection.
        """
        ctx = trace_ctx or TraceContext.start(
            caller.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()
        capped_targets = max(1, min(max_targets, 10))
        excluded = set(exclude_node_ids or ())
        excluded.add(caller.id)
        per_q_k = max(12, (capped_targets + len(excluded)) * 2)
        primary_hits = await self.rag.search(
            tree_id=tree.id,
            query=query,
            scope=scope,
            current_node=caller,
            top_k=per_q_k,
            max_per_node=2,
        )
        ranked_lists: list[list[str]] = [
            list(dict.fromkeys(h.node_id for h in primary_hits if h.node_id not in excluded))
        ]
        for sub_q in (extra_queries or [])[:3]:
            sq = sub_q.strip()
            if not sq or sq.lower() == query.strip().lower():
                continue
            sub_hits = await self.rag.search(
                tree_id=tree.id,
                query=sq,
                scope=scope,
                current_node=caller,
                top_k=max(8, capped_targets + len(excluded)),
                max_per_node=2,
            )
            ranked_lists.append(
                list(dict.fromkeys(h.node_id for h in sub_hits if h.node_id not in excluded))
            )

        # Pick top distinct target nodes via round-robin interleaving across primary + sub-queries,
        # with causal freshness filtering: if candidate A has a stale dependency on candidate B,
        # suppress stale A so authoritative revised branches take precedence over outdated consumers.
        raw_candidates: list[str] = []
        max_len = max((len(lst) for lst in ranked_lists), default=0)
        for idx in range(max_len):
            for lst in ranked_lists:
                if idx < len(lst):
                    nid = lst[idx]
                    if nid not in raw_candidates:
                        raw_candidates.append(nid)

        stale_map = await self.repo.compute_stale_dependencies_map(
            tree.id,
            transitive=self.ablations.enable_transitive_staleness,
            max_hops=self.guardrails.max_staleness_hops,
        )
        candidate_set = set(raw_candidates)
        suppressed_stale: set[str] = set()
        promoted_sources: list[str] = []
        for cid in raw_candidates:
            for dep in stale_map.get(cid, []):
                if dep.source_node_id in candidate_set and dep.source_node_id != cid:
                    suppressed_stale.add(cid)
                    if dep.source_node_id not in promoted_sources:
                        promoted_sources.append(dep.source_node_id)

        ordered_candidates = [
            *promoted_sources,
            *(cid for cid in raw_candidates if cid not in suppressed_stale and cid not in promoted_sources),
        ]
        target_ids = ordered_candidates[:capped_targets]

        if not target_ids:
            out: dict[str, Any] = {
                "query": query,
                "scope": scope,
                "targets_queried": [],
                "answers": [],
            }
            ev = await self.repo.create_ipc_event(
                tree_id=tree.id,
                trace_id=ctx.trace_id,
                hop_depth=ctx.hop_depth,
                from_node_id=caller.id,
                to_node_id=None,
                event_type="search",
                payload={"compound_ask": True, "query": query, "scope": scope},
                response=out,
                status="ok",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            return IpcExecutionResult(tool_output=out, ipc_event=ev)

        eff_shadow_prompt = shadow_prompt or query
        if self.ablations.enable_parallel_ipc and len(target_ids) > 1:
            sub_results = await asyncio.gather(
                *(
                    self.request(
                        tree=tree,
                        caller=caller,
                        target_id=tid,
                        prompt=eff_shadow_prompt,
                        mutate=False,
                        trace_ctx=ctx,
                    )
                    for tid in target_ids
                )
            )
        else:
            sub_results = [
                await self.request(
                    tree=tree,
                    caller=caller,
                    target_id=tid,
                    prompt=eff_shadow_prompt,
                    mutate=False,
                    trace_ctx=ctx,
                )
                for tid in target_ids
            ]

        answers = [r.tool_output for r in sub_results]
        found_answers = [
            a
            for a in answers
            if a.get("found_in_branch", True)
            and not str(a.get("answer") or "").strip().upper().startswith("NOT_FOUND_IN_BRANCH")
        ]
        out = {
            "query": query,
            "scope": scope,
            "targets_queried": target_ids,
            "answers": answers,
            "found_answers": found_answers,
        }
        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=ctx.trace_id,
            hop_depth=ctx.hop_depth,
            from_node_id=caller.id,
            to_node_id=target_ids[0] if len(target_ids) == 1 else None,
            event_type="search",
            payload={"compound_ask": True, "query": query, "scope": scope, "targets": target_ids},
            response={"targets_queried": target_ids, "answer_count": len(answers)},
            status="ok",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        return IpcExecutionResult(tool_output=out, ipc_event=ev)

