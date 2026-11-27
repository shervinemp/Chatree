from __future__ import annotations

import inspect
import json
from typing import Any, AsyncIterator, Callable

from chatree.config import GuardrailPolicy, resolve_model_context_window
from chatree.engine.context import ContextAssembler, _message_to_openai
from chatree.engine.guardrails import TraceContext
from chatree.engine.ipc import IPCRouter
from chatree.engine.rag import TreeRAGService
from chatree.engine.reflog import ReflogService
from chatree.engine.staleness import StalenessService
from chatree.hooks import HookBus
from chatree.protocols import LLMProvider, ModelRouter
from chatree.providers.mock import estimate_messages_tokens
from chatree.storage.repository import SQLiteRepository
from chatree.tools import TOOL_SEND, ToolRegistry
from chatree.types import (
    DEFAULT_SESSION_ID,
    IpcEventModel,
    LLMResponse,
    MailboxItemModel,
    MessageModel,
    NodeModel,
    TreeModel,
    TurnResult,
)


class TurnOrchestrator:
    """
    Runs a streaming or awaited user turn on a node with:
      - Prefix-preserving context assembly
      - Scoped tool schemas (core IPC + branch-permitted custom tools)
      - Parallel IPC fan-out via `IPCRouter.execute_tool_batch`
      - Automatic Tier-1 chunk indexing on turn completion
      - Lifecycle HookBus emissions
    """

    def __init__(
        self,
        repo: SQLiteRepository,
        context: ContextAssembler,
        llm: LLMProvider,
        router: ModelRouter,
        ipc: IPCRouter,
        reflog: ReflogService,
        staleness: StalenessService,
        rag: TreeRAGService,
        tool_registry: ToolRegistry,
        *,
        hooks: HookBus | None = None,
        guardrails: GuardrailPolicy | None = None,
        on_first_turn_title: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.repo = repo
        self.context = context
        self.llm = llm
        self.router = router
        self.ipc = ipc
        self.reflog = reflog
        self.staleness = staleness
        self.rag = rag
        self.tools = tool_registry
        self.hooks = hooks
        self.guardrails = guardrails or GuardrailPolicy()
        self.on_first_turn_title = on_first_turn_title
        self._last_envelope_sig: dict[tuple[str, str], tuple[str, tuple[tuple[str, str, int, str], ...]]] = {}

    async def run_turn(
        self,
        node_id: str,
        user_content: str,
        *,
        session_id: str = DEFAULT_SESSION_ID,
        max_hop_depth: int | None = None,
        token_budget: int | None = None,
        model_override: str | None = None,
        include_core_tools: bool = True,
    ) -> TurnResult:
        """Non-streaming convenience wrapper collecting the complete `TurnResult`."""
        trace_id = ""
        head_node_id = node_id
        user_msg: MessageModel | None = None
        assistant_msg: MessageModel | None = None
        ipc_events: list[IpcEventModel] = []
        pending_handoffs: list[MailboxItemModel] = []
        notices: list[str] = []
        stale_ids: list[str] = []
        compacted_from_id: str | None = None
        actual_node_id = node_id
        tokens_used = 0
        budget = token_budget or self.guardrails.token_budget
        error: dict[str, Any] | None = None

        async for ev in self.run_turn_stream(
            node_id=node_id,
            user_content=user_content,
            session_id=session_id,
            max_hop_depth=max_hop_depth,
            token_budget=token_budget,
            model_override=model_override,
            include_core_tools=include_core_tools,
        ):
            name = ev["event"]
            data = ev["data"]
            if name == "branch_compacted":
                compacted_from_id = data.get("from_node_id")
                if data.get("to_node_id"):
                    actual_node_id = str(data["to_node_id"])
            elif name == "turn_start":
                trace_id = data["trace_id"]
                if data.get("node_id"):
                    actual_node_id = str(data["node_id"])
                user_msg = MessageModel(**data["user_message"])
                notices = list(data.get("ephemeral_notices") or [])
            elif name == "ipc_activity":
                ipc_events.append(IpcEventModel(**data))
            elif name == "handoff_pending":
                pending_handoffs.append(MailboxItemModel(**data))
            elif name == "staleness_updated":
                stale_ids.extend(data.get("stale_node_ids") or [])
            elif name == "turn_complete":
                head_node_id = data.get("head_node_id", actual_node_id)
                if data.get("assistant_message"):
                    assistant_msg = MessageModel(**data["assistant_message"])
                ts = data.get("trace_summary") or {}
                tokens_used = int(ts.get("tokens_used", 0))
                budget = int(ts.get("token_budget", budget))
            elif name == "error":
                error = data

        return TurnResult(
            trace_id=trace_id,
            node_id=actual_node_id,
            head_node_id=head_node_id,
            session_id=session_id,
            user_message=user_msg,
            assistant_message=assistant_msg,
            content=assistant_msg.content if assistant_msg else "",
            ipc_events=ipc_events,
            pending_handoffs=pending_handoffs,
            ephemeral_notices=notices,
            stale_node_ids=list(dict.fromkeys(stale_ids)),
            compacted_from_node_id=compacted_from_id,
            tokens_used=tokens_used,
            token_budget=budget,
            error=error,
        )

    async def _maybe_auto_compact(
        self,
        tree: TreeModel,
        node: NodeModel,
        trace_ctx: TraceContext,
        *,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> tuple[TreeModel, NodeModel, list[dict[str, Any]]]:
        """Run non-destructive auto-compaction if `node` crossed the configured token threshold."""
        events: list[dict[str, Any]] = []
        if (
            not self.guardrails.auto_compact_enabled
            or node.last_msg_seq < self.guardrails.auto_compact_min_messages
        ):
            return tree, node, events
        active_model = self.router.resolve_model(tree, node, "chat")
        resolved_window = resolve_model_context_window(
            active_model,
            overrides=self.guardrails.model_context_windows,
            configured_max=self.guardrails.max_branch_context_tokens,
        )
        ratio_threshold = int(resolved_window * self.guardrails.auto_compact_trigger_ratio)
        if self.guardrails.auto_compact_threshold_tokens is not None:
            eff_threshold = min(self.guardrails.auto_compact_threshold_tokens, ratio_threshold)
        else:
            eff_threshold = ratio_threshold
        live_tokens = await self.context.estimate_node_context_tokens(node)
        if live_tokens < eff_threshold:
            return tree, node, events

        comp_res = await self.ipc.compact_branch(
            tree=tree,
            source=node,
            keep_recent_turns=self.guardrails.auto_compact_keep_recent_turns,
            reason=f"Auto-compacted at {live_tokens} tokens >= threshold {eff_threshold}",
            switch_head=True,
            session_id=session_id,
            trace_ctx=trace_ctx,
        )
        events.append({"event": "branch_compacted", "data": comp_res.tool_output})
        if comp_res.ipc_event is not None:
            events.append({"event": "ipc_activity", "data": comp_res.ipc_event.model_dump()})
        if comp_res.head_changed is not None:
            f_id, t_id, rf_ent = comp_res.head_changed
            events.append(
                {
                    "event": "head_changed",
                    "data": {
                        "from_node_id": f_id,
                        "to_node_id": t_id,
                        "reflog_entry": rf_ent.model_dump(),
                    },
                }
            )
        new_id = str(comp_res.tool_output["to_node_id"])
        new_node = await self.repo.get_node(new_id)
        new_tree = await self.repo.get_tree(tree.id)
        if new_node is not None and new_tree is not None:
            return new_tree, new_node, events
        return tree, node, events

    async def _collect_preturn_hints(
        self,
        tree: TreeModel,
        refreshed_node: NodeModel,
        user_content: str,
    ) -> tuple[list[Any], list[Any]]:
        """Run fast local (<5ms, 0 LLM tokens) speculative pointer & Tier-2.5 atomic fact lookup."""
        speculative_hits: list[Any] = []
        fact_hits: list[Any] = []
        if not self.rag.cfg.enabled or len(await self.repo.list_nodes(tree.id)) <= 1:
            return speculative_hits, fact_hits

        if (
            self.context.ablations.enable_speculative_pointers
            and self.rag.cfg.speculative_pointer_top_k > 0
        ):
            raw_hits = await self.rag.search(
                tree_id=tree.id,
                query=user_content,
                scope="all",
                current_node=refreshed_node,
                top_k=self.rag.cfg.speculative_pointer_top_k + 2,
                use_reranker=False,
            )
            speculative_hits = [
                h
                for h in raw_hits
                if h.node_id != refreshed_node.id
                and h.score >= self.rag.cfg.speculative_pointer_min_score
            ][: self.rag.cfg.speculative_pointer_top_k]

        if (
            self.context.ablations.enable_atomic_facts
            and self.rag.cfg.enable_atomic_facts
            and self.rag.cfg.fact_hint_top_k > 0
        ):
            raw_facts = await self.rag.search(
                tree_id=tree.id,
                query=user_content,
                scope="all",
                current_node=refreshed_node,
                top_k=self.rag.cfg.fact_hint_top_k + 2,
                granularities={"fact"},
                use_reranker=False,
            )
            fact_hits = [
                h
                for h in raw_facts
                if h.node_id != refreshed_node.id
                and h.score >= self.rag.cfg.speculative_pointer_min_score
            ][: self.rag.cfg.fact_hint_top_k]

        return speculative_hits, fact_hits

    async def run_turn_stream(
        self,
        node_id: str,
        user_content: str,
        *,
        session_id: str = DEFAULT_SESSION_ID,
        max_hop_depth: int | None = None,
        token_budget: int | None = None,
        model_override: str | None = None,
        include_core_tools: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        node = await self.repo.get_node(node_id)
        if node is None:
            err = {"code": "NODE_NOT_FOUND", "message": f"Node {node_id} not found."}
            if self.hooks:
                await self.hooks.emit("error", err)
            yield {"event": "error", "data": err}
            return

        tree = await self.repo.get_tree(node.tree_id)
        if tree is None:
            err = {"code": "TREE_NOT_FOUND", "message": f"Tree {node.tree_id} not found."}
            if self.hooks:
                await self.hooks.emit("error", err)
            yield {"event": "error", "data": err}
            return

        trace_ctx = TraceContext.start(
            root_node_id=node.id,
            max_hop_depth=max_hop_depth if max_hop_depth is not None else self.guardrails.max_hop_depth,
            token_budget=token_budget if token_budget is not None else self.guardrails.token_budget,
            min_turn_reserve=self.guardrails.min_turn_reserve,
        )

        # Auto-compact / non-destructive split threshold for unbounded infinite-context sessions
        tree, node, compact_events = await self._maybe_auto_compact(
            tree, node, trace_ctx, session_id=session_id
        )
        node_id = node.id
        for cev in compact_events:
            yield cev

        # Promotable subagent lifecycle: if a human directly chats inside a subagent node,
        # seamlessly adopt it as a first-class interactive 'chat' branch.
        if node.node_kind == "subagent":
            promoted = await self.repo.update_node(node.id, node_kind="chat")
            if promoted is not None:
                node = promoted
            if self.hooks:
                await self.hooks.emit(
                    "subagent_promoted",
                    {"tree_id": tree.id, "node_id": node.id, "title": node.title},
                )

        reopened = False
        if node.status == "COMPLETED":
            await self.repo.update_node(node.id, status="ACTIVE", is_dirty=True)
            await self.reflog.record_transition(
                tree_id=tree.id,
                from_node=node,
                to_node=node,
                action="reopen",
                note=f"Reopened '{node.title}' for new turn",
                update_head=True,
                session_id=session_id,
            )
            reopened = True

        cur_session = await self.repo.get_session(tree.id, session_id)
        session_head_id = cur_session.head_node_id if cur_session is not None else tree.head_node_id
        if session_head_id != node.id:
            prev_head = (
                await self.repo.get_node(session_head_id) if session_head_id else None
            )
            await self.reflog.record_transition(
                tree_id=tree.id,
                from_node=prev_head,
                to_node=node,
                action="jump",
                note=f"Focused node '{node.title}'",
                update_head=True,
                session_id=session_id,
            )
            tree = await self.repo.get_tree(tree.id)
            assert tree is not None

        user_msg = await self.repo.append_message(
            node_id=node.id,
            tree_id=tree.id,
            role="user",
            content=user_content,
            mark_dirty=True,
        )

        affected_consumers = await self.staleness.get_affected_consumers(node.id)
        if affected_consumers:
            stale_payload = {"source_node_id": node.id, "stale_node_ids": affected_consumers}
            if self.hooks:
                await self.hooks.emit("stale_propagated", stale_payload)
            yield {"event": "staleness_updated", "data": stale_payload}

        refreshed_node = await self.repo.get_node(node.id)
        assert refreshed_node is not None

        # Pre-turn speculative local search (<5ms, 0 LLM tokens) + Tier-2.5 atomic dated facts
        speculative_hits, fact_hits = await self._collect_preturn_hints(
            tree, refreshed_node, user_content
        )

        # Drain unread background-actor mailbox deliveries for injection into this turn
        unread_mailbox = await self.repo.list_mailbox_items(
            tree.id, to_node_id=node.id, status="unread"
        )

        # Conservative steady-state envelope: omit redundant multi-line BRANCH SUMMARIES only when
        # continuing on the exact same node with zero branch/summary changes and no pending cues.
        compact_steady = False
        if self.guardrails.compact_steady_state_envelope and not reopened:
            all_tree_nodes = await self.repo.list_nodes(tree.id)
            other_sig = tuple(
                (n.id, n.status, n.summary_version, n.title)
                for n in all_tree_nodes
                if n.id != node.id and n.status != "ARCHIVED" and n.summary
            )
            cur_sig = (node.id, other_sig)
            sig_key = (tree.id, session_id)
            prev_sig = self._last_envelope_sig.get(sig_key)
            has_cues = bool(
                await self.repo.get_unconsumed_orientation_cues(
                    tree.id, node.id, session_id=session_id
                )
            )
            if prev_sig == cur_sig and not has_cues and not unread_mailbox:
                compact_steady = True
            self._last_envelope_sig[sig_key] = cur_sig

        llm_messages, cue_ids, notices = await self.context.assemble_for_active_turn(
            tree=tree,
            node=refreshed_node,
            newest_user_msg=user_msg,
            speculative_hits=speculative_hits,
            fact_hits=fact_hits,
            unread_mailbox=unread_mailbox,
            session_id=session_id,
            compact_steady_state=compact_steady,
        )
        turn_start_data = {
            "trace_id": trace_ctx.trace_id,
            "session_id": session_id,
            "node_id": node.id,
            "user_message": user_msg.model_dump(),
            "ephemeral_notices": notices,
            "delivered_mailbox_ids": [mb.id for mb in unread_mailbox],
            "reopened": reopened,
        }
        if self.hooks:
            await self.hooks.emit("turn_start", turn_start_data)
        yield {"event": "turn_start", "data": turn_start_data}

        if cue_ids:
            await self.repo.mark_orientation_cues_consumed(cue_ids)
        for mb in unread_mailbox:
            await self.repo.update_mailbox_item(mb.id, status="delivered")

        model = model_override or self.router.resolve_model(tree, refreshed_node, "chat")
        tool_schemas = self.tools.schemas_for_node(
            refreshed_node.tool_permission,
            include_core=include_core_tools,
            include_compound_ask=self.guardrails.enable_compound_ask_tool,
            is_shadow=False,
        )
        # When no tools are exposed on this turn, use a tool-free system prompt so models
        # (e.g. DeepSeek) never emit raw XML/DSML tool-call markup in plain text.
        if not tool_schemas and llm_messages and llm_messages[0].get("role") == "system":
            llm_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a concise, accurate assistant operating inside a conversation branch. "
                        "Answer directly in plain text based on the conversation history; no tools are available on this turn."
                    ),
                },
                *llm_messages[1:],
            ]
        # Freeze Round-0 prefix + user envelope across tool rounds so mid-turn tool loops
        # NEVER mutate the user message (preserving provider KV caches & preventing amnesia).
        frozen_turn_base = list(llm_messages)
        trailing_msgs: list[MessageModel] = []
        final_assistant_msg: MessageModel | None = None
        ipc_events_emitted: list[dict[str, Any]] = []
        finished_target_id: str | None = None
        finished_auto_committed = False

        fork_bp = await self.context.get_fork_breakpoint_index(refreshed_node)
        stream_params = inspect.signature(self.llm.stream_complete).parameters
        stream_extra_kw: dict[str, Any] = {}
        if fork_bp is not None and "fork_breakpoint_index" in stream_params:
            stream_extra_kw["fork_breakpoint_index"] = fork_bp

        for _round in range(self.guardrails.max_tool_rounds):
            if _round > 0:
                llm_messages = [
                    *frozen_turn_base,
                    *(_message_to_openai(m) for m in trailing_msgs),
                ]

            est_tokens = max(128, estimate_messages_tokens(llm_messages) // 4)
            reserved = trace_ctx.ledger.try_reserve(
                min(est_tokens, max(128, trace_ctx.ledger.remaining())), min_reserve=128
            )
            grace_synthesis = False
            if reserved is None:
                if _round > 0 and trailing_msgs:
                    # Tool rounds already fetched evidence; execute one final tool-free synthesis
                    # turn instead of dropping all tool outputs with an empty string!
                    grace_synthesis = True
                    reserved = 0
                else:
                    err = {
                        "code": "TOKEN_BUDGET_EXCEEDED",
                        "message": f"Trace token budget ({trace_ctx.ledger.token_budget}) reached.",
                    }
                    if self.hooks:
                        await self.hooks.emit("error", err)
                    yield {"event": "error", "data": err}
                    break

            round_msgs = llm_messages
            round_tools = None if grace_synthesis else (tool_schemas or None)
            if grace_synthesis and round_msgs and round_msgs[0].get("role") == "system":
                round_msgs = [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise, accurate assistant. Synthesize one direct final "
                            "answer from the tool outputs above without calling any further tools."
                        ),
                    },
                    *round_msgs[1:],
                ]
            # Enforce whole-request context-window budget (prefix + envelope + newest user msg +
            # trailing tool outputs + tool schemas + reply reserve) before EVERY provider call.
            round_msgs = self.context.fit_messages_to_budget(
                round_msgs, model=model, tools=round_tools
            )

            response: LLMResponse | None = None
            try:
                async for chunk in self.llm.stream_complete(
                    model=model,
                    messages=round_msgs,
                    tools=round_tools,
                    **stream_extra_kw,
                ):
                    if chunk.kind == "delta" and chunk.delta:
                        yield {"event": "token", "data": {"delta": chunk.delta}}
                    elif chunk.kind == "done":
                        response = chunk.response
            except Exception as exc:
                trace_ctx.ledger.release(reserved)
                err = {"code": "LLM_PROVIDER_ERROR", "message": f"Model '{model}' failed: {exc}"}
                if self.hooks:
                    await self.hooks.emit("error", err)
                yield {"event": "error", "data": err}
                return

            if response is None:
                trace_ctx.ledger.release(reserved)
                err = {"code": "EMPTY_RESPONSE", "message": "Model returned no response."}
                if self.hooks:
                    await self.hooks.emit("error", err)
                yield {"event": "error", "data": err}
                return

            self.ipc.mark_node_llm_warm(node.id)
            trace_ctx.ledger.settle(
                reserved,
                response.tokens_in,
                response.tokens_out,
                cache_read_tokens=response.cache_read_tokens,
                cache_read_weight=self.guardrails.cache_read_budget_weight,
            )

            if response.tool_calls:
                assistant_tc_msg = await self.repo.append_message(
                    node_id=node.id,
                    tree_id=tree.id,
                    role="assistant",
                    content=response.content,
                    tool_calls=[tc.to_openai_dict() for tc in response.tool_calls],
                    model_used=response.model_used,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    cache_read_tokens=response.cache_read_tokens,
                    latency_ms=response.latency_ms,
                    mark_dirty=True,
                )
                trailing_msgs.append(assistant_tc_msg)

                for tc in response.tool_calls:
                    yield {
                        "event": "tool_call",
                        "data": {
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }

                current_node = await self.repo.get_node(node.id)
                current_tree = await self.repo.get_tree(tree.id)
                assert current_node is not None and current_tree is not None

                batch_results = await self.ipc.execute_tool_batch(
                    tree=current_tree,
                    caller=current_node,
                    tool_calls=response.tool_calls,
                    trace_ctx=trace_ctx,
                    session_id=session_id,
                )

                for tc, ipc_res in batch_results:
                    if ipc_res.ipc_event is not None:
                        ev_dict = ipc_res.ipc_event.model_dump()
                        ipc_events_emitted.append(ev_dict)
                        yield {"event": "ipc_activity", "data": ev_dict}

                    if ipc_res.handoff_pending is not None:
                        yield {
                            "event": "handoff_pending",
                            "data": ipc_res.handoff_pending.model_dump(),
                        }

                    if ipc_res.branch_suggested is not None:
                        yield {
                            "event": "branch_suggested",
                            "data": ipc_res.branch_suggested,
                        }

                    if ipc_res.head_changed is not None:
                        from_id, to_id, rf_entry = ipc_res.head_changed
                        yield {
                            "event": "head_changed",
                            "data": {
                                "from_node_id": from_id,
                                "to_node_id": to_id,
                                "reflog_entry": rf_entry.model_dump(),
                            },
                        }

                    if ipc_res.stale_node_ids:
                        sp = {"source_node_id": node.id, "stale_node_ids": ipc_res.stale_node_ids}
                        if self.hooks:
                            await self.hooks.emit("stale_propagated", sp)
                        yield {"event": "staleness_updated", "data": sp}

                    if (
                        tc.name == TOOL_SEND
                        and tc.arguments.get("finish")
                        and not ipc_res.tool_output.get("error")
                    ):
                        finished_target_id = ipc_res.tool_output.get("target_id")
                        finished_auto_committed = (
                            ipc_res.tool_output.get("handoff_status") == "delivered"
                        )

                    tool_msg = await self.repo.append_message(
                        node_id=node.id,
                        tree_id=tree.id,
                        role="tool",
                        content=json.dumps(ipc_res.tool_output),
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        metadata={
                            "ipc_event_id": ipc_res.ipc_event.id if ipc_res.ipc_event else None
                        },
                        mark_dirty=not bool(finished_target_id),
                    )
                    trailing_msgs.append(tool_msg)
                continue

            final_assistant_msg = await self.repo.append_message(
                node_id=node.id,
                tree_id=tree.id,
                role="assistant",
                content=response.content,
                model_used=response.model_used,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                cache_read_tokens=response.cache_read_tokens,
                latency_ms=response.latency_ms,
                metadata={"ipc_events": [e["id"] for e in ipc_events_emitted]}
                if ipc_events_emitted
                else None,
                mark_dirty=not bool(finished_target_id),
            )
            if finished_target_id and finished_auto_committed:
                latest_sender = await self.repo.get_node(node.id)
                if latest_sender is not None:
                    await self.repo.upsert_dependency(
                        tree_id=tree.id,
                        consumer_node_id=finished_target_id,
                        source_node_id=node.id,
                        seen_summary_version=latest_sender.summary_version,
                        seen_msg_seq=latest_sender.last_msg_seq,
                    )
            break

        # Refresh Tier-1 chunk embeddings for this node after turn completion
        await self.rag.index_node_chunks(node.id)

        updated_node = await self.repo.get_node(node.id)
        # Non-blocking Turn-1 auto-title when branch still has a generic placeholder title
        if (
            self.on_first_turn_title is not None
            and self.guardrails.auto_title_enabled
            and updated_node is not None
            and updated_node.title_source == "placeholder"
        ):
            own_msgs = await self.repo.list_messages(updated_node.id)
            user_turns = sum(1 for m in own_msgs if m.role == "user")
            if user_turns == 1:
                self.on_first_turn_title(tree.id, updated_node.id)
        updated_session = await self.repo.get_session(tree.id, session_id)
        updated_tree = await self.repo.get_tree(tree.id)
        final_head_id = (
            updated_session.head_node_id
            if updated_session and updated_session.head_node_id
            else (updated_tree.head_node_id if updated_tree else node.id)
        )
        complete_data = {
            "trace_id": trace_ctx.trace_id,
            "session_id": session_id,
            "assistant_message": final_assistant_msg.model_dump() if final_assistant_msg else None,
            "node": updated_node.model_dump() if updated_node else None,
            "head_node_id": final_head_id,
            "trace_summary": {
                "tokens_used": trace_ctx.ledger.tokens_used,
                "token_budget": trace_ctx.ledger.token_budget,
                "ipc_count": len(ipc_events_emitted),
            },
        }
        if self.hooks:
            await self.hooks.emit("turn_complete", complete_data)
        yield {"event": "turn_complete", "data": complete_data}
