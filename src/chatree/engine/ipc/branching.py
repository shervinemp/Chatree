from __future__ import annotations

import time
from typing import Any

from chatree.config import ToolPermission
from chatree.engine.guardrails import TraceContext
from chatree.engine.ipc.types import IpcExecutionResult
from chatree.types import (
    DEFAULT_SESSION_ID,
    ChildContextMode,
    NodeKind,
    NodeModel,
    TreeModel,
)


class IpcBranchingMixin:
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
    ablations: Any
    sandbox: Any
    on_branch_switch_away: Any
    _resolve_model: Any

    # ------------------------------------------------------------------
    # 1. spawn_child
    # ------------------------------------------------------------------

    async def spawn_child(
        self,
        tree: TreeModel,
        parent: NodeModel,
        goal: str,
        title: str | None = None,
        context_mode: ChildContextMode = "fork",
        summary_brief: str = "",
        slice_turns: int = 6,
        autonomous: bool = False,
        model_override: str | None = None,
        tool_permission: ToolPermission | None = None,
        node_kind: NodeKind | None = None,
        switch_head: bool = True,
        initiated_by_model: bool = False,
        session_id: str = DEFAULT_SESSION_ID,
        trace_ctx: TraceContext | None = None,
    ) -> IpcExecutionResult:
        ctx = trace_ctx or TraceContext.start(
            parent.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()

        # Spawn hysteresis & mixed-initiative policy (prevents weak-model branch thrashing)
        if initiated_by_model and not autonomous:
            if self.guardrails.spawn_policy == "manual":
                ev = await self.repo.create_ipc_event(
                    tree_id=tree.id,
                    trace_id=ctx.trace_id,
                    hop_depth=ctx.hop_depth,
                    from_node_id=parent.id,
                    to_node_id=None,
                    event_type="spawn",
                    payload={"goal": goal, "context_mode": context_mode},
                    response={"error": "SPAWN_POLICY_MANUAL", "detail": "Model-initiated spawning is disabled (spawn_policy='manual')."},
                    status="rejected_policy",
                )
                return IpcExecutionResult(
                    tool_output={"error": "SPAWN_POLICY_MANUAL", "detail": "Model-initiated spawning is disabled."},
                    ipc_event=ev,
                )
            if self.guardrails.spawn_policy == "suggest":
                suggestion = {
                    "parent_id": parent.id,
                    "title": (title or "").strip() or " ".join(goal.split()[:5]).rstrip(".,?!"),
                    "goal": goal,
                    "context_mode": context_mode,
                    "summary_brief": summary_brief,
                }
                if self.hooks is not None:
                    await self.hooks.emit("branch_suggested", {"tree_id": tree.id, **suggestion})
                ev = await self.repo.create_ipc_event(
                    tree_id=tree.id,
                    trace_id=ctx.trace_id,
                    hop_depth=ctx.hop_depth,
                    from_node_id=parent.id,
                    to_node_id=None,
                    event_type="spawn",
                    payload=suggestion,
                    response={"status": "suggested", "suggested_branch": suggestion},
                    status="suggested",
                )
                return IpcExecutionResult(
                    tool_output={"status": "suggested", "suggested_branch": suggestion},
                    ipc_event=ev,
                    branch_suggested=suggestion,
                )
            active_children = await self.repo.count_active_children(parent.id)
            if active_children >= self.guardrails.max_active_children:
                detail = (
                    f"Parent '{parent.id}' already has {active_children} active children "
                    f"(max_active_children={self.guardrails.max_active_children})."
                )
                ev = await self.repo.create_ipc_event(
                    tree_id=tree.id,
                    trace_id=ctx.trace_id,
                    hop_depth=ctx.hop_depth,
                    from_node_id=parent.id,
                    to_node_id=None,
                    event_type="spawn",
                    payload={"goal": goal},
                    response={"error": "MAX_ACTIVE_CHILDREN_EXCEEDED", "detail": detail},
                    status="rejected_policy",
                )
                return IpcExecutionResult(
                    tool_output={"error": "MAX_ACTIVE_CHILDREN_EXCEEDED", "detail": detail},
                    ipc_event=ev,
                )

        if autonomous:
            decision = ctx.check_ipc_call(
                f"spawn:{parent.id}:{ctx.hop_depth}", costs_hop=True, costs_tokens=True
            )
            if not decision.allowed:
                ev = await self.repo.create_ipc_event(
                    tree_id=tree.id,
                    trace_id=ctx.trace_id,
                    hop_depth=ctx.hop_depth,
                    from_node_id=parent.id,
                    to_node_id=None,
                    event_type="spawn",
                    payload={"goal": goal, "context_mode": context_mode, "autonomous": True},
                    response={"error": decision.error_code, "detail": decision.detail},
                    status=decision.status,
                )
                return IpcExecutionResult(
                    tool_output={"error": decision.error_code, "detail": decision.detail},
                    ipc_event=ev,
                )

        if context_mode not in ("fork", "slice", "clean"):
            context_mode = "fork"

        resolved_perm: ToolPermission = (
            tool_permission
            if tool_permission in ("read_only", "read_write")
            else self.guardrails.default_child_permission
        )
        resolved_kind: NodeKind = (
            node_kind
            if node_kind in ("chat", "subagent")
            else ("subagent" if autonomous else "chat")
        )
        explicit_title = (title or "").strip()
        short_title = (
            explicit_title
            or " ".join(goal.split()[:5]).rstrip(".,?!")
            or ("Sub-Agent Worker" if resolved_kind == "subagent" else "Child Branch")
        )
        from chatree.types import TitleSource, is_placeholder_title

        resolved_title_source: TitleSource = (
            "placeholder"
            if is_placeholder_title(short_title)
            else ("user" if explicit_title else "auto")
        )
        safe_fork_seq = await self.repo.get_last_safe_fork_seq(parent.id)

        child = await self.repo.create_child_node(
            parent=parent,
            title=short_title,
            goal=goal,
            context_mode=context_mode,
            summary_brief=summary_brief,
            spawn_msg_seq=safe_fork_seq,
            slice_turn_count=slice_turns,
            model_override=model_override,
            tool_permission=resolved_perm,
            node_kind=resolved_kind,
            title_source=resolved_title_source,
        )

        head_changed = None
        autonomous_result: str | None = None
        tokens_in = 0
        tokens_out = 0

        if autonomous:
            seed_prompt = f"Goal: {goal}" + (f"\nBrief: {summary_brief}" if summary_brief else "")
            user_msg = await self.repo.append_message(
                node_id=child.id, tree_id=tree.id, role="user", content=seed_prompt
            )
            refreshed_child = await self.repo.get_node(child.id)
            assert refreshed_child is not None
            llm_msgs, _, _ = await self.context.assemble_for_active_turn(
                tree, refreshed_child, user_msg
            )
            child_model = self.router.resolve_model(tree, refreshed_child, "chat")
            resp = await self.llm.complete(model=child_model, messages=llm_msgs, tools=None)
            ctx.ledger.record(resp.tokens_in, resp.tokens_out)
            tokens_in, tokens_out = resp.tokens_in, resp.tokens_out

            await self.repo.append_message(
                node_id=child.id,
                tree_id=tree.id,
                role="assistant",
                content=resp.content,
                model_used=resp.model_used,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                cache_read_tokens=resp.cache_read_tokens,
                latency_ms=resp.latency_ms,
            )
            child = await self.repo.commit_node_summary(
                child.id, resp.content, status="COMPLETED"
            )
            await self.rag.index_node_chunks(child.id)
            await self.rag.index_node_summary(child.id)
            await self.rag.propagate_rollups_upward(child.id)
            autonomous_result = resp.content
        elif switch_head:
            # Strict HEAD-only reflog invariant: record only interactive HEAD transitions.
            entry = await self.reflog.record_transition(
                tree_id=tree.id,
                from_node=parent,
                to_node=child,
                action="spawn",
                note=f"Spawned {context_mode} branch: {child.title}",
                update_head=True,
                session_id=session_id,
            )
            head_changed = (parent.id, child.id, entry)
            if (
                self.on_branch_switch_away is not None
                and parent.parent_id is not None
                and parent.is_dirty
            ):
                self.on_branch_switch_away(tree.id, parent.id)

        await self.rag.index_node_summary(child.id)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        out: dict[str, Any] = {
            "child_id": child.id,
            "title": child.title,
            "node_kind": child.node_kind,
            "context_mode": child.context_mode,
            "tool_permission": child.tool_permission,
            "spawn_msg_seq": child.spawn_msg_seq,
            "status": child.status,
            "switched_head": bool(head_changed),
        }
        if autonomous_result is not None:
            out["autonomous_result"] = autonomous_result
            out["summary_version"] = child.summary_version

        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=ctx.trace_id,
            hop_depth=ctx.hop_depth,
            from_node_id=parent.id,
            to_node_id=child.id,
            event_type="spawn",
            payload={
                "goal": goal,
                "title": child.title,
                "node_kind": child.node_kind,
                "context_mode": context_mode,
                "tool_permission": child.tool_permission,
                "summary_brief": summary_brief,
                "autonomous": autonomous,
            },
            response=out,
            status="ok",
            target_summary_version=child.summary_version,
            target_msg_seq=child.last_msg_seq,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )

        if autonomous:
            await self.repo.upsert_dependency(
                tree_id=tree.id,
                consumer_node_id=parent.id,
                source_node_id=child.id,
                seen_summary_version=child.summary_version,
                seen_msg_seq=child.last_msg_seq,
                last_ipc_event_id=ev.id,
            )

        if self.hooks is not None:
            await self.hooks.emit(
                "spawn",
                {
                    "tree_id": tree.id,
                    "parent_id": parent.id,
                    "child": child.model_dump(),
                    "autonomous": autonomous,
                },
            )

        return IpcExecutionResult(tool_output=out, ipc_event=ev, head_changed=head_changed)
