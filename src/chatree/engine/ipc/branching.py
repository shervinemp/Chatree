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
        import_branch_ids: list[str] | None = None,
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

        # Multi-branch context imports: freeze bounded read-only summary/artifact snapshots from
        # up to 4 sibling/ancestor branches into `summary_brief` (0% KV invalidation) and record
        # automatic causal dependency edges so staleness propagates from every imported branch.
        imported_nodes: list[NodeModel] = []
        skipped_imports: list[str] = []
        effective_brief = summary_brief
        if import_branch_ids:
            dedup_ids = list(dict.fromkeys(str(bid).strip() for bid in import_branch_ids if str(bid).strip()))
            import_lines: list[str] = []
            for bid in dedup_ids:
                if len(imported_nodes) >= 4:
                    skipped_imports.append(f"{bid} (cap=4)")
                    continue
                if bid == parent.id and context_mode == "fork":
                    skipped_imports.append(f"{bid} (already inherited by fork)")
                    continue
                imp = await self.repo.get_node(bid)
                if imp is None or imp.tree_id != tree.id:
                    skipped_imports.append(f"{bid} (not found)")
                    continue
                if imp.status == "ARCHIVED":
                    skipped_imports.append(f"{bid} (archived)")
                    continue
                imp_summary = (imp.summary or "").strip()
                if not imp_summary:
                    imp_msgs = await self.repo.list_messages(imp.id)
                    imp_summary = " | ".join(
                        f"{m.role}: {' '.join(m.content.split())[:140]}"
                        for m in imp_msgs[-2:]
                        if m.role in ("user", "assistant") and m.content.strip()
                    ) or imp.goal or imp.title
                imp_summary = " ".join(imp_summary.split())[:400]
                imp_arts = await self.repo.list_artifacts_by_node(imp.id)
                art_tag = (
                    " | artifacts: " + ", ".join(f"{a.slug}@v{a.version}" for a in imp_arts[:5])
                    if imp_arts
                    else ""
                )
                import_lines.append(
                    f'- {imp.id} "{imp.title}" (v{imp.summary_version}, {imp.status}): {imp_summary}{art_tag}'
                )
                imported_nodes.append(imp)
            if import_lines:
                import_block = (
                    "[Imported Context Snapshots (frozen at spawn — query branch IDs if flagged STALE)]\n"
                    + "\n".join(import_lines)
                )
                effective_brief = (
                    f"{summary_brief.strip()}\n\n{import_block}" if summary_brief.strip() else import_block
                )

        child = await self.repo.create_child_node(
            parent=parent,
            title=short_title,
            goal=goal,
            context_mode=context_mode,
            summary_brief=effective_brief,
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
        if imported_nodes:
            out["imported_branch_ids"] = [imp.id for imp in imported_nodes]
        if skipped_imports:
            out["skipped_imports"] = skipped_imports

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
                "summary_brief": effective_brief,
                "autonomous": autonomous,
                "imported_branch_ids": [imp.id for imp in imported_nodes],
            },
            response=out,
            status="ok",
            target_summary_version=child.summary_version,
            target_msg_seq=child.last_msg_seq,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )

        for imp in imported_nodes:
            await self.repo.upsert_dependency(
                tree_id=tree.id,
                consumer_node_id=child.id,
                source_node_id=imp.id,
                seen_summary_version=imp.summary_version,
                seen_msg_seq=imp.last_msg_seq,
                last_ipc_event_id=ev.id,
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
