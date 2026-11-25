from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import json
import re
import time
from typing import Any, Callable

from chatree.config import AblationConfig, GuardrailPolicy, ToolPermission
from chatree.engine.context import ContextAssembler
from chatree.engine.guardrails import TraceContext
from chatree.engine.rag import TreeRAGService, extract_key_identifiers
from chatree.engine.reflog import ReflogService
from chatree.engine.staleness import StalenessService
from chatree.hooks import HookBus
from chatree.protocols import LLMProvider, ModelRouter
from chatree.providers.mock import estimate_messages_tokens
from chatree.sandbox import BranchSandbox
from chatree.storage.repository import SQLiteRepository, cosine_similarity
from chatree.tools import (
    CORE_TOOL_NAMES,
    SHADOW_READ_ONLY_CORE_TOOLS,
    TOOL_ARCHIVE_BRANCH,
    TOOL_ASK_TREE,
    TOOL_COMPACT_BRANCH,
    TOOL_PIN_INVARIANT,
    TOOL_REFLOG_JUMP,
    TOOL_REQUEST,
    TOOL_SEARCH_TREE,
    TOOL_SEND,
    TOOL_SPAWN_CHILD,
    ToolCallContext,
    ToolRegistry,
)
from chatree.types import (
    ArtifactInput,
    ArtifactRef,
    ChildContextMode,
    IpcEventModel,
    LLMToolCall,
    MailboxItemModel,
    NodeModel,
    ReflogEntryModel,
    SearchScope,
    TreeModel,
)


from chatree.engine.ipc.types import IpcExecutionResult

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
        switch_head: bool = True,
        initiated_by_model: bool = False,
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
        short_title = (
            (title or "").strip()
            or " ".join(goal.split()[:5]).rstrip(".,?!")
            or "Child Branch"
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

            await self.reflog.record_transition(
                tree_id=tree.id,
                from_node=parent,
                to_node=child,
                action="spawn",
                note=f"Autonomous child '{child.title}' completed",
                update_head=False,
            )
        elif switch_head:
            entry = await self.reflog.record_transition(
                tree_id=tree.id,
                from_node=parent,
                to_node=child,
                action="spawn",
                note=f"Spawned {context_mode} branch: {child.title}",
                update_head=True,
            )
            head_changed = (parent.id, child.id, entry)
            if (
                self.on_branch_switch_away is not None
                and parent.parent_id is not None
                and parent.is_dirty
            ):
                self.on_branch_switch_away(tree.id, parent.id)
        else:
            await self.reflog.record_transition(
                tree_id=tree.id,
                from_node=parent,
                to_node=child,
                action="spawn",
                note=f"Created {context_mode} branch: {child.title}",
                update_head=False,
            )

        await self.rag.index_node_summary(child.id)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        out: dict[str, Any] = {
            "child_id": child.id,
            "title": child.title,
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

    # ------------------------------------------------------------------
    # 2. send
    # ------------------------------------------------------------------

    async def send(
        self,
        tree: TreeModel,
        sender: NodeModel,
        target_id: str,
        content: str,
        artifacts: list[ArtifactInput] | None = None,
        finish: bool = False,
        auto_commit_override: bool | None = None,
        return_to_parent: bool = True,
        trace_ctx: TraceContext | None = None,
    ) -> IpcExecutionResult:
        ctx = trace_ctx or TraceContext.start(
            sender.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()

        resolved_target_id = target_id or sender.parent_id
        if not resolved_target_id:
            return IpcExecutionResult(
                tool_output={"error": "NO_TARGET", "detail": "Root node has no parent target for send()."}
            )
        target = await self.repo.get_node(resolved_target_id)
        if target is None or target.tree_id != tree.id:
            return IpcExecutionResult(
                tool_output={
                    "error": "TARGET_NOT_FOUND",
                    "detail": f"Target node {resolved_target_id} does not exist in this tree.",
                }
            )

        artifact_refs: list[ArtifactRef] = []
        for art_in in artifacts or []:
            created = await self.repo.create_artifact(
                tree_id=tree.id, art=art_in, created_by_node_id=sender.id
            )
            artifact_refs.append(
                ArtifactRef(
                    id=created.id,
                    slug=created.slug,
                    version=created.version,
                    kind=created.kind,
                    title=created.title,
                    summary=created.summary,
                )
            )

        if finish:
            updated_sender = await self.staleness.commit_summary_with_materiality(
                sender, content, status="COMPLETED"
            )

            should_auto_commit = (
                auto_commit_override
                if auto_commit_override is not None
                else tree.auto_commit_handoffs
            )
            mb_status = "delivered" if should_auto_commit else "pending_review"
            latency_ms = int((time.perf_counter() - t0) * 1000)

            out = {
                "ok": True,
                "finish": True,
                "from_node_id": sender.id,
                "target_id": target.id,
                "summary_version": updated_sender.summary_version,
                "handoff_status": mb_status,
                "artifacts": [r.ref_label for r in artifact_refs],
            }
            ev = await self.repo.create_ipc_event(
                tree_id=tree.id,
                trace_id=ctx.trace_id,
                hop_depth=ctx.hop_depth,
                from_node_id=sender.id,
                to_node_id=target.id,
                event_type="send_finish",
                payload={
                    "summary": content,
                    "artifacts": [r.model_dump() for r in artifact_refs],
                    "finish": True,
                },
                response=out,
                status="ok" if should_auto_commit else "pending_review",
                target_summary_version=updated_sender.summary_version,
                target_msg_seq=updated_sender.last_msg_seq,
                latency_ms=latency_ms,
            )
            mb_item = await self.repo.create_mailbox_item(
                tree_id=tree.id,
                to_node_id=target.id,
                from_node_id=sender.id,
                kind="handoff",
                status=mb_status,
                summary=content,
                artifact_refs=artifact_refs,
                payload={"sender_title": sender.title, "summary_version": updated_sender.summary_version},
                ipc_event_id=ev.id,
            )

            if self.hooks is not None:
                await self.hooks.emit(
                    "handoff_created",
                    {
                        "tree_id": tree.id,
                        "mailbox_item": mb_item.model_dump(),
                        "auto_commit": should_auto_commit,
                    },
                )

            head_changed = None
            stale_consumers = await self.staleness.get_affected_consumers(sender.id)

            if should_auto_commit:
                await self._deliver_handoff_to_target(
                    tree=tree,
                    sender=updated_sender,
                    target=target,
                    summary=content,
                    artifact_refs=artifact_refs,
                    ipc_event_id=ev.id,
                )
                if self.hooks is not None:
                    await self.hooks.emit(
                        "handoff_committed",
                        {"tree_id": tree.id, "mailbox_item": mb_item.model_dump()},
                    )
                if return_to_parent:
                    rf = await self.reflog.record_transition(
                        tree_id=tree.id,
                        from_node=updated_sender,
                        to_node=target,
                        action="return",
                        note=f"Auto-merged summary v{updated_sender.summary_version}",
                        update_head=True,
                    )
                    head_changed = (sender.id, target.id, rf)

            return IpcExecutionResult(
                tool_output=out,
                ipc_event=ev,
                head_changed=head_changed,
                handoff_pending=None if should_auto_commit else mb_item,
                stale_node_ids=[nid for nid in stale_consumers if nid != target.id],
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        out = {
            "ok": True,
            "finish": False,
            "from_node_id": sender.id,
            "target_id": target.id,
            "artifacts": [r.ref_label for r in artifact_refs],
        }
        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=ctx.trace_id,
            hop_depth=ctx.hop_depth,
            from_node_id=sender.id,
            to_node_id=target.id,
            event_type="send",
            payload={
                "content": content,
                "artifacts": [r.model_dump() for r in artifact_refs],
                "finish": False,
            },
            response=out,
            status="ok",
            latency_ms=latency_ms,
        )
        await self.repo.create_mailbox_item(
            tree_id=tree.id,
            to_node_id=target.id,
            from_node_id=sender.id,
            kind="message",
            status="delivered",
            summary=content,
            artifact_refs=artifact_refs,
            payload={"sender_title": sender.title},
            ipc_event_id=ev.id,
        )
        art_line = (
            "\nArtifacts: " + ", ".join(r.ref_label for r in artifact_refs)
            if artifact_refs
            else ""
        )
        await self.repo.append_message(
            node_id=target.id,
            tree_id=tree.id,
            role="user",
            content=f"[IPC Message from {sender.title} (`{sender.id}`)]:\n{content}{art_line}",
            metadata={
                "ipc_delivery": True,
                "from_node_id": sender.id,
                "ipc_event_id": ev.id,
                "artifact_refs": [r.model_dump() for r in artifact_refs],
            },
        )
        await self.rag.index_node_chunks(target.id)
        return IpcExecutionResult(tool_output=out, ipc_event=ev)

    async def commit_pending_handoff(
        self,
        mailbox_id: str,
        edited_summary: str | None = None,
        edited_artifacts: list[ArtifactInput] | None = None,
        return_head_to_target: bool = True,
    ) -> tuple[MailboxItemModel, NodeModel, ReflogEntryModel | None]:
        mb = await self.repo.get_mailbox_item(mailbox_id)
        if mb is None:
            raise ValueError(f"Mailbox item not found: {mailbox_id}")
        if mb.status != "pending_review":
            raise ValueError(f"Mailbox item {mailbox_id} is already {mb.status}")

        tree = await self.repo.get_tree(mb.tree_id)
        sender = await self.repo.get_node(mb.from_node_id)
        target = await self.repo.get_node(mb.to_node_id)
        if not tree or not sender or not target:
            raise ValueError("Tree, sender, or target node missing for handoff commit.")

        final_summary = edited_summary.strip() if edited_summary is not None else mb.summary
        final_refs = list(mb.artifact_refs)

        if edited_artifacts is not None:
            final_refs = []
            for art_in in edited_artifacts:
                created = await self.repo.create_artifact(
                    tree_id=tree.id, art=art_in, created_by_node_id=sender.id
                )
                final_refs.append(
                    ArtifactRef(
                        id=created.id,
                        slug=created.slug,
                        version=created.version,
                        kind=created.kind,
                        title=created.title,
                        summary=created.summary,
                    )
                )

        if edited_summary is not None and final_summary != (sender.summary or ""):
            sender = await self.repo.commit_node_summary(
                sender.id, summary=final_summary, status="COMPLETED"
            )
            await self.rag.index_node_summary(sender.id)
            await self.rag.propagate_rollups_upward(sender.id)

        await self._deliver_handoff_to_target(
            tree=tree,
            sender=sender,
            target=target,
            summary=final_summary,
            artifact_refs=final_refs,
            ipc_event_id=mb.ipc_event_id,
        )
        updated_mb = await self.repo.update_mailbox_item(
            mailbox_id, status="delivered", summary=final_summary, artifact_refs=final_refs
        )
        assert updated_mb is not None

        if self.hooks is not None:
            await self.hooks.emit(
                "handoff_committed",
                {"tree_id": tree.id, "mailbox_item": updated_mb.model_dump()},
            )

        reflog_entry: ReflogEntryModel | None = None
        if return_head_to_target:
            reflog_entry = await self.reflog.record_transition(
                tree_id=tree.id,
                from_node=sender,
                to_node=target,
                action="return",
                note=f"Merged handoff v{sender.summary_version} from '{sender.title}'",
                update_head=True,
            )

        refreshed_target = await self.repo.get_node(target.id)
        assert refreshed_target is not None
        return updated_mb, refreshed_target, reflog_entry

    async def _deliver_handoff_to_target(
        self,
        tree: TreeModel,
        sender: NodeModel,
        target: NodeModel,
        summary: str,
        artifact_refs: list[ArtifactRef],
        ipc_event_id: str | None,
    ) -> None:
        art_block = ""
        if artifact_refs:
            art_lines = [
                f"- `{r.ref_label}` ({r.kind}): {r.title}" + (f" — {r.summary}" if r.summary else "")
                for r in artifact_refs
            ]
            art_block = "\n\n**Attached Artifacts:**\n" + "\n".join(art_lines)

        handoff_text = (
            f"[Handoff from child `{sender.id}` ({sender.title}) · summary v{sender.summary_version}]\n\n"
            f"{summary}{art_block}"
        )
        await self.repo.append_message(
            node_id=target.id,
            tree_id=tree.id,
            role="user",
            content=handoff_text,
            metadata={
                "ipc_handoff": True,
                "from_node_id": sender.id,
                "from_node_title": sender.title,
                "summary_version": sender.summary_version,
                "artifact_refs": [r.model_dump() for r in artifact_refs],
            },
        )
        await self.repo.upsert_dependency(
            tree_id=tree.id,
            consumer_node_id=target.id,
            source_node_id=sender.id,
            seen_summary_version=sender.summary_version,
            seen_msg_seq=sender.last_msg_seq,
            last_ipc_event_id=ipc_event_id,
        )
        await self.rag.index_node_chunks(target.id)


    # ------------------------------------------------------------------
    # 5. reflog_jump
    # ------------------------------------------------------------------

    async def reflog_jump(
        self,
        tree: TreeModel,
        caller: NodeModel,
        target: str,
        note: str = "",
        follow_up_prompt: str | None = None,
        trace_ctx: TraceContext | None = None,
    ) -> IpcExecutionResult:
        ctx = trace_ctx or TraceContext.start(
            caller.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()
        try:
            dest, entry, _cursor = await self.reflog.jump(tree.id, target, note=note)
        except ValueError as exc:
            return IpcExecutionResult(
                tool_output={"error": "REFLOG_JUMP_FAILED", "detail": str(exc)}
            )

        # Trigger optional non-blocking Tier-2 background summarization for the branch we left
        if (
            self.on_branch_switch_away is not None
            and caller.id != dest.id
            and caller.parent_id is not None
            and caller.is_dirty
        ):
            self.on_branch_switch_away(tree.id, caller.id)

        # Build compact destination context preview so the model has immediate continuity on arrival
        dest_msgs = await self.repo.list_messages(dest.id)
        recent_turns: list[dict[str, str]] = []
        for m in dest_msgs:
            if m.role in ("user", "assistant") and m.content.strip():
                recent_turns.append({"role": m.role, "content": m.content.strip()[:220]})
        recent_turns = recent_turns[-2:]

        clean_follow = (follow_up_prompt or "").strip()
        follow_up_response: str | None = None
        follow_reopened = False
        tokens_in = 0
        tokens_out = 0
        stale_node_ids: list[str] = []

        if clean_follow:
            decision = ctx.check_ipc_call(dest.id, costs_hop=True, costs_tokens=True)
            if not decision.allowed:
                ev = await self.repo.create_ipc_event(
                    tree_id=tree.id,
                    trace_id=ctx.trace_id,
                    hop_depth=ctx.hop_depth,
                    from_node_id=caller.id,
                    to_node_id=dest.id,
                    event_type="reflog_jump",
                    payload={"target": target, "note": note, "follow_up_prompt": clean_follow},
                    response={"error": decision.error_code, "detail": decision.detail},
                    status=decision.status,
                )
                return IpcExecutionResult(
                    tool_output={"error": decision.error_code, "detail": decision.detail},
                    ipc_event=ev,
                )

            if dest.status == "COMPLETED":
                await self.repo.update_node(dest.id, status="ACTIVE", is_dirty=True)
                follow_reopened = True

            user_msg = await self.repo.append_message(
                node_id=dest.id,
                tree_id=tree.id,
                role="user",
                content=clean_follow,
                mark_dirty=True,
            )
            refreshed_dest = await self.repo.get_node(dest.id)
            refreshed_tree = await self.repo.get_tree(tree.id)
            assert refreshed_dest is not None and refreshed_tree is not None

            dest_llm_msgs, cue_ids, _ = await self.context.assemble_for_active_turn(
                tree=refreshed_tree, node=refreshed_dest, newest_user_msg=user_msg
            )
            if cue_ids:
                await self.repo.mark_orientation_cues_consumed(cue_ids)

            dest_model = self._resolve_model(refreshed_tree, refreshed_dest, "chat")
            est_prompt = max(
                self.guardrails.min_turn_reserve, estimate_messages_tokens(dest_llm_msgs)
            )
            reserved = ctx.ledger.try_reserve(
                est_prompt, min_reserve=self.guardrails.min_turn_reserve
            )
            if reserved is None:
                ev = await self.repo.create_ipc_event(
                    tree_id=tree.id,
                    trace_id=ctx.trace_id,
                    hop_depth=ctx.hop_depth,
                    from_node_id=caller.id,
                    to_node_id=dest.id,
                    event_type="reflog_jump",
                    payload={"target": target, "note": note, "follow_up_prompt": clean_follow},
                    response={
                        "error": "TOKEN_BUDGET_EXCEEDED",
                        "detail": "Trace token budget exhausted before follow_up_prompt execution.",
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

            dest_tools = (
                self.tools.schemas_for_node(
                    refreshed_dest.tool_permission,
                    include_core=True,
                    include_compound_ask=self.guardrails.enable_compound_ask_tool,
                    is_shadow=False,
                )
                if self.guardrails.preserve_kv_cache_tool_schemas
                else None
            )
            supports_tc = "tool_choice" in inspect.signature(self.llm.complete).parameters
            try:
                resp = None
                if dest_tools and supports_tc:
                    try:
                        resp = await self.llm.complete(
                            model=dest_model,
                            messages=dest_llm_msgs,
                            tools=dest_tools,
                            temperature=0.2,
                            tool_choice="none",  # type: ignore[call-arg]
                        )
                    except Exception:
                        resp = None
                if resp is None:
                    tf_msgs = list(dest_llm_msgs)
                    if tf_msgs and tf_msgs[0].get("role") == "system":
                        tf_msgs[0] = {
                            "role": "system",
                            "content": (
                                "You are a concise, accurate assistant operating inside a conversation branch. "
                                "Answer directly in plain text based on this branch's history; no tools are available."
                            ),
                        }
                    resp = await self.llm.complete(
                        model=dest_model, messages=tf_msgs, tools=None, temperature=0.2
                    )
            except Exception:
                ctx.ledger.release(reserved)
                raise

            ctx.ledger.settle(reserved, resp.tokens_in, resp.tokens_out)
            tokens_in, tokens_out = resp.tokens_in, resp.tokens_out
            follow_up_response = resp.content

            await self.repo.append_message(
                node_id=dest.id,
                tree_id=tree.id,
                role="assistant",
                content=resp.content,
                model_used=resp.model_used,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                cache_read_tokens=resp.cache_read_tokens,
                latency_ms=resp.latency_ms,
                mark_dirty=True,
            )
            await self.rag.index_node_chunks(dest.id)
            dest = await self.repo.get_node(dest.id) or refreshed_dest
            stale_node_ids = [
                cid
                for cid in await self.staleness.get_affected_consumers(dest.id)
                if cid != dest.id
            ]
            recent_turns = [
                {"role": "user", "content": clean_follow[:220]},
                {"role": "assistant", "content": resp.content.strip()[:220]},
            ]

        latency_ms = int((time.perf_counter() - t0) * 1000)
        out: dict[str, Any] = {
            "ok": True,
            "from_node_id": caller.id,
            "to_node_id": dest.id,
            "to_node_title": dest.title,
            "head_ref": entry.head_ref if entry else "HEAD@{0}",
            "dest_goal": dest.goal or "",
            "dest_status": dest.status,
            "dest_summary": (dest.summary or "")[:360],
            "dest_recent_turns": recent_turns,
        }
        if follow_up_response is not None:
            out["follow_up_executed"] = True
            out["follow_up_reopened"] = follow_reopened
            out["follow_up_response"] = follow_up_response
            out["stale_node_ids"] = stale_node_ids

        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=ctx.trace_id,
            hop_depth=ctx.hop_depth,
            from_node_id=caller.id,
            to_node_id=dest.id,
            event_type="reflog_jump",
            payload={
                "target": target,
                "note": note,
                "follow_up_prompt": clean_follow or None,
            },
            response=out,
            status="ok",
            target_summary_version=dest.summary_version,
            target_msg_seq=dest.last_msg_seq,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        head_changed = (caller.id, dest.id, entry) if entry else None
        return IpcExecutionResult(
            tool_output=out,
            ipc_event=ev,
            head_changed=head_changed,
            stale_node_ids=stale_node_ids,
        )

