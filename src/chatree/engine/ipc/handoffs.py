from __future__ import annotations

import time
from typing import Any

from chatree.engine.guardrails import TraceContext
from chatree.engine.ipc.types import IpcExecutionResult
from chatree.types import (
    DEFAULT_SESSION_ID,
    ArtifactInput,
    ArtifactRef,
    MailboxItemModel,
    NodeModel,
    ReflogEntryModel,
    TreeModel,
)


class IpcHandoffsMixin:
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
        session_id: str = DEFAULT_SESSION_ID,
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
                        session_id=session_id,
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
        session_id: str = DEFAULT_SESSION_ID,
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
                session_id=session_id,
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
