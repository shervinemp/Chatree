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

class IpcLifecycleMixin:
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

    # ------------------------------------------------------------------
    # 6. compact_branch (Non-Destructive Continuation Chaining)
    # ------------------------------------------------------------------

    @staticmethod
    def _next_continuation_title(base_title: str) -> str:
        m = re.search(r"\s*\(cont\.\s*(\d+)\)\s*$", base_title.strip())
        if m:
            num = int(m.group(1)) + 1
            stem = base_title[: m.start()].strip()
            return f"{stem} (cont. {num})"
        return f"{base_title.strip()} (cont. 2)"

    async def compact_branch(
        self,
        tree: TreeModel,
        source: NodeModel,
        *,
        keep_recent_turns: int | None = None,
        title: str | None = None,
        reason: str = "",
        switch_head: bool = True,
        trace_ctx: TraceContext | None = None,
    ) -> IpcExecutionResult:
        """
        Non-destructively compact `source` into a linked continuation child node:
          1. Leaves `source`'s verbatim messages & KV-cache prefix 100% untouched.
          2. Generates a structured 5-section anchor-preserving handoff using `source`'s warm
             KV prefix and commits it as `source`'s Tier-2 summary (marking `source` COMPLETED
             and indexing Tier-1 chunks + Tier-2 summary + Tier-3 subtree rollups).
          3. Spawns a linked continuation child under `source` (`slice` keeping the last
             `keep_recent_turns` turns, or `clean` if 0) with the compaction handoff + explicit
             ancestor back-pointer, records a `new -> source` dependency edge, and moves HEAD.
        """
        ctx = trace_ctx or TraceContext.start(
            source.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()
        keep_n = (
            self.guardrails.auto_compact_keep_recent_turns
            if keep_recent_turns is None
            else max(0, min(keep_recent_turns, 24))
        )
        tokens_before = await self.context.estimate_node_context_tokens(source)

        # Ensure all verbatim turns of `source` are indexed in Tier-1 RAG before archiving
        await self.rag.index_node_chunks(source.id)

        updated_source, handoff_text, t_in, t_out, _llm_ms = (
            await self.staleness.generate_compaction_handoff(
                tree=tree,
                node=source,
                trace_ctx=ctx,
                reason=reason,
            )
        )
        self.mark_node_llm_warm(source.id)

        cont_title = (title or "").strip() or self._next_continuation_title(updated_source.title)
        child_mode: ChildContextMode = "slice" if keep_n > 0 else "clean"
        safe_seq = await self.repo.get_last_safe_fork_seq(updated_source.id)
        back_pointer = (
            f'[Compacted from ancestor {updated_source.id} "{updated_source.title}" '
            f"(verbatim history preserved — call {TOOL_REQUEST}(\"{updated_source.id}\", prompt=...) "
            f'or {TOOL_ASK_TREE}(query=..., scope="ancestors") for older details)]\n'
            f"{handoff_text}"
        )

        cont_node = await self.repo.create_child_node(
            parent=updated_source,
            title=cont_title,
            goal=updated_source.goal or updated_source.title,
            context_mode=child_mode,
            summary_brief=back_pointer,
            spawn_msg_seq=safe_seq,
            slice_turn_count=max(1, keep_n),
            model_override=updated_source.model_override,
            tool_permission=updated_source.tool_permission,
        )
        await self.rag.index_node_summary(cont_node.id)

        tokens_after = await self.context.estimate_node_context_tokens(cont_node)
        # Compaction savings guard: if `slice` mode keeps too many recent turns and expands
        # working context (`tokens_after >= tokens_before`), step down to `clean` mode so
        # compaction actually reduces context rather than growing it.
        if (
            self.guardrails.compaction_enforce_savings
            and child_mode == "slice"
            and tokens_before > 0
            and tokens_after >= tokens_before
        ):
            async with self.repo.db.write_transaction():
                await self.repo.conn.execute(
                    "UPDATE nodes SET context_mode = 'clean', slice_turn_count = 0 WHERE id = ?",
                    (cont_node.id,),
                )
            refreshed_cont = await self.repo.get_node(cont_node.id)
            if refreshed_cont is not None:
                cont_node = refreshed_cont
                child_mode = "clean"
                keep_n = 0
                tokens_after = await self.context.estimate_node_context_tokens(cont_node)
        compression_ratio = (
            round(tokens_before / max(1, tokens_after), 2) if tokens_before > 0 else 1.0
        )

        note_txt = (
            reason.strip()
            or f"Compacted '{updated_source.title}' ({tokens_before} -> {tokens_after} tok, {compression_ratio}x)"
        )
        cue = (
            f"Compacted continuation of {updated_source.id} ({updated_source.title}, "
            f"{tokens_before}->{tokens_after} tok). Ancestor history remains verbatim & queryable."
        )
        entry = await self.reflog.record_transition(
            tree_id=tree.id,
            from_node=updated_source,
            to_node=cont_node,
            action="spawn",
            note=note_txt,
            orientation_cue=cue,
            update_head=switch_head,
        )
        head_changed = (updated_source.id, cont_node.id, entry) if switch_head else None

        latency_ms = int((time.perf_counter() - t0) * 1000)
        out: dict[str, Any] = {
            "ok": True,
            "compacted": True,
            "from_node_id": updated_source.id,
            "from_node_title": updated_source.title,
            "to_node_id": cont_node.id,
            "to_node_title": cont_node.title,
            "context_mode": child_mode,
            "keep_recent_turns": keep_n,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "compression_ratio": compression_ratio,
            "source_summary_version": updated_source.summary_version,
            "handoff_preview": handoff_text[:400],
            "switched_head": switch_head,
        }
        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=ctx.trace_id,
            hop_depth=ctx.hop_depth,
            from_node_id=updated_source.id,
            to_node_id=cont_node.id,
            event_type="spawn",
            payload={
                "compact": True,
                "keep_recent_turns": keep_n,
                "reason": reason,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
            },
            response=out,
            status="ok",
            target_summary_version=updated_source.summary_version,
            target_msg_seq=updated_source.last_msg_seq,
            tokens_in=t_in,
            tokens_out=t_out,
            latency_ms=latency_ms,
        )
        await self.repo.upsert_dependency(
            tree_id=tree.id,
            consumer_node_id=cont_node.id,
            source_node_id=updated_source.id,
            seen_summary_version=updated_source.summary_version,
            seen_msg_seq=updated_source.last_msg_seq,
            last_ipc_event_id=ev.id,
        )
        if self.hooks is not None:
            await self.hooks.emit("branch_compacted", {"tree_id": tree.id, **out})

        return IpcExecutionResult(
            tool_output=out, ipc_event=ev, head_changed=head_changed
        )

    # ------------------------------------------------------------------
    # 7. pin_invariant (Permission-Gated Bounded Tree Invariants)
    # ------------------------------------------------------------------

    async def pin_invariant(
        self,
        tree: TreeModel,
        caller: NodeModel,
        key: str,
        value: str = "",
        action: str = "pin",
        *,
        is_shadow: bool = False,
        trace_ctx: TraceContext | None = None,
    ) -> IpcExecutionResult:
        ctx = trace_ctx or TraceContext.start(
            caller.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()
        if is_shadow:
            return IpcExecutionResult(
                tool_output={
                    "error": "READ_ONLY_SHADOW",
                    "detail": f"Tool '{TOOL_PIN_INVARIANT}' is disabled during a read-only shadow turn.",
                }
            )
        if caller.tool_permission != "read_write":
            detail = (
                f"Branch '{caller.title}' (`{caller.id}`) has '{caller.tool_permission}' permission; "
                "only `read_write` branches may modify tree-wide PINNED invariants."
            )
            ev = await self.repo.create_ipc_event(
                tree_id=tree.id,
                trace_id=ctx.trace_id,
                hop_depth=ctx.hop_depth,
                from_node_id=caller.id,
                to_node_id=caller.id,
                event_type="custom_tool",
                payload={"tool": TOOL_PIN_INVARIANT, "key": key, "action": action},
                response={"error": "PERMISSION_DENIED", "detail": detail},
                status="rejected_permission",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            return IpcExecutionResult(
                tool_output={"error": "PERMISSION_DENIED", "detail": detail},
                ipc_event=ev,
            )

        try:
            if action == "unpin":
                removed = await self.repo.delete_tree_invariant(tree.id, key)
                out: dict[str, Any] = {"ok": True, "action": "unpin", "key": key, "removed": removed}
            else:
                inv = await self.repo.upsert_tree_invariant(
                    tree.id,
                    key,
                    value,
                    source_node_id=caller.id,
                    max_items=self.guardrails.max_pinned_invariants,
                    max_value_chars=self.guardrails.max_pinned_value_chars,
                )
                out = {"ok": True, "action": "pin", "invariant": inv.model_dump()}
        except ValueError as exc:
            return IpcExecutionResult(
                tool_output={"error": "INVARIANT_ERROR", "detail": str(exc)}
            )

        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=ctx.trace_id,
            hop_depth=ctx.hop_depth,
            from_node_id=caller.id,
            to_node_id=caller.id,
            event_type="custom_tool",
            payload={"tool": TOOL_PIN_INVARIANT, "key": key, "action": action},
            response=out,
            status="ok",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        return IpcExecutionResult(tool_output=out, ipc_event=ev)

    # ------------------------------------------------------------------
    # 8. archive_branch (Dead-End Quarantine + Clean Parent Return)
    # ------------------------------------------------------------------

    async def archive_branch(
        self,
        tree: TreeModel,
        caller: NodeModel,
        *,
        target_id: str | None = None,
        reason: str = "",
        switch_to_parent: bool = True,
        is_shadow: bool = False,
        trace_ctx: TraceContext | None = None,
    ) -> IpcExecutionResult:
        """
        Mark a non-root dead-end or scrapped branch as `ARCHIVED`, immediately quarantining
        its chunks, summaries, and Tier-2.5 atomic facts from default active search while
        keeping its verbatim history preserved in SQLite for explicit audit.
        """
        ctx = trace_ctx or TraceContext.start(
            caller.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()
        if is_shadow:
            return IpcExecutionResult(
                tool_output={
                    "error": "READ_ONLY_SHADOW",
                    "detail": f"Tool '{TOOL_ARCHIVE_BRANCH}' is disabled during a read-only shadow turn.",
                }
            )

        tid = (target_id or caller.id).strip()
        target = await self.repo.get_node(tid)
        if target is None or target.tree_id != tree.id:
            return IpcExecutionResult(
                tool_output={"error": "TARGET_NOT_FOUND", "detail": f"Branch '{tid}' not found in tree."}
            )
        if target.parent_id is None:
            return IpcExecutionResult(
                tool_output={
                    "error": "ROOT_PROTECTED",
                    "detail": "Cannot archive the root branch of a dialogue tree.",
                }
            )
        if target.id != caller.id and caller.tool_permission != "read_write":
            detail = (
                f"Branch '{caller.title}' (`{caller.id}`) has '{caller.tool_permission}' permission; "
                "only `read_write` branches may archive a different branch."
            )
            ev = await self.repo.create_ipc_event(
                tree_id=tree.id,
                trace_id=ctx.trace_id,
                hop_depth=ctx.hop_depth,
                from_node_id=caller.id,
                to_node_id=target.id,
                event_type="custom_tool",
                payload={"tool": TOOL_ARCHIVE_BRANCH, "target_id": target.id, "reason": reason},
                response={"error": "PERMISSION_DENIED", "detail": detail},
                status="rejected_permission",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            return IpcExecutionResult(
                tool_output={"error": "PERMISSION_DENIED", "detail": detail},
                ipc_event=ev,
            )

        archived = await self.repo.update_node(target.id, status="ARCHIVED", is_dirty=False)
        assert archived is not None
        # Refresh parent's Tier-3 subtree rollup so archived branch no longer leaks into active rollups
        if archived.parent_id:
            await self.rag.propagate_rollups_upward(archived.id)

        head_changed: tuple[str, str, ReflogEntryModel] | None = None
        switched_to_id: str | None = None
        if switch_to_parent and archived.parent_id and (
            tree.head_node_id == archived.id or caller.id == archived.id
        ):
            parent = await self.repo.get_node(archived.parent_id)
            if parent is not None:
                note_txt = reason.strip() or f"Archived dead-end branch '{archived.title}'"
                cue = (
                    f"Archived dead-end branch {archived.id} ('{archived.title}') — quarantined from "
                    f"active search. Returned HEAD to '{parent.title}'."
                )
                entry = await self.reflog.record_transition(
                    tree_id=tree.id,
                    from_node=archived,
                    to_node=parent,
                    action="return",
                    note=note_txt,
                    orientation_cue=cue,
                    update_head=True,
                )
                head_changed = (archived.id, parent.id, entry)
                switched_to_id = parent.id

        out: dict[str, Any] = {
            "ok": True,
            "archived_node_id": archived.id,
            "archived_node_title": archived.title,
            "status": archived.status,
            "quarantined_from_search": True,
            "reason": reason.strip(),
            "switched_head_to": switched_to_id,
        }
        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=ctx.trace_id,
            hop_depth=ctx.hop_depth,
            from_node_id=caller.id,
            to_node_id=archived.id,
            event_type="custom_tool",
            payload={"tool": TOOL_ARCHIVE_BRANCH, "target_id": archived.id, "reason": reason},
            response=out,
            status="ok",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        if self.hooks is not None:
            await self.hooks.emit("branch_archived", {"tree_id": tree.id, **out})
        return IpcExecutionResult(tool_output=out, ipc_event=ev, head_changed=head_changed)
