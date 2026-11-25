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


@dataclass
class IpcExecutionResult:
    tool_output: dict[str, Any]
    ipc_event: IpcEventModel | None = None
    head_changed: tuple[str, str, ReflogEntryModel] | None = None
    handoff_pending: MailboxItemModel | None = None
    branch_suggested: dict[str, Any] | None = None
    stale_node_ids: list[str] = field(default_factory=list)


class IPCRouter:
    """
    Executes the 5 core tree IPC primitives + custom domain tools with:
      - Deterministic cycle, hop-depth, and shared token-budget guardrails
      - Parallel `asyncio.gather` fan-out for concurrent read-only `request()` / `search_tree()` calls
      - Per-branch read_only / read_write external tool permission scoping
      - 3-tier Hybrid Tree RAG indexing & search
      - Lifecycle HookBus emissions
    """

    def __init__(
        self,
        repo: SQLiteRepository,
        context: ContextAssembler,
        llm: LLMProvider,
        router: ModelRouter,
        reflog: ReflogService,
        staleness: StalenessService,
        rag: TreeRAGService,
        tool_registry: ToolRegistry,
        *,
        hooks: HookBus | None = None,
        guardrails: GuardrailPolicy | None = None,
        ablations: AblationConfig | None = None,
        sandbox: BranchSandbox | None = None,
        on_branch_switch_away: Callable[[str, str], None] | None = None,
    ) -> None:
        self.repo = repo
        self.context = context
        self.llm = llm
        self.router = router
        self.reflog = reflog
        self.staleness = staleness
        self.rag = rag
        self.tools = tool_registry
        self.hooks = hooks
        self.guardrails = guardrails or GuardrailPolicy()
        self.ablations = ablations or AblationConfig()
        self.sandbox = sandbox
        self.on_branch_switch_away = on_branch_switch_away
        self._node_last_llm_at: dict[str, float] = {}

    def mark_node_llm_warm(self, node_id: str) -> None:
        """Record real provider KV-cache warm timestamp on actual LLM call completion."""
        self._node_last_llm_at[node_id] = time.monotonic()

    def get_node_cache_age_seconds(self, node: NodeModel) -> float | None:
        last_mono = self._node_last_llm_at.get(node.id)
        if last_mono is not None:
            return max(0.0, time.monotonic() - last_mono)
        try:
            return max(
                0.0,
                (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(node.updated_at)
                ).total_seconds(),
            )
        except Exception:
            return None

    def _resolve_model(
        self,
        tree: TreeModel,
        node: NodeModel,
        role: Any,
        *,
        context_tokens: int | None = None,
        cache_age_seconds: float | None = None,
    ) -> str:
        sig = inspect.signature(self.router.resolve_model)
        if "context_tokens" in sig.parameters:
            return self.router.resolve_model(
                tree,
                node,
                role,
                context_tokens=context_tokens,
                cache_age_seconds=cache_age_seconds,
            )  # type: ignore[call-arg]
        return self.router.resolve_model(tree, node, role)

    def _is_parallel_safe(self, tc: LLMToolCall) -> bool:
        if tc.name in (TOOL_SEARCH_TREE, TOOL_ASK_TREE):
            return True
        if tc.name == TOOL_REQUEST:
            return not bool(tc.arguments.get("mutate", False))
        spec = self.tools.get(tc.name)
        if spec is not None and spec.side_effect == "read_only":
            return True
        return False

    async def execute_tool_batch(
        self,
        tree: TreeModel,
        caller: NodeModel,
        tool_calls: list[LLMToolCall],
        trace_ctx: TraceContext,
    ) -> list[tuple[LLMToolCall, IpcExecutionResult]]:
        """
        Execute a batch of tool calls from one assistant turn.
        When `ablations.enable_parallel_ipc=True` and multiple read-only calls appear
        consecutively (e.g., fanning out `request()` across 3 sibling branches), they run
        concurrently via `asyncio.gather` while stateful calls stay strictly serialized.
        """
        if not tool_calls:
            return []

        if not self.ablations.enable_parallel_ipc or len(tool_calls) == 1:
            results: list[tuple[LLMToolCall, IpcExecutionResult]] = []
            for tc in tool_calls:
                cur_node = await self.repo.get_node(caller.id) or caller
                cur_tree = await self.repo.get_tree(tree.id) or tree
                res = await self.execute_tool(cur_tree, cur_node, tc.name, tc.arguments, trace_ctx)
                results.append((tc, res))
            return results

        # Partition into consecutive runs of parallel-safe calls vs. serialized stateful calls
        out: list[tuple[LLMToolCall, IpcExecutionResult]] = []
        idx = 0
        while idx < len(tool_calls):
            if self._is_parallel_safe(tool_calls[idx]):
                run: list[LLMToolCall] = []
                while idx < len(tool_calls) and self._is_parallel_safe(tool_calls[idx]):
                    run.append(tool_calls[idx])
                    idx += 1
                cur_node = await self.repo.get_node(caller.id) or caller
                cur_tree = await self.repo.get_tree(tree.id) or tree
                if len(run) > 1:
                    batch_res = await asyncio.gather(
                        *(
                            self.execute_tool(
                                cur_tree, cur_node, tc.name, tc.arguments, trace_ctx
                            )
                            for tc in run
                        )
                    )
                    out.extend(zip(run, batch_res))
                else:
                    res = await self.execute_tool(
                        cur_tree, cur_node, run[0].name, run[0].arguments, trace_ctx
                    )
                    out.append((run[0], res))
            else:
                tc = tool_calls[idx]
                idx += 1
                cur_node = await self.repo.get_node(caller.id) or caller
                cur_tree = await self.repo.get_tree(tree.id) or tree
                res = await self.execute_tool(cur_tree, cur_node, tc.name, tc.arguments, trace_ctx)
                out.append((tc, res))
        return out

    async def execute_tool(
        self,
        tree: TreeModel,
        caller: NodeModel,
        tool_name: str,
        arguments: dict[str, Any],
        trace_ctx: TraceContext,
        *,
        is_shadow: bool = False,
    ) -> IpcExecutionResult:
        if self.hooks is not None:
            await self.hooks.emit(
                "ipc_before",
                {
                    "tree_id": tree.id,
                    "caller_id": caller.id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "trace_id": trace_ctx.trace_id,
                    "hop_depth": trace_ctx.hop_depth,
                },
            )

        res = await self._dispatch_tool(
            tree, caller, tool_name, arguments, trace_ctx, is_shadow=is_shadow
        )

        if self.hooks is not None:
            await self.hooks.emit(
                "ipc_after",
                {
                    "tree_id": tree.id,
                    "caller_id": caller.id,
                    "tool_name": tool_name,
                    "trace_id": trace_ctx.trace_id,
                    "ipc_event": res.ipc_event.model_dump() if res.ipc_event else None,
                    "output": res.tool_output,
                },
            )
        return res

    async def _dispatch_tool(
        self,
        tree: TreeModel,
        caller: NodeModel,
        tool_name: str,
        arguments: dict[str, Any],
        trace_ctx: TraceContext,
        *,
        is_shadow: bool = False,
    ) -> IpcExecutionResult:
        if tool_name == TOOL_SPAWN_CHILD:
            perm = arguments.get("permission")
            if perm not in ("read_only", "read_write"):
                perm = None
            return await self.spawn_child(
                tree=tree,
                parent=caller,
                goal=str(arguments.get("goal") or "Explore tangent"),
                title=arguments.get("title"),
                context_mode=arguments.get("context_mode", "fork"),
                summary_brief=str(arguments.get("summary_brief") or ""),
                slice_turns=int(arguments.get("slice_turns") or 6),
                autonomous=bool(arguments.get("autonomous", False)),
                model_override=arguments.get("model"),
                tool_permission=perm,
                initiated_by_model=True,
                trace_ctx=trace_ctx,
            )

        if tool_name == TOOL_ASK_TREE:
            return await self.ask_tree(
                tree=tree,
                caller=caller,
                query=str(arguments.get("query") or ""),
                scope=arguments.get("scope", "all"),
                max_targets=int(arguments.get("max_targets") or 2),
                trace_ctx=trace_ctx,
            )

        if tool_name == TOOL_SEND:
            raw_artifacts = arguments.get("artifacts") or []
            parsed_arts = [ArtifactInput(**a) for a in raw_artifacts if isinstance(a, dict)]
            return await self.send(
                tree=tree,
                sender=caller,
                target_id=str(arguments.get("target_id") or caller.parent_id or ""),
                content=str(arguments.get("content") or ""),
                artifacts=parsed_arts,
                finish=bool(arguments.get("finish", False)),
                trace_ctx=trace_ctx,
            )

        if tool_name == TOOL_REQUEST:
            prompt_val = arguments.get("prompt")
            if isinstance(prompt_val, str) and not prompt_val.strip():
                prompt_val = None
            return await self.request(
                tree=tree,
                caller=caller,
                target_id=str(arguments.get("target_id") or ""),
                prompt=prompt_val,
                mutate=bool(arguments.get("mutate", False)),
                trace_ctx=trace_ctx,
            )

        if tool_name == TOOL_SEARCH_TREE:
            return await self.search_tree(
                tree=tree,
                caller=caller,
                query=str(arguments.get("query") or ""),
                scope=arguments.get("scope", "all"),
                top_k=int(arguments.get("top_k") or 5),
                trace_ctx=trace_ctx,
            )

        if tool_name == TOOL_REFLOG_JUMP:
            follow_val = arguments.get("follow_up_prompt")
            if isinstance(follow_val, str) and not follow_val.strip():
                follow_val = None
            return await self.reflog_jump(
                tree=tree,
                caller=caller,
                target=str(arguments.get("target") or ""),
                note=str(arguments.get("note") or ""),
                follow_up_prompt=follow_val,
                trace_ctx=trace_ctx,
            )

        if tool_name == TOOL_COMPACT_BRANCH:
            raw_keep = arguments.get("keep_recent_turns")
            keep_val = int(raw_keep) if raw_keep is not None else None
            return await self.compact_branch(
                tree=tree,
                source=caller,
                keep_recent_turns=keep_val,
                title=arguments.get("title"),
                reason=str(arguments.get("reason") or ""),
                trace_ctx=trace_ctx,
            )

        if tool_name == TOOL_PIN_INVARIANT:
            return await self.pin_invariant(
                tree=tree,
                caller=caller,
                key=str(arguments.get("key") or ""),
                value=str(arguments.get("value") or ""),
                action=str(arguments.get("action") or "pin"),
                is_shadow=is_shadow,
                trace_ctx=trace_ctx,
            )

        if tool_name == TOOL_ARCHIVE_BRANCH:
            raw_tid = arguments.get("target_id")
            tid = str(raw_tid).strip() if raw_tid else None
            return await self.archive_branch(
                tree=tree,
                caller=caller,
                target_id=tid,
                reason=str(arguments.get("reason") or ""),
                switch_to_parent=bool(arguments.get("switch_to_parent", True)),
                is_shadow=is_shadow,
                trace_ctx=trace_ctx,
            )

        # Custom user-registered domain tool with deterministic per-branch permission scoping
        if tool_name not in CORE_TOOL_NAMES:
            return await self._execute_custom_tool(
                tree=tree,
                caller=caller,
                tool_name=tool_name,
                arguments=arguments,
                trace_ctx=trace_ctx,
                is_shadow=is_shadow,
            )

        return IpcExecutionResult(
            tool_output={"error": "UNKNOWN_TOOL", "detail": f"Unsupported tool: {tool_name}"}
        )

    async def _execute_custom_tool(
        self,
        tree: TreeModel,
        caller: NodeModel,
        tool_name: str,
        arguments: dict[str, Any],
        trace_ctx: TraceContext,
        *,
        is_shadow: bool = False,
    ) -> IpcExecutionResult:
        t0 = time.perf_counter()
        ws = self.sandbox.get_workspace(caller.id) if self.sandbox is not None else None
        tctx = ToolCallContext(
            tree=tree,
            node=caller,
            trace_id=trace_ctx.trace_id,
            hop_depth=trace_ctx.hop_depth,
            is_shadow=is_shadow,
            workspace_path=ws.path if ws is not None else None,
        )
        out = await self.tools.execute_custom(tool_name, arguments, tctx)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        status = (
            "rejected_permission"
            if out.get("error") == "PERMISSION_DENIED"
            else ("error" if out.get("error") else "ok")
        )
        ev = await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=trace_ctx.trace_id,
            hop_depth=trace_ctx.hop_depth,
            from_node_id=caller.id,
            to_node_id=caller.id,
            event_type="custom_tool",
            payload={"tool": tool_name, "arguments": arguments, "permission": caller.tool_permission},
            response=out,
            status=status,
            latency_ms=latency_ms,
        )
        if self.hooks is not None:
            await self.hooks.emit(
                "custom_tool",
                {
                    "tree_id": tree.id,
                    "node_id": caller.id,
                    "tool_name": tool_name,
                    "status": status,
                    "output": out,
                },
            )
        return IpcExecutionResult(tool_output=out, ipc_event=ev)

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
    ) -> IpcExecutionResult:
        """
        Fuses `search_tree` + parallel `asyncio.gather` shadow `request()` calls into one
        single round trip, avoiding the serial `search_tree -> parent LLM -> request` latency tax.
        """
        ctx = trace_ctx or TraceContext.start(
            caller.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()
        capped_targets = max(1, min(max_targets, 10))
        excluded = set(exclude_node_ids or ())
        excluded.add(caller.id)
        hits = await self.rag.search(
            tree_id=tree.id,
            query=query,
            scope=scope,
            current_node=caller,
            top_k=max(12, (capped_targets + len(excluded)) * 2),
            max_per_node=2,
        )
        # Pick top distinct target nodes excluding caller/excluded, with causal freshness filtering:
        # if candidate A has a stale dependency on candidate B (another hit), suppress stale A so
        # authoritative revised branches always take precedence over outdated consumers.
        raw_candidates: list[str] = []
        for h in hits:
            if h.node_id not in excluded and h.node_id not in raw_candidates:
                raw_candidates.append(h.node_id)

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
            out = {
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
