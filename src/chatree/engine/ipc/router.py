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


from chatree.engine.ipc.branching import IpcBranchingMixin
from chatree.engine.ipc.lifecycle import IpcLifecycleMixin
from chatree.engine.ipc.shadow import IpcShadowMixin
from chatree.engine.ipc.types import IpcExecutionResult


class IPCRouter(IpcBranchingMixin, IpcShadowMixin, IpcLifecycleMixin):
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

