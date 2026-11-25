from __future__ import annotations

import inspect
import time
from typing import Any

from chatree.engine.guardrails import TraceContext
from chatree.engine.ipc.types import IpcExecutionResult
from chatree.providers.mock import estimate_messages_tokens
from chatree.types import (
    DEFAULT_SESSION_ID,
    NodeModel,
    TreeModel,
)


class IpcNavigationMixin:
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

    async def reflog_jump(
        self,
        tree: TreeModel,
        caller: NodeModel,
        target: str,
        note: str = "",
        follow_up_prompt: str | None = None,
        session_id: str = DEFAULT_SESSION_ID,
        trace_ctx: TraceContext | None = None,
    ) -> IpcExecutionResult:
        ctx = trace_ctx or TraceContext.start(
            caller.id, self.guardrails.max_hop_depth, self.guardrails.token_budget
        )
        t0 = time.perf_counter()
        try:
            dest, entry, _cursor = await self.reflog.jump(
                tree.id, target, note=note, session_id=session_id
            )
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
