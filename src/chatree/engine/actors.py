from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from chatree.config import GuardrailPolicy, ToolPermission
from chatree.engine.context import ContextAssembler
from chatree.engine.guardrails import TraceContext
from chatree.engine.ipc import IPCRouter
from chatree.engine.rag import TreeRAGService
from chatree.engine.reflog import ReflogService
from chatree.engine.staleness import StalenessService
from chatree.hooks import HookBus
from chatree.protocols import LLMProvider, ModelRouter
from chatree.storage.repository import SQLiteRepository, new_id, utc_now
from chatree.tools import BACKGROUND_ACTOR_BLOCKED_CORE_TOOLS, ToolRegistry
from chatree.types import (
    ArtifactRef,
    BackgroundJobModel,
    ChildContextMode,
    NodeModel,
    TreeModel,
)


class BackgroundActorManager:
    """
    Non-blocking async sub-agent actor runtime.
    Spawns isolated child nodes executing bounded autonomous tasks concurrently in the
    background while the user continues chatting with the parent; completed children
    enqueue their terminal summary + artifacts into the parent's `mailbox` (`status='unread'`)
    for automatic injection on the parent's next turn.
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
        tools: ToolRegistry,
        *,
        hooks: HookBus | None = None,
        guardrails: GuardrailPolicy | None = None,
    ) -> None:
        self.repo = repo
        self.context = context
        self.llm = llm
        self.router = router
        self.ipc = ipc
        self.reflog = reflog
        self.staleness = staleness
        self.rag = rag
        self.tools = tools
        self.hooks = hooks
        self.guardrails = guardrails or GuardrailPolicy()
        self._jobs: dict[str, BackgroundJobModel] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._semaphore = asyncio.Semaphore(self.guardrails.max_concurrent_background_jobs)

    async def hydrate(self) -> None:
        """Load persisted background jobs from SQLite so job history survives restarts."""
        for job in await self.repo.list_background_jobs():
            # Any job marked 'running' in a prior dead process is now interrupted/cancelled
            if job.status == "running" and job.job_id not in self._tasks:
                job.status = "cancelled"
                job.error = "Interrupted by process restart"
                job.completed_at = utc_now()
                await self.repo.upsert_background_job(job)
            self._jobs.setdefault(job.job_id, job)

    def list_jobs(self, tree_id: str | None = None) -> list[BackgroundJobModel]:
        jobs = list(self._jobs.values())
        if tree_id is not None:
            jobs = [j for j in jobs if j.tree_id == tree_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    async def list_jobs_persisted(self, tree_id: str | None = None) -> list[BackgroundJobModel]:
        return await self.repo.list_background_jobs(tree_id=tree_id)

    def get_job(self, job_id: str) -> BackgroundJobModel | None:
        return self._jobs.get(job_id)

    async def spawn_background(
        self,
        tree: TreeModel,
        parent: NodeModel,
        goal: str,
        *,
        title: str | None = None,
        context_mode: ChildContextMode = "clean",
        summary_brief: str = "",
        slice_turns: int = 6,
        model_override: str | None = None,
        tool_permission: ToolPermission | None = None,
        max_rounds: int = 3,
        token_budget: int | None = None,
    ) -> tuple[NodeModel, BackgroundJobModel]:
        short_title = (
            (title or "").strip()
            or " ".join(goal.split()[:5]).rstrip(".,?!")
            or "Background Agent"
        )
        resolved_perm: ToolPermission = (
            tool_permission
            if tool_permission in ("read_only", "read_write")
            else self.guardrails.default_child_permission
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
            node_kind="subagent",
        )

        job = BackgroundJobModel(
            job_id=new_id("job"),
            tree_id=tree.id,
            parent_node_id=parent.id,
            child_node_id=child.id,
            title=child.title,
            goal=goal,
            status="running",
            created_at=utc_now(),
        )
        self._jobs[job.job_id] = job
        await self.repo.upsert_background_job(job)
        # Strict HEAD-only reflog invariant: log background sub-agent spawns in ipc_events, never reflog.
        await self.repo.create_ipc_event(
            tree_id=tree.id,
            trace_id=new_id("tr"),
            hop_depth=0,
            from_node_id=parent.id,
            to_node_id=child.id,
            event_type="spawn",
            payload={
                "goal": goal,
                "title": child.title,
                "context_mode": context_mode,
                "tool_permission": resolved_perm,
                "node_kind": "subagent",
                "background": True,
                "job_id": job.job_id,
            },
            response={
                "child_id": child.id,
                "job_id": job.job_id,
                "status": "running",
                "node_kind": "subagent",
            },
            status="ok",
        )

        if self.hooks is not None:
            await self.hooks.emit("background_started", job.model_dump())

        task = asyncio.create_task(
            self._run_guarded_actor_loop(
                job=job,
                tree=tree,
                parent=parent,
                child=child,
                goal=goal,
                summary_brief=summary_brief,
                max_rounds=max(1, min(max_rounds, 6)),
                token_budget=token_budget or self.guardrails.token_budget,
            ),
            name=f"chatree-bg-{job.job_id}",
        )
        self._tasks[job.job_id] = task
        return child, job

    async def _run_guarded_actor_loop(self, **kwargs: Any) -> None:
        async with self._semaphore:
            await self._run_actor_loop(**kwargs)

    async def _run_actor_loop(
        self,
        *,
        job: BackgroundJobModel,
        tree: TreeModel,
        parent: NodeModel,
        child: NodeModel,
        goal: str,
        summary_brief: str,
        max_rounds: int,
        token_budget: int,
    ) -> None:
        t0 = time.perf_counter()
        trace_ctx = TraceContext.start(
            root_node_id=child.id,
            max_hop_depth=self.guardrails.max_hop_depth,
            token_budget=token_budget,
            min_turn_reserve=self.guardrails.min_turn_reserve,
        )
        try:
            seed_prompt = (
                f"[Autonomous Background Sub-Agent Task]\nGoal: {goal}"
                + (f"\nBrief: {summary_brief}" if summary_brief else "")
            )
            user_msg = await self.repo.append_message(
                node_id=child.id, tree_id=tree.id, role="user", content=seed_prompt
            )
            cur_child = await self.repo.get_node(child.id)
            assert cur_child is not None
            model = self.router.resolve_model(tree, cur_child, "chat")
            # Background actors get read-only IPC + ask_tree + branch-permitted custom tools
            # (never reflog_jump / spawn_child / send, since actor loop owns terminal handoff)
            tool_schemas = [
                t
                for t in self.tools.schemas_for_node(
                    cur_child.tool_permission,
                    include_core=True,
                    include_compound_ask=self.guardrails.enable_compound_ask_tool,
                    is_shadow=False,
                )
                if t["function"]["name"] not in BACKGROUND_ACTOR_BLOCKED_CORE_TOOLS
            ]

            trailing_msgs: list[Any] = []
            final_text = ""

            for rnd in range(max_rounds):
                job.turns_executed = rnd + 1
                cur_child = await self.repo.get_node(child.id) or cur_child
                cur_tree = await self.repo.get_tree(tree.id) or tree
                llm_msgs, _, _ = await self.context.assemble_for_active_turn(
                    cur_tree,
                    cur_child,
                    user_msg,
                    trailing_tool_messages=trailing_msgs or None,
                )
                resp = await self.llm.complete(
                    model=model,
                    messages=llm_msgs,
                    tools=tool_schemas or None,
                    temperature=0.2,
                )
                trace_ctx.ledger.record(resp.tokens_in, resp.tokens_out)
                job.tokens_used = trace_ctx.ledger.tokens_used

                if resp.tool_calls:
                    tc_msg = await self.repo.append_message(
                        node_id=child.id,
                        tree_id=tree.id,
                        role="assistant",
                        content=resp.content,
                        tool_calls=[tc.to_openai_dict() for tc in resp.tool_calls],
                        model_used=resp.model_used,
                        tokens_in=resp.tokens_in,
                        tokens_out=resp.tokens_out,
                        cache_read_tokens=resp.cache_read_tokens,
                        latency_ms=resp.latency_ms,
                    )
                    trailing_msgs.append(tc_msg)
                    batch = await self.ipc.execute_tool_batch(
                        tree=cur_tree,
                        caller=cur_child,
                        tool_calls=resp.tool_calls,
                        trace_ctx=trace_ctx,
                    )
                    for tc, ipc_res in batch:
                        t_msg = await self.repo.append_message(
                            node_id=child.id,
                            tree_id=tree.id,
                            role="tool",
                            content=json.dumps(ipc_res.tool_output),
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                        )
                        trailing_msgs.append(t_msg)
                    continue

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
                final_text = resp.content.strip()
                break

            if not final_text:
                final_text = f"Background task '{goal}' reached max_rounds={max_rounds}."

            # Guard: if parent branch was archived or deleted while worker was running, abort delivery cleanly
            cur_parent = await self.repo.get_node(parent.id)
            if cur_parent is None or cur_parent.status == "ARCHIVED":
                job.status = "cancelled"
                job.error = "Parent branch archived or deleted before completion"
                job.completed_at = utc_now()
                await self.repo.upsert_background_job(job)
                return

            # Commit summary, refresh 4-tier RAG, and enqueue unread mailbox delivery to parent!
            cur_child = await self.repo.get_node(child.id) or child
            done_child = await self.staleness.commit_summary_with_materiality(
                cur_child, final_text, status="COMPLETED"
            )
            await self.rag.index_node_chunks(done_child.id)

            arts = await self.repo.list_artifacts_by_node(done_child.id)
            art_refs = [
                ArtifactRef(
                    id=a.id,
                    slug=a.slug,
                    version=a.version,
                    kind=a.kind,
                    title=a.title,
                    summary=a.summary,
                )
                for a in arts
            ]
            latency_ms = int((time.perf_counter() - t0) * 1000)
            ev = await self.repo.create_ipc_event(
                tree_id=tree.id,
                trace_id=trace_ctx.trace_id,
                hop_depth=trace_ctx.hop_depth,
                from_node_id=done_child.id,
                to_node_id=parent.id,
                event_type="send_finish",
                payload={"background_job_id": job.job_id, "summary": final_text, "finish": True},
                response={"ok": True, "mailbox_status": "unread", "summary_version": done_child.summary_version},
                status="ok",
                target_summary_version=done_child.summary_version,
                target_msg_seq=done_child.last_msg_seq,
                tokens_in=done_child.tokens_in,
                tokens_out=done_child.tokens_out,
                latency_ms=latency_ms,
            )
            mb = await self.repo.create_mailbox_item(
                tree_id=tree.id,
                to_node_id=parent.id,
                from_node_id=done_child.id,
                kind="handoff",
                status="unread",
                summary=final_text,
                artifact_refs=art_refs,
                payload={
                    "sender_title": done_child.title,
                    "background_job_id": job.job_id,
                    "summary_version": done_child.summary_version,
                },
                ipc_event_id=ev.id,
            )
            await self.repo.upsert_dependency(
                tree_id=tree.id,
                consumer_node_id=parent.id,
                source_node_id=done_child.id,
                seen_summary_version=done_child.summary_version,
                seen_msg_seq=done_child.last_msg_seq,
                last_ipc_event_id=ev.id,
            )

            job.status = "completed"
            job.summary = final_text
            job.mailbox_id = mb.id
            job.completed_at = utc_now()
            await self.repo.upsert_background_job(job)
            if self.hooks is not None:
                await self.hooks.emit("background_completed", job.model_dump())

        except asyncio.CancelledError:
            job.status = "cancelled"
            job.completed_at = utc_now()
            try:
                await self.repo.upsert_background_job(job)
            except Exception:
                pass
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = utc_now()
            # Deliver an unread failure notice to the parent mailbox so parent is never blind!
            try:
                fail_mb = await self.repo.create_mailbox_item(
                    tree_id=tree.id,
                    to_node_id=parent.id,
                    from_node_id=child.id,
                    kind="message",
                    status="unread",
                    summary=f"[FAILED] Background task '{goal}' failed: {exc}",
                    artifact_refs=[],
                    payload={
                        "sender_title": child.title,
                        "background_job_id": job.job_id,
                        "error": str(exc),
                        "failed": True,
                    },
                )
                job.mailbox_id = fail_mb.id
                await self.repo.upsert_background_job(job)
            except Exception:
                pass
            if self.hooks is not None:
                await self.hooks.emit("background_failed", job.model_dump())

    async def wait(
        self, job_id: str | None = None, *, timeout: float | None = 30.0
    ) -> list[BackgroundJobModel]:
        if job_id is not None:
            task = self._tasks.get(job_id)
            if task is not None and not task.done():
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            job = self._jobs.get(job_id)
            return [job] if job is not None else []

        pending = [t for t in self._tasks.values() if not t.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
        return list(self._jobs.values())

    async def cancel_for_subtree(
        self,
        tree_id: str,
        node_id: str | None = None,
        *,
        reason: str = "Parent branch archived or deleted",
    ) -> list[str]:
        """Supervisor cascade: cancel all running background sub-agents scoped to `node_id`'s subtree (or entire tree)."""
        affected_node_ids: set[str] | None = None
        if node_id is not None:
            target = await self.repo.get_node(node_id)
            if target is not None:
                all_nodes = await self.repo.list_nodes(tree_id)
                affected_node_ids = {
                    n.id for n in all_nodes if n.id == node_id or n.path.startswith(target.path)
                }
            else:
                affected_node_ids = {node_id}

        cancelled_ids: list[str] = []
        tasks_to_await: list[asyncio.Task[None]] = []
        now = utc_now()
        for job in list(self._jobs.values()):
            if job.tree_id != tree_id or job.status != "running":
                continue
            if affected_node_ids is not None and (
                job.parent_node_id not in affected_node_ids
                and job.child_node_id not in affected_node_ids
            ):
                continue
            job.status = "cancelled"
            job.error = reason
            job.completed_at = now
            cancelled_ids.append(job.job_id)
            task = self._tasks.get(job.job_id)
            if task is not None and not task.done():
                task.cancel()
                tasks_to_await.append(task)
            try:
                await self.repo.upsert_background_job(job)
            except Exception:
                pass
        if tasks_to_await:
            await asyncio.gather(*tasks_to_await, return_exceptions=True)
        return cancelled_ids

    async def cancel_all(self) -> None:
        pending = [t for t in self._tasks.values() if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
