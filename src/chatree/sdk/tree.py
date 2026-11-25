from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

from chatree.config import ToolPermission
from chatree.engine.guardrails import TraceContext
from chatree.sandbox import SandboxWorkspace
from chatree.sdk.node import NodeHandle
from chatree.sdk.session import TreeSession
from chatree.types import (
    DEFAULT_SESSION_ID,
    ArtifactInput,
    ArtifactModel,
    BackgroundJobModel,
    BranchComparison,
    BranchFactModel,
    ChildContextMode,
    IpcEventModel,
    MailboxItemModel,
    MessageModel,
    NodeKind,
    NodeModel,
    NodeStatus,
    RagGranularity,
    ReflogEntryModel,
    SearchHit,
    SearchScope,
    SessionKind,
    TreeInvariantModel,
    TreeMetrics,
    TreeModel,
    TreeSessionModel,
    TurnResult,
)

if TYPE_CHECKING:
    from chatree.sdk.client import Chatree


class DialogueTree:
    """High-level ergonomic handle bound to a single Chatree dialogue tree."""

    def __init__(self, client: Chatree, tree_id: str) -> None:
        self._ct = client
        self.id = tree_id

    async def model(self) -> TreeModel:
        t = await self._ct.repo.get_tree(self.id)
        if t is None:
            raise ValueError(f"Tree not found: {self.id}")
        return t

    async def head(self, *, session_id: str = DEFAULT_SESSION_ID) -> NodeHandle:
        sess = await self._ct.repo.get_session(self.id, session_id)
        if sess is not None and sess.head_node_id:
            return NodeHandle(self, sess.head_node_id, session_id=session_id)
        t = await self.model()
        assert t.head_node_id is not None
        return NodeHandle(self, t.head_node_id, session_id=session_id)

    def session(self, session_id: str = DEFAULT_SESSION_ID) -> TreeSession:
        """Return a stateful visitor/cursor handle scoped to `session_id` on this tree."""
        return TreeSession(self, session_id)

    async def create_session(
        self,
        title: str = "New Session",
        *,
        kind: SessionKind = "human",
        start_node_id: str | None = None,
        session_id: str | None = None,
    ) -> TreeSession:
        """Open a new concurrent visitor session (independent HEAD & reflog) on this shared tree."""
        sess, _ = await self._ct.repo.create_session(
            self.id,
            title=title,
            kind=kind,
            start_node_id=start_node_id,
            session_id=session_id,
        )
        return TreeSession(self, sess.id)

    async def sessions(self) -> list[TreeSessionModel]:
        """List all visitor sessions attached to this tree."""
        return await self._ct.repo.list_sessions(self.id)

    async def delete_session(self, session_id: str) -> bool:
        """Close and remove a non-default visitor session and its private reflog trail."""
        return await self._ct.repo.delete_session(self.id, session_id)

    async def root(self) -> NodeHandle:
        t = await self.model()
        assert t.root_node_id is not None
        return NodeHandle(self, t.root_node_id)

    def node(self, node_id: str, *, session_id: str = DEFAULT_SESSION_ID) -> NodeHandle:
        return NodeHandle(self, node_id, session_id=session_id)

    async def nodes(self) -> list[NodeModel]:
        return await self._ct.repo.list_nodes(self.id)

    async def active_chat_nodes(self) -> list[NodeModel]:
        """Return non-archived interactive chat branches (`node_kind == 'chat'`, `status == 'ACTIVE'`)."""
        return [
            n
            for n in await self.nodes()
            if n.node_kind == "chat" and n.status == "ACTIVE"
        ]

    async def subagents(self) -> list[NodeModel]:
        """Return all spawned sub-agent worker nodes (`node_kind == 'subagent'`) in this tree."""
        return [n for n in await self.nodes() if n.node_kind == "subagent"]

    async def chat(
        self,
        content: str,
        *,
        node_id: str | None = None,
        session_id: str = DEFAULT_SESSION_ID,
        model_override: str | None = None,
        max_hop_depth: int | None = None,
        token_budget: int | None = None,
        include_core_tools: bool = True,
    ) -> TurnResult:
        if node_id is None:
            sess = await self._ct.repo.get_session(self.id, session_id)
            target_id = (sess.head_node_id if sess else None) or (await self.model()).head_node_id
        else:
            target_id = node_id
        assert target_id is not None
        return await self._ct.orchestrator.run_turn(
            target_id,
            content,
            session_id=session_id,
            model_override=model_override,
            max_hop_depth=max_hop_depth,
            token_budget=token_budget,
            include_core_tools=include_core_tools,
        )

    async def stream(
        self,
        content: str,
        *,
        node_id: str | None = None,
        session_id: str = DEFAULT_SESSION_ID,
        model_override: str | None = None,
        max_hop_depth: int | None = None,
        token_budget: int | None = None,
        include_core_tools: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        if node_id is None:
            sess = await self._ct.repo.get_session(self.id, session_id)
            target_id = (sess.head_node_id if sess else None) or (await self.model()).head_node_id
        else:
            target_id = node_id
        assert target_id is not None
        async for ev in self._ct.orchestrator.run_turn_stream(
            target_id,
            content,
            session_id=session_id,
            model_override=model_override,
            max_hop_depth=max_hop_depth,
            token_budget=token_budget,
            include_core_tools=include_core_tools,
        ):
            yield ev

    async def spawn(
        self,
        goal: str,
        *,
        parent_id: str | None = None,
        title: str | None = None,
        mode: ChildContextMode = "fork",
        summary_brief: str = "",
        slice_turns: int = 6,
        autonomous: bool = False,
        model: str | None = None,
        permission: ToolPermission | None = None,
        node_kind: NodeKind | None = None,
        switch_head: bool = True,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> NodeHandle:
        t = await self.model()
        if parent_id is None:
            sess = await self._ct.repo.get_session(self.id, session_id)
            pid = (sess.head_node_id if sess else None) or t.head_node_id or t.root_node_id
        else:
            pid = parent_id
        assert pid is not None
        parent = await self._ct.repo.get_node(pid)
        if parent is None:
            raise ValueError(f"Parent node not found: {pid}")
        res = await self._ct.ipc.spawn_child(
            tree=t,
            parent=parent,
            goal=goal,
            title=title,
            context_mode=mode,
            summary_brief=summary_brief,
            slice_turns=slice_turns,
            autonomous=autonomous,
            model_override=model,
            tool_permission=permission,
            node_kind=node_kind,
            switch_head=switch_head,
            session_id=session_id,
        )
        if res.tool_output.get("error"):
            raise RuntimeError(f"{res.tool_output['error']}: {res.tool_output.get('detail')}")
        child_id = str(res.tool_output["child_id"])
        # Automatically provision an isolated sandbox workspace for writable child branches!
        if (
            self._ct.sandbox is not None
            and res.tool_output.get("tool_permission") == "read_write"
        ):
            await self._ct.sandbox.create_workspace(child_id, base_node_id=parent.id)
        return NodeHandle(self, child_id, session_id=session_id)

    def workspace(self, node_id: str) -> SandboxWorkspace | None:
        if self._ct.sandbox is None:
            return None
        return self._ct.sandbox.get_workspace(node_id)

    async def ask(
        self,
        query: str,
        *,
        scope: SearchScope = "all",
        max_targets: int = 2,
        caller_id: str | None = None,
        exclude_node_ids: set[str] | None = None,
        shadow_prompt: str | None = None,
        deep: bool = False,
        enable_recursive_hop: bool = True,
        include_branch_summary_map: bool = True,
        concise_synthesis: bool = True,
    ) -> dict[str, Any]:
        """
        Single-hop compound Retrieve-then-Ask fusing hybrid search + parallel shadow turns.
        Pass `deep=True` to execute the unified `TreeQueryEngine.deep_ask()` pipeline
        (Hop-1 shadows + Tier-2.5 atomic dated facts + optional Hop-2 drill-down + chronological synthesis).
        """
        t = await self.model()
        cid = caller_id or t.head_node_id or t.root_node_id
        assert cid is not None
        caller = await self._ct.repo.get_node(cid)
        assert caller is not None
        if deep:
            return await self._ct.query_engine.deep_ask(
                tree=t,
                caller=caller,
                query=query,
                scope=scope,
                max_targets=max_targets,
                enable_recursive_hop=enable_recursive_hop,
                include_branch_summary_map=include_branch_summary_map,
                concise_synthesis=concise_synthesis,
            )
        res = await self._ct.ipc.ask_tree(
            tree=t,
            caller=caller,
            query=query,
            scope=scope,
            max_targets=max_targets,
            exclude_node_ids=exclude_node_ids,
            shadow_prompt=shadow_prompt,
        )
        return res.tool_output

    async def request(
        self,
        target_id: str,
        *,
        prompt: str | None = None,
        mutate: bool = False,
        caller_id: str | None = None,
    ) -> dict[str, Any]:
        t = await self.model()
        cid = caller_id or t.head_node_id or t.root_node_id
        assert cid is not None
        caller = await self._ct.repo.get_node(cid)
        assert caller is not None
        # When an external SDK/CLI/MCP caller omits caller_id and queries HEAD/root directly,
        # start the trace from "external" so querying HEAD isn't rejected as a self-cycle
        # (while recursive loops initiated from inside target_id are still blocked).
        trace_ctx = None
        if caller_id is None and cid == target_id:
            trace_ctx = TraceContext.start(
                "external",
                self._ct.config.guardrails.max_hop_depth,
                self._ct.config.guardrails.token_budget,
            )
        res = await self._ct.ipc.request(
            tree=t,
            caller=caller,
            target_id=target_id,
            prompt=prompt,
            mutate=mutate,
            trace_ctx=trace_ctx,
        )
        return res.tool_output

    async def search(
        self,
        query: str,
        *,
        scope: SearchScope = "all",
        node_id: str | None = None,
        top_k: int = 5,
        status_filter: set[NodeStatus] | None = None,
        granularities: set[RagGranularity] | None = None,
    ) -> list[SearchHit]:
        t = await self.model()
        cid = node_id or t.head_node_id
        curr = await self._ct.repo.get_node(cid) if cid else None
        return await self._ct.rag.search(
            tree_id=self.id,
            query=query,
            scope=scope,
            current_node=curr,
            top_k=top_k,
            status_filter=status_filter,
            granularities=granularities,
        )

    async def jump(
        self, target: str, note: str = "", *, session_id: str = DEFAULT_SESSION_ID
    ) -> NodeHandle:
        prev_sess = await self._ct.repo.get_session(self.id, session_id)
        prev_head_id = (
            prev_sess.head_node_id if prev_sess is not None else (await self.model()).head_node_id
        )
        dest, _, _ = await self._ct.reflog.jump(self.id, target, note=note, session_id=session_id)
        if (
            self._ct.config.auto_background_summarize_on_switch
            and prev_head_id
            and prev_head_id != dest.id
        ):
            prev_node = await self._ct.repo.get_node(prev_head_id)
            if prev_node is not None and prev_node.parent_id is not None and prev_node.is_dirty:
                self._ct._schedule_bg_summary(self.id, prev_head_id)
        return NodeHandle(self, dest.id, session_id=session_id)

    def refresh_summary_background(self, node_id: str) -> asyncio.Task[str]:
        """Trigger a non-blocking background Tier-2 summary refresh for `node_id` without closing it."""
        return self._ct._schedule_bg_summary(self.id, node_id)

    async def compact(
        self,
        *,
        node_id: str | None = None,
        keep_recent_turns: int | None = None,
        title: str | None = None,
        reason: str = "",
        switch_head: bool = True,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> tuple[NodeHandle, dict[str, Any]]:
        """
        Non-destructively compact `node_id` (defaults to session HEAD) into a linked continuation child
        branch while preserving the previous branch's verbatim messages & KV cache untouched.
        """
        t = await self.model()
        sess = await self._ct.repo.get_session(self.id, session_id)
        nid = node_id or (sess.head_node_id if sess else None) or t.head_node_id or t.root_node_id
        assert nid is not None
        src = await self._ct.repo.get_node(nid)
        if src is None:
            raise ValueError(f"Node not found: {nid}")
        res = await self._ct.ipc.compact_branch(
            tree=t,
            source=src,
            keep_recent_turns=keep_recent_turns,
            title=title,
            reason=reason,
            switch_head=switch_head,
            session_id=session_id,
        )
        return NodeHandle(
            self, str(res.tool_output["to_node_id"]), session_id=session_id
        ), res.tool_output

    async def archive(
        self,
        node_id: str | None = None,
        *,
        reason: str = "",
        switch_to_parent: bool = True,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> dict[str, Any]:
        """Archive a non-root dead-end branch, quarantining its chunks/facts from default active search."""
        t = await self.model()
        sess = await self._ct.repo.get_session(self.id, session_id)
        sess_head = (sess.head_node_id if sess else None) or t.head_node_id
        nid = node_id or sess_head
        if not nid:
            raise ValueError("No branch node specified or active.")
        caller_id = sess_head or t.root_node_id or nid
        caller = await self._ct.repo.get_node(caller_id)
        if caller is None:
            raise ValueError(f"Node not found: {caller_id}")
        res = await self._ct.ipc.archive_branch(
            tree=t,
            caller=caller,
            target_id=nid,
            reason=reason,
            switch_to_parent=switch_to_parent,
            session_id=session_id,
        )
        return res.tool_output

    async def pin(
        self, key: str, value: str, *, source_node_id: str | None = None
    ) -> TreeInvariantModel:
        """Pin or update a bounded tree-wide invariant surfaced in every branch's `<dialogue_tree_context>`."""
        return await self._ct.repo.upsert_tree_invariant(
            self.id,
            key,
            value,
            source_node_id=source_node_id,
            max_items=self._ct.config.guardrails.max_pinned_invariants,
            max_value_chars=self._ct.config.guardrails.max_pinned_value_chars,
        )

    async def unpin(self, key: str) -> bool:
        """Remove a tree-wide pinned invariant by key."""
        return await self._ct.repo.delete_tree_invariant(self.id, key)

    async def invariants(self) -> list[TreeInvariantModel]:
        """List all pinned tree-wide invariants."""
        return await self._ct.repo.list_tree_invariants(self.id)

    async def facts(
        self,
        *,
        node_id: str | None = None,
        active_only: bool = True,
        exclude_archived: bool = True,
    ) -> list[BranchFactModel]:
        """List Tier-2.5 provenance-linked atomic temporal facts (excluding ARCHIVED dead ends by default)."""
        return await self._ct.repo.list_branch_facts(
            self.id,
            node_id=node_id,
            active_only=active_only,
            exclude_archived=exclude_archived,
        )

    async def add_facts(
        self, node_id: str, facts: list[dict[str, Any]], *, source_msg_seq: int | None = None
    ) -> list[BranchFactModel]:
        """Store and index Tier-2.5 atomic temporal facts scoped strictly to `node_id`."""
        created = await self._ct.repo.replace_branch_facts(
            self.id, node_id, facts, source_msg_seq=source_msg_seq
        )
        await self._ct.rag.index_node_facts(node_id)
        return created

    async def back(self, *, session_id: str = DEFAULT_SESSION_ID) -> NodeHandle:
        return await self.jump("back", session_id=session_id)

    async def forward(self, *, session_id: str = DEFAULT_SESSION_ID) -> NodeHandle:
        return await self.jump("forward", session_id=session_id)

    async def reflog(
        self, limit: int = 100, *, session_id: str | None = DEFAULT_SESSION_ID
    ) -> list[ReflogEntryModel]:
        return await self._ct.repo.list_reflog(self.id, limit=limit, session_id=session_id)

    async def ipc_log(self, limit: int = 100) -> list[IpcEventModel]:
        return await self._ct.repo.list_ipc_events(self.id, limit=limit)

    async def artifacts(self) -> list[ArtifactModel]:
        return await self._ct.repo.list_artifacts(self.id)

    async def add_artifact(
        self,
        slug: str,
        title: str,
        content: str,
        *,
        kind: str = "markdown",
        summary: str = "",
        created_by_node_id: str | None = None,
    ) -> ArtifactModel:
        return await self._ct.repo.create_artifact(
            self.id,
            ArtifactInput(slug=slug, title=title, kind=kind, content=content, summary=summary),
            created_by_node_id=created_by_node_id,
        )

    async def pending_handoffs(self) -> list[MailboxItemModel]:
        return await self._ct.repo.list_mailbox_items(self.id, status="pending_review")

    async def commit_handoff(
        self,
        mailbox_id: str,
        *,
        summary: str | None = None,
        artifacts: list[ArtifactInput] | None = None,
        return_head_to_target: bool = True,
    ) -> tuple[MailboxItemModel, NodeModel]:
        mb, target, _ = await self._ct.ipc.commit_pending_handoff(
            mailbox_id=mailbox_id,
            edited_summary=summary,
            edited_artifacts=artifacts,
            return_head_to_target=return_head_to_target,
        )
        return mb, target

    # ------------------------------------------------------------------
    # Branch ergonomics & compilation
    # ------------------------------------------------------------------

    async def extract_to_child(
        self,
        parent_node_id: str,
        seq_start: int,
        seq_end: int,
        *,
        title: str,
        goal: str = "",
        summary: str | None = None,
    ) -> tuple[NodeHandle, NodeHandle]:
        parent, child = await self._ct.ops.extract_to_child(
            parent_node_id, seq_start, seq_end, title=title, goal=goal, summary=summary
        )
        return NodeHandle(self, parent.id), NodeHandle(self, child.id)

    async def fork_on_edit(
        self,
        node_id: str,
        at_seq: int,
        *,
        title: str | None = None,
        switch_head: bool = True,
    ) -> NodeHandle:
        forked = await self._ct.ops.fork_on_edit(
            node_id, at_seq, title=title, switch_head=switch_head
        )
        return NodeHandle(self, forked.id)

    async def compare_siblings(
        self, parent_id: str, sibling_ids: list[str] | None = None
    ) -> BranchComparison:
        return await self._ct.ops.compare_siblings(parent_id, sibling_ids=sibling_ids)

    async def cherry_pick(
        self, source_node_id: str, target_node_id: str, excerpt: str, *, note: str = ""
    ) -> MessageModel:
        return await self._ct.ops.cherry_pick(source_node_id, target_node_id, excerpt, note=note)

    async def promote_child(
        self, child_id: str, *, archive_siblings: bool = True, switch_head_to_parent: bool = True
    ) -> NodeHandle:
        child_node = await self._ct.repo.get_node(child_id)
        if child_node is None or not child_node.parent_id:
            raise ValueError("Only a non-root child node can be promoted.")
        parent_before = await self._ct.repo.get_node(child_node.parent_id)
        if parent_before is None:
            raise ValueError(f"Parent node not found: {child_node.parent_id}")

        # Dual-commit atomicity: merge sandbox FIRST and abort WITHOUT deleting worktree on conflict!
        merged_sandbox = False
        if self._ct.sandbox is not None and self._ct.sandbox.get_workspace(child_id) is not None:
            parent_target = (
                parent_before.id
                if self._ct.sandbox.get_workspace(parent_before.id) is not None
                else None
            )
            merge_res = await self._ct.sandbox.merge_into(child_id, target_node_id=parent_target)
            if not merge_res.get("merged"):
                ws = self._ct.sandbox.get_workspace(child_id)
                raise RuntimeError(
                    f"Sandbox merge failed for branch '{child_id}': "
                    f"{merge_res.get('error') or merge_res.get('reason')}. "
                    f"Worktree and git branch preserved at {ws.path if ws else '(unknown)'}."
                )
            merged_sandbox = True

        parent = await self._ct.ops.promote_child(
            child_id,
            archive_siblings=archive_siblings,
            switch_head_to_parent=switch_head_to_parent,
        )

        # Only clean up worktrees AFTER both sandbox merge AND dialogue promotion succeeded!
        if self._ct.sandbox is not None:
            if merged_sandbox:
                await self._ct.sandbox.cleanup(child_id)
            if archive_siblings:
                for sib in await self._ct.repo.get_children(parent.id):
                    if sib.id != child_id and self._ct.sandbox.get_workspace(sib.id) is not None:
                        await self._ct.sandbox.cleanup(sib.id)
        return NodeHandle(self, parent.id)

    async def compile_markdown(
        self,
        *,
        leaf_node_id: str | None = None,
        include_transcript: bool = True,
        include_side_branches: bool = True,
        include_artifacts: bool = True,
    ) -> str:
        return await self._ct.ops.compile_markdown(
            self.id,
            leaf_node_id=leaf_node_id,
            include_transcript=include_transcript,
            include_side_branches=include_side_branches,
            include_artifacts=include_artifacts,
        )

    async def reindex_rag(self) -> dict[str, int]:
        return await self._ct.rag.index_tree(self.id)

    async def spawn_background(
        self,
        goal: str,
        *,
        parent_id: str | None = None,
        title: str | None = None,
        mode: ChildContextMode = "clean",
        summary_brief: str = "",
        slice_turns: int = 6,
        model: str | None = None,
        permission: ToolPermission | None = None,
        max_rounds: int = 3,
        token_budget: int | None = None,
    ) -> tuple[NodeHandle, BackgroundJobModel]:
        """Spawn a non-blocking async background sub-agent that delivers its result to parent's mailbox."""
        t = await self.model()
        pid = parent_id or t.head_node_id or t.root_node_id
        assert pid is not None
        parent = await self._ct.repo.get_node(pid)
        if parent is None:
            raise ValueError(f"Parent node not found: {pid}")
        child, job = await self._ct.actors.spawn_background(
            tree=t,
            parent=parent,
            goal=goal,
            title=title,
            context_mode=mode,
            summary_brief=summary_brief,
            slice_turns=slice_turns,
            model_override=model,
            tool_permission=permission,
            max_rounds=max_rounds,
            token_budget=token_budget,
        )
        return NodeHandle(self, child.id), job

    def background_jobs(self) -> list[BackgroundJobModel]:
        return self._ct.actors.list_jobs(tree_id=self.id)

    async def wait_background(
        self, job_id: str | None = None, *, timeout: float | None = 30.0
    ) -> list[BackgroundJobModel]:
        return await self._ct.actors.wait(job_id=job_id, timeout=timeout)

    async def unread_mailbox(self, node_id: str | None = None) -> list[MailboxItemModel]:
        return await self._ct.repo.list_mailbox_items(
            self.id, to_node_id=node_id, status="unread"
        )

    async def export_directory(self, output_dir: str | Path) -> Path:
        return await self._ct.export_directory(self.id, output_dir)

    async def export_sft_examples(
        self, *, only_tool_turns: bool = False, include_context_envelope: bool = True
    ) -> list[dict[str, Any]]:
        from chatree.eval.trajectories import TrajectoryExporter

        return await TrajectoryExporter.export_sft_examples(
            self._ct.repo,
            self._ct.context,
            self.id,
            only_tool_turns=only_tool_turns,
            include_context_envelope=include_context_envelope,
            tools_registry=self._ct.tools,
        )

    async def export_sft_jsonl(
        self, output_path: str | Path, *, only_tool_turns: bool = False
    ) -> Path:
        from chatree.eval.trajectories import TrajectoryExporter

        examples = await self.export_sft_examples(only_tool_turns=only_tool_turns)
        return TrajectoryExporter.save_jsonl(examples, output_path)

    async def metrics(self) -> TreeMetrics:
        nodes = await self._ct.repo.list_nodes(self.id)
        ipcs = await self._ct.repo.list_ipc_events(self.id, limit=10_000)
        reflog = await self._ct.repo.list_reflog(self.id, limit=10_000)
        arts = await self._ct.repo.list_artifacts(self.id)
        total_msgs = await self._ct.repo.count_tree_messages(self.id)
        emb_count = await self._ct.repo.count_embeddings(self.id)

        t_in = sum(n.tokens_in for n in nodes)
        t_out = sum(n.tokens_out for n in nodes)
        t_cache = sum(n.cache_read_tokens for n in nodes)
        stale_edges = sum(len(n.stale_dependencies) for n in nodes)

        ipc_by_type: dict[str, int] = {}
        shadow_turns = 0
        shadow_cache_hits = 0
        cached_hits = 0
        blocks = 0
        for e in ipcs:
            ipc_by_type[e.event_type] = ipc_by_type.get(e.event_type, 0) + 1
            if e.event_type == "request_shadow" and e.status == "ok":
                if e.response.get("cache_hit") is True:
                    shadow_cache_hits += 1
                else:
                    shadow_turns += 1
            if e.event_type == "request_summary" and e.response.get("cache_hit") is True:
                cached_hits += 1
            if e.status.startswith("rejected_"):
                blocks += 1

        return TreeMetrics(
            tree_id=self.id,
            node_count=len(nodes),
            active_nodes=sum(1 for n in nodes if n.status == "ACTIVE"),
            completed_nodes=sum(1 for n in nodes if n.status == "COMPLETED"),
            max_depth=max((n.depth for n in nodes), default=0),
            total_messages=total_msgs,
            total_tokens_in=t_in,
            total_tokens_out=t_out,
            total_cache_read_tokens=t_cache,
            cache_hit_ratio=round(t_cache / t_in, 4) if t_in > 0 else 0.0,
            ipc_total=len(ipcs),
            ipc_by_type=ipc_by_type,
            shadow_turns=shadow_turns,
            shadow_cache_hits=shadow_cache_hits,
            cached_summary_hits=cached_hits,
            guardrail_blocks=blocks,
            stale_edges_count=stale_edges,
            reflog_jumps=sum(1 for r in reflog if r.action in ("jump", "return", "reopen")),
            artifact_count=len(arts),
            indexed_embeddings=emb_count,
        )
