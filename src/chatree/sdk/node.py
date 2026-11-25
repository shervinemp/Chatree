from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chatree.engine.guardrails import TraceContext
from chatree.types import (
    DEFAULT_SESSION_ID,
    ArtifactInput,
    BackgroundJobModel,
    MessageModel,
    NodeModel,
    StaleDependencyInfo,
    TurnResult,
)

if TYPE_CHECKING:
    from chatree.sdk.tree import DialogueTree


class NodeHandle:
    """Ergonomic handle for interacting directly with a specific tree node (bound to a visitor session)."""

    def __init__(
        self, tree: DialogueTree, node_id: str, *, session_id: str = DEFAULT_SESSION_ID
    ) -> None:
        self.tree = tree
        self.id = node_id
        self.session_id = session_id

    async def model(self) -> NodeModel:
        n = await self.tree._ct.repo.get_node(self.id)
        if n is None:
            raise ValueError(f"Node not found: {self.id}")
        return n

    async def promote_to_chat(self) -> NodeModel:
        """Promote a `subagent` worker node into a first-class interactive `chat` branch."""
        updated = await self.tree._ct.repo.update_node(self.id, node_kind="chat")
        if updated is None:
            raise ValueError(f"Node not found: {self.id}")
        return updated

    async def messages(self) -> list[MessageModel]:
        return await self.tree._ct.repo.list_messages(self.id)

    async def stale_dependencies(self) -> list[StaleDependencyInfo]:
        return (await self.model()).stale_dependencies

    async def chat(self, content: str, **kwargs: Any) -> TurnResult:
        kwargs.setdefault("session_id", self.session_id)
        return await self.tree.chat(content, node_id=self.id, **kwargs)

    async def spawn(self, goal: str, **kwargs: Any) -> NodeHandle:
        kwargs.setdefault("session_id", self.session_id)
        return await self.tree.spawn(goal, parent_id=self.id, **kwargs)

    async def spawn_background(
        self, goal: str, **kwargs: Any
    ) -> tuple[NodeHandle, BackgroundJobModel]:
        return await self.tree.spawn_background(goal, parent_id=self.id, **kwargs)

    async def compact(
        self,
        *,
        keep_recent_turns: int | None = None,
        title: str | None = None,
        reason: str = "",
        switch_head: bool = True,
        session_id: str | None = None,
    ) -> tuple[NodeHandle, dict[str, Any]]:
        """Non-destructively compact this branch into a linked continuation child branch."""
        return await self.tree.compact(
            node_id=self.id,
            keep_recent_turns=keep_recent_turns,
            title=title,
            reason=reason,
            switch_head=switch_head,
            session_id=session_id or self.session_id,
        )

    async def archive(
        self,
        *,
        reason: str = "",
        switch_to_parent: bool = True,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Mark this non-root branch ARCHIVED to quarantine its turns/facts from active search."""
        return await self.tree.archive(
            node_id=self.id,
            reason=reason,
            switch_to_parent=switch_to_parent,
            session_id=session_id or self.session_id,
        )

    async def request(
        self, target_id: str, *, prompt: str | None = None, mutate: bool = False
    ) -> dict[str, Any]:
        return await self.tree.request(
            target_id, prompt=prompt, mutate=mutate, caller_id=self.id
        )

    async def finish(
        self,
        *,
        summary: str | None = None,
        artifacts: list[ArtifactInput] | None = None,
        auto_commit: bool | None = None,
        return_to_parent: bool = True,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        node = await self.model()
        if not node.parent_id:
            raise ValueError("Cannot finish the root node into a parent.")
        t = await self.tree.model()
        summary_text = (summary or "").strip()
        if not summary_text:
            trace_ctx = TraceContext.start(
                node.id,
                self.tree._ct.config.guardrails.max_hop_depth,
                self.tree._ct.config.guardrails.token_budget,
            )
            refreshed, *_ = await self.tree._ct.staleness.ensure_fresh_summary(
                t, node, trace_ctx=trace_ctx, force=True
            )
            summary_text = refreshed.summary or f"Completed '{node.title}'."
            node = refreshed

        combined_artifacts = list(artifacts or [])
        if self.tree._ct.sandbox is not None and self.tree._ct.sandbox.get_workspace(node.id) is not None:
            diff_art = await self.tree._ct.sandbox.diff_artifact(
                node.id, title=f"Git Diff ({node.title})"
            )
            if diff_art is not None:
                combined_artifacts.append(diff_art)

        assert node.parent_id is not None
        res = await self.tree._ct.ipc.send(
            tree=t,
            sender=node,
            target_id=node.parent_id,
            content=summary_text,
            artifacts=combined_artifacts,
            finish=True,
            auto_commit_override=auto_commit,
            return_to_parent=return_to_parent,
            session_id=session_id or self.session_id,
        )
        return {
            "result": res.tool_output,
            "handoff_pending": res.handoff_pending,
            "ipc_event": res.ipc_event,
        }
