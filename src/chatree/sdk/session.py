from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from chatree.config import ToolPermission
from chatree.sdk.node import NodeHandle
from chatree.types import (
    DEFAULT_SESSION_ID,
    ChildContextMode,
    NodeKind,
    ReflogEntryModel,
    TreeSessionModel,
    TurnResult,
)

if TYPE_CHECKING:
    from chatree.sdk.tree import DialogueTree


class TreeSession:
    """
    Stateful Visitor / Cursor handle attached to a `DialogueTree`.

    Each `TreeSession` walks the shared tree with its own independent `HEAD` focus,
    `Back`/`Forward` reflog trail, and orientation cues while sharing all nodes,
    4-tier RAG indexes, artifacts, `PINNED` invariants, and IPC provenance.
    """

    def __init__(self, tree: DialogueTree, session_id: str = DEFAULT_SESSION_ID) -> None:
        self.tree = tree
        self.id = session_id

    async def model(self) -> TreeSessionModel:
        s = await self.tree._ct.repo.get_session(self.tree.id, self.id)
        if s is None:
            raise ValueError(f"Session '{self.id}' not found in tree '{self.tree.id}'")
        return s

    async def head(self) -> NodeHandle:
        return await self.tree.head(session_id=self.id)

    async def chat(
        self,
        content: str,
        *,
        node_id: str | None = None,
        model_override: str | None = None,
        max_hop_depth: int | None = None,
        token_budget: int | None = None,
        include_core_tools: bool = True,
    ) -> TurnResult:
        return await self.tree.chat(
            content,
            node_id=node_id,
            session_id=self.id,
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
        model_override: str | None = None,
        max_hop_depth: int | None = None,
        token_budget: int | None = None,
        include_core_tools: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        async for ev in self.tree.stream(
            content,
            node_id=node_id,
            session_id=self.id,
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
    ) -> NodeHandle:
        return await self.tree.spawn(
            goal,
            parent_id=parent_id,
            title=title,
            mode=mode,
            summary_brief=summary_brief,
            slice_turns=slice_turns,
            autonomous=autonomous,
            model=model,
            permission=permission,
            node_kind=node_kind,
            switch_head=switch_head,
            session_id=self.id,
        )

    async def jump(self, target: str, note: str = "") -> NodeHandle:
        return await self.tree.jump(target, note=note, session_id=self.id)

    async def back(self) -> NodeHandle:
        return await self.tree.back(session_id=self.id)

    async def forward(self) -> NodeHandle:
        return await self.tree.forward(session_id=self.id)

    async def reflog(self, limit: int = 100) -> list[ReflogEntryModel]:
        return await self.tree.reflog(limit=limit, session_id=self.id)

    async def ask(self, query: str, **kwargs: Any) -> dict[str, Any]:
        if "caller_id" not in kwargs or kwargs["caller_id"] is None:
            kwargs["caller_id"] = (await self.head()).id
        return await self.tree.ask(query, **kwargs)

    async def request(
        self, target_id: str, *, prompt: str | None = None, mutate: bool = False
    ) -> dict[str, Any]:
        return await self.tree.request(
            target_id, prompt=prompt, mutate=mutate, caller_id=(await self.head()).id
        )

    async def compact(
        self,
        *,
        node_id: str | None = None,
        keep_recent_turns: int | None = None,
        title: str | None = None,
        reason: str = "",
        switch_head: bool = True,
    ) -> tuple[NodeHandle, dict[str, Any]]:
        return await self.tree.compact(
            node_id=node_id,
            keep_recent_turns=keep_recent_turns,
            title=title,
            reason=reason,
            switch_head=switch_head,
            session_id=self.id,
        )

    async def archive(
        self,
        node_id: str | None = None,
        *,
        reason: str = "",
        switch_to_parent: bool = True,
    ) -> dict[str, Any]:
        return await self.tree.archive(
            node_id=node_id,
            reason=reason,
            switch_to_parent=switch_to_parent,
            session_id=self.id,
        )
