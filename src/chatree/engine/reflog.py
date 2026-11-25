from __future__ import annotations

import re

from chatree.hooks import HookBus
from chatree.storage.repository import SQLiteRepository
from chatree.types import DEFAULT_SESSION_ID, NodeModel, ReflogAction, ReflogEntryModel


class ReflogService:
    """Git-style append-only per-session navigation ledger + IDE back/forward cursor stepping."""

    def __init__(self, repo: SQLiteRepository, hooks: HookBus | None = None) -> None:
        self.repo = repo
        self.hooks = hooks

    async def record_transition(
        self,
        tree_id: str,
        from_node: NodeModel | None,
        to_node: NodeModel,
        action: ReflogAction,
        note: str = "",
        orientation_cue: str | None = None,
        update_head: bool = True,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> ReflogEntryModel:
        if orientation_cue is None:
            orientation_cue = self._build_default_cue(from_node, to_node, action, note)
        entry = await self.repo.append_reflog(
            tree_id=tree_id,
            from_node_id=from_node.id if from_node else None,
            to_node_id=to_node.id,
            action=action,
            note=note,
            orientation_cue=orientation_cue,
            update_head=update_head,
            session_id=session_id,
        )
        if self.hooks is not None:
            await self.hooks.emit(
                "reflog_jump",
                {
                    "tree_id": tree_id,
                    "session_id": session_id,
                    "action": action,
                    "from_node_id": from_node.id if from_node else None,
                    "to_node_id": to_node.id,
                    "reflog_entry": entry.model_dump(),
                },
            )
        return entry

    async def jump(
        self,
        tree_id: str,
        target: str,
        note: str = "",
        *,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> tuple[NodeModel, ReflogEntryModel | None, int]:
        tree = await self.repo.get_tree(tree_id)
        if tree is None:
            raise ValueError(f"Tree not found: {tree_id}")
        session = await self.repo.get_session(tree_id, session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' not found in tree {tree_id}")

        entries = await self.repo.list_reflog(tree_id, limit=200, session_id=session_id)
        current_head_id = session.head_node_id
        current_head = (
            await self.repo.get_node(current_head_id) if current_head_id else None
        )
        clean_target = target.strip()

        if clean_target in ("back", "forward"):
            if not entries:
                if current_head is None:
                    raise ValueError("Tree has no nodes.")
                return current_head, None, 0
            cur_idx = max(0, min(session.reflog_cursor, len(entries) - 1))
            step = 1 if clean_target == "back" else -1
            new_idx = cur_idx
            while 0 <= new_idx + step < len(entries):
                new_idx += step
                if entries[new_idx].to_node_id != (current_head.id if current_head else None):
                    break
            dest_entry = entries[new_idx]
            dest_node = await self.repo.get_node(dest_entry.to_node_id)
            if dest_node is None:
                raise ValueError(f"Reflog target node no longer exists: {dest_entry.to_node_id}")
            await self.repo.update_session(
                tree_id, session_id, head_node_id=dest_node.id, reflog_cursor=new_idx
            )
            return dest_node, dest_entry, new_idx

        ref_match = re.fullmatch(r"HEAD@\{(\d+)\}", clean_target, flags=re.IGNORECASE)
        if ref_match:
            idx = int(ref_match.group(1))
            if idx < 0 or idx >= len(entries):
                raise ValueError(
                    f"Reflog index {clean_target} out of range (0..{max(0, len(entries) - 1)})"
                )
            dest_id = entries[idx].to_node_id
            default_note = note or f"Jumped via {clean_target}"
        else:
            dest_id = clean_target
            default_note = note or f"Jumped to {dest_id}"

        dest_node = await self.repo.get_node(dest_id)
        if dest_node is None or dest_node.tree_id != tree_id:
            raise ValueError(f"Node not found in tree {tree_id}: {dest_id}")

        if current_head and current_head.id == dest_node.id:
            await self.repo.update_session(
                tree_id, session_id, head_node_id=dest_node.id, reflog_cursor=0
            )
            return dest_node, entries[0] if entries else None, 0

        entry = await self.record_transition(
            tree_id=tree_id,
            from_node=current_head,
            to_node=dest_node,
            action="jump",
            note=default_note,
            update_head=True,
            session_id=session_id,
        )
        return dest_node, entry, 0

    @staticmethod
    def _build_default_cue(
        from_node: NodeModel | None,
        to_node: NodeModel,
        action: ReflogAction,
        note: str,
    ) -> str:
        if action == "init":
            return ""
        if action == "spawn" and from_node:
            return (
                f"Spawned from parent {from_node.id} ({from_node.title}) "
                f"in {to_node.context_mode} mode."
            )
        if action == "return" and from_node:
            ver = f"summary v{from_node.summary_version}" if from_node.summary_version else "no summary"
            suffix = f" Note: {note}" if note else ""
            return f"Returned from child {from_node.id} ({from_node.title}, {ver}).{suffix}"
        if action == "reopen":
            return f"Reopened completed branch {to_node.id} ({to_node.title}) for revision."
        if action == "extract":
            return note or f"Extracted tangent into branch {to_node.id} ({to_node.title})."
        if action == "promote":
            return note or f"Promoted branch {to_node.id} ({to_node.title})."
        if from_node:
            suffix = f" ({note})" if note else ""
            return f"Jumped from {from_node.id} ({from_node.title}) to {to_node.id} ({to_node.title}){suffix}."
        return note or f"Active HEAD moved to {to_node.id} ({to_node.title})."
