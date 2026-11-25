from __future__ import annotations

import re
from collections import Counter

from chatree.engine.rag import TreeRAGService
from chatree.engine.reflog import ReflogService
from chatree.engine.staleness import StalenessService
from chatree.hooks import HookBus
from chatree.storage.repository import SQLiteRepository, new_id, utc_now
from chatree.types import (
    BranchComparison,
    MessageModel,
    NodeModel,
    SiblingDiffEntry,
)

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "with", "this", "from", "have", "are", "was", "were",
        "been", "will", "would", "could", "should", "into", "about", "using", "used", "use",
        "branch", "summary", "concise", "points", "conclusions", "user", "assistant", "node",
        "mode", "demo", "mock", "recorded", "noted", "key",
    }
)


def _extract_terms(text: str) -> Counter[str]:
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if w.lower() not in _STOPWORDS
    ]
    return Counter(words)


class BranchOperations:
    """
    Daily-driver branch ergonomics implemented directly in the core library:
      - Retroactive tangent extraction (`extract_to_child`)
      - Fork-on-edit / retry immutability (`fork_on_edit`)
      - Side-by-side sibling comparison (`compare_siblings`)
      - Cherry-pick conclusion (`cherry_pick`)
      - Promote / fast-forward winning child (`promote_child`)
      - Compile-to-document Markdown ADR export (`compile_markdown`)
    """

    def __init__(
        self,
        repo: SQLiteRepository,
        reflog: ReflogService,
        staleness: StalenessService,
        rag: TreeRAGService,
        hooks: HookBus | None = None,
    ) -> None:
        self.repo = repo
        self.reflog = reflog
        self.staleness = staleness
        self.rag = rag
        self.hooks = hooks

    async def extract_to_child(
        self,
        parent_node_id: str,
        seq_start: int,
        seq_end: int,
        *,
        title: str,
        goal: str = "",
        summary: str | None = None,
    ) -> tuple[NodeModel, NodeModel]:
        """
        Retroactively refactor turns `seq_start..seq_end` out of `parent_node_id` into a new
        child branch and collapse them in the parent into a single summary stub.
        Returns `(updated_parent, extracted_child)`.
        """
        parent = await self.repo.get_node(parent_node_id)
        if parent is None:
            raise ValueError(f"Node not found: {parent_node_id}")
        tree = await self.repo.get_tree(parent.tree_id)
        assert tree is not None

        messages = await self.repo.list_messages(parent.id)
        if not messages:
            raise ValueError("Cannot extract turns from an empty node.")
        if seq_start < 1 or seq_end < seq_start or seq_end > messages[-1].seq:
            raise ValueError(
                f"Invalid turn range {seq_start}..{seq_end} (valid 1..{messages[-1].seq})."
            )

        before = [m for m in messages if m.seq < seq_start]
        extracted = [m for m in messages if seq_start <= m.seq <= seq_end]
        after = [m for m in messages if m.seq > seq_end]
        if not extracted:
            raise ValueError("No messages found in the requested range.")

        # Spawn child forked immediately before the extracted tangent
        child = await self.repo.create_child_node(
            parent=parent,
            title=title,
            goal=goal or title,
            context_mode="fork",
            summary_brief=f"Extracted turns {seq_start}..{seq_end} from '{parent.title}'.",
            spawn_msg_seq=max(0, seq_start - 1),
        )

        for m in extracted:
            await self.repo.append_message(
                node_id=child.id,
                tree_id=tree.id,
                role=m.role,
                content=m.content,
                tool_calls=m.tool_calls,
                tool_call_id=m.tool_call_id,
                tool_name=m.tool_name,
                model_used=m.model_used,
                tokens_in=m.tokens_in,
                tokens_out=m.tokens_out,
                cache_read_tokens=m.cache_read_tokens,
                latency_ms=m.latency_ms,
                metadata={**m.metadata, "extracted_from_seq": m.seq},
            )

        child_fresh = await self.repo.get_node(child.id)
        assert child_fresh is not None
        if summary and summary.strip():
            child_done = await self.repo.commit_node_summary(
                child.id, summary=summary.strip(), status="COMPLETED"
            )
        else:
            child_done, *_ = await self.staleness.ensure_fresh_summary(
                tree, child_fresh, force=True, new_status="COMPLETED"
            )

        stub_msg = MessageModel(
            id=new_id("m"),
            node_id=parent.id,
            tree_id=tree.id,
            seq=seq_start,
            role="user",
            content=(
                f"[Extracted tangent -> child `{child_done.id}` ({child_done.title}) · "
                f"summary v{child_done.summary_version}]\n\n{child_done.summary or ''}"
            ),
            metadata={
                "ipc_extracted_stub": True,
                "child_node_id": child_done.id,
                "original_seq_range": [seq_start, seq_end],
            },
            created_at=utc_now(),
        )
        compacted = [*before, stub_msg, *after]
        await self.repo.replace_node_messages_for_extraction(parent.id, tree.id, compacted)
        await self.repo.upsert_dependency(
            tree_id=tree.id,
            consumer_node_id=parent.id,
            source_node_id=child_done.id,
            seen_summary_version=child_done.summary_version,
            seen_msg_seq=child_done.last_msg_seq,
        )

        await self.rag.index_node_chunks(child_done.id)
        await self.rag.index_node_summary(child_done.id)
        await self.rag.index_node_chunks(parent.id)
        await self.rag.propagate_rollups_upward(child_done.id)

        updated_parent = await self.repo.get_node(parent.id)
        assert updated_parent is not None
        if self.hooks is not None:
            await self.hooks.emit(
                "branch_extracted",
                {
                    "tree_id": tree.id,
                    "parent_id": parent.id,
                    "child_id": child_done.id,
                    "seq_range": [seq_start, seq_end],
                },
            )
        return updated_parent, child_done

    async def fork_on_edit(
        self,
        node_id: str,
        at_seq: int,
        *,
        title: str | None = None,
        switch_head: bool = True,
    ) -> NodeModel:
        """
        Immutably branch off at `at_seq - 1` instead of overwriting history when editing an
        earlier prompt or retrying an assistant generation.
        """
        node = await self.repo.get_node(node_id)
        if node is None:
            raise ValueError(f"Node not found: {node_id}")
        if at_seq < 1 or at_seq > node.last_msg_seq + 1:
            raise ValueError(f"Invalid seq {at_seq} for node '{node.id}' (last_msg_seq={node.last_msg_seq}).")

        # Fork from `node` itself at `at_seq - 1` so original messages remain 100% append-only
        branch_title = title or f"{node.title} (edit @{at_seq})"
        forked = await self.repo.create_child_node(
            parent=node,
            title=branch_title,
            goal=node.goal or branch_title,
            context_mode="fork",
            summary_brief=f"Immutable fork-on-edit of '{node.title}' before turn #{at_seq}.",
            spawn_msg_seq=max(0, at_seq - 1),
            model_override=node.model_override,
            tool_permission=node.tool_permission,
        )
        if switch_head:
            await self.reflog.record_transition(
                tree_id=node.tree_id,
                from_node=node,
                to_node=forked,
                action="fork_edit",
                note=f"Forked '{node.title}' before turn #{at_seq} into '{forked.title}'",
                update_head=True,
            )
        return forked

    async def compare_siblings(
        self,
        parent_id: str,
        sibling_ids: list[str] | None = None,
    ) -> BranchComparison:
        parent = await self.repo.get_node(parent_id)
        if parent is None:
            raise ValueError(f"Parent node not found: {parent_id}")
        tree = await self.repo.get_tree(parent.tree_id)
        assert tree is not None

        children = await self.repo.get_children(parent_id)
        if sibling_ids:
            wanted = set(sibling_ids)
            children = [c for c in children if c.id in wanted]

        entries: list[SiblingDiffEntry] = []
        term_sets: dict[str, set[str]] = {}

        for ch in children:
            fresh_ch, *_ = await self.staleness.ensure_fresh_summary(tree, ch)
            arts = await self.repo.list_artifacts_by_node(fresh_ch.id)
            summary_txt = fresh_ch.summary or fresh_ch.goal or ""
            entries.append(
                SiblingDiffEntry(
                    node_id=fresh_ch.id,
                    title=fresh_ch.title,
                    status=fresh_ch.status,
                    summary_version=fresh_ch.summary_version,
                    summary=summary_txt,
                    artifact_refs=[f"{a.slug}@v{a.version}" for a in arts],
                    tokens_total=fresh_ch.tokens_in + fresh_ch.tokens_out,
                    message_count=fresh_ch.last_msg_seq,
                )
            )
            term_sets[fresh_ch.id] = set(_extract_terms(summary_txt).keys())

        if len(term_sets) >= 2:
            shared = sorted(set.intersection(*term_sets.values()))[:15]
        else:
            shared = []
        shared_set = set(shared)

        distinctive: dict[str, list[str]] = {}
        for nid, tset in term_sets.items():
            other_union: set[str] = set()
            for oid, oset in term_sets.items():
                if oid != nid:
                    other_union |= oset
            distinctive[nid] = sorted((tset - other_union) - shared_set)[:12]

        return BranchComparison(
            parent_id=parent.id,
            parent_title=parent.title,
            siblings=entries,
            shared_terms=shared,
            distinctive_terms_by_node=distinctive,
        )

    async def cherry_pick(
        self,
        source_node_id: str,
        target_node_id: str,
        excerpt: str,
        *,
        note: str = "",
    ) -> MessageModel:
        source = await self.repo.get_node(source_node_id)
        target = await self.repo.get_node(target_node_id)
        if source is None or target is None or source.tree_id != target.tree_id:
            raise ValueError("Source and target nodes must exist in the same tree.")

        header = f"[Cherry-picked from `{source.id}` ({source.title}) · v{source.summary_version}]"
        body = f"{header}\n" + (f"Note: {note}\n\n" if note else "\n") + excerpt.strip()
        msg = await self.repo.append_message(
            node_id=target.id,
            tree_id=target.tree_id,
            role="user",
            content=body,
            metadata={"ipc_cherry_pick": True, "source_node_id": source.id},
        )
        await self.repo.upsert_dependency(
            tree_id=target.tree_id,
            consumer_node_id=target.id,
            source_node_id=source.id,
            seen_summary_version=source.summary_version,
            seen_msg_seq=source.last_msg_seq,
        )
        await self.rag.index_node_chunks(target.id)
        return msg

    async def promote_child(
        self,
        child_id: str,
        *,
        archive_siblings: bool = True,
        switch_head_to_parent: bool = True,
    ) -> NodeModel:
        """
        Fast-forward / promote a winning child branch onto its parent as the canonical
        conclusion, optionally archiving competing siblings.
        """
        child = await self.repo.get_node(child_id)
        if child is None or not child.parent_id:
            raise ValueError("Only a non-root child node can be promoted.")
        parent = await self.repo.get_node(child.parent_id)
        tree = await self.repo.get_tree(child.tree_id)
        assert parent is not None and tree is not None

        fresh_child, *_ = await self.staleness.ensure_fresh_summary(
            tree, child, new_status="COMPLETED"
        )
        arts = await self.repo.list_artifacts_by_node(fresh_child.id)
        art_lines = (
            "\n\n**Promoted Artifacts:**\n"
            + "\n".join(f"- `{a.slug}@v{a.version}`: {a.title}" for a in arts)
            if arts
            else ""
        )

        promoted_text = (
            f"[Promoted Winning Branch `{fresh_child.id}` ({fresh_child.title}) · "
            f"summary v{fresh_child.summary_version}]\n\n"
            f"{fresh_child.summary or ''}{art_lines}"
        )
        await self.repo.append_message(
            node_id=parent.id,
            tree_id=tree.id,
            role="user",
            content=promoted_text,
            metadata={"ipc_promoted_branch": True, "promoted_node_id": fresh_child.id},
        )
        await self.repo.upsert_dependency(
            tree_id=tree.id,
            consumer_node_id=parent.id,
            source_node_id=fresh_child.id,
            seen_summary_version=fresh_child.summary_version,
            seen_msg_seq=fresh_child.last_msg_seq,
        )

        archived_ids: list[str] = []
        if archive_siblings:
            for sib in await self.repo.get_children(parent.id):
                if sib.id != fresh_child.id and sib.status != "ARCHIVED":
                    await self.repo.update_node(sib.id, status="ARCHIVED")
                    archived_ids.append(sib.id)

        await self.rag.index_node_chunks(parent.id)
        await self.rag.propagate_rollups_upward(fresh_child.id)

        if switch_head_to_parent:
            await self.reflog.record_transition(
                tree_id=tree.id,
                from_node=fresh_child,
                to_node=parent,
                action="promote",
                note=f"Promoted winning branch '{fresh_child.title}' into '{parent.title}'",
                update_head=True,
            )

        updated_parent = await self.repo.get_node(parent.id)
        assert updated_parent is not None
        if self.hooks is not None:
            await self.hooks.emit(
                "branch_promoted",
                {
                    "tree_id": tree.id,
                    "parent_id": parent.id,
                    "promoted_child_id": fresh_child.id,
                    "archived_sibling_ids": archived_ids,
                },
            )
        return updated_parent

    async def compile_markdown(
        self,
        tree_id: str,
        *,
        leaf_node_id: str | None = None,
        include_transcript: bool = True,
        include_side_branches: bool = True,
        include_artifacts: bool = True,
    ) -> str:
        """
        Walk the active root-to-leaf path, fold in resolved child summaries and versioned
        artifacts, and emit a clean, shareable Markdown ADR / design document.
        """
        tree = await self.repo.get_tree(tree_id)
        if tree is None:
            raise ValueError(f"Tree not found: {tree_id}")
        target_id = leaf_node_id or tree.head_node_id or tree.root_node_id
        if not target_id:
            return f"# {tree.title}\n\n*(Empty dialogue tree)*\n"

        leaf = await self.repo.get_node(target_id)
        if leaf is None:
            raise ValueError(f"Leaf node not found: {target_id}")

        ancestors = await self.repo.get_ancestors(leaf)
        active_path = [*ancestors, leaf]
        active_ids = {n.id for n in active_path}

        lines: list[str] = [
            f"# {tree.title}",
            "",
            f"- **Tree ID:** `{tree.id}`",
            f"- **Active Path:** {' → '.join(f'{n.title} (`{n.id}`)' for n in active_path)}",
            f"- **Default Model:** `{tree.default_model}`",
            "",
        ]

        # Executive Summary from root/leaf summaries
        lines.append("## Executive Path Summary")
        lines.append("")
        for node in active_path:
            status_tag = f"`{node.status}` · `v{node.summary_version}`"
            lines.append(f"### {node.title} (`{node.id}` · {status_tag})")
            if node.goal:
                lines.append(f"**Goal:** {node.goal}")
                lines.append("")
            if node.summary:
                lines.append(node.summary)
                lines.append("")
            elif node.rollup_summary:
                lines.append(node.rollup_summary)
                lines.append("")

            if include_side_branches:
                side_children = [
                    c for c in await self.repo.get_children(node.id) if c.id not in active_ids
                ]
                if side_children:
                    lines.append("#### Resolved Side Branches")
                    lines.append("")
                    for sc in side_children:
                        body = sc.summary or sc.goal or "(No summary yet)"
                        lines.append(
                            f"- **{sc.title}** (`{sc.id}`, `{sc.status}`, `v{sc.summary_version}`): {body}"
                        )
                    lines.append("")

            if include_transcript:
                msgs = [
                    m
                    for m in await self.repo.list_messages(node.id)
                    if m.role in ("user", "assistant") and m.content.strip()
                ]
                if msgs:
                    lines.append("#### Key Dialogue Turns")
                    lines.append("")
                    for m in msgs:
                        speaker = "User" if m.role == "user" else "Assistant"
                        lines.append(f"- **{speaker} (#{m.seq}):** {m.content.strip()}")
                    lines.append("")

        if include_artifacts:
            arts = await self.repo.list_artifacts(tree.id)
            if arts:
                lines.append("## Versioned Artifacts Blackboard")
                lines.append("")
                for art in arts:
                    lines.append(f"### `{art.slug}@v{art.version}` — {art.title} ({art.kind})")
                    if art.summary:
                        lines.append(f"_{art.summary}_")
                    lines.append("")
                    lang = art.kind if art.kind != "markdown" else "markdown"
                    lines.append(f"```{lang}")
                    lines.append(art.content.rstrip())
                    lines.append("```")
                    lines.append("")

        return "\n".join(lines).rstrip() + "\n"
