from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chatree.engine.rag import TreeRAGService
from chatree.storage.repository import SQLiteRepository
from chatree.types import (
    ArtifactModel,
    BranchFactModel,
    IpcEventModel,
    MailboxItemModel,
    MessageModel,
    NodeModel,
    ReflogEntryModel,
    TreeInvariantModel,
    TreeModel,
    TreeSessionModel,
)

_KIND_EXT: dict[str, str] = {
    "markdown": "md",
    "sql": "sql",
    "json": "json",
    "diff": "diff",
    "code": "txt",
    "python": "py",
    "typescript": "ts",
}


def _safe_slug(slug: str) -> str:
    return slug.replace("/", "__").replace("\\", "__").replace(":", "_")


class DialogueDirectoryStore:
    """
    Portable, human-inspectable, git-greppable `.dialogue/` directory format:
      <dir>/
        tree.json
        reflog.jsonl
        ipc_events.jsonl
        dependencies.jsonl
        nodes/<node_id>.json
        nodes/<node_id>.jsonl
        nodes/<node_id>.md
        artifacts/<slug>@v<N>.meta.json
        artifacts/<slug>@v<N>.<ext>
    Supports lossless round-trip export and import across SQLite databases.
    """

    @staticmethod
    async def export_tree(
        repo: SQLiteRepository,
        tree_id: str,
        output_dir: str | Path,
    ) -> Path:
        tree = await repo.get_tree(tree_id)
        if tree is None:
            raise ValueError(f"Tree not found: {tree_id}")

        out = Path(output_dir).resolve()
        nodes_dir = out / "nodes"
        arts_dir = out / "artifacts"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        arts_dir.mkdir(parents=True, exist_ok=True)

        nodes = await repo.list_nodes(tree_id)
        sessions = await repo.list_sessions(tree_id)
        reflog = await repo.list_reflog(tree_id, limit=100_000, session_id=None)
        ipc_events = await repo.list_ipc_events(tree_id, limit=100_000)
        artifacts = await repo.list_artifacts(tree_id)
        mailbox_items = await repo.list_mailbox_items(tree_id)
        deps = await repo.list_dependencies(tree_id)
        facts = await repo.list_branch_facts(tree_id, active_only=False, exclude_archived=False)
        invariants = await repo.list_tree_invariants(tree_id)

        (out / "tree.json").write_text(
            json.dumps(
                {
                    "format_version": 3,
                    "tree": tree.model_dump(),
                    "node_ids": [n.id for n in nodes],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Write oldest -> newest JSONL ledgers for clean git diffs
        (out / "reflog.jsonl").write_text(
            "\n".join(json.dumps(r.model_dump()) for r in reversed(reflog))
            + ("\n" if reflog else ""),
            encoding="utf-8",
        )
        (out / "ipc_events.jsonl").write_text(
            "\n".join(json.dumps(e.model_dump()) for e in reversed(ipc_events))
            + ("\n" if ipc_events else ""),
            encoding="utf-8",
        )
        (out / "dependencies.jsonl").write_text(
            "\n".join(json.dumps(d) for d in deps) + ("\n" if deps else ""),
            encoding="utf-8",
        )
        (out / "mailbox.jsonl").write_text(
            "\n".join(json.dumps(mb.model_dump()) for mb in reversed(mailbox_items))
            + ("\n" if mailbox_items else ""),
            encoding="utf-8",
        )
        (out / "facts.jsonl").write_text(
            "\n".join(json.dumps(f.model_dump()) for f in facts) + ("\n" if facts else ""),
            encoding="utf-8",
        )
        (out / "invariants.json").write_text(
            json.dumps([inv.model_dump() for inv in invariants], indent=2) + "\n",
            encoding="utf-8",
        )
        (out / "sessions.json").write_text(
            json.dumps([s.model_dump() for s in sessions], indent=2) + "\n",
            encoding="utf-8",
        )

        for node in nodes:
            msgs = await repo.list_messages(node.id)
            (nodes_dir / f"{node.id}.json").write_text(
                json.dumps(node.model_dump(), indent=2), encoding="utf-8"
            )
            (nodes_dir / f"{node.id}.jsonl").write_text(
                "\n".join(json.dumps(m.model_dump()) for m in msgs)
                + ("\n" if msgs else ""),
                encoding="utf-8",
            )
            # Human-readable Markdown mirror for effortless `rg` / `grep`
            md_lines = [
                f"# {node.title} (`{node.id}`)",
                "",
                f"- **Path:** `{node.path}`",
                f"- **Mode:** `{node.context_mode}` | **Kind:** `{node.node_kind}` | **Status:** `{node.status}` | **Permission:** `{node.tool_permission}`",
                f"- **Summary Version:** `v{node.summary_version}.{node.summary_minor_version}`",
            ]
            if node.goal:
                md_lines.append(f"- **Goal:** {node.goal}")
            if node.summary:
                md_lines.extend(["", "## Summary", "", node.summary])
            if msgs:
                md_lines.extend(["", "## Messages", ""])
                for m in msgs:
                    md_lines.append(f"### #{m.seq} · {m.role}")
                    md_lines.append("")
                    md_lines.append(m.content or "(tool call)")
                    md_lines.append("")
            (nodes_dir / f"{node.id}.md").write_text(
                "\n".join(md_lines).rstrip() + "\n", encoding="utf-8"
            )

        for art in artifacts:
            base = f"{_safe_slug(art.slug)}@v{art.version}"
            ext = _KIND_EXT.get(art.kind, "txt")
            (arts_dir / f"{base}.meta.json").write_text(
                json.dumps(art.model_dump(exclude={"content"}), indent=2), encoding="utf-8"
            )
            (arts_dir / f"{base}.{ext}").write_text(art.content, encoding="utf-8")

        return out

    @staticmethod
    async def import_tree(
        repo: SQLiteRepository,
        input_dir: str | Path,
        *,
        rag: TreeRAGService | None = None,
        overwrite: bool = True,
    ) -> TreeModel:
        in_dir = Path(input_dir).resolve()
        meta_path = in_dir / "tree.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Not a valid .dialogue directory (missing {meta_path})")

        bundle: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        tree = TreeModel(**bundle["tree"])

        existing = await repo.get_tree(tree.id)
        if existing is not None:
            if not overwrite:
                raise ValueError(f"Tree '{tree.id}' already exists (pass overwrite=True to replace).")
            await repo.delete_tree(tree.id)

        async with repo.db.write_transaction():
            await DialogueDirectoryStore._insert_bundle(repo, tree, in_dir, bundle)
        repo.invalidate_vector_cache(tree.id)

        if rag is not None:
            await rag.index_tree(tree.id)

        restored = await repo.get_tree(tree.id)
        assert restored is not None
        return restored

    @staticmethod
    async def _insert_bundle(
        repo: SQLiteRepository,
        tree: TreeModel,
        in_dir: Path,
        bundle: dict[str, Any],
    ) -> None:
        await repo.conn.execute(
            """
            INSERT INTO trees (
                id, title, root_node_id, head_node_id,
                default_model, shadow_model, summary_model,
                auto_commit_handoffs, reflog_cursor, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tree.id,
                tree.title,
                tree.root_node_id,
                tree.head_node_id,
                tree.default_model,
                tree.shadow_model,
                tree.summary_model,
                1 if tree.auto_commit_handoffs else 0,
                tree.reflog_cursor,
                tree.created_at,
                tree.updated_at,
            ),
        )

        # Load nodes ordered by depth ASC so foreign-key parent_id constraints always succeed
        nodes_dir = in_dir / "nodes"
        loaded_nodes: list[NodeModel] = []
        for nid in bundle.get("node_ids", []):
            npath = nodes_dir / f"{nid}.json"
            if npath.exists():
                loaded_nodes.append(NodeModel(**json.loads(npath.read_text(encoding="utf-8"))))
        loaded_nodes.sort(key=lambda n: (n.depth, n.created_at))

        for n in loaded_nodes:
            await repo.conn.execute(
                """
                INSERT INTO nodes (
                    id, tree_id, parent_id, path, depth, title, goal, summary_brief,
                    context_mode, spawn_msg_seq, slice_turn_count, status, node_kind, tool_permission,
                    model_override, summary, summary_version, summary_minor_version,
                    last_change_severity, last_change_delta, rollup_summary, is_dirty,
                    last_msg_seq, tokens_in, tokens_out, cache_read_tokens, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    n.id,
                    n.tree_id,
                    n.parent_id,
                    n.path,
                    n.depth,
                    n.title,
                    n.goal,
                    n.summary_brief,
                    n.context_mode,
                    n.spawn_msg_seq,
                    n.slice_turn_count,
                    n.status,
                    n.node_kind,
                    n.tool_permission,
                    n.model_override,
                    n.summary,
                    n.summary_version,
                    n.summary_minor_version,
                    n.last_change_severity,
                    n.last_change_delta,
                    n.rollup_summary,
                    1 if n.is_dirty else 0,
                    n.last_msg_seq,
                    n.tokens_in,
                    n.tokens_out,
                    n.cache_read_tokens,
                    n.created_at,
                    n.updated_at,
                ),
            )
            await repo._upsert_node_fts(
                n.id, n.tree_id, n.title, n.goal, n.summary or n.summary_brief, n.rollup_summary or ""
            )

            msg_path = nodes_dir / f"{n.id}.jsonl"
            if msg_path.exists():
                for line in msg_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    m = MessageModel(**json.loads(line))
                    await repo.conn.execute(
                        """
                        INSERT INTO messages (
                            id, node_id, tree_id, seq, role, content,
                            tool_calls_json, tool_call_id, tool_name, model_used,
                            tokens_in, tokens_out, cache_read_tokens, latency_ms,
                            metadata_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            m.id,
                            m.node_id,
                            m.tree_id,
                            m.seq,
                            m.role,
                            m.content,
                            json.dumps(m.tool_calls) if m.tool_calls is not None else None,
                            m.tool_call_id,
                            m.tool_name,
                            m.model_used,
                            m.tokens_in,
                            m.tokens_out,
                            m.cache_read_tokens,
                            m.latency_ms,
                            json.dumps(m.metadata) if m.metadata else None,
                            m.created_at,
                        ),
                    )
                    if m.content.strip():
                        await repo.conn.execute(
                            """
                            INSERT INTO messages_fts (message_id, tree_id, node_id, seq, content)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (m.id, m.tree_id, m.node_id, str(m.seq), m.content),
                        )

        # Restore visitor sessions (or backfill default session for pre-session exports)
        sess_path = in_dir / "sessions.json"
        if sess_path.exists():
            for raw_s in json.loads(sess_path.read_text(encoding="utf-8")):
                sess = TreeSessionModel(**raw_s)
                await repo.conn.execute(
                    """
                    INSERT OR REPLACE INTO tree_sessions (
                        tree_id, id, title, kind, head_node_id, reflog_cursor,
                        is_default, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sess.tree_id,
                        sess.id,
                        sess.title,
                        sess.kind,
                        sess.head_node_id,
                        sess.reflog_cursor,
                        1 if sess.is_default else 0,
                        sess.created_at,
                        sess.updated_at,
                    ),
                )
        else:
            await repo.conn.execute(
                """
                INSERT OR IGNORE INTO tree_sessions (
                    tree_id, id, title, kind, head_node_id, reflog_cursor,
                    is_default, created_at, updated_at
                ) VALUES (?, 'default', 'Main Session', 'human', ?, ?, 1, ?, ?)
                """,
                (
                    tree.id,
                    tree.head_node_id,
                    tree.reflog_cursor,
                    tree.created_at,
                    tree.updated_at,
                ),
            )

        # Restore reflog
        rf_path = in_dir / "reflog.jsonl"
        if rf_path.exists():
            for line in rf_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rf = ReflogEntryModel(**json.loads(line))
                await repo.conn.execute(
                    """
                    INSERT INTO reflog (
                        id, tree_id, session_id, seq, from_node_id, to_node_id, action,
                        note, orientation_cue, cue_consumed, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rf.id,
                        rf.tree_id,
                        rf.session_id,
                        rf.seq,
                        rf.from_node_id,
                        rf.to_node_id,
                        rf.action,
                        rf.note,
                        rf.orientation_cue,
                        1 if rf.cue_consumed else 0,
                        rf.created_at,
                    ),
                )

        # Restore IPC events
        ipc_path = in_dir / "ipc_events.jsonl"
        if ipc_path.exists():
            for line in ipc_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = IpcEventModel(**json.loads(line))
                await repo.conn.execute(
                    """
                    INSERT INTO ipc_events (
                        id, tree_id, trace_id, hop_depth, from_node_id, to_node_id,
                        event_type, payload_json, response_json, status,
                        target_summary_version, target_msg_seq,
                        tokens_in, tokens_out, latency_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ev.id,
                        ev.tree_id,
                        ev.trace_id,
                        ev.hop_depth,
                        ev.from_node_id,
                        ev.to_node_id,
                        ev.event_type,
                        json.dumps(ev.payload),
                        json.dumps(ev.response),
                        ev.status,
                        ev.target_summary_version,
                        ev.target_msg_seq,
                        ev.tokens_in,
                        ev.tokens_out,
                        ev.latency_ms,
                        ev.created_at,
                    ),
                )

        # Restore dependencies
        dep_path = in_dir / "dependencies.jsonl"
        if dep_path.exists():
            for line in dep_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                await repo.conn.execute(
                    """
                    INSERT INTO node_dependencies (
                        consumer_node_id, source_node_id, tree_id,
                        seen_summary_version, seen_msg_seq, artifact_slug,
                        last_ipc_event_id, dismissed_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        d["consumer_node_id"],
                        d["source_node_id"],
                        d["tree_id"],
                        d["seen_summary_version"],
                        d["seen_msg_seq"],
                        d.get("artifact_slug"),
                        d.get("last_ipc_event_id"),
                        d.get("dismissed_at"),
                        d["updated_at"],
                    ),
                )

        # Restore mailbox items (pending_review handoffs, unread background deliveries, etc.)
        mb_path = in_dir / "mailbox.jsonl"
        if mb_path.exists():
            for line in mb_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                mb = MailboxItemModel(**json.loads(line))
                await repo.conn.execute(
                    """
                    INSERT INTO mailbox (
                        id, tree_id, to_node_id, from_node_id, ipc_event_id,
                        kind, status, summary, artifact_refs_json, payload_json,
                        created_at, delivered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mb.id,
                        mb.tree_id,
                        mb.to_node_id,
                        mb.from_node_id,
                        mb.ipc_event_id,
                        mb.kind,
                        mb.status,
                        mb.summary,
                        json.dumps([r.model_dump() for r in mb.artifact_refs]),
                        json.dumps(mb.payload),
                        mb.created_at,
                        mb.delivered_at,
                    ),
                )

        # Restore artifacts
        arts_dir = in_dir / "artifacts"
        if arts_dir.exists():
            for meta_file in sorted(arts_dir.glob("*.meta.json")):
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                base = meta_file.name.removesuffix(".meta.json")
                ext = _KIND_EXT.get(meta.get("kind", "markdown"), "txt")
                content_file = arts_dir / f"{base}.{ext}"
                content = content_file.read_text(encoding="utf-8") if content_file.exists() else ""
                art = ArtifactModel(content=content, **meta)
                await repo.conn.execute(
                    """
                    INSERT INTO artifacts (
                        id, tree_id, slug, version, kind, title, content, summary,
                        created_by_node_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        art.id,
                        art.tree_id,
                        art.slug,
                        art.version,
                        art.kind,
                        art.title,
                        art.content,
                        art.summary,
                        art.created_by_node_id,
                        art.created_at,
                    ),
                )

        # Restore Tier-2.5 atomic temporal facts (lossless round-trip!)
        facts_path = in_dir / "facts.jsonl"
        if facts_path.exists():
            for line in facts_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                fact = BranchFactModel(**json.loads(line))
                await repo.conn.execute(
                    """
                    INSERT INTO branch_facts (
                        id, tree_id, node_id, fact_text, entities_json,
                        resolved_date, source_msg_seq, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact.id,
                        fact.tree_id,
                        fact.node_id,
                        fact.fact_text,
                        json.dumps(fact.entities),
                        fact.resolved_date,
                        fact.source_msg_seq,
                        fact.status,
                        fact.created_at,
                    ),
                )

        # Restore tree-level PINNED invariants (including source version/seq staleness snapshots!)
        inv_path = in_dir / "invariants.json"
        if inv_path.exists():
            raw_invs = json.loads(inv_path.read_text(encoding="utf-8"))
            for item in raw_invs if isinstance(raw_invs, list) else []:
                inv = TreeInvariantModel(**item)
                await repo.conn.execute(
                    """
                    INSERT INTO tree_invariants (
                        tree_id, key, value, source_node_id,
                        source_summary_version, source_msg_seq, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        inv.tree_id,
                        inv.key,
                        inv.value,
                        inv.source_node_id,
                        inv.source_summary_version,
                        inv.source_msg_seq,
                        inv.updated_at,
                    ),
                )
