from __future__ import annotations

import json
import re
from typing import Any

from chatree.storage.repos.base import (
    _row_to_artifact,
    _row_to_mailbox,
    new_id,
    utc_now,
)
from chatree.types import (
    DEFAULT_SESSION_ID,
    ArtifactInput,
    ArtifactModel,
    ArtifactRef,
    BackgroundJobModel,
    MailboxItemModel,
    MailboxStatus,
    ReflogAction,
    ReflogEntryModel,
    SessionKind,
    TreeSessionModel,
)


def _row_to_session(row: Any) -> TreeSessionModel:
    return TreeSessionModel(
        id=str(row["id"]),
        tree_id=str(row["tree_id"]),
        title=str(row["title"]),
        kind=row["kind"],
        head_node_id=row["head_node_id"],
        reflog_cursor=int(row["reflog_cursor"] or 0),
        is_default=bool(row["is_default"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


class ReflogArtifactsMailboxMixin:
    db: Any
    conn: Any

    # Tree Sessions (Stateful Visitors / Cursors)
    # ------------------------------------------------------------------

    async def create_session(
        self,
        tree_id: str,
        *,
        title: str = "New Session",
        kind: SessionKind = "human",
        start_node_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[TreeSessionModel, ReflogEntryModel]:
        cur = await self.conn.execute(
            "SELECT id, root_node_id, head_node_id FROM trees WHERE id = ?", (tree_id,)
        )
        t_row = await cur.fetchone()
        if t_row is None:
            raise ValueError(f"Tree not found: {tree_id}")

        target_node_id = start_node_id or t_row["root_node_id"] or t_row["head_node_id"]
        if not target_node_id:
            raise ValueError(f"Tree '{tree_id}' has no nodes to attach session to.")

        n_cur = await self.conn.execute(
            "SELECT id, title FROM nodes WHERE id = ? AND tree_id = ?", (target_node_id, tree_id)
        )
        n_row = await n_cur.fetchone()
        if n_row is None:
            raise ValueError(f"Start node '{target_node_id}' not found in tree '{tree_id}'.")

        sid = (session_id or "").strip() or new_id("s")
        clean_title = (title or "").strip() or "New Session"
        now = utc_now()
        rf_id = new_id("rf")

        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO tree_sessions (
                    tree_id, id, title, kind, head_node_id, reflog_cursor,
                    is_default, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    tree_id,
                    sid,
                    clean_title,
                    kind,
                    target_node_id,
                    1 if sid == DEFAULT_SESSION_ID else 0,
                    now,
                    now,
                ),
            )
            seq_cur = await self.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM reflog WHERE tree_id = ?",
                (tree_id,),
            )
            next_seq = (await seq_cur.fetchone())["max_seq"] + 1
            await self.conn.execute(
                """
                INSERT INTO reflog (
                    id, tree_id, session_id, seq, from_node_id, to_node_id, action,
                    note, orientation_cue, cue_consumed, created_at
                ) VALUES (?, ?, ?, ?, NULL, ?, 'init', ?, '', 1, ?)
                """,
                (
                    rf_id,
                    tree_id,
                    sid,
                    next_seq,
                    target_node_id,
                    f"Opened session '{clean_title}' at '{n_row['title']}'",
                    now,
                ),
            )
        sess = await self.get_session(tree_id, sid)
        entries = await self.list_reflog(tree_id, limit=1, session_id=sid)
        assert sess is not None and entries
        return sess, entries[0]

    async def get_session(
        self, tree_id: str, session_id: str = DEFAULT_SESSION_ID
    ) -> TreeSessionModel | None:
        cur = await self.conn.execute(
            "SELECT * FROM tree_sessions WHERE tree_id = ? AND id = ?",
            (tree_id, session_id),
        )
        row = await cur.fetchone()
        if row is not None:
            return _row_to_session(row)
        if session_id == DEFAULT_SESSION_ID:
            t_cur = await self.conn.execute("SELECT * FROM trees WHERE id = ?", (tree_id,))
            t_row = await t_cur.fetchone()
            if t_row is None:
                return None
            async with self.db.write_transaction():
                await self.conn.execute(
                    """
                    INSERT OR IGNORE INTO tree_sessions (
                        tree_id, id, title, kind, head_node_id, reflog_cursor,
                        is_default, created_at, updated_at
                    ) VALUES (?, 'default', 'Main Session', 'human', ?, ?, 1, ?, ?)
                    """,
                    (
                        tree_id,
                        t_row["head_node_id"],
                        t_row["reflog_cursor"],
                        t_row["created_at"],
                        t_row["updated_at"],
                    ),
                )
            cur2 = await self.conn.execute(
                "SELECT * FROM tree_sessions WHERE tree_id = ? AND id = 'default'", (tree_id,)
            )
            row2 = await cur2.fetchone()
            return _row_to_session(row2) if row2 is not None else None
        return None

    async def list_sessions(self, tree_id: str) -> list[TreeSessionModel]:
        # Ensure primary default session exists for legacy/imported trees
        await self.get_session(tree_id, DEFAULT_SESSION_ID)
        cur = await self.conn.execute(
            """
            SELECT * FROM tree_sessions
            WHERE tree_id = ?
            ORDER BY is_default DESC, created_at ASC
            """,
            (tree_id,),
        )
        return [_row_to_session(r) for r in await cur.fetchall()]

    async def update_session(
        self, tree_id: str, session_id: str, **fields: Any
    ) -> TreeSessionModel | None:
        allowed = {"title", "kind", "head_node_id", "reflog_cursor"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return await self.get_session(tree_id, session_id)
        now = utc_now()
        updates["updated_at"] = now
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        async with self.db.write_transaction():
            await self.conn.execute(
                f"UPDATE tree_sessions SET {set_clause} WHERE tree_id = ? AND id = ?",
                [*updates.values(), tree_id, session_id],
            )
            if session_id == DEFAULT_SESSION_ID:
                tree_sync: dict[str, Any] = {}
                if "head_node_id" in updates:
                    tree_sync["head_node_id"] = updates["head_node_id"]
                if "reflog_cursor" in updates:
                    tree_sync["reflog_cursor"] = updates["reflog_cursor"]
                if tree_sync:
                    tree_sync["updated_at"] = now
                    t_set = ", ".join(f"{k} = ?" for k in tree_sync)
                    await self.conn.execute(
                        f"UPDATE trees SET {t_set} WHERE id = ?",
                        [*tree_sync.values(), tree_id],
                    )
        return await self.get_session(tree_id, session_id)

    async def delete_session(self, tree_id: str, session_id: str) -> bool:
        if session_id == DEFAULT_SESSION_ID:
            raise ValueError("Cannot delete the primary 'default' session of a tree.")
        async with self.db.write_transaction():
            await self.conn.execute(
                "DELETE FROM reflog WHERE tree_id = ? AND session_id = ?",
                (tree_id, session_id),
            )
            cur = await self.conn.execute(
                "DELETE FROM tree_sessions WHERE tree_id = ? AND id = ?",
                (tree_id, session_id),
            )
            deleted = bool((cur.rowcount or 0) > 0)
        return deleted

    # Reflog (Session-Scoped Visitor Walk Trail)
    # ------------------------------------------------------------------

    async def append_reflog(
        self,
        tree_id: str,
        from_node_id: str | None,
        to_node_id: str,
        action: ReflogAction,
        note: str = "",
        orientation_cue: str = "",
        update_head: bool = True,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> ReflogEntryModel:
        rf_id = new_id("rf")
        now = utc_now()
        cue_consumed = 0 if orientation_cue else 1

        async with self.db.write_transaction():
            cur = await self.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM reflog WHERE tree_id = ?",
                (tree_id,),
            )
            next_seq = (await cur.fetchone())["max_seq"] + 1
            await self.conn.execute(
                """
                INSERT INTO reflog (
                    id, tree_id, session_id, seq, from_node_id, to_node_id, action,
                    note, orientation_cue, cue_consumed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rf_id,
                    tree_id,
                    session_id,
                    next_seq,
                    from_node_id,
                    to_node_id,
                    action,
                    note,
                    orientation_cue,
                    cue_consumed,
                    now,
                ),
            )
            if update_head:
                await self.conn.execute(
                    """
                    UPDATE tree_sessions
                    SET head_node_id = ?, reflog_cursor = 0, updated_at = ?
                    WHERE tree_id = ? AND id = ?
                    """,
                    (to_node_id, now, tree_id, session_id),
                )
                if session_id == DEFAULT_SESSION_ID:
                    await self.conn.execute(
                        "UPDATE trees SET head_node_id = ?, reflog_cursor = 0, updated_at = ? WHERE id = ?",
                        (to_node_id, now, tree_id),
                    )
        entries = await self.list_reflog(tree_id, limit=1, session_id=session_id)
        return entries[0]

    async def list_reflog(
        self,
        tree_id: str,
        limit: int = 100,
        *,
        session_id: str | None = DEFAULT_SESSION_ID,
    ) -> list[ReflogEntryModel]:
        if session_id is not None:
            cur = await self.conn.execute(
                "SELECT * FROM reflog WHERE tree_id = ? AND session_id = ? ORDER BY seq DESC LIMIT ?",
                (tree_id, session_id, limit),
            )
        else:
            cur = await self.conn.execute(
                "SELECT * FROM reflog WHERE tree_id = ? ORDER BY seq DESC LIMIT ?",
                (tree_id, limit),
            )
        rows = await cur.fetchall()
        return [
            ReflogEntryModel(
                id=r["id"],
                tree_id=r["tree_id"],
                session_id=(
                    str(r["session_id"])
                    if "session_id" in r.keys() and r["session_id"]
                    else DEFAULT_SESSION_ID
                ),
                seq=r["seq"],
                head_ref=f"HEAD@{{{idx}}}",
                from_node_id=r["from_node_id"],
                to_node_id=r["to_node_id"],
                action=r["action"],
                note=r["note"],
                orientation_cue=r["orientation_cue"],
                cue_consumed=bool(r["cue_consumed"]),
                created_at=r["created_at"],
            )
            for idx, r in enumerate(rows)
        ]

    async def get_unconsumed_orientation_cues(
        self,
        tree_id: str,
        node_id: str,
        *,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> list[ReflogEntryModel]:
        cur = await self.conn.execute(
            """
            SELECT * FROM reflog
            WHERE tree_id = ? AND session_id = ? AND to_node_id = ?
              AND cue_consumed = 0 AND orientation_cue != ''
            ORDER BY seq DESC
            LIMIT 3
            """,
            (tree_id, session_id, node_id),
        )
        return [
            ReflogEntryModel(
                id=r["id"],
                tree_id=r["tree_id"],
                session_id=(
                    str(r["session_id"])
                    if "session_id" in r.keys() and r["session_id"]
                    else DEFAULT_SESSION_ID
                ),
                seq=r["seq"],
                head_ref="HEAD@{0}",
                from_node_id=r["from_node_id"],
                to_node_id=r["to_node_id"],
                action=r["action"],
                note=r["note"],
                orientation_cue=r["orientation_cue"],
                cue_consumed=False,
                created_at=r["created_at"],
            )
            for r in await cur.fetchall()
        ]

    async def mark_orientation_cues_consumed(self, reflog_ids: list[str]) -> None:
        if not reflog_ids:
            return
        placeholders = ",".join("?" for _ in reflog_ids)
        async with self.db.write_transaction():
            await self.conn.execute(
                f"UPDATE reflog SET cue_consumed = 1 WHERE id IN ({placeholders})", reflog_ids
            )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def create_artifact(
        self,
        tree_id: str,
        art: ArtifactInput,
        created_by_node_id: str | None = None,
    ) -> ArtifactModel:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", art.slug.strip()).strip("-") or "artifact"
        art_id = new_id("art")
        now = utc_now()
        async with self.db.write_transaction():
            cur = await self.conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_v FROM artifacts WHERE tree_id = ? AND slug = ?",
                (tree_id, slug),
            )
            next_v = (await cur.fetchone())["max_v"] + 1
            await self.conn.execute(
                """
                INSERT INTO artifacts (
                    id, tree_id, slug, version, kind, title, content, summary,
                    created_by_node_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    art_id,
                    tree_id,
                    slug,
                    next_v,
                    art.kind,
                    art.title,
                    art.content,
                    art.summary,
                    created_by_node_id,
                    now,
                ),
            )
        cur_row = await self.conn.execute("SELECT * FROM artifacts WHERE id = ?", (art_id,))
        row = await cur_row.fetchone()
        assert row is not None
        return _row_to_artifact(row)

    async def list_artifacts(self, tree_id: str) -> list[ArtifactModel]:
        cur = await self.conn.execute(
            "SELECT * FROM artifacts WHERE tree_id = ? ORDER BY created_at DESC",
            (tree_id,),
        )
        return [_row_to_artifact(r) for r in await cur.fetchall()]

    async def list_artifacts_by_node(self, node_id: str) -> list[ArtifactModel]:
        cur = await self.conn.execute(
            "SELECT * FROM artifacts WHERE created_by_node_id = ? ORDER BY created_at DESC",
            (node_id,),
        )
        return [_row_to_artifact(r) for r in await cur.fetchall()]

    async def get_artifact(self, artifact_id: str) -> ArtifactModel | None:
        cur = await self.conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        row = await cur.fetchone()
        return _row_to_artifact(row) if row else None

    # ------------------------------------------------------------------
    # Mailbox
    # ------------------------------------------------------------------

    async def create_mailbox_item(
        self,
        tree_id: str,
        to_node_id: str,
        from_node_id: str,
        kind: str,
        status: MailboxStatus,
        summary: str,
        artifact_refs: list[ArtifactRef],
        payload: dict[str, Any] | None = None,
        ipc_event_id: str | None = None,
    ) -> MailboxItemModel:
        mb_id = new_id("mb")
        now = utc_now()
        delivered_at = now if status == "delivered" else None
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO mailbox (
                    id, tree_id, to_node_id, from_node_id, ipc_event_id,
                    kind, status, summary, artifact_refs_json, payload_json,
                    created_at, delivered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mb_id,
                    tree_id,
                    to_node_id,
                    from_node_id,
                    ipc_event_id,
                    kind,
                    status,
                    summary,
                    json.dumps([r.model_dump() for r in artifact_refs]),
                    json.dumps(payload or {}),
                    now,
                    delivered_at,
                ),
            )
        item = await self.get_mailbox_item(mb_id)
        assert item is not None
        return item

    async def get_mailbox_item(self, mailbox_id: str) -> MailboxItemModel | None:
        cur = await self.conn.execute("SELECT * FROM mailbox WHERE id = ?", (mailbox_id,))
        row = await cur.fetchone()
        return _row_to_mailbox(row) if row else None

    async def list_mailbox_items(
        self,
        tree_id: str,
        to_node_id: str | None = None,
        status: MailboxStatus | None = None,
    ) -> list[MailboxItemModel]:
        clauses = ["tree_id = ?"]
        params: list[Any] = [tree_id]
        if to_node_id is not None:
            clauses.append("to_node_id = ?")
            params.append(to_node_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        cur = await self.conn.execute(
            f"SELECT * FROM mailbox WHERE {' AND '.join(clauses)} ORDER BY created_at DESC",
            params,
        )
        return [_row_to_mailbox(r) for r in await cur.fetchall()]

    async def update_mailbox_item(
        self,
        mailbox_id: str,
        status: MailboxStatus | None = None,
        summary: str | None = None,
        artifact_refs: list[ArtifactRef] | None = None,
    ) -> MailboxItemModel | None:
        updates: dict[str, Any] = {}
        if status is not None:
            updates["status"] = status
            if status == "delivered":
                updates["delivered_at"] = utc_now()
        if summary is not None:
            updates["summary"] = summary
        if artifact_refs is not None:
            updates["artifact_refs_json"] = json.dumps([r.model_dump() for r in artifact_refs])
        if not updates:
            return await self.get_mailbox_item(mailbox_id)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        async with self.db.write_transaction():
            await self.conn.execute(
                f"UPDATE mailbox SET {set_clause} WHERE id = ?",
                list(updates.values()) + [mailbox_id],
            )
        return await self.get_mailbox_item(mailbox_id)

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Runtime Settings
    # ------------------------------------------------------------------

    async def get_all_settings(self) -> dict[str, Any]:
        cur = await self.conn.execute("SELECT key, value_json FROM app_settings")
        return {r["key"]: json.loads(r["value_json"]) for r in await cur.fetchall()}

    async def set_settings(self, values: dict[str, Any]) -> None:
        now = utc_now()
        async with self.db.write_transaction():
            for k, v in values.items():
                await self.conn.execute(
                    """
                    INSERT INTO app_settings (key, value_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                    """,
                    (k, json.dumps(v), now),
                )

    # ------------------------------------------------------------------
    # Durable Background Jobs
    # ------------------------------------------------------------------

    async def upsert_background_job(self, job: BackgroundJobModel) -> None:
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO background_jobs (
                    job_id, tree_id, parent_node_id, child_node_id, title, goal,
                    status, turns_executed, tokens_used, summary, error,
                    mailbox_id, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    turns_executed = excluded.turns_executed,
                    tokens_used = excluded.tokens_used,
                    summary = excluded.summary,
                    error = excluded.error,
                    mailbox_id = excluded.mailbox_id,
                    completed_at = excluded.completed_at
                """,
                (
                    job.job_id,
                    job.tree_id,
                    job.parent_node_id,
                    job.child_node_id,
                    job.title,
                    job.goal,
                    job.status,
                    job.turns_executed,
                    job.tokens_used,
                    job.summary,
                    job.error,
                    job.mailbox_id,
                    job.created_at,
                    job.completed_at,
                ),
            )

    async def get_background_job(self, job_id: str) -> BackgroundJobModel | None:
        cur = await self.conn.execute(
            "SELECT * FROM background_jobs WHERE job_id = ?", (job_id,)
        )
        r = await cur.fetchone()
        return BackgroundJobModel(**dict(r)) if r else None

    async def list_background_jobs(
        self, tree_id: str | None = None, *, limit: int | None = None
    ) -> list[BackgroundJobModel]:
        lim_clause = f" LIMIT {int(limit)}" if limit is not None and limit > 0 else ""
        if tree_id is not None:
            cur = await self.conn.execute(
                f"SELECT * FROM background_jobs WHERE tree_id = ? ORDER BY created_at DESC{lim_clause}",
                (tree_id,),
            )
        else:
            cur = await self.conn.execute(
                f"SELECT * FROM background_jobs ORDER BY created_at DESC{lim_clause}"
            )
        return [BackgroundJobModel(**dict(r)) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # Durable Sandbox Workspaces
    # ------------------------------------------------------------------

    async def upsert_sandbox_workspace(
        self, node_id: str, path: str, branch_name: str, base_ref: str
    ) -> None:
        now = utc_now()
        async with self.db.write_transaction():
            await self.conn.execute(
                """
                INSERT INTO sandbox_workspaces (node_id, path, branch_name, base_ref, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    path = excluded.path,
                    branch_name = excluded.branch_name,
                    base_ref = excluded.base_ref
                """,
                (node_id, path, branch_name, base_ref, now),
            )

    async def delete_sandbox_workspace(self, node_id: str) -> None:
        async with self.db.write_transaction():
            await self.conn.execute(
                "DELETE FROM sandbox_workspaces WHERE node_id = ?", (node_id,)
            )

    async def list_sandbox_workspaces(self) -> list[dict[str, str]]:
        cur = await self.conn.execute("SELECT * FROM sandbox_workspaces")
        return [dict(r) for r in await cur.fetchall()]

