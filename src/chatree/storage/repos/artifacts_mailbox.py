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
    ArtifactInput,
    ArtifactModel,
    ArtifactRef,
    BackgroundJobModel,
    MailboxItemModel,
    MailboxStatus,
    ReflogAction,
    ReflogEntryModel,
)


class ReflogArtifactsMailboxMixin:
    db: Any
    conn: Any

    # Reflog
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
                    id, tree_id, seq, from_node_id, to_node_id, action,
                    note, orientation_cue, cue_consumed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rf_id,
                    tree_id,
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
                    "UPDATE trees SET head_node_id = ?, reflog_cursor = 0, updated_at = ? WHERE id = ?",
                    (to_node_id, now, tree_id),
                )
        entries = await self.list_reflog(tree_id, limit=1)
        return entries[0]

    async def list_reflog(self, tree_id: str, limit: int = 100) -> list[ReflogEntryModel]:
        cur = await self.conn.execute(
            "SELECT * FROM reflog WHERE tree_id = ? ORDER BY seq DESC LIMIT ?",
            (tree_id, limit),
        )
        rows = await cur.fetchall()
        return [
            ReflogEntryModel(
                id=r["id"],
                tree_id=r["tree_id"],
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
        self, tree_id: str, node_id: str
    ) -> list[ReflogEntryModel]:
        cur = await self.conn.execute(
            """
            SELECT * FROM reflog
            WHERE tree_id = ? AND to_node_id = ? AND cue_consumed = 0 AND orientation_cue != ''
            ORDER BY seq DESC
            LIMIT 3
            """,
            (tree_id, node_id),
        )
        return [
            ReflogEntryModel(
                id=r["id"],
                tree_id=r["tree_id"],
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

