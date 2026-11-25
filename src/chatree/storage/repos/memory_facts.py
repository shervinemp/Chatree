from __future__ import annotations

import json
import re
from typing import Any

from chatree.storage.repos.base import (
    new_id,
    utc_now,
)
from chatree.types import (
    BranchFactModel,
    TreeInvariantModel,
)


class MemoryFactsInvariantsMixin:
    db: Any
    conn: Any

    # ------------------------------------------------------------------
    # Tier-2.5 Branch-Scoped Atomic Temporal Facts
    # ------------------------------------------------------------------

    async def replace_branch_facts(
        self,
        tree_id: str,
        node_id: str,
        facts: list[dict[str, Any]],
        *,
        source_msg_seq: int | None = None,
    ) -> list[BranchFactModel]:
        """
        Supersede prior active facts for THIS `node_id` ONLY (never touching other branches!)
        and insert the newly extracted atomic temporal facts for `node_id`.
        """
        now = utc_now()
        created: list[BranchFactModel] = []
        async with self.db.write_transaction():
            await self.conn.execute(
                "UPDATE branch_facts SET status = 'superseded' WHERE node_id = ? AND status = 'active'",
                (node_id,),
            )
            for item in facts:
                fact_text = str(item.get("fact") or item.get("fact_text") or "").strip()
                if not fact_text:
                    continue
                raw_ents = item.get("entities") or []
                if isinstance(raw_ents, str):
                    raw_ents = [raw_ents]
                entities = [
                    str(e).strip()
                    for e in raw_ents
                    if str(e).strip()
                ][:12]
                resolved_date = str(item.get("resolved_date") or "").strip()[:40]
                msg_seq = item.get("source_msg_seq", source_msg_seq)
                fid = new_id("fact")
                await self.conn.execute(
                    """
                    INSERT INTO branch_facts (
                        id, tree_id, node_id, fact_text, entities_json,
                        resolved_date, source_msg_seq, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (
                        fid,
                        tree_id,
                        node_id,
                        fact_text[:600],
                        json.dumps(entities),
                        resolved_date,
                        msg_seq,
                        now,
                    ),
                )
                created.append(
                    BranchFactModel(
                        id=fid,
                        tree_id=tree_id,
                        node_id=node_id,
                        fact_text=fact_text[:600],
                        entities=entities,
                        resolved_date=resolved_date,
                        source_msg_seq=msg_seq,
                        status="active",
                        created_at=now,
                    )
                )
        return created

    async def list_branch_facts(
        self,
        tree_id: str,
        *,
        node_id: str | None = None,
        active_only: bool = True,
        exclude_archived: bool = True,
    ) -> list[BranchFactModel]:
        clauses = ["f.tree_id = ?"]
        params: list[Any] = [tree_id]
        if node_id is not None:
            clauses.append("f.node_id = ?")
            params.append(node_id)
        if active_only:
            clauses.append("f.status = 'active'")
        if exclude_archived:
            clauses.append("n.status != 'ARCHIVED'")
        cur = await self.conn.execute(
            f"""
            SELECT f.*
            FROM branch_facts f
            JOIN nodes n ON n.id = f.node_id
            WHERE {' AND '.join(clauses)}
            ORDER BY f.created_at ASC
            """,
            params,
        )
        return [
            BranchFactModel(
                id=r["id"],
                tree_id=r["tree_id"],
                node_id=r["node_id"],
                fact_text=r["fact_text"],
                entities=json.loads(r["entities_json"]) if r["entities_json"] else [],
                resolved_date=r["resolved_date"] or "",
                source_msg_seq=r["source_msg_seq"],
                status=r["status"],
                created_at=r["created_at"],
            )
            for r in await cur.fetchall()
        ]

    # ------------------------------------------------------------------
    # Tree-Level PINNED Invariants (Bounded Core Memory)
    # ------------------------------------------------------------------

    async def upsert_tree_invariant(
        self,
        tree_id: str,
        key: str,
        value: str,
        *,
        source_node_id: str | None = None,
        max_items: int = 8,
        max_value_chars: int = 120,
    ) -> TreeInvariantModel:
        clean_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip()).strip("_")[:48]
        if not clean_key:
            raise ValueError("Invariant key must contain at least one alphanumeric character.")
        clean_val = " ".join(value.strip().split())[:max_value_chars]
        if not clean_val:
            raise ValueError("Invariant value cannot be empty.")
        now = utc_now()
        src_sv = 0
        src_seq = 0
        if source_node_id:
            scur = await self.conn.execute(
                "SELECT summary_version, last_msg_seq FROM nodes WHERE id = ?",
                (source_node_id,),
            )
            srow = await scur.fetchone()
            if srow is not None:
                src_sv = int(srow["summary_version"])
                src_seq = int(srow["last_msg_seq"])
        async with self.db.write_transaction():
            cur = await self.conn.execute(
                "SELECT key FROM tree_invariants WHERE tree_id = ?", (tree_id,)
            )
            existing_keys = {r["key"] for r in await cur.fetchall()}
            if clean_key not in existing_keys and len(existing_keys) >= max_items:
                raise ValueError(
                    f"Pinned invariant limit reached ({max_items}). Unpin or update an existing key first."
                )
            await self.conn.execute(
                """
                INSERT INTO tree_invariants (
                    tree_id, key, value, source_node_id, source_summary_version, source_msg_seq, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tree_id, key) DO UPDATE SET
                    value = excluded.value,
                    source_node_id = excluded.source_node_id,
                    source_summary_version = excluded.source_summary_version,
                    source_msg_seq = excluded.source_msg_seq,
                    updated_at = excluded.updated_at
                """,
                (tree_id, clean_key, clean_val, source_node_id, src_sv, src_seq, now),
            )
        return TreeInvariantModel(
            tree_id=tree_id,
            key=clean_key,
            value=clean_val,
            source_node_id=source_node_id,
            source_summary_version=src_sv,
            source_msg_seq=src_seq,
            updated_at=now,
        )

    async def delete_tree_invariant(self, tree_id: str, key: str) -> bool:
        clean_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip()).strip("_")[:48]
        async with self.db.write_transaction():
            cur = await self.conn.execute(
                "DELETE FROM tree_invariants WHERE tree_id = ? AND key = ?",
                (tree_id, clean_key),
            )
            return (cur.rowcount or 0) > 0

    async def list_tree_invariants(self, tree_id: str) -> list[TreeInvariantModel]:
        cur = await self.conn.execute(
            "SELECT * FROM tree_invariants WHERE tree_id = ? ORDER BY key ASC",
            (tree_id,),
        )
        return [
            TreeInvariantModel(
                tree_id=r["tree_id"],
                key=r["key"],
                value=r["value"],
                source_node_id=r["source_node_id"],
                source_summary_version=int(r["source_summary_version"]) if "source_summary_version" in r.keys() else 0,
                source_msg_seq=int(r["source_msg_seq"]) if "source_msg_seq" in r.keys() else 0,
                updated_at=r["updated_at"],
            )
            for r in await cur.fetchall()
        ]

