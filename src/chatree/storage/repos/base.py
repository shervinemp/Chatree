from __future__ import annotations

import json
import math
import re
import struct
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from chatree.types import (
    ArtifactModel,
    ArtifactRef,
    IpcEventModel,
    MailboxItemModel,
    MessageModel,
    NodeModel,
    StaleDependencyInfo,
    TreeModel,
)

_np: Any
try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:
    _np = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> tuple[float, ...]:
    n = len(blob) // 4
    return struct.unpack(f"<{n}f", blob)


def cosine_similarity(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 1e-12:
        return 0.0
    return dot / denom


def normalize_bm25(raw: float) -> float:
    """Strictly monotone mapping of SQLite FTS5 bm25() (more negative = stronger match) into (0, 1)."""
    clamped = max(-40.0, min(40.0, raw + 2.0))
    return 1.0 / (1.0 + math.exp(0.35 * clamped))


def sanitize_fts_query(raw: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", raw.strip())
    if not tokens:
        return ""
    seen: set[str] = set()
    parts: list[str] = []
    for tok in tokens[:14]:
        low = tok.lower()
        if low not in seen:
            seen.add(low)
            parts.append(f'"{tok}"*')
    return " OR ".join(parts)


def _row_to_tree(row: aiosqlite.Row) -> TreeModel:
    return TreeModel(
        id=row["id"],
        title=row["title"],
        root_node_id=row["root_node_id"],
        head_node_id=row["head_node_id"],
        default_model=row["default_model"],
        shadow_model=row["shadow_model"],
        summary_model=row["summary_model"],
        auto_commit_handoffs=bool(row["auto_commit_handoffs"]),
        reflog_cursor=row["reflog_cursor"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_node(
    row: aiosqlite.Row,
    stale_dependencies: list[StaleDependencyInfo] | None = None,
    child_count: int = 0,
    pending_handoffs_count: int = 0,
) -> NodeModel:
    return NodeModel(
        id=row["id"],
        tree_id=row["tree_id"],
        parent_id=row["parent_id"],
        path=row["path"],
        depth=row["depth"],
        title=row["title"],
        goal=row["goal"],
        summary_brief=row["summary_brief"],
        context_mode=row["context_mode"],
        spawn_msg_seq=row["spawn_msg_seq"],
        slice_turn_count=row["slice_turn_count"],
        status=row["status"],
        node_kind=(
            row["node_kind"]
            if "node_kind" in row.keys() and row["node_kind"] in ("chat", "subagent")
            else "chat"
        ),
        tool_permission=row["tool_permission"],
        model_override=row["model_override"],
        summary=row["summary"],
        summary_version=row["summary_version"],
        summary_minor_version=row["summary_minor_version"],
        last_change_severity=row["last_change_severity"],
        last_change_delta=row["last_change_delta"],
        rollup_summary=row["rollup_summary"],
        is_dirty=bool(row["is_dirty"]),
        last_msg_seq=row["last_msg_seq"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        cache_read_tokens=row["cache_read_tokens"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        stale_dependencies=stale_dependencies or [],
        child_count=child_count,
        pending_handoffs_count=pending_handoffs_count,
    )


def _row_to_message(row: aiosqlite.Row) -> MessageModel:
    return MessageModel(
        id=row["id"],
        node_id=row["node_id"],
        tree_id=row["tree_id"],
        seq=row["seq"],
        role=row["role"],
        content=row["content"],
        tool_calls=json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else None,
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
        model_used=row["model_used"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        cache_read_tokens=row["cache_read_tokens"],
        latency_ms=row["latency_ms"],
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        created_at=row["created_at"],
    )


def _row_to_ipc(row: aiosqlite.Row) -> IpcEventModel:
    return IpcEventModel(
        id=row["id"],
        tree_id=row["tree_id"],
        trace_id=row["trace_id"],
        hop_depth=row["hop_depth"],
        from_node_id=row["from_node_id"],
        to_node_id=row["to_node_id"],
        event_type=row["event_type"],
        payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
        response=json.loads(row["response_json"]) if row["response_json"] else {},
        status=row["status"],
        target_summary_version=row["target_summary_version"],
        target_msg_seq=row["target_msg_seq"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        latency_ms=row["latency_ms"],
        created_at=row["created_at"],
    )


def _row_to_artifact(row: aiosqlite.Row) -> ArtifactModel:
    return ArtifactModel(
        id=row["id"],
        tree_id=row["tree_id"],
        slug=row["slug"],
        version=row["version"],
        kind=row["kind"],
        title=row["title"],
        content=row["content"],
        summary=row["summary"],
        created_by_node_id=row["created_by_node_id"],
        created_at=row["created_at"],
    )


def _row_to_mailbox(row: aiosqlite.Row) -> MailboxItemModel:
    raw_refs = json.loads(row["artifact_refs_json"]) if row["artifact_refs_json"] else []
    refs = [ArtifactRef(**r) for r in raw_refs]
    return MailboxItemModel(
        id=row["id"],
        tree_id=row["tree_id"],
        to_node_id=row["to_node_id"],
        from_node_id=row["from_node_id"],
        ipc_event_id=row["ipc_event_id"],
        kind=row["kind"],
        status=row["status"],
        summary=row["summary"],
        artifact_refs=refs,
        payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
        created_at=row["created_at"],
        delivered_at=row["delivered_at"],
    )


def eligible_node_ids_safe(ids: set[str]) -> set[str]:
    return ids
