from __future__ import annotations

from typing import Any
import aiosqlite

from chatree.storage.database import Database
from chatree.storage.repos.artifacts_mailbox import ReflogArtifactsMailboxMixin
from chatree.storage.repos.base import (
    _np,
    _row_to_artifact,
    _row_to_ipc,
    _row_to_mailbox,
    _row_to_message,
    _row_to_node,
    _row_to_tree,
    cosine_similarity,
    eligible_node_ids_safe,
    new_id,
    normalize_bm25,
    pack_vector,
    sanitize_fts_query,
    unpack_vector,
    utc_now,
)
from chatree.storage.repos.ipc_deps import IpcDepsShadowMixin
from chatree.storage.repos.memory_facts import MemoryFactsInvariantsMixin
from chatree.storage.repos.search_vectors import SearchVectorsMixin
from chatree.storage.repos.trees import TreesNodesMessagesMixin


class SQLiteRepository(
    TreesNodesMessagesMixin,
    IpcDepsShadowMixin,
    ReflogArtifactsMailboxMixin,
    SearchVectorsMixin,
    MemoryFactsInvariantsMixin,
):
    """
    Composable async SQLite repository assembled from cohesive domain mixins:
      - `TreesNodesMessagesMixin`: trees, nodes, copy-on-write messages
      - `IpcDepsShadowMixin`: IPC audit events, transitive staleness, shadow cache
      - `ReflogArtifactsMailboxMixin`: reflog, cues, artifacts, mailbox, settings, jobs, sandboxes
      - `SearchVectorsMixin`: per-tree cached NumPy vector search, sqlite-vec, FTS5 BM25
      - `MemoryFactsInvariantsMixin`: Tier-2.5 atomic temporal facts & tree-wide PINNED invariants
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self._vector_cache: dict[str, dict[int, tuple[list[Any], Any]]] = {}
        self._vector_cache_hits: int = 0
        self._vector_cache_misses: int = 0

    @property
    def conn(self) -> aiosqlite.Connection:
        return self.db.conn


__all__ = [
    "SQLiteRepository",
    "cosine_similarity",
    "eligible_node_ids_safe",
    "new_id",
    "normalize_bm25",
    "pack_vector",
    "sanitize_fts_query",
    "unpack_vector",
    "utc_now",
]
