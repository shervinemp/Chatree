from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class AdapterAnswer:
    answer: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    peak_context_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemorySystemAdapter(Protocol):
    """
    Unified interface for head-to-head benchmarking of Chatree against official upstream
    memory/context frameworks (`mem0ai`, `letta`, `graphiti-core`, Linear, Flat-RAG) on
    identical multi-session datasets (LongMemEval, LoCoMo) and branching scenarios.
    """

    @property
    def name(self) -> str: ...

    async def reset(self, conversation_id: str) -> None: ...

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None: ...

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None: ...

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None: ...

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer: ...

    async def close(self) -> None: ...

