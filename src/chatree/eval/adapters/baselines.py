from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Protocol, runtime_checkable

_RLM_LOCAL_REPL_LOCK = threading.Lock()

from chatree.config import AblationConfig, ChatreeConfig
from chatree.protocols import Embedder, LLMProvider, Reranker
from chatree.providers.mock import MockLLMProvider
from chatree.providers.openai_compat import resolve_provider_endpoint
from chatree.sdk.client import Chatree, DialogueTree


from chatree.eval.adapters.base import AdapterAnswer



class FullContextLinearAdapter:
    """Fair Full-Context Linear baseline: puts all prior session turns in one flat prompt."""

    def __init__(
        self,
        *,
        name: str = "full_context_linear",
        model: str = "mock/chatree-demo",
        llm: LLMProvider | None = None,
        max_context_tokens: int = 120_000,
    ) -> None:
        self._name = name
        self.model = model
        self.llm = llm or MockLLMProvider(stream_delay_s=0.0)
        self.max_context_tokens = max_context_tokens
        self._turns: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        return self._name

    def clone(self) -> FullContextLinearAdapter:
        return FullContextLinearAdapter(
            name=self._name,
            model=self.model,
            llm=self.llm,
            max_context_tokens=self.max_context_tokens,
        )

    async def reset(self, conversation_id: str) -> None:
        del conversation_id
        self._turns = []

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del parent_session_id
        header = f"--- Session {title or session_id}" + (f" ({date_time})" if date_time else "") + " ---"
        self._turns.append({"role": "user", "content": header})

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        del session_id
        norm_role = "user" if role.lower() in ("user", "human", "speaker_a") else "assistant"
        stamped = f"[{date_time}] {content}" if date_time else content
        self._turns.append({"role": norm_role, "content": stamped})

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del session_id, summary

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        # Guard against exceeding provider context window (e.g. DeepSeek 128k on LongMemEval)
        budget_chars = max(4000, (self.max_context_tokens - 600) * 4)
        kept_turns: list[dict[str, str]] = []
        used_chars = 0
        truncated = False
        for t in reversed(self._turns):
            tlen = len(t.get("content", "")) + 16
            if used_chars + tlen > budget_chars and kept_turns:
                truncated = True
                break
            kept_turns.append(t)
            used_chars += tlen
        kept_turns.reverse()
        msgs: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Answer the user's question concisely based on the full conversation history. "
                    "If information changed over time, use the latest update. If unknown, say so."
                ),
            },
            *kept_turns,
            {"role": "user", "content": prompt_q},
        ]
        resp = await self.llm.complete(model=self.model, messages=msgs, tools=None, temperature=0.0)
        return AdapterAnswer(
            answer=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cache_read_tokens=resp.cache_read_tokens,
            peak_context_tokens=resp.tokens_in,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            metadata={"truncated": truncated},
        )

    async def close(self) -> None:
        self._turns = []


class FlatHybridRAGAdapter:
    """Fair Flat Hybrid RAG baseline (BM25 + dense vector chunks -> LLM synthesis, no shadow turns)."""

    def __init__(
        self,
        *,
        name: str = "flat_hybrid_rag",
        model: str = "mock/chatree-demo",
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        top_k: int = 5,
    ) -> None:
        self._name = name
        self.model = model
        self.llm = llm or (MockLLMProvider(stream_delay_s=0.0) if model.startswith("mock/") else None)
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self._ct: Chatree | None = None
        self._tree: DialogueTree | None = None
        self._root_id: str = ""
        self._dirty_index: bool = False

    def clone(self) -> FlatHybridRAGAdapter:
        return FlatHybridRAGAdapter(
            name=self._name,
            model=self.model,
            llm=self.llm,
            embedder=self.embedder,
            reranker=self.reranker,
            top_k=self.top_k,
        )

    @property
    def name(self) -> str:
        return self._name

    async def reset(self, conversation_id: str) -> None:
        await self.close()
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            ablations=AblationConfig(enable_subtree_rollups=False, enable_shadow_turns=False),
        )
        self._ct = Chatree.in_memory(
            config=cfg, llm=self.llm, embedder=self.embedder, reranker=self.reranker
        )
        await self._ct.startup()
        self._tree = await self._ct.create_tree(conversation_id, default_model=self.model)
        self._root_id = (await self._tree.root()).id

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del parent_session_id
        assert self._ct is not None and self._tree is not None
        header = f"[Session {title or session_id}" + (f" | {date_time}" if date_time else "") + "]"
        await self._ct.repo.append_message(self._root_id, self._tree.id, "user", header)

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        del session_id
        assert self._ct is not None and self._tree is not None
        norm_role = "user" if role.lower() in ("user", "human", "speaker_a") else "assistant"
        stamped = f"[{date_time}] {content}" if date_time else content
        await self._ct.repo.append_message(self._root_id, self._tree.id, norm_role, stamped)

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del session_id, summary
        self._dirty_index = True

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._ct is not None and self._tree is not None
        if self._dirty_index:
            await self._ct.rag.index_node_chunks(self._root_id)
            self._dirty_index = False
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        hits = await self._tree.search(
            prompt_q, scope="all", top_k=self.top_k, granularities={"chunk"}
        )
        chunks_txt = "\n\n".join(f"- {h.snippet}" for h in hits)
        resp = await self._ct.llm.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question concisely using only the retrieved conversation snippets. "
                        "If facts changed over time, prefer the latest dated update. If absent, say so."
                    ),
                },
                {"role": "user", "content": f"[RAG Context]\n{chunks_txt}\n\nQuestion: {prompt_q}"},
            ],
            tools=None,
            temperature=0.0,
        )
        return AdapterAnswer(
            answer=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cache_read_tokens=resp.cache_read_tokens,
            peak_context_tokens=resp.tokens_in,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            metadata={"hits_count": len(hits)},
        )

    async def close(self) -> None:
        if self._ct is not None:
            await self._ct.close()
            self._ct = None
            self._tree = None




class TreeSummaryOnlyAdapter:
    """
    Context-Folding / Hierarchical Summary-Only baseline (ReSum / FoldGRPO-style folding WITHOUT
    live read-only shadow turns). Summarizes each session at `finish_session` (tracking write-path
    LLM tokens!) and answers queries solely from retrieved session summaries.
    """

    def __init__(
        self,
        *,
        name: str = "tree_summary_only",
        model: str = "mock/chatree-demo",
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        top_k: int = 6,
    ) -> None:
        self._name = name
        self.model = model
        self.llm = llm or (MockLLMProvider(stream_delay_s=0.0) if model.startswith("mock/") else None)
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = top_k
        self._ct: Chatree | None = None
        self._tree: DialogueTree | None = None
        self._root_id: str = ""
        self._session_nodes: dict[str, str] = {}
        self._node_dates: dict[str, str] = {}
        self._node_order: dict[str, int] = {}
        self._node_titles: dict[str, str] = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0

    @property
    def name(self) -> str:
        return self._name

    def clone(self) -> TreeSummaryOnlyAdapter:
        return TreeSummaryOnlyAdapter(
            name=self._name,
            model=self.model,
            llm=self.llm,
            embedder=self.embedder,
            reranker=self.reranker,
            top_k=self.top_k,
        )

    def get_ingest_usage(self) -> tuple[int, int]:
        return self._ingest_tokens_in, self._ingest_tokens_out

    async def reset(self, conversation_id: str) -> None:
        await self.close()
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            ablations=AblationConfig(enable_shadow_turns=False),
        )
        self._ct = Chatree.in_memory(
            config=cfg, llm=self.llm, embedder=self.embedder, reranker=self.reranker
        )
        await self._ct.startup()
        self._tree = await self._ct.create_tree(conversation_id, default_model=self.model)
        self._root_id = (await self._tree.root()).id
        self._session_nodes = {"root": self._root_id}
        self._node_dates = {}
        self._node_order = {}
        self._node_titles = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        assert self._tree is not None
        parent_id = self._session_nodes.get(parent_session_id or "root", self._root_id)
        label = title or session_id
        child = await self._tree.spawn(
            goal=f"{label} | Date: {date_time}" if date_time else label,
            parent_id=parent_id,
            title=label,
            mode="clean",
            switch_head=False,
        )
        self._session_nodes[session_id] = child.id
        self._node_dates[child.id] = date_time
        self._node_order[child.id] = len(self._node_order)
        self._node_titles[child.id] = label

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        assert self._ct is not None and self._tree is not None
        if session_id not in self._session_nodes:
            await self.start_session(session_id, title=session_id, date_time=date_time)
        nid = self._session_nodes[session_id]
        stamped = f"[{date_time}] {content}" if date_time else content
        norm_role = "user" if role.lower() in ("user", "human", "speaker_a") else "assistant"
        await self._ct.repo.append_message(nid, self._tree.id, norm_role, stamped, mark_dirty=True)

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        assert self._ct is not None and self._tree is not None
        nid = self._session_nodes.get(session_id)
        if not nid or nid == self._root_id:
            return
        if summary and summary.strip():
            sum_text = summary.strip()
        else:
            msgs = await self._ct.repo.list_messages(nid)
            transcript = "\n".join(f"{m.role}: {m.content}" for m in msgs)[:6000]
            resp = await self._ct.llm.complete(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize this conversation session concisely in 3-5 bullet points, "
                            "preserving key facts, names, dates, preferences, and decisions."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ],
                tools=None,
                temperature=0.0,
            )
            self._ingest_tokens_in += resp.tokens_in
            self._ingest_tokens_out += resp.tokens_out
            sum_text = resp.content
        await self._ct.repo.commit_node_summary(nid, sum_text, status="COMPLETED")
        await self._ct.rag.index_node_summary(nid)

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._ct is not None and self._tree is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        hits = await self._tree.search(
            prompt_q, scope="all", top_k=self.top_k, granularities={"summary"}
        )
        ordered = sorted(
            hits,
            key=lambda h: (
                self._node_dates.get(h.node_id, ""),
                self._node_order.get(h.node_id, 0),
            ),
        )
        summaries_txt = "\n\n".join(
            f"[Session: {h.node_title}"
            + (f" | Date: {self._node_dates[h.node_id]}" if self._node_dates.get(h.node_id) else "")
            + f"]\n{h.summary or h.snippet}"
            for h in ordered
        )
        resp = await self._ct.llm.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer concisely using ONLY the retrieved folded session summaries below. "
                        "If facts changed over time, prefer the latest date. If unknown, say so."
                    ),
                },
                {"role": "user", "content": f"[Folded Summaries]\n{summaries_txt}\n\nQuestion: {prompt_q}"},
            ],
            tools=None,
            temperature=0.0,
        )
        return AdapterAnswer(
            answer=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cache_read_tokens=resp.cache_read_tokens,
            peak_context_tokens=resp.tokens_in,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            metadata={"summaries_retrieved": len(ordered)},
        )

    async def close(self) -> None:
        if self._ct is not None:
            await self._ct.close()
            self._ct = None
            self._tree = None
            self._session_nodes.clear()

