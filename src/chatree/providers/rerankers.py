from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Any

from chatree.protocols import LLMProvider
from chatree.providers.embeddings import (
    GLOBAL_ONNX_INFER_LOCK,
    OnnxDevice,
    resolve_onnx_provider_candidates,
    resolve_onnx_providers,
)
from chatree.types import SearchHit


class FastEmbedCrossEncoderReranker:
    """
    Zero-API-cost local ONNX cross-encoder reranker backed by `fastembed`
    (default `Xenova/ms-marco-MiniLM-L-6-v2`) with automatic CUDA/DirectML GPU acceleration.
    Install with: `uv sync --extra local-rag` or `uv pip install fastembed`.
    """

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int | None = None,
        threads: int | None = None,
        device: OnnxDevice = "auto",
    ) -> None:
        self.model_name = model_name
        self.device: OnnxDevice = device
        self.providers, self.uses_gpu = resolve_onnx_providers(device)
        self.batch_size = max(1, batch_size if batch_size is not None else (64 if self.uses_gpu else 16))
        self.threads = threads if threads is not None else min(4, os.cpu_count() or 2)
        self._encoder: Any = None
        self._init_lock = threading.Lock()
        self._infer_lock = GLOBAL_ONNX_INFER_LOCK

    def _build_encoder_with_providers(self, providers: list[str]) -> Any:
        try:
            from fastembed.rerank.cross_encoder import (  # type: ignore[import-not-found]
                TextCrossEncoder,
            )
        except ImportError as exc:
            raise RuntimeError(
                "FastEmbedCrossEncoderReranker requires `fastembed`. "
                "Install with `uv sync --extra local-rag` or `uv pip install fastembed`."
            ) from exc
        return TextCrossEncoder(
            model_name=self.model_name,
            threads=self.threads,
            providers=providers,
        )

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            with self._init_lock:
                if self._encoder is None:
                    candidates = resolve_onnx_provider_candidates(self.device)
                    last_exc: Exception | None = None
                    for provs, is_gpu in candidates:
                        try:
                            candidate = self._build_encoder_with_providers(provs)
                            # Runtime smoke-test catches missing cuDNN/GPU kernel DLLs at init time
                            if is_gpu:
                                with self._infer_lock:
                                    _ = list(candidate.rerank("ok", ["ok"], batch_size=1))
                            self.providers = provs
                            self.uses_gpu = is_gpu
                            if not is_gpu:
                                self.batch_size = min(self.batch_size, 16)
                            self._encoder = candidate
                            break
                        except Exception as exc:
                            last_exc = exc
                            continue
                    if self._encoder is None:
                        assert last_exc is not None
                        raise last_exc
        return self._encoder

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits or len(hits) <= 1:
            return hits[:top_k]
        docs = [f"{h.node_title}: {h.snippet[:1200]}" for h in hits]

        def _score() -> list[float]:
            encoder = self._get_encoder()
            with self._infer_lock:
                try:
                    return [
                        float(s)
                        for s in encoder.rerank(query, docs, batch_size=self.batch_size)
                    ]
                except Exception:
                    if not self.uses_gpu:
                        raise
                    self.providers = ["CPUExecutionProvider"]
                    self.uses_gpu = False
                    self.batch_size = min(self.batch_size, 16)
                    self._encoder = self._build_encoder_with_providers(self.providers)
                    return [
                        float(s)
                        for s in self._encoder.rerank(query, docs, batch_size=self.batch_size)
                    ]

        scores = await asyncio.to_thread(_score)
        paired = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        out: list[SearchHit] = []
        for sc, h in paired[:top_k]:
            out.append(h.model_copy(update={"score": round(sc, 6)}))
        return out


class LLMReranker:
    """
    Zero-extra-dependency listwise LLM reranker compatible with any `LLMProvider`.
    Asks the model to return the top-k candidate indices ordered by direct relevance to `query`.
    """

    def __init__(self, llm: LLMProvider, model: str) -> None:
        self.llm = llm
        self.model = model

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits or len(hits) <= 1:
            return hits[:top_k]
        candidates = hits[: min(len(hits), 15)]
        lines = [
            f"[{i}] Branch='{h.node_title}' ({h.granularity}): {' '.join(h.snippet.split())[:220]}"
            for i, h in enumerate(candidates)
        ]
        prompt = (
            "You are a search reranker. Select and order the candidate indices most relevant to "
            f"answering the query (at most {top_k} indices). Return strict JSON: {{\"indices\": [int, ...]}}\n"
            f"Query: {query}\n\nCandidates:\n" + "\n".join(lines)
        )
        try:
            resp = await self.llm.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
            )
            m = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                idxs = parsed.get("indices")
                if isinstance(idxs, list):
                    ordered: list[SearchHit] = []
                    seen: set[int] = set()
                    for raw_i in idxs:
                        if isinstance(raw_i, int) and 0 <= raw_i < len(candidates) and raw_i not in seen:
                            seen.add(raw_i)
                            ordered.append(candidates[raw_i])
                        if len(ordered) >= top_k:
                            break
                    if ordered:
                        # Fill remaining slots with original ranking if fewer than top_k returned
                        for idx, h in enumerate(candidates):
                            if len(ordered) >= top_k:
                                break
                            if idx not in seen:
                                ordered.append(h)
                        return ordered[:top_k]
        except Exception:
            pass
        return hits[:top_k]
