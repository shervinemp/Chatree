from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import threading
from typing import Any, Literal

from chatree.protocols import Embedder

OnnxDevice = Literal["auto", "cuda", "directml", "cpu"]


def resolve_onnx_provider_candidates(device: OnnxDevice = "auto") -> list[tuple[list[str], bool]]:
    """
    Return an ordered fallback chain of `(providers, uses_gpu)` candidates so broken GPU
    runtimes (e.g. `CUDAExecutionProvider` missing `cudnn64_9.dll`) automatically step down
    to `DmlExecutionProvider` and ultimately `CPUExecutionProvider`.
    """
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        avail = set(ort.get_available_providers())
    except Exception:
        return [(["CPUExecutionProvider"], False)]

    if device == "cpu":
        return [(["CPUExecutionProvider"], False)]
    candidates: list[tuple[list[str], bool]] = []
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
        candidates.append((["CUDAExecutionProvider", "CPUExecutionProvider"], True))
    if device in ("auto", "directml") and "DmlExecutionProvider" in avail:
        candidates.append((["DmlExecutionProvider", "CPUExecutionProvider"], True))
    candidates.append((["CPUExecutionProvider"], False))
    return candidates


def resolve_onnx_providers(device: OnnxDevice = "auto") -> tuple[list[str], bool]:
    """
    Resolve top-priority available ONNX Runtime execution providers
    (`CUDAExecutionProvider` -> `DmlExecutionProvider` -> `CPUExecutionProvider`).
    Returns `(providers, uses_gpu)`.
    """
    return resolve_onnx_provider_candidates(device)[0]


class HashEmbedder:
    """
    Fast, zero-dependency, deterministic feature-hashing embedder combining word tokens,
    bigrams, and character 3-grams into an L2-normalized float vector.
    Ideal for offline use, CI tests, and zero-key development; swap in LiteLLMEmbedder
    or any custom Embedder for dense neural embeddings.
    """

    def __init__(self, dimensions: int = 128) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        if not tokens:
            return vec

        features: list[tuple[str, float]] = []
        for tok in tokens:
            features.append((f"w:{tok}", 2.0))
            if len(tok) >= 4:
                for i in range(len(tok) - 2):
                    features.append((f"c:{tok[i : i + 3]}", 0.5))
        for a, b in zip(tokens, tokens[1:]):
            features.append((f"b:{a}_{b}", 1.5))

        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dimensions
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-12:
            vec = [v / norm for v in vec]
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class LiteLLMEmbedder:
    """Neural embedding provider backed by `litellm.aembedding` (OpenAI, Gemini, Ollama, etc.)."""

    def __init__(
        self,
        model: str,
        dimensions: int = 1536,
        api_base: str | None = None,
    ) -> None:
        self.model = model
        self._dimensions = dimensions
        self.api_base = api_base

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLM is required for LiteLLMEmbedder. Install with `pip install chatree[litellm]`."
            ) from exc
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        resp = await litellm.aembedding(**kwargs)
        vectors = [list(item["embedding"]) for item in resp.data]
        if vectors:
            self._dimensions = len(vectors[0])
        return vectors


class FastEmbedEmbedder:
    """
    Zero-API-cost local dense neural embedder backed by `fastembed` ONNX CPU inference
    (default `BAAI/bge-small-en-v1.5`, 384-dim L2-normalized semantic embeddings).
    Install with: `uv sync --extra local-rag` or `uv pip install fastembed`.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimensions: int = 384,
        *,
        batch_size: int | None = None,
        threads: int | None = None,
        max_chars: int = 2000,
        device: OnnxDevice = "auto",
    ) -> None:
        self.model_name = model_name
        self._dimensions = dimensions
        self.device: OnnxDevice = device
        self.providers, self.uses_gpu = resolve_onnx_providers(device)
        # GPU tensor cores shine at batch_size=64 (<1GB VRAM); CPU caps at 16 to avoid multi-GB arenas
        self.batch_size = max(1, batch_size if batch_size is not None else (64 if self.uses_gpu else 16))
        self.threads = threads if threads is not None else min(4, os.cpu_count() or 2)
        self.max_chars = max_chars
        self._model: Any = None
        self._init_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _build_model_with_providers(self, providers: list[str]) -> Any:
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "FastEmbedEmbedder requires `fastembed`. Install with `uv sync --extra local-rag` "
                "or `uv pip install fastembed`."
            ) from exc
        return TextEmbedding(
            model_name=self.model_name,
            threads=self.threads,
            providers=providers,
        )

    def _get_model(self) -> Any:
        if self._model is None:
            with self._init_lock:
                if self._model is None:
                    candidates = resolve_onnx_provider_candidates(self.device)
                    last_exc: Exception | None = None
                    for provs, is_gpu in candidates:
                        try:
                            candidate = self._build_model_with_providers(provs)
                            # Runtime kernel smoke-test catches missing cuDNN / driver DLL failures immediately
                            if is_gpu:
                                _ = list(candidate.embed(["ok"], batch_size=1))
                            self.providers = provs
                            self.uses_gpu = is_gpu
                            if not is_gpu:
                                self.batch_size = min(self.batch_size, 16)
                            self._model = candidate
                            break
                        except Exception as exc:
                            last_exc = exc
                            continue
                    if self._model is None:
                        assert last_exc is not None
                        raise last_exc
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clipped = [t[: self.max_chars] for t in texts]

        def _run() -> list[list[float]]:
            model = self._get_model()
            # Serialize ONNX inference across async workers and cap batch_size so
            # ONNX Runtime's CPU memory arena stays <500 MB instead of multiplying per thread.
            with self._infer_lock:
                try:
                    vecs = [
                        v.tolist()
                        for v in model.embed(clipped, batch_size=self.batch_size)
                    ]
                except Exception:
                    if not self.uses_gpu:
                        raise
                    # Mid-flight GPU failure -> clean fallback to CPUExecutionProvider
                    self.providers = ["CPUExecutionProvider"]
                    self.uses_gpu = False
                    self.batch_size = min(self.batch_size, 16)
                    self._model = self._build_model_with_providers(self.providers)
                    vecs = [
                        v.tolist()
                        for v in self._model.embed(clipped, batch_size=self.batch_size)
                    ]
            if vecs:
                self._dimensions = len(vecs[0])
            return vecs

        return await asyncio.to_thread(_run)


def build_embedder(
    model: str = "hash/local-128",
    dimensions: int = 128,
    api_base: str | None = None,
) -> Embedder:
    if not model or model.startswith("hash/"):
        return HashEmbedder(dimensions=dimensions)
    if model.startswith("local/") or model.startswith("fastembed/"):
        clean = model.split("/", 1)[1] or "BAAI/bge-small-en-v1.5"
        return FastEmbedEmbedder(model_name=clean)
    clean_model = model.removeprefix("litellm/")
    return LiteLLMEmbedder(model=clean_model, dimensions=dimensions, api_base=api_base)
