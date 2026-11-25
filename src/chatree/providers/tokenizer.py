from __future__ import annotations

import json
import re
from typing import Any


_TOKEN_SPLIT_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


class HeuristicTokenCounter:
    """
    Zero-dependency calibrated token estimator accounting for prose words, code/JSON
    punctuation tokens, and non-ASCII/CJK characters (significantly more accurate on
    code, schemas, and JSON tool payloads than naive `len(text) // 4`).
    """

    def count_text(self, text: str, *, model: str | None = None) -> int:
        del model
        if not text:
            return 0
        # Fast path for very large blobs to keep O(1) overhead under microsecond budgets
        if len(text) > 80_000:
            return max(1, int(len(text) / 3.6))
        pieces = _TOKEN_SPLIT_RE.findall(text)
        total = 0
        for p in pieces:
            n = len(p)
            if n == 1:
                total += 1
            elif n <= 5:
                total += 1
            else:
                # Long identifiers / snake_case / camelCase split into ~1 token per 4 chars
                total += max(1, (n + 3) // 4)
        return max(1, total)

    def count_messages(self, messages: list[dict[str, Any]], *, model: str | None = None) -> int:
        total = 3
        for m in messages:
            total += 4
            content = m.get("content")
            if isinstance(content, str):
                total += self.count_text(content, model=model)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += self.count_text(part["text"], model=model)
            if m.get("tool_calls"):
                total += self.count_text(json.dumps(m["tool_calls"]), model=model)
        return total


class TiktokenCounter:
    """Exact BPE token counter backed by `tiktoken` when installed (`pip install tiktoken`)."""

    def __init__(self, default_encoding: str = "cl100k_base") -> None:
        import tiktoken  # type: ignore[import-not-found]

        self._tiktoken = tiktoken
        self._default_enc = tiktoken.get_encoding(default_encoding)
        self._cache: dict[str, Any] = {}

    def _encoding_for(self, model: str | None) -> Any:
        if not model:
            return self._default_enc
        bare = model.split("/", 1)[-1]
        enc = self._cache.get(bare)
        if enc is None:
            try:
                enc = self._tiktoken.encoding_for_model(bare)
            except KeyError:
                enc = self._default_enc
            self._cache[bare] = enc
        return enc

    def count_text(self, text: str, *, model: str | None = None) -> int:
        if not text:
            return 0
        return max(1, len(self._encoding_for(model).encode(text, disallowed_special=())))

    def count_messages(self, messages: list[dict[str, Any]], *, model: str | None = None) -> int:
        total = 3
        for m in messages:
            total += 4
            content = m.get("content")
            if isinstance(content, str):
                total += self.count_text(content, model=model)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += self.count_text(part["text"], model=model)
            if m.get("tool_calls"):
                total += self.count_text(json.dumps(m["tool_calls"]), model=model)
        return total


def build_token_counter(*, prefer_tiktoken: bool = True) -> HeuristicTokenCounter | TiktokenCounter:
    """Return `TiktokenCounter` when `tiktoken` is installed, else `HeuristicTokenCounter`."""
    if prefer_tiktoken:
        try:
            return TiktokenCounter()
        except Exception:
            pass
    return HeuristicTokenCounter()


_DEFAULT_COUNTER = build_token_counter()


def get_default_token_counter() -> HeuristicTokenCounter | TiktokenCounter:
    return _DEFAULT_COUNTER


def set_default_token_counter(counter: Any) -> None:
    """Override the process-wide default `TokenCounter` implementation."""
    global _DEFAULT_COUNTER
    _DEFAULT_COUNTER = counter
