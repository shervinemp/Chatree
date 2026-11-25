from __future__ import annotations

import importlib.util
from typing import Literal

from chatree.protocols import LLMProvider
from chatree.providers.litellm import LiteLLMProvider
from chatree.providers.mock import MockLLMProvider
from chatree.providers.openai_compat import OpenAICompatibleProvider

LLMBackendKind = Literal["auto", "litellm", "openai_compat", "mock"]


def _has_litellm() -> bool:
    return importlib.util.find_spec("litellm") is not None


def build_llm_provider(
    backend: LLMBackendKind = "auto",
    *,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
    enable_cache_control: bool = True,
    max_retries: int = 4,
    retry_base_delay_s: float = 1.0,
    timeout_s: float = 90.0,
    max_tokens: int | None = None,
) -> LLMProvider:
    """
    Provider-agnostic LLM backend factory:
      - `backend="mock"` (or `model="mock/..."` when `backend="auto"` and no external backend needed):
        returns `MockLLMProvider`.
      - `backend="openai_compat"`: zero-LiteLLM OpenAI-compatible HTTP/SSE provider (works with
        OpenAI, Ollama, vLLM, SGLang, LM Studio, OpenRouter, DeepSeek, Groq, Together, Mistral, xAI,
        Gemini OpenAI-compat, and custom proxies).
      - `backend="litellm"`: multi-provider `LiteLLMProvider` (100+ providers including native
        Anthropic prompt caching, Bedrock, Vertex AI, Azure, Gemini, Cohere, etc.).
      - `backend="auto"` (default): uses `LiteLLMProvider` if `litellm` is installed (unless an
        explicit `openai_compat/` / `vllm/` / `sglang/` / `lm_studio/` / `llamacpp/` prefix is used
        without `litellm`), automatically falling back to zero-dependency `OpenAICompatibleProvider`
        when `litellm` is not installed!
    """
    if backend == "mock":
        return MockLLMProvider()

    if backend == "openai_compat":
        return OpenAICompatibleProvider(
            api_base=api_base,
            api_key=api_key,
            extra_headers=extra_headers,
            max_retries=max_retries,
            retry_base_delay_s=retry_base_delay_s,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )

    if backend == "litellm":
        return LiteLLMProvider(
            api_base=api_base,
            api_key=api_key,
            extra_headers=extra_headers,
            enable_cache_control=enable_cache_control,
            max_retries=max_retries,
            retry_base_delay_s=retry_base_delay_s,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )

    # backend == "auto"
    low = (model or "").strip().lower()
    if low.startswith(("openai_compat/", "sglang/", "lm_studio/", "llamacpp/")):
        return OpenAICompatibleProvider(
            api_base=api_base,
            api_key=api_key,
            extra_headers=extra_headers,
            max_retries=max_retries,
            retry_base_delay_s=retry_base_delay_s,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )
    if _has_litellm():
        return LiteLLMProvider(
            api_base=api_base,
            api_key=api_key,
            extra_headers=extra_headers,
            enable_cache_control=enable_cache_control,
            max_retries=max_retries,
            retry_base_delay_s=retry_base_delay_s,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )
    return OpenAICompatibleProvider(
        api_base=api_base,
        api_key=api_key,
        extra_headers=extra_headers,
        max_retries=max_retries,
        retry_base_delay_s=retry_base_delay_s,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )
