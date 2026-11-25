from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from typing import Any, AsyncIterator

from chatree.providers.mock import MockLLMProvider, estimate_messages_tokens, estimate_tokens
from chatree.providers.openai_compat import _PROVIDER_PREFIX_MAP, resolve_provider_endpoint
from chatree.types import LLMResponse, LLMStreamEvent, LLMToolCall


def _is_retryable_exception(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    retryable_names = (
        "ratelimit",
        "timeout",
        "apiconnection",
        "serviceunavailable",
        "internalserver",
        "badgateway",
        "overloaded",
    )
    if any(k in name for k in retryable_names):
        return True
    return any(
        tok in msg
        for tok in ("429", "500", "502", "503", "504", "rate limit", "timed out", "temporarily unavailable")
    )


def _safe_parse_tool_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError as exc:
        return {"_raw_arguments": raw, "_parse_error": str(exc)}


class LiteLLMProvider:
    """
    Multi-provider LLM backend backed by LiteLLM (OpenRouter, Gemini, Anthropic, OpenAI,
    Ollama, and OpenAI-compatible endpoints) with transparent delegation to MockLLMProvider
    for `mock/*` model IDs.
    """

    def __init__(
        self,
        api_base: str | None = None,
        *,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        enable_cache_control: bool = True,
        max_retries: int = 4,
        retry_base_delay_s: float = 1.0,
        timeout_s: float = 90.0,
        max_tokens: int | None = None,
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.extra_headers = dict(extra_headers or {})
        self.enable_cache_control = enable_cache_control
        self.max_retries = max(0, max_retries)
        self.retry_base_delay_s = max(0.05, retry_base_delay_s)
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._runtime_keys: dict[str, str] = {}
        self._mock = MockLLMProvider()

    async def _call_with_retry(self, fn: Any, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            try:
                return await fn(**kwargs)
            except Exception as exc:
                if attempt >= self.max_retries or not _is_retryable_exception(exc):
                    raise
                delay = self.retry_base_delay_s * (2**attempt) + random.uniform(0.0, 0.35)
                attempt += 1
                await asyncio.sleep(delay)

    def set_runtime_keys(self, keys: dict[str, str]) -> None:
        """Store provider API keys strictly in-instance memory (never mutating process os.environ)."""
        for k, v in keys.items():
            if v is not None and str(v).strip():
                self._runtime_keys[k] = str(v).strip()

    def _resolve_api_key(self, model: str) -> str | None:
        _, _, key = resolve_provider_endpoint(
            model,
            api_base=self.api_base,
            api_key=self.api_key,
            runtime_keys=self._runtime_keys,
        )
        return key

    def configured_providers(self) -> dict[str, bool]:
        status: dict[str, bool] = {"mock": True}
        for name, (key_env, _, base_env) in _PROVIDER_PREFIX_MAP.items():
            if name in ("google", "together", "fireworks", "ollama_chat", "hosted_vllm", "openai_compat", "custom"):
                continue
            has_key = bool(key_env and (os.environ.get(key_env) or self._runtime_keys.get(key_env)))
            has_base = bool(base_env and os.environ.get(base_env))
            status[name] = has_key or has_base
        if self.api_base:
            status["custom_api_base"] = True
        return status

    def _import_litellm(self) -> Any:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLM is not installed. Install with `pip install chatree[litellm]` "
                "or pass a custom `LLMProvider` implementation."
            ) from exc
        litellm.suppress_debug_info = True
        return litellm

    @staticmethod
    def apply_cache_control(
        model: str,
        messages: list[dict[str, Any]],
        *,
        fork_breakpoint_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Annotate static prefix boundaries with Anthropic ephemeral `cache_control` breakpoints:
          1) `system` prompt (index 0)
          2) Optional `fork` split boundary (`fork_breakpoint_index`) so sibling forks share a warm trunk
          3) Latest non-empty committed message before the tail (`len - 2` walking backward)
        Combined with the `tools[-1]` breakpoint, this uses at most Anthropic's 4-breakpoint cap.
        """
        low = model.lower()
        if "anthropic" not in low and "claude" not in low:
            return messages
        if not messages:
            return messages

        def _mark(msg: dict[str, Any]) -> dict[str, Any]:
            content = msg.get("content")
            if isinstance(content, str) and content:
                cloned = dict(msg)
                cloned["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
                return cloned
            return msg

        annotated = [dict(m) for m in messages]
        # Breakpoint 1: static system prompt (shared across root + all fork children)
        if annotated[0].get("role") == "system":
            annotated[0] = _mark(annotated[0])

        # Breakpoint 2: walk backward from len-2 to find the latest non-empty cacheable message
        tail_bp_idx: int | None = None
        if len(annotated) >= 3:
            for idx in range(len(annotated) - 2, 0, -1):
                role = annotated[idx].get("role")
                content = annotated[idx].get("content")
                if role in ("user", "assistant", "tool") and isinstance(content, str) and content:
                    annotated[idx] = _mark(annotated[idx])
                    tail_bp_idx = idx
                    break

        # Breakpoint 3 (4th overall including tools[-1]): fork split boundary for sibling CoW sharing
        if (
            fork_breakpoint_index is not None
            and 1 <= fork_breakpoint_index < len(annotated) - 1
            and fork_breakpoint_index != tail_bp_idx
        ):
            for idx in range(fork_breakpoint_index, 0, -1):
                if idx == tail_bp_idx:
                    break
                role = annotated[idx].get("role")
                content = annotated[idx].get("content")
                if role in ("user", "assistant", "tool") and isinstance(content, str) and content:
                    annotated[idx] = _mark(annotated[idx])
                    break
        return annotated

    @staticmethod
    def apply_tools_cache_control(
        model: str, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Annotate the final tool schema with Anthropic ephemeral cache_control."""
        if not tools:
            return tools
        low = model.lower()
        if "anthropic" not in low and "claude" not in low:
            return tools
        cloned = [dict(t) for t in tools]
        cloned[-1] = {**cloned[-1], "cache_control": {"type": "ephemeral"}}
        return cloned

    def _build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        stream: bool,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        fork_breakpoint_index: int | None = None,
    ) -> dict[str, Any]:
        wire_messages = (
            self.apply_cache_control(
                model, messages, fork_breakpoint_index=fork_breakpoint_index
            )
            if self.enable_cache_control
            else messages
        )
        wire_tools = (
            self.apply_tools_cache_control(model, tools)
            if (self.enable_cache_control and tools)
            else tools
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": wire_messages,
            "temperature": temperature,
            "stream": stream,
            "timeout": self.timeout_s,
        }
        eff_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if eff_max_tokens is not None:
            kwargs["max_tokens"] = eff_max_tokens
        api_key = self._resolve_api_key(model)
        if api_key:
            kwargs["api_key"] = api_key
        if self.extra_headers:
            kwargs["extra_headers"] = dict(self.extra_headers)
        if wire_tools:
            kwargs["tools"] = wire_tools
            kwargs["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        # Forward explicit `api_base` to ANY provider/proxy (OpenAI, Ollama, vLLM, SGLang, DeepSeek,
        # Azure, OpenRouter, LM Studio, self-hosted gateways) instead of whitelisting 3 prefixes.
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if stream and not model.lower().startswith(("ollama/", "ollama_chat/", "cohere/")):
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    @staticmethod
    def _extract_usage(
        usage: Any, messages: list[dict[str, Any]], content: str
    ) -> tuple[int, int, int]:
        if usage is None:
            return estimate_messages_tokens(messages), estimate_tokens(content), 0
        if isinstance(usage, dict):
            tokens_in = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            tokens_out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
            if not isinstance(details, dict):
                details = {}
            cache_read = int(
                details.get("cached_tokens")
                or usage.get("cache_read_input_tokens")
                or usage.get("prompt_cache_hit_tokens")
                or usage.get("cached_content_token_count")
                or 0
            )
            return tokens_in, tokens_out, cache_read

        tokens_in = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(
            getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
        )
        details = getattr(usage, "prompt_tokens_details", None) or getattr(
            usage, "input_tokens_details", None
        )
        cache_read = 0
        if details is not None:
            cache_read = int(
                (
                    details.get("cached_tokens")
                    if isinstance(details, dict)
                    else getattr(details, "cached_tokens", 0)
                )
                or 0
            )
        if not cache_read:
            cache_read = int(
                getattr(usage, "cache_read_input_tokens", 0)
                or getattr(usage, "prompt_cache_hit_tokens", 0)
                or getattr(usage, "cached_content_token_count", 0)
                or 0
            )
        return tokens_in, tokens_out, cache_read

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        fork_breakpoint_index: int | None = None,
    ) -> LLMResponse:
        if model.startswith("mock/"):
            return await self._mock.complete(
                model,
                messages,
                tools=None if tool_choice == "none" else tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        litellm = self._import_litellm()
        t0 = time.perf_counter()
        raw = await self._call_with_retry(
            litellm.acompletion,
            **self._build_kwargs(
                model,
                messages,
                tools,
                temperature,
                stream=False,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                fork_breakpoint_index=fork_breakpoint_index,
            ),
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        msg = raw.choices[0].message
        content = msg.content or ""
        parsed_tools: list[LLMToolCall] = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                args = _safe_parse_tool_args(tc.function.arguments)
                parsed_tools.append(
                    LLMToolCall(
                        id=tc.id or f"call_{uuid.uuid4().hex[:8]}",
                        name=tc.function.name,
                        arguments=args,
                    )
                )
        t_in, t_out, cache_read = self._extract_usage(
            getattr(raw, "usage", None), messages, content
        )
        return LLMResponse(
            content=content,
            tool_calls=parsed_tools,
            model_used=getattr(raw, "model", None) or model,
            tokens_in=t_in,
            tokens_out=t_out,
            cache_read_tokens=cache_read,
            latency_ms=latency_ms,
        )

    async def stream_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        *,
        max_tokens: int | None = None,
        fork_breakpoint_index: int | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        if model.startswith("mock/"):
            async for ev in self._mock.stream_complete(
                model, messages, tools=tools, temperature=temperature
            ):
                yield ev
            return

        litellm = self._import_litellm()
        t0 = time.perf_counter()
        stream = await self._call_with_retry(
            litellm.acompletion,
            **self._build_kwargs(
                model,
                messages,
                tools,
                temperature,
                stream=True,
                max_tokens=max_tokens,
                fork_breakpoint_index=fork_breakpoint_index,
            ),
        )

        content_parts: list[str] = []
        tool_accum: dict[int, dict[str, str]] = {}
        usage_obj: Any = None
        model_used = model

        async for chunk in stream:
            if getattr(chunk, "model", None):
                model_used = chunk.model
            if getattr(chunk, "usage", None):
                usage_obj = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            if getattr(delta, "content", None):
                text = delta.content
                content_parts.append(text)
                yield LLMStreamEvent(kind="delta", delta=text)
            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = getattr(tc_delta, "index", 0) or 0
                    slot = tool_accum.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc_delta, "id", None):
                        slot["id"] = tc_delta.id
                    fn = getattr(tc_delta, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["arguments"] += fn.arguments

        full_content = "".join(content_parts)
        parsed_tools: list[LLMToolCall] = []
        for idx in sorted(tool_accum.keys()):
            slot = tool_accum[idx]
            args = _safe_parse_tool_args(slot["arguments"])
            parsed_tools.append(
                LLMToolCall(
                    id=slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
                    name=slot["name"],
                    arguments=args,
                )
            )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        t_in, t_out, cache_read = self._extract_usage(usage_obj, messages, full_content)
        yield LLMStreamEvent(
            kind="done",
            response=LLMResponse(
                content=full_content,
                tool_calls=parsed_tools,
                model_used=model_used,
                tokens_in=t_in,
                tokens_out=t_out,
                cache_read_tokens=cache_read,
                latency_ms=latency_ms,
            ),
        )
