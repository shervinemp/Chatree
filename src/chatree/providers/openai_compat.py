from __future__ import annotations

import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from chatree.providers.mock import MockLLMProvider, estimate_messages_tokens, estimate_tokens
from chatree.types import LLMResponse, LLMStreamEvent, LLMToolCall


@dataclass(frozen=True)
class ProviderSpec:
    """Normalized endpoint & credential metadata for a model prefix."""

    provider: str
    model_name: str
    api_key_env: str | None = None
    default_base_url: str | None = None
    base_url_env: str | None = None


_PROVIDER_PREFIX_MAP: dict[str, tuple[str | None, str | None, str | None]] = {
    # prefix -> (api_key_env, default_openai_compat_base_url, base_url_env)
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "OPENAI_API_BASE"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1", "ANTHROPIC_API_BASE"),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_BASE"),
    "google": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_BASE"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "DEEPSEEK_API_BASE"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "OPENROUTER_API_BASE"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "GROQ_API_BASE"),
    "mistral": ("MISTRAL_API_KEY", "https://api.mistral.ai/v1", "MISTRAL_API_BASE"),
    "together_ai": ("TOGETHERAI_API_KEY", "https://api.together.xyz/v1", "TOGETHER_API_BASE"),
    "together": ("TOGETHERAI_API_KEY", "https://api.together.xyz/v1", "TOGETHER_API_BASE"),
    "fireworks_ai": ("FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_BASE"),
    "fireworks": ("FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_BASE"),
    "xai": ("XAI_API_KEY", "https://api.x.ai/v1", "XAI_API_BASE"),
    "perplexity": ("PERPLEXITYAI_API_KEY", "https://api.perplexity.ai", "PERPLEXITY_API_BASE"),
    "cohere": ("COHERE_API_KEY", "https://api.cohere.com/compatibility/v1", "COHERE_API_BASE"),
    "cerebras": ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "CEREBRAS_API_BASE"),
    "sambanova": ("SAMBANOVA_API_KEY", "https://api.sambanova.ai/v1", "SAMBANOVA_API_BASE"),
    "huggingface": ("HUGGINGFACE_API_KEY", "https://router.huggingface.co/v1", "HF_API_BASE"),
    "azure": ("AZURE_API_KEY", None, "AZURE_API_BASE"),
    "bedrock": ("AWS_ACCESS_KEY_ID", None, None),
    "vertex_ai": ("GOOGLE_APPLICATION_CREDENTIALS", None, None),
    "ollama": (None, "http://localhost:11434/v1", "OLLAMA_API_BASE"),
    "ollama_chat": (None, "http://localhost:11434/v1", "OLLAMA_API_BASE"),
    "vllm": ("VLLM_API_KEY", "http://localhost:8000/v1", "VLLM_API_BASE"),
    "hosted_vllm": ("VLLM_API_KEY", "http://localhost:8000/v1", "VLLM_API_BASE"),
    "sglang": ("SGLANG_API_KEY", "http://localhost:30000/v1", "SGLANG_API_BASE"),
    "lm_studio": ("LM_STUDIO_API_KEY", "http://localhost:1234/v1", "LM_STUDIO_API_BASE"),
    "llamacpp": ("LLAMACPP_API_KEY", "http://localhost:8080/v1", "LLAMACPP_API_BASE"),
    "openai_compat": ("OPENAI_API_KEY", None, "OPENAI_API_BASE"),
    "custom": ("OPENAI_API_KEY", None, "CHATREE_API_BASE"),
}


def resolve_provider_spec(model: str) -> ProviderSpec:
    """
    Parse any `<provider>/<model>` or bare model name (e.g. `claude-sonnet-4-5`, `gpt-4o`,
    `deepseek/deepseek-chat`, `ollama/llama3.2`, `vllm/Qwen/Qwen2.5-72B`, `openrouter/anthropic/...`)
    into a normalized `ProviderSpec`.
    """
    raw = (model or "").strip()
    low = raw.lower()
    if "/" in raw:
        prefix, rest = raw.split("/", 1)
        p_low = prefix.lower()
        if p_low in _PROVIDER_PREFIX_MAP:
            key_env, def_base, base_env = _PROVIDER_PREFIX_MAP[p_low]
            return ProviderSpec(
                provider=p_low,
                model_name=rest,
                api_key_env=key_env,
                default_base_url=def_base,
                base_url_env=base_env,
            )
        return ProviderSpec(provider=p_low, model_name=rest, api_key_env=f"{p_low.upper()}_API_KEY")

    # Bare model ID heuristics
    if low.startswith("claude"):
        return ProviderSpec("anthropic", raw, *_PROVIDER_PREFIX_MAP["anthropic"])
    if low.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return ProviderSpec("openai", raw, *_PROVIDER_PREFIX_MAP["openai"])
    if low.startswith("gemini"):
        return ProviderSpec("gemini", raw, *_PROVIDER_PREFIX_MAP["gemini"])
    if low.startswith("deepseek"):
        return ProviderSpec("deepseek", raw, *_PROVIDER_PREFIX_MAP["deepseek"])
    if low.startswith(("mistral", "codestral", "pixtral", "ministral")):
        return ProviderSpec("mistral", raw, *_PROVIDER_PREFIX_MAP["mistral"])
    if low.startswith("grok"):
        return ProviderSpec("xai", raw, *_PROVIDER_PREFIX_MAP["xai"])
    return ProviderSpec(provider="openai_compat", model_name=raw, api_key_env="OPENAI_API_KEY")


def resolve_provider_endpoint(
    model: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    runtime_keys: dict[str, str] | None = None,
) -> tuple[ProviderSpec, str | None, str | None]:
    """Resolve `(ProviderSpec, resolved_base_url, resolved_api_key)` without mutating `os.environ`."""
    spec = resolve_provider_spec(model)
    rkeys = runtime_keys or {}

    resolved_key = api_key
    if not resolved_key and spec.api_key_env:
        resolved_key = rkeys.get(spec.api_key_env) or os.environ.get(spec.api_key_env)
    if not resolved_key and spec.provider == "gemini":
        resolved_key = rkeys.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_key and spec.provider in ("together", "together_ai"):
        resolved_key = rkeys.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_API_KEY")

    resolved_base = api_base
    if not resolved_base and spec.base_url_env:
        resolved_base = os.environ.get(spec.base_url_env)
    if not resolved_base:
        resolved_base = spec.default_base_url
    return spec, resolved_base, resolved_key


def _safe_parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError as exc:
        return {"_raw_arguments": str(raw), "_parse_error": str(exc)}


def _extract_usage_from_dict(
    usage: dict[str, Any] | None, messages: list[dict[str, Any]], content: str
) -> tuple[int, int, int]:
    if not isinstance(usage, dict):
        return estimate_messages_tokens(messages), estimate_tokens(content), 0
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


class OpenAICompatibleProvider:
    """
    Zero-extra-dependency, provider-agnostic LLM backend speaking the standard OpenAI
    `/v1/chat/completions` HTTP + SSE protocol.

    Works out-of-the-box (without installing `litellm`) with:
      - Local & self-hosted engines: Ollama, vLLM, SGLang, LM Studio, llama.cpp server
      - Hosted APIs: OpenAI, OpenRouter, DeepSeek, Groq, Together, Fireworks, Mistral, xAI,
        Cerebras, SambaNova, HuggingFace Router, Gemini OpenAI-compat, and custom proxies
      - Transparent `mock/*` delegation for zero-key offline testing
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        max_retries: int = 4,
        retry_base_delay_s: float = 1.0,
        timeout_s: float = 90.0,
        max_tokens: int | None = None,
    ) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.extra_headers = dict(extra_headers or {})
        self.max_retries = max(0, max_retries)
        self.retry_base_delay_s = max(0.05, retry_base_delay_s)
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._runtime_keys: dict[str, str] = {}
        self._mock = MockLLMProvider()

    def set_runtime_keys(self, keys: dict[str, str]) -> None:
        for k, v in keys.items():
            if v is not None and str(v).strip():
                self._runtime_keys[k] = str(v).strip()

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

    def _build_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        stream: bool,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        include_stream_usage: bool = True,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        spec, base_url, key = resolve_provider_endpoint(
            model,
            api_base=self.api_base,
            api_key=self.api_key,
            runtime_keys=self._runtime_keys,
        )
        if not base_url:
            raise ValueError(
                f"No base URL resolved for model '{model}'. Pass `api_base='https://.../v1'` "
                "or set `CHATREE_API_BASE`."
            )
        clean_base = base_url.rstrip("/")
        url = clean_base if clean_base.endswith("/chat/completions") else f"{clean_base}/chat/completions"

        # OpenRouter expects full upstream slug (e.g. `anthropic/claude-sonnet-4`); strip only `openrouter/`
        wire_model = spec.model_name
        body: dict[str, Any] = {
            "model": wire_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        eff_max = max_tokens if max_tokens is not None else self.max_tokens
        if eff_max is not None:
            body["max_tokens"] = eff_max
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        if stream and include_stream_usage and spec.provider not in ("ollama", "ollama_chat"):
            body["stream_options"] = {"include_usage": True}

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            **self.extra_headers,
        }
        if key:
            headers.setdefault("Authorization", f"Bearer {key}")
        elif spec.provider in ("ollama", "ollama_chat", "vllm", "hosted_vllm", "sglang", "lm_studio", "llamacpp"):
            headers.setdefault("Authorization", "Bearer local")
        return url, headers, body

    @staticmethod
    def _is_retryable_http(exc: Exception) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code in (429, 500, 502, 503, 504)
        if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
            return True
        msg = str(exc).lower()
        return any(tok in msg for tok in ("429", "502", "503", "504", "timed out", "temporarily unavailable"))

    async def _post_json(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")

        def _sync_call() -> dict[str, Any]:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                    parsed: dict[str, Any] = json.loads(raw)
                    return parsed
            except urllib.error.HTTPError as err:
                err_body = ""
                try:
                    err_body = err.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                raise urllib.error.HTTPError(
                    err.url, err.code, f"{err.reason}: {err_body}", err.headers, None
                ) from None

        attempt = 0
        while True:
            try:
                return await asyncio.to_thread(_sync_call)
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_retryable_http(exc):
                    raise
                delay = self.retry_base_delay_s * (2**attempt) + random.uniform(0.0, 0.35)
                attempt += 1
                await asyncio.sleep(delay)

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
        del fork_breakpoint_index
        if model.startswith("mock/"):
            return await self._mock.complete(
                model,
                messages,
                tools=None if tool_choice == "none" else tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        url, headers, body = self._build_request(
            model,
            messages,
            tools,
            temperature,
            stream=False,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
        t0 = time.perf_counter()
        raw = await self._post_json(url, headers, body)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        choices = raw.get("choices") or []
        msg = choices[0].get("message", {}) if choices else {}
        content = str(msg.get("content") or "")
        parsed_tools: list[LLMToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            parsed_tools.append(
                LLMToolCall(
                    id=str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
                    name=str(fn.get("name") or ""),
                    arguments=_safe_parse_args(fn.get("arguments")),
                )
            )
        t_in, t_out, cache_read = _extract_usage_from_dict(raw.get("usage"), messages, content)
        return LLMResponse(
            content=content,
            tool_calls=parsed_tools,
            model_used=str(raw.get("model") or model),
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
        del fork_breakpoint_index
        if model.startswith("mock/"):
            async for ev in self._mock.stream_complete(
                model, messages, tools=tools, temperature=temperature
            ):
                yield ev
            return

        url, headers, body = self._build_request(
            model,
            messages,
            tools,
            temperature,
            stream=True,
            max_tokens=max_tokens,
        )
        t0 = time.perf_counter()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _stream_worker(req_body: dict[str, Any]) -> None:
            payload = json.dumps(req_body).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    ctype = (resp.headers.get("Content-Type") or "").lower()
                    if "application/json" in ctype and "text/event-stream" not in ctype:
                        obj = json.loads(resp.read().decode("utf-8"))
                        loop.call_soon_threadsafe(queue.put_nowait, ("json_body", obj))
                        return
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
                            except json.JSONDecodeError:
                                continue
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("eof", None))

        worker_fut = loop.run_in_executor(None, _stream_worker, body)
        content_parts: list[str] = []
        tool_accum: dict[int, dict[str, str]] = {}
        usage_obj: dict[str, Any] | None = None
        model_used = model

        try:
            while True:
                kind, payload = await queue.get()
                if kind == "eof":
                    break
                if kind == "error":
                    # Retry once without `stream_options` if an OpenAI-compat server rejected it with HTTP 400/422
                    if (
                        isinstance(payload, urllib.error.HTTPError)
                        and payload.code in (400, 422)
                        and "stream_options" in body
                    ):
                        await worker_fut
                        fallback_resp = await self.complete(
                            model,
                            messages,
                            tools=tools,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        if fallback_resp.content:
                            yield LLMStreamEvent(kind="delta", delta=fallback_resp.content)
                        yield LLMStreamEvent(kind="done", response=fallback_resp)
                        return
                    raise payload
                if kind == "json_body":
                    # Some servers return a single non-SSE JSON completion even when stream=True
                    choices = payload.get("choices") or []
                    msg = choices[0].get("message", {}) if choices else {}
                    txt = str(msg.get("content") or "")
                    if txt:
                        content_parts.append(txt)
                        yield LLMStreamEvent(kind="delta", delta=txt)
                    for idx, tc in enumerate(msg.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        tool_accum[idx] = {
                            "id": str(tc.get("id") or ""),
                            "name": str(fn.get("name") or ""),
                            "arguments": str(fn.get("arguments") or ""),
                        }
                    if isinstance(payload.get("usage"), dict):
                        usage_obj = payload["usage"]
                    if payload.get("model"):
                        model_used = str(payload["model"])
                    continue
                if kind == "chunk":
                    if payload.get("model"):
                        model_used = str(payload["model"])
                    if isinstance(payload.get("usage"), dict):
                        usage_obj = payload["usage"]
                    choices = payload.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        txt = str(delta["content"])
                        content_parts.append(txt)
                        yield LLMStreamEvent(kind="delta", delta=txt)
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = int(tc_delta.get("index") or 0)
                        slot = tool_accum.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc_delta.get("id"):
                            slot["id"] = str(tc_delta["id"])
                        fn = tc_delta.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += str(fn["name"])
                        if fn.get("arguments"):
                            slot["arguments"] += str(fn["arguments"])
        finally:
            await worker_fut

        full_content = "".join(content_parts)
        parsed_tools: list[LLMToolCall] = []
        for idx in sorted(tool_accum.keys()):
            slot = tool_accum[idx]
            parsed_tools.append(
                LLMToolCall(
                    id=slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
                    name=slot["name"],
                    arguments=_safe_parse_args(slot["arguments"]),
                )
            )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        t_in, t_out, cache_read = _extract_usage_from_dict(usage_obj, messages, full_content)
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


class CallableLLMProvider:
    """
    Adapter letting users plug ANY sync or async Python function (e.g., native Anthropic SDK,
    Google GenAI SDK, AWS Bedrock, HuggingFace Transformers, LangChain, or custom router)
    directly into Chatree as an `LLMProvider` in 3 lines of code.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
    ) -> None:
        self._fn = fn

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> LLMResponse:
        import inspect

        sig = inspect.signature(self._fn)
        call_kw: dict[str, Any] = {}
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        for k, v in (
            ("model", model),
            ("messages", messages),
            ("tools", tools),
            ("temperature", temperature),
            *kwargs.items(),
        ):
            if k in params or has_varkw:
                call_kw[k] = v

        res = self._fn(**call_kw)
        if inspect.isawaitable(res):
            res = await res
        if isinstance(res, LLMResponse):
            return res
        if isinstance(res, str):
            return LLMResponse(
                content=res,
                model_used=model,
                tokens_in=estimate_messages_tokens(messages),
                tokens_out=estimate_tokens(res),
            )
        if isinstance(res, dict):
            return LLMResponse(**res)
        raise TypeError(
            f"CallableLLMProvider function must return LLMResponse, str, or dict; got {type(res).__name__}."
        )

    async def stream_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        resp = await self.complete(model, messages, tools=tools, temperature=temperature, **kwargs)
        if resp.content:
            yield LLMStreamEvent(kind="delta", delta=resp.content)
        yield LLMStreamEvent(kind="done", response=resp)
