from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any

import pytest

from chatree import (
    CallableLLMProvider,
    Chatree,
    ChatreeConfig,
    CostModel,
    LiteLLMProvider,
    Mem0OfficialAdapter,
    OpenAICompatibleEmbedder,
    OpenAICompatibleProvider,
    RLMOfficialAdapter,
    build_embedder,
    build_llm_provider,
    resolve_provider_endpoint,
)
from chatree.types import LLMResponse


def test_resolve_provider_endpoint_across_hosted_and_local_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k_openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k_anthropic")
    monkeypatch.setenv("GOOGLE_API_KEY", "k_google")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k_groq")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k_or")

    s1, b1, k1 = resolve_provider_endpoint("openai/gpt-4o")
    assert (s1.provider, s1.model_name, b1, k1) == ("openai", "gpt-4o", "https://api.openai.com/v1", "k_openai")

    s2, _, k2 = resolve_provider_endpoint("claude-sonnet-4-5")
    assert (s2.provider, s2.model_name, k2) == ("anthropic", "claude-sonnet-4-5", "k_anthropic")

    s3, b3, k3 = resolve_provider_endpoint("gemini/gemini-2.5-pro")
    assert s3.provider == "gemini" and k3 == "k_google" and "googleapis.com" in (b3 or "")

    s4, b4, k4 = resolve_provider_endpoint("openrouter/anthropic/claude-sonnet-4")
    assert (s4.provider, s4.model_name, b4, k4) == (
        "openrouter",
        "anthropic/claude-sonnet-4",
        "https://openrouter.ai/api/v1",
        "k_or",
    )

    s5, b5, k5 = resolve_provider_endpoint("groq/llama-3.3-70b-versatile")
    assert (s5.provider, s5.model_name, k5) == ("groq", "llama-3.3-70b-versatile", "k_groq")
    assert "groq.com" in (b5 or "")

    s6, b6, _ = resolve_provider_endpoint("ollama/llama3.2")
    assert (s6.provider, s6.model_name, b6) == ("ollama", "llama3.2", "http://localhost:11434/v1")

    s7, b7, _ = resolve_provider_endpoint("vllm/Qwen/Qwen2.5-72B-Instruct")
    assert (s7.provider, s7.model_name, b7) == ("vllm", "Qwen/Qwen2.5-72B-Instruct", "http://localhost:8000/v1")

    s8, b8, _ = resolve_provider_endpoint("sglang/meta-llama/Llama-3.3-70B")
    assert (s8.provider, b8) == ("sglang", "http://localhost:30000/v1")


@pytest.mark.asyncio
async def test_openai_compatible_provider_complete_stream_tools_and_embeddings_live_http() -> None:
    """
    Verify zero-LiteLLM `OpenAICompatibleProvider` and `OpenAICompatibleEmbedder` end-to-end
    against a local HTTP/SSE OpenAI-compatible server (simulating vLLM / Ollama / LM Studio / proxy).
    """
    seen_headers: list[dict[str, str]] = []
    seen_bodies: list[dict[str, Any]] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            body = json.loads(raw)
            seen_headers.append({k.lower(): v for k, v in self.headers.items()})
            seen_bodies.append(body)

            if self.path.endswith("/embeddings"):
                resp = {
                    "data": [
                        {"index": i, "embedding": [0.5, 0.5, 0.5, 0.5]}
                        for i in range(len(body.get("input") or []))
                    ]
                }
                encoded = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {"model": body["model"], "choices": [{"delta": {"content": "Hello "}}]},
                    {"model": body["model"], "choices": [{"delta": {"content": "from local vLLM!"}}]},
                    {
                        "model": body["model"],
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 120,
                            "completion_tokens": 12,
                            "prompt_tokens_details": {"cached_tokens": 96},
                        },
                    },
                ]
                for ch in chunks:
                    self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            resp = {
                "model": body["model"],
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Sync completion OK",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "ping", "arguments": '{"n": 7}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_cache_hit_tokens": 80,
                },
            }
            encoded = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base_url = f"http://127.0.0.1:{port}/v1"
        prov = OpenAICompatibleProvider(
            api_base=base_url,
            api_key="test-key",
            extra_headers={"X-Custom-Tenant": "chatree-test"},
        )

        # 1. Sync completion + tool call + cache hit usage
        r = await prov.complete(
            model="vllm/Qwen/Qwen2.5-7B",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}],
        )
        assert r.content == "Sync completion OK"
        assert len(r.tool_calls) == 1 and r.tool_calls[0].name == "ping" and r.tool_calls[0].arguments == {"n": 7}
        assert (r.tokens_in, r.tokens_out, r.cache_read_tokens) == (100, 20, 80)
        assert seen_bodies[-1]["model"] == "Qwen/Qwen2.5-7B"
        assert seen_headers[-1]["x-custom-tenant"] == "chatree-test"
        assert seen_headers[-1]["authorization"] == "Bearer test-key"

        # 2. Full Chatree SDK turn via SSE streaming with zero LiteLLM!
        cfg = ChatreeConfig(
            db_path=":memory:",
            llm_provider="openai_compat",
            default_model="vllm/Qwen/Qwen2.5-7B",
            api_base=base_url,
            api_key="test-key",
        )
        async with Chatree.in_memory(config=cfg) as ct:
            assert isinstance(ct.llm, OpenAICompatibleProvider)
            tree = await ct.create_tree("Zero-LiteLLM Tree")
            turn = await tree.chat("Say hello", include_core_tools=False)
            assert turn.content == "Hello from local vLLM!"
            metrics = await tree.metrics()
            assert metrics.total_cache_read_tokens == 96

        # 3. Zero-LiteLLM OpenAICompatibleEmbedder
        emb = build_embedder("openai_compat/nomic-embed-text", api_base=base_url, api_key="test-key")
        assert isinstance(emb, OpenAICompatibleEmbedder)
        vecs = await emb.embed(["alpha", "beta"])
        assert vecs == [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
        assert emb.dimensions == 4
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.asyncio
async def test_callable_llm_provider_wraps_any_sync_or_async_fn() -> None:
    sync_llm = CallableLLMProvider(lambda model, messages: f"[{model}] echo: {messages[-1]['content'][-12:]}")
    async with Chatree.in_memory(llm=sync_llm) as ct:
        tree = await ct.create_tree("Callable Provider")
        turn = await tree.chat("hello custom", include_core_tools=False)
        assert "hello custom" in turn.content

    async def _async_fn(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="async-sdk-ok", model_used=model, tokens_in=42, tokens_out=5, cache_read_tokens=20)

    async_llm = CallableLLMProvider(_async_fn)
    resp = await async_llm.complete("custom/my-model", [{"role": "user", "content": "ping"}])
    assert resp.content == "async-sdk-ok" and resp.cache_read_tokens == 20


def test_build_llm_provider_auto_fallback_without_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    import chatree.providers.factory as fac_mod

    monkeypatch.setattr(fac_mod, "_has_litellm", lambda: False)
    p_auto = build_llm_provider("auto", model="openai/gpt-4o", api_base="http://localhost:8000/v1")
    assert isinstance(p_auto, OpenAICompatibleProvider)

    monkeypatch.setattr(fac_mod, "_has_litellm", lambda: True)
    p_lite = build_llm_provider("auto", model="anthropic/claude-sonnet-4-5")
    assert isinstance(p_lite, LiteLLMProvider)

    p_sglang = build_llm_provider("auto", model="sglang/Qwen/Qwen2.5-72B")
    assert isinstance(p_sglang, OpenAICompatibleProvider)


def test_litellm_provider_forwards_api_base_and_extracts_all_vendor_cache_usage() -> None:
    p = LiteLLMProvider(
        api_base="https://my-gateway.internal/v1",
        api_key="gw_key",
        extra_headers={"X-Org": "research"},
    )
    kw = p._build_kwargs(
        "openrouter/anthropic/claude-sonnet-4",
        [{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.1,
        stream=False,
    )
    assert kw["api_base"] == "https://my-gateway.internal/v1"
    assert kw["api_key"] == "gw_key"
    assert kw["extra_headers"] == {"X-Org": "research"}

    # Verify cache extraction across OpenAI, Anthropic, DeepSeek, and Gemini usage shapes
    assert LiteLLMProvider._extract_usage({"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 70}}, [], "") == (100, 10, 70)
    assert LiteLLMProvider._extract_usage({"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 60}, [], "") == (100, 10, 60)
    assert LiteLLMProvider._extract_usage({"prompt_tokens": 100, "completion_tokens": 10, "prompt_cache_hit_tokens": 50}, [], "") == (100, 10, 50)
    assert LiteLLMProvider._extract_usage({"prompt_tokens": 100, "completion_tokens": 10, "cached_content_token_count": 40}, [], "") == (100, 10, 40)


def test_eval_adapters_mem0_rlm_and_pricing_are_provider_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k_anth")
    monkeypatch.setenv("GEMINI_API_KEY", "k_gem")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k_ds")

    # Mem0 config mapping across arbitrary providers (always paired with local FastEmbed)
    c_anth = Mem0OfficialAdapter.build_provider_config("anthropic/claude-sonnet-4-5", "/tmp/m1")
    assert c_anth["llm"]["provider"] == "anthropic"
    assert c_anth["llm"]["config"]["model"] == "claude-sonnet-4-5"
    assert c_anth["embedder"]["provider"] == "fastembed"

    c_oll = Mem0OfficialAdapter.build_provider_config("ollama/llama3.2", "/tmp/m2")
    assert c_oll["llm"]["provider"] == "ollama"
    assert c_oll["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"

    c_or = Mem0OfficialAdapter.build_provider_config("openrouter/openai/gpt-4o-mini", "/tmp/m3")
    assert c_or["llm"]["provider"] == "openai"
    assert "openrouter.ai" in c_or["llm"]["config"]["openai_base_url"]

    # RLM backend mapping across arbitrary providers
    b_anth, kw_anth = RLMOfficialAdapter.resolve_rlm_backend_args("anthropic/claude-sonnet-4-5")
    assert b_anth == "anthropic" and kw_anth["model_name"] == "claude-sonnet-4-5" and kw_anth["api_key"] == "k_anth"

    b_gem, kw_gem = RLMOfficialAdapter.resolve_rlm_backend_args("gemini/gemini-2.5-pro")
    assert b_gem == "gemini" and kw_gem["model_name"] == "gemini-2.5-pro"

    b_vllm, kw_vllm = RLMOfficialAdapter.resolve_rlm_backend_args("vllm/Qwen/Qwen2.5-72B")
    assert b_vllm == "openai" and kw_vllm["base_url"] == "http://localhost:8000/v1" and kw_vllm["api_key"] == "local"

    # Pricing recognizes zero-cost local runtimes & frontier providers
    cm = CostModel()
    assert cm.estimate_usd("vllm/Qwen/Qwen2.5-72B", tokens_in=100_000, tokens_out=10_000) == 0.0
    assert cm.estimate_usd("anthropic/claude-sonnet-4-5", tokens_in=100_000, tokens_out=10_000, cache_read_tokens=80_000) > 0.0
