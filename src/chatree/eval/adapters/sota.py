from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from chatree.eval.adapters.base import AdapterAnswer
from chatree.protocols import LLMProvider
from chatree.providers.openai_compat import resolve_provider_endpoint

_RLM_LOCAL_REPL_LOCK = threading.Lock()


class Mem0OfficialAdapter:
    """
    Thin adapter wrapping the OFFICIAL upstream `mem0ai` Python SDK (`from mem0 import Memory`)
    for apples-to-apples head-to-head comparison—zero toy reimplementation.
    Requires `pip install mem0ai`.
    """

    def __init__(
        self,
        *,
        name: str = "mem0_official",
        model: str = "openai/gpt-4o-mini",
        llm: LLMProvider | None = None,
        mem0_config: dict[str, Any] | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        top_k: int = 8,
    ) -> None:
        self._name = name
        self.model = model
        self.llm = llm
        self.mem0_config = mem0_config
        self.api_base = api_base
        self.api_key = api_key
        self.top_k = top_k
        self._mem: Any = None
        self._user_id = "eval_user"
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._session_buf: dict[str, list[dict[str, str]]] = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0
        self._ingest_cache_read_tokens = 0
        self._ingest_llm_calls = 0
        self._ingest_usage_metered = False
        self._metering_ingest = False
        self._meter_lock = threading.Lock()

    @staticmethod
    def build_provider_config(
        model: str,
        tmpdir: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Build a provider-agnostic Mem0 configuration for any LLM provider + local FastEmbed."""
        spec, resolved_base, resolved_key = resolve_provider_endpoint(
            model, api_base=api_base, api_key=api_key
        )
        # Native Mem0 LLM providers; everything else routes via OpenAI-compatible base_url or litellm
        native_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "gemini": "gemini",
            "google": "gemini",
            "deepseek": "deepseek",
            "groq": "groq",
            "together": "together",
            "together_ai": "together",
            "ollama": "ollama",
            "ollama_chat": "ollama",
            "azure": "azure_openai",
            "bedrock": "aws_bedrock",
            "vllm": "vllm",
            "hosted_vllm": "vllm",
        }
        mem0_provider = native_map.get(spec.provider)
        llm_cfg: dict[str, Any] = {"model": spec.model_name, "temperature": 0.0}
        if mem0_provider:
            if resolved_key:
                llm_cfg["api_key"] = resolved_key
            if mem0_provider == "ollama" and resolved_base:
                llm_cfg["ollama_base_url"] = resolved_base.removesuffix("/v1")
            elif mem0_provider in ("openai", "vllm") and resolved_base:
                llm_cfg["openai_base_url"] = resolved_base
            llm_block = {"provider": mem0_provider, "config": llm_cfg}
        elif resolved_base:
            llm_cfg["openai_base_url"] = resolved_base
            if resolved_key:
                llm_cfg["api_key"] = resolved_key
            llm_block = {"provider": "openai", "config": llm_cfg}
        else:
            llm_block = {"provider": "litellm", "config": {"model": model, "temperature": 0.0}}

        return {
            "llm": llm_block,
            "embedder": {
                "provider": "fastembed",
                "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(Path(tmpdir) / "qdrant"),
                    "embedding_model_dims": 384,
                },
            },
            "history_db_path": str(Path(tmpdir) / "mem0_history.db"),
        }

    @property
    def name(self) -> str:
        return self._name

    def get_ingest_usage(self) -> tuple[int, int]:
        return self._ingest_tokens_in, self._ingest_tokens_out

    def get_ingest_metering_info(self) -> dict[str, Any]:
        """Return exact write-path metering metadata (client-metered vs. fallback heuristic)."""
        return {
            "tokens_in": self._ingest_tokens_in,
            "tokens_out": self._ingest_tokens_out,
            "cache_read_tokens": self._ingest_cache_read_tokens,
            "llm_calls": self._ingest_llm_calls,
            "metered": self._ingest_usage_metered,
        }

    def _install_llm_usage_meter(self) -> bool:
        """
        Wrap `self._mem.llm.client.chat.completions.create` so every internal `mem0ai`
        write-path LLM call records exact provider-reported `prompt_tokens`, `completion_tokens`,
        and cached input tokens (DeepSeek `prompt_cache_hit_tokens` or OpenAI
        `prompt_tokens_details.cached_tokens`). Falls back cleanly when a non-OpenAI client is used.
        """
        llm_obj = getattr(self._mem, "llm", None)
        client = getattr(llm_obj, "client", None)
        completions = getattr(getattr(client, "chat", None), "completions", None)
        orig_create = getattr(completions, "create", None)
        if completions is None or not callable(orig_create):
            return False
        if getattr(orig_create, "_chatree_metered", False):
            return True

        def _extract_usage_counts(resp: Any) -> tuple[int, int, int] | None:
            usage = getattr(resp, "usage", None)
            if usage is None and isinstance(resp, dict):
                usage = resp.get("usage")
            if usage is None:
                return None

            def _get_field(obj: Any, key: str) -> Any:
                if isinstance(obj, dict):
                    return obj.get(key)
                return getattr(obj, key, None)

            p_tok = int(_get_field(usage, "prompt_tokens") or 0)
            c_tok = int(_get_field(usage, "completion_tokens") or 0)
            cache_tok = int(_get_field(usage, "prompt_cache_hit_tokens") or 0)
            if cache_tok == 0:
                details = _get_field(usage, "prompt_tokens_details")
                if details is not None:
                    cache_tok = int(_get_field(details, "cached_tokens") or 0)
            return p_tok, c_tok, cache_tok

        def _metered_create(*args: Any, **kwargs: Any) -> Any:
            resp = orig_create(*args, **kwargs)
            if self._metering_ingest:
                counts = _extract_usage_counts(resp)
                if counts is not None:
                    p_tok, c_tok, cache_tok = counts
                    with self._meter_lock:
                        self._ingest_tokens_in += p_tok
                        self._ingest_tokens_out += c_tok
                        self._ingest_cache_read_tokens += cache_tok
                        self._ingest_llm_calls += 1
                        self._ingest_usage_metered = True
            return resp

        setattr(_metered_create, "_chatree_metered", True)
        completions.create = _metered_create
        return True

    def clone(self) -> Mem0OfficialAdapter:
        """Create an isolated worker adapter (safe for concurrency when MEM0_TELEMETRY=false)."""
        return Mem0OfficialAdapter(
            name=self._name,
            model=self.model,
            llm=self.llm,
            mem0_config=dict(self.mem0_config) if self.mem0_config else None,
            api_base=self.api_base,
            api_key=self.api_key,
            top_k=self.top_k,
        )

    async def reset(self, conversation_id: str) -> None:
        await self.close()
        # Disable upstream Mem0's shared ~/.mem0/migrations_qdrant telemetry store so isolated
        # concurrent workers never hit portalocker contention on Windows/Linux.
        os.environ.setdefault("MEM0_TELEMETRY", "false")
        try:
            from mem0 import Memory  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Official Mem0 adapter requires the upstream package: `uv pip install mem0ai`."
            ) from exc
        self._tmpdir = tempfile.TemporaryDirectory(prefix="chatree-mem0-")
        self._user_id = f"chatree_{conversation_id}"
        self._session_buf = {}
        self._ingest_tokens_in = 0
        self._ingest_tokens_out = 0
        self._ingest_cache_read_tokens = 0
        self._ingest_llm_calls = 0
        self._ingest_usage_metered = False
        self._metering_ingest = False
        cfg = (
            dict(self.mem0_config)
            if self.mem0_config
            else self.build_provider_config(
                self.model,
                self._tmpdir.name,
                api_base=self.api_base,
                api_key=self.api_key,
            )
        )
        cfg.setdefault("history_db_path", str(Path(self._tmpdir.name) / "mem0_history.db"))
        self._mem = Memory.from_config(cfg) if cfg else Memory()
        self._install_llm_usage_meter()

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del title, date_time, parent_session_id
        self._session_buf[session_id] = []

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        norm_role = "user" if role.lower() in ("user", "human", "speaker_a") else "assistant"
        stamped = f"[{date_time}] {content}" if date_time else content
        self._session_buf.setdefault(session_id, []).append({"role": norm_role, "content": stamped})

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del summary
        msgs = self._session_buf.pop(session_id, [])
        if msgs and self._mem is not None:
            calls_before = self._ingest_llm_calls
            self._metering_ingest = True
            try:
                res = await asyncio.to_thread(self._mem.add, msgs, user_id=self._user_id)
            finally:
                self._metering_ingest = False
            if self._ingest_llm_calls == calls_before:
                # Fallback heuristic ONLY when the underlying Mem0 LLM client could not be metered
                chars = sum(len(m["content"]) for m in msgs)
                self._ingest_tokens_in += max(200, chars // 4 + 350)
                out_chars = len(json.dumps(res, default=str))
                self._ingest_tokens_out += max(40, out_chars // 4)

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._mem is not None and self.llm is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question

        def _do_search() -> Any:
            try:
                return self._mem.search(
                    prompt_q, filters={"user_id": self._user_id}, top_k=self.top_k
                )
            except (TypeError, ValueError):
                try:
                    return self._mem.search(
                        prompt_q, filters={"user_id": self._user_id}, limit=self.top_k
                    )
                except (TypeError, ValueError):
                    return self._mem.search(prompt_q, user_id=self._user_id, limit=self.top_k)

        raw = await asyncio.to_thread(_do_search)
        items = raw.get("results", raw) if isinstance(raw, dict) else raw
        mem_lines = [
            f"- {it.get('memory', it) if isinstance(it, dict) else it}" for it in (items or [])
        ]
        resp = await self.llm.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the user's question concisely using the retrieved Mem0 memories.",
                },
                {
                    "role": "user",
                    "content": "[Mem0 Memories]\n" + "\n".join(mem_lines) + f"\n\nQuestion: {prompt_q}",
                },
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
            metadata={"memories_retrieved": len(mem_lines)},
        )

    async def close(self) -> None:
        if self._mem is not None:
            for attr in ("vector_store", "_telemetry_vector_store"):
                try:
                    vs = getattr(self._mem, attr, None)
                    client = getattr(vs, "client", None)
                    if client is not None and hasattr(client, "close"):
                        client.close()
                except Exception:
                    pass
            try:
                db = getattr(self._mem, "db", None)
                conn = getattr(db, "connection", None)
                if conn is not None and hasattr(conn, "close"):
                    conn.close()
            except Exception:
                pass
        self._mem = None
        self._session_buf.clear()
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None


class LettaOfficialAdapter:
    """
    Thin adapter wrapping the OFFICIAL upstream `letta_client` / `letta` SDK for apples-to-apples
    head-to-head comparison against MemGPT/Letta—zero toy reimplementation.
    Requires `pip install letta-client` (pointing at a local Letta server).
    """

    def __init__(
        self,
        *,
        name: str = "letta_official",
        base_url: str = "http://localhost:8283",
        model: str = "openai/gpt-4o-mini",
        embedding: str = "openai/text-embedding-3-small",
    ) -> None:
        self._name = name
        self.base_url = base_url
        self.model = model
        self.embedding = embedding
        self._client: Any = None
        self._agent_id: str | None = None

    @property
    def name(self) -> str:
        return self._name

    async def reset(self, conversation_id: str) -> None:
        await self.close()
        try:
            from letta_client import Letta  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Official Letta adapter requires the upstream package: `uv pip install letta-client`."
            ) from exc
        self._client = Letta(base_url=self.base_url)
        agent = await asyncio.to_thread(
            self._client.agents.create,
            name=f"chatree_eval_{conversation_id}",
            model=self.model,
            embedding=self.embedding,
        )
        self._agent_id = agent.id

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del session_id, title, date_time, parent_session_id

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        assert self._client is not None and self._agent_id is not None
        stamped = f"[{session_id} | {date_time}] {role}: {content}"
        await asyncio.to_thread(
            self._client.agents.passages.create, agent_id=self._agent_id, text=stamped
        )

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del session_id, summary

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._client is not None and self._agent_id is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        resp = await asyncio.to_thread(
            self._client.agents.messages.create,
            agent_id=self._agent_id,
            messages=[{"role": "user", "content": prompt_q}],
        )
        texts = [
            getattr(m, "content", "")
            for m in getattr(resp, "messages", [])
            if getattr(m, "message_type", "") == "assistant_message"
        ]
        usage = getattr(resp, "usage", None)
        return AdapterAnswer(
            answer="\n".join(str(t) for t in texts if t),
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    async def close(self) -> None:
        if self._client is not None and self._agent_id is not None:
            try:
                await asyncio.to_thread(self._client.agents.delete, agent_id=self._agent_id)
            except Exception:
                pass
        self._client = None
        self._agent_id = None


class GraphitiOfficialAdapter:
    """
    Thin adapter wrapping the OFFICIAL upstream `graphiti-core` (Zep bi-temporal KG engine).
    Requires `pip install graphiti-core` and a configured graph driver instance.
    """

    def __init__(
        self,
        *,
        name: str = "graphiti_official",
        graphiti_instance: Any = None,
        model: str = "deepseek/deepseek-chat",
        llm: LLMProvider | None = None,
        top_k: int = 8,
    ) -> None:
        self._name = name
        self._graphiti = graphiti_instance
        self.model = model
        self.llm = llm
        self.top_k = top_k
        self._group_id = "eval"
        self._session_buf: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return self._name

    async def reset(self, conversation_id: str) -> None:
        if self._graphiti is None:
            try:
                import graphiti_core  # type: ignore[import-not-found]  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "Official Graphiti/Zep adapter requires `uv pip install graphiti-core` "
                    "plus a connected Graphiti instance passed to `GraphitiOfficialAdapter(graphiti_instance=...)`."
                ) from exc
            raise ValueError("Pass an initialized `graphiti_instance` to GraphitiOfficialAdapter.")
        self._group_id = f"chatree_{conversation_id}"
        self._session_buf = {}

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del title, date_time, parent_session_id
        self._session_buf[session_id] = []

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        self._session_buf.setdefault(session_id, []).append(f"[{date_time}] {role}: {content}")

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del summary
        from datetime import datetime, timezone

        lines = self._session_buf.pop(session_id, [])
        if lines and self._graphiti is not None:
            await self._graphiti.add_episode(
                name=f"{self._group_id}_{session_id}",
                episode_body="\n".join(lines),
                source_description="conversation session",
                reference_time=datetime.now(timezone.utc),
                group_id=self._group_id,
            )

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        assert self._graphiti is not None and self.llm is not None
        t0 = time.perf_counter()
        prompt_q = f"[Current date: {date_time}] {question}" if date_time else question
        edges = await self._graphiti.search(prompt_q, group_ids=[self._group_id], num_results=self.top_k)
        facts = [f"- {getattr(e, 'fact', str(e))}" for e in (edges or [])]
        resp = await self.llm.complete(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer concisely using the retrieved bi-temporal knowledge graph facts.",
                },
                {
                    "role": "user",
                    "content": "[Graphiti Facts]\n" + "\n".join(facts) + f"\n\nQuestion: {prompt_q}",
                },
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
            metadata={"facts_retrieved": len(facts)},
        )

    async def close(self) -> None:
        self._session_buf.clear()




class RLMOfficialAdapter:
    """
    Thin adapter wrapping MIT's OFFICIAL upstream Recursive Language Models library
    (`pip install rlms` -> `from rlm import RLM`, Zhang, Kraska & Khattab, arXiv:2512.24601)
    for apples-to-apples comparison on identical multi-session datasets.
    """

    def __init__(
        self,
        *,
        name: str = "rlm_official",
        model: str = "openai/gpt-4o-mini",
        backend: str = "auto",
        base_url: str | None = None,
        api_key: str | None = None,
        max_depth: int = 1,
        max_iterations: int = 10,
        max_concurrent_subcalls: int = 4,
        max_timeout: float = 90.0,
    ) -> None:
        self._name = name
        self.model = model
        self.backend = backend
        self.base_url = base_url
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_concurrent_subcalls = max_concurrent_subcalls
        self.max_timeout = max_timeout
        self._sessions: list[str] = []
        self._cur_lines: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    # Note: intentionally no `clone()` because upstream `rlms` LocalREPL mutates
    # process-global `os.chdir()` and `sys.stdout/sys.stderr` during execution.

    async def reset(self, conversation_id: str) -> None:
        del conversation_id
        self._sessions = []
        self._cur_lines = []

    async def start_session(
        self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
    ) -> None:
        del parent_session_id
        hdr = f"=== Session {title or session_id}" + (f" ({date_time})" if date_time else "") + " ==="
        self._cur_lines = [hdr]

    async def ingest_turn(
        self, session_id: str, role: str, content: str, *, date_time: str = ""
    ) -> None:
        del session_id
        stamped = f"[{date_time}] {role}: {content}" if date_time else f"{role}: {content}"
        self._cur_lines.append(stamped)

    async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
        del session_id, summary
        if self._cur_lines:
            self._sessions.append("\n".join(self._cur_lines))
            self._cur_lines = []

    @staticmethod
    def resolve_rlm_backend_args(
        model: str,
        *,
        backend: str = "auto",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Resolve `(rlm_backend, backend_kwargs)` cleanly for ANY provider prefix."""
        spec, resolved_base, resolved_key = resolve_provider_endpoint(
            model, api_base=base_url, api_key=api_key
        )
        eff_backend = backend
        if eff_backend == "auto":
            if spec.provider == "anthropic":
                eff_backend = "anthropic"
            elif spec.provider in ("gemini", "google") and not base_url:
                eff_backend = "gemini"
            elif spec.provider == "openrouter":
                eff_backend = "openrouter"
            elif spec.provider in ("bedrock", "vertex_ai", "azure", "cohere"):
                eff_backend = "litellm"
            else:
                eff_backend = "openai"

        b_kwargs: dict[str, Any] = {
            "model_name": model if eff_backend == "litellm" else spec.model_name
        }
        if resolved_key:
            b_kwargs["api_key"] = resolved_key
        if eff_backend == "openai":
            if resolved_base and resolved_base != "https://api.openai.com/v1":
                b_kwargs["base_url"] = resolved_base
            elif base_url:
                b_kwargs["base_url"] = base_url
            if spec.provider in ("ollama", "ollama_chat", "vllm", "hosted_vllm", "sglang", "lm_studio", "llamacpp") and "api_key" not in b_kwargs:
                b_kwargs["api_key"] = "local"
        elif base_url:
            b_kwargs["base_url"] = base_url
        return eff_backend, b_kwargs

    def _build_rlm(self) -> Any:
        try:
            from rlm import RLM  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Official RLM adapter requires the upstream package: `uv pip install rlms`."
            ) from exc

        eff_backend, b_kwargs = self.resolve_rlm_backend_args(
            self.model, backend=self.backend, base_url=self.base_url, api_key=self.api_key
        )
        return RLM(
            backend=eff_backend,  # type: ignore[arg-type]
            backend_kwargs=b_kwargs,
            environment="local",
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_concurrent_subcalls=self.max_concurrent_subcalls,
            max_timeout=self.max_timeout,
            verbose=False,
        )

    async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
        t0 = time.perf_counter()
        if self._cur_lines:
            self._sessions.append("\n".join(self._cur_lines))
            self._cur_lines = []
        full_context = "\n\n".join(self._sessions)
        prompt_q = (
            f"{f'[Current date: {date_time}] ' if date_time else ''}"
            f"Answer concisely based on the conversation history stored in `context`. "
            f"If facts changed over time, use the latest dated update. If unmentioned, say you do not know.\n"
            f"Question: {question}"
        )

        def _run_sync() -> Any:
            with _RLM_LOCAL_REPL_LOCK:
                saved_cwd = os.getcwd()
                inst = self._build_rlm()
                try:
                    return inst.completion(prompt=full_context, root_prompt=prompt_q)
                except SystemExit as sexc:
                    raise RuntimeError(f"RLM REPL invoked SystemExit({sexc.code})") from None
                finally:
                    try:
                        os.chdir(saved_cwd)
                    except OSError:
                        pass

        try:
            comp = await asyncio.to_thread(_run_sync)
            answer_txt = str(getattr(comp, "response", "") or "")
            t_in = 0
            t_out = 0
            calls = 0
            usage = getattr(comp, "usage_summary", None)
            if usage is not None:
                for msum in (getattr(usage, "model_usage_summaries", None) or {}).values():
                    t_in += int(getattr(msum, "total_input_tokens", 0) or 0)
                    t_out += int(getattr(msum, "total_output_tokens", 0) or 0)
                    calls += int(getattr(msum, "total_calls", 0) or 0)
            peak_tok = t_in // max(1, calls) if calls else t_in
            return AdapterAnswer(
                answer=answer_txt,
                tokens_in=t_in,
                tokens_out=t_out,
                cache_read_tokens=0,
                peak_context_tokens=peak_tok,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                metadata={"rlm_calls": calls},
            )
        except (Exception, SystemExit) as exc:
            return AdapterAnswer(
                answer=f"I do not know (RLM error: {type(exc).__name__}).",
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                metadata={"error": str(exc)},
            )

    async def close(self) -> None:
        self._sessions.clear()
        self._cur_lines.clear()
