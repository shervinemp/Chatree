from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chatree import (
    BenchmarkRunner,
    Chatree,
    ChatreeConfig,
    ChatreeSystemAdapter,
    ExternalBenchmarkRunner,
    FastEmbedCrossEncoderReranker,
    FastEmbedEmbedder,
    FullContextLinearAdapter,
    GuardrailPolicy,
    LLMReranker,
    LiteLLMProvider,
    LongMemEvalDataset,
    MemorySystemAdapter,
    MockLLMProvider,
    RLMOfficialAdapter,
    TreeSummaryOnlyAdapter,
    build_embedder,
    unscripted_agentic_scenario,
)
from chatree.cli import run_cli_async
from chatree.protocols import Embedder, Reranker
from chatree.providers.litellm import _is_retryable_exception, _safe_parse_tool_args
from chatree.types import LLMResponse, LLMToolCall, SearchHit


# ---------------------------------------------------------------------------
# 1. Shadow KV-cache tool schema preservation + deterministic runtime blocking
# ---------------------------------------------------------------------------


class _ShadowToolsSpyLLM(MockLLMProvider):
    """Records exact `tools` arrays seen on active vs shadow turns and tests read-only enforcement."""

    def __init__(self, *, attempt_forbidden_in_shadow: bool = False) -> None:
        super().__init__(stream_delay_s=0.0)
        self.active_tools: list[list[dict[str, Any]] | None] = []
        self.shadow_tools: list[list[dict[str, Any]] | None] = []
        self.shadow_tool_results: list[dict[str, Any]] = []
        self.attempt_forbidden_in_shadow = attempt_forbidden_in_shadow
        self._did_forbidden_attempt = False

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        is_shadow = any(
            m.get("role") == "user"
            and ("[Shadow Query |" in str(m.get("content", "")) or "[IPC Shadow Query" in str(m.get("content", "")))
            for m in messages
        )
        if is_shadow:
            self.shadow_tools.append(tools)
            # Capture any tool responses returned to the shadow turn
            for m in messages:
                if m.get("role") == "tool":
                    try:
                        self.shadow_tool_results.append(
                            {"name": m.get("name"), "output": json.loads(str(m.get("content", "{}")))}
                        )
                    except json.JSONDecodeError:
                        pass
            if self.attempt_forbidden_in_shadow and not self._did_forbidden_attempt:
                self._did_forbidden_attempt = True
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call_bad_spawn",
                            name="dialogue_tree_create_branch",
                            arguments={"title": "Sneaky", "goal": "Mutate during shadow"},
                        ),
                        LLMToolCall(
                            id="call_bad_custom",
                            name="write_secret",
                            arguments={"value": "hacked"},
                        ),
                    ],
                    model_used=model,
                    tokens_in=50,
                    tokens_out=20,
                )
        else:
            self.active_tools.append(tools)
        return await super().complete(model, messages, tools=tools, temperature=temperature)


@pytest.mark.asyncio
async def test_shadow_kv_cache_tool_schema_preservation_and_runtime_enforcement() -> None:
    spy = _ShadowToolsSpyLLM(attempt_forbidden_in_shadow=True)
    secret_writes: list[str] = []

    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(preserve_kv_cache_tool_schemas=True),
    )
    async with Chatree.in_memory(config=cfg, llm=spy) as ct:
        @ct.tool(
            name="write_secret",
            description="Write a secret file (forbidden in shadow turns).",
            side_effect="write",
            allowed_in_shadow=False,
        )
        async def write_secret(value: str) -> dict[str, Any]:
            secret_writes.append(value)
            return {"written": value}

        # Direct ToolRegistry check: canonical schema matches active turn byte-for-byte
        active_schemas = ct.tools.schemas_for_node("read_write", is_shadow=False)
        shadow_schemas_preserved = ct.tools.schemas_for_node(
            "read_write", is_shadow=True, preserve_kv_cache_schemas=True
        )
        shadow_schemas_stripped = ct.tools.schemas_for_node(
            "read_write", is_shadow=True, preserve_kv_cache_schemas=False
        )
        assert shadow_schemas_preserved == active_schemas
        assert len(shadow_schemas_stripped) < len(active_schemas)

        tree = await ct.create_tree("KV Cache Safety Test")
        root = await tree.root()
        child = await root.spawn("Hold DB pool config", title="DB Config", permission="read_write")
        await child.chat("We picked PostgreSQL pool_size=42 with pgbouncer.")

        # Execute read-only shadow turn against child
        res = await root.request(child.id, prompt="What PostgreSQL pool_size did we pick?")
        assert res["mode"] == "shadow_turn"
        assert res["mutated"] is False

        # Active turn on child & shadow turn on child MUST receive identical `tools` schema
        assert spy.active_tools and spy.shadow_tools
        assert spy.shadow_tools[0] == spy.active_tools[-1]

        # Even though tools were visible for KV cache preservation, mutating core & custom tools
        # were deterministically blocked at runtime!
        assert secret_writes == []
        by_name = {r["name"]: r["output"] for r in spy.shadow_tool_results}
        assert by_name["dialogue_tree_create_branch"]["error"] == "READ_ONLY_SHADOW"
        assert by_name["write_secret"]["error"] == "PERMISSION_DENIED"
        # And no sneaky child was created under `child`
        nodes = await tree.nodes()
        assert not any(n.title == "Sneaky" for n in nodes)


# ---------------------------------------------------------------------------
# 2. Shadow NOT_FOUND_IN_BRANCH abstention + chronological synthesis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_not_found_abstention_and_chronological_synthesis() -> None:
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Abstention Test")
        root = await tree.root()
        c_cooking = await root.spawn("Cooking recipes", title="Cooking", mode="clean")
        await c_cooking.chat("For pasta carbonara use guanciale, pecorino romano, and egg yolks.")
        await c_cooking.finish(summary="Pasta carbonara recipe with guanciale.", auto_commit=True)

        # Shadow query on unrelated topic should abstain with NOT_FOUND_IN_BRANCH
        res1 = await root.request(c_cooking.id, prompt="What Kubernetes ingress controller version was chosen?")
        assert res1["found_in_branch"] is False
        assert str(res1["answer"]).startswith("NOT_FOUND_IN_BRANCH")

        # Repeat exact same query -> hits shadow cache AND preserves found_in_branch=False!
        res2 = await root.request(c_cooking.id, prompt="What Kubernetes ingress controller version was chosen?")
        assert res2["cache_hit"] is True
        assert res2["found_in_branch"] is False

        # ask_tree filters out abstained answers in `found_answers`
        ask_out = await tree.ask("What Kubernetes ingress controller version was chosen?", scope="all")
        assert ask_out["found_answers"] == []

    # Test ChatreeSystemAdapter on LongMemEval: both abstention (`_abs`) and chronological update
    records = [
        {
            "question_id": "lme_abs_1_abs",
            "question_type": "single-session-user",
            "question": "What color submarine did the user buy last winter?",
            "answer": "Not mentioned",
            "question_date": "2024/06/01",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024/01/10"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I planted basil and rosemary on my balcony."},
                    {"role": "assistant", "content": "Herbs grow great with full sun!"},
                ]
            ],
        },
        {
            "question_id": "lme_upd_1",
            "question_type": "knowledge-update",
            "question": "Which cloud region is our production database hosted in now?",
            "answer": "eu-central-1",
            "question_date": "2024/06/01",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2024/01/10", "2024/04/22"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "Our production database is hosted in cloud region us-east-1."},
                    {"role": "assistant", "content": "Noted: production database region is us-east-1."},
                ],
                [
                    {"role": "user", "content": "Update: we migrated our production database to cloud region eu-central-1."},
                    {"role": "assistant", "content": "Updated: production database region is now eu-central-1."},
                ],
            ],
        },
    ]
    convs = LongMemEvalDataset.from_records(records)
    runner = ExternalBenchmarkRunner(model="mock/chatree-demo", max_concurrency=2)
    report = await runner.evaluate(convs, [ChatreeSystemAdapter(model="mock/chatree-demo")])
    res = report.results[0]
    assert res.n_questions == 2
    by_qid = {it.question_id: it for it in res.item_results}
    assert by_qid["lme_abs_1_abs"].correct is True
    assert "do not know" in by_qid["lme_abs_1_abs"].prediction.lower()
    assert by_qid["lme_upd_1"].correct is True
    assert "eu-central-1" in by_qid["lme_upd_1"].prediction


# ---------------------------------------------------------------------------
# 3. LiteLLMProvider retry/backoff, malformed tool JSON, and kwargs guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_litellm_retry_backoff_and_safe_tool_json_parsing() -> None:
    # Non-destructive JSON parsing
    assert _safe_parse_tool_args('{"a": 1}') == {"a": 1}
    assert _safe_parse_tool_args("[1, 2]") == {"value": [1, 2]}
    broken = _safe_parse_tool_args("{unterminated: true")
    assert broken["_raw_arguments"] == "{unterminated: true"
    assert "_parse_error" in broken

    # Retry classification
    class RateLimitError(Exception):
        pass

    assert _is_retryable_exception(RateLimitError("too many requests")) is True
    assert _is_retryable_exception(RuntimeError("HTTP 503 temporarily unavailable")) is True
    assert _is_retryable_exception(ValueError("bad schema")) is False

    provider = LiteLLMProvider(
        api_base="https://api.deepseek.com",
        max_retries=3,
        retry_base_delay_s=0.05,
        timeout_s=45.0,
        max_tokens=1024,
    )

    # Succeeds on 3rd attempt after 2 transient 429 failures
    attempts = {"n": 0}

    async def flaky_call(**kwargs: Any) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitError("429 rate limit")
        return {"ok": True, "seen": kwargs}

    out = await provider._call_with_retry(flaky_call, ping="pong")
    assert out == {"ok": True, "seen": {"ping": "pong"}}
    assert attempts["n"] == 3

    # Non-retryable error fails immediately without retries
    bad_attempts = {"n": 0}

    async def non_retryable(**_kwargs: Any) -> None:
        bad_attempts["n"] += 1
        raise ValueError("invalid model")

    with pytest.raises(ValueError, match="invalid model"):
        await provider._call_with_retry(non_retryable)
    assert bad_attempts["n"] == 1

    # Kwargs builder checks (timeout, max_tokens, api_base, ollama stream_options guard)
    kw_ds = provider._build_kwargs(
        "deepseek/deepseek-chat", [{"role": "user", "content": "hi"}], None, 0.1, stream=True
    )
    assert kw_ds["timeout"] == 45.0
    assert kw_ds["max_tokens"] == 1024
    assert kw_ds["api_base"] == "https://api.deepseek.com"
    assert kw_ds["stream_options"] == {"include_usage": True}

    kw_ollama = provider._build_kwargs(
        "ollama/llama3", [{"role": "user", "content": "hi"}], None, 0.1, stream=True
    )
    assert "stream_options" not in kw_ollama


# ---------------------------------------------------------------------------
# 4. Local dense embedder factory & built-in rerankers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_embedder_factory_and_builtin_rerankers() -> None:
    emb_local = build_embedder("local/BAAI/bge-small-en-v1.5")
    emb_fast = build_embedder("fastembed/BAAI/bge-small-en-v1.5")
    assert isinstance(emb_local, FastEmbedEmbedder)
    assert isinstance(emb_fast, FastEmbedEmbedder)
    assert emb_local.model_name == "BAAI/bge-small-en-v1.5"
    assert isinstance(emb_local, Embedder)

    ce = FastEmbedCrossEncoderReranker()
    assert isinstance(ce, Reranker)

    class _RankLLM:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        async def complete(
            self,
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.2,
        ) -> LLMResponse:
            del model, messages, tools, temperature
            return LLMResponse(content=self.reply, model_used="stub", tokens_in=10, tokens_out=5)

        async def stream_complete(self, *args: Any, **kwargs: Any):  # pragma: no cover
            raise NotImplementedError

    hits = [
        SearchHit(
            node_id="n0",
            node_title="Alpha",
            node_status="COMPLETED",
            path="Root / Alpha",
            granularity="chunk",
            snippet="old fact",
            score=0.9,
            summary_version=1,
        ),
        SearchHit(
            node_id="n1",
            node_title="Beta",
            node_status="COMPLETED",
            path="Root / Beta",
            granularity="chunk",
            snippet="noise",
            score=0.8,
            summary_version=1,
        ),
        SearchHit(
            node_id="n2",
            node_title="Gamma",
            node_status="COMPLETED",
            path="Root / Gamma",
            granularity="chunk",
            snippet="exact answer",
            score=0.7,
            summary_version=1,
        ),
    ]
    reranker = LLMReranker(_RankLLM('{"indices": [2, 0]}'), model="stub")
    assert isinstance(reranker, Reranker)
    reordered = await reranker.rerank("find exact answer", hits, top_k=2)
    assert [h.node_id for h in reordered] == ["n2", "n0"]

    # Graceful fallback on malformed JSON
    fallback = await LLMReranker(_RankLLM("not json"), model="stub").rerank("q", hits, top_k=2)
    assert [h.node_id for h in fallback] == ["n0", "n1"]


# ---------------------------------------------------------------------------
# 5. Checkpoint/resume, bounded concurrency, dry-run, manifest, and CLI
# ---------------------------------------------------------------------------


class _CountingMockLLM(MockLLMProvider):
    def __init__(self) -> None:
        super().__init__(stream_delay_s=0.0)
        self.calls = 0

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        self.calls += 1
        return await super().complete(model, messages, tools=tools, temperature=temperature)


@pytest.mark.asyncio
async def test_external_eval_checkpoint_resume_concurrency_dry_run_and_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lme_records = [
        {
            "question_id": "q_a",
            "question_type": "single-session-user",
            "question": "What programming language did we pick for the billing worker?",
            "answer": "Rust",
            "question_date": "2024/05/01",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024/04/01"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "We picked Rust for the billing worker."},
                    {"role": "assistant", "content": "Great choice: Rust for the billing worker."},
                ]
            ],
        },
        {
            "question_id": "q_b",
            "question_type": "single-session-user",
            "question": "Which queue broker handles webhook retries?",
            "answer": "NATS JetStream",
            "question_date": "2024/05/01",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024/04/02"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "NATS JetStream handles webhook retries."},
                    {"role": "assistant", "content": "Confirmed: NATS JetStream handles webhook retries."},
                ]
            ],
        },
    ]
    convs = LongMemEvalDataset.from_records(lme_records)
    runner = ExternalBenchmarkRunner(model="mock/chatree-demo", max_concurrency=2)

    # 5a. Dry-run cost estimation makes 0 LLM calls and produces valid summary
    llm1 = _CountingMockLLM()
    adapters1 = [
        ChatreeSystemAdapter(model="mock/chatree-demo", llm=llm1),
        FullContextLinearAdapter(model="mock/chatree-demo", llm=llm1),
    ]
    est = runner.estimate_cost_dry_run(convs, adapters1)
    assert llm1.calls == 0
    assert est.n_conversations == 2
    assert est.n_questions == 2
    assert est.estimated_llm_calls > 0
    assert est.estimated_input_tokens > 0
    assert "[DRY RUN]" in est.to_summary()

    # 5b. Full run writes JSONL checkpoint
    cp_path = tmp_path / "lme.ckpt.jsonl"
    rep1 = await runner.evaluate(convs, adapters1, checkpoint_path=cp_path)
    assert cp_path.exists()
    cp_lines = [ln for ln in cp_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(cp_lines) == 4  # 2 systems x 2 questions
    assert llm1.calls > 0

    # 5c. Resume from checkpoint executes ZERO new LLM calls and preserves identical results
    llm2 = _CountingMockLLM()
    adapters2 = [
        ChatreeSystemAdapter(model="mock/chatree-demo", llm=llm2),
        FullContextLinearAdapter(model="mock/chatree-demo", llm=llm2),
    ]
    rep2 = await runner.evaluate(convs, adapters2, checkpoint_path=cp_path)
    assert llm2.calls == 0
    assert [r.accuracy for r in rep2.results] == [r.accuracy for r in rep1.results]

    # 5d. Reproducibility run_manifest.json
    manifest_path = tmp_path / "run_manifest.json"
    rep2.save_manifest(
        manifest_path,
        judge_model="mock/chatree-demo",
        embed_model="local/BAAI/bge-small-en-v1.5",
        reranker_name="llm",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "longmemeval"
    assert manifest["embed_model"] == "local/BAAI/bge-small-en-v1.5"
    assert manifest["reranker"] == "llm"
    assert manifest["systems"] == ["chatree_full", "full_context_linear"]
    assert "timestamp_utc" in manifest and "chatree_version" in manifest

    # 5e. CLI flags: --dry-run, --checkpoint, --manifest-out, --reranker llm
    lme_file = tmp_path / "lme.json"
    lme_file.write_text(json.dumps(lme_records), encoding="utf-8")
    rc_dry = await run_cli_async(["eval", "--longmemeval", str(lme_file), "--dry-run"])
    assert rc_dry == 0
    assert "[DRY RUN]" in capsys.readouterr().out

    cli_cp = tmp_path / "cli.ckpt.jsonl"
    cli_manifest = tmp_path / "cli_manifest.json"
    rc_run = await run_cli_async(
        [
            "eval",
            "--longmemeval",
            str(lme_file),
            "--checkpoint",
            str(cli_cp),
            "--manifest-out",
            str(cli_manifest),
            "--reranker",
            "llm",
            "--concurrency",
            "2",
        ]
    )
    assert rc_run == 0
    assert cli_cp.exists()
    assert cli_manifest.exists()


# ---------------------------------------------------------------------------
# 6. Flexible unscripted agentic tool matching (`expected_tools_any_of`)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unscripted_agentic_flexible_expected_tools_any_of() -> None:
    scenario = unscripted_agentic_scenario()
    assert scenario.steps[-1].expected_tools == []
    assert scenario.steps[-1].expected_tools_any_of == [
        ["dialogue_tree_search_branches", "dialogue_tree_search_and_query", "dialogue_tree_query_branch"]
    ]

    # Default MockLLMProvider calls search_tree on Turn 4 -> perfect precision & recall
    runner = BenchmarkRunner(model="mock/chatree-demo")
    rep = await runner.run_suite(scenario)
    ct_row = next(r for r in rep.results if r.mode == "tree_ipc_full")
    assert ct_row.tool_precision == 1.0
    assert ct_row.tool_recall == 1.0

    # Custom model that chooses single-hop `ask_tree` on Turn 4 also gets full credit (1.0 / 1.0)!
    class _AskTreeMockLLM(MockLLMProvider):
        def __init__(self) -> None:
            super().__init__(stream_delay_s=0.0)
            self._used_ask = False

        async def complete(
            self,
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.2,
        ) -> LLMResponse:
            last_user = next(
                (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            has_tool_res = any(m.get("role") == "tool" for m in messages[-3:])
            if (
                "search tree for token_bucket_100rps_burst_20" in last_user.lower()
                and tools
                and not has_tool_res
                and not self._used_ask
            ):
                self._used_ask = True
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call_ask_1",
                            name="dialogue_tree_search_and_query",
                            arguments={"query": "TOKEN_BUCKET_100RPS_BURST_20", "scope": "all"},
                        )
                    ],
                    model_used=model,
                    tokens_in=60,
                    tokens_out=20,
                )
            return await super().complete(model, messages, tools=tools, temperature=temperature)

    runner_ask = BenchmarkRunner(model="mock/chatree-demo", llm=_AskTreeMockLLM())
    rep_ask = await runner_ask.run_suite(scenario)
    ct_ask = next(r for r in rep_ask.results if r.mode == "tree_ipc_full")
    assert ct_ask.tool_precision == 1.0
    assert ct_ask.tool_recall == 1.0


@pytest.mark.asyncio
async def test_shadow_cache_read_tokens_and_folding_rlm_ingest_metrics() -> None:
    # 1. Shadow turns must accumulate real provider `cache_read_tokens` (not hardcode 0!)
    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Shadow Cache Accounting")
        root = await tree.root()
        child = await root.spawn("Cache Branch", mode="clean")
        await child.chat("Turn 1 context line with pool_size=88.")
        await child.chat("Turn 2 context line with retry_cap=5.")
        m_before = await tree.metrics()
        res = await root.request(child.id, prompt="What pool_size and retry_cap did we pick?")
        assert res["mode"] == "shadow_turn"
        m_after = await tree.metrics()
        assert m_after.total_cache_read_tokens > m_before.total_cache_read_tokens

    # 2. TreeSummaryOnlyAdapter & RLMOfficialAdapter conform to MemorySystemAdapter + ingest metrics
    assert isinstance(TreeSummaryOnlyAdapter(model="mock/chatree-demo"), MemorySystemAdapter)
    assert isinstance(RLMOfficialAdapter(model="deepseek/deepseek-chat"), MemorySystemAdapter)

    records = [
        {
            "question_id": "fold_q1",
            "question_type": "single-session-user",
            "question": "Which queue broker did we choose?",
            "answer": "NATS",
            "question_date": "2024/05/01",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024/04/01"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "We chose NATS as our queue broker."},
                    {"role": "assistant", "content": "Confirmed: NATS queue broker."},
                ]
            ],
        }
    ]
    convs = LongMemEvalDataset.from_records(records)
    runner = ExternalBenchmarkRunner(model="mock/chatree-demo", max_concurrency=1)
    rep = await runner.evaluate(
        convs, [TreeSummaryOnlyAdapter(model="mock/chatree-demo")]
    )
    fold_res = rep.results[0]
    assert fold_res.ingest_tokens_in > 0
    assert fold_res.ingest_tokens_out > 0
    assert fold_res.ingest_latency_ms >= 0.0

