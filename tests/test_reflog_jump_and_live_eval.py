from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from chatree import (
    Chatree,
    ChatreeConfig,
    GuardrailPolicy,
    JumpAroundBenchmarkRunner,
    MockLLMProvider,
    default_jump_around_scenario,
)
from chatree.cli import run_cli_async
from chatree.types import LLMResponse


@pytest.mark.asyncio
async def test_tree_reflog_branch_summaries_separation_and_immutable_prefix() -> None:
    """
    Verify `<dialogue_tree_context>` cleanly separates structural TREE (parent/siblings/children),
    temporal REFLOG (`HEAD@{k}` visits), and Tier-2 BRANCH SUMMARIES while keeping
    `build_static_prefix()` 100% byte-identical across reflog jumps and sibling edits.
    """
    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Platform Architecture")
        root = await tree.root()
        await root.chat("Root SLO is P99_LATENCY_120MS.")

        auth = await root.spawn("Design JWT signing", title="Auth Design", mode="clean")
        await auth.chat("We picked ED25519_SIGNING and JWT_TTL_15M.")
        await auth.finish(summary="Auth settled on ED25519_SIGNING with JWT_TTL_15M.", auto_commit=True)

        db = await root.spawn("Design Postgres cluster", title="Database Design", mode="clean")
        await db.chat("We picked POSTGRES_16_PGBOUNCER with POOL_SIZE_64.")
        db_model_before = await db.model()
        static_before = await ct.context.build_static_prefix(db_model_before)

        # Jump away and back to populate temporal REFLOG + create another sibling
        cache = await root.spawn("Design Valkey caching", title="Caching Layer", mode="clean")
        await cache.chat("Valkey cluster with 300s TTL.")
        await tree.jump(db.id)

        db_model_after = await db.model()
        static_after = await ct.context.build_static_prefix(db_model_after)
        assert static_after == static_before  # KV-cache prefix stayed 100% byte-identical!

        tm = await tree.model()
        envelope, _, _ = await ct.context.build_ephemeral_envelope(tm, db_model_after)
        assert "TREE [" in envelope
        assert f'parent={root.id} "Platform Architecture"' in envelope
        assert "siblings:" in envelope
        assert '"Auth Design"' in envelope
        assert '"Caching Layer"' in envelope
        assert "REFLOG [HEAD@{k}=" in envelope
        assert "@{0}=" in envelope and "@{1}=" in envelope
        assert "BRANCH SUMMARIES [" in envelope
        assert "ED25519_SIGNING" in envelope


@pytest.mark.asyncio
async def test_switch_branch_destination_preview_and_one_turn_followup_staleness() -> None:
    """
    Verify `dialogue_tree_switch_branch` returns destination goal/status/summary/recent turns AND
    supports 1-turn `follow_up_prompt` (reopening a completed branch, mutating destination, and
    propagating staleness to downstream consumers).
    """
    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Gateway Spec")
        root = await tree.root()

        auth = await root.spawn("Auth policy", title="Auth Design", mode="clean")
        await auth.chat("Initial auth policy: JWT_TTL_15M with ED25519_SIGNING.")
        await auth.finish(summary="Auth uses JWT_TTL_15M and ED25519_SIGNING.", auto_commit=True)

        # Root reads auth (establishing dependency edge root -> auth)
        await root.request(auth.id, prompt="What is our JWT TTL?")
        assert (await root.stale_dependencies()) == []

        # Jump from root -> auth WITH follow_up_prompt in one tool call
        tm = await tree.model()
        rm = await root.model()
        res = await ct.ipc.reflog_jump(
            tree=tm,
            caller=rm,
            target=auth.id,
            note="Security revision",
            follow_up_prompt="Revision: change JWT TTL to JWT_TTL_5M_ROTATING.",
        )
        out = res.tool_output
        assert out["ok"] is True
        assert out["to_node_id"] == auth.id
        assert out["dest_goal"] == "Auth policy"
        assert "JWT_TTL_15M" in out["dest_summary"]
        assert out["follow_up_executed"] is True
        assert out["follow_up_reopened"] is True
        assert out["follow_up_response"]
        assert len(out["dest_recent_turns"]) == 2
        assert root.id in out["stale_node_ids"]
        assert root.id in res.stale_node_ids

        # Destination was reopened to ACTIVE, holds the new turn, and root sees staleness
        auth_after = await auth.model()
        assert auth_after.status == "ACTIVE"
        assert (await tree.head()).id == auth.id
        root_stale = await root.stale_dependencies()
        assert any(dep.source_node_id == auth.id for dep in root_stale)


class _SummaryToolChoiceSpyLLM(MockLLMProvider):
    """Verifies `ensure_fresh_summary` preserves canonical tools + `tool_choice='none'`."""

    def __init__(self) -> None:
        super().__init__(stream_delay_s=0.0)
        self.summary_calls: list[dict[str, Any]] = []

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        *,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        if any("[Chatree Summarizer]" in str(m.get("content", "")) for m in messages):
            self.summary_calls.append({"tools": tools, "tool_choice": tool_choice})
        return await super().complete(
            model, messages, tools=tools, temperature=temperature, tool_choice=tool_choice
        )


@pytest.mark.asyncio
async def test_background_summary_on_switch_and_kv_cache_preservation() -> None:
    """
    Verify:
      1. `auto_background_summarize_on_switch=True` summarizes a dirty open branch asynchronously
         when HEAD jumps away, keeping status='ACTIVE' (not closing it) and indexing Tier-2 summary.
      2. Summarization preserves canonical tool schemas with `tool_choice='none'` for byte-0 KV-cache reuse.
      3. `refresh_summary_background` and `Chatree.close()` handle cancellation cleanly.
    """
    spy = _SummaryToolChoiceSpyLLM()
    cfg = ChatreeConfig(
        db_path=":memory:",
        auto_background_summarize_on_switch=True,
        guardrails=GuardrailPolicy(preserve_kv_cache_tool_schemas=True),
    )
    async with Chatree.in_memory(config=cfg, llm=spy) as ct:
        tree = await ct.create_tree("Background Summary Test")
        root = await tree.root()
        auth = await root.spawn("Auth policy", title="Auth Design", mode="clean")
        await auth.chat("We settled on ED25519_SIGNING and ARGON2ID_64MB.")
        assert (await auth.model()).summary is None
        assert (await auth.model()).is_dirty is True

        # Switch away to root WITHOUT closing auth -> triggers non-blocking bg summary
        await tree.jump(root.id)
        await ct.wait_background_summaries(timeout=10.0)

        auth_after = await auth.model()
        assert auth_after.status == "ACTIVE"  # Still open, NOT forcibly closed!
        assert auth_after.is_dirty is False
        assert auth_after.summary is not None and "ED25519_SIGNING" in auth_after.summary
        assert spy.summary_calls, "Expected summarizer call to run"
        assert spy.summary_calls[-1]["tools"] is not None
        assert spy.summary_calls[-1]["tool_choice"] == "none"

        # Explicit refresh_summary_background works and clean close cancels pending tasks
        await auth.chat("Follow-up note: enable REFRESH_REUSE_DETECTION.")
        task = tree.refresh_summary_background(auth.id)
        assert isinstance(task, asyncio.Task)
        summary_out = await task
        assert "REFRESH_REUSE_DETECTION" in summary_out


@pytest.mark.asyncio
async def test_jump_around_benchmark_runner_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sc = default_jump_around_scenario()
    assert len(sc.turns) == 18

    runner = JumpAroundBenchmarkRunner(model="mock/chatree-demo")
    rep = await runner.run(sc)
    assert [r.system for r in rep.results] == [
        "chatree_live",
        "linear",
        "sliding_window",
        "flat_rag",
    ]
    ct_res = rep.results[0]
    assert ct_res.branches_created >= 3
    assert ct_res.reflog_jumps >= 2
    assert ct_res.bg_summaries_completed >= 1
    assert ct_res.navigation_accuracy is not None and ct_res.navigation_accuracy >= 0.5
    assert len(ct_res.per_turn_context_tokens) == 18
    assert "Track 5" in rep.to_markdown_table()
    assert "\\begin{table}" in rep.to_latex_table()

    csv_p = tmp_path / "ja.csv"
    tex_p = tmp_path / "ja.tex"
    rc = await run_cli_async(
        ["eval", "--jump-around", "--csv-out", str(csv_p), "--latex-out", str(tex_p)]
    )
    assert rc == 0
    assert csv_p.exists() and "chatree_live" in csv_p.read_text(encoding="utf-8")
    assert tex_p.exists() and "tab:jump_around_live" in tex_p.read_text(encoding="utf-8")
    assert "Track 5" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_six_engine_and_caching_optimizations() -> None:
    """
    Verify all 6 performance & reliability optimizations:
      1. Incremental tail chunk indexing (unchanged head chunks are NOT re-embedded)
      2. Cache-discounted BudgetLedger + graceful final-synthesis reserve (never returns '')
      3. Vectorized NumPy cosine scoring parity
      4. Fast RRF-only speculative pointers (skips cross-encoder on normal chat turns)
      5. Anthropic 4th fork-boundary breakpoint + real `last_llm_at` warm-cache tracking
      6. Semantic near-duplicate shadow_cache hit with key-identifier false-positive guard
    """
    from chatree import HashEmbedder, LiteLLMProvider, RagConfig
    from chatree.engine.guardrails import BudgetLedger
    from chatree.types import LLMToolCall, SearchHit

    # 1 & 4. Spy embedder & reranker to verify incremental chunk indexing + RRF-only speculative pointers
    class _CountingEmbedder(HashEmbedder):
        def __init__(self) -> None:
            super().__init__(dimensions=64)
            self.embed_batches: list[int] = []

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.embed_batches.append(len(texts))
            return await super().embed(texts)

    class _SpyReranker:
        def __init__(self) -> None:
            self.calls = 0

        async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
            del query
            self.calls += 1
            return hits[:top_k]

    emb = _CountingEmbedder()
    reranker = _SpyReranker()
    cfg = ChatreeConfig(
        db_path=":memory:",
        rag=RagConfig(chunk_target_tokens=60, chunk_overlap_tokens=10),
    )
    async with Chatree.in_memory(
        config=cfg, llm=MockLLMProvider(stream_delay_s=0.0), embedder=emb, reranker=reranker
    ) as ct:
        tree = await ct.create_tree("Opt Test")
        root = await tree.root()
        # Append 4 big messages -> produces multiple chunks
        for i in range(4):
            await ct.repo.append_message(
                root.id, tree.id, "user", f"Block {i}: " + (f"token_{i} " * 45)
            )
        emb.embed_batches.clear()
        n1 = await ct.rag.index_node_chunks(root.id)
        assert n1 >= 3
        assert emb.embed_batches == [n1]

        # Re-indexing with no new messages performs ZERO embeddings!
        emb.embed_batches.clear()
        assert await ct.rag.index_node_chunks(root.id) == n1
        assert emb.embed_batches == []

        # Appending 1 new message embeds ONLY the changed/new tail chunk(s), not all n1+1!
        await ct.repo.append_message(root.id, tree.id, "user", "Block 4: " + ("token_4 " * 45))
        emb.embed_batches.clear()
        n2 = await ct.rag.index_node_chunks(root.id)
        assert n2 > n1
        assert sum(emb.embed_batches) < n2  # Incremental tail only!

        # 4. Pre-turn speculative pointers skip cross-encoder reranker; explicit search uses it
        child = await root.spawn("Sibling branch", title="Sibling", mode="clean")
        await child.chat("Valkey caching policy VALKEY_CLUSTER_7.")
        reranker.calls = 0
        await root.chat("Just confirming we will revisit Valkey later.")
        assert reranker.calls == 0  # Speculative pointers stayed fast RRF-only!
        await tree.search("VALKEY_CLUSTER_7", top_k=3)
        assert reranker.calls == 1

        # 5. Real `last_llm_at` warm timestamp recorded on LLM turn
        age = ct.ipc.get_node_cache_age_seconds(await child.model())
        assert age is not None and age < 5.0

    # 2. Cache-discounted BudgetLedger + graceful final-synthesis reserve
    ledger = BudgetLedger(token_budget=10_000)
    r = ledger.try_reserve(1_000)
    assert r == 1_000
    # 5000 tokens in with 4500 cached at 0.1 weight -> 500 + 450 + 100 = 1050 effective tokens!
    ledger.settle(r, tokens_in=5_000, tokens_out=100, cache_read_tokens=4_500, cache_read_weight=0.1)
    assert ledger.tokens_used == 1_050

    # Even if trace budget is completely exhausted right after a tool round, orchestrator runs
    # one tool-free grace synthesis turn instead of returning empty ""!
    class _BudgetExhaustingToolLLM(MockLLMProvider):
        def __init__(self) -> None:
            super().__init__(stream_delay_s=0.0)
            self.round = 0

        async def complete(
            self,
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.2,
            **kwargs: Any,
        ) -> LLMResponse:
            self.round += 1
            if self.round == 1 and tools:
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id="call_s1",
                            name="dialogue_tree_search_branches",
                            arguments={"query": "gateway"},
                        )
                    ],
                    model_used=model,
                    tokens_in=600,
                    tokens_out=50,
                    cache_read_tokens=0,
                )
            return LLMResponse(
                content="Grace synthesis answer from tool evidence.",
                model_used=model,
                tokens_in=200,
                tokens_out=20,
            )

    cfg_tight = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(token_budget=550, min_turn_reserve=128),
    )
    async with Chatree.in_memory(config=cfg_tight, llm=_BudgetExhaustingToolLLM()) as ct2:
        t2 = await ct2.create_tree("Grace Synthesis")
        res_g = await t2.chat("Search gateway and synthesize.", token_budget=550)
        assert "Grace synthesis answer" in res_g.content

    # 5b. Anthropic 4th fork-boundary breakpoint
    fork_msgs = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Parent turn 1"},
        {"role": "assistant", "content": "Parent reply 1"},  # index 2 = fork boundary
        {"role": "user", "content": "Child turn 1"},
        {"role": "assistant", "content": "Child reply 1"},   # index 4 = tail breakpoint
        {"role": "user", "content": "Child turn 2"},
    ]
    wire = LiteLLMProvider.apply_cache_control(
        "anthropic/claude-sonnet-4", fork_msgs, fork_breakpoint_index=2
    )
    assert isinstance(wire[0]["content"], list)  # BP 1: system
    assert isinstance(wire[2]["content"], list)  # BP 3: fork boundary!
    assert isinstance(wire[4]["content"], list)  # BP 2: tail committed turn!

    # 6. Semantic near-duplicate shadow_cache hit + identifier false-positive guard
    class _NearDupEmbedder:
        @property
        def dimensions(self) -> int:
            return 4

        async def embed(self, texts: list[str]) -> list[list[float]]:
            out: list[list[float]] = []
            for t in texts:
                low = t.lower()
                if "jwt" in low or "token" in low:
                    out.append([0.99, 0.14, 0.0, 0.0])
                else:
                    out.append([0.0, 0.0, 1.0, 0.0])
            return out

    cfg_sem = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(semantic_shadow_cache_threshold=0.95),
    )
    async with Chatree.in_memory(
        config=cfg_sem, llm=MockLLMProvider(stream_delay_s=0.0), embedder=_NearDupEmbedder()
    ) as ct3:
        t3 = await ct3.create_tree("Semantic Shadow Cache")
        r3 = await t3.root()
        a3 = await r3.spawn("Auth", title="Auth", mode="clean")
        await a3.chat("We picked ED25519_SIGNING and JWT_TTL_5M_ROTATING.")

        first = await r3.request(a3.id, prompt="What JWT TTL did we select?")
        assert first["cache_hit"] is False
        # Rephrased query with same semantics hits semantic shadow cache in 0 LLM tokens!
        second = await r3.request(a3.id, prompt="Which JWT lifetime was chosen?")
        assert second["cache_hit"] is True
        assert second["semantic_cache_hit"] is True
        # Query introducing a DIFFERENT concrete identifier must NOT falsely hit!
        third = await r3.request(a3.id, prompt="What JWT POOL_SIZE_64 value was chosen?")
        assert third["cache_hit"] is False

    # 7. GPU ONNX provider auto-detection & graceful CPU fallback
    from chatree.providers.embeddings import FastEmbedEmbedder, resolve_onnx_providers
    from chatree.providers.rerankers import FastEmbedCrossEncoderReranker

    cpu_provs, cpu_gpu = resolve_onnx_providers("cpu")
    assert cpu_provs == ["CPUExecutionProvider"] and cpu_gpu is False
    auto_provs, _ = resolve_onnx_providers("auto")
    assert "CPUExecutionProvider" in auto_provs
    emb_cpu = FastEmbedEmbedder(device="cpu")
    assert emb_cpu.batch_size == 16 and emb_cpu.uses_gpu is False
    rr_cpu = FastEmbedCrossEncoderReranker(device="cpu")
    assert rr_cpu.batch_size == 16 and rr_cpu.uses_gpu is False


