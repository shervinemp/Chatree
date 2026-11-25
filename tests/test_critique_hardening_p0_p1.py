from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chatree import (
    TOOL_ARCHIVE_BRANCH,
    Chatree,
    ChatreeConfig,
    CompactionBenchmarkRunner,
    ExactFactScorer,
    ExternalBenchmarkRunner,
    FastEmbedCrossEncoderReranker,
    FastEmbedEmbedder,
    GuardrailPolicy,
    LoCoMoDataset,
    MockLLMProvider,
    RagConfig,
    SearchHit,
    default_compaction_scenario,
    is_token_asserted_as_active,
    natural_jump_around_scenario,
)
from chatree.eval.external_benchmarks import ExternalQAResult, score_external_qa
from chatree.types import LLMResponse


@pytest.mark.asyncio
async def test_archive_branch_tool_quarantines_dead_end_and_protects_root() -> None:
    """
    Verify `dialogue_tree_archive_branch`:
      - Protects root branch (`ROOT_PROTECTED`)
      - Allows a speculative `read_only` branch to self-archive, switching HEAD to parent
      - Blocks `read_only` branches from archiving *other* branches (`PERMISSION_DENIED`)
      - Blocks archiving during read-only shadow turns (`READ_ONLY_SHADOW`)
      - Quarantines archived chunks, summaries, rollups, and atomic facts from active search
    """
    archived_events: list[dict[str, Any]] = []
    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        @ct.on("branch_archived")
        def _on_arch(payload: dict[str, Any]) -> None:
            archived_events.append(payload)

        tree = await ct.create_tree("Archive Quarantine")
        root = await tree.root()
        tm = await tree.model()
        rm = await root.model()

        # 1. Root cannot be archived
        r_root = await ct.ipc.archive_branch(tm, rm, target_id=root.id, reason="bad")
        assert r_root.tool_output["error"] == "ROOT_PROTECTED"

        # 2. Shadow turn cannot archive anything
        proto = await root.spawn("GraphQL idea", title="GraphQL Prototype", mode="clean")
        await proto.chat("Testing GRAPHQL_FEDERATION_V0 and APOLLO_ROUTER_EXPERIMENTAL.")
        await tree.add_facts(
            proto.id,
            [{"fact": "Prototype tested GRAPHQL_FEDERATION_V0.", "entities": ["GraphQL"]}],
        )
        pm = await proto.model()
        r_shadow = await ct.ipc.archive_branch(tm, pm, is_shadow=True)
        assert r_shadow.tool_output["error"] == "READ_ONLY_SHADOW"

        # 3. read_only child cannot archive a *different* sibling branch
        sib = await root.spawn("Auth Design", title="Auth Design", mode="clean", switch_head=False)
        r_deny = await ct.ipc.archive_branch(tm, pm, target_id=sib.id)
        assert r_deny.tool_output["error"] == "PERMISSION_DENIED"

        # 4. read_only dead-end branch CAN self-archive via IPC tool, moving HEAD back to root!
        r_ok = await ct.ipc.execute_tool(
            tree=await tree.model(),
            caller=pm,
            tool_name=TOOL_ARCHIVE_BRANCH,
            arguments={"reason": "Exceeds latency budget"},
            trace_ctx=ct.ipc.repo and __import__("chatree.engine.guardrails", fromlist=["TraceContext"]).TraceContext.start(pm.id, 3, 10_000),
        )
        assert r_ok.tool_output["ok"] is True
        assert r_ok.tool_output["status"] == "ARCHIVED"
        assert r_ok.tool_output["switched_head_to"] == root.id
        assert (await tree.head()).id == root.id
        assert len(archived_events) == 1 and archived_events[0]["archived_node_id"] == proto.id

        # 5. Quarantined from default RAG search, active facts, and BRANCH SUMMARIES!
        assert all(h.node_id != proto.id for h in await tree.search("GRAPHQL_FEDERATION_V0", top_k=10))
        assert all(f.node_id != proto.id for f in await tree.facts())
        env, _, _ = await ct.context.build_ephemeral_envelope(await tree.model(), await root.model())
        assert "GRAPHQL_FEDERATION_V0" not in env


@pytest.mark.asyncio
async def test_onnx_gpu_runtime_smoke_test_falls_back_cleanly_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Simulate the exact Windows failure mode (`CUDAExecutionProvider` available but missing
    `cudnn64_9.dll` during kernel execution) and verify BOTH `FastEmbedEmbedder` and
    `FastEmbedCrossEncoderReranker` automatically step down to `CPUExecutionProvider`.
    """
    import chatree.providers.embeddings as emb_mod
    import chatree.providers.rerankers as rer_mod

    fake_candidates = [
        (["CUDAExecutionProvider", "CPUExecutionProvider"], True),
        (["CPUExecutionProvider"], False),
    ]
    monkeypatch.setattr(emb_mod, "resolve_onnx_provider_candidates", lambda _dev="auto": list(fake_candidates))
    monkeypatch.setattr(rer_mod, "resolve_onnx_provider_candidates", lambda _dev="auto": list(fake_candidates))

    class _FakeVec:
        def __init__(self, vals: list[float]) -> None:
            self._vals = vals

        def tolist(self) -> list[float]:
            return list(self._vals)

    class _FakeEmbedModel:
        def __init__(self, providers: list[str]) -> None:
            self.providers = providers

        def embed(self, texts: list[str], batch_size: int = 16) -> list[_FakeVec]:
            if "CUDAExecutionProvider" in self.providers:
                raise RuntimeError("LoadLibrary failed for cudnn64_9.dll with error 2")
            return [_FakeVec([1.0, 0.0, 0.0, 0.0]) for _ in texts]

    class _FakeCrossEncoder:
        def __init__(self, providers: list[str]) -> None:
            self.providers = providers

        def rerank(self, query: str, docs: list[str], batch_size: int = 16) -> list[float]:
            if "CUDAExecutionProvider" in self.providers:
                raise RuntimeError("LoadLibrary failed for cudnn64_9.dll with error 2")
            return [0.9 - 0.1 * i for i in range(len(docs))]

    emb = FastEmbedEmbedder(device="auto")
    monkeypatch.setattr(emb, "_build_model_with_providers", lambda provs: _FakeEmbedModel(provs))
    vecs = await emb.embed(["hello world"])
    assert vecs == [[1.0, 0.0, 0.0, 0.0]]
    assert emb.providers == ["CPUExecutionProvider"]
    assert emb.uses_gpu is False
    assert emb.batch_size <= 16

    rer = FastEmbedCrossEncoderReranker(device="auto")
    monkeypatch.setattr(rer, "_build_encoder_with_providers", lambda provs: _FakeCrossEncoder(provs))
    hits = [
        SearchHit(node_id="n1", node_title="A", node_status="ACTIVE", path="/r/n1", summary_version=1, snippet="alpha", score=0.1),
        SearchHit(node_id="n2", node_title="B", node_status="ACTIVE", path="/r/n2", summary_version=1, snippet="beta", score=0.2),
    ]
    ranked = await rer.rerank("alpha", hits, top_k=2)
    assert len(ranked) == 2
    assert rer.providers == ["CPUExecutionProvider"]
    assert rer.uses_gpu is False
    assert rer.batch_size <= 16


@pytest.mark.asyncio
async def test_pinned_invariant_staleness_and_live_ephemeral_facts_injection() -> None:
    """
    Verify:
      1. Pinning an invariant from a source branch snapshots its version/seq; materially revising
         or dirtily editing the source branch flags `[STALE: ...]` in `<dialogue_tree_context>`
         until re-pinned — while keeping `build_static_prefix()` byte-identical.
      2. Live `TurnOrchestrator` injects relevant Tier-2.5 atomic facts into ephemeral `FACTS [...]`
         without mutating the static prefix cache.
    """
    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Pinned Staleness & Facts")
        root = await tree.root()
        auth = await root.spawn("Auth policy", title="Auth Design", mode="clean", permission="read_write")
        await auth.chat("We picked JWT_TTL_15M on 2026-05-01.")
        await tree.add_facts(
            auth.id,
            [{"fact": "Auth Design picked JWT_TTL_15M on 2026-05-01.", "entities": ["Auth", "JWT_TTL_15M"], "resolved_date": "2026-05-01"}],
        )

        # Pin invariant authored by `auth`
        await tree.pin("jwt_ttl", "JWT_TTL_15M", source_node_id=auth.id)
        db_branch = await root.spawn("DB Design", title="DB Design", mode="clean")
        await db_branch.chat("Designing session schema.")
        static_before = await ct.context.build_static_prefix(await db_branch.model())

        # Initially fresh — no STALE tag on jwt_ttl
        env_fresh, _, _ = await ct.context.build_ephemeral_envelope(await tree.model(), await db_branch.model())
        assert "jwt_ttl=JWT_TTL_15M" in env_fresh
        assert "[STALE:" not in env_fresh

        # Now revise `auth` without updating the pinned invariant -> STALE tag MUST appear!
        await auth.chat("Security review revised TTL to JWT_TTL_5M_ROTATING!")
        env_stale, _, notices = await ct.context.build_ephemeral_envelope(await tree.model(), await db_branch.model())
        assert "jwt_ttl=JWT_TTL_15M [STALE:" in env_stale
        assert any("PINNED jwt_ttl" in n for n in notices)

        # Re-pinning updates snapshot and clears STALE
        await tree.pin("jwt_ttl", "JWT_TTL_5M_ROTATING", source_node_id=auth.id)
        env_clean, _, _ = await ct.context.build_ephemeral_envelope(await tree.model(), await db_branch.model())
        assert "jwt_ttl=JWT_TTL_5M_ROTATING" in env_clean
        assert "[STALE:" not in env_clean

        # Ephemeral FACTS injection surfaces sibling atomic dated facts while preserving static KV prefix!
        fact_hits = await tree.search("JWT_TTL_15M 2026-05-01", granularities={"fact"}, top_k=4)
        env_with_facts, _, _ = await ct.context.build_ephemeral_envelope(
            await tree.model(), await db_branch.model(), fact_hits=fact_hits
        )
        assert "FACTS [" in env_with_facts
        assert "2026-05-01" in env_with_facts
        assert await ct.context.build_static_prefix(await db_branch.model()) == static_before


@pytest.mark.asyncio
async def test_compaction_savings_guard_and_rollup_depth_cap() -> None:
    """
    Verify:
      1. `compaction_enforce_savings=True` steps down from `slice` -> `clean` if keeping recent
         turns would leave `tokens_after >= tokens_before` (preventing compaction expansion).
      2. `RagConfig.max_rollup_depth` caps `propagate_rollups_upward()` so deep compaction chains
         stay O(1) instead of O(D).
    """
    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(compaction_enforce_savings=True),
        rag=RagConfig(max_rollup_depth=2),
    )
    async with Chatree.in_memory(config=cfg, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Compaction Guard")
        root = await tree.root()
        await root.chat("Short turn 1.")
        await root.chat("Short turn 2.")

        # Requesting slice with keep_recent_turns=4 on a 2-turn branch would expand context;
        # the savings guard automatically steps down to `clean` mode!
        cont, meta = await root.compact(keep_recent_turns=4, reason="test savings guard")
        assert meta["context_mode"] == "clean"
        assert meta["keep_recent_turns"] == 0
        assert (await cont.model()).context_mode == "clean"

        # Build a 4-level chain under root and verify rollup propagation stops at max_rollup_depth=2
        n1 = await root.spawn("L1", title="L1", mode="clean")
        n2 = await n1.spawn("L2", title="L2", mode="clean")
        n3 = await n2.spawn("L3", title="L3", mode="clean")
        n4 = await n3.spawn("L4", title="L4", mode="clean")
        updated_count = await ct.rag.propagate_rollups_upward(n4.id)
        assert updated_count == 2  # Only n3 and n2 updated, NOT n1 or root!


@pytest.mark.asyncio
async def test_locomo_adversarial_loader_rubric_and_rejudge_checkpoint(tmp_path: Path) -> None:
    """
    Verify:
      1. `LoCoMoDataset` stores `adversarial_answer` in `distractor_answer` (never `ground_truth`).
      2. `score_external_qa` never labels a distractor as `Ground Truth:` and accepts truthful
         false-premise/speaker-swap corrections.
      3. `ExternalBenchmarkRunner.rejudge_checkpoint` filters strictly to requested QA IDs (ignoring
         leftover extra rows) and re-judges without re-running adapters.
    """
    sample_records = [
        {
            "sample_id": "conv_adv_1",
            "conversation": {
                "speaker_a": "Caroline",
                "speaker_b": "Melanie",
                "session_1_date_time": "2023-05-25",
                "session_1": [
                    {"speaker": "Melanie", "text": "I ran a charity race and realized self-care is important."}
                ],
            },
            "qa": [
                {
                    "question": "What did Caroline realize after her charity race?",
                    "adversarial_answer": "self-care is important",
                    "category": 5,
                },
                {
                    "question": "Who ran the charity race?",
                    "answer": "Melanie",
                    "category": 4,
                },
            ],
        }
    ]
    convs = LoCoMoDataset.from_records(sample_records)
    adv_qa = convs[0].qa_items[0]
    assert adv_qa.is_abstention is True
    assert adv_qa.ground_truth == ""
    assert adv_qa.distractor_answer == "self-care is important"

    # Capture exact judge prompt sent to LLM
    captured_prompts: list[str] = []

    class _JudgeSpy(MockLLMProvider):
        async def complete(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
            prompt = str(messages[0]["content"])
            captured_prompts.append(prompt)
            if "Caroline did not run the charity race" in prompt:
                return LLMResponse(content='{"correct": true, "rationale": "Explicitly rejected false premise"}')
            return LLMResponse(content='{"correct": false, "rationale": "Accepted false premise"}')

    spy = _JudgeSpy(stream_delay_s=0.0)
    ok_corr, _, _, _ = await score_external_qa(
        question=adv_qa.question,
        ground_truth=adv_qa.ground_truth,
        distractor_answer=adv_qa.distractor_answer,
        prediction="Caroline did not run the charity race — Melanie did.",
        category="adversarial",
        is_abstention=True,
        judge_llm=spy,
        judge_model="real/judge-model",
    )
    assert ok_corr is True
    assert captured_prompts
    assert "Ground Truth:" not in captured_prompts[0]
    assert "Wrong-Premise Distractor" in captured_prompts[0]
    assert "rejects the false premise" in captured_prompts[0]

    # Offline mock abstention check also accepts false-premise rejection
    ok_offline, _, _, _ = await score_external_qa(
        question=adv_qa.question,
        ground_truth="",
        distractor_answer=adv_qa.distractor_answer,
        prediction="Caroline did not run the charity race — Melanie did.",
        category="adversarial",
        is_abstention=True,
        judge_llm=None,
        judge_model=None,
    )
    assert ok_offline is True

    # Re-judge a checkpoint containing an extra leftover row (`conv_adv_1_q99`) and verify exact filtering!
    ckpt = tmp_path / "locomo.ckpt.jsonl"
    rows = [
        {
            "system_name": "sys_a",
            "result": ExternalQAResult(
                conversation_id="conv_adv_1",
                question_id="conv_adv_1_q0",
                category="adversarial",
                question=adv_qa.question,
                ground_truth="self-care is important",
                prediction="Caroline did not run the charity race — Melanie did.",
                correct=False,
                f1=0.0,
                bleu1=0.0,
            ).model_dump(),
        },
        {
            "system_name": "sys_a",
            "result": ExternalQAResult(
                conversation_id="conv_adv_1",
                question_id="conv_adv_1_q1",
                category="single_hop",
                question="Who ran the charity race?",
                ground_truth="Melanie",
                prediction="Melanie ran the charity race.",
                correct=True,
                f1=0.5,
                bleu1=0.5,
            ).model_dump(),
        },
        {
            "system_name": "sys_a",
            "result": ExternalQAResult(
                conversation_id="conv_adv_1",
                question_id="conv_adv_1_q99_leftover",
                category="single_hop",
                question="Leftover pilot question",
                ground_truth="foo",
                prediction="foo",
                correct=True,
                f1=1.0,
                bleu1=1.0,
            ).model_dump(),
        },
    ]
    ckpt.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    runner = ExternalBenchmarkRunner(model="real/model", judge_model="real/judge-model", judge_llm=spy)
    re_rep = await runner.rejudge_checkpoint(
        convs, ckpt, categories_only={"adversarial"}, reference_system="sys_a"
    )
    assert len(re_rep.results) == 1
    assert re_rep.results[0].n_questions == 2  # Leftover q99 cleanly excluded!
    assert re_rep.results[0].category_accuracy["adversarial"] == 1.0  # Flipped False -> True!
    assert re_rep.results[0].accuracy == 1.0


def test_negation_aware_exact_fact_scorer_and_natural_jump_around() -> None:
    """Verify negation-aware scoring distinguishes retired mentions from active leakage."""
    scorer = ExactFactScorer()

    # Retired/excluded mention should NOT trigger stale_contradiction or rejected_leakage
    clean_ans = (
        "Active configuration: ED25519_SIGNING and JWT_TTL_5M_ROTATING.\n"
        "Note: JWT_TTL_15M is retired and superseded.\n"
        "### Excluded / Scrapped:\n"
        "- GRAPHQL_FEDERATION_V0"
    )
    res_ok = scorer.score(
        query="Confirm active config",
        target_branch="root",
        answer_text=clean_ans,
        expected_keywords=["ED25519_SIGNING", "JWT_TTL_5M_ROTATING"],
        rejected_keywords=["GRAPHQL_FEDERATION_V0"],
        stale_keywords=["JWT_TTL_15M"],
    )
    assert res_ok.passed is True
    assert res_ok.stale_contradiction is False
    assert res_ok.rejected_leakage is False

    # Active assertion MUST still be caught!
    bad_ans = "Active configuration: ED25519_SIGNING with JWT_TTL_15M and GRAPHQL_FEDERATION_V0."
    res_bad = scorer.score(
        query="Confirm active config",
        target_branch="root",
        answer_text=bad_ans,
        expected_keywords=["ED25519_SIGNING"],
        rejected_keywords=["GRAPHQL_FEDERATION_V0"],
        stale_keywords=["JWT_TTL_15M"],
    )
    assert res_bad.passed is False
    assert res_bad.stale_contradiction is True
    assert res_bad.rejected_leakage is True
    assert is_token_asserted_as_active("Use JWT_TTL_5M_ROTATING instead of JWT_TTL_15M.", "JWT_TTL_15M") is False

    # Natural jump-around scenario has zero tool jargon
    nat = natural_jump_around_scenario()
    joined = "\n".join(t.user_message for t in nat.turns)
    assert len(nat.turns) == 18
    assert "mode='clean'" not in joined
    assert "HEAD@{1}" not in joined


@pytest.mark.asyncio
async def test_track6_compaction_benchmark_non_destructive_beats_destructive() -> None:
    """
    Verify Track 6 `CompactionBenchmarkRunner` distinguishes non-destructive compaction
    (recovers BOTH anchored decisions AND un-anchored Turn-1 verbatim diagnostics) from
    destructive rolling compaction (keeps anchors, loses un-anchored Turn-1 details) and
    sliding window (loses both).
    """
    sc = default_compaction_scenario(num_filler_turns=6)
    sc.compact_threshold_tokens = 900
    sc.keep_recent_turns = 1

    runner = CompactionBenchmarkRunner(model="mock/chatree-demo")
    report = await runner.run(sc)
    by_sys = {r.system: r for r in report.results}

    nd = by_sys["chatree_non_destructive"]
    dr = by_sys["destructive_rolling"]
    sw = by_sys["sliding_window"]

    assert nd.compactions >= 2
    assert nd.gen1_messages_preserved >= 2
    assert nd.anchored_recall == 1.0
    assert nd.non_anchored_recall == 1.0
    assert nd.overall_recall == 1.0

    # Destructive rolling loses un-anchored Turn-1 verbatim details permanently!
    assert dr.compactions >= 1
    assert dr.gen1_messages_preserved == 0
    assert dr.non_anchored_recall == 0.0
    assert nd.non_anchored_recall > dr.non_anchored_recall

    # Sliding window drops Turn 1 completely
    assert sw.non_anchored_recall == 0.0

    md = report.to_markdown_table()
    tex = report.to_latex_table()
    assert "chatree_non_destructive" in md and "destructive_rolling" in md
    assert r"\begin{table}" in tex
