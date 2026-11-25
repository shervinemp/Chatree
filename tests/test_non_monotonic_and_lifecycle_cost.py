from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from chatree import (
    ChatreeSystemAdapter,
    ExternalBenchmarkRunner,
    ExternalQAResult,
    FlatHybridRAGAdapter,
    HashEmbedder,
    LLMResponse,
    MockLLMProvider,
    build_non_monotonic_engineering_benchmark,
)
from chatree.eval.external_benchmarks import score_external_qa
from chatree.eval.runner import BenchmarkRunner
from chatree.eval.scenarios import proposal_vs_decision_scenario


def _load_eval_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_publishable_eval.py"
    spec = importlib.util.spec_from_file_location("run_publishable_eval", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_non_monotonic_benchmark_corpus_integrity() -> None:
    """Track 7 corpus must cover 6 engineering projects, 36 QAs, all 5 failure categories, and fair cancellations."""
    convs = build_non_monotonic_engineering_benchmark()
    assert len(convs) == 6
    assert all(c.benchmark == "non_monotonic" for c in convs)

    all_qas = [qa for c in convs for qa in c.qa_items]
    assert len(all_qas) == 36

    categories = {qa.category for qa in all_qas}
    assert categories == {
        "spec_reversal",
        "dead_end_quarantine",
        "downstream_staleness",
        "verbatim_under_noise",
        "cross_branch_synthesis",
    }

    archived_sessions = [s for c in convs for s in c.sessions if s.archived]
    assert len(archived_sessions) == 6
    for sess in archived_sessions:
        assert sess.archive_reason.strip()
        # Baseline fairness guarantee: every archived session must contain an explicit textual cancellation turn
        joined = " ".join(t.content.lower() for t in sess.turns)
        assert any(
            kw in joined
            for kw in (
                "cancelled",
                "scrap",
                "abandoned",
                "discarded",
                "drop and archive",
                "do not ship",
                "decommissioned",
                "rejected",
                "retired",
            )
        ), f"Archived session {sess.session_id} missing explicit textual cancellation turn"


@pytest.mark.asyncio
async def test_chatree_adapter_archive_session_quarantines_dead_end() -> None:
    """ChatreeSystemAdapter.archive_session must quarantine scrapped branches from 4-tier RAG & summary map."""
    llm = MockLLMProvider(stream_delay_s=0.0)
    adapter = ChatreeSystemAdapter(
        name="chatree_two_tier_v3",
        model="mock/chatree-demo",
        llm=llm,
        embedder=HashEmbedder(),
        generate_session_summaries=False,
    )
    await adapter.reset("quarantine_test")
    try:
        await adapter.start_session("s_active", title="Active Auth Spec", date_time="2026-03-01")
        await adapter.ingest_turn(
            "s_active", "assistant", "Active production auth token is ED25519_PROD_SPEC_42."
        )
        await adapter.finish_session("s_active", summary="Production auth uses ED25519_PROD_SPEC_42.")

        await adapter.start_session("s_dead", title="Scrapped GraphQL Prototype", date_time="2026-03-02")
        await adapter.ingest_turn(
            "s_dead",
            "assistant",
            "Experimental GRAPHQL_FEDERATION_SCRAPPED_99 adds 24 hops. Cancelled and scrapped.",
        )
        await adapter.finish_session("s_dead", summary="Scrapped GRAPHQL_FEDERATION_SCRAPPED_99.")
        await adapter.archive_session("s_dead", reason="Violates latency SLO")

        assert adapter._tree is not None and adapter._ct is not None
        dead_nid = adapter._session_nodes["s_dead"]
        dead_node = await adapter._ct.repo.get_node(dead_nid)
        assert dead_node is not None and dead_node.status == "ARCHIVED"
        assert dead_nid not in adapter._node_summaries

        hits = await adapter._tree.search("GRAPHQL_FEDERATION_SCRAPPED_99", top_k=5)
        assert all(h.node_id != dead_nid for h in hits)

        active_hits = await adapter._tree.search("ED25519_PROD_SPEC_42", top_k=5)
        assert any(h.node_id == adapter._session_nodes["s_active"] for h in active_hits)
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_external_benchmark_report_total_lifecycle_cost_and_track7_mock(tmp_path: Path) -> None:
    """ExternalBenchmarkReport must report Query / Ingest / Total USD, p50/p95 ms, and CSV total_cost_usd."""
    convs = build_non_monotonic_engineering_benchmark()[:1]
    # Keep 2 QAs for instant offline test execution
    tiny_conv = convs[0].model_copy(update={"qa_items": convs[0].qa_items[:2]})

    llm = MockLLMProvider(stream_delay_s=0.0)
    embedder = HashEmbedder()
    adapters = [
        ChatreeSystemAdapter(
            name="chatree_two_tier_v3",
            model="mock/chatree-demo",
            llm=llm,
            embedder=embedder,
            generate_session_summaries=True,
        ),
        FlatHybridRAGAdapter(
            name="flat_hybrid_rag",
            model="mock/chatree-demo",
            llm=llm,
            embedder=embedder,
            top_k=3,
        ),
    ]
    runner = ExternalBenchmarkRunner(model="mock/chatree-demo", judge_model="mock/judge")
    est = runner.estimate_cost_dry_run([tiny_conv], adapters)
    assert est.benchmark == "non_monotonic"
    assert est.n_questions == 2
    assert "est_total=$" in est.to_summary()

    ckpt = tmp_path / "exp7.ckpt.jsonl"
    report = await runner.evaluate(
        [tiny_conv],
        adapters,
        reference_system="chatree_two_tier_v3",
        checkpoint_path=ckpt,
    )
    assert report.benchmark == "non_monotonic"
    assert len(report.results) == 2

    md = report.to_markdown_table()
    assert "Query / Ingest / Total USD" in md
    assert "Ingest s" in md
    assert "p50 / p95 ms" in md

    tex = report.to_latex_table()
    assert "Total USD (Q+I)" in tex
    assert "p50 / p95 ms" in tex

    csv_path = report.save_csv(tmp_path / "exp7.csv")
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) == 2
    assert "total_cost_usd" in rows[0]
    for r in rows:
        expected_total = round(float(r["estimated_cost_usd"]) + float(r["ingest_cost_usd"]), 6)
        assert float(r["total_cost_usd"]) == pytest.approx(expected_total, abs=1e-6)

    manifest_path = report.save_manifest(tmp_path / "exp7_manifest.json", judge_model="mock/judge")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "total_query_cost_usd" in manifest["extra"]
    assert "total_ingest_cost_usd" in manifest["extra"]


@pytest.mark.asyncio
async def test_cross_family_judge_llm_wired_distinctly() -> None:
    """Dedicated cross-family judge_llm must receive judge calls instead of solver llm."""
    mod = _load_eval_script_module()
    solver_llm = MockLLMProvider(stream_delay_s=0.0)
    same = mod._build_judge_llm(
        solver_llm,
        model="mock/chatree-demo",
        judge_model="mock/chatree-demo",
        provider="mock",
        judge_provider=None,
        api_base=None,
        judge_api_base=None,
    )
    assert same is solver_llm

    different = mod._build_judge_llm(
        solver_llm,
        model="mock/chatree-demo",
        judge_model="mock/cross-family-judge",
        provider="mock",
        judge_provider="mock",
        api_base=None,
        judge_api_base=None,
    )
    assert different is not solver_llm

    class RecordingJudgeLLM(MockLLMProvider):
        def __init__(self) -> None:
            super().__init__(stream_delay_s=0.0)
            self.called_models: list[str] = []

        async def complete(
            self,
            *,
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.2,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            self.called_models.append(model)
            return LLMResponse(
                content='{"correct": true, "score": 1.0, "rationale": "cross-family ok"}',
                tokens_in=50,
                tokens_out=15,
            )

    rec_judge = RecordingJudgeLLM()
    ok, _, _, rat = await score_external_qa(
        question="What algorithm is active?",
        ground_truth="ED25519_SIGNING",
        prediction="ED25519_SIGNING",
        category="spec_reversal",
        is_abstention=False,
        judge_llm=rec_judge,
        judge_model="openai/gpt-4o",
    )
    assert ok is True
    assert rat == "cross-family ok"
    assert rec_judge.called_models == ["openai/gpt-4o"]

    # Also confirm BenchmarkRunner routes judge calls through explicit judge_llm
    rec_judge.called_models.clear()
    bench = BenchmarkRunner(
        model="mock/chatree-demo",
        judge_model="openai/gpt-4o",
        llm=solver_llm,
        judge_llm=rec_judge,
        embedder=HashEmbedder(),
    )
    rep = await bench.run_suite(proposal_vs_decision_scenario(), modes=["linear"])
    assert rep.results[0].avg_judge_score == pytest.approx(1.0)
    assert len(rec_judge.called_models) >= 1
    assert all(m == "openai/gpt-4o" for m in rec_judge.called_models)


@pytest.mark.asyncio
async def test_eval_runner_dry_run_and_universal_rejudge_zero_paid_calls(tmp_path: Path) -> None:
    """run_publishable_eval.py --dry-run and --rejudge-ckpt must work cleanly at $0."""
    mod = _load_eval_script_module()
    out_dir = tmp_path / "out"

    # 1. Dry-run Track 7 with DeepSeek pricing preview (0 network / API calls)
    await mod.run_main(
        [
            "--model",
            "deepseek/deepseek-chat",
            "--judge-model",
            "openai/gpt-4o-mini",
            "--provider",
            "mock",
            "--judge-provider",
            "mock",
            "--tracks",
            "7",
            "--paper-baselines",
            "--dry-run",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert not (out_dir / "exp7_non_monotonic.ckpt.jsonl").exists()

    # 2. Universal --rejudge-ckpt on a synthetic Track 7 checkpoint with mock judge
    convs = build_non_monotonic_engineering_benchmark()
    first_conv = convs[0]
    first_qa = first_conv.qa_items[0]
    ckpt_in = tmp_path / "synthetic_exp7.ckpt.jsonl"
    rec = ExternalQAResult(
        conversation_id=first_conv.conversation_id,
        question_id=first_qa.question_id,
        category=first_qa.category,
        question=first_qa.question,
        ground_truth=first_qa.ground_truth,
        prediction=first_qa.ground_truth,
        correct=False,  # Deliberately flipped to prove rejudge updates it
        f1=0.0,
        bleu1=0.0,
        tokens_in=500,
        tokens_out=40,
        ingest_tokens_in=1000,
        ingest_tokens_out=200,
        latency_ms=1200.0,
    )
    ckpt_in.write_text(
        json.dumps({"system_name": "chatree_two_tier_v3", "result": rec.model_dump()}) + "\n",
        encoding="utf-8",
    )

    await mod.run_main(
        [
            "--model",
            "mock/chatree-demo",
            "--judge-model",
            "mock/cross-family-judge",
            "--provider",
            "mock",
            "--judge-provider",
            "mock",
            "--rejudge-ckpt",
            str(ckpt_in),
            "--rejudge-dataset",
            "non_monotonic",
            "--out-dir",
            str(out_dir),
        ]
    )
    rejudged_csv = out_dir / "exp7_non_monotonic_rejudged_mock_cross_family_judge.csv"
    rejudged_md = out_dir / "exp7_non_monotonic_rejudged_mock_cross_family_judge.md"
    rejudged_manifest = out_dir / "exp7_non_monotonic_rejudged_mock_cross_family_judge_manifest.json"
    assert rejudged_csv.exists() and rejudged_md.exists() and rejudged_manifest.exists()
    rows = list(csv.DictReader(rejudged_csv.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert float(rows[0]["accuracy"]) == pytest.approx(1.0)
    assert "Query / Ingest / Total USD" in rejudged_md.read_text(encoding="utf-8")
