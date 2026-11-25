from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatree import (
    AblationMatrixRunner,
    ChatreeSystemAdapter,
    ExternalBenchmarkRunner,
    FlatHybridRAGAdapter,
    FullContextLinearAdapter,
    GraphitiOfficialAdapter,
    LettaOfficialAdapter,
    LoCoMoDataset,
    LongMemEvalDataset,
    Mem0OfficialAdapter,
    MemorySystemAdapter,
    bleu1,
    bootstrap_ci,
    exact_match,
    format_latex_table,
    mcnemar_test,
    paired_bootstrap_test,
    token_f1,
)
from chatree.cli import run_cli_async


def test_stats_metrics_bootstrap_mcnemar_and_latex() -> None:
    assert exact_match("The Denver, CO!", "denver co") == 1.0
    assert token_f1("Denver Colorado", "Denver") == pytest.approx(0.6667, abs=1e-3)
    assert bleu1("Denver Colorado", "Denver") == pytest.approx(0.5, abs=1e-3)
    assert bleu1("Denver", "Denver Colorado") < 1.0  # brevity penalty

    ci = bootstrap_ci([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], n_resamples=1000, seed=7)
    assert ci.ci_low <= ci.mean <= ci.ci_high
    assert ci.mean == 0.75

    # Paired bootstrap detects real improvement vs identical series
    pb_sig = paired_bootstrap_test([1.0] * 15, [0.0] * 15, n_resamples=1000, seed=7)
    assert pb_sig.delta_mean == 1.0
    assert pb_sig.significant_05 is True
    pb_eq = paired_bootstrap_test([0.5] * 10, [0.5] * 10, n_resamples=500)
    assert pb_eq.p_value == 1.0 and pb_eq.significant_05 is False

    # McNemar exact binomial (discordant < 25) and chi2 continuity-corrected (discordant >= 25)
    mc_small = mcnemar_test([True] * 10 + [False] * 2, [False] * 10 + [False] * 2)
    assert mc_small.method == "exact_binomial"
    assert mc_small.b10 == 10 and mc_small.b01 == 0
    assert mc_small.significant_05 is True

    mc_large = mcnemar_test([True] * 30 + [False] * 5, [False] * 30 + [False] * 5)
    assert mc_large.method == "chi2_continuity"
    assert mc_large.significant_05 is True

    tex = format_latex_table(
        ["System", "Acc_%", "Cost_$"],
        [["chatree_full", "92.5%", "$0.012"]],
        caption="Test_Table 100%",
        label="tab:test",
    )
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert "chatree\\_full" in tex and "92.5\\%" in tex and "\\$0.012" in tex


@pytest.mark.asyncio
async def test_locomo_and_longmemeval_loaders_sota_adapters_and_runner(tmp_path: Path) -> None:
    # Verify official SOTA adapters implement the unified protocol (zero toy reimplementations)
    assert isinstance(ChatreeSystemAdapter(), MemorySystemAdapter)
    assert isinstance(FullContextLinearAdapter(), MemorySystemAdapter)
    assert isinstance(FlatHybridRAGAdapter(), MemorySystemAdapter)
    assert isinstance(Mem0OfficialAdapter(), MemorySystemAdapter)
    assert isinstance(LettaOfficialAdapter(), MemorySystemAdapter)
    assert isinstance(GraphitiOfficialAdapter(), MemorySystemAdapter)

    # 1) LoCoMo official schema round-trip
    locomo_file = tmp_path / "locomo10.json"
    locomo_file.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv_0",
                    "conversation": {
                        "speaker_a": "Alice",
                        "speaker_b": "Bob",
                        "session_1_date_time": "2024-03-01 10:00",
                        "session_1": [
                            {"speaker": "Alice", "text": "I adopted a golden retriever named Koda yesterday."},
                            {"speaker": "Bob", "text": "Koda is a great name!"},
                        ],
                        "session_2_date_time": "2024-03-14 14:00",
                        "session_2": [
                            {"speaker": "Alice", "text": "We moved our team sprint standup to 9am CET."},
                        ],
                    },
                    "session_summary": {
                        "session_1_summary": "Alice adopted golden retriever Koda.",
                        "session_2_summary": "Alice's team standup moved to 9am CET.",
                    },
                    "qa": [
                        {"question": "What is the name of Alice's dog?", "answer": "Koda", "category": 4},
                        {"question": "What time is Alice's standup?", "answer": "9am CET", "category": 1},
                        {"question": "What is Alice's cat's name?", "adversarial_answer": "unanswerable", "category": 5},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    locomo_convs = LoCoMoDataset.from_file(locomo_file)
    assert len(locomo_convs) == 1
    assert len(locomo_convs[0].sessions) == 2
    assert [q.category for q in locomo_convs[0].qa_items] == ["single_hop", "multi_hop", "adversarial"]

    ext_runner = ExternalBenchmarkRunner(model="mock/chatree-demo")
    locomo_rep = await ext_runner.evaluate(
        locomo_convs,
        adapters=[
            ChatreeSystemAdapter(model="mock/chatree-demo"),
            FullContextLinearAdapter(model="mock/chatree-demo"),
            FlatHybridRAGAdapter(model="mock/chatree-demo"),
        ],
        reference_system="chatree_full",
    )
    assert len(locomo_rep.results) == 3
    by_sys = {r.system_name: r for r in locomo_rep.results}
    assert by_sys["chatree_full"].n_questions == 3
    assert by_sys["chatree_full"].accuracy > 0.0
    assert by_sys["full_context_linear"].mcnemar_vs_ref is not None
    assert "chatree_full" in locomo_rep.to_markdown_table()
    assert "\\toprule" in locomo_rep.to_latex_table()
    csv_path = locomo_rep.save_csv(tmp_path / "locomo.csv")
    assert "chatree_full" in csv_path.read_text(encoding="utf-8")

    # 2) LongMemEval official schema round-trip
    lme_file = tmp_path / "longmemeval_s_cleaned.json"
    lme_file.write_text(
        json.dumps(
            [
                {
                    "question_id": "q_update_1",
                    "question_type": "knowledge-update",
                    "question": "Which cloud region hosts our billing service now?",
                    "question_date": "2025/05/10",
                    "answer": "eu-central-1",
                    "haystack_session_ids": ["s1", "s2"],
                    "haystack_dates": ["2025/01/05", "2025/04/20"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "Billing service currently runs in us-east-1."}],
                        [{"role": "user", "content": "Revision: we migrated billing service to eu-central-1."}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    lme_convs = LongMemEvalDataset.from_file(lme_file)
    assert len(lme_convs) == 1
    assert lme_convs[0].qa_items[0].category == "knowledge-update"
    lme_rep = await ext_runner.evaluate(
        lme_convs,
        adapters=[
            ChatreeSystemAdapter(model="mock/chatree-demo"),
            FullContextLinearAdapter(model="mock/chatree-demo"),
        ],
        reference_system="chatree_full",
    )
    assert lme_rep.results[0].category_accuracy.get("knowledge-update") == 1.0


@pytest.mark.asyncio
async def test_systematic_ablation_matrix_crossover_sweep_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    abl_runner = AblationMatrixRunner(model="mock/chatree-demo")
    abl_rep = await abl_runner.run_ablation_matrix(
        preset_names=[
            "chatree_full",
            "chatree_oracle_routing",
            "no_shadow_turns_folding",
            "no_staleness",
            "no_transitive_staleness",
        ]
    )
    by_name = {r.ablation_name: r for r in abl_rep.rows}
    assert by_name["chatree_full"].pass_rate == 1.0
    assert by_name["no_shadow_turns_folding"].pass_rate < by_name["chatree_full"].pass_rate
    assert by_name["no_staleness"].pass_rate < by_name["chatree_full"].pass_rate
    assert by_name["no_transitive_staleness"].pass_rate < by_name["chatree_full"].pass_rate
    assert "\\begin{table}" in abl_rep.to_latex_table()

    cross_rep = await abl_runner.run_crossover_sweep(ks=[2, 4])
    pts_k4 = {p.mode: p for p in cross_rep.points if p.k_branches == 4}
    assert pts_k4["tree_ipc_full"].pass_rate == 1.0
    assert pts_k4["tree_ipc_full"].peak_root_context_tokens < pts_k4["linear"].peak_root_context_tokens

    # CLI flags: --ablation-matrix + --latex-out + --crossover-ks
    tex_out = tmp_path / "ablations.tex"
    assert await run_cli_async(["eval", "--ablation-matrix", "--latex-out", str(tex_out)]) == 0
    assert "chatree_full" in capsys.readouterr().out
    assert tex_out.exists() and "\\toprule" in tex_out.read_text(encoding="utf-8")

    cross_tex = tmp_path / "crossover.tex"
    assert await run_cli_async(["eval", "--crossover-ks", "2,4", "--latex-out", str(cross_tex)]) == 0
    assert "tree_ipc_full" in capsys.readouterr().out
    assert cross_tex.exists()
