"""
Publishable Evaluation & SOTA Comparison Workflow:
  - Track A: Standardized External Benchmarks (LongMemEval & LoCoMo official schemas + Token F1,
             BLEU-1, LLM-judge binary accuracy, 95% bootstrap CIs, McNemar p-values, LaTeX booktabs).
  - Track B: Systematic 10-row Ablation Matrix & K-scaling Crossover Curve.
  - Official upstream SOTA adapters (`Mem0OfficialAdapter`, `LettaOfficialAdapter`,
    `GraphitiOfficialAdapter`) plug directly into `ExternalBenchmarkRunner.evaluate(...)`.

Run offline (zero API keys):
    uv run python examples/05_publishable_eval_and_sota.py

Run with a real model (e.g. DeepSeek / Gemini / Anthropic):
    $env:DEEPSEEK_API_KEY="sk-..."
    uv run chatree eval --ablation-matrix --model deepseek/deepseek-chat --latex-out ablations.tex
    uv run chatree eval --longmemeval ./data/longmemeval_s_cleaned.json --model deepseek/deepseek-chat --latex-out lme.tex
"""

from __future__ import annotations

import asyncio

from chatree import (
    AblationMatrixRunner,
    ChatreeSystemAdapter,
    ExternalBenchmarkRunner,
    FlatHybridRAGAdapter,
    FullContextLinearAdapter,
    LongMemEvalDataset,
)


async def main() -> None:
    # 1. Systematic Ablation Matrix (10 presets with 95% bootstrap CIs + McNemar p-values)
    abl_runner = AblationMatrixRunner(model="mock/chatree-demo")
    abl_report = await abl_runner.run_ablation_matrix()
    print(abl_report.to_markdown_table())
    print()

    # 2. Scaling & Cost/Context Crossover Curve (K = 2, 4, 6)
    cross_report = await abl_runner.run_crossover_sweep(ks=[2, 4, 6])
    print(cross_report.to_markdown_table())
    print()

    # 3. Track A: LongMemEval-formatted head-to-head comparison (Chatree vs Linear vs Flat-RAG)
    #    Swap in `LongMemEvalDataset.from_file("./data/longmemeval_s_cleaned.json")` +
    #    `Mem0OfficialAdapter(...)` / `LettaOfficialAdapter(...)` for full paper runs.
    sample_lme = LongMemEvalDataset.from_records(
        [
            {
                "question_id": "lme_demo_1",
                "question_type": "knowledge-update",
                "question": "What city does Alex currently live in?",
                "question_date": "2025/05/01",
                "answer": "Denver",
                "haystack_session_ids": ["s1", "s2"],
                "haystack_dates": ["2025/01/10", "2025/04/15"],
                "haystack_sessions": [
                    [{"role": "user", "content": "I live in Austin and work on distributed databases."}],
                    [{"role": "user", "content": "Revision: I moved from Austin to Denver last month."}],
                ],
            },
            {
                "question_id": "lme_demo_2_abs",
                "question_type": "single-session-user",
                "question": "What breed is Alex's pet parrot?",
                "question_date": "2025/05/01",
                "answer": "unanswerable",
                "haystack_session_ids": ["s1"],
                "haystack_dates": ["2025/01/10"],
                "haystack_sessions": [
                    [{"role": "user", "content": "I enjoy hiking and cycling on weekends."}],
                ],
            },
        ]
    )

    ext_runner = ExternalBenchmarkRunner(model="mock/chatree-demo")
    ext_report = await ext_runner.evaluate(
        sample_lme,
        adapters=[
            ChatreeSystemAdapter(model="mock/chatree-demo"),
            FullContextLinearAdapter(model="mock/chatree-demo"),
            FlatHybridRAGAdapter(model="mock/chatree-demo"),
        ],
        reference_system="chatree_full",
    )
    print(ext_report.to_markdown_table())
    print("\n--- Camera-Ready LaTeX Table ---")
    print(abl_report.to_latex_table())


if __name__ == "__main__":
    asyncio.run(main())
