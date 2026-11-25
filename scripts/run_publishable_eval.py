"""
Provider-Agnostic Publishable Evaluation Runner for Chatree.

Runs Tracks 1-6 against ANY LLM provider (OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter,
Groq, Mistral, Together, xAI, Ollama, vLLM, SGLang, LM Studio, or `mock/chatree-demo`) with
crash-safe JSONL checkpointing, exact provenance manifests, CSVs, and LaTeX `booktabs` tables:
  - Track 1: 5-Family Multi-Tangent & Autonomous Tool-Use Suite (6 baselines + LLM Judge)
  - Track 2: Systematic 10-Row Ablation Matrix + K-Scaling Crossover Curve (K=2, 4, 8)
  - Track 3: Official LoCoMo (ACL 2024) stratified head-to-head
  - Track 4: Official LongMemEval-S or LongMemEval-M (>1M-token overflow) head-to-head
  - Track 5: Live 18-Turn Jump-Around Chatbot Benchmark (`--natural-prompts` supported)
  - Track 6: Multi-Generation Non-Destructive vs. Destructive Compaction Benchmark

Never prints or logs any API keys.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from chatree import (
    AblationMatrixRunner,
    BenchmarkRunner,
    ChatreeSystemAdapter,
    CompactionBenchmarkRunner,
    ExternalBenchmarkRunner,
    ExternalConversation,
    FastEmbedCrossEncoderReranker,
    FlatHybridRAGAdapter,
    FullContextLinearAdapter,
    JumpAroundBenchmarkRunner,
    LoCoMoDataset,
    LongMemEvalDataset,
    Mem0OfficialAdapter,
    RLMOfficialAdapter,
    SuiteBenchmarkReport,
    TrajectoryExporter,
    TreeSummaryOnlyAdapter,
    build_embedder,
    build_full_eval_suite,
    build_llm_provider,
    default_jump_around_scenario,
    natural_jump_around_scenario,
    write_run_manifest,
)

DEFAULT_MODEL = "deepseek/deepseek-chat"
DEFAULT_EMBED_MODEL = "local/BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"


def _slugify_model(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")[:48] or "eval"


def select_stratified_locomo(
    path: Path, *, max_conversations: int = 2, per_category_per_conv: int = 6
) -> list[ExternalConversation]:
    convs = LoCoMoDataset.from_file(path, max_conversations=max_conversations)
    selected: list[ExternalConversation] = []
    for c in convs:
        counts: Counter[str] = Counter()
        picked = []
        for qa in c.qa_items:
            if counts[qa.category] < per_category_per_conv:
                picked.append(qa)
                counts[qa.category] += 1
        selected.append(c.model_copy(update={"qa_items": picked}))
    return selected


def select_stratified_longmemeval(
    path: Path, *, per_category: int = 5
) -> list[ExternalConversation]:
    convs = LongMemEvalDataset.from_file(path)
    by_cat: dict[str, list[ExternalConversation]] = {}
    for c in convs:
        cat = c.qa_items[0].category
        by_cat.setdefault(cat, []).append(c)

    picked: list[ExternalConversation] = []
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        abs_items = [x for x in items if x.qa_items[0].is_abstention]
        reg_items = [x for x in items if not x.qa_items[0].is_abstention]
        chosen = (abs_items[:1] if abs_items else []) + reg_items
        picked.extend(chosen[:per_category])
    del convs, by_cat
    gc.collect()
    return picked


async def run_track1_and_2(
    out_dir: Path,
    llm,
    embedder,
    reranker,
    *,
    model: str,
    shadow_model: str,
    summary_model: str,
    judge_model: str,
    embed_model: str,
    reranker_name: str,
) -> dict:
    summary: dict = {}
    t0 = time.perf_counter()
    print(f"\n=== TRACK 1: 5-Family Suite (model={model}, judge={judge_model}) ===", flush=True)
    bench = BenchmarkRunner(
        model=model,
        shadow_model=shadow_model,
        summary_model=summary_model,
        judge_model=judge_model,
        oracle_routing=False,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        max_concurrency=6,
    )
    scenarios = build_full_eval_suite(num_distractors=4)
    reports = []
    for idx, sc in enumerate(scenarios, 1):
        ts = time.perf_counter()
        print(f"  [Track 1] ({idx}/{len(scenarios)}) Running '{sc.name}'...", flush=True)
        reports.append(await bench.run_suite(sc))
        print(f"  [Track 1] Finished '{sc.name}' in {time.perf_counter() - ts:.1f}s", flush=True)

    suite = SuiteBenchmarkReport(reports=reports)
    suite.save_csv(out_dir / "exp1_suite.csv")
    suite.save_jsonl(out_dir / "exp1_suite.jsonl")
    TrajectoryExporter.save_jsonl(
        TrajectoryExporter.export_grpo_episodes(suite), out_dir / "exp1_grpo.jsonl"
    )
    md1 = suite.to_markdown_table()
    (out_dir / "exp1_suite.md").write_text(md1, encoding="utf-8")
    print(md1, flush=True)
    t1_cost = sum(r.estimated_cost_usd for rep in suite.reports for r in rep.results)
    write_run_manifest(
        out_dir / "exp1_manifest.json",
        benchmark="chatree_5family_suite",
        model=model,
        judge_model=judge_model,
        embed_model=embed_model,
        reranker_name=reranker_name,
        total_tokens_in=sum(r.total_tokens_in for rep in suite.reports for r in rep.results),
        total_tokens_out=sum(r.total_tokens_out for rep in suite.reports for r in rep.results),
        total_estimated_cost_usd=round(t1_cost, 6),
        systems=["linear", "sliding_window", "flat_rag", "tree_summary_only", "tree_ipc_no_staleness", "tree_ipc_full"],
    )
    summary["track1_cost_usd"] = round(t1_cost, 5)
    summary["track1_seconds"] = round(time.perf_counter() - t0, 1)

    t1 = time.perf_counter()
    print(f"\n=== TRACK 2: 10-Row Ablation Matrix + K-Scaling Crossover (model={model}) ===", flush=True)
    abl = AblationMatrixRunner(
        model=model,
        shadow_model=shadow_model,
        summary_model=summary_model,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        max_concurrency=6,
    )
    abl_rep = await abl.run_ablation_matrix()
    abl_rep.save_csv(out_dir / "exp2_ablations.csv")
    (out_dir / "exp2_ablations.tex").write_text(abl_rep.to_latex_table(), encoding="utf-8")
    (out_dir / "exp2_ablations.md").write_text(abl_rep.to_markdown_table(), encoding="utf-8")
    print(abl_rep.to_markdown_table(), flush=True)

    cross_rep = await abl.run_crossover_sweep(ks=[2, 4, 8])
    (out_dir / "exp2_crossover.tex").write_text(cross_rep.to_latex_table(), encoding="utf-8")
    (out_dir / "exp2_crossover.md").write_text(cross_rep.to_markdown_table(), encoding="utf-8")
    print("\n" + cross_rep.to_markdown_table(), flush=True)

    t2_cost = sum(r.estimated_cost_usd for r in abl_rep.rows) + sum(
        p.estimated_cost_usd for p in cross_rep.points
    )
    write_run_manifest(
        out_dir / "exp2_manifest.json",
        benchmark="ablation_matrix_and_crossover",
        model=model,
        embed_model=embed_model,
        reranker_name=reranker_name,
        total_estimated_cost_usd=round(t2_cost, 6),
        systems=[r.ablation_name for r in abl_rep.rows],
    )
    summary["track2_cost_usd"] = round(t2_cost, 5)
    summary["track2_seconds"] = round(time.perf_counter() - t1, 1)
    return summary


async def run_track3_and_4(
    out_dir: Path,
    llm,
    embedder,
    reranker,
    *,
    model: str,
    shadow_model: str,
    summary_model: str,
    judge_model: str,
    embed_model: str,
    reranker_name: str,
    api_base: str | None = None,
    tracks: set[str] | None = None,
    concurrency: int = 4,
    locomo_convs: int = 2,
    locomo_per_cat: int = 6,
    lme_per_cat: int = 5,
    lme_split: str = "s",
    include_paper_baselines: bool = False,
) -> dict:
    active = tracks or {"3", "4"}
    summary: dict = {}
    runner = ExternalBenchmarkRunner(
        model=model,
        judge_model=judge_model,
        judge_llm=llm,
        max_concurrency=concurrency,
    )
    adapters: list = [
        ChatreeSystemAdapter(
            name="chatree_two_tier" if include_paper_baselines else "chatree_full",
            model=model,
            shadow_model=shadow_model,
            summary_model=summary_model,
            llm=llm,
            embedder=embedder,
            reranker=reranker,
            max_ask_targets=4,
            generate_session_summaries=include_paper_baselines,
            use_provided_summaries=False,
            include_branch_summary_map=include_paper_baselines,
            concise_synthesis=True,
            enable_recursive_hop=include_paper_baselines,
        ),
        FlatHybridRAGAdapter(
            name="flat_hybrid_rag",
            model=model,
            llm=llm,
            embedder=embedder,
            reranker=reranker,
            top_k=5,
        ),
        FullContextLinearAdapter(
            name="full_context_linear",
            model=model,
            llm=llm,
            max_context_tokens=120_000,
        ),
    ]
    if include_paper_baselines:
        adapters.extend(
            [
                TreeSummaryOnlyAdapter(
                    name="tree_summary_only",
                    model=model,
                    llm=llm,
                    embedder=embedder,
                    reranker=reranker,
                    top_k=6,
                ),
                Mem0OfficialAdapter(
                    name="mem0_official",
                    model=model,
                    llm=llm,
                    api_base=api_base,
                    top_k=8,
                ),
                RLMOfficialAdapter(
                    name="rlm_official",
                    model=model,
                    backend="auto",
                    base_url=api_base,
                    max_depth=1,
                    max_iterations=8,
                    max_concurrent_subcalls=4,
                    max_timeout=90.0,
                ),
            ]
        )

    ref_sys = "chatree_two_tier" if include_paper_baselines else "chatree_full"

    if "3" in active:
        t0 = time.perf_counter()
        loco_path = Path("data/benchmarks/locomo10.json")
        if not loco_path.exists():
            LoCoMoDataset.download(loco_path)
        loco_convs_list = select_stratified_locomo(
            loco_path, max_conversations=locomo_convs, per_category_per_conv=locomo_per_cat
        )
        n_loco_q = sum(len(c.qa_items) for c in loco_convs_list)
        print(
            f"\n=== TRACK 3: Official LoCoMo ({len(loco_convs_list)} convs, {n_loco_q} QAs, model={model}, judge={judge_model}) ===",
            flush=True,
        )
        print(runner.estimate_cost_dry_run(loco_convs_list, adapters).to_summary(), flush=True)
        loco_rep = await runner.evaluate(
            loco_convs_list,
            adapters,
            reference_system=ref_sys,
            checkpoint_path=out_dir / "exp3_locomo.ckpt.jsonl",
        )
        loco_rep.save_csv(out_dir / "exp3_locomo.csv")
        (out_dir / "exp3_locomo.tex").write_text(loco_rep.to_latex_table(), encoding="utf-8")
        loco_rep.save_manifest(
            out_dir / "exp3_locomo_manifest.json",
            judge_model=judge_model,
            embed_model=embed_model,
            reranker_name=reranker_name,
            extra={"per_category_per_conv": locomo_per_cat, "n_questions": n_loco_q},
        )
        all_cats3 = sorted({k for r in loco_rep.results for k in r.category_accuracy})
        cat_lines3 = [
            "\n#### LoCoMo Per-Category Accuracy",
            "| System | " + " | ".join(all_cats3) + " |",
            "|---|" + "|".join("---:" for _ in all_cats3) + "|",
        ]
        for r in loco_rep.results:
            cat_lines3.append(
                f"| `{r.system_name}` | "
                + " | ".join(f"{r.category_accuracy.get(c, 0.0) * 100:.1f}%" for c in all_cats3)
                + " |"
            )
        full_md3 = loco_rep.to_markdown_table() + "\n" + "\n".join(cat_lines3)
        (out_dir / "exp3_locomo.md").write_text(full_md3, encoding="utf-8")
        print(full_md3, flush=True)
        summary["track3_cost_usd"] = round(
            sum(r.estimated_cost_usd + r.ingest_cost_usd for r in loco_rep.results), 5
        )
        summary["track3_seconds"] = round(time.perf_counter() - t0, 1)

    if "4" in active:
        t1 = time.perf_counter()
        split_clean = "m" if lme_split.lower() == "m" else "s"
        lme_path = Path(f"data/benchmarks/longmemeval_{split_clean}_cleaned.json")
        if not lme_path.exists():
            print(f"  [Track 4] Downloading LongMemEval-{split_clean.upper()} to {lme_path}...", flush=True)
            LongMemEvalDataset.download(lme_path, split=split_clean)
        lme_convs = select_stratified_longmemeval(lme_path, per_category=lme_per_cat)
        n_lme_q = sum(len(c.qa_items) for c in lme_convs)
        pfx = "exp4_longmemeval" if split_clean == "s" else "exp4_longmemeval_m"
        print(
            f"\n=== TRACK 4: Official LongMemEval-{split_clean.upper()} ({n_lme_q} items, model={model}, judge={judge_model}) ===",
            flush=True,
        )
        lme_adapters = [a for a in adapters if a.name not in ("mem0_official", "tree_summary_only")]
        print(runner.estimate_cost_dry_run(lme_convs, lme_adapters).to_summary(), flush=True)
        lme_rep = await runner.evaluate(
            lme_convs,
            lme_adapters,
            reference_system=ref_sys,
            checkpoint_path=out_dir / f"{pfx}.ckpt.jsonl",
        )
        lme_rep.save_csv(out_dir / f"{pfx}.csv")
        (out_dir / f"{pfx}.tex").write_text(lme_rep.to_latex_table(), encoding="utf-8")
        lme_rep.save_manifest(
            out_dir / f"{pfx}_manifest.json",
            judge_model=judge_model,
            embed_model=embed_model,
            reranker_name=reranker_name,
            extra={"split": split_clean, "per_category": lme_per_cat, "n_questions": n_lme_q},
        )
        all_cats4 = sorted({k for r in lme_rep.results for k in r.category_accuracy})
        cat_lines4 = [
            f"\n#### LongMemEval-{split_clean.upper()} Per-Category Accuracy",
            "| System | " + " | ".join(all_cats4) + " |",
            "|---|" + "|".join("---:" for _ in all_cats4) + "|",
        ]
        for r in lme_rep.results:
            cat_lines4.append(
                f"| `{r.system_name}` | "
                + " | ".join(f"{r.category_accuracy.get(c, 0.0) * 100:.1f}%" for c in all_cats4)
                + " |"
            )
        full_md4 = lme_rep.to_markdown_table() + "\n" + "\n".join(cat_lines4)
        (out_dir / f"{pfx}.md").write_text(full_md4, encoding="utf-8")
        print(full_md4, flush=True)
        summary[f"track4_{split_clean}_cost_usd"] = round(
            sum(r.estimated_cost_usd + r.ingest_cost_usd for r in lme_rep.results), 5
        )
        summary[f"track4_{split_clean}_seconds"] = round(time.perf_counter() - t1, 1)
    return summary


async def run_main(argv: Sequence[str] | None = None, *, default_out_dir: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Provider-agnostic Chatree publishable evaluation runner")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Any LiteLLM or OpenAI-compatible model ID")
    parser.add_argument("--shadow-model", default="inherit")
    parser.add_argument("--summary-model", default="inherit")
    parser.add_argument("--judge-model", default=None, help="LLM judge model (defaults to --model; set cross-family for audit)")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--provider",
        choices=["auto", "litellm", "openai_compat", "mock"],
        default="auto",
        help="LLM backend driver (auto uses litellm if installed, else zero-dep openai_compat)",
    )
    parser.add_argument("--api-base", default=None, help="Optional custom OpenAI-compatible base URL (Ollama, vLLM, SGLang, proxy)")
    parser.add_argument("--tracks", default="1,2,3,4", help="Comma-separated tracks to run (1,2,3,4,5,6)")
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--locomo-convs", type=int, default=2)
    parser.add_argument("--locomo-per-cat", type=int, default=6)
    parser.add_argument("--lme-per-cat", type=int, default=5)
    parser.add_argument("--lme-split", choices=["s", "m"], default="s")
    parser.add_argument("--natural-prompts", action="store_true", help="Use system-agnostic natural prompts for Track 5")
    parser.add_argument("--rejudge-locomo-ckpt", default="", help="Re-judge existing LoCoMo checkpoint with fixed rubric / --judge-model")
    parser.add_argument("--paper-baselines", action="store_true", help="Include Context-Folding, official Mem0, and official MIT RLM")
    args = parser.parse_args(argv)

    model = args.model
    judge_model = args.judge_model or model
    out_dir = Path(args.out_dir or f"results/eval_{_slugify_model(model)}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = {t.strip() for t in args.tracks.split(",") if t.strip()}

    llm = build_llm_provider(
        args.provider,
        model=model,
        api_base=args.api_base,
        max_retries=4,
        retry_base_delay_s=1.0,
        timeout_s=120.0,
        max_tokens=512,
    )

    if args.rejudge_locomo_ckpt:
        loco_convs = select_stratified_locomo(
            Path("data/benchmarks/locomo10.json"),
            max_conversations=args.locomo_convs,
            per_category_per_conv=args.locomo_per_cat,
        )
        re_runner = ExternalBenchmarkRunner(
            model=model, judge_model=judge_model, judge_llm=llm, max_concurrency=args.concurrency
        )
        re_rep = await re_runner.rejudge_checkpoint(
            loco_convs,
            Path(args.rejudge_locomo_ckpt),
            output_checkpoint_path=out_dir / "exp3_locomo_rejudged.ckpt.jsonl",
            reference_system="chatree_two_tier",
        )
        (out_dir / "exp3_locomo_rejudged.md").write_text(re_rep.to_markdown_table(), encoding="utf-8")
        (out_dir / "exp3_locomo_rejudged.tex").write_text(re_rep.to_latex_table(), encoding="utf-8")
        print(re_rep.to_markdown_table(), flush=True)
        return

    embedder = build_embedder(args.embed_model, api_base=args.api_base)
    reranker = FastEmbedCrossEncoderReranker() if not model.startswith("mock/") else None
    reranker_name = DEFAULT_RERANKER_NAME if reranker is not None else "none"

    totals: dict = {"model": model, "judge_model": judge_model, "provider": args.provider}
    if "1" in tracks or "2" in tracks:
        totals.update(
            await run_track1_and_2(
                out_dir,
                llm,
                embedder,
                reranker,
                model=model,
                shadow_model=args.shadow_model,
                summary_model=args.summary_model,
                judge_model=judge_model,
                embed_model=args.embed_model,
                reranker_name=reranker_name,
            )
        )
    if "3" in tracks or "4" in tracks:
        totals.update(
            await run_track3_and_4(
                out_dir,
                llm,
                embedder,
                reranker,
                model=model,
                shadow_model=args.shadow_model,
                summary_model=args.summary_model,
                judge_model=judge_model,
                embed_model=args.embed_model,
                reranker_name=reranker_name,
                api_base=args.api_base,
                tracks=tracks,
                concurrency=args.concurrency,
                locomo_convs=args.locomo_convs,
                locomo_per_cat=args.locomo_per_cat,
                lme_per_cat=args.lme_per_cat,
                lme_split=args.lme_split,
                include_paper_baselines=args.paper_baselines,
            )
        )
    if "5" in tracks:
        t5 = time.perf_counter()
        ja_runner = JumpAroundBenchmarkRunner(
            model=model,
            shadow_model=args.shadow_model,
            judge_model=judge_model,
            llm=llm,
            embedder=embedder,
            reranker=reranker,
        )
        sc = natural_jump_around_scenario() if args.natural_prompts else default_jump_around_scenario()
        ja_systems = ["chatree_live", "linear", "sliding_window", "flat_rag"]
        if args.paper_baselines:
            ja_systems.append("rlm_official")
        ja_rep = await ja_runner.run(scenario=sc, systems=ja_systems)  # type: ignore[arg-type]
        pfx5 = "exp5_natural_jump_around" if args.natural_prompts else "exp5_jump_around"
        ja_rep.save_csv(out_dir / f"{pfx5}.csv")
        ja_rep.save_json(out_dir / f"{pfx5}.json")
        (out_dir / f"{pfx5}.tex").write_text(ja_rep.to_latex_table(), encoding="utf-8")
        print(ja_rep.to_markdown_table(), flush=True)
        totals["track5_seconds"] = round(time.perf_counter() - t5, 1)
    if "6" in tracks:
        t6 = time.perf_counter()
        comp_runner = CompactionBenchmarkRunner(
            model=model, llm=llm, embedder=embedder, reranker=reranker
        )
        comp_rep = await comp_runner.run()
        comp_rep.save_json(out_dir / "exp6_compaction_bench.json")
        (out_dir / "exp6_compaction_bench.md").write_text(comp_rep.to_markdown_table(), encoding="utf-8")
        (out_dir / "exp6_compaction_bench.tex").write_text(comp_rep.to_latex_table(), encoding="utf-8")
        print(comp_rep.to_markdown_table(), flush=True)
        totals["track6_seconds"] = round(time.perf_counter() - t6, 1)

    (out_dir / "run_summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print("\n=== DONE ===", json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(run_main())
