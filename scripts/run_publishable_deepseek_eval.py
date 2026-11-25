"""
Publishable Live-LLM Evaluation Runner for Chatree (DeepSeek + Local ONNX BGE-small + Cross-Encoder).

Executes 4 complementary publishable evaluation tracks with crash-safe JSONL checkpointing,
exact provenance manifests, CSVs, and camera-ready LaTeX `booktabs` tables:
  - Track 1: 5-Family Multi-Tangent & Autonomous Tool-Use Suite across 6 baselines (with LLM Judge)
  - Track 2: Systematic 10-Row Ablation Matrix + K-Scaling Crossover Curve (K=2, 4, 8)
  - Track 3: Official LoCoMo (ACL 2024, snap-research/locomo) stratified head-to-head
  - Track 4: Official LongMemEval-S (ICLR 2025, xiaowu0162/longmemeval-cleaned) stratified head-to-head

Never prints or logs any API keys. Requires `DEEPSEEK_API_KEY` in process environment.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import gc
import json
from pathlib import Path
import time

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
    LiteLLMProvider,
    LoCoMoDataset,
    LongMemEvalDataset,
    Mem0OfficialAdapter,
    RLMOfficialAdapter,
    TrajectoryExporter,
    TreeSummaryOnlyAdapter,
    build_embedder,
    build_full_eval_suite,
    write_run_manifest,
)


MODEL = "deepseek/deepseek-chat"
EMBED_MODEL = "local/BAAI/bge-small-en-v1.5"
RERANKER_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"


def select_stratified_locomo(
    path: Path, *, max_conversations: int = 2, per_category_per_conv: int = 6
) -> list[ExternalConversation]:
    """Select balanced QA items across all 5 official LoCoMo categories without altering text."""
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
    """Select balanced items across all 6 official LongMemEval-S categories (including `_abs`)."""
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
        # Ensure at least 1 abstention item when available in that category
        chosen = (abs_items[:1] if abs_items else []) + reg_items
        picked.extend(chosen[:per_category])
    del convs, by_cat
    gc.collect()
    return picked


async def run_track1_and_2(out_dir: Path, llm: LiteLLMProvider, embedder, reranker) -> dict:
    summary: dict = {}
    t0 = time.perf_counter()
    print("\n=== TRACK 1: 5-Family Multi-Tangent & Agentic Suite (6 Baselines + LLM Judge) ===", flush=True)
    bench = BenchmarkRunner(
        model=MODEL,
        shadow_model=MODEL,
        summary_model=MODEL,
        judge_model=MODEL,
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
        print(f"  [Track 1] ({idx}/{len(scenarios)}) Running scenario '{sc.name}'...", flush=True)
        rep = await bench.run_suite(sc)
        reports.append(rep)
        print(f"  [Track 1] Finished '{sc.name}' in {time.perf_counter() - ts:.1f}s", flush=True)
    from chatree import SuiteBenchmarkReport

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
        model=MODEL,
        judge_model=MODEL,
        embed_model=EMBED_MODEL,
        reranker_name=RERANKER_NAME,
        total_tokens_in=sum(r.total_tokens_in for rep in suite.reports for r in rep.results),
        total_tokens_out=sum(r.total_tokens_out for rep in suite.reports for r in rep.results),
        total_estimated_cost_usd=round(t1_cost, 6),
        systems=["linear", "sliding_window", "flat_rag", "tree_summary_only", "tree_ipc_no_staleness", "tree_ipc_full"],
    )
    summary["track1_cost_usd"] = round(t1_cost, 5)
    summary["track1_seconds"] = round(time.perf_counter() - t0, 1)

    t1 = time.perf_counter()
    print("\n=== TRACK 2: Systematic 10-Row Ablation Matrix + K-Scaling Crossover (K=2,4,8) ===", flush=True)
    abl = AblationMatrixRunner(
        model=MODEL,
        shadow_model=MODEL,
        summary_model=MODEL,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        max_concurrency=6,
    )
    print("  [Track 2a] Running 10-row systematic ablation matrix (concurrency=6)...", flush=True)
    abl_rep = await abl.run_ablation_matrix()
    abl_rep.save_csv(out_dir / "exp2_ablations.csv")
    (out_dir / "exp2_ablations.tex").write_text(abl_rep.to_latex_table(), encoding="utf-8")
    md2a = abl_rep.to_markdown_table()
    (out_dir / "exp2_ablations.md").write_text(md2a, encoding="utf-8")
    print(md2a, flush=True)

    print("  [Track 2b] Running K-scaling crossover sweep (K=2, 4, 8, concurrency=6)...", flush=True)
    cross_rep = await abl.run_crossover_sweep(ks=[2, 4, 8])
    (out_dir / "exp2_crossover.tex").write_text(cross_rep.to_latex_table(), encoding="utf-8")
    md2b = cross_rep.to_markdown_table()
    (out_dir / "exp2_crossover.md").write_text(md2b, encoding="utf-8")
    print("\n" + md2b, flush=True)

    t2_cost = sum(r.estimated_cost_usd for r in abl_rep.rows) + sum(
        p.estimated_cost_usd for p in cross_rep.points
    )
    write_run_manifest(
        out_dir / "exp2_manifest.json",
        benchmark="ablation_matrix_and_crossover",
        model=MODEL,
        embed_model=EMBED_MODEL,
        reranker_name=RERANKER_NAME,
        total_estimated_cost_usd=round(t2_cost, 6),
        systems=[r.ablation_name for r in abl_rep.rows],
    )
    summary["track2_cost_usd"] = round(t2_cost, 5)
    summary["track2_seconds"] = round(time.perf_counter() - t1, 1)
    return summary


async def run_track3_and_4(
    out_dir: Path,
    llm: LiteLLMProvider,
    embedder,
    reranker,
    *,
    tracks: set[str] | None = None,
    concurrency: int = 4,
    locomo_convs: int = 2,
    locomo_per_cat: int = 6,
    lme_per_cat: int = 5,
    lme_split: str = "s",
    judge_model: str = MODEL,
    include_paper_baselines: bool = False,
    rlm_max_items: int | None = None,
) -> dict:
    active = tracks or {"3", "4"}
    summary: dict = {}
    runner = ExternalBenchmarkRunner(
        model=MODEL,
        judge_model=judge_model,
        judge_llm=llm,
        max_concurrency=concurrency,
    )
    adapters: list = [
        ChatreeSystemAdapter(
            name="chatree_two_tier" if include_paper_baselines else "chatree_full",
            model=MODEL,
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
            model=MODEL,
            llm=llm,
            embedder=embedder,
            reranker=reranker,
            top_k=5,
        ),
        FullContextLinearAdapter(
            name="full_context_linear",
            model=MODEL,
            llm=llm,
            max_context_tokens=120_000,
        ),
    ]
    if include_paper_baselines:
        adapters.extend(
            [
                TreeSummaryOnlyAdapter(
                    name="tree_summary_only",
                    model=MODEL,
                    llm=llm,
                    embedder=embedder,
                    reranker=reranker,
                    top_k=6,
                ),
                Mem0OfficialAdapter(
                    name="mem0_official",
                    model=MODEL,
                    llm=llm,
                    top_k=8,
                ),
                RLMOfficialAdapter(
                    name="rlm_official",
                    model=MODEL,
                    max_depth=1,
                    max_iterations=8,
                    max_concurrent_subcalls=4,
                    max_timeout=90.0,
                ),
            ]
        )

    ref_sys = "chatree_two_tier" if include_paper_baselines else "chatree_full"

    # --- Track 3: Official LoCoMo ---
    if "3" in active:
        t0 = time.perf_counter()
        loco_convs = select_stratified_locomo(
            Path("data/benchmarks/locomo10.json"),
            max_conversations=locomo_convs,
            per_category_per_conv=locomo_per_cat,
        )
        n_loco_q = sum(len(c.qa_items) for c in loco_convs)
        print(f"\n=== TRACK 3: Official LoCoMo ({len(loco_convs)} convs, {n_loco_q} stratified QAs) ===", flush=True)
        print(runner.estimate_cost_dry_run(loco_convs, adapters).to_summary(), flush=True)
        loco_rep = await runner.evaluate(
            loco_convs,
            adapters,
            reference_system=ref_sys,
            checkpoint_path=out_dir / "exp3_locomo.ckpt.jsonl",
        )
        loco_rep.save_csv(out_dir / "exp3_locomo.csv")
        (out_dir / "exp3_locomo.tex").write_text(loco_rep.to_latex_table(), encoding="utf-8")
        loco_rep.save_manifest(
            out_dir / "exp3_locomo_manifest.json",
            judge_model=MODEL,
            embed_model=EMBED_MODEL,
            reranker_name=RERANKER_NAME,
            extra={"per_category_per_conv": locomo_per_cat, "n_questions": n_loco_q},
        )
        md3 = loco_rep.to_markdown_table()
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
        full_md3 = md3 + "\n" + "\n".join(cat_lines3)
        (out_dir / "exp3_locomo.md").write_text(full_md3, encoding="utf-8")
        print(full_md3, flush=True)
        summary["track3_cost_usd"] = round(sum(r.estimated_cost_usd + r.ingest_cost_usd for r in loco_rep.results), 5)
        summary["track3_seconds"] = round(time.perf_counter() - t0, 1)

    # --- Track 4: Official LongMemEval-S or LongMemEval-M (>1M-token overflow!) ---
    if "4" in active:
        t1 = time.perf_counter()
        split_clean = "m" if lme_split.lower() == "m" else "s"
        lme_path = Path(f"data/benchmarks/longmemeval_{split_clean}_cleaned.json")
        if not lme_path.exists():
            print(f"  [Track 4] Downloading official LongMemEval-{split_clean.upper()} to {lme_path}...", flush=True)
            LongMemEvalDataset.download(lme_path, split=split_clean)
        lme_convs = select_stratified_longmemeval(lme_path, per_category=lme_per_cat)
        n_lme_q = sum(len(c.qa_items) for c in lme_convs)
        pfx = "exp4_longmemeval" if split_clean == "s" else "exp4_longmemeval_m"
        print(
            f"\n=== TRACK 4: Official LongMemEval-{split_clean.upper()} "
            f"({n_lme_q} stratified items across 6 categories, judge={judge_model}) ===",
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
            embed_model=EMBED_MODEL,
            reranker_name=RERANKER_NAME,
            extra={"split": split_clean, "per_category": lme_per_cat, "n_questions": n_lme_q},
        )
        md4 = lme_rep.to_markdown_table()
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
        full_md4 = md4 + "\n" + "\n".join(cat_lines4)
        (out_dir / f"{pfx}.md").write_text(full_md4, encoding="utf-8")
        print(full_md4, flush=True)
        summary[f"track4_{split_clean}_cost_usd"] = round(
            sum(r.estimated_cost_usd + r.ingest_cost_usd for r in lme_rep.results), 5
        )
        summary[f"track4_{split_clean}_seconds"] = round(time.perf_counter() - t1, 1)
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", default="1,2,3,4", help="Comma-separated track numbers to run (1,2,3,4,6)")
    parser.add_argument("--out-dir", default="results/deepseek_eval")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--locomo-convs", type=int, default=2)
    parser.add_argument("--locomo-per-cat", type=int, default=6)
    parser.add_argument("--lme-per-cat", type=int, default=5)
    parser.add_argument("--lme-split", choices=["s", "m"], default="s", help="LongMemEval split: 's' (~115k tok) or 'm' (~1.2M tok overflow)")
    parser.add_argument("--judge-model", default=MODEL, help="LLM judge model (set to e.g. openai/gpt-4o or anthropic/claude-sonnet-4-5 for cross-family audit)")
    parser.add_argument(
        "--rejudge-locomo-ckpt",
        default="",
        help="Optional path to an existing LoCoMo checkpoint to re-judge with the fixed adversarial rubric / --judge-model without re-running adapters.",
    )
    parser.add_argument(
        "--paper-baselines",
        action="store_true",
        help="Include upgraded chatree_v2, tree_summary_only (Context-Folding), official Mem0, and official MIT RLM",
    )
    args = parser.parse_args()

    tracks = {t.strip() for t in args.tracks.split(",") if t.strip()}
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LiteLLMProvider(max_retries=4, retry_base_delay_s=1.0, timeout_s=120.0, max_tokens=512)
    if args.rejudge_locomo_ckpt:
        loco_convs = select_stratified_locomo(
            Path("data/benchmarks/locomo10.json"),
            max_conversations=args.locomo_convs,
            per_category_per_conv=args.locomo_per_cat,
        )
        re_runner = ExternalBenchmarkRunner(
            model=MODEL,
            judge_model=args.judge_model,
            judge_llm=llm,
            max_concurrency=args.concurrency,
        )
        re_rep = await re_runner.rejudge_checkpoint(
            loco_convs,
            Path(args.rejudge_locomo_ckpt),
            output_checkpoint_path=out_dir / "exp3_locomo_rejudged.ckpt.jsonl",
            reference_system="chatree_two_tier",
        )
        re_md = re_rep.to_markdown_table()
        (out_dir / "exp3_locomo_rejudged.md").write_text(re_md, encoding="utf-8")
        (out_dir / "exp3_locomo_rejudged.tex").write_text(re_rep.to_latex_table(), encoding="utf-8")
        print(re_md, flush=True)
        return

    embedder = build_embedder(EMBED_MODEL)
    reranker = FastEmbedCrossEncoderReranker()

    totals: dict = {}
    if "1" in tracks or "2" in tracks:
        totals.update(await run_track1_and_2(out_dir, llm, embedder, reranker))
    if "3" in tracks or "4" in tracks:
        totals.update(
            await run_track3_and_4(
                out_dir,
                llm,
                embedder,
                reranker,
                tracks=tracks,
                concurrency=args.concurrency,
                locomo_convs=args.locomo_convs,
                locomo_per_cat=args.locomo_per_cat,
                lme_per_cat=args.lme_per_cat,
                lme_split=args.lme_split,
                judge_model=args.judge_model,
                include_paper_baselines=args.paper_baselines,
            )
        )
    if "6" in tracks:
        t6 = time.perf_counter()
        comp_runner = CompactionBenchmarkRunner(
            model=MODEL, llm=llm, embedder=embedder, reranker=reranker
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
    asyncio.run(main())
