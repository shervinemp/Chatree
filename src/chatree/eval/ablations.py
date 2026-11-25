from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field

from chatree.config import AblationConfig, ChatreeConfig
from chatree.eval.runner import BaselineMode, BenchmarkRunner, ModeBenchmarkResult
from chatree.eval.scenarios import (
    MultiTangentScenario,
    build_full_eval_suite,
    distractor_scaling_scenario,
)
from chatree.eval.scoring import CostModel, ExactFactScorer, QueryEvalResult
from chatree.eval.stats import (
    BootstrapCI,
    McNemarResult,
    bootstrap_ci,
    format_latex_table,
    mcnemar_test,
)
from chatree.protocols import Embedder, LLMProvider, Reranker
from chatree.providers.mock import MockLLMProvider
from chatree.sdk.client import Chatree


# Canonical systematic ablation matrix isolating every Chatree architectural mechanism
ABLATION_PRESETS: dict[str, tuple[AblationConfig, bool]] = {
    # name -> (AblationConfig, oracle_routing)
    "chatree_full": (AblationConfig(), False),
    "chatree_oracle_routing": (AblationConfig(), True),
    "no_shadow_turns_folding": (AblationConfig(enable_shadow_turns=False), False),
    "no_staleness": (
        AblationConfig(enable_staleness_notices=False, enable_transitive_staleness=False),
        False,
    ),
    "no_transitive_staleness": (AblationConfig(enable_transitive_staleness=False), False),
    "no_materiality_gating": (AblationConfig(enable_materiality_gating=False), False),
    "no_shadow_cache": (AblationConfig(enable_shadow_cache=False), False),
    "no_speculative_pointers": (AblationConfig(enable_speculative_pointers=False), False),
    "no_subtree_rollups": (AblationConfig(enable_subtree_rollups=False), False),
    "no_hybrid_rag_fts_only": (AblationConfig(enable_hybrid_rag=False), False),
}


class AblationRow(BaseModel):
    ablation_name: str
    queries_passed: int
    queries_total: int
    pass_rate: float
    pass_ci: BootstrapCI
    rejected_leakage_rate: float
    stale_contradiction_rate: float
    peak_root_context_tokens: int
    total_tokens: int
    cache_hit_ratio: float
    estimated_cost_usd: float
    shadow_turns: int
    shadow_cache_hits: int
    stale_detections: int
    mcnemar_vs_full: McNemarResult | None = None
    query_passed_flags: list[bool] = Field(default_factory=list)


class AblationMatrixReport(BaseModel):
    model: str
    scenarios: list[str]
    rows: list[AblationRow]

    def to_markdown_table(self) -> str:
        lines = [
            f"### Chatree Systematic Ablation Matrix (model=`{self.model}`, scenarios={len(self.scenarios)})",
            "",
            "| Variant | Strict Pass (95% CI) | Leakage | Stale Err | Peak Tok | Total Tok | Est. USD | Shadow (Cache) | McNemar p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.rows:
            ci = f"{r.pass_rate * 100:.1f}% [{r.pass_ci.ci_low * 100:.1f}, {r.pass_ci.ci_high * 100:.1f}]"
            p_str = (
                "ref"
                if r.ablation_name == "chatree_full"
                else (
                    f"{r.mcnemar_vs_full.p_value:.4f}" + ("*" if r.mcnemar_vs_full.significant_05 else "")
                    if r.mcnemar_vs_full
                    else "—"
                )
            )
            lines.append(
                f"| `{r.ablation_name}` | {ci} | {r.rejected_leakage_rate * 100:.1f}% "
                f"| {r.stale_contradiction_rate * 100:.1f}% | {r.peak_root_context_tokens} "
                f"| {r.total_tokens} | ${r.estimated_cost_usd:.5f} "
                f"| {r.shadow_turns} ({r.shadow_cache_hits}) | {p_str} |"
            )
        return "\n".join(lines)

    def to_latex_table(
        self,
        *,
        caption: str = "Chatree Systematic Ablation Matrix across Multi-Tangent Stress Scenarios",
        label: str = "tab:chatree-ablations",
    ) -> str:
        headers = [
            "Variant",
            "Pass (95% CI)",
            "Leakage",
            "Stale Err",
            "Peak Tok",
            "Cost (USD)",
            "p-val",
        ]
        rows = [
            [
                r.ablation_name,
                f"{r.pass_rate * 100:.1f}% [{r.pass_ci.ci_low * 100:.1f}, {r.pass_ci.ci_high * 100:.1f}]",
                f"{r.rejected_leakage_rate * 100:.1f}%",
                f"{r.stale_contradiction_rate * 100:.1f}%",
                str(r.peak_root_context_tokens),
                f"${r.estimated_cost_usd:.5f}",
                "ref"
                if r.ablation_name == "chatree_full"
                else (
                    f"{r.mcnemar_vs_full.p_value:.4f}" + ("*" if r.mcnemar_vs_full.significant_05 else "")
                    if r.mcnemar_vs_full
                    else "-"
                ),
            ]
            for r in self.rows
        ]
        return format_latex_table(headers, rows, caption=caption, label=label)

    def save_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        flat = [
            {
                "ablation": r.ablation_name,
                "queries_passed": r.queries_passed,
                "queries_total": r.queries_total,
                "pass_rate": r.pass_rate,
                "pass_ci_low": r.pass_ci.ci_low,
                "pass_ci_high": r.pass_ci.ci_high,
                "rejected_leakage_rate": r.rejected_leakage_rate,
                "stale_contradiction_rate": r.stale_contradiction_rate,
                "peak_root_context_tokens": r.peak_root_context_tokens,
                "total_tokens": r.total_tokens,
                "cache_hit_ratio": r.cache_hit_ratio,
                "estimated_cost_usd": r.estimated_cost_usd,
                "shadow_turns": r.shadow_turns,
                "shadow_cache_hits": r.shadow_cache_hits,
                "stale_detections": r.stale_detections,
                "mcnemar_p_vs_full": r.mcnemar_vs_full.p_value if r.mcnemar_vs_full else None,
            }
            for r in self.rows
        ]
        buf = io.StringIO()
        if flat:
            w = csv.DictWriter(buf, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
        p.write_text(buf.getvalue(), encoding="utf-8")
        return p


class CrossoverPoint(BaseModel):
    k_branches: int
    mode: BaselineMode
    pass_rate: float
    rejected_leakage_rate: float
    peak_root_context_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_p50_ms: float


class CrossoverSweepReport(BaseModel):
    model: str
    ks: list[int]
    points: list[CrossoverPoint]

    def to_markdown_table(self) -> str:
        lines = [
            f"### Scaling & Crossover Curve ($K \\in {self.ks}$, model=`{self.model}`)",
            "",
            "| K | Mode | Strict Pass | Leakage | Peak Root Tok | Total Tok | Est. USD | p50 ms |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for pt in self.points:
            lines.append(
                f"| {pt.k_branches} | `{pt.mode}` | {pt.pass_rate * 100:.1f}% "
                f"| {pt.rejected_leakage_rate * 100:.1f}% | {pt.peak_root_context_tokens} "
                f"| {pt.total_tokens} | ${pt.estimated_cost_usd:.5f} | {pt.latency_p50_ms:.1f} |"
            )
        return "\n".join(lines)

    def to_latex_table(
        self,
        *,
        caption: str = "Scaling and Cost/Context Crossover Curve across Distractor Count K",
        label: str = "tab:chatree-crossover",
    ) -> str:
        headers = ["K", "Mode", "Pass", "Leakage", "Peak Tok", "Total Tok", "Cost (USD)", "p50 ms"]
        rows = [
            [
                str(pt.k_branches),
                pt.mode,
                f"{pt.pass_rate * 100:.1f}%",
                f"{pt.rejected_leakage_rate * 100:.1f}%",
                str(pt.peak_root_context_tokens),
                str(pt.total_tokens),
                f"${pt.estimated_cost_usd:.5f}",
                f"{pt.latency_p50_ms:.1f}",
            ]
            for pt in self.points
        ]
        return format_latex_table(headers, rows, caption=caption, label=label)


class AblationMatrixRunner:
    """Executes systematic feature ablations and K-scaling crossover sweeps for camera-ready tables."""

    def __init__(
        self,
        *,
        model: str = "mock/chatree-demo",
        shadow_model: str = "inherit",
        summary_model: str = "inherit",
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        cost_model: CostModel | None = None,
        max_concurrency: int = 6,
    ) -> None:
        self.model = model
        self.shadow_model = shadow_model
        self.summary_model = summary_model
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        self.cost_model = cost_model or CostModel()
        self.max_concurrency = max(1, max_concurrency)
        self.scorer = ExactFactScorer()

    async def _run_single_ablation_on_scenario(
        self,
        scenario: MultiTangentScenario,
        ablations: AblationConfig,
        oracle_routing: bool,
    ) -> tuple[list[QueryEvalResult], int, int, int, int, int, int, int]:
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            shadow_model=self.shadow_model,
            summary_model=self.summary_model,
            auto_commit_handoffs=True,
            ablations=ablations,
        )
        eff_llm = self.llm or (MockLLMProvider(stream_delay_s=0.0) if self.model.startswith("mock/") else None)
        async with Chatree.in_memory(
            config=cfg, llm=eff_llm, embedder=self.embedder, reranker=self.reranker
        ) as ct:
            tree = await ct.create_tree(
                scenario.name,
                default_model=self.model,
                shadow_model=self.shadow_model,
                summary_model=self.summary_model,
                auto_commit_handoffs=True,
            )
            root = await tree.root()
            branch_map: dict[str, str] = {"root": root.id}
            q_results: list[QueryEvalResult] = []
            peak_tok = 0
            stale_det = 0

            async def _track_root() -> None:
                nonlocal peak_tok
                tok = await ct.context.estimate_node_context_tokens(await root.model())
                if tok > peak_tok:
                    peak_tok = tok

            for step in scenario.steps:
                if step.kind == "spawn":
                    pid = branch_map.get(step.parent, root.id)
                    ch = await tree.spawn(
                        step.goal or step.title or step.branch,
                        parent_id=pid,
                        title=step.title or step.branch,
                        mode=step.context_mode,
                        switch_head=True,
                    )
                    branch_map[step.branch] = ch.id
                elif step.kind == "chat":
                    nid = branch_map.get(step.branch, root.id)
                    await tree.chat(step.content, node_id=nid, include_core_tools=False)
                    await _track_root()
                elif step.kind == "finish":
                    nid = branch_map.get(step.branch)
                    if nid and nid != root.id:
                        await tree.node(nid).finish(
                            summary=step.content or None, auto_commit=True, return_to_parent=True
                        )
                        await _track_root()
                elif step.kind == "revisit":
                    nid = branch_map.get(step.branch, root.id)
                    await tree.jump(nid, note="Ablation revisit")
                    await tree.chat(step.content, node_id=nid, include_core_tools=False)
                    await _track_root()
                elif step.kind == "verify_query":
                    target_name = step.target_branch or "root"
                    target_id = branch_map.get(target_name, root.id)
                    rm = await root.model()
                    if ablations.enable_staleness_notices and rm.stale_dependencies:
                        relevant_stale = [
                            d
                            for d in rm.stale_dependencies
                            if ablations.enable_transitive_staleness or d.hops == 1
                        ]
                        stale_det += len(relevant_stale)
                    else:
                        relevant_stale = []

                    if not ablations.enable_shadow_turns:
                        out = await tree.request(target_id, prompt=step.content, caller_id=root.id)
                        ans = str(out.get("summary") or out.get("answer") or "")
                    elif not ablations.enable_staleness_notices and step.is_post_revision:
                        res = await tree.chat(step.content, node_id=root.id, include_core_tools=False)
                        ans = res.content
                    elif (
                        not ablations.enable_transitive_staleness
                        and step.is_post_revision
                        and not any(d.hops == 1 for d in relevant_stale)
                    ):
                        # Without transitive staleness, 2-hop upstream root receives no notice -> answers from stale handoff
                        res = await tree.chat(step.content, node_id=root.id, include_core_tools=False)
                        ans = res.content
                    elif oracle_routing:
                        out = await tree.request(target_id, prompt=step.content, caller_id=root.id)
                        ans = str(out.get("answer") or out.get("summary") or "")
                    else:
                        ask_out = await tree.ask(
                            step.content, scope="all", max_targets=2, caller_id=root.id
                        )
                        answers = ask_out.get("found_answers") or ask_out.get("answers") or []
                        ans = "\n".join(str(a.get("answer") or a.get("summary") or "") for a in answers)

                    q_results.append(
                        self.scorer.score(
                            query=step.content,
                            target_branch=target_name,
                            answer_text=ans,
                            expected_keywords=step.expected_keywords,
                            rejected_keywords=step.rejected_keywords,
                            stale_keywords=step.stale_keywords,
                        )
                    )
                    await _track_root()

            m = await tree.metrics()
            return (
                q_results,
                peak_tok,
                m.total_tokens_in,
                m.total_tokens_out,
                m.total_cache_read_tokens,
                m.shadow_turns,
                m.shadow_cache_hits,
                stale_det,
            )

    async def run_ablation_matrix(
        self,
        scenarios: Sequence[MultiTangentScenario] | None = None,
        *,
        preset_names: Sequence[str] | None = None,
    ) -> AblationMatrixReport:
        selected_scenarios = [
            s
            for s in (scenarios or build_full_eval_suite(num_distractors=3, crossover_branches=3, crossover_turns=2))
            if s.execution_mode == "scripted"
        ]
        names = list(preset_names or ABLATION_PRESETS.keys())
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _eval_preset(name: str) -> AblationRow:
            abl, oracle = ABLATION_PRESETS[name]
            async with sem:
                all_qs: list[QueryEvalResult] = []
                max_peak = 0
                tot_in = 0
                tot_out = 0
                tot_cache = 0
                tot_shadow = 0
                tot_scache = 0
                tot_stale = 0

                for sc in selected_scenarios:
                    qs, pk, tin, tout, tcache, sh, sch, st = await self._run_single_ablation_on_scenario(
                        sc, abl, oracle
                    )
                    all_qs.extend(qs)
                    max_peak = max(max_peak, pk)
                    tot_in += tin
                    tot_out += tout
                    tot_cache += tcache
                    tot_shadow += sh
                    tot_scache += sch
                    tot_stale += st

                flags = [q.passed for q in all_qs]
                n = len(all_qs)
                passed = sum(1 for f in flags if f)
                leak = sum(1 for q in all_qs if q.rejected_leakage) / n if n else 0.0
                stale_err = sum(1 for q in all_qs if q.stale_contradiction) / n if n else 0.0
                return AblationRow(
                    ablation_name=name,
                    queries_passed=passed,
                    queries_total=n,
                    pass_rate=round(passed / n, 4) if n else 1.0,
                    pass_ci=bootstrap_ci([1.0 if f else 0.0 for f in flags]),
                    rejected_leakage_rate=round(leak, 4),
                    stale_contradiction_rate=round(stale_err, 4),
                    peak_root_context_tokens=max_peak,
                    total_tokens=tot_in + tot_out,
                    cache_hit_ratio=round(tot_cache / tot_in, 4) if tot_in else 0.0,
                    estimated_cost_usd=self.cost_model.estimate_usd(
                        self.model, tokens_in=tot_in, tokens_out=tot_out, cache_read_tokens=tot_cache
                    ),
                    shadow_turns=tot_shadow,
                    shadow_cache_hits=tot_scache,
                    stale_detections=tot_stale,
                    query_passed_flags=flags,
                )

        rows = list(await asyncio.gather(*(_eval_preset(n) for n in names)))

        ref = next((r for r in rows if r.ablation_name == "chatree_full"), None)
        if ref is not None:
            for r in rows:
                if r.ablation_name != "chatree_full" and len(r.query_passed_flags) == len(ref.query_passed_flags):
                    r.mcnemar_vs_full = mcnemar_test(ref.query_passed_flags, r.query_passed_flags)

        return AblationMatrixReport(
            model=self.model,
            scenarios=[s.name for s in selected_scenarios],
            rows=rows,
        )

    async def run_crossover_sweep(
        self,
        ks: Sequence[int] = (2, 4, 6),
        *,
        modes: Sequence[BaselineMode] = (
            "linear",
            "sliding_window",
            "flat_rag",
            "tree_summary_only",
            "tree_ipc_full",
        ),
    ) -> CrossoverSweepReport:
        runner = BenchmarkRunner(
            model=self.model,
            shadow_model=self.shadow_model,
            summary_model=self.summary_model,
            oracle_routing=False,
            llm=self.llm,
            embedder=self.embedder,
            reranker=self.reranker,
            cost_model=self.cost_model,
            max_concurrency=self.max_concurrency,
        )
        points: list[CrossoverPoint] = []
        for k in ks:
            rep = await runner.run_suite(distractor_scaling_scenario(num_distractors=k), modes=list(modes))
            for r in rep.results:
                points.append(
                    CrossoverPoint(
                        k_branches=k,
                        mode=r.mode,
                        pass_rate=r.pass_rate,
                        rejected_leakage_rate=r.rejected_leakage_rate,
                        peak_root_context_tokens=r.peak_root_context_tokens,
                        total_tokens=r.total_tokens_in + r.total_tokens_out,
                        estimated_cost_usd=r.estimated_cost_usd,
                        latency_p50_ms=r.latency_p50_ms,
                    )
                )
        return CrossoverSweepReport(model=self.model, ks=list(ks), points=points)
