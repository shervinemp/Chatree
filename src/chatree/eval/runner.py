from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from chatree.config import AblationConfig, ChatreeConfig
from chatree.eval.scenarios import MultiTangentScenario, TaskFamily
from chatree.eval.scoring import CostModel, ExactFactScorer, LLMJudgeScorer, QueryEvalResult
from chatree.protocols import Embedder, LLMProvider, Reranker
from chatree.providers.litellm import LiteLLMProvider
from chatree.providers.mock import MockLLMProvider
from chatree.sdk.client import Chatree
from chatree.tools import (
    TOOL_ASK_TREE,
    TOOL_REFLOG_JUMP,
    TOOL_REQUEST,
    TOOL_SEARCH_TREE,
    TOOL_SEND,
    TOOL_SPAWN_CHILD,
)
from chatree.types import TreeMetrics

BaselineMode = Literal[
    "linear",
    "sliding_window",
    "flat_rag",
    "tree_summary_only",
    "tree_ipc_no_staleness",
    "tree_ipc_full",
]

ALL_BASELINE_MODES: list[BaselineMode] = [
    "linear",
    "sliding_window",
    "flat_rag",
    "tree_summary_only",
    "tree_ipc_no_staleness",
    "tree_ipc_full",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 2)


class ModeBenchmarkResult(BaseModel):
    mode: BaselineMode
    scenario_name: str
    family: TaskFamily
    queries_passed: int
    queries_total: int
    pass_rate: float
    recall_rate: float
    rejected_leakage_rate: float = 0.0
    stale_contradiction_rate: float = 0.0
    avg_judge_score: float | None = None
    tool_precision: float | None = None
    tool_recall: float | None = None
    peak_root_context_tokens: int
    total_tokens_in: int
    total_tokens_out: int
    cache_hit_ratio: float
    estimated_cost_usd: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    shadow_turns: int
    stale_detections: int
    transitive_stale_detections: int = 0
    guardrail_blocks: int = 0
    metrics: TreeMetrics
    query_results: list[QueryEvalResult] = Field(default_factory=list)


class BenchmarkReport(BaseModel):
    scenario_name: str
    family: TaskFamily = "proposal_vs_decision"
    results: list[ModeBenchmarkResult]

    def to_markdown_table(self) -> str:
        lines = [
            f"### Chatree Evaluation: `{self.scenario_name}` ({self.family})",
            "",
            "| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.results:
            lines.append(
                f"| `{r.mode}` | {r.queries_passed}/{r.queries_total} ({int(r.pass_rate * 100)}%) "
                f"| {int(r.recall_rate * 100)}% "
                f"| {int(r.rejected_leakage_rate * 100)}% "
                f"| {int(r.stale_contradiction_rate * 100)}% "
                f"| {r.peak_root_context_tokens} "
                f"| ${r.estimated_cost_usd:.5f} "
                f"| {int(r.cache_hit_ratio * 100)}% "
                f"| {r.latency_p50_ms:.1f} / {r.latency_p95_ms:.1f} "
                f"| {r.shadow_turns} "
                f"| {r.stale_detections} ({r.transitive_stale_detections}) |"
            )
        return "\n".join(lines)

    def to_flat_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for r in self.results:
            rows.append(
                {
                    "scenario": r.scenario_name,
                    "family": r.family,
                    "mode": r.mode,
                    "queries_passed": r.queries_passed,
                    "queries_total": r.queries_total,
                    "pass_rate": r.pass_rate,
                    "recall_rate": r.recall_rate,
                    "rejected_leakage_rate": r.rejected_leakage_rate,
                    "stale_contradiction_rate": r.stale_contradiction_rate,
                    "avg_judge_score": r.avg_judge_score,
                    "tool_precision": r.tool_precision,
                    "tool_recall": r.tool_recall,
                    "peak_root_context_tokens": r.peak_root_context_tokens,
                    "total_tokens_in": r.total_tokens_in,
                    "total_tokens_out": r.total_tokens_out,
                    "cache_hit_ratio": r.cache_hit_ratio,
                    "estimated_cost_usd": r.estimated_cost_usd,
                    "latency_p50_ms": r.latency_p50_ms,
                    "latency_p95_ms": r.latency_p95_ms,
                    "shadow_turns": r.shadow_turns,
                    "stale_detections": r.stale_detections,
                    "transitive_stale_detections": r.transitive_stale_detections,
                    "guardrail_blocks": r.guardrail_blocks,
                }
            )
        return rows

    def to_csv(self) -> str:
        rows = self.to_flat_rows()
        if not rows:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(r.model_dump()) for r in self.results) + "\n"

    def save_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_csv(), encoding="utf-8")
        return p

    def save_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_jsonl(), encoding="utf-8")
        return p


class SuiteBenchmarkReport(BaseModel):
    reports: list[BenchmarkReport]

    def to_markdown_table(self) -> str:
        return "\n\n".join(r.to_markdown_table() for r in self.reports)

    def to_csv(self) -> str:
        all_rows: list[dict[str, object]] = []
        for rep in self.reports:
            all_rows.extend(rep.to_flat_rows())
        if not all_rows:
            return ""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
        return buf.getvalue()

    def to_jsonl(self) -> str:
        lines: list[str] = []
        for rep in self.reports:
            for r in rep.results:
                lines.append(json.dumps(r.model_dump()))
        return "\n".join(lines) + ("\n" if lines else "")

    def save_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_csv(), encoding="utf-8")
        return p

    def save_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_jsonl(), encoding="utf-8")
        return p


class BenchmarkRunner:
    """
    Runs controlled baselines and Chatree ablations over single or multi-family
    `MultiTangentScenario` suites, supporting real LiteLLM models, custom providers,
    optional LLM-as-a-Judge scoring, USD cost modeling, latency percentiles, and
    CSV/JSONL export.
    """

    def __init__(
        self,
        *,
        model: str = "mock/chatree-demo",
        shadow_model: str = "inherit",
        summary_model: str = "inherit",
        judge_model: str | None = None,
        sliding_window_messages: int = 6,
        oracle_routing: bool = False,
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        cost_model: CostModel | None = None,
        max_concurrency: int = 6,
    ) -> None:
        self.model = model
        self.shadow_model = shadow_model
        self.summary_model = summary_model
        self.judge_model = judge_model
        self.sliding_window_messages = sliding_window_messages
        self.oracle_routing = oracle_routing
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker
        self.cost_model = cost_model or CostModel()
        self.max_concurrency = max(1, max_concurrency)
        self.scorer = ExactFactScorer()

    async def run_scenarios(
        self,
        scenarios: list[MultiTangentScenario],
        modes: list[BaselineMode] | None = None,
    ) -> SuiteBenchmarkReport:
        reports = [await self.run_suite(sc, modes=modes) for sc in scenarios]
        return SuiteBenchmarkReport(reports=reports)

    async def run_suite(
        self,
        scenario: MultiTangentScenario,
        modes: list[BaselineMode] | None = None,
    ) -> BenchmarkReport:
        if scenario.execution_mode == "unscripted":
            selected: list[BaselineMode] = modes or ["linear", "tree_ipc_full"]
        else:
            selected = modes or list(ALL_BASELINE_MODES)
        if self.max_concurrency <= 1 or len(selected) <= 1:
            results = [await self.run_mode(scenario, m) for m in selected]
        else:
            sem = asyncio.Semaphore(self.max_concurrency)

            async def _run(m: BaselineMode) -> ModeBenchmarkResult:
                async with sem:
                    return await self.run_mode(scenario, m)

            results = list(await asyncio.gather(*(_run(m) for m in selected)))
        return BenchmarkReport(
            scenario_name=scenario.name,
            family=scenario.family,
            results=results,
        )

    async def run_mode(
        self, scenario: MultiTangentScenario, mode: BaselineMode
    ) -> ModeBenchmarkResult:
        ablations = AblationConfig(
            enable_shadow_turns=(mode in ("tree_ipc_full", "tree_ipc_no_staleness")),
            enable_staleness_notices=(mode == "tree_ipc_full"),
            enable_transitive_staleness=(mode == "tree_ipc_full"),
            enable_hybrid_rag=(mode not in ("linear", "sliding_window")),
            enable_subtree_rollups=(mode.startswith("tree_")),
        )
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            shadow_model=self.shadow_model,
            summary_model=self.summary_model,
            auto_commit_handoffs=True,
            ablations=ablations,
        )

        effective_llm = self.llm
        if effective_llm is None and self.model.startswith("mock/"):
            effective_llm = MockLLMProvider(stream_delay_s=0.0)

        async with Chatree.in_memory(
            config=cfg, llm=effective_llm, embedder=self.embedder, reranker=self.reranker
        ) as ct:
            tree = await ct.create_tree(
                scenario.name,
                default_model=self.model,
                shadow_model=self.shadow_model,
                summary_model=self.summary_model,
                auto_commit_handoffs=True,
            )
            root = await tree.root()
            judge: LLMJudgeScorer | None = None
            if self.judge_model:
                judge_llm = self.llm or ct.llm
                if not isinstance(judge_llm, LiteLLMProvider) and self.judge_model.startswith("mock/"):
                    judge_llm = ct.llm
                judge = LLMJudgeScorer(judge_llm, self.judge_model)

            branch_map: dict[str, str] = {"root": root.id}
            query_results: list[QueryEvalResult] = []
            latencies_ms: list[float] = []
            stale_detections = 0
            transitive_stale_detections = 0
            peak_root_tokens = 0

            # Track autonomous tool precision/recall for unscripted tasks
            expected_tool_calls = 0
            actual_tool_calls = 0
            matched_tool_calls = 0

            # Rolling window buffer for `sliding_window` baseline
            window_summary = ""

            async def _record_root_tokens() -> None:
                nonlocal peak_root_tokens
                rm = await root.model()
                tok = await ct.context.estimate_node_context_tokens(rm)
                if mode == "sliding_window":
                    msgs = await root.messages()
                    kept = msgs[-self.sliding_window_messages :]
                    tok = sum(max(4, len(m.content) // 4) for m in kept) + max(0, len(window_summary) // 4)
                if tok > peak_root_tokens:
                    peak_root_tokens = tok

            async def _score_and_record(
                *,
                query: str,
                target_branch: str,
                answer_text: str,
                expected_kw: list[str],
                rejected_kw: list[str],
                stale_kw: list[str],
                elapsed_ms: float,
            ) -> None:
                qres = self.scorer.score(
                    query=query,
                    target_branch=target_branch,
                    answer_text=answer_text,
                    expected_keywords=expected_kw,
                    rejected_keywords=rejected_kw,
                    stale_keywords=stale_kw,
                    latency_ms=elapsed_ms,
                )
                if judge is not None:
                    j_score, j_rat = await judge.judge(
                        query=query,
                        answer_text=answer_text,
                        expected_keywords=expected_kw,
                        forbidden_keywords=[*rejected_kw, *stale_kw],
                    )
                    qres.judge_score = j_score
                    qres.judge_rationale = j_rat
                query_results.append(qres)
                latencies_ms.append(elapsed_ms)

            for step in scenario.steps:
                # ------------------------------------------------------
                # Unscripted agentic step (real autonomous tool decisions)
                # ------------------------------------------------------
                if step.kind == "unscripted_turn":
                    target_id = (
                        root.id
                        if step.branch == "root"
                        else (await tree.head()).id
                    )
                    use_tools = mode != "linear"
                    t0 = time.perf_counter()
                    res = await tree.chat(
                        step.content,
                        node_id=target_id,
                        include_core_tools=use_tools,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    await _record_root_tokens()

                    fired_tools: list[str] = []
                    for ev in res.ipc_events:
                        if ev.event_type == "spawn":
                            fired_tools.append(TOOL_SPAWN_CHILD)
                        elif ev.event_type in ("send", "send_finish"):
                            fired_tools.append(TOOL_SEND)
                        elif ev.event_type == "search":
                            fired_tools.append(
                                TOOL_ASK_TREE if ev.payload.get("compound_ask") else TOOL_SEARCH_TREE
                            )
                        elif ev.event_type.startswith("request_"):
                            fired_tools.append(TOOL_REQUEST)
                        elif ev.event_type == "reflog_jump":
                            fired_tools.append(TOOL_REFLOG_JUMP)

                    # Evaluate exact expected tools AND semantically equivalent any-of groups
                    expected_groups: list[list[str]] = [
                        *([et] for et in step.expected_tools),
                        *step.expected_tools_any_of,
                    ]
                    expected_tool_calls += len(expected_groups)
                    actual_tool_calls += len(fired_tools)
                    remaining_fired = list(fired_tools)
                    for grp in expected_groups:
                        for candidate in grp:
                            if candidate in remaining_fired:
                                matched_tool_calls += 1
                                remaining_fired.remove(candidate)
                                break

                    if step.expected_keywords:
                        await _score_and_record(
                            query=step.content,
                            target_branch=step.branch,
                            answer_text=res.content,
                            expected_kw=step.expected_keywords,
                            rejected_kw=step.rejected_keywords,
                            stale_kw=step.stale_keywords,
                            elapsed_ms=elapsed_ms,
                        )
                    else:
                        latencies_ms.append(elapsed_ms)
                    continue

                # ------------------------------------------------------
                # Linear & Sliding-Window baselines (fair LLM synthesis)
                # ------------------------------------------------------
                if mode in ("linear", "sliding_window"):
                    if step.kind in ("chat", "revisit"):
                        await tree.chat(step.content, node_id=root.id, include_core_tools=False)
                        if mode == "sliding_window":
                            all_msgs = await root.messages()
                            if len(all_msgs) > self.sliding_window_messages:
                                evicted = all_msgs[: -self.sliding_window_messages]
                                # Real LLM rolling summary of evicted turns (capped concise budget)
                                sum_prompt = (
                                    "[Chatree Summarizer] Condense these evicted earlier turns into "
                                    "one concise high-level paragraph (omitting low-level parameter details):\n"
                                    + "\n".join(
                                        f"- {m.role}: {m.content[:140]}" for m in evicted[-6:]
                                    )
                                )
                                sum_resp = await ct.llm.complete(
                                    model=self.model,
                                    messages=[{"role": "user", "content": sum_prompt}],
                                    tools=None,
                                    temperature=0.1,
                                )
                                window_summary = sum_resp.content[:220]
                                await ct.repo.increment_node_usage(
                                    root.id, sum_resp.tokens_in, sum_resp.tokens_out, sum_resp.cache_read_tokens
                                )
                        await _record_root_tokens()
                    elif step.kind == "verify_query":
                        t0 = time.perf_counter()
                        if mode == "sliding_window":
                            all_msgs = await root.messages()
                            visible = all_msgs[-self.sliding_window_messages :]
                            win_messages: list[dict[str, str]] = [
                                {
                                    "role": "system",
                                    "content": "Answer accurately using only the rolling summary and recent window.",
                                }
                            ]
                            if window_summary:
                                win_messages.append(
                                    {"role": "assistant", "content": f"Rolling summary: {window_summary}"}
                                )
                            for m in visible:
                                if m.role in ("user", "assistant"):
                                    win_messages.append({"role": m.role, "content": m.content})
                            win_messages.append(
                                {
                                    "role": "user",
                                    "content": f"[Sliding Window Context]\nQuestion: {step.content}",
                                }
                            )
                            win_resp = await ct.llm.complete(
                                model=self.model, messages=win_messages, tools=None, temperature=0.1
                            )
                            await ct.repo.increment_node_usage(
                                root.id, win_resp.tokens_in, win_resp.tokens_out, win_resp.cache_read_tokens
                            )
                            answer_text = win_resp.content
                        else:
                            # Fair linear baseline: score ONLY the model's actual reply (`res.content`),
                            # NEVER concatenating the entire raw transcript!
                            res = await tree.chat(step.content, node_id=root.id, include_core_tools=False)
                            answer_text = res.content

                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        await _record_root_tokens()

                        await _score_and_record(
                            query=step.content,
                            target_branch=step.target_branch or "root",
                            answer_text=answer_text,
                            expected_kw=step.expected_keywords,
                            rejected_kw=step.rejected_keywords,
                            stale_kw=step.stale_keywords,
                            elapsed_ms=elapsed_ms,
                        )
                    continue

                # ------------------------------------------------------
                # Tree-based baselines & Chatree ablations
                # ------------------------------------------------------
                if step.kind == "spawn":
                    parent_id = branch_map.get(step.parent, root.id)
                    child = await tree.spawn(
                        step.goal or step.title or step.branch,
                        parent_id=parent_id,
                        title=step.title or step.branch,
                        mode=step.context_mode,
                        switch_head=True,
                    )
                    branch_map[step.branch] = child.id

                elif step.kind == "chat":
                    nid = branch_map.get(step.branch, root.id)
                    await tree.chat(step.content, node_id=nid, include_core_tools=False)
                    await _record_root_tokens()

                elif step.kind == "finish":
                    finish_nid = branch_map.get(step.branch)
                    if finish_nid and finish_nid != root.id:
                        await tree.node(finish_nid).finish(
                            summary=step.content or None, auto_commit=True, return_to_parent=True
                        )
                        await _record_root_tokens()

                elif step.kind == "revisit":
                    nid = branch_map.get(step.branch, root.id)
                    await tree.jump(nid, note="Benchmark revisit")
                    await tree.chat(step.content, node_id=nid, include_core_tools=False)
                    await _record_root_tokens()

                elif step.kind == "verify_query":
                    target_name = step.target_branch or "root"
                    target_id = branch_map.get(target_name, root.id)
                    root_model = await root.model()
                    if root_model.stale_dependencies:
                        stale_detections += len(root_model.stale_dependencies)
                        transitive_stale_detections += sum(
                            1 for d in root_model.stale_dependencies if d.hops > 1
                        )

                    t0 = time.perf_counter()
                    answer_text = ""
                    if mode == "flat_rag":
                        # Fair Flat-RAG baseline: retrieve top-3 chunks AND synthesize through the LLM
                        hits = await tree.search(
                            step.content,
                            scope="all",
                            top_k=3,
                            granularities={"chunk"},
                        )
                        chunks_txt = "\n".join(f"- {h.snippet}" for h in hits)
                        rag_resp = await ct.llm.complete(
                            model=self.model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Answer the question concisely based on the retrieved context snippets.",
                                },
                                {
                                    "role": "user",
                                    "content": f"[RAG Context]\n{chunks_txt}\n\nQuestion: {step.content}",
                                },
                            ],
                            tools=None,
                            temperature=0.1,
                        )
                        await ct.repo.increment_node_usage(
                            root.id, rag_resp.tokens_in, rag_resp.tokens_out, rag_resp.cache_read_tokens
                        )
                        answer_text = rag_resp.content
                    elif mode == "tree_summary_only":
                        out = await tree.request(target_id, prompt=step.content, caller_id=root.id)
                        answer_text = str(out.get("summary") or out.get("answer") or "")
                    elif mode == "tree_ipc_no_staleness":
                        # Without staleness alerts, if parent already committed a handoff for this
                        # branch, parent answers from its own context instead of re-querying!
                        target_stale = any(
                            d.source_node_id == target_id for d in root_model.stale_dependencies
                        )
                        if target_stale or step.is_post_revision:
                            res = await tree.chat(
                                step.content, node_id=root.id, include_core_tools=False
                            )
                            answer_text = res.content
                        else:
                            out = await tree.request(
                                target_id, prompt=step.content, caller_id=root.id
                            )
                            answer_text = str(out.get("answer") or out.get("summary") or "")
                    else:  # tree_ipc_full
                        if self.oracle_routing:
                            out = await tree.request(
                                target_id, prompt=step.content, caller_id=root.id
                            )
                            answer_text = str(out.get("answer") or out.get("summary") or "")
                        else:
                            # Non-oracle Retrieve-then-Ask: real 3-tier hybrid search + causal
                            # freshness filtering + shadow turn (level playing field with flat_rag!)
                            ask_out = await tree.ask(
                                step.content,
                                scope="all",
                                max_targets=2,
                                caller_id=root.id,
                            )
                            answers = ask_out.get("found_answers") or ask_out.get("answers") or []
                            answer_text = "\n".join(
                                str(a.get("answer") or a.get("summary") or "") for a in answers
                            )

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    await _record_root_tokens()

                    await _score_and_record(
                        query=step.content,
                        target_branch=target_name,
                        answer_text=answer_text,
                        expected_kw=step.expected_keywords,
                        rejected_kw=step.rejected_keywords,
                        stale_kw=step.stale_keywords,
                        elapsed_ms=elapsed_ms,
                    )

            metrics = await tree.metrics()
            total_cnt = len(query_results)
            passed_cnt = sum(1 for q in query_results if q.passed)
            recall_avg = (
                round(sum(q.recall_score for q in query_results) / total_cnt, 4)
                if total_cnt
                else 1.0
            )
            leak_rate = (
                round(sum(1 for q in query_results if q.rejected_leakage) / total_cnt, 4)
                if total_cnt
                else 0.0
            )
            stale_err_rate = (
                round(sum(1 for q in query_results if q.stale_contradiction) / total_cnt, 4)
                if total_cnt
                else 0.0
            )
            judge_scores = [q.judge_score for q in query_results if q.judge_score is not None]
            avg_judge = (
                round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None
            )

            tool_prec = (
                round(matched_tool_calls / actual_tool_calls, 4)
                if scenario.execution_mode == "unscripted" and actual_tool_calls > 0
                else (1.0 if scenario.execution_mode == "unscripted" and expected_tool_calls == 0 else (0.0 if scenario.execution_mode == "unscripted" else None))
            )
            tool_rec = (
                round(matched_tool_calls / expected_tool_calls, 4)
                if scenario.execution_mode == "unscripted" and expected_tool_calls > 0
                else None
            )

            est_usd = self.cost_model.estimate_usd(
                self.model,
                tokens_in=metrics.total_tokens_in,
                tokens_out=metrics.total_tokens_out,
                cache_read_tokens=metrics.total_cache_read_tokens,
            )

            return ModeBenchmarkResult(
                mode=mode,
                scenario_name=scenario.name,
                family=scenario.family,
                queries_passed=passed_cnt,
                queries_total=total_cnt,
                pass_rate=round(passed_cnt / total_cnt, 4) if total_cnt else 1.0,
                recall_rate=recall_avg,
                rejected_leakage_rate=leak_rate,
                stale_contradiction_rate=stale_err_rate,
                avg_judge_score=avg_judge,
                tool_precision=tool_prec,
                tool_recall=tool_rec,
                peak_root_context_tokens=peak_root_tokens,
                total_tokens_in=metrics.total_tokens_in,
                total_tokens_out=metrics.total_tokens_out,
                cache_hit_ratio=metrics.cache_hit_ratio,
                estimated_cost_usd=est_usd,
                latency_p50_ms=_percentile(latencies_ms, 0.50),
                latency_p95_ms=_percentile(latencies_ms, 0.95),
                shadow_turns=metrics.shadow_turns,
                stale_detections=stale_detections,
                transitive_stale_detections=transitive_stale_detections,
                guardrail_blocks=metrics.guardrail_blocks,
                metrics=metrics,
                query_results=query_results,
            )
