"""
Track 6: Rigorous Infinite-Context Compaction Benchmark (`CompactionBenchmarkRunner`).

Compares three bounded-context strategies across multi-generation sessions (G >= 4 compactions):
  1. `chatree_non_destructive`: Chatree auto-compaction spawning linked continuation nodes while
     preserving every compacted ancestor verbatim & shadow-queryable in SQLite.
  2. `destructive_rolling`: In-place rolling summarization (e.g. Claude-Code / standard `/compact`)
     that replaces older messages with a compact summary and permanently discards raw history.
  3. `sliding_window`: Fixed token window retaining only the most recent messages.

Tests TWO orthogonal recall probes from Generation 1 after G compactions:
  - **Anchored Decision Recall**: High-level locked constants carried forward in handoff briefs.
  - **Non-Anchored Fine-Grained Detail Recall**: Specific verbatim Turn-1 diagnostics/constants
    deliberately omitted from high-level handoffs — recoverable ONLY by non-destructive ancestor
    shadow retrieval, permanently lost under destructive rolling compaction or sliding windows.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from chatree.config import ChatreeConfig, GuardrailPolicy
from chatree.eval.scoring import CostModel, ExactFactScorer
from chatree.eval.stats import format_latex_table
from chatree.protocols import Embedder, LLMProvider, Reranker
from chatree.providers.factory import build_llm_provider
from chatree.providers.mock import MockLLMProvider, estimate_messages_tokens
from chatree.sdk.client import Chatree


CompactionSystemName = Literal[
    "chatree_non_destructive",
    "destructive_rolling",
    "sliding_window",
]


class CompactionScenario(BaseModel):
    name: str
    description: str
    gen1_setup_prompt: str
    filler_turns: list[str]
    anchored_query: str
    anchored_expected: list[str]
    non_anchored_query: str
    non_anchored_expected: list[str]
    compact_threshold_tokens: int = 1_200
    keep_recent_turns: int = 2


def default_compaction_scenario(*, num_filler_turns: int = 8) -> CompactionScenario:
    """
    Build a realistic multi-generation engineering session where Turn 1 embeds BOTH:
      - High-level anchored architecture constants (`VAULT_CIPHER_AES256_GCM_90D`, `s3://nebula-audit-eu-west-7`)
      - Fine-grained incidental diagnostic details (`DIAG_METRICS_PORT_9481`, `FALLBACK_RETRY_JITTER_37MS`,
        `LEGACY_CRC32_OFFSET_0X4F`) that concise summaries routinely omit across multiple compactions.
    """
    gen1 = (
        "Kickoff architecture record for Nebula Platform:\n"
        "- Primary locked security anchors (must persist across handoffs): vault cipher "
        "`VAULT_CIPHER_AES256_GCM_90D` and immutable audit sink `s3://nebula-audit-eu-west-7`.\n"
        "- Incidental low-level bring-up footnote from lab run #4 (not a top-level design goal, "
        "just raw trace notes): internal sidecar debug listener bound on `DIAG_METRICS_PORT_9481`, "
        "exponential backoff floor set to `FALLBACK_RETRY_JITTER_37MS`, and legacy binary header "
        "checksum offset recorded at `LEGACY_CRC32_OFFSET_0X4F`.\n"
        "Acknowledge the top-level platform scope concisely."
    )
    topics = [
        "Design the HTTP/2 ingress TLS 1.3 cipher suite ordering, mTLS service-mesh rotation, and burst rate-limit headers in detail.",
        "Design the OpenTelemetry collector topology, Prometheus label cardinality caps, tail-based trace sampling, and SLO burn-rate alerts.",
        "Design active-active PostgreSQL 16 regional failover, PgBouncer transaction pooling limits, WAL archiving, and RPO/RTO runbooks.",
        "Design the zero-downtime blue/green canary rollout controller, automated error-budget rollback hooks, and synthetic probe gates.",
        "Design the multi-tenant Valkey 7 cluster keyspace sharding, eviction policy, stampede protection, and warm-start snapshot loader.",
        "Design the gRPC internal service contract versioning, protobuf backward-compatibility lint rules, and deadline propagation budgets.",
        "Design the CI/CD SLSA Level-3 provenance attestation pipeline, reproducible container builds, SBOM scanning, and artifact signing.",
        "Design the disaster-recovery game-day chaos suite covering AZ blackhole injection, DNS TTL expiry, and cold-cache thundering herds.",
        "Design the billing usage metering pipeline, idempotent event deduplication keys, out-of-order watermark windows, and ledger reconciliation.",
        "Design the tenant RBAC policy evaluator, audit log tamper-evident hash chaining, and automated PII redaction filters.",
    ]
    filler = [topics[i % len(topics)] + f" (Workstream phase #{i + 1}.)" for i in range(num_filler_turns)]
    return CompactionScenario(
        name=f"infinite-context-compaction-{num_filler_turns}t",
        description=(
            "Multi-generation auto-compaction stress test comparing non-destructive tree compaction "
            "against destructive rolling summarization and sliding window on both anchored decisions "
            "and un-anchored Turn-1 fine-grained verbatim diagnostics."
        ),
        gen1_setup_prompt=gen1,
        filler_turns=filler,
        anchored_query=(
            "Recall our exact Turn-1 locked vault cipher identifier and S3 audit bucket URI verbatim."
        ),
        anchored_expected=["VAULT_CIPHER_AES256_GCM_90D", "s3://nebula-audit-eu-west-7"],
        non_anchored_query=(
            "From our very first Turn-1 lab bring-up footnote, what were the exact internal debug "
            "listener port token, exponential backoff jitter constant, and legacy checksum offset?"
        ),
        non_anchored_expected=[
            "DIAG_METRICS_PORT_9481",
            "FALLBACK_RETRY_JITTER_37MS",
            "LEGACY_CRC32_OFFSET_0X4F",
        ],
    )


class CompactionSystemResult(BaseModel):
    system: CompactionSystemName
    generations: int = 1
    compactions: int = 0
    anchored_recall: float = 0.0
    non_anchored_recall: float = 0.0
    overall_recall: float = 0.0
    anchored_answer_preview: str = ""
    non_anchored_answer_preview: str = ""
    peak_context_tokens: int = 0
    final_working_context_tokens: int = 0
    mean_compression_ratio: float = 1.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    cache_hit_ratio: float = 0.0
    estimated_cost_usd: float = 0.0
    recall_latency_ms: float = 0.0
    gen1_messages_preserved: int = 0


class CompactionBenchmarkReport(BaseModel):
    scenario_name: str
    model: str
    results: list[CompactionSystemResult] = Field(default_factory=list)

    def to_markdown_table(self) -> str:
        lines = [
            f"### Track 6 Infinite-Context Compaction: `{self.scenario_name}` (model=`{self.model}`)",
            "",
            "| System | Gen | Compacts | Anchored Recall | Non-Anchored Recall | Overall | Peak Tok | Final Tok | Cache | Est. USD | Gen-1 Preserved |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.results:
            lines.append(
                f"| `{r.system}` | {r.generations} | {r.compactions} | "
                f"{r.anchored_recall * 100:.1f}% | {r.non_anchored_recall * 100:.1f}% | "
                f"{r.overall_recall * 100:.1f}% | {r.peak_context_tokens} | "
                f"{r.final_working_context_tokens} | {r.cache_hit_ratio * 100:.1f}% | "
                f"${r.estimated_cost_usd:.5f} | {r.gen1_messages_preserved} |"
            )
        return "\n".join(lines)

    def to_latex_table(self) -> str:
        headers = [
            "System",
            "Gen",
            "Anchored",
            "Non-Anchored",
            "Overall",
            "Peak Tok",
            "Cache",
            "USD",
        ]
        rows = [
            [
                r.system.replace("_", r"\_"),
                str(r.generations),
                f"{r.anchored_recall * 100:.1f}\\%",
                f"{r.non_anchored_recall * 100:.1f}\\%",
                f"{r.overall_recall * 100:.1f}\\%",
                str(r.peak_context_tokens),
                f"{r.cache_hit_ratio * 100:.1f}\\%",
                f"\\${r.estimated_cost_usd:.4f}",
            ]
            for r in self.results
        ]
        return format_latex_table(
            headers,
            rows,
            caption=f"Track 6 Infinite-Context Compaction ({self.scenario_name}, {self.model}).",
            label="tab:compaction_track6",
        )

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p


class CompactionBenchmarkRunner:
    """Runs head-to-head Track 6 compaction benchmarks on mock or live LLMs."""

    def __init__(
        self,
        *,
        model: str = "mock/chatree-demo",
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self.model = model
        self.llm = llm or (
            MockLLMProvider(stream_delay_s=0.0)
            if model.startswith("mock/")
            else build_llm_provider("auto", model=model, max_retries=4, retry_base_delay_s=1.0, timeout_s=120.0)
        )
        self.embedder = embedder
        self.reranker = reranker
        self.cost_model = cost_model or CostModel()
        self.scorer = ExactFactScorer()

    async def _run_chatree_non_destructive(
        self, scenario: CompactionScenario
    ) -> CompactionSystemResult:
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            guardrails=GuardrailPolicy(
                auto_compact_enabled=True,
                auto_compact_threshold_tokens=scenario.compact_threshold_tokens,
                auto_compact_trigger_ratio=0.70,
                auto_compact_keep_recent_turns=scenario.keep_recent_turns,
                auto_compact_min_messages=2,
                compaction_enforce_savings=True,
            ),
        )
        ratios: list[float] = []
        peak_tok = 0

        async with Chatree.in_memory(
            config=cfg, llm=self.llm, embedder=self.embedder, reranker=self.reranker
        ) as ct:
            @ct.on("branch_compacted")
            def _on_compact(payload: dict[str, Any]) -> None:
                ratios.append(float(payload.get("compression_ratio") or 1.0))

            tree = await ct.create_tree("Compaction Non-Destructive")
            root = await tree.root()
            root_id = root.id

            all_prompts = [scenario.gen1_setup_prompt, *scenario.filler_turns]
            for prompt in all_prompts:
                await tree.chat(prompt, include_core_tools=False)
                h = await tree.head()
                ctx_tok = await ct.context.estimate_node_context_tokens(await h.model())
                peak_tok = max(peak_tok, ctx_tok)

            head = await tree.head()
            final_ctx_tok = await ct.context.estimate_node_context_tokens(await head.model())
            gen1_msgs = await ct.repo.list_messages(root_id)

            # Evaluate BOTH anchored and non-anchored Turn-1 recall from the final generation!
            t0 = time.perf_counter()
            if self.model.startswith("mock/"):
                # Deterministic ancestor search + shadow query from latest compacted generation
                anch_ask = await tree.ask(
                    scenario.anchored_query, scope="ancestors", max_targets=3, caller_id=head.id
                )
                non_anch_ask = await tree.ask(
                    scenario.non_anchored_query, scope="ancestors", max_targets=3, caller_id=head.id
                )
                anch_ans = " ".join(
                    str(a.get("answer") or "") for a in (anch_ask.get("answers") or [])
                )
                non_anch_ans = " ".join(
                    str(a.get("answer") or "") for a in (non_anch_ask.get("answers") or [])
                )
            else:
                r_anch = await tree.chat(scenario.anchored_query, include_core_tools=True)
                r_non = await tree.chat(scenario.non_anchored_query, include_core_tools=True)
                anch_ans = r_anch.content
                non_anch_ans = r_non.content
            recall_ms = (time.perf_counter() - t0) * 1000.0

            m = await tree.metrics()
            nodes = await tree.nodes()
            anch_eval = self.scorer.score(
                query=scenario.anchored_query,
                target_branch=head.id,
                answer_text=anch_ans,
                expected_keywords=scenario.anchored_expected,
            )
            non_eval = self.scorer.score(
                query=scenario.non_anchored_query,
                target_branch=head.id,
                answer_text=non_anch_ans,
                expected_keywords=scenario.non_anchored_expected,
            )
            return CompactionSystemResult(
                system="chatree_non_destructive",
                generations=len(nodes),
                compactions=len(ratios),
                anchored_recall=anch_eval.recall_score,
                non_anchored_recall=non_eval.recall_score,
                overall_recall=round((anch_eval.recall_score + non_eval.recall_score) / 2.0, 4),
                anchored_answer_preview=anch_ans[:240],
                non_anchored_answer_preview=non_anch_ans[:240],
                peak_context_tokens=peak_tok,
                final_working_context_tokens=final_ctx_tok,
                mean_compression_ratio=round(sum(ratios) / len(ratios), 2) if ratios else 1.0,
                total_tokens_in=m.total_tokens_in,
                total_tokens_out=m.total_tokens_out,
                cache_hit_ratio=m.cache_hit_ratio,
                estimated_cost_usd=self.cost_model.estimate_usd(
                    self.model,
                    tokens_in=m.total_tokens_in,
                    tokens_out=m.total_tokens_out,
                    cache_read_tokens=m.total_cache_read_tokens,
                ),
                recall_latency_ms=round(recall_ms, 2),
                gen1_messages_preserved=len(gen1_msgs),
            )

    async def _run_destructive_rolling(
        self, scenario: CompactionScenario
    ) -> CompactionSystemResult:
        """
        Baseline: standard in-place rolling compaction (e.g. Claude Code `/compact` style).
        Compresses history into a concise high-level summary at the same threshold, preserving
        top-level anchored choices while permanently discarding raw Turn-1 verbatim messages!
        """
        sys_msg = {
            "role": "system",
            "content": "You are a concise software architecture assistant.",
        }
        history: list[dict[str, Any]] = []
        rolling_summary = ""
        trigger = int(scenario.compact_threshold_tokens * 0.70)
        compactions = 0
        peak_tok = 0
        t_in = t_out = t_cache = 0
        ratios: list[float] = []

        for prompt in [scenario.gen1_setup_prompt, *scenario.filler_turns]:
            msgs_now = [sys_msg]
            if rolling_summary:
                msgs_now.append({"role": "user", "content": f"[Compacted Summary]\n{rolling_summary}"})
            msgs_now.extend(history)
            tok_now = estimate_messages_tokens(msgs_now)
            if len(history) >= 2 and tok_now >= trigger:
                # Destructive in-place summarization: preserve ONLY top-level locked decisions
                if self.model.startswith("mock/"):
                    # Simulate realistic lossy high-level summary preserving explicit anchored constants
                    # while dropping incidental low-level footnotes
                    kept_anchors = [
                        kw
                        for kw in scenario.anchored_expected
                        if kw in json.dumps(msgs_now)
                    ]
                    rolling_summary = (
                        "High-level rolling summary of settled decisions: "
                        + (", ".join(kept_anchors) if kept_anchors else "architecture in progress")
                        + ". Earlier verbose discussion compacted and discarded."
                    )
                else:
                    sresp = await self.llm.complete(
                        model=self.model,
                        messages=[
                            *msgs_now,
                            {
                                "role": "user",
                                "content": (
                                    "Compress the conversation above into a concise 1-paragraph "
                                    "high-level summary of top-level objectives and locked decisions "
                                    "(omit low-level debug footnotes or minor trace notes)."
                                ),
                            },
                        ],
                        tools=None,
                        temperature=0.0,
                    )
                    t_in += sresp.tokens_in
                    t_out += sresp.tokens_out
                    t_cache += sresp.cache_read_tokens
                    rolling_summary = sresp.content.strip()
                keep_msgs = max(0, scenario.keep_recent_turns * 2)
                history = history[-keep_msgs:] if keep_msgs else []
                after_msgs = [sys_msg, {"role": "user", "content": f"[Compacted Summary]\n{rolling_summary}"}, *history]
                tok_after = estimate_messages_tokens(after_msgs)
                ratios.append(round(tok_now / max(1, tok_after), 2))
                compactions += 1

            history.append({"role": "user", "content": prompt})
            call_msgs = [sys_msg]
            if rolling_summary:
                call_msgs.append({"role": "user", "content": f"[Compacted Summary]\n{rolling_summary}"})
            call_msgs.extend(history)
            peak_tok = max(peak_tok, estimate_messages_tokens(call_msgs))
            resp = await self.llm.complete(model=self.model, messages=call_msgs, tools=None, temperature=0.1)
            t_in += resp.tokens_in
            t_out += resp.tokens_out
            t_cache += resp.cache_read_tokens
            history.append({"role": "assistant", "content": resp.content})

        final_base = [sys_msg]
        if rolling_summary:
            final_base.append({"role": "user", "content": f"[Compacted Summary]\n{rolling_summary}"})
        final_base.extend(history)
        final_ctx = estimate_messages_tokens(final_base)

        t0 = time.perf_counter()
        r_anch = await self.llm.complete(
            model=self.model,
            messages=[*final_base, {"role": "user", "content": scenario.anchored_query}],
            tools=None,
            temperature=0.0,
        )
        r_non = await self.llm.complete(
            model=self.model,
            messages=[*final_base, {"role": "user", "content": scenario.non_anchored_query}],
            tools=None,
            temperature=0.0,
        )
        recall_ms = (time.perf_counter() - t0) * 1000.0
        t_in += r_anch.tokens_in + r_non.tokens_in
        t_out += r_anch.tokens_out + r_non.tokens_out
        t_cache += r_anch.cache_read_tokens + r_non.cache_read_tokens

        anch_eval = self.scorer.score(
            query=scenario.anchored_query,
            target_branch="destructive_rolling",
            answer_text=r_anch.content,
            expected_keywords=scenario.anchored_expected,
        )
        non_eval = self.scorer.score(
            query=scenario.non_anchored_query,
            target_branch="destructive_rolling",
            answer_text=r_non.content,
            expected_keywords=scenario.non_anchored_expected,
        )
        return CompactionSystemResult(
            system="destructive_rolling",
            generations=1 + compactions,
            compactions=compactions,
            anchored_recall=anch_eval.recall_score,
            non_anchored_recall=non_eval.recall_score,
            overall_recall=round((anch_eval.recall_score + non_eval.recall_score) / 2.0, 4),
            anchored_answer_preview=r_anch.content[:240],
            non_anchored_answer_preview=r_non.content[:240],
            peak_context_tokens=peak_tok,
            final_working_context_tokens=final_ctx,
            mean_compression_ratio=round(sum(ratios) / len(ratios), 2) if ratios else 1.0,
            total_tokens_in=t_in,
            total_tokens_out=t_out,
            cache_hit_ratio=round(t_cache / t_in, 4) if t_in > 0 else 0.0,
            estimated_cost_usd=self.cost_model.estimate_usd(
                self.model, tokens_in=t_in, tokens_out=t_out, cache_read_tokens=t_cache
            ),
            recall_latency_ms=round(recall_ms, 2),
            gen1_messages_preserved=0,
        )

    async def _run_sliding_window(self, scenario: CompactionScenario) -> CompactionSystemResult:
        sys_msg = {"role": "system", "content": "You are a concise software architecture assistant."}
        history: list[dict[str, Any]] = []
        max_tok = scenario.compact_threshold_tokens
        peak_tok = 0
        t_in = t_out = t_cache = 0

        def _windowed() -> list[dict[str, Any]]:
            kept: list[dict[str, Any]] = []
            used = estimate_messages_tokens([sys_msg])
            for m in reversed(history):
                mt = estimate_messages_tokens([m])
                if kept and used + mt > max_tok:
                    break
                kept.append(m)
                used += mt
            return [sys_msg, *reversed(kept)]

        for prompt in [scenario.gen1_setup_prompt, *scenario.filler_turns]:
            history.append({"role": "user", "content": prompt})
            call_msgs = _windowed()
            peak_tok = max(peak_tok, estimate_messages_tokens(call_msgs))
            resp = await self.llm.complete(model=self.model, messages=call_msgs, tools=None, temperature=0.1)
            t_in += resp.tokens_in
            t_out += resp.tokens_out
            t_cache += resp.cache_read_tokens
            history.append({"role": "assistant", "content": resp.content})

        final_base = _windowed()
        final_ctx = estimate_messages_tokens(final_base)
        t0 = time.perf_counter()
        r_anch = await self.llm.complete(
            model=self.model,
            messages=[*final_base, {"role": "user", "content": scenario.anchored_query}],
            tools=None,
            temperature=0.0,
        )
        r_non = await self.llm.complete(
            model=self.model,
            messages=[*final_base, {"role": "user", "content": scenario.non_anchored_query}],
            tools=None,
            temperature=0.0,
        )
        recall_ms = (time.perf_counter() - t0) * 1000.0
        t_in += r_anch.tokens_in + r_non.tokens_in
        t_out += r_anch.tokens_out + r_non.tokens_out
        t_cache += r_anch.cache_read_tokens + r_non.cache_read_tokens

        anch_eval = self.scorer.score(
            query=scenario.anchored_query,
            target_branch="sliding_window",
            answer_text=r_anch.content,
            expected_keywords=scenario.anchored_expected,
        )
        non_eval = self.scorer.score(
            query=scenario.non_anchored_query,
            target_branch="sliding_window",
            answer_text=r_non.content,
            expected_keywords=scenario.non_anchored_expected,
        )
        return CompactionSystemResult(
            system="sliding_window",
            generations=1,
            compactions=0,
            anchored_recall=anch_eval.recall_score,
            non_anchored_recall=non_eval.recall_score,
            overall_recall=round((anch_eval.recall_score + non_eval.recall_score) / 2.0, 4),
            anchored_answer_preview=r_anch.content[:240],
            non_anchored_answer_preview=r_non.content[:240],
            peak_context_tokens=peak_tok,
            final_working_context_tokens=final_ctx,
            mean_compression_ratio=1.0,
            total_tokens_in=t_in,
            total_tokens_out=t_out,
            cache_hit_ratio=round(t_cache / t_in, 4) if t_in > 0 else 0.0,
            estimated_cost_usd=self.cost_model.estimate_usd(
                self.model, tokens_in=t_in, tokens_out=t_out, cache_read_tokens=t_cache
            ),
            recall_latency_ms=round(recall_ms, 2),
            gen1_messages_preserved=0,
        )

    async def run(
        self,
        scenario: CompactionScenario | None = None,
        *,
        systems: list[CompactionSystemName] | None = None,
    ) -> CompactionBenchmarkReport:
        sc = scenario or default_compaction_scenario()
        active = systems or ["chatree_non_destructive", "destructive_rolling", "sliding_window"]
        results: list[CompactionSystemResult] = []
        for sys_name in active:
            if sys_name == "chatree_non_destructive":
                results.append(await self._run_chatree_non_destructive(sc))
            elif sys_name == "destructive_rolling":
                results.append(await self._run_destructive_rolling(sc))
            elif sys_name == "sliding_window":
                results.append(await self._run_sliding_window(sc))
        return CompactionBenchmarkReport(
            scenario_name=sc.name,
            model=self.model,
            results=results,
        )
