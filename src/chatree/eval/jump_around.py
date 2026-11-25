from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from chatree.config import AblationConfig, ChatreeConfig, GuardrailPolicy
from chatree.eval.scoring import CostModel, ExactFactScorer, LLMJudgeScorer, QueryEvalResult
from chatree.eval.stats import format_latex_table
from chatree.eval.stats import percentile as _percentile
from chatree.protocols import Embedder, LLMProvider, Reranker
from chatree.providers.litellm import LiteLLMProvider
from chatree.providers.mock import MockLLMProvider, estimate_messages_tokens
from chatree.sdk.client import Chatree

JumpAroundTurnKind = Literal[
    "chat",
    "branch",
    "jump_back",
    "dead_end",
    "query",
    "repeat_query",
]

JumpAroundSystemName = Literal[
    "chatree_live",
    "linear",
    "sliding_window",
    "flat_rag",
    "rlm_official",
]


class JumpAroundTurn(BaseModel):
    """One interactive user turn in a realistic multi-branch jump-around chatbot session."""

    turn_index: int
    kind: JumpAroundTurnKind
    user_message: str
    expected_branch_hint: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    rejected_keywords: list[str] = Field(default_factory=list)
    stale_keywords: list[str] = Field(default_factory=list)
    must_use_jump: bool = False
    expect_shadow_cache_hit: bool = False


class JumpAroundScenario(BaseModel):
    name: str
    description: str
    turns: list[JumpAroundTurn]


def default_jump_around_scenario() -> JumpAroundScenario:
    """
    Realistic 18-turn interactive multi-topic chatbot session testing:
      1. Unscripted branching (Auth, Database, Caching)
      2. Leaving a branch unfinished and switching away (triggers background Tier-2 summary)
      3. Cross-branch shadow query from one active branch (Database) into an unfinished sibling (Auth)
      4. Dead-end tangent exploration & explicit abandonment (GraphQL Federation prototype)
      5. `HEAD@{k}` / reflog jump-back mid-stream revision (JWT_TTL_15M -> JWT_TTL_5M_ROTATING)
      6. Final cross-branch integration (zero dead-end leakage, zero stale revision contradiction)
      7. Repeated cross-branch verification exercising `shadow_cache` / warm KV-cache reuse
    """
    turns = [
        JumpAroundTurn(
            turn_index=1,
            kind="chat",
            user_message=(
                "Let's design the architecture for Nebula API Gateway in Rust/Axum with a hard SLO "
                "of P99_LATENCY_120MS. Keep this root thread focused on top-level decisions."
            ),
            expected_branch_hint="Nebula",
        ),
        JumpAroundTurn(
            turn_index=2,
            kind="branch",
            user_message=(
                "Create a new child branch titled 'Auth Design' (mode='clean') to design JWT "
                "signing, token TTL, and password hashing."
            ),
            expected_branch_hint="Auth",
        ),
        JumpAroundTurn(
            turn_index=3,
            kind="chat",
            user_message=(
                "In Auth Design, let's lock in ED25519_SIGNING for JWTs, ARGON2ID_64MB for "
                "password hashing, and an initial token lifetime of JWT_TTL_15M."
            ),
            expected_branch_hint="Auth",
        ),
        JumpAroundTurn(
            turn_index=4,
            kind="chat",
            user_message=(
                "Also note in Auth Design that key rotation publishes public keys at "
                "/.well-known/jwks.json every 24 hours. Leave this branch open for now."
            ),
            expected_branch_hint="Auth",
        ),
        JumpAroundTurn(
            turn_index=5,
            kind="jump_back",
            user_message=(
                "Switch branch back to the root thread (HEAD@{1}) without closing Auth Design — "
                "we'll revisit Auth later."
            ),
            expected_branch_hint="Nebula",
            must_use_jump=True,
        ),
        JumpAroundTurn(
            turn_index=6,
            kind="branch",
            user_message=(
                "Create a new child branch titled 'Database Design' (mode='clean') to design our "
                "primary SQL storage, connection pooling, and replication."
            ),
            expected_branch_hint="Database",
        ),
        JumpAroundTurn(
            turn_index=7,
            kind="chat",
            user_message=(
                "In Database Design, our settled stack is POSTGRES_16_PGBOUNCER with POOL_SIZE_64 "
                "and one read replica in READ_REPLICA_EU_WEST."
            ),
            expected_branch_hint="Database",
        ),
        JumpAroundTurn(
            turn_index=8,
            kind="query",
            user_message=(
                "While we stay here in Database Design to plan the sessions table, search and query "
                "our Auth Design branch: what JWT signing algorithm and current JWT TTL did we pick?"
            ),
            expected_branch_hint="Database",
            expected_keywords=["ED25519_SIGNING", "JWT_TTL_15M"],
        ),
        JumpAroundTurn(
            turn_index=9,
            kind="dead_end",
            user_message=(
                "Hold on — create a clean side branch titled 'GraphQL Prototype' to explore a "
                "speculative GraphQL gateway idea."
            ),
            expected_branch_hint="GraphQL",
        ),
        JumpAroundTurn(
            turn_index=10,
            kind="dead_end",
            user_message=(
                "For this speculative prototype, suppose we tried GRAPHQL_FEDERATION_V0 with "
                "APOLLO_ROUTER_EXPERIMENTAL and 24 subgraph hops."
            ),
            expected_branch_hint="GraphQL",
        ),
        JumpAroundTurn(
            turn_index=11,
            kind="jump_back",
            user_message=(
                "Wait, that adds too much latency for P99_LATENCY_120MS. Scrap the GraphQL "
                "Prototype completely — do NOT merge it into Nebula's architecture. Switch branch "
                "back to the root Nebula thread."
            ),
            expected_branch_hint="Nebula",
            must_use_jump=True,
        ),
        JumpAroundTurn(
            turn_index=12,
            kind="branch",
            user_message=(
                "Create a clean child branch titled 'Caching Layer' for response caching and "
                "token revocation list caching."
            ),
            expected_branch_hint="Caching",
        ),
        JumpAroundTurn(
            turn_index=13,
            kind="chat",
            user_message=(
                "In Caching Layer, let's settle on VALKEY_CLUSTER_7 with CACHE_TTL_300S and a "
                "revocation bloom filter aligned with Auth's JWT TTL."
            ),
            expected_branch_hint="Caching",
        ),
        JumpAroundTurn(
            turn_index=14,
            kind="jump_back",
            user_message=(
                "Security review just came back! Switch branch to Auth Design and immediately "
                "revise our token lifetime: replace JWT_TTL_15M with JWT_TTL_5M_ROTATING and "
                "enable REFRESH_REUSE_DETECTION."
            ),
            expected_branch_hint="Auth",
            must_use_jump=True,
        ),
        JumpAroundTurn(
            turn_index=15,
            kind="chat",
            user_message=(
                "Confirm the updated Auth Design policy in this branch: ED25519_SIGNING, "
                "ARGON2ID_64MB, JWT_TTL_5M_ROTATING, and REFRESH_REUSE_DETECTION (JWT_TTL_15M is "
                "superseded and retired)."
            ),
            expected_branch_hint="Auth",
        ),
        JumpAroundTurn(
            turn_index=16,
            kind="jump_back",
            user_message="Now switch branch back to the root Nebula thread to assemble the final spec.",
            expected_branch_hint="Nebula",
            must_use_jump=True,
        ),
        JumpAroundTurn(
            turn_index=17,
            kind="query",
            user_message=(
                "Assemble the final production configuration for Nebula API Gateway by checking our "
                "settled Auth Design, Database Design, and Caching Layer branches. State the exact "
                "tokens for: (1) JWT signing algorithm, current JWT TTL, and password hashing, "
                "(2) Database engine and pool size, and (3) Caching store. Exclude scrapped prototypes "
                "and retired TTLs."
            ),
            expected_branch_hint="Nebula",
            expected_keywords=[
                "ED25519_SIGNING",
                "JWT_TTL_5M_ROTATING",
                "ARGON2ID_64MB",
                "POSTGRES_16_PGBOUNCER",
                "POOL_SIZE_64",
                "VALKEY_CLUSTER_7",
            ],
            rejected_keywords=["GRAPHQL_FEDERATION_V0", "APOLLO_ROUTER_EXPERIMENTAL"],
            stale_keywords=["JWT_TTL_15M"],
        ),
        JumpAroundTurn(
            turn_index=18,
            kind="repeat_query",
            user_message=(
                "Quick final verification before sign-off: confirm Auth Design's exact JWT signing "
                "algorithm and current active JWT TTL (not any retired value)."
            ),
            expected_branch_hint="Nebula",
            expected_keywords=["ED25519_SIGNING", "JWT_TTL_5M_ROTATING"],
            stale_keywords=["JWT_TTL_15M"],
            expect_shadow_cache_hit=True,
        ),
    ]
    return JumpAroundScenario(
        name="live-jump-around-18t",
        description=(
            "18-turn realistic interactive chatbot session with unscripted branching, unfinished "
            "switches, cross-branch shadow reads, dead-end tangent abandonment, reflog jump-back "
            "revision, and final multi-branch synthesis."
        ),
        turns=turns,
    )


def natural_jump_around_scenario() -> JumpAroundScenario:
    """
    System-agnostic 18-turn interactive multi-topic session using purely natural human prompts
    (zero tool jargon like `mode='clean'` or `HEAD@{1}`) so tool-free baselines (`linear`,
    `sliding_window`, `flat_rag`) are NEVER tripped up by missing-tool meta-instructions while
    Chatree autonomously branches, archives dead-ends, and jumps across topics.
    """
    turns = [
        JumpAroundTurn(
            turn_index=1,
            kind="chat",
            user_message=(
                "Let's design the architecture for Nebula API Gateway in Rust/Axum with a hard SLO "
                "of P99_LATENCY_120MS. Keep our top-level plan clean and concise."
            ),
            expected_branch_hint="Nebula",
        ),
        JumpAroundTurn(
            turn_index=2,
            kind="branch",
            user_message=(
                "Let's dive into a separate Auth Design discussion to settle JWT signing, "
                "token TTL, and password hashing."
            ),
            expected_branch_hint="Auth",
        ),
        JumpAroundTurn(
            turn_index=3,
            kind="chat",
            user_message=(
                "For Auth Design, let's lock in ED25519_SIGNING for JWTs, ARGON2ID_64MB for "
                "password hashing, and an initial token lifetime of JWT_TTL_15M."
            ),
            expected_branch_hint="Auth",
        ),
        JumpAroundTurn(
            turn_index=4,
            kind="chat",
            user_message=(
                "Also note for Auth Design that public keys rotate at /.well-known/jwks.json "
                "every 24 hours. Keep Auth open — we may revise it after security review."
            ),
            expected_branch_hint="Auth",
        ),
        JumpAroundTurn(
            turn_index=5,
            kind="jump_back",
            user_message=(
                "Pause Auth Design for now and go back to our top-level Nebula architecture plan."
            ),
            expected_branch_hint="Nebula",
            must_use_jump=True,
        ),
        JumpAroundTurn(
            turn_index=6,
            kind="branch",
            user_message=(
                "Now let's open a focused Database Design track for primary SQL storage, "
                "connection pooling, and replication."
            ),
            expected_branch_hint="Database",
        ),
        JumpAroundTurn(
            turn_index=7,
            kind="chat",
            user_message=(
                "For Database Design, our settled stack is POSTGRES_16_PGBOUNCER with POOL_SIZE_64 "
                "and one read replica in READ_REPLICA_EU_WEST."
            ),
            expected_branch_hint="Database",
        ),
        JumpAroundTurn(
            turn_index=8,
            kind="query",
            user_message=(
                "While staying focused on Database Design to plan the sessions table, double-check "
                "our Auth Design decisions: what JWT signing algorithm and current JWT TTL did we pick?"
            ),
            expected_branch_hint="Database",
            expected_keywords=["ED25519_SIGNING", "JWT_TTL_15M"],
        ),
        JumpAroundTurn(
            turn_index=9,
            kind="dead_end",
            user_message=(
                "Wait — let's spin up an isolated GraphQL Prototype exploration to test a "
                "speculative federated gateway idea."
            ),
            expected_branch_hint="GraphQL",
        ),
        JumpAroundTurn(
            turn_index=10,
            kind="dead_end",
            user_message=(
                "Suppose this speculative prototype used GRAPHQL_FEDERATION_V0 with "
                "APOLLO_ROUTER_EXPERIMENTAL across 24 subgraph hops."
            ),
            expected_branch_hint="GraphQL",
        ),
        JumpAroundTurn(
            turn_index=11,
            kind="jump_back",
            user_message=(
                "That completely blows past P99_LATENCY_120MS. Scrap and archive the GraphQL "
                "Prototype completely — do NOT include it in Nebula — and return to the main Nebula plan."
            ),
            expected_branch_hint="Nebula",
            must_use_jump=True,
        ),
        JumpAroundTurn(
            turn_index=12,
            kind="branch",
            user_message=(
                "Next, let's start a Caching Layer track for response caching and token revocation caching."
            ),
            expected_branch_hint="Caching",
        ),
        JumpAroundTurn(
            turn_index=13,
            kind="chat",
            user_message=(
                "For Caching Layer, let's settle on VALKEY_CLUSTER_7 with CACHE_TTL_300S and a "
                "revocation bloom filter aligned with Auth's JWT TTL."
            ),
            expected_branch_hint="Caching",
        ),
        JumpAroundTurn(
            turn_index=14,
            kind="jump_back",
            user_message=(
                "Security review just came back! Let's jump back to Auth Design and revise our "
                "token lifetime immediately: replace JWT_TTL_15M with JWT_TTL_5M_ROTATING and "
                "enable REFRESH_REUSE_DETECTION."
            ),
            expected_branch_hint="Auth",
            must_use_jump=True,
        ),
        JumpAroundTurn(
            turn_index=15,
            kind="chat",
            user_message=(
                "Confirm the updated Auth Design policy: ED25519_SIGNING, ARGON2ID_64MB, "
                "JWT_TTL_5M_ROTATING, and REFRESH_REUSE_DETECTION (JWT_TTL_15M is retired)."
            ),
            expected_branch_hint="Auth",
        ),
        JumpAroundTurn(
            turn_index=16,
            kind="jump_back",
            user_message="Now return to the main Nebula plan to assemble the final production spec.",
            expected_branch_hint="Nebula",
            must_use_jump=True,
        ),
        JumpAroundTurn(
            turn_index=17,
            kind="query",
            user_message=(
                "Assemble the final production configuration for Nebula API Gateway from our "
                "settled Auth Design, Database Design, and Caching Layer decisions. State the exact "
                "tokens for: (1) JWT signing algorithm, current JWT TTL, and password hashing, "
                "(2) Database engine and pool size, and (3) Caching store. Exclude scrapped prototypes "
                "and retired TTLs."
            ),
            expected_branch_hint="Nebula",
            expected_keywords=[
                "ED25519_SIGNING",
                "JWT_TTL_5M_ROTATING",
                "ARGON2ID_64MB",
                "POSTGRES_16_PGBOUNCER",
                "POOL_SIZE_64",
                "VALKEY_CLUSTER_7",
            ],
            rejected_keywords=["GRAPHQL_FEDERATION_V0", "APOLLO_ROUTER_EXPERIMENTAL"],
            stale_keywords=["JWT_TTL_15M"],
        ),
        JumpAroundTurn(
            turn_index=18,
            kind="repeat_query",
            user_message=(
                "Quick final verification before sign-off: confirm Auth Design's exact JWT signing "
                "algorithm and current active JWT TTL."
            ),
            expected_branch_hint="Nebula",
            expected_keywords=["ED25519_SIGNING", "JWT_TTL_5M_ROTATING"],
            stale_keywords=["JWT_TTL_15M"],
            expect_shadow_cache_hit=True,
        ),
    ]
    return JumpAroundScenario(
        name="natural-jump-around-18t",
        description=(
            "System-agnostic 18-turn jump-around benchmark with natural prompts (no tool jargon), "
            "dead-end archiving, mid-stream revision, and final multi-topic synthesis."
        ),
        turns=turns,
    )


class JumpAroundTurnTrace(BaseModel):
    turn_index: int
    kind: JumpAroundTurnKind
    head_node_id: str = ""
    head_title: str = ""
    navigation_matched: bool | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    latency_ms: float = 0.0
    tools_called: list[str] = Field(default_factory=list)
    answer_preview: str = ""


class JumpAroundSystemResult(BaseModel):
    system: JumpAroundSystemName
    scenario_name: str
    turns_total: int
    queries_passed: int
    queries_total: int
    strict_pass_rate: float
    mean_recall: float
    dead_end_leakage_rate: float
    stale_revision_error_rate: float
    avg_judge_score: float | None = None
    navigation_accuracy: float | None = None
    branches_created: int = 0
    reflog_jumps: int = 0
    shadow_turns: int = 0
    shadow_cache_hits: int = 0
    bg_summaries_completed: int = 0
    peak_context_tokens: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cache_read_tokens: int = 0
    cache_hit_ratio: float = 0.0
    estimated_cost_usd: float = 0.0
    turn_latency_p50_ms: float = 0.0
    turn_latency_p95_ms: float = 0.0
    per_turn_context_tokens: list[int] = Field(default_factory=list)
    query_results: list[QueryEvalResult] = Field(default_factory=list)
    turn_traces: list[JumpAroundTurnTrace] = Field(default_factory=list)


class JumpAroundReport(BaseModel):
    scenario_name: str
    model: str
    results: list[JumpAroundSystemResult]

    def to_markdown_table(self) -> str:
        lines = [
            f"### Track 5 — Live Multi-Turn Jump-Around Chatbot Eval (`{self.scenario_name}`, model=`{self.model}`)",
            "",
            "| System | Strict Pass | Recall | Dead-End Leak | Stale Err | Judge | Nav Acc | Peak Tok | Total Tok In | Cache Hit | Est. USD | Turn p50 / p95 ms | Branches / Jumps / Shadow (Cache) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.results:
            judge_s = f"{r.avg_judge_score:.2f}" if r.avg_judge_score is not None else "—"
            nav_s = f"{r.navigation_accuracy * 100:.1f}%" if r.navigation_accuracy is not None else "—"
            lines.append(
                f"| `{r.system}` "
                f"| {r.queries_passed}/{r.queries_total} ({r.strict_pass_rate * 100:.0f}%) "
                f"| {r.mean_recall * 100:.1f}% "
                f"| {r.dead_end_leakage_rate * 100:.1f}% "
                f"| {r.stale_revision_error_rate * 100:.1f}% "
                f"| {judge_s} "
                f"| {nav_s} "
                f"| {r.peak_context_tokens} "
                f"| {r.total_tokens_in} "
                f"| {r.cache_hit_ratio * 100:.1f}% "
                f"| ${r.estimated_cost_usd:.5f} "
                f"| {r.turn_latency_p50_ms:.0f} / {r.turn_latency_p95_ms:.0f} "
                f"| {r.branches_created} / {r.reflog_jumps} / {r.shadow_turns} ({r.shadow_cache_hits}) |"
            )
        return "\n".join(lines)

    def to_flat_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in self.results:
            rows.append(
                {
                    "scenario": r.scenario_name,
                    "system": r.system,
                    "turns_total": r.turns_total,
                    "queries_passed": r.queries_passed,
                    "queries_total": r.queries_total,
                    "strict_pass_rate": r.strict_pass_rate,
                    "mean_recall": r.mean_recall,
                    "dead_end_leakage_rate": r.dead_end_leakage_rate,
                    "stale_revision_error_rate": r.stale_revision_error_rate,
                    "avg_judge_score": r.avg_judge_score,
                    "navigation_accuracy": r.navigation_accuracy,
                    "branches_created": r.branches_created,
                    "reflog_jumps": r.reflog_jumps,
                    "shadow_turns": r.shadow_turns,
                    "shadow_cache_hits": r.shadow_cache_hits,
                    "bg_summaries_completed": r.bg_summaries_completed,
                    "peak_context_tokens": r.peak_context_tokens,
                    "total_tokens_in": r.total_tokens_in,
                    "total_tokens_out": r.total_tokens_out,
                    "total_cache_read_tokens": r.total_cache_read_tokens,
                    "cache_hit_ratio": r.cache_hit_ratio,
                    "estimated_cost_usd": r.estimated_cost_usd,
                    "turn_latency_p50_ms": r.turn_latency_p50_ms,
                    "turn_latency_p95_ms": r.turn_latency_p95_ms,
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

    def to_latex_table(self) -> str:
        headers = [
            "System",
            "Strict Pass",
            "Recall",
            "Leak",
            "Stale",
            "Nav Acc",
            "Peak Tok",
            "Cache %",
            "USD",
            "p50 ms",
        ]
        rows: list[list[str]] = []
        for r in self.results:
            nav_s = f"{r.navigation_accuracy * 100:.0f}\\%" if r.navigation_accuracy is not None else "--"
            rows.append(
                [
                    r.system,
                    f"{r.strict_pass_rate * 100:.0f}\\%",
                    f"{r.mean_recall * 100:.1f}\\%",
                    f"{r.dead_end_leakage_rate * 100:.0f}\\%",
                    f"{r.stale_revision_error_rate * 100:.0f}\\%",
                    nav_s,
                    str(r.peak_context_tokens),
                    f"{r.cache_hit_ratio * 100:.1f}\\%",
                    f"\\${r.estimated_cost_usd:.4f}",
                    f"{r.turn_latency_p50_ms:.0f}",
                ]
            )
        return format_latex_table(
            headers=headers,
            rows=rows,
            caption=f"Track 5: Live 18-turn jump-around chatbot benchmark ({self.scenario_name}, {self.model}).",
            label="tab:jump_around_live",
        )

    def save_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_csv(), encoding="utf-8")
        return p

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p


class JumpAroundBenchmarkRunner:
    """
    Evaluates real multi-turn conversational jumping, branching, abandonment, and mid-stream
    revision across:
      - `chatree_live` (live HEAD turns with `dialogue_tree_*` tools + bg summaries + shadow turns)
      - `linear` (single accumulating linear thread)
      - `sliding_window` (rolling summary + bounded recent window)
      - `flat_rag` (flat conversation log + hybrid BM25/vector top-k chunk synthesis on queries)
      - optional `rlm_official` (MIT `rlms` REPL over accumulated live transcript on query turns)
    """

    def __init__(
        self,
        *,
        model: str = "mock/chatree-demo",
        shadow_model: str = "inherit",
        judge_model: str | None = None,
        sliding_window_messages: int = 8,
        llm: LLMProvider | None = None,
        judge_llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self.model = model
        self.shadow_model = shadow_model
        self.judge_model = judge_model
        self.sliding_window_messages = sliding_window_messages
        self.llm = llm
        self.judge_llm = judge_llm
        self.embedder = embedder
        self.reranker = reranker
        self.cost_model = cost_model or CostModel()
        self.scorer = ExactFactScorer()

    def _effective_llm(self) -> LLMProvider:
        if self.llm is not None:
            return self.llm
        if self.model.startswith("mock/"):
            return MockLLMProvider(stream_delay_s=0.0)
        return LiteLLMProvider()

    async def run(
        self,
        scenario: JumpAroundScenario | None = None,
        *,
        systems: list[JumpAroundSystemName] | None = None,
    ) -> JumpAroundReport:
        sc = scenario or default_jump_around_scenario()
        selected: list[JumpAroundSystemName] = systems or [
            "chatree_live",
            "linear",
            "sliding_window",
            "flat_rag",
        ]
        results: list[JumpAroundSystemResult] = []
        for sys_name in selected:
            results.append(await self._run_system(sc, sys_name))
        return JumpAroundReport(scenario_name=sc.name, model=self.model, results=results)

    async def _score_query(
        self,
        turn: JumpAroundTurn,
        answer_text: str,
        elapsed_ms: float,
        judge: LLMJudgeScorer | None,
    ) -> QueryEvalResult:
        qres = self.scorer.score(
            query=turn.user_message,
            target_branch=turn.expected_branch_hint or "root",
            answer_text=answer_text,
            expected_keywords=turn.expected_keywords,
            rejected_keywords=turn.rejected_keywords,
            stale_keywords=turn.stale_keywords,
            latency_ms=elapsed_ms,
        )
        if judge is not None:
            j_score, j_rat = await judge.judge(
                query=turn.user_message,
                answer_text=answer_text,
                expected_keywords=turn.expected_keywords,
                forbidden_keywords=[*turn.rejected_keywords, *turn.stale_keywords],
            )
            qres.judge_score = j_score
            qres.judge_rationale = j_rat
        return qres

    async def _run_system(
        self, scenario: JumpAroundScenario, system: JumpAroundSystemName
    ) -> JumpAroundSystemResult:
        llm = self._effective_llm()
        judge = LLMJudgeScorer(self.judge_llm or llm, self.judge_model) if self.judge_model else None

        if system == "chatree_live":
            return await self._run_chatree_live(scenario, llm, judge)
        if system == "linear":
            return await self._run_linear(scenario, llm, judge)
        if system == "sliding_window":
            return await self._run_sliding_window(scenario, llm, judge)
        if system == "flat_rag":
            return await self._run_flat_rag(scenario, llm, judge)
        if system == "rlm_official":
            return await self._run_rlm_official(scenario, llm, judge)
        raise ValueError(f"Unsupported jump-around system: {system}")

    async def _run_chatree_live(
        self,
        scenario: JumpAroundScenario,
        llm: LLMProvider,
        judge: LLMJudgeScorer | None,
    ) -> JumpAroundSystemResult:
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            shadow_model=self.shadow_model,
            auto_commit_handoffs=True,
            auto_background_summarize_on_switch=True,
            guardrails=GuardrailPolicy(
                preserve_kv_cache_tool_schemas=True,
                enable_compound_ask_tool=True,
                max_tool_rounds=6,
            ),
        )
        async with Chatree.in_memory(
            config=cfg, llm=llm, embedder=self.embedder, reranker=self.reranker
        ) as ct:
            tree = await ct.create_tree(
                "Nebula API Gateway",
                goal="Top-level architecture spec for Nebula API Gateway",
                default_model=self.model,
                shadow_model=self.shadow_model,
                auto_commit_handoffs=True,
            )
            query_results: list[QueryEvalResult] = []
            traces: list[JumpAroundTurnTrace] = []
            per_turn_ctx: list[int] = []
            latencies: list[float] = []
            nav_checks = 0
            nav_hits = 0
            peak_ctx = 0

            for turn in scenario.turns:
                m_before = await tree.metrics()
                t0 = time.perf_counter()
                res = await tree.chat(turn.user_message, include_core_tools=True)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)

                # Wait briefly for any non-blocking background summary triggered on branch switch
                await ct.wait_background_summaries(timeout=15.0)

                head_node = await (await tree.head()).model()
                head_ctx = await ct.context.estimate_node_context_tokens(head_node)
                per_turn_ctx.append(head_ctx)
                if head_ctx > peak_ctx:
                    peak_ctx = head_ctx

                m_after = await tree.metrics()
                nav_match: bool | None = None
                if turn.expected_branch_hint and turn.kind in ("branch", "jump_back", "dead_end"):
                    nav_checks += 1
                    hint_low = turn.expected_branch_hint.lower()
                    nav_match = hint_low in head_node.title.lower() or (
                        hint_low == "nebula" and head_node.parent_id is None
                    )
                    if nav_match:
                        nav_hits += 1

                tools_called = [ev.event_type for ev in res.ipc_events]
                traces.append(
                    JumpAroundTurnTrace(
                        turn_index=turn.turn_index,
                        kind=turn.kind,
                        head_node_id=head_node.id,
                        head_title=head_node.title,
                        navigation_matched=nav_match,
                        tokens_in=max(0, m_after.total_tokens_in - m_before.total_tokens_in),
                        tokens_out=max(0, m_after.total_tokens_out - m_before.total_tokens_out),
                        cache_read_tokens=max(
                            0, m_after.total_cache_read_tokens - m_before.total_cache_read_tokens
                        ),
                        latency_ms=round(elapsed_ms, 2),
                        tools_called=[str(t) for t in tools_called],
                        answer_preview=res.content[:220],
                    )
                )

                if turn.expected_keywords:
                    qres = await self._score_query(turn, res.content, elapsed_ms, judge)
                    query_results.append(qres)

            final_metrics = await tree.metrics()
            nodes = await tree.nodes()
            bg_summaries = sum(1 for n in nodes if n.parent_id is not None and bool(n.summary))
            cost_usd = self.cost_model.estimate_usd(
                self.model,
                tokens_in=final_metrics.total_tokens_in,
                tokens_out=final_metrics.total_tokens_out,
                cache_read_tokens=final_metrics.total_cache_read_tokens,
            )
            return self._build_result(
                system="chatree_live",
                scenario=scenario,
                query_results=query_results,
                traces=traces,
                latencies=latencies,
                per_turn_ctx=per_turn_ctx,
                peak_ctx=peak_ctx,
                total_in=final_metrics.total_tokens_in,
                total_out=final_metrics.total_tokens_out,
                total_cache=final_metrics.total_cache_read_tokens,
                cost_usd=cost_usd,
                nav_acc=round(nav_hits / nav_checks, 4) if nav_checks else None,
                branches_created=max(0, final_metrics.node_count - 1),
                reflog_jumps=final_metrics.reflog_jumps,
                shadow_turns=final_metrics.shadow_turns,
                shadow_cache_hits=final_metrics.shadow_cache_hits,
                bg_summaries=bg_summaries,
            )

    async def _run_linear(
        self,
        scenario: JumpAroundScenario,
        llm: LLMProvider,
        judge: LLMJudgeScorer | None,
    ) -> JumpAroundSystemResult:
        sys_prompt = (
            "You are a concise, accurate architecture assistant helping design Nebula API Gateway. "
            "Track latest settled decisions, respect explicitly abandoned prototypes, and answer "
            "directly without tools."
        )
        history: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
        query_results: list[QueryEvalResult] = []
        traces: list[JumpAroundTurnTrace] = []
        per_turn_ctx: list[int] = []
        latencies: list[float] = []
        total_in = total_out = total_cache = peak_ctx = 0

        for turn in scenario.turns:
            history.append({"role": "user", "content": turn.user_message})
            t0 = time.perf_counter()
            resp = await llm.complete(model=self.model, messages=history, tools=None, temperature=0.1)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            history.append({"role": "assistant", "content": resp.content})

            ctx_tok = resp.tokens_in or estimate_messages_tokens(history)
            per_turn_ctx.append(ctx_tok)
            peak_ctx = max(peak_ctx, ctx_tok)
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            total_cache += resp.cache_read_tokens

            traces.append(
                JumpAroundTurnTrace(
                    turn_index=turn.turn_index,
                    kind=turn.kind,
                    head_node_id="linear",
                    head_title="Linear Thread",
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    cache_read_tokens=resp.cache_read_tokens,
                    latency_ms=round(elapsed_ms, 2),
                    answer_preview=resp.content[:220],
                )
            )
            if turn.expected_keywords:
                query_results.append(await self._score_query(turn, resp.content, elapsed_ms, judge))

        cost_usd = self.cost_model.estimate_usd(
            self.model, tokens_in=total_in, tokens_out=total_out, cache_read_tokens=total_cache
        )
        return self._build_result(
            system="linear",
            scenario=scenario,
            query_results=query_results,
            traces=traces,
            latencies=latencies,
            per_turn_ctx=per_turn_ctx,
            peak_ctx=peak_ctx,
            total_in=total_in,
            total_out=total_out,
            total_cache=total_cache,
            cost_usd=cost_usd,
        )

    async def _run_sliding_window(
        self,
        scenario: JumpAroundScenario,
        llm: LLMProvider,
        judge: LLMJudgeScorer | None,
    ) -> JumpAroundSystemResult:
        sys_prompt = (
            "You are a concise architecture assistant operating with a rolling summary plus the "
            "most recent conversation turns. Answer directly based on visible context."
        )
        all_msgs: list[dict[str, str]] = []
        rolling_summary = ""
        query_results: list[QueryEvalResult] = []
        traces: list[JumpAroundTurnTrace] = []
        per_turn_ctx: list[int] = []
        latencies: list[float] = []
        total_in = total_out = total_cache = peak_ctx = 0

        for turn in scenario.turns:
            all_msgs.append({"role": "user", "content": turn.user_message})
            if len(all_msgs) > self.sliding_window_messages:
                evicted = all_msgs[: -self.sliding_window_messages]
                rolling_summary = " | ".join(m["content"][:90] for m in evicted[-4:])
            window = all_msgs[-self.sliding_window_messages :]
            msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
            if rolling_summary:
                msgs.append(
                    {
                        "role": "user",
                        "content": f"[Sliding Window Context]\nEarlier summary: {rolling_summary}",
                    }
                )
            msgs.extend(window)

            t0 = time.perf_counter()
            resp = await llm.complete(model=self.model, messages=msgs, tools=None, temperature=0.1)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            all_msgs.append({"role": "assistant", "content": resp.content})

            ctx_tok = resp.tokens_in or estimate_messages_tokens(msgs)
            per_turn_ctx.append(ctx_tok)
            peak_ctx = max(peak_ctx, ctx_tok)
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            total_cache += resp.cache_read_tokens

            traces.append(
                JumpAroundTurnTrace(
                    turn_index=turn.turn_index,
                    kind=turn.kind,
                    head_node_id="sliding_window",
                    head_title="Sliding Window",
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    cache_read_tokens=resp.cache_read_tokens,
                    latency_ms=round(elapsed_ms, 2),
                    answer_preview=resp.content[:220],
                )
            )
            if turn.expected_keywords:
                query_results.append(await self._score_query(turn, resp.content, elapsed_ms, judge))

        cost_usd = self.cost_model.estimate_usd(
            self.model, tokens_in=total_in, tokens_out=total_out, cache_read_tokens=total_cache
        )
        return self._build_result(
            system="sliding_window",
            scenario=scenario,
            query_results=query_results,
            traces=traces,
            latencies=latencies,
            per_turn_ctx=per_turn_ctx,
            peak_ctx=peak_ctx,
            total_in=total_in,
            total_out=total_out,
            total_cache=total_cache,
            cost_usd=cost_usd,
        )

    async def _run_flat_rag(
        self,
        scenario: JumpAroundScenario,
        llm: LLMProvider,
        judge: LLMJudgeScorer | None,
    ) -> JumpAroundSystemResult:
        cfg = ChatreeConfig(
            db_path=":memory:",
            default_model=self.model,
            ablations=AblationConfig(enable_subtree_rollups=False, enable_shadow_turns=False),
        )
        async with Chatree.in_memory(
            config=cfg, llm=llm, embedder=self.embedder, reranker=self.reranker
        ) as ct:
            tree = await ct.create_tree("Nebula Flat RAG", default_model=self.model)
            root = await tree.root()
            query_results: list[QueryEvalResult] = []
            traces: list[JumpAroundTurnTrace] = []
            per_turn_ctx: list[int] = []
            latencies: list[float] = []
            total_in = total_out = total_cache = peak_ctx = 0

            for turn in scenario.turns:
                t0 = time.perf_counter()
                await ct.repo.append_message(
                    root.id, tree.id, "user", f"[Turn {turn.turn_index}] {turn.user_message}"
                )
                await ct.rag.index_node_chunks(root.id)
                hits = await tree.search(
                    turn.user_message, scope="all", top_k=5, granularities={"chunk"}
                )
                chunks_txt = "\n".join(f"- {h.snippet}" for h in hits)
                msgs: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": (
                            "Answer concisely using only the retrieved conversation snippets. "
                            "Prefer latest updates and exclude scrapped prototypes."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"[RAG Context]\n{chunks_txt}\n\nQuestion: {turn.user_message}",
                    },
                ]
                resp = await llm.complete(model=self.model, messages=msgs, tools=None, temperature=0.1)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                await ct.repo.append_message(
                    root.id, tree.id, "assistant", f"[Turn {turn.turn_index}] {resp.content}"
                )

                ctx_tok = resp.tokens_in or estimate_messages_tokens(msgs)
                per_turn_ctx.append(ctx_tok)
                peak_ctx = max(peak_ctx, ctx_tok)
                total_in += resp.tokens_in
                total_out += resp.tokens_out
                total_cache += resp.cache_read_tokens

                traces.append(
                    JumpAroundTurnTrace(
                        turn_index=turn.turn_index,
                        kind=turn.kind,
                        head_node_id="flat_rag",
                        head_title="Flat Hybrid RAG",
                        tokens_in=resp.tokens_in,
                        tokens_out=resp.tokens_out,
                        cache_read_tokens=resp.cache_read_tokens,
                        latency_ms=round(elapsed_ms, 2),
                        answer_preview=resp.content[:220],
                    )
                )
                if turn.expected_keywords:
                    query_results.append(
                        await self._score_query(turn, resp.content, elapsed_ms, judge)
                    )

            cost_usd = self.cost_model.estimate_usd(
                self.model, tokens_in=total_in, tokens_out=total_out, cache_read_tokens=total_cache
            )
            return self._build_result(
                system="flat_rag",
                scenario=scenario,
                query_results=query_results,
                traces=traces,
                latencies=latencies,
                per_turn_ctx=per_turn_ctx,
                peak_ctx=peak_ctx,
                total_in=total_in,
                total_out=total_out,
                total_cache=total_cache,
                cost_usd=cost_usd,
            )

    async def _run_rlm_official(
        self,
        scenario: JumpAroundScenario,
        llm: LLMProvider,
        judge: LLMJudgeScorer | None,
    ) -> JumpAroundSystemResult:
        from chatree.eval.adapters import RLMOfficialAdapter

        adapter = RLMOfficialAdapter(model=self.model, max_iterations=8)
        await adapter.reset(scenario.name)
        await adapter.start_session("live", title="Live Session")
        query_results: list[QueryEvalResult] = []
        traces: list[JumpAroundTurnTrace] = []
        per_turn_ctx: list[int] = []
        latencies: list[float] = []
        total_in = total_out = total_cache = peak_ctx = 0

        try:
            for turn in scenario.turns:
                t0 = time.perf_counter()
                if turn.expected_keywords:
                    ans = await adapter.answer_query(turn.user_message)
                    elapsed_ms = ans.latency_ms or (time.perf_counter() - t0) * 1000.0
                    reply = ans.answer
                    t_in, t_out, t_cache, p_ctx = (
                        ans.tokens_in,
                        ans.tokens_out,
                        ans.cache_read_tokens,
                        ans.peak_context_tokens,
                    )
                else:
                    await adapter.ingest_turn("live", "user", turn.user_message)
                    # Lightweight acknowledgement turn on non-query steps
                    resp = await llm.complete(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "Acknowledge the user's update in one brief sentence."},
                            {"role": "user", "content": turn.user_message},
                        ],
                        tools=None,
                        temperature=0.0,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    reply = resp.content
                    t_in, t_out, t_cache, p_ctx = (
                        resp.tokens_in,
                        resp.tokens_out,
                        resp.cache_read_tokens,
                        resp.tokens_in,
                    )
                    await adapter.ingest_turn("live", "assistant", reply)

                latencies.append(elapsed_ms)
                per_turn_ctx.append(p_ctx)
                peak_ctx = max(peak_ctx, p_ctx)
                total_in += t_in
                total_out += t_out
                total_cache += t_cache

                traces.append(
                    JumpAroundTurnTrace(
                        turn_index=turn.turn_index,
                        kind=turn.kind,
                        head_node_id="rlm_official",
                        head_title="RLM Official",
                        tokens_in=t_in,
                        tokens_out=t_out,
                        cache_read_tokens=t_cache,
                        latency_ms=round(elapsed_ms, 2),
                        answer_preview=reply[:220],
                    )
                )
                if turn.expected_keywords:
                    query_results.append(await self._score_query(turn, reply, elapsed_ms, judge))
        finally:
            await adapter.close()

        cost_usd = self.cost_model.estimate_usd(
            self.model, tokens_in=total_in, tokens_out=total_out, cache_read_tokens=total_cache
        )
        return self._build_result(
            system="rlm_official",
            scenario=scenario,
            query_results=query_results,
            traces=traces,
            latencies=latencies,
            per_turn_ctx=per_turn_ctx,
            peak_ctx=peak_ctx,
            total_in=total_in,
            total_out=total_out,
            total_cache=total_cache,
            cost_usd=cost_usd,
        )

    @staticmethod
    def _build_result(
        *,
        system: JumpAroundSystemName,
        scenario: JumpAroundScenario,
        query_results: list[QueryEvalResult],
        traces: list[JumpAroundTurnTrace],
        latencies: list[float],
        per_turn_ctx: list[int],
        peak_ctx: int,
        total_in: int,
        total_out: int,
        total_cache: int,
        cost_usd: float,
        nav_acc: float | None = None,
        branches_created: int = 0,
        reflog_jumps: int = 0,
        shadow_turns: int = 0,
        shadow_cache_hits: int = 0,
        bg_summaries: int = 0,
    ) -> JumpAroundSystemResult:
        q_total = len(query_results)
        q_passed = sum(1 for q in query_results if q.passed)
        mean_rec = (
            round(sum(q.recall_score for q in query_results) / q_total, 4) if q_total else 0.0
        )
        leak_rate = (
            round(sum(1 for q in query_results if q.rejected_leakage) / q_total, 4)
            if q_total
            else 0.0
        )
        stale_rate = (
            round(sum(1 for q in query_results if q.stale_contradiction) / q_total, 4)
            if q_total
            else 0.0
        )
        judged = [q.judge_score for q in query_results if q.judge_score is not None]
        avg_judge = round(sum(judged) / len(judged), 4) if judged else None
        return JumpAroundSystemResult(
            system=system,
            scenario_name=scenario.name,
            turns_total=len(scenario.turns),
            queries_passed=q_passed,
            queries_total=q_total,
            strict_pass_rate=round(q_passed / q_total, 4) if q_total else 0.0,
            mean_recall=mean_rec,
            dead_end_leakage_rate=leak_rate,
            stale_revision_error_rate=stale_rate,
            avg_judge_score=avg_judge,
            navigation_accuracy=nav_acc,
            branches_created=branches_created,
            reflog_jumps=reflog_jumps,
            shadow_turns=shadow_turns,
            shadow_cache_hits=shadow_cache_hits,
            bg_summaries_completed=bg_summaries,
            peak_context_tokens=peak_ctx,
            total_tokens_in=total_in,
            total_tokens_out=total_out,
            total_cache_read_tokens=total_cache,
            cache_hit_ratio=round(total_cache / total_in, 4) if total_in > 0 else 0.0,
            estimated_cost_usd=cost_usd,
            turn_latency_p50_ms=_percentile(latencies, 0.50),
            turn_latency_p95_ms=_percentile(latencies, 0.95),
            per_turn_context_tokens=per_turn_ctx,
            query_results=query_results,
            turn_traces=traces,
        )
