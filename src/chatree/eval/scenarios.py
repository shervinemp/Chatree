from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TaskFamily = Literal[
    "distractor_scaling",
    "proposal_vs_decision",
    "multihop_revision",
    "unscripted_agentic",
    "crossover_scaling",
]


class ScenarioStep(BaseModel):
    """One declarative step in a reproducible multi-tangent evaluation scenario."""

    kind: Literal["chat", "spawn", "finish", "revisit", "request_sync", "verify_query", "unscripted_turn"]
    branch: str = "root"
    parent: str = "root"
    title: str | None = None
    goal: str = ""
    context_mode: Literal["fork", "slice", "clean"] = "clean"
    content: str = ""
    target_branch: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    rejected_keywords: list[str] = Field(default_factory=list)
    stale_keywords: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    # Semantically equivalent tool alternatives (e.g. [["search_tree", "ask_tree", "request"]])
    expected_tools_any_of: list[list[str]] = Field(default_factory=list)
    is_post_revision: bool = False


class MultiTangentScenario(BaseModel):
    name: str
    family: TaskFamily = "proposal_vs_decision"
    execution_mode: Literal["scripted", "unscripted"] = "scripted"
    description: str = ""
    metadata: dict[str, int | str | float] = Field(default_factory=dict)
    steps: list[ScenarioStep]


def distractor_scaling_scenario(num_distractors: int = 4) -> MultiTangentScenario:
    """
    Task Family 1 — Distractor & Dead-End Scaling ("Logical Context Poisoning"):
    Creates `num_distractors` side branches containing rejected alternatives + noise,
    plus target branches with settled constraints, then probes exact decisions and
    measures rejected-proposal leakage as K grows.
    """
    k = max(2, num_distractors)
    steps: list[ScenarioStep] = [
        ScenarioStep(
            kind="chat",
            branch="root",
            content=f"We are designing a distributed platform across {k} subsystems. Keep final decisions clean.",
        )
    ]

    for i in range(1, k + 1):
        bname = f"subsys_{i}"
        rejected = f"REJECTED_ARCH_OPT_{i}"
        settled = f"SETTLED_SPEC_V{i}_99"
        steps.extend(
            [
                ScenarioStep(
                    kind="spawn",
                    branch=bname,
                    parent="root",
                    title=f"Subsystem {i}",
                    goal=f"Select architecture for subsystem {i}",
                    context_mode="clean",
                ),
                ScenarioStep(
                    kind="chat",
                    branch=bname,
                    content=(
                        f"Initial proposal: consider {rejected} for subsystem {i} throughput handling "
                        f"with naive synchronous polling and unindexed blobs."
                    ),
                ),
                ScenarioStep(
                    kind="chat",
                    branch=bname,
                    content=(
                        f"Rejected {rejected} due to tail latency spikes. "
                        f"Final decision: subsystem {i} adopts {settled} with bounded backpressure."
                    ),
                ),
                ScenarioStep(
                    kind="finish",
                    branch=bname,
                    content=f"Subsystem {i} architecture finalized for production.",
                ),
            ]
        )

    # Probe the first and last subsystem to test long-range recall without distractor leakage
    for idx in (1, k):
        steps.append(
            ScenarioStep(
                kind="verify_query",
                branch="root",
                target_branch=f"subsys_{idx}",
                content=f"What final architecture spec did Subsystem {idx} settle on?",
                expected_keywords=[f"SETTLED_SPEC_V{idx}_99"],
                rejected_keywords=[f"REJECTED_ARCH_OPT_{idx}"],
            )
        )

    return MultiTangentScenario(
        name=f"distractor-scaling-k{k}",
        family="distractor_scaling",
        execution_mode="scripted",
        description=f"Scales to K={k} tangent branches with rejected proposals to measure context poisoning and leakage.",
        metadata={"num_distractors": k},
        steps=steps,
    )


def proposal_vs_decision_scenario() -> MultiTangentScenario:
    """
    Task Family 2 — Lossy-Summary Detail Recovery & Proposal-vs-Decision Trap:
    Each branch explicitly debates an attractive keyword-rich proposal, rejects it,
    adopts a specific final parameter, and returns a realistic concise (lossy) summary.
    Tests Retrieve-then-Ask shadow turns vs. flat chunk RAG and summary-only folding.
    """
    return MultiTangentScenario(
        name="proposal-vs-decision-trap",
        family="proposal_vs_decision",
        execution_mode="scripted",
        description=(
            "Branches debate and reject an initial proposal before settling on an exact parameter "
            "omitted from the prose summary; isolates shadow turns vs. flat RAG and one-way folding."
        ),
        steps=[
            ScenarioStep(
                kind="chat",
                branch="root",
                content="Let's finalize vector indexing, auth signing, and streaming transport.",
            ),
            # Branch 1: Vector index
            ScenarioStep(
                kind="spawn",
                branch="indexing",
                parent="root",
                title="Vector Indexing",
                goal="Choose SQLite vector index parameters",
                context_mode="clean",
            ),
            ScenarioStep(
                kind="chat",
                branch="indexing",
                content="Initial proposal: use BRUTE_FORCE_EXACT_L2 scan for vector indexing.",
            ),
            ScenarioStep(
                kind="chat",
                branch="indexing",
                content=(
                    "Benchmarks showed brute-force scans fail past 50k chunks. "
                    "Final decision: use HNSW_PQ_M16_EF64 cosine index in sqlite-vec."
                ),
            ),
            ScenarioStep(
                kind="finish",
                branch="indexing",
                content="Vector indexing uses local SQLite storage.",  # lossy prose summary!
            ),
            # Branch 2: Auth signing
            ScenarioStep(
                kind="spawn",
                branch="auth",
                parent="root",
                title="Auth Signing",
                goal="Choose JWT signing algorithm and rotation window",
                context_mode="clean",
            ),
            ScenarioStep(
                kind="chat",
                branch="auth",
                content="Initial proposal: use HS256_SHARED_SECRET with no key rotation.",
            ),
            ScenarioStep(
                kind="chat",
                branch="auth",
                content=(
                    "Security review rejected shared symmetric secrets. "
                    "Final decision: use ED25519_JWKS_ROTATE_24H asymmetric signing."
                ),
            ),
            ScenarioStep(
                kind="finish",
                branch="auth",
                content="Authentication uses stateless signed tokens.",  # lossy prose summary!
            ),
            # Verification queries probing fine-grained settled decisions AND checking rejected leakage
            ScenarioStep(
                kind="verify_query",
                branch="root",
                target_branch="indexing",
                content="What exact vector indexing configuration did Vector Indexing settle on?",
                expected_keywords=["HNSW_PQ_M16_EF64"],
                rejected_keywords=["BRUTE_FORCE_EXACT_L2"],
            ),
            ScenarioStep(
                kind="verify_query",
                branch="root",
                target_branch="auth",
                content="What exact JWT signing scheme did Auth Signing adopt?",
                expected_keywords=["ED25519_JWKS_ROTATE_24H"],
                rejected_keywords=["HS256_SHARED_SECRET"],
            ),
        ],
    )


def multihop_revision_scenario() -> MultiTangentScenario:
    """
    Task Family 3 — Mid-Stream Branch Revision & Multi-Hop Transitive Staleness:
    Root (A) <- Gateway (B) <- Auth (C).
    C closes with V1_COOKIE_SESSION -> B reads C and closes -> A reads B.
    Then user reflog-jumps back into C and revises to V2_OAUTH2_PKCE_ED25519.
    Verifies both direct (B) and transitive (A) staleness detection and stale-contradiction avoidance.
    """
    return MultiTangentScenario(
        name="multihop-revision-staleness",
        family="multihop_revision",
        execution_mode="scripted",
        description=(
            "2-hop dependency chain A <- B <- C with mid-stream reflog revision of C; "
            "tests transitive staleness detection and stale-contradiction prevention."
        ),
        steps=[
            ScenarioStep(
                kind="chat",
                branch="root",
                content="Coordinate API gateway and auth architecture.",
            ),
            ScenarioStep(
                kind="spawn",
                branch="gateway",
                parent="root",
                title="API Gateway",
                goal="Design edge middleware depending on auth spec",
                context_mode="clean",
            ),
            ScenarioStep(
                kind="spawn",
                branch="auth",
                parent="gateway",
                title="Auth Core",
                goal="Select core session/token mechanism",
                context_mode="clean",
            ),
            ScenarioStep(
                kind="chat",
                branch="auth",
                content="Initial plan: V1_COOKIE_SESSION with server-side Redis sessions.",
            ),
            ScenarioStep(
                kind="finish",
                branch="auth",
                content="Auth Core v1 decision: V1_COOKIE_SESSION with Redis store.",
            ),
            # Gateway integrates Auth v1 and finishes back to Root -> creates A <- B <- C chain!
            ScenarioStep(
                kind="chat",
                branch="gateway",
                content="Gateway middleware validates V1_COOKIE_SESSION headers from Auth Core.",
            ),
            ScenarioStep(
                kind="finish",
                branch="gateway",
                content="Gateway integrates Auth Core v1 (V1_COOKIE_SESSION).",
            ),
            # Mid-stream reflog jump back into leaf branch `auth` to revise the decision!
            ScenarioStep(
                kind="revisit",
                branch="auth",
                content="Revision: drop cookies entirely and adopt V2_OAUTH2_PKCE_ED25519 bearer tokens.",
            ),
            # Verify root (2 hops upstream!) reflects V2 and does NOT assert stale V1!
            ScenarioStep(
                kind="verify_query",
                branch="root",
                target_branch="auth",
                content="What is our current active Auth Core mechanism after recent revisions?",
                expected_keywords=["V2_OAUTH2_PKCE_ED25519"],
                stale_keywords=["V1_COOKIE_SESSION"],
                is_post_revision=True,
            ),
        ],
    )


def unscripted_agentic_scenario() -> MultiTangentScenario:
    """
    Task Family 4 — Unscripted Autonomous Tool-Use Policy Evaluation:
    Sends natural prompts directly to HEAD and evaluates whether the model invokes
    appropriate tree IPC tools (`spawn_child`, `search_tree`, `request`, `send`)
    without runaway loops or guardrail violations.
    """
    return MultiTangentScenario(
        name="unscripted-agentic-policy",
        family="unscripted_agentic",
        execution_mode="unscripted",
        description=(
            "Tests autonomous model decisions on when to spawn branches, search the tree, "
            "and query dormant nodes with read-only shadow turns."
        ),
        steps=[
            ScenarioStep(
                kind="unscripted_turn",
                branch="root",
                content="Please branch off to design the rate limiter policy",
                expected_tools=["dialogue_tree_create_branch"],
            ),
            ScenarioStep(
                kind="unscripted_turn",
                branch="head",
                content="Final decision: rate limiter uses TOKEN_BUCKET_100RPS_BURST_20 per tenant.",
                expected_tools=[],
            ),
            ScenarioStep(
                kind="unscripted_turn",
                branch="head",
                content="wrap up and return to parent with the rate limiter summary",
                expected_tools=["dialogue_tree_send_to_branch"],
            ),
            ScenarioStep(
                kind="unscripted_turn",
                branch="root",
                content="search tree for TOKEN_BUCKET_100RPS_BURST_20",
                expected_tools_any_of=[
                    ["dialogue_tree_search_branches", "dialogue_tree_search_and_query", "dialogue_tree_query_branch"]
                ],
                expected_keywords=["TOKEN_BUCKET_100RPS_BURST_20"],
            ),
        ],
    )


def crossover_scaling_scenario(
    num_branches: int = 4, filler_turns_per_branch: int = 3
) -> MultiTangentScenario:
    """
    Task Family 5 — Cost, Latency & KV-Cache Crossover Benchmark:
    Simulates a multi-turn engineering session with `num_branches` isolated deep-dives
    each taking `filler_turns_per_branch` turns, measuring peak root context savings,
    cumulative token/USD economics, cache-hit rate, and p50/p95 query latency.
    """
    steps: list[ScenarioStep] = [
        ScenarioStep(
            kind="chat",
            branch="root",
            content="Session kickoff: architecting multi-tenant analytics pipeline.",
        )
    ]
    for b in range(1, num_branches + 1):
        bname = f"topic_{b}"
        steps.append(
            ScenarioStep(
                kind="spawn",
                branch=bname,
                parent="root",
                title=f"Topic {b}",
                goal=f"Deep dive into component {b}",
                context_mode="fork",
            )
        )
        for t in range(1, filler_turns_per_branch):
            steps.append(
                ScenarioStep(
                    kind="chat",
                    branch=bname,
                    content=(
                        f"Component {b} exploration turn {t}: evaluating trade-offs across throughput, "
                        f"memory overhead, checkpoint intervals, retry backoff jitter, and schema evolution."
                    ),
                )
            )
        steps.append(
            ScenarioStep(
                kind="chat",
                branch=bname,
                content=f"Final decision: component {b} locks in CONFIG_TOKEN_C{b}_OK.",
            )
        )
        steps.append(
            ScenarioStep(
                kind="finish",
                branch=bname,
                content=f"Component {b} design finalized.",
            )
        )

    for b in (1, num_branches):
        steps.append(
            ScenarioStep(
                kind="verify_query",
                branch="root",
                target_branch=f"topic_{b}",
                content=f"Confirm the final config token for Topic {b}.",
                expected_keywords=[f"CONFIG_TOKEN_C{b}_OK"],
            )
        )

    return MultiTangentScenario(
        name=f"crossover-b{num_branches}-t{filler_turns_per_branch}",
        family="crossover_scaling",
        execution_mode="scripted",
        description=(
            f"Multi-turn crossover scaling ({num_branches} forked branches x {filler_turns_per_branch} turns) "
            "measuring peak context reduction, USD cost with KV cache discounts, and p50/p95 latency."
        ),
        metadata={"num_branches": num_branches, "filler_turns_per_branch": filler_turns_per_branch},
        steps=steps,
    )


def default_multi_tangent_scenario() -> MultiTangentScenario:
    """Default combined scenario covering lossy-summary shadow retrieval + mid-stream revision."""
    return MultiTangentScenario(
        name="multi-tangent-architecture-revision",
        family="proposal_vs_decision",
        execution_mode="scripted",
        description=(
            "Evaluates cross-branch fine-grained detail retrieval and post-reflog branch "
            "revision coherence across linear, sliding window, flat-RAG, summary-only, "
            "no-staleness, and full Chatree."
        ),
        steps=[
            ScenarioStep(kind="chat", branch="root", content="We are architecting our local-first AI platform."),
            ScenarioStep(
                kind="spawn",
                branch="db",
                parent="root",
                title="DB Storage",
                goal="Select database engine and indexing parameters",
                context_mode="clean",
            ),
            ScenarioStep(
                kind="chat",
                branch="db",
                content="Final decision: SQLite WAL with HNSW_EF_64 and materialized path /root/n1/ indexing.",
            ),
            ScenarioStep(
                kind="finish",
                branch="db",
                content="Database uses SQLite WAL for local storage.",
            ),
            ScenarioStep(
                kind="spawn",
                branch="auth",
                parent="root",
                title="Auth Tokens",
                goal="Choose signing algorithm and TTL",
                context_mode="clean",
            ),
            ScenarioStep(
                kind="chat",
                branch="auth",
                content="Initial plan: HS256_LEGACY signing with 30-day TTL.",
            ),
            ScenarioStep(
                kind="finish",
                branch="auth",
                content="Auth uses HS256_LEGACY signing with 30-day TTL.",
            ),
            ScenarioStep(
                kind="verify_query",
                branch="root",
                target_branch="db",
                content="What exact HNSW parameter did we pick in DB Storage?",
                expected_keywords=["HNSW_EF_64"],
            ),
            ScenarioStep(
                kind="revisit",
                branch="auth",
                content="Revision: replace HS256 with ED25519_ROTATING_24H keys.",
            ),
            ScenarioStep(
                kind="verify_query",
                branch="root",
                target_branch="auth",
                content="What is our current revised signing key policy in Auth Tokens?",
                expected_keywords=["ED25519_ROTATING_24H"],
                stale_keywords=["HS256_LEGACY"],
                is_post_revision=True,
            ),
        ],
    )


def build_full_eval_suite(
    *,
    num_distractors: int = 4,
    crossover_branches: int = 4,
    crossover_turns: int = 3,
) -> list[MultiTangentScenario]:
    return [
        distractor_scaling_scenario(num_distractors=num_distractors),
        proposal_vs_decision_scenario(),
        multihop_revision_scenario(),
        unscripted_agentic_scenario(),
        crossover_scaling_scenario(
            num_branches=crossover_branches,
            filler_turns_per_branch=crossover_turns,
        ),
    ]
