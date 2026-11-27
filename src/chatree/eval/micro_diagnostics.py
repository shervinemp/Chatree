"""
Architectural & Micro-Diagnostic Proof Suite (Track 8) for Chatree vs. Distilled Flat Memory (Mem0).

Provides two strictly separated, scientifically transparent proof layers:
  1. **Part A — Zero-Token Deterministic Simulation & Real-Framework Execution Trace (`$0.00`, 0 API tokens):**
     - Sim 1 (Byte-0 Prefix KV-Cache Stability): Measures exact Longest Common Token Prefix (LCP)
       across 16 turns through real `Chatree` (`ContextAssembler`) vs. Mem0-style prefix/header injection.
     - Sim 2 (Write-Time vs. Query-Time Scaling): Traces exact LLM call counts, prompt-schema overhead
       (measuring upstream `mem0ai` `ADDITIVE_EXTRACTION_PROMPT` directly when installed), and USD curves
       across N in {5, 20, 50, 200} sessions, cross-referenced against empirical LongMemEval-S runs.
     - Sim 3 (Step-by-Step Framework Execution Traces on Probes 1-3): Executes real `Chatree` alongside
       real unmodified `mem0ai` (routed through an in-process localhost zero-token OpenAI-compatible HTTP
       proxy when `mem0ai` is installed, with a labeled deterministic fallback if absent), giving both
       frameworks IDENTICAL lossy summary/fact distillation to isolate pure architectural guarantees:
         * Probe 1: Un-Anchored Verbatim Detail Recovery (lossless SQLite leaves + shadow turn)
         * Probe 2: Transitive Downstream Staleness (`A -> B -> C` causal DAG invalidation)
         * Probe 3: Silent Dead-End Quarantine (native `archive()` with zero cancellation essay)
  2. **Part B — Ultra-Lean Live Head-to-Head Micro-Diagnostic (`run_live_micro_diagnostics`):**
     - Runs the EXACT same 3 surgical probes (~$0.02-$0.04 off-peak total) against real `Chatree`
       and real unmodified `Mem0OfficialAdapter` (`mem0ai`) on any configured LLM provider, recording
       raw answers, exact stored memories/tree states, tokens, USD, KV-cache %, and pass/fail rubrics.
"""

from __future__ import annotations

import asyncio
import csv
import http.server
import importlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field

from chatree.config import ChatreeConfig, GuardrailPolicy
from chatree.eval.adapters.chatree import ChatreeSystemAdapter
from chatree.eval.adapters.sota import Mem0OfficialAdapter
from chatree.eval.scoring import CostModel, is_token_asserted_as_active
from chatree.protocols import Embedder, LLMProvider, Reranker
from chatree.providers.embeddings import HashEmbedder
from chatree.providers.mock import estimate_messages_tokens, estimate_tokens
from chatree.sdk.client import Chatree
from chatree.tools import STATIC_SYSTEM_PROMPT
from chatree.types import LLMResponse, LLMStreamEvent, LLMToolCall

# Documented byte/char lengths verified directly on upstream `mem0ai==2.2.1`
DEFAULT_MEM0_EXTRACTION_PROMPT_CHARS = 33_653
DEFAULT_MEM0_UPDATE_PROMPT_CHARS = 5_310


# =============================================================================
# Shared Probe Scenario Definitions (used identically by Part A & Part B)
# =============================================================================


class ProbeTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    date_time: str = ""


class ProbeSession(BaseModel):
    session_id: str
    title: str
    date_time: str = ""
    turns: list[ProbeTurn] = Field(default_factory=list)
    summary: str | None = None
    archived: bool = False
    archive_reason: str = ""
    imports: list[str] = Field(default_factory=list)
    is_revision_of: str | None = None


class ProbeQuery(BaseModel):
    query_id: str
    probe_id: str
    question: str
    expected_any_groups: list[list[str]] = Field(default_factory=list)
    forbidden_active_tokens: list[str] = Field(default_factory=list)
    require_warning_or_abstention: bool = False
    target_session_id: str | None = None
    rationale: str = ""


class MicroDiagnosticProbeSpec(BaseModel):
    probe_id: str
    title: str
    architectural_claim: str
    sessions: list[ProbeSession]
    queries: list[ProbeQuery]


def build_probe1_verbatim_recovery() -> MicroDiagnosticProbeSpec:
    """
    Probe 1: Un-Anchored Verbatim Detail Recovery.
    12-turn gateway incident session mixing 3 headline architecture decisions with 4 incidental
    verbatim literals embedded in configs/headers/traces/SQL plans, plus 2 distractor sessions.
    """
    incident_turns = [
        ProbeTurn(
            role="user",
            date_time="2026-04-02 09:10",
            content=(
                "We hit a p99 checkout latency spike on the payment gateway overnight. "
                "Let's walk through the ingress config, response traces, shard logs, and DB plan "
                "to settle our production stack changes."
            ),
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-02 09:11",
            content=(
                "Let's start with the edge proxy and ingress timeouts. Paste the current nginx/Envoy "
                "upstream block and let me know which proxy stack we're standardizing on."
            ),
        ),
        ProbeTurn(
            role="user",
            date_time="2026-04-02 09:14",
            content=(
                "Here is the legacy nginx ingress snippet from staging before we migrate:\n"
                "```nginx\n"
                "location /v2/charge {\n"
                "    proxy_connect_timeout 5s;\n"
                "    proxy_read_timeout 43s;\n"
                "    proxy_send_timeout 15s;\n"
                "}\n"
                "```\n"
                "Going forward, our headline architectural decision #1 is migrating the edge gateway "
                "to Envoy 1.31 across all regions."
            ),
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-02 09:15",
            content=(
                "Got it — standardizing on Envoy 1.31 for the edge gateway. "
                "Now paste a failing downstream response header capture so we can check retries."
            ),
        ),
        ProbeTurn(
            role="user",
            date_time="2026-04-02 09:18",
            content=(
                "Captured headers from a throttled checkout request:\n"
                "HTTP/2 503\n"
                "X-Region: us-east-1b\n"
                "X-Retry-Epoch: 17\n"
                "X-Upstream-Zone: pay-core-a\n"
                "Also, headline architectural decision #2: all internal service-to-service hops "
                "will switch from REST/JSON to gRPC mTLS."
            ),
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-02 09:19",
            content=(
                "Understood — gRPC mTLS for internal service hops. "
                "What showed up in the coordinator worker logs during the failover window?"
            ),
        ),
        ProbeTurn(
            role="user",
            date_time="2026-04-02 09:22",
            content=(
                "Coordinator worker stack excerpt:\n"
                "[WARN] lease_mgr.rs:412 renew_failed node=pay-04 code=ERR_SHARD_LEASE_419 elapsed=912ms\n"
                "[INFO] fallback_router triggered drain on partition 6\n"
                "To prevent repeat state contention, headline architectural decision #3: replace the "
                "legacy Redis 6 cluster with Valkey 8.0 for rate-limit and idempotency keys."
            ),
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-02 09:23",
            content=(
                "Locked in Valkey 8.0 for rate-limiting and idempotency. "
                "Lastly, paste the slow-query EXPLAIN ANALYZE output from the ledger database."
            ),
        ),
        ProbeTurn(
            role="user",
            date_time="2026-04-02 09:27",
            content=(
                "Slow-query plan on ledger_entries:\n"
                "Index Scan using idx_ledger_tenant_ts_v3 on ledger_entries "
                "(cost=0.56..142.88 rows=18 width=96) (actual time=18.4..41.2 ms)\n"
                "Filter: (status = 'PENDING')\n"
                "Buffers: shared hit=312 read=9"
            ),
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-02 09:28",
            content=(
                "Great — the summary of our three locked platform decisions from this incident is: "
                "(1) Envoy 1.31 at the edge, (2) gRPC mTLS internally, and (3) Valkey 8.0 for "
                "rate-limit/idempotency storage."
            ),
        ),
        ProbeTurn(
            role="user",
            date_time="2026-04-02 09:30",
            content="Confirmed. Those three architecture decisions are final for Q2 rollout.",
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-02 09:31",
            content="All three Q2 platform decisions recorded. Ready to close out the incident review.",
        ),
    ]

    ui_turns = [
        ProbeTurn(
            role="user",
            date_time="2026-04-03 11:00",
            content="For the checkout UI redesign we picked React 19 with TanStack Query and Tailwind v4.",
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-03 11:01",
            content="Recorded React 19, TanStack Query, and Tailwind v4 for the checkout frontend.",
        ),
        ProbeTurn(
            role="user",
            date_time="2026-04-03 11:05",
            content="Accessibility audit target is WCAG 2.2 AA with keyboard focus traps on modals.",
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-03 11:06",
            content="WCAG 2.2 AA modal accessibility target locked in.",
        ),
    ]

    webhook_turns = [
        ProbeTurn(
            role="user",
            date_time="2026-04-04 14:00",
            content="Billing webhook retries will use exponential backoff capped at 6 attempts over 24 hours.",
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-04 14:01",
            content="Logged 6-attempt 24h exponential backoff policy for billing webhooks.",
        ),
        ProbeTurn(
            role="user",
            date_time="2026-04-04 14:04",
            content="Signatures use HMAC-SHA256 in the X-Billing-Sig header with a 300s replay window.",
        ),
        ProbeTurn(
            role="assistant",
            date_time="2026-04-04 14:05",
            content="Recorded HMAC-SHA256 X-Billing-Sig header and 300s replay tolerance.",
        ),
    ]

    return MicroDiagnosticProbeSpec(
        probe_id="probe1_verbatim_recovery",
        title="Probe 1: Un-Anchored Verbatim Detail Recovery",
        architectural_claim=(
            "Chatree keeps untouched verbatim branch transcripts in SQLite + FTS5 and answers "
            "via read-only shadow turns even when Tier-2 summaries omit low-salience literals; "
            "write-time distilled flat stores lose un-extracted constants permanently."
        ),
        sessions=[
            ProbeSession(
                session_id="incident_409_gateway",
                title="Incident 409 Payment Gateway Postmortem",
                date_time="2026-04-02",
                turns=incident_turns,
            ),
            ProbeSession(
                session_id="frontend_checkout_ui",
                title="Frontend Checkout UI Stack",
                date_time="2026-04-03",
                turns=ui_turns,
            ),
            ProbeSession(
                session_id="billing_webhook_retries",
                title="Billing Webhook Retry Policy",
                date_time="2026-04-04",
                turns=webhook_turns,
            ),
        ],
        queries=[
            ProbeQuery(
                query_id="p1_q1_timeout",
                probe_id="probe1_verbatim_recovery",
                question=(
                    "In the payment gateway incident debugging session, what exact "
                    "`proxy_read_timeout` value appeared in the legacy nginx ingress snippet?"
                ),
                expected_any_groups=[["43s", "43 seconds", "43 sec"]],
                rationale="Incidental verbatim nginx directive inside turn 3 snippet.",
            ),
            ProbeQuery(
                query_id="p1_q2_retry_epoch",
                probe_id="probe1_verbatim_recovery",
                question=(
                    "In the payment gateway incident response header capture, what exact "
                    "numeric value appeared in the `X-Retry-Epoch` header?"
                ),
                expected_any_groups=[["17"]],
                rationale="Incidental HTTP response header literal inside turn 5.",
            ),
            ProbeQuery(
                query_id="p1_q3_error_code",
                probe_id="probe1_verbatim_recovery",
                question=(
                    "In the coordinator worker log during the gateway failover window, "
                    "what exact error code appeared on the `renew_failed` line?"
                ),
                expected_any_groups=[["ERR_SHARD_LEASE_419"]],
                rationale="Incidental verbatim error token inside turn 7 log line.",
            ),
            ProbeQuery(
                query_id="p1_q4_index_name",
                probe_id="probe1_verbatim_recovery",
                question=(
                    "In the slow-query `EXPLAIN ANALYZE` plan on `ledger_entries`, "
                    "what exact PostgreSQL index name was scanned?"
                ),
                expected_any_groups=[["idx_ledger_tenant_ts_v3"]],
                rationale="Incidental SQL index identifier inside turn 9 EXPLAIN output.",
            ),
        ],
    )


def build_probe2_transitive_staleness() -> MicroDiagnosticProbeSpec:
    """
    Probe 2: Transitive Downstream Staleness (`A -> B -> C` Causal Invalidation).
    Branch A (`Auth-Token-Spec`) -> Branch B (`Edge-Cache-Policy`) -> Branch C (`Mobile-Refresh-Flow`).
    Only A is later revised (`JWT_TTL=60m -> 10m`, `RS256 -> ED25519`) without touching B or C.
    Then an indirect release-safety question is asked in C WITHOUT mentioning 'JWT' or 'auth'.
    """
    return MicroDiagnosticProbeSpec(
        probe_id="probe2_transitive_staleness",
        title="Probe 2: Transitive Downstream Staleness (A -> B -> C)",
        architectural_claim=(
            "Chatree maintains explicit `node_dependencies` edges and runs cycle-safe transitive BFS "
            "to flag downstream branches B (1-hop) and C (2-hop) as `[STALE]` when A materially changes, "
            "injecting alerts into C's ephemeral envelope even when the user's query never mentions A."
        ),
        sessions=[
            ProbeSession(
                session_id="branch_a_auth",
                title="Auth Token Spec",
                date_time="2026-04-10",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-10 10:00",
                        content=(
                            "Let's finalize the core authentication token specification for v1. "
                            "We will issue access tokens with JWT_TTL=60m signed with JWT_ALG=RS256."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-10 10:01",
                        content=(
                            "Locked v1 Auth Token Spec: JWT_TTL=60m (60 minutes) and JWT_ALG=RS256."
                        ),
                    ),
                ],
            ),
            ProbeSession(
                session_id="branch_b_cache",
                title="Edge Cache Policy",
                date_time="2026-04-11",
                imports=["branch_a_auth"],
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-11 10:00",
                        content=(
                            "Based on the Auth Token Spec (JWT_TTL=60m), set the CDN/edge authorization "
                            "cache policy to CACHE_TTL=30m. Invariant: CACHE_TTL must always be at most "
                            "half of JWT_TTL so revoked tokens expire within half a token lifetime."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-11 10:01",
                        content=(
                            "Locked Edge Cache Policy: CACHE_TTL=30m (derived as <= 0.5 * JWT_TTL=60m)."
                        ),
                    ),
                ],
            ),
            ProbeSession(
                session_id="branch_c_mobile",
                title="Mobile Refresh Flow",
                date_time="2026-04-12",
                imports=["branch_b_cache"],
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-12 10:00",
                        content=(
                            "For the iOS/Android background token refresh scheduler, set "
                            "REFRESH_WINDOW=25m based on the Edge Cache Policy (CACHE_TTL=30m). "
                            "Invariant: REFRESH_WINDOW must stay 5 minutes shorter than CACHE_TTL so "
                            "clients proactively rotate credentials before edge cache eviction."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-12 10:01",
                        content=(
                            "Locked Mobile Refresh Flow: REFRESH_WINDOW=25m (derived as CACHE_TTL - 5m)."
                        ),
                    ),
                ],
            ),
            ProbeSession(
                session_id="mobile_release_checklist",
                title="Mobile Release Cut Checklist",
                date_time="2026-04-13",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-13 09:00",
                        content=(
                            "For the mobile release v4.8.0 cut checklist, freeze TestFlight beta 24h "
                            "before App Store submission and stage Play Console rollout at 10% -> 50% -> 100%."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-13 09:01",
                        content="Recorded mobile release v4.8.0 checklist: 24h TestFlight freeze and 10/50/100% staged rollout.",
                    ),
                ],
            ),
            ProbeSession(
                session_id="ios_bg_sync_scheduler",
                title="iOS Background Refresh Scheduler",
                date_time="2026-04-13",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-13 11:00",
                        content=(
                            "On iOS background refresh tasks, BGAppRefreshTask adds 120s jitter to "
                            "earliestBeginDate and defers non-critical mobile sync in Low Power Mode."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-13 11:01",
                        content="Logged iOS BGAppRefreshTask 120s jitter and Low Power Mode deferral for mobile refresh.",
                    ),
                ],
            ),
            ProbeSession(
                session_id="android_workmanager_sync",
                title="Android WorkManager Refresh Policy",
                date_time="2026-04-13",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-13 14:00",
                        content=(
                            "Android mobile refresh uses WorkManager PeriodicWorkRequest with a 15m flex "
                            "window, CONNECTED network constraint, and 30s exponential backoff."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-13 14:01",
                        content="Recorded Android WorkManager 15m flex window and 30s backoff for mobile refresh.",
                    ),
                ],
            ),
            ProbeSession(
                session_id="cdn_edge_cache_purge",
                title="CDN Edge Cache Purge and Vary Rules",
                date_time="2026-04-14",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-14 09:30",
                        content=(
                            "Edge cache invalidation uses Fastly surrogate-key purges (<5s global propagation) "
                            "and Vary headers on Accept-Encoding and X-Tenant-Tier."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-14 09:31",
                        content="Recorded Fastly <5s surrogate-key edge cache purge and Vary header rules.",
                    ),
                ],
            ),
            ProbeSession(
                session_id="mobile_crashlytics_slo",
                title="Mobile Release Stability Gate",
                date_time="2026-04-14",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-14 13:00",
                        content=(
                            "Mobile release promotion gate requires Crashlytics crash-free sessions >= 99.6% "
                            "and cold-start p95 latency <= 1.2s across iOS and Android."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-14 13:01",
                        content="Logged mobile release stability gates: >=99.6% crash-free and <=1.2s cold-start p95.",
                    ),
                ],
            ),
            ProbeSession(
                session_id="edge_cache_swr_tuning",
                title="Edge Cache Stale-While-Revalidate Policy",
                date_time="2026-04-14",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-14 15:00",
                        content=(
                            "Public catalog endpoints on the edge cache use stale-while-revalidate=60s, "
                            "while authenticated mobile endpoints enforce private cache directives."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-14 15:01",
                        content="Recorded edge cache SWR=60s for public catalog and private cache rules for mobile.",
                    ),
                ],
            ),
            ProbeSession(
                session_id="branch_a_auth_v2",
                title="Auth Token Spec Security Revision",
                date_time="2026-04-15",
                is_revision_of="branch_a_auth",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-15 16:00",
                        content=(
                            "Security audit revision for the Auth Token Spec: shorten access token "
                            "lifetime to JWT_TTL=10m and migrate signing from RS256 to JWT_ALG=ED25519 "
                            "effective immediately."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-15 16:01",
                        content=(
                            "Updated Auth Token Spec to v2: JWT_TTL=10m and JWT_ALG=ED25519 "
                            "(superseding 60m / RS256)."
                        ),
                    ),
                ],
            ),
        ],
        queries=[
            ProbeQuery(
                query_id="p2_q1_release_safety",
                probe_id="probe2_transitive_staleness",
                target_session_id="branch_c_mobile",
                question=(
                    "Before we cut the mobile release, double-check our Mobile Refresh Flow "
                    "settings (`REFRESH_WINDOW=25m` and upstream `CACHE_TTL=30m`): are they safe "
                    "to ship as-is, or has any upstream dependency changed that invalidates them? "
                    "Be specific about what changed and why."
                ),
                expected_any_groups=[
                    ["10m", "10 min", "10-minute", "10 minute"],
                    ["ed25519", "stale", "invalid", "unsafe", "not safe", "exceeds", "outdated"],
                ],
                require_warning_or_abstention=True,
                rationale=(
                    "Indirect downstream query from Branch C that never mentions 'JWT' or 'auth': "
                    "tests whether 2-hop transitive staleness alerts propagate from A -> B -> C."
                ),
            )
        ],
    )


def build_probe3_silent_quarantine() -> MicroDiagnosticProbeSpec:
    """
    Probe 3: Silent Dead-End Quarantine (No Cancellation Essay).
    Main branch settles on PostgreSQL 16.4; exploratory MongoDB 7.0 branch is silently archived
    via native `archive()` with ZERO textual 'CANCELLATION' turn typed in the transcript.
    """
    return MicroDiagnosticProbeSpec(
        probe_id="probe3_silent_quarantine",
        title="Probe 3: Silent Dead-End Quarantine (No Cancellation Essay)",
        architectural_claim=(
            "Chatree's native `status='ARCHIVED'` atomically removes scrapped exploration branches "
            "from all 4 retrieval tiers without requiring users to write cancellation essays; "
            "flat vector stores retain silent dead-ends as active memories and leak them."
        ),
        sessions=[
            ProbeSession(
                session_id="main_storage_arch",
                title="Main Ledger Storage Architecture",
                date_time="2026-04-20",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-20 10:00",
                        content=(
                            "Let's lock in the production storage architecture for the ledger service: "
                            "PostgreSQL 16.4 in primary_region=us-east-1 with pgbouncer_pool_size=120 "
                            "and isolation_level=serializable."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-20 10:01",
                        content=(
                            "Confirmed production ledger storage: PostgreSQL 16.4, "
                            "pgbouncer_pool_size=120, isolation_level=serializable, us-east-1."
                        ),
                    ),
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-20 10:05",
                        content="Backups run via pgBackRest continuous WAL archiving to S3 with 35-day retention.",
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-20 10:06",
                        content="Recorded pgBackRest WAL archiving to S3 (35-day retention) for PostgreSQL 16.4.",
                    ),
                ],
            ),
            ProbeSession(
                session_id="explore_mongodb_spike",
                title="Explore MongoDB Document Ledger Spike",
                date_time="2026-04-21",
                archived=True,
                archive_reason="Exploratory spike silently archived after evaluating latency & ACID tradeoffs",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-21 13:00",
                        content=(
                            "Let's prototype an alternative ledger design on MongoDB 7.0 using replica set "
                            "`rs_ledger_0`, `writeConcern=majority`, and `shardKey=tenant_id_hashed`."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-21 13:01",
                        content=(
                            "Setting up the MongoDB 7.0 ledger prototype with replica set `rs_ledger_0`, "
                            "`writeConcern=majority`, and hashed shard key `tenant_id_hashed`."
                        ),
                    ),
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-21 13:08",
                        content=(
                            "For the MongoDB 7.0 ledger nodes, set `wiredTigerCacheSizeGB=24` and "
                            "`readPreference=primaryPreferred` on `rs_ledger_0`."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-21 13:09",
                        content=(
                            "Configured `wiredTigerCacheSizeGB=24` and `readPreference=primaryPreferred` "
                            "on MongoDB 7.0 replica set `rs_ledger_0`."
                        ),
                    ),
                ],
            ),
            ProbeSession(
                session_id="observability_metrics",
                title="Observability and Telemetry Stack",
                date_time="2026-04-22",
                turns=[
                    ProbeTurn(
                        role="user",
                        date_time="2026-04-22 09:00",
                        content=(
                            "Telemetry goes through OpenTelemetry collector `otel-gw-02` with a "
                            "Prometheus scrape interval of 15s and alert SLO p99 < 180ms."
                        ),
                    ),
                    ProbeTurn(
                        role="assistant",
                        date_time="2026-04-22 09:01",
                        content="Recorded otel-gw-02, 15s Prometheus scrape interval, and p99 < 180ms SLO.",
                    ),
                ],
            ),
        ],
        queries=[
            ProbeQuery(
                query_id="p3_q1_active_db",
                probe_id="probe3_silent_quarantine",
                question=(
                    "What database engine, PgBouncer pool size, and transaction isolation level "
                    "are active for our production ledger service?"
                ),
                expected_any_groups=[
                    ["postgresql 16.4", "postgres 16.4", "postgresql"],
                    ["120"],
                    ["serializable"],
                ],
                forbidden_active_tokens=["MongoDB", "rs_ledger_0"],
                rationale="Must recall active PostgreSQL 16.4 stack without asserting archived MongoDB.",
            ),
            ProbeQuery(
                query_id="p3_q2_dead_end_leakage",
                probe_id="probe3_silent_quarantine",
                question=(
                    "What are our active production MongoDB replica set name, write concern, "
                    "and WiredTiger cache size for the ledger service? If none are part of the "
                    "active production architecture, state that clearly."
                ),
                expected_any_groups=[
                    [
                        "none",
                        "not part of",
                        "not using mongodb",
                        "no active",
                        "no mongodb",
                        "postgresql",
                        "do not use mongodb",
                        "not mentioned",
                        "no information",
                        "don't have",
                        "do not have",
                        "not configured",
                    ]
                ],
                forbidden_active_tokens=["rs_ledger_0", "writeConcern=majority", "wiredTigerCacheSizeGB"],
                require_warning_or_abstention=True,
                rationale=(
                    "Silent archival quarantine check: archived MongoDB spike must never leak "
                    "`rs_ledger_0`, `writeConcern=majority`, or `wiredTigerCacheSizeGB=24` as active."
                ),
            ),
        ],
    )


def build_all_micro_diagnostic_probes() -> list[MicroDiagnosticProbeSpec]:
    return [
        build_probe1_verbatim_recovery(),
        build_probe2_transitive_staleness(),
        build_probe3_silent_quarantine(),
    ]


# =============================================================================
# Shared Deterministic Scoring Helpers (Transparent, Zero-Token, Reproducible)
# =============================================================================

_WARNING_OR_INVALID_RE = re.compile(
    r"\b("
    r"not safe|unsafe|invalid|invalidated|stale|outdated|superseded|violat\w+|"
    r"must be (?:reduced|shortened|updated|changed|lowered)|needs? (?:update|updating|revision)|"
    r"do not ship|should not ship|cannot ship|block(?:ed|ing)? release|"
    r"exceeds?|longer than|greater than|conflicts? with"
    r")\b",
    flags=re.IGNORECASE,
)


def score_probe_answer(query: ProbeQuery, answer: str) -> dict[str, Any]:
    """Deterministic, auditable scoring for micro-diagnostic probe answers."""
    low = (answer or "").lower()
    matched_groups: list[str] = []
    missed_groups: list[list[str]] = []
    for grp in query.expected_any_groups:
        hit = next((tok for tok in grp if tok.lower() in low), None)
        if hit is not None:
            matched_groups.append(hit)
        else:
            missed_groups.append(grp)

    leaked_tokens: list[str] = [
        tok for tok in query.forbidden_active_tokens if is_token_asserted_as_active(answer, tok)
    ]

    warning_ok = True
    if query.require_warning_or_abstention and query.probe_id == "probe2_transitive_staleness":
        warning_ok = bool(_WARNING_OR_INVALID_RE.search(answer or ""))

    passed = (len(missed_groups) == 0) and (len(leaked_tokens) == 0) and warning_ok
    return {
        "passed": passed,
        "matched_groups": matched_groups,
        "missed_groups": missed_groups,
        "leaked_tokens": leaked_tokens,
        "warning_ok": warning_ok,
    }


# =============================================================================
# Part A — Zero-Token Deterministic Simulation & Real-Framework Execution Trace
# =============================================================================


class PrefixTurnMetric(BaseModel):
    turn: int
    chatree_prompt_tokens: int
    chatree_lcp_tokens: int
    chatree_lcp_ratio: float
    mem0_end_of_sys_prompt_tokens: int = 0
    mem0_end_of_sys_lcp_tokens: int = 0
    mem0_end_of_sys_lcp_ratio: float = 0.0
    mem0_prompt_tokens: int
    mem0_lcp_tokens: int
    mem0_lcp_ratio: float


class PrefixStabilitySimReport(BaseModel):
    num_turns: int
    chatree_mean_lcp_ratio: float
    mem0_end_of_sys_mean_lcp_ratio: float = 0.0
    mem0_mean_lcp_ratio: float
    chatree_total_raw_tokens: int
    chatree_effective_billed_tokens: int
    mem0_end_of_sys_total_raw_tokens: int = 0
    mem0_end_of_sys_effective_billed_tokens: int = 0
    mem0_total_raw_tokens: int
    mem0_effective_billed_tokens: int
    effective_input_savings_vs_end_of_sys_pct: float = 0.0
    effective_input_savings_pct: float
    per_turn: list[PrefixTurnMetric]


class MaterialityCase(BaseModel):
    case_id: str
    category: Literal[
        "identifier_numeric",
        "polarity_flip",
        "semantic_non_alphanumeric",
        "cosmetic_rewording",
    ]
    expected_severity: Literal["material", "cosmetic"]
    old_summary: str
    new_summary: str


class MaterialityCaseResult(BaseModel):
    case_id: str
    category: str
    expected_severity: str
    predicted_severity: str
    correct: bool
    delta_reason: str


class MaterialityEvalReport(BaseModel):
    embedder_backend: str
    cosine_threshold: float
    total_cases: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    category_breakdown: dict[str, dict[str, Any]]
    cases: list[MaterialityCaseResult]


class ScalingPoint(BaseModel):
    sessions: int
    queries: int
    chatree_write_calls: int
    chatree_write_tokens_in: int
    chatree_write_usd: float
    chatree_query_usd: float
    chatree_total_usd: float
    mem0_write_calls: int
    mem0_write_tokens_in: int
    mem0_write_usd_uncached: float
    mem0_write_usd_70pct_sys_cached: float
    mem0_query_usd: float
    mem0_total_usd_uncached: float
    mem0_total_usd_70pct_sys_cached: float


class ScalingSimReport(BaseModel):
    model: str
    mem0_extraction_prompt_chars: int
    mem0_extraction_prompt_tokens_est: int
    mem0_update_prompt_chars: int
    mem0_prompt_source: str
    chatree_cold_trim_char_budget: int
    empirical_longmemeval_anchor: dict[str, Any]
    points: list[ScalingPoint]


class TraceStep(BaseModel):
    step: int
    operation: str
    chatree_action: str
    mem0_action: str


class ProbeTraceComparison(BaseModel):
    probe_id: str
    title: str
    mem0_trace_backend: str
    chatree_passed: int
    chatree_total: int
    mem0_passed: int
    mem0_total: int
    chatree_extra_metrics: dict[str, Any] = Field(default_factory=dict)
    mem0_extra_metrics: dict[str, Any] = Field(default_factory=dict)
    steps: list[TraceStep] = Field(default_factory=list)
    query_outcomes: list[dict[str, Any]] = Field(default_factory=list)


class ZeroTokenSimulationReport(BaseModel):
    proof_type: Literal["zero_token_deterministic_simulation"] = "zero_token_deterministic_simulation"
    real_api_tokens_used: int = 0
    real_api_usd_cost: float = 0.0
    mem0_trace_backend: str
    prefix_stability: PrefixStabilitySimReport
    scaling: ScalingSimReport
    probe_traces: list[ProbeTraceComparison]
    materiality_eval: MaterialityEvalReport | None = None

    def to_markdown(self) -> str:
        ps = self.prefix_stability
        lines: list[str] = [
            "# Track 8A: Zero-Token Structural Simulation & Real-Framework Execution Trace",
            "",
            (
                f"> **Zero-Token Guarantee:** `real_api_tokens_used = {self.real_api_tokens_used}`, "
                f"`real_api_usd_cost = ${self.real_api_usd_cost:.2f}`. "
                f"Chatree executes its real SQLite + ContextAssembler + StalenessService + TreeRAG engine; "
                f"Mem0 trace backend = `{self.mem0_trace_backend}`."
            ),
            "",
            "## 1. Prompt Prefix KV-Cache Stability (Exact 3-Way LCP Trace over 16 Turns)",
            "",
            (
                "All three layouts process the identical 16-turn multi-topic engineering dialogue with the "
                "exact same base system prompt. Chatree appends its dynamic `<dialogue_tree_context>` block "
                "**ephemerally at the tail** of the newest user turn, keeping both the system prompt and all "
                "prior turns immutable. We compare against two Mem0 system-prompt injection layouts: "
                "(a) **End-of-System Injection** (appending retrieved memories after the static system prompt, "
                "which preserves the system prompt in cache but invalidates all prior turns on every top-k change), "
                "and (b) **Byte-0 Prepend** (prepending memories before the system prompt). "
                "*Note: A Mem0 integration that injects memories inside the newest user turn (tail injection) "
                "can match static-turn cacheability; this simulation evaluates system-prompt injection layouts.*"
            ),
            "",
            "| Metric | Chatree (Tail-Only Envelope) | Mem0 (End-of-System Injection) | Mem0 (Byte-0 Prepend) |",
            "|---|---:|---:|---:|",
            (
                f"| Mean Prefix Cacheability `LCP(t, t-1) / |P_t|` (Turns 2-{ps.num_turns}) "
                f"| **{ps.chatree_mean_lcp_ratio * 100:.1f}%** "
                f"| {ps.mem0_end_of_sys_mean_lcp_ratio * 100:.1f}% "
                f"| {ps.mem0_mean_lcp_ratio * 100:.1f}% |"
            ),
            (
                f"| Raw Input Tokens ({ps.num_turns} turns) "
                f"| {ps.chatree_total_raw_tokens:,} "
                f"| {ps.mem0_end_of_sys_total_raw_tokens:,} "
                f"| {ps.mem0_total_raw_tokens:,} |"
            ),
            (
                f"| Effective Billed Input Tokens (`0.1x` cached read) "
                f"| **{ps.chatree_effective_billed_tokens:,}** "
                f"| {ps.mem0_end_of_sys_effective_billed_tokens:,} "
                f"| {ps.mem0_effective_billed_tokens:,} |"
            ),
            (
                f"| Effective Input Token Reduction (Chatree vs. Layout) "
                f"| baseline "
                f"| **-{ps.effective_input_savings_vs_end_of_sys_pct:.1f}%** "
                f"| **-{ps.effective_input_savings_pct:.1f}%** |"
            ),
            "",
            "<details><summary>Turn-by-Turn Exact LCP Trace (Turns 1-16)</summary>",
            "",
            "| Turn | Chatree Tok | Chatree Cache % | Mem0 End-of-Sys Cache % | Mem0 Byte-0 Cache % |",
            "|---:|---:|---:|---:|---:|",
        ]
        for pt in ps.per_turn:
            lines.append(
                f"| {pt.turn} | {pt.chatree_prompt_tokens} "
                f"| {pt.chatree_lcp_ratio * 100:.1f}% "
                f"| {pt.mem0_end_of_sys_lcp_ratio * 100:.1f}% "
                f"| {pt.mem0_lcp_ratio * 100:.1f}% |"
            )
        lines.extend(["", "</details>", ""])

        sc = self.scaling
        lines.extend(
            [
                "## 2. Write-Time vs. Query-Time LLM Call & Cost Scaling",
                "",
                (
                    f"- **Measured upstream `mem0ai` extraction system prompt:** "
                    f"`{sc.mem0_extraction_prompt_chars:,}` chars (~`{sc.mem0_extraction_prompt_tokens_est:,}` tokens; "
                    f"source: `{sc.mem0_prompt_source}`) + `{sc.mem0_update_prompt_chars:,}`-char update prompt on reconciliation."
                ),
                (
                    f"- **Chatree write path:** `0` extra LLM calls per live chat turn; `1` cold-trimmed summary call "
                    f"capped at `{sc.chatree_cold_trim_char_budget:,}` chars (~1,875 tok, 0 tool schemas) on branch finish."
                ),
                (
                    "- **Empirical LongMemEval-S anchor (`n=30`, `deepseek/deepseek-chat`):** "
                    "Mem0 `$1.5460` ingest (character-heuristic estimate prior to v0.2.0 client metering) + `$0.0088` query = **`$1.5548`** (`176x` write tax); "
                    "Chatree `$0.892` ingest + `$0.163` query = **`$1.055`** (`32%` lower total USD)."
                ),
                "",
                "| Sessions (N) | Queries (Q) | Chatree Write Calls | Chatree Total USD | Mem0 Write Calls | Mem0 Total USD (70% sys-cached) | Mem0 Total USD (uncached) |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for sp in sc.points:
            lines.append(
                f"| {sp.sessions} | {sp.queries} | {sp.chatree_write_calls} "
                f"| **${sp.chatree_total_usd:.4f}** | {sp.mem0_write_calls} "
                f"| ${sp.mem0_total_usd_70pct_sys_cached:.4f} | ${sp.mem0_total_usd_uncached:.4f} |"
            )

        lines.extend(
            [
                "",
                "## 3. Controlled Lossy-Distillation Stress Traces (Probes 1-3)",
                "",
                (
                    "To isolate pure architectural recovery guarantees under distillation loss, **both frameworks "
                    "receive the exact same deliberately lossy distillation** (high-level decisions kept; incidental "
                    "low-salience constants omitted). Note the honest contrast with live Track 8B on the short 12-turn "
                    "Probe 1 dialogue, where DeepSeek+Mem0 emitted 13 granular facts and passed 4/4 — demonstrating "
                    "that flat distilled stores rely on extraction verbosity rather than a structural verbatim fallback."
                ),
                "",
                "| Probe | Chatree Pass | Mem0 Pass | Key Structural Mechanism |",
                "|---|---:|---:|---|",
            ]
        )
        for tr in self.probe_traces:
            mech = ""
            if tr.probe_id == "probe1_verbatim_recovery":
                mech = "Verbatim SQLite log + FTS5 + read-only lookup recovers omitted constants"
            elif tr.probe_id == "probe2_transitive_staleness":
                mech = (
                    f"`node_dependencies` BFS flags B (1h) + C (2h) STALE "
                    f"({tr.chatree_extra_metrics.get('stale_dependents_flagged', 0)}/2 vs. 0/2)"
                )
            elif tr.probe_id == "probe3_silent_quarantine":
                mech = "Native `status='ARCHIVED'` excludes dead-end from all search levels (0% leakage)"
            lines.append(
                f"| {tr.title} | **{tr.chatree_passed}/{tr.chatree_total}** "
                f"| {tr.mem0_passed}/{tr.mem0_total} | {mech} |"
            )

        for tr in self.probe_traces:
            lines.extend(["", f"### {tr.title}", ""])
            lines.append("| Step | Operation | Chatree State & Execution | Mem0 State & Execution |")
            lines.append("|---:|---|---|---|")
            for st in tr.steps:
                lines.append(
                    f"| {st.step} | {st.operation} | {st.chatree_action} | {st.mem0_action} |"
                )
            if tr.mem0_extra_metrics:
                lines.append("")
                lines.append(
                    f"- **Chatree metrics:** `{json.dumps(tr.chatree_extra_metrics, sort_keys=True)}`"
                )
                lines.append(
                    f"- **Mem0 metrics:** `{json.dumps(tr.mem0_extra_metrics, sort_keys=True)}`"
                )

        if self.materiality_eval is not None:
            me = self.materiality_eval
            lines.extend(
                [
                    "",
                    "## 4. Zero-Token Materiality Classifier Benchmark (20 Branch-Revision Pairs)",
                    "",
                    (
                        f"Evaluates `StalenessService.classify_change_materiality()` (`embedder={me.embedder_backend}`, "
                        f"`cosine_threshold={me.cosine_threshold:.2f}`) on 20 paired branch summaries: "
                        f"10 true material revisions (4 identifier/numeric, 3 polarity flips, 3 subtle non-alphanumeric "
                        f"semantic changes) and 10 true cosmetic rewordings."
                    ),
                    "",
                    "| Metric | Score |",
                    "|---|---:|",
                    f"| Precision (material) | **{me.precision * 100:.1f}%** (TP={me.tp}, FP={me.fp}) |",
                    f"| Recall (material) | **{me.recall * 100:.1f}%** (TP={me.tp}, FN={me.fn}) |",
                    f"| F1 (material) | **{me.f1 * 100:.1f}%** |",
                    f"| Overall Accuracy | **{me.accuracy * 100:.1f}%** ({me.tp + me.tn}/{me.total_cases}) |",
                    "",
                    "| Category | Expected | Correct / Total | Recall / Specificity |",
                    "|---|---|---:|---:|",
                ]
            )
            for cat_name, stats in me.category_breakdown.items():
                lines.append(
                    f"| `{cat_name}` | `{stats['expected']}` | {stats['correct']}/{stats['total']} "
                    f"| {stats['rate'] * 100:.1f}% |"
                )
        return "\n".join(lines) + "\n"

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def _tokenize_messages_for_lcp(messages: list[dict[str, Any]]) -> list[str]:
    """Deterministic sub-token stream preserving exact message boundaries for LCP calculation."""
    tokens: list[str] = []
    for m in messages:
        tokens.append(f"<|role:{m.get('role', '')}|>")
        content = str(m.get("content") or "")
        tokens.extend(re.findall(r"\w+|[^\w\s]", content))
        tokens.append("<|end|>")
    return tokens


def _lcp_len(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


async def simulate_prefix_kv_cache_stability(num_turns: int = 16) -> PrefixStabilitySimReport:
    """
    Sim 1: Run `num_turns` real interactive turns through Chatree's real `ContextAssembler`
    (with live sibling branches & dynamic tail envelopes) vs. two Mem0-style system-prompt
    injection layouts (end-of-system-prompt append and byte-0 prepend), measuring exact
    turn-to-turn Longest Common Token Prefix (LCP).
    """
    topics = [
        ("auth", "Reviewing JWT RS256 key rotation cadence and JWKS cache TTL."),
        ("db", "Tuning PostgreSQL 16.4 autovacuum and PgBouncer pool sizing."),
        ("edge", "Configuring Envoy 1.31 circuit breakers and retry budgets."),
        ("mobile", "Auditing iOS background refresh window and token skew tolerance."),
    ]
    mem0_pool = [
        "Service uses PostgreSQL 16.4 with PgBouncer pool size 120 in us-east-1.",
        "Edge gateway migrated to Envoy 1.31 with gRPC mTLS for internal hops.",
        "Rate limiting and idempotency keys run on Valkey 8.0.",
        "Access tokens use JWT_TTL=10m signed with ED25519.",
        "Mobile background refresh window must stay 5 minutes below edge cache TTL.",
        "Telemetry pipelines flow through otel-gw-02 with 15s Prometheus scrape.",
        "Checkout UI uses React 19, TanStack Query, and Tailwind v4.",
        "Billing webhooks sign payloads with HMAC-SHA256 in X-Billing-Sig.",
    ]

    cfg = ChatreeConfig(
        db_path=":memory:",
        default_model="mock/chatree-demo",
        guardrails=GuardrailPolicy(auto_title_enabled=False),
    )
    chatree_prompts: list[list[str]] = []
    mem0_end_of_sys_prompts: list[list[str]] = []
    mem0_prompts: list[list[str]] = []
    mem0_history: list[dict[str, Any]] = []

    async with Chatree.in_memory(config=cfg, embedder=HashEmbedder()) as ct:
        tree = await ct.create_tree("Prefix Stability Sim")
        root = await tree.root()
        # Spawn 2 completed sibling branches so Chatree's tail envelope carries live branch metadata
        sib1 = await root.spawn("Database tuning", title="Postgres Tuning", mode="clean", switch_head=False)
        await ct.repo.commit_node_summary(sib1.id, "PostgreSQL 16.4 with PgBouncer 120.", status="COMPLETED")
        sib2 = await root.spawn("Edge proxy setup", title="Envoy Gateway", mode="clean", switch_head=False)
        await ct.repo.commit_node_summary(sib2.id, "Envoy 1.31 with gRPC mTLS and Valkey 8.0.", status="COMPLETED")
        await ct.rag.index_tree(tree.id)

        for t_idx in range(1, num_turns + 1):
            tag, detail = topics[(t_idx - 1) % len(topics)]
            user_txt = (
                f"Turn {t_idx} [{tag}]: {detail} Verify rollout checklist item #{t_idx} "
                f"and confirm latency SLO impact."
            )
            asst_txt = (
                f"Turn {t_idx} [{tag}] analysis complete: checklist item #{t_idx} verified against "
                f"current production guardrails with zero regression."
            )

            u_msg = await ct.repo.append_message(
                node_id=root.id, tree_id=tree.id, role="user", content=user_txt, mark_dirty=True
            )
            tree_m = await tree.model()
            root_m = await root.model()
            spec_hits = await ct.rag.search(tree.id, user_txt, top_k=2)
            assembled, _, _ = await ct.context.assemble_for_active_turn(
                tree_m,
                root_m,
                u_msg,
                speculative_hits=spec_hits,
                compact_steady_state=(t_idx > 2),
            )
            chatree_prompts.append(_tokenize_messages_for_lcp(assembled))

            await ct.repo.append_message(
                node_id=root.id, tree_id=tree.id, role="assistant", content=asst_txt, mark_dirty=True
            )

            # Compare both Mem0 system-prompt injection layouts using the IDENTICAL base `STATIC_SYSTEM_PROMPT`:
            #   (a) End-of-system-prompt append (preserves system prompt; invalidates prior turns)
            #   (b) Byte-0 prepend (mutates byte 0 directly)
            start_k = (t_idx - 1) % len(mem0_pool)
            retrieved = [mem0_pool[(start_k + j) % len(mem0_pool)] for j in range(4)]
            mem_block = "Relevant Mem0 memories for this turn:\n" + "\n".join(
                f"- {fact}" for fact in retrieved
            )
            mem0_end_sys = STATIC_SYSTEM_PROMPT + "\n\n" + mem_block
            mem0_end_msgs = [
                {"role": "system", "content": mem0_end_sys},
                *mem0_history,
                {"role": "user", "content": user_txt},
            ]
            mem0_end_of_sys_prompts.append(_tokenize_messages_for_lcp(mem0_end_msgs))

            mem0_byte0_sys = mem_block + "\n\n" + STATIC_SYSTEM_PROMPT
            mem0_byte0_msgs = [
                {"role": "system", "content": mem0_byte0_sys},
                *mem0_history,
                {"role": "user", "content": user_txt},
            ]
            mem0_prompts.append(_tokenize_messages_for_lcp(mem0_byte0_msgs))

            mem0_history.append({"role": "user", "content": user_txt})
            mem0_history.append({"role": "assistant", "content": asst_txt})

    per_turn: list[PrefixTurnMetric] = []
    c_raw_total = 0
    c_eff_total = 0.0
    me_raw_total = 0
    me_eff_total = 0.0
    m_raw_total = 0
    m_eff_total = 0.0
    c_ratios: list[float] = []
    me_ratios: list[float] = []
    m_ratios: list[float] = []

    for idx in range(num_turns):
        c_tok = len(chatree_prompts[idx])
        me_tok = len(mem0_end_of_sys_prompts[idx])
        m_tok = len(mem0_prompts[idx])
        if idx == 0:
            c_lcp = 0
            me_lcp = 0
            m_lcp = 0
        else:
            c_lcp = _lcp_len(chatree_prompts[idx - 1], chatree_prompts[idx])
            me_lcp = _lcp_len(mem0_end_of_sys_prompts[idx - 1], mem0_end_of_sys_prompts[idx])
            m_lcp = _lcp_len(mem0_prompts[idx - 1], mem0_prompts[idx])
        c_ratio = c_lcp / max(1, c_tok)
        me_ratio = me_lcp / max(1, me_tok)
        m_ratio = m_lcp / max(1, m_tok)
        if idx > 0:
            c_ratios.append(c_ratio)
            me_ratios.append(me_ratio)
            m_ratios.append(m_ratio)
        c_raw_total += c_tok
        me_raw_total += me_tok
        m_raw_total += m_tok
        # Standard 90% KV-cache discount (cached read billed at 0.1x)
        c_eff_total += (c_tok - c_lcp) + 0.1 * c_lcp
        me_eff_total += (me_tok - me_lcp) + 0.1 * me_lcp
        m_eff_total += (m_tok - m_lcp) + 0.1 * m_lcp
        per_turn.append(
            PrefixTurnMetric(
                turn=idx + 1,
                chatree_prompt_tokens=c_tok,
                chatree_lcp_tokens=c_lcp,
                chatree_lcp_ratio=round(c_ratio, 4),
                mem0_end_of_sys_prompt_tokens=me_tok,
                mem0_end_of_sys_lcp_tokens=me_lcp,
                mem0_end_of_sys_lcp_ratio=round(me_ratio, 4),
                mem0_prompt_tokens=m_tok,
                mem0_lcp_tokens=m_lcp,
                mem0_lcp_ratio=round(m_ratio, 4),
            )
        )

    c_eff_int = int(round(c_eff_total))
    me_eff_int = int(round(me_eff_total))
    m_eff_int = int(round(m_eff_total))
    savings_vs_end = (1.0 - (c_eff_total / max(1.0, me_eff_total))) * 100.0
    savings_vs_byte0 = (1.0 - (c_eff_total / max(1.0, m_eff_total))) * 100.0
    return PrefixStabilitySimReport(
        num_turns=num_turns,
        chatree_mean_lcp_ratio=round(sum(c_ratios) / max(1, len(c_ratios)), 4),
        mem0_end_of_sys_mean_lcp_ratio=round(sum(me_ratios) / max(1, len(me_ratios)), 4),
        mem0_mean_lcp_ratio=round(sum(m_ratios) / max(1, len(m_ratios)), 4),
        chatree_total_raw_tokens=c_raw_total,
        chatree_effective_billed_tokens=c_eff_int,
        mem0_end_of_sys_total_raw_tokens=me_raw_total,
        mem0_end_of_sys_effective_billed_tokens=me_eff_int,
        mem0_total_raw_tokens=m_raw_total,
        mem0_effective_billed_tokens=m_eff_int,
        effective_input_savings_vs_end_of_sys_pct=round(savings_vs_end, 2),
        effective_input_savings_pct=round(savings_vs_byte0, 2),
        per_turn=per_turn,
    )


DEFAULT_MATERIALITY_CASES: list[MaterialityCase] = [
    # --- 10 True Material Revisions ---
    # 4 Identifier / Numeric / Config revisions
    MaterialityCase(
        case_id="mat_id_1",
        category="identifier_numeric",
        expected_severity="material",
        old_summary="Auth service issues JWT tokens with JWT_TTL=60m signed with RS256.",
        new_summary="Auth service issues JWT tokens with JWT_TTL=10m signed with ED25519.",
    ),
    MaterialityCase(
        case_id="mat_id_2",
        category="identifier_numeric",
        expected_severity="material",
        old_summary="Primary database runs PostgreSQL 16.4 with PgBouncer pool_size=120.",
        new_summary="Primary database runs PostgreSQL 16.4 with PgBouncer pool_size=40.",
    ),
    MaterialityCase(
        case_id="mat_id_3",
        category="identifier_numeric",
        expected_severity="material",
        old_summary="Checkout gateway retries upstream calls up to max_retries=3 with timeout_ms=2500.",
        new_summary="Checkout gateway retries upstream calls up to max_retries=1 with timeout_ms=800.",
    ),
    MaterialityCase(
        case_id="mat_id_4",
        category="identifier_numeric",
        expected_severity="material",
        old_summary="Event ingestion writes to Kafka topic orders.v1 on cluster prod-us-east-1.",
        new_summary="Event ingestion writes to Kafka topic orders.v2 on cluster prod-eu-west-1.",
    ),
    # 3 Polarity / Constraint flips
    MaterialityCase(
        case_id="mat_pol_1",
        category="polarity_flip",
        expected_severity="material",
        old_summary="Allow public unauthenticated access to the metrics healthcheck endpoint.",
        new_summary="Forbid public unauthenticated access to the metrics healthcheck endpoint.",
    ),
    MaterialityCase(
        case_id="mat_pol_2",
        category="polarity_flip",
        expected_severity="material",
        old_summary="Edge routing layer stays stateless across all regional nodes.",
        new_summary="Edge routing layer stays stateful across all regional nodes.",
    ),
    MaterialityCase(
        case_id="mat_pol_3",
        category="polarity_flip",
        expected_severity="material",
        old_summary="Audit log writes run in synchronous mode before returning a response.",
        new_summary="Audit log writes run in asynchronous mode before returning a response.",
    ),
    # 3 Subtle Non-Alphanumeric Semantic Requirement Changes (no IDs, numbers, or polarity keywords changed)
    MaterialityCase(
        case_id="mat_sem_1",
        category="semantic_non_alphanumeric",
        expected_severity="material",
        old_summary="Billing invoices settle in dollars at the end of every month.",
        new_summary="Billing invoices settle in euros at the end of every quarter.",
    ),
    MaterialityCase(
        case_id="mat_sem_2",
        category="semantic_non_alphanumeric",
        expected_severity="material",
        old_summary="Customer incidents escalate directly to the on-call platform engineers.",
        new_summary="Customer incidents escalate first to daytime operations triage staff.",
    ),
    MaterialityCase(
        case_id="mat_sem_3",
        category="semantic_non_alphanumeric",
        expected_severity="material",
        old_summary="Nightly snapshots replicate across distant regions for disaster recovery.",
        new_summary="Nightly snapshots stay inside the local datacenter only.",
    ),
    # --- 10 True Cosmetic Rewordings (zero decision/spec changes) ---
    MaterialityCase(
        case_id="cos_1",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Auth service issues JWT tokens with JWT_TTL=10m signed using ED25519.",
        new_summary="The auth service signs JWT tokens with ED25519 and uses JWT_TTL=10m.",
    ),
    MaterialityCase(
        case_id="cos_2",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Primary database runs PostgreSQL 16.4 with PgBouncer pool_size=120.",
        new_summary="Our primary database uses PostgreSQL 16.4 and PgBouncer with pool_size=120.",
    ),
    MaterialityCase(
        case_id="cos_3",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Edge gateway uses Envoy 1.31 with gRPC mTLS for internal traffic.",
        new_summary="Internal traffic goes through the Envoy 1.31 edge gateway using gRPC mTLS.",
    ),
    MaterialityCase(
        case_id="cos_4",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Rate limiting and idempotency keys run on Valkey 8.0.",
        new_summary="Valkey 8.0 handles both rate limiting and idempotency keys.",
    ),
    MaterialityCase(
        case_id="cos_5",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Checkout gateway retries upstream calls up to max_retries=3 with timeout_ms=2500.",
        new_summary="For upstream calls, the checkout gateway uses timeout_ms=2500 and up to max_retries=3.",
    ),
    MaterialityCase(
        case_id="cos_6",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Telemetry pipelines send metrics through otel-gw-02 on a 15s scrape interval.",
        new_summary="Metrics flow via otel-gw-02 in the telemetry pipeline with a 15s scrape interval.",
    ),
    MaterialityCase(
        case_id="cos_7",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Billing webhooks verify payloads with HMAC-SHA256 in the X-Billing-Sig header.",
        new_summary="The X-Billing-Sig header carries an HMAC-SHA256 signature to verify billing webhooks.",
    ),
    MaterialityCase(
        case_id="cos_8",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Audit log writes remain synchronous before returning a response to callers.",
        new_summary="Before returning a response to callers, audit log writes remain synchronous.",
    ),
    MaterialityCase(
        case_id="cos_9",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Customer incidents escalate directly to on-call platform engineers.",
        new_summary="On-call platform engineers receive customer incident escalations directly.",
    ),
    MaterialityCase(
        case_id="cos_10",
        category="cosmetic_rewording",
        expected_severity="cosmetic",
        old_summary="Nightly backups keep snapshots for 30d in object bucket ledger-backups-prod.",
        new_summary="Snapshots from nightly backups are retained for 30d in the ledger-backups-prod bucket.",
    ),
]


async def evaluate_materiality_classifier(
    *,
    cases: list[MaterialityCase] | None = None,
    embedder: Embedder | None = None,
    prefer_fastembed: bool = True,
) -> MaterialityEvalReport:
    """
    Zero-token evaluation of Chatree's real `StalenessService.classify_change_materiality()`
    across 20 realistic branch-summary revision pairs (10 true material + 10 true cosmetic),
    reporting exact precision, recall, F1, accuracy, and per-category breakdown.
    """
    eval_cases = cases if cases is not None else DEFAULT_MATERIALITY_CASES
    resolved_embedder = embedder
    backend_label = type(embedder).__name__ if embedder is not None else ""
    if resolved_embedder is None and prefer_fastembed:
        try:
            from chatree.providers.embeddings import FastEmbedEmbedder

            resolved_embedder = FastEmbedEmbedder()
            backend_label = "FastEmbedEmbedder(BAAI/bge-small-en-v1.5)"
        except Exception:
            resolved_embedder = None
    if resolved_embedder is None:
        resolved_embedder = HashEmbedder()
        backend_label = "HashEmbedder"

    cfg = ChatreeConfig(
        db_path=":memory:",
        default_model="mock/chatree-demo",
        guardrails=GuardrailPolicy(auto_title_enabled=False),
    )
    results: list[MaterialityCaseResult] = []
    tp = fp = tn = fn = 0
    cat_counts: dict[str, dict[str, Any]] = {}

    async with Chatree.in_memory(config=cfg, embedder=resolved_embedder) as ct:
        threshold = ct.staleness.guardrails.materiality_cosine_threshold
        for case in eval_cases:
            pred, reason = await ct.staleness.classify_change_materiality(
                case.old_summary, case.new_summary
            )
            correct = pred == case.expected_severity
            if case.expected_severity == "material":
                if pred == "material":
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == "cosmetic":
                    tn += 1
                else:
                    fp += 1
            entry = cat_counts.setdefault(
                case.category, {"expected": case.expected_severity, "correct": 0, "total": 0}
            )
            entry["total"] += 1
            if correct:
                entry["correct"] += 1
            results.append(
                MaterialityCaseResult(
                    case_id=case.case_id,
                    category=case.category,
                    expected_severity=case.expected_severity,
                    predicted_severity=pred,
                    correct=correct,
                    delta_reason=reason,
                )
            )

    for entry in cat_counts.values():
        entry["rate"] = round(entry["correct"] / max(1, entry["total"]), 4)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    total = len(eval_cases)
    accuracy = (tp + tn) / max(1, total)

    return MaterialityEvalReport(
        embedder_backend=backend_label,
        cosine_threshold=threshold,
        total_cases=total,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        category_breakdown=cat_counts,
        cases=results,
    )


def _measure_mem0_prompt_chars() -> tuple[int, int, str]:
    """Read exact prompt sizes from installed `mem0ai` when present, else documented v2.2.1 constants."""
    try:
        os.environ.setdefault("MEM0_TELEMETRY", "false")
        mem0_prompts = importlib.import_module("mem0.configs.prompts")

        ext = len(getattr(mem0_prompts, "ADDITIVE_EXTRACTION_PROMPT", "") or "")
        upd = len(getattr(mem0_prompts, "DEFAULT_UPDATE_MEMORY_PROMPT", "") or "")
        if ext > 0:
            mem0_pkg = importlib.import_module("mem0")

            ver = getattr(mem0_pkg, "__version__", "installed")
            return ext, upd or DEFAULT_MEM0_UPDATE_PROMPT_CHARS, f"live_import:mem0ai=={ver}"
    except Exception:
        pass
    return (
        DEFAULT_MEM0_EXTRACTION_PROMPT_CHARS,
        DEFAULT_MEM0_UPDATE_PROMPT_CHARS,
        "documented_constant:mem0ai==2.2.1",
    )


def simulate_write_vs_query_scaling(
    *,
    model: str = "deepseek/deepseek-chat",
    session_counts: tuple[int, ...] = (5, 20, 50, 200),
    queries_list: tuple[int, ...] = (10, 50),
    raw_tokens_per_session: int = 2_200,
) -> ScalingSimReport:
    """
    Sim 2: Analytical + schema-measured scaling trace of write-time vs. query-time calls & USD
    comparing Chatree's cold-trimmed 1-call-per-finished-branch path against upstream `mem0ai`.
    """
    cost = CostModel()
    ext_chars, upd_chars, src = _measure_mem0_prompt_chars()
    ext_sys_tok = max(1, ext_chars // 4)
    upd_sys_tok = max(1, upd_chars // 4)
    chatree_cold_budget_chars = 7_500
    chatree_cold_in_tok = min(raw_tokens_per_session, chatree_cold_budget_chars // 4) + 180
    chatree_cold_out_tok = 160

    points: list[ScalingPoint] = []
    for n_sess in session_counts:
        for n_q in queries_list:
            # Chatree: 1 cold-trimmed summary call per finished session, 0 during chat turns
            c_write_calls = n_sess
            c_write_in = n_sess * chatree_cold_in_tok
            c_write_out = n_sess * chatree_cold_out_tok
            c_write_usd = cost.estimate_usd(
                model, tokens_in=c_write_in, tokens_out=c_write_out, cache_read_tokens=0
            )
            # Chatree query: anchored to empirical v3.3 LongMemEval-S (~$0.0054/query on DeepSeek)
            c_q_in = n_q * 5_600
            c_q_cache = int(c_q_in * 0.65)
            c_q_out = n_q * 280
            c_query_usd = cost.estimate_usd(
                model, tokens_in=c_q_in, tokens_out=c_q_out, cache_read_tokens=c_q_cache
            )

            # Mem0: 1 ADDITIVE_EXTRACTION_PROMPT call per session + ~0.8 reconcile calls after session 1
            reconcile_calls = max(0, int(round((n_sess - 1) * 0.8)))
            m_write_calls = n_sess + reconcile_calls
            # Existing-memory context grows modestly as top-k related facts accumulate (~220 tok/call)
            m_ext_in = n_sess * (ext_sys_tok + raw_tokens_per_session + 220)
            m_upd_in = reconcile_calls * (upd_sys_tok + 420)
            m_write_in = m_ext_in + m_upd_in
            m_write_out = n_sess * 220 + reconcile_calls * 110
            m_write_usd_uncached = cost.estimate_usd(
                model, tokens_in=m_write_in, tokens_out=m_write_out, cache_read_tokens=0
            )
            # Fair optimistic bound: even if 70% of Mem0's static system prompt hits provider KV cache
            m_sys_cached = int((max(0, n_sess - 1) * ext_sys_tok + reconcile_calls * upd_sys_tok) * 0.70)
            m_write_usd_cached = cost.estimate_usd(
                model, tokens_in=m_write_in, tokens_out=m_write_out, cache_read_tokens=m_sys_cached
            )
            # Mem0 query: tiny top-8 vector lookup (~1,125 tok input, ~60 tok output, 0% KV cache)
            m_query_usd = cost.estimate_usd(
                model, tokens_in=n_q * 1_125, tokens_out=n_q * 60, cache_read_tokens=0
            )

            points.append(
                ScalingPoint(
                    sessions=n_sess,
                    queries=n_q,
                    chatree_write_calls=c_write_calls,
                    chatree_write_tokens_in=c_write_in,
                    chatree_write_usd=round(c_write_usd, 4),
                    chatree_query_usd=round(c_query_usd, 4),
                    chatree_total_usd=round(c_write_usd + c_query_usd, 4),
                    mem0_write_calls=m_write_calls,
                    mem0_write_tokens_in=m_write_in,
                    mem0_write_usd_uncached=round(m_write_usd_uncached, 4),
                    mem0_write_usd_70pct_sys_cached=round(m_write_usd_cached, 4),
                    mem0_query_usd=round(m_query_usd, 4),
                    mem0_total_usd_uncached=round(m_write_usd_uncached + m_query_usd, 4),
                    mem0_total_usd_70pct_sys_cached=round(m_write_usd_cached + m_query_usd, 4),
                )
            )

    return ScalingSimReport(
        model=model,
        mem0_extraction_prompt_chars=ext_chars,
        mem0_extraction_prompt_tokens_est=ext_sys_tok,
        mem0_update_prompt_chars=upd_chars,
        mem0_prompt_source=src,
        chatree_cold_trim_char_budget=chatree_cold_budget_chars,
        empirical_longmemeval_anchor={
            "dataset": "LongMemEval-S (n=30, ~115k tok/conv, deepseek/deepseek-chat)",
            "chatree_v3": {"accuracy": 0.800, "query_usd": 0.2348, "ingest_usd": 0.8837, "total_usd": 1.1185},
            "chatree_v3_3_opt": {"query_usd": 0.163, "ingest_usd": 0.892, "total_usd": 1.055, "kv_cache": 0.646},
            "mem0_official": {
                "accuracy": 0.700,
                "query_usd": 0.0088,
                "ingest_usd": 1.5460,
                "total_usd": 1.5548,
                "write_to_query_ratio": 175.7,
            },
        },
        points=points,
    )


# -----------------------------------------------------------------------------
# Shared Deterministic Lossy Distillation (used by BOTH Chatree & Mem0 in Sim 3)
# -----------------------------------------------------------------------------


def _extract_high_level_facts_lossy(transcript_text: str) -> list[str]:
    """
    Realistic lossy extractor applied IDENTICALLY to both Chatree's Tier-2 summary stub
    and Mem0's write-time extraction proxy in Sim 3.
    Preserves headline architectural decisions and config choices, while omitting low-salience
    incidental verbatim log/header/snippet tokens (mirroring real LLM summarization loss).
    """
    low = transcript_text.lower()
    facts: list[str] = []

    # Probe 1: Headline decisions kept; incidental verbatim tokens (43s, Epoch 17, 419, idx_) DROPPED
    if "envoy 1.31" in low:
        facts.append("Headline decision: migrate edge payment gateway to Envoy 1.31 across all regions.")
    if "grpc mtls" in low:
        facts.append("Headline decision: switch internal service-to-service hops from REST/JSON to gRPC mTLS.")
    if "valkey 8.0" in low:
        facts.append("Headline decision: replace legacy Redis 6 with Valkey 8.0 for rate-limit and idempotency keys.")
    if "react 19" in low:
        facts.append("Checkout UI redesign uses React 19, TanStack Query, Tailwind v4, and WCAG 2.2 AA.")
    if "x-billing-sig" in low:
        facts.append("Billing webhooks use 6-attempt 24h exponential backoff and HMAC-SHA256 X-Billing-Sig.")

    # Probe 2: Auth / Edge Cache / Mobile Refresh specs
    if "jwt_ttl=10m" in low or "ed25519" in low:
        facts.append("Auth Token Spec revised to JWT_TTL=10m and signing algorithm JWT_ALG=ED25519.")
    elif "jwt_ttl=60m" in low or "rs256" in low:
        facts.append("Auth Token Spec v1 uses JWT_TTL=60m signed with JWT_ALG=RS256.")
    if "cache_ttl=30m" in low:
        facts.append("Edge Cache Policy sets CACHE_TTL=30m (must stay <= half of JWT_TTL).")
    if "refresh_window=25m" in low:
        facts.append("Mobile Refresh Flow sets REFRESH_WINDOW=25m (5 minutes shorter than CACHE_TTL).")
    if "testflight beta" in low or "v4.8.0" in low:
        facts.append("Mobile release v4.8.0 checklist requires 24h TestFlight freeze and 10/50/100% Play Console staged rollout.")
    if "bgapprefreshtask" in low:
        facts.append("iOS background mobile refresh uses BGAppRefreshTask with 120s jitter and Low Power Mode deferral.")
    if "periodicworkrequest" in low:
        facts.append("Android mobile refresh uses WorkManager PeriodicWorkRequest with 15m flex window and 30s backoff.")
    if "surrogate-key" in low:
        facts.append("Edge cache invalidation uses Fastly surrogate-key purges (<5s) and Vary on Accept-Encoding, X-Tenant-Tier.")
    if "crashlytics" in low:
        facts.append("Mobile release stability gate requires >=99.6% crash-free sessions and <=1.2s cold-start p95.")
    if "stale-while-revalidate=60s" in low:
        facts.append("Edge cache uses stale-while-revalidate=60s for public catalog and private directives for mobile.")

    # Probe 3: Production Postgres + Exploratory MongoDB Spike + Observability
    if "postgresql 16.4" in low:
        facts.append(
            "Production ledger database uses PostgreSQL 16.4 in us-east-1 with "
            "pgbouncer_pool_size=120, isolation_level=serializable, and pgBackRest WAL backups."
        )
    if "mongodb 7.0" in low or "rs_ledger_0" in low:
        facts.append(
            "Ledger MongoDB 7.0 setup uses replica set rs_ledger_0, writeConcern=majority, "
            "shardKey=tenant_id_hashed, wiredTigerCacheSizeGB=24, and readPreference=primaryPreferred."
        )
    if "otel-gw-02" in low:
        facts.append("Observability uses OpenTelemetry collector otel-gw-02, 15s Prometheus scrape, p99 < 180ms SLO.")

    return facts


class _TraceRecordingSimLLM:
    """
    Deterministic zero-token LLM stub for Chatree in Sim 3:
      - Summarizer uses `_extract_high_level_facts_lossy` (same lossy extraction as Mem0!)
      - Shadow query reads verbatim branch turns present in `messages`
      - Active turn on Branch C reacts ONLY to what is actually present in `<dialogue_tree_context>`
        (so if `STALE` is injected, it invokes `dialogue_tree_query_branch` on the stale upstream
        branch to verify the new spec; if not, it declares settings safe!)
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del temperature, tool_choice, kwargs
        tok_in = estimate_messages_tokens(messages)
        last_role = str(messages[-1].get("role") or "")
        last_content = str(messages[-1].get("content") or "")
        body_blob = "\n".join(str(m.get("content") or "") for m in messages[1:-1])

        # 1. Summarizer (intentionally lossy — omits incidental verbatim literals!)
        if "[Chatree Summarizer]" in last_content:
            facts = _extract_high_level_facts_lossy(body_blob)
            summary = " ".join(facts) if facts else "General technical discussion."
            facts_json = json.dumps([{"fact": f, "date": "", "speaker": "user"} for f in facts])
            out = f"{summary}\nFACTS_JSON: {facts_json}"
            self.calls.append({"kind": "summarizer", "tokens_in": tok_in, "facts": len(facts)})
            return LLMResponse(content=out, model_used=model, tokens_in=tok_in, tokens_out=estimate_tokens(out))

        # 2. Read-only shadow turn (reads verbatim leaf turns present in this branch's static prefix!)
        if "[Shadow Query |" in last_content:
            ans = self._answer_from_evidence(last_content, body_blob, allow_not_found=True)
            self.calls.append({"kind": "shadow_turn", "tokens_in": tok_in, "found": ans != "NOT_FOUND_IN_BRANCH"})
            return LLMResponse(content=ans, model_used=model, tokens_in=tok_in, tokens_out=estimate_tokens(ans))

        # 3. TreeQueryEngine planner or final synthesis
        if "Respond ONLY with JSON" in last_content or "branch_ids" in last_content:
            out = json.dumps({"branch_ids": [], "followup_query": ""})
            self.calls.append({"kind": "hop2_planner", "tokens_in": tok_in})
            return LLMResponse(content=out, model_used=model, tokens_in=tok_in, tokens_out=12)

        if "Synthesize a concise" in last_content or "[Branch Evidence]" in last_content or "Question:" in last_content and tools is None:
            ans = self._answer_from_evidence(last_content, last_content + "\n" + body_blob, allow_not_found=False)
            self.calls.append({"kind": "synthesis", "tokens_in": tok_in})
            return LLMResponse(content=ans, model_used=model, tokens_in=tok_in, tokens_out=estimate_tokens(ans))

        # 4. Interactive turn with tools (Probe 2: Branch C checks STALE notice in `<dialogue_tree_context>`)
        if last_role == "tool":
            tool_blob = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "tool")
            ans = self._answer_probe2_from_context(body_blob + "\n" + tool_blob)
            self.calls.append({"kind": "interactive_post_tool", "tokens_in": tok_in})
            return LLMResponse(content=ans, model_used=model, tokens_in=tok_in, tokens_out=estimate_tokens(ans))

        ctx_m = re.search(r"<dialogue_tree_context>(.*?)</dialogue_tree_context>", last_content, re.DOTALL)
        ctx_block = ctx_m.group(1) if ctx_m else ""
        stale_m = re.search(r'STALE \[[^\]]*\]:\s*\n-\s*(\S+)\s+"([^"]+)"', ctx_block)
        if stale_m and tools:
            stale_src_id = stale_m.group(1)
            tc = LLMToolCall(
                id="call_sim_stale_check",
                name="dialogue_tree_query_branch",
                arguments={
                    "target_node_id": stale_src_id,
                    "prompt": "What are the latest Auth Token Spec JWT_TTL and JWT_ALG values?",
                },
            )
            self.calls.append({"kind": "interactive_tool_call", "target": stale_src_id, "tokens_in": tok_in})
            return LLMResponse(
                content="",
                tool_calls=[tc],
                model_used=model,
                tokens_in=tok_in,
                tokens_out=32,
            )

        ans = self._answer_probe2_from_context(body_blob + "\n" + last_content)
        self.calls.append({"kind": "interactive_direct", "tokens_in": tok_in})
        return LLMResponse(content=ans, model_used=model, tokens_in=tok_in, tokens_out=estimate_tokens(ans))

    @staticmethod
    def _answer_from_evidence(question_blob: str, evidence_blob: str, *, allow_not_found: bool) -> str:
        q_low = question_blob.lower()
        ev = evidence_blob
        ev_low = ev.lower()

        if "proxy_read_timeout" in q_low:
            m = re.search(r"proxy_read_timeout\s+([0-9]+s)", ev)
            if m:
                return f"The legacy nginx ingress snippet specifies `proxy_read_timeout {m.group(1)}`."
            return "NOT_FOUND_IN_BRANCH" if allow_not_found else "Not found in the available evidence."

        if "x-retry-epoch" in q_low:
            m = re.search(r"X-Retry-Epoch:\s*([0-9]+)", ev) or re.search(
                r"X-Retry-Epoch[^0-9]{0,40}([0-9]+)", ev, flags=re.IGNORECASE
            )
            if m:
                return f"The captured response header was `X-Retry-Epoch: {m.group(1)}`."
            return "NOT_FOUND_IN_BRANCH" if allow_not_found else "Not found in the available evidence."

        if "renew_failed" in q_low or "error code" in q_low:
            m = re.search(r"(ERR_SHARD_LEASE_[0-9]+)", ev)
            if m:
                return f"The coordinator `renew_failed` log reported error code `{m.group(1)}`."
            return "NOT_FOUND_IN_BRANCH" if allow_not_found else "Not found in the available evidence."

        if "explain analyze" in q_low or "index name" in q_low:
            m = re.search(r"(idx_[a-z0-9_]+)", ev)
            if m:
                return f"The `EXPLAIN ANALYZE` plan scanned PostgreSQL index `{m.group(1)}`."
            return "NOT_FOUND_IN_BRANCH" if allow_not_found else "Not found in the available evidence."

        if "jwt_ttl" in q_low or "jwt_alg" in q_low:
            if "jwt_ttl=10m" in ev_low and "ed25519" in ev_low:
                return "Latest Auth Token Spec v2: JWT_TTL=10m and JWT_ALG=ED25519 (superseding 60m / RS256)."
            if "jwt_ttl=60m" in ev_low:
                return "Auth Token Spec v1: JWT_TTL=60m and JWT_ALG=RS256."
            return "NOT_FOUND_IN_BRANCH" if allow_not_found else "No auth token spec found."

        if "mongodb" in q_low or "wiredtiger" in q_low or "replica set" in q_low:
            if "rs_ledger_0" in ev_low:
                return (
                    "Active MongoDB settings found in memory: replica set `rs_ledger_0`, "
                    "`writeConcern=majority`, and `wiredTigerCacheSizeGB=24`."
                )
            if allow_not_found:
                return "NOT_FOUND_IN_BRANCH"
            return (
                "None — no MongoDB replica set, write concern, or WiredTiger cache is part of "
                "the active production ledger architecture (which uses PostgreSQL 16.4)."
            )

        if "pgbouncer" in q_low or "isolation level" in q_low:
            parts: list[str] = []
            if "postgresql 16.4" in ev_low:
                parts.append("PostgreSQL 16.4 with pgbouncer_pool_size=120 and isolation_level=serializable")
            if "rs_ledger_0" in ev_low:
                parts.append("MongoDB 7.0 replica set rs_ledger_0")
            if parts:
                return "Active production ledger storage: " + "; ".join(parts) + "."
            return "NOT_FOUND_IN_BRANCH" if allow_not_found else "Not found."

        return "NOT_FOUND_IN_BRANCH" if allow_not_found else "Not found in the available evidence."

    @staticmethod
    def _answer_probe2_from_context(context_blob: str) -> str:
        low = context_blob.lower()
        if "jwt_ttl=10m" in low or ("10m" in low and "ed25519" in low):
            return (
                "NOT SAFE to ship as-is — upstream `Auth Token Spec` was materially revised to "
                "`JWT_TTL=10m` and `JWT_ALG=ED25519` (superseding 60m / RS256). Both downstream "
                "`CACHE_TTL=30m` (must be <= half of JWT_TTL, i.e., <= 5m) and `REFRESH_WINDOW=25m` "
                "(must be shorter than CACHE_TTL) are now INVALID and STALE."
            )
        return (
            "Settings appear consistent and safe to ship as-is: `CACHE_TTL=30m` satisfies the "
            "half-of-JWT_TTL rule and `REFRESH_WINDOW=25m` sits 5 minutes below `CACHE_TTL=30m`."
        )

    async def stream_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        resp = await self.complete(
            model=model, messages=messages, tools=tools, temperature=temperature, **kwargs
        )
        if resp.content:
            yield LLMStreamEvent(kind="delta", delta=resp.content)
        yield LLMStreamEvent(kind="done", response=resp)


class _LocalMem0ZeroTokenProxy:
    """
    In-process localhost OpenAI-compatible HTTP proxy for zero-token execution tracing through
    REAL, UNMODIFIED `mem0ai` (`from mem0 import Memory`) + FastEmbed + Qdrant!
    Captures exact HTTP prompt payloads sent by `mem0ai` with 0 external API tokens ($0.00).
    """

    def __init__(self) -> None:
        self.captured_calls: list[dict[str, Any]] = []
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> None:
        proxy = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                del format, args

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                msgs = body.get("messages", [])
                sys_txt = str(msgs[0].get("content", "")) if msgs else ""
                user_txt = "\n".join(str(m.get("content", "")) for m in msgs[1:])
                tok_est = estimate_messages_tokens(msgs)

                if "ADDITIVE" in sys_txt.upper() or "Memory Extractor" in sys_txt:
                    call_kind = "mem0_additive_extract"
                    # Extract ONLY from `## New Messages` block so old memories aren't re-extracted
                    nm_idx = user_txt.find("## New Messages")
                    new_block = user_txt[nm_idx:] if nm_idx != -1 else user_txt
                    facts = _extract_high_level_facts_lossy(new_block)
                    reply_obj = {
                        "memory": [
                            {"id": str(i), "text": f, "attributed_to": "user", "linked_memory_ids": []}
                            for i, f in enumerate(facts)
                        ]
                    }
                    reply_str = json.dumps(reply_obj)
                elif "UPDATE" in sys_txt and "DELETE" in sys_txt:
                    call_kind = "mem0_reconcile_update"
                    facts = _extract_high_level_facts_lossy(user_txt)
                    reply_obj = {
                        "memory": [
                            {"id": str(i), "text": f, "event": "ADD"} for i, f in enumerate(facts)
                        ]
                    }
                    reply_str = json.dumps(reply_obj)
                else:
                    call_kind = "mem0_fact_extract"
                    facts = _extract_high_level_facts_lossy(user_txt)
                    reply_str = json.dumps({"facts": facts})

                proxy.captured_calls.append(
                    {
                        "kind": call_kind,
                        "system_chars": len(sys_txt),
                        "user_chars": len(user_txt),
                        "prompt_tokens_est": tok_est,
                        "extracted_count": len(facts),
                    }
                )
                resp_payload = {
                    "id": f"chatcmpl-sim-{len(proxy.captured_calls)}",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": body.get("model", "gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply_str},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": tok_est,
                        "completion_tokens": max(16, len(reply_str) // 4),
                        "total_tokens": tok_est + max(16, len(reply_str) // 4),
                    },
                }
                raw = json.dumps(resp_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
            self._thread = None


class _DeterministicMem0Fallback:
    """Clean deterministic Mem0 flat-store simulator used ONLY when `mem0ai` is not installed."""

    def __init__(self) -> None:
        self.memories: list[str] = []
        self.http_calls: list[dict[str, Any]] = []

    def add(self, msgs: list[dict[str, str]]) -> list[str]:
        blob = "\n".join(m["content"] for m in msgs)
        facts = _extract_high_level_facts_lossy(blob)
        for f in facts:
            if f not in self.memories:
                self.memories.append(f)
        self.http_calls.append(
            {
                "kind": "sim_mem0_extract",
                "system_chars": DEFAULT_MEM0_EXTRACTION_PROMPT_CHARS,
                "user_chars": len(blob),
                "prompt_tokens_est": (DEFAULT_MEM0_EXTRACTION_PROMPT_CHARS + len(blob)) // 4,
                "extracted_count": len(facts),
            }
        )
        return facts

    def search(self, query: str, limit: int = 4) -> list[str]:
        q_terms = {w for w in re.findall(r"[a-z0-9_]+", query.lower()) if len(w) >= 3}
        scored: list[tuple[int, int, str]] = []
        for idx, mem in enumerate(self.memories):
            m_terms = set(re.findall(r"[a-z0-9_]+", mem.lower()))
            overlap = len(q_terms & m_terms)
            scored.append((overlap, idx, mem))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [m for ov, _, m in scored[:limit] if ov > 0] or self.memories[:limit]


class _Mem0TraceHarness:
    """Unified wrapper executing either real `mem0ai` via localhost proxy or deterministic fallback."""

    def __init__(self, *, prefer_real_mem0: bool = True) -> None:
        self.prefer_real_mem0 = prefer_real_mem0
        self.backend_name: str = "deterministic_mem0_fallback"
        self._proxy: _LocalMem0ZeroTokenProxy | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._mem: Any = None
        self._fallback: _DeterministicMem0Fallback | None = None
        self._user_id: str = "sim_user"

    def start(self, user_id: str) -> None:
        self.close()
        self._user_id = user_id
        if self.prefer_real_mem0:
            try:
                os.environ.setdefault("MEM0_TELEMETRY", "false")
                mem0_pkg = importlib.import_module("mem0")
                memory_cls = getattr(mem0_pkg, "Memory")

                self._proxy = _LocalMem0ZeroTokenProxy()
                self._proxy.start()
                self._tmpdir = tempfile.TemporaryDirectory(prefix="chatree-sim-mem0-")
                cfg = {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "gpt-4o-mini",
                            "api_key": "sim-zero-token",
                            "openai_base_url": self._proxy.base_url,
                            "temperature": 0.0,
                        },
                    },
                    "embedder": {
                        "provider": "fastembed",
                        "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384},
                    },
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "path": str(Path(self._tmpdir.name) / "qdrant"),
                            "embedding_model_dims": 384,
                        },
                    },
                    "history_db_path": str(Path(self._tmpdir.name) / "mem0_hist.db"),
                }
                self._mem = memory_cls.from_config(cfg)
                ver = getattr(mem0_pkg, "__version__", "unknown")
                self.backend_name = f"real_mem0ai_{ver}_local_proxy"
                return
            except Exception:
                self.close()
        self._fallback = _DeterministicMem0Fallback()
        self.backend_name = "deterministic_mem0_fallback"

    def add_session(self, session: ProbeSession) -> list[str]:
        msgs = [
            {
                "role": t.role,
                "content": f"[{t.date_time}] {t.content}" if t.date_time else t.content,
            }
            for t in session.turns
        ]
        if self._mem is not None:
            res = self._mem.add(msgs, user_id=self._user_id)
            items = res.get("results", []) if isinstance(res, dict) else []
            return [str(it.get("memory", it)) for it in items]
        assert self._fallback is not None
        return self._fallback.add(msgs)

    def get_all_memories(self) -> list[str]:
        if self._mem is not None:
            try:
                raw = self._mem.get_all(filters={"user_id": self._user_id})
            except (TypeError, ValueError):
                raw = self._mem.get_all(user_id=self._user_id)
            items = raw.get("results", raw) if isinstance(raw, dict) else raw
            return [str(it.get("memory", it)) if isinstance(it, dict) else str(it) for it in (items or [])]
        assert self._fallback is not None
        return list(self._fallback.memories)

    def search(self, query: str, limit: int = 4) -> list[str]:
        if self._mem is not None:
            try:
                raw = self._mem.search(query, filters={"user_id": self._user_id}, top_k=limit)
            except (TypeError, ValueError):
                try:
                    raw = self._mem.search(query, filters={"user_id": self._user_id}, limit=limit)
                except (TypeError, ValueError):
                    raw = self._mem.search(query, user_id=self._user_id, limit=limit)
            items = raw.get("results", raw) if isinstance(raw, dict) else raw
            return [
                str(it.get("memory", it)) if isinstance(it, dict) else str(it)
                for it in (items or [])[:limit]
            ]
        assert self._fallback is not None
        return self._fallback.search(query, limit=limit)

    @property
    def captured_calls(self) -> list[dict[str, Any]]:
        if self._proxy is not None:
            return list(self._proxy.captured_calls)
        if self._fallback is not None:
            return list(self._fallback.http_calls)
        return []

    def close(self) -> None:
        if self._mem is not None:
            for attr in ("vector_store", "_telemetry_vector_store"):
                try:
                    vs = getattr(self._mem, attr, None)
                    client = getattr(vs, "client", None)
                    if client is not None and hasattr(client, "close"):
                        client.close()
                except Exception:
                    pass
            try:
                db = getattr(self._mem, "db", None)
                conn = getattr(db, "connection", None)
                if conn is not None and hasattr(conn, "close"):
                    conn.close()
            except Exception:
                pass
            self._mem = None
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None
        self._fallback = None


async def _trace_probe1_or_3_zero_token(
    spec: MicroDiagnosticProbeSpec, *, prefer_real_mem0_proxy: bool
) -> ProbeTraceComparison:
    steps: list[TraceStep] = []
    step_no = 0
    sim_llm = _TraceRecordingSimLLM()
    embedder = HashEmbedder()

    chatree_adapter = ChatreeSystemAdapter(
        name="chatree_sim",
        model="sim/zero-token",
        llm=sim_llm,
        embedder=embedder,
        reranker=None,
        max_ask_targets=4,
        generate_session_summaries=True,
        include_branch_summary_map=True,
        enable_recursive_hop=False,
        ingest_concurrency=1,
    )
    mem0_h = _Mem0TraceHarness(prefer_real_mem0=prefer_real_mem0_proxy)

    try:
        await chatree_adapter.reset(spec.probe_id)
        await asyncio.to_thread(mem0_h.start, spec.probe_id)

        for sess in spec.sessions:
            step_no += 1
            await chatree_adapter.start_session(
                sess.session_id, title=sess.title, date_time=sess.date_time
            )
            for turn in sess.turns:
                await chatree_adapter.ingest_turn(
                    sess.session_id, turn.role, turn.content, date_time=turn.date_time or sess.date_time
                )
            await chatree_adapter.finish_session(sess.session_id)
            m_added = await asyncio.to_thread(mem0_h.add_session, sess)
            nid = chatree_adapter._session_nodes[sess.session_id]
            c_summary = chatree_adapter._node_summaries.get(nid, "")
            steps.append(
                TraceStep(
                    step=step_no,
                    operation=f"Ingest `{sess.session_id}` ({len(sess.turns)} turns)",
                    chatree_action=(
                        f"Persisted {len(sess.turns)} verbatim turns in SQLite + FTS5; "
                        f"lossy Tier-2 summary ({len(c_summary)} chars)"
                    ),
                    mem0_action=(
                        f"`mem.add()` extracted {len(m_added)} distilled fact(s) into Qdrant; "
                        f"raw transcript discarded"
                    ),
                )
            )
            if sess.archived:
                step_no += 1
                await chatree_adapter.archive_session(sess.session_id, reason=sess.archive_reason)
                steps.append(
                    TraceStep(
                        step=step_no,
                        operation=f"Silent archive `{sess.session_id}` (NO cancellation essay)",
                        chatree_action=(
                            "Native `archive()` set `status='ARCHIVED'`; pruned from Tier 1/2/2.5/3 RAG "
                            "(kept only in SQLite audit log)"
                        ),
                        mem0_action=(
                            "No branch-archive primitive and no cancellation essay -> "
                            f"{len(m_added)} dead-end fact(s) remain ACTIVE in Qdrant"
                        ),
                    )
                )

        mem0_all = await asyncio.to_thread(mem0_h.get_all_memories)
        c_passed = 0
        m_passed = 0
        outcomes: list[dict[str, Any]] = []

        for q in spec.queries:
            step_no += 1
            c_ans = await chatree_adapter.answer_query(q.question)
            c_sc = score_probe_answer(q, c_ans.answer)
            if c_sc["passed"]:
                c_passed += 1

            # For Mem0, search top-4 memories (top-3 for Probe 2/3) and synthesize with IDENTICAL deterministic reader
            m_hits = await asyncio.to_thread(mem0_h.search, q.question, 4)
            m_ev = "\n".join(f"- {h}" for h in m_hits)
            m_ans_txt = _TraceRecordingSimLLM._answer_from_evidence(q.question, m_ev, allow_not_found=False)
            m_sc = score_probe_answer(q, m_ans_txt)
            if m_sc["passed"]:
                m_passed += 1

            steps.append(
                TraceStep(
                    step=step_no,
                    operation=f"Query `{q.query_id}`",
                    chatree_action=(
                        f"{'PASS' if c_sc['passed'] else 'FAIL'} via targets={c_ans.metadata.get('targets_queried', [])} "
                        f"-> `{c_ans.answer[:95]}`"
                    ),
                    mem0_action=(
                        f"{'PASS' if m_sc['passed'] else 'FAIL'} via {len(m_hits)} retrieved fact(s) "
                        f"-> `{m_ans_txt[:95]}`"
                    ),
                )
            )
            outcomes.append(
                {
                    "query_id": q.query_id,
                    "question": q.question,
                    "chatree_answer": c_ans.answer,
                    "chatree_passed": c_sc["passed"],
                    "chatree_leaked": c_sc["leaked_tokens"],
                    "mem0_retrieved": m_hits,
                    "mem0_answer": m_ans_txt,
                    "mem0_passed": m_sc["passed"],
                    "mem0_leaked": m_sc["leaked_tokens"],
                }
            )

        return ProbeTraceComparison(
            probe_id=spec.probe_id,
            title=spec.title,
            mem0_trace_backend=mem0_h.backend_name,
            chatree_passed=c_passed,
            chatree_total=len(spec.queries),
            mem0_passed=m_passed,
            mem0_total=len(spec.queries),
            chatree_extra_metrics={
                "shadow_turns_executed": sum(1 for c in sim_llm.calls if c["kind"] == "shadow_turn"),
                "summarizer_calls": sum(1 for c in sim_llm.calls if c["kind"] == "summarizer"),
            },
            mem0_extra_metrics={
                "mem0_http_calls": len(mem0_h.captured_calls),
                "mem0_max_system_prompt_chars": max(
                    (int(c.get("system_chars", 0)) for c in mem0_h.captured_calls), default=0
                ),
                "stored_memories_count": len(mem0_all),
                "stored_memories": mem0_all,
            },
            steps=steps,
            query_outcomes=outcomes,
        )
    finally:
        await chatree_adapter.close()
        mem0_h.close()


async def _trace_probe2_zero_token(
    spec: MicroDiagnosticProbeSpec, *, prefer_real_mem0_proxy: bool
) -> ProbeTraceComparison:
    """Trace Probe 2 (`A -> B -> C` transitive staleness) through real Chatree + Mem0."""
    steps: list[TraceStep] = []
    step_no = 0
    sim_llm = _TraceRecordingSimLLM()
    cfg = ChatreeConfig(
        db_path=":memory:",
        default_model="sim/zero-token",
        auto_commit_handoffs=True,
        guardrails=GuardrailPolicy(auto_title_enabled=False),
    )
    mem0_h = _Mem0TraceHarness(prefer_real_mem0=prefer_real_mem0_proxy)

    try:
        await asyncio.to_thread(mem0_h.start, spec.probe_id)
        async with Chatree.in_memory(config=cfg, llm=sim_llm, embedder=HashEmbedder()) as ct:
            tree = await ct.create_tree("Probe 2 Transitive Staleness", default_model="sim/zero-token")
            root = await tree.root()
            node_by_sess: dict[str, Any] = {}

            for sess in spec.sessions:
                step_no += 1
                if sess.is_revision_of:
                    target_h = node_by_sess[sess.is_revision_of]
                    for turn in sess.turns:
                        await ct.repo.append_message(
                            node_id=target_h.id,
                            tree_id=tree.id,
                            role=turn.role,
                            content=f"[{turn.date_time}] {turn.content}",
                            mark_dirty=True,
                        )
                    t_m = await tree.model()
                    n_m = await target_h.model()
                    updated_a, _, _, _, _ = await ct.staleness.ensure_fresh_summary(
                        t_m, n_m, force=True, new_status="COMPLETED", session_date=sess.date_time
                    )
                    m_added = await asyncio.to_thread(mem0_h.add_session, sess)
                    b_stale = await node_by_sess["branch_b_cache"].stale_dependencies()
                    c_stale = await node_by_sess["branch_c_mobile"].stale_dependencies()
                    steps.append(
                        TraceStep(
                            step=step_no,
                            operation=f"Revise ONLY `{sess.is_revision_of}` (JWT_TTL 60m->10m, RS256->ED25519)",
                            chatree_action=(
                                f"`commit_summary_with_materiality` v{updated_a.summary_version} "
                                f"(`severity={updated_a.last_change_severity}`) -> transitive BFS flagged "
                                f"B ({len(b_stale)} stale, hops={[d.hops for d in b_stale]}) & "
                                f"C ({len(c_stale)} stale, hops={[d.hops for d in c_stale]})"
                            ),
                            mem0_action=(
                                f"`mem.add()` added {len(m_added)} revision fact(s); "
                                f"0 dependency edges -> B (`CACHE_TTL=30m`) & C (`REFRESH_WINDOW=25m`) unflagged"
                            ),
                        )
                    )
                else:
                    imp_ids = [node_by_sess[s].id for s in sess.imports if s in node_by_sess]
                    br = await root.spawn(
                        goal=sess.title,
                        title=sess.title,
                        mode="clean",
                        imports=imp_ids or None,
                        switch_head=False,
                    )
                    node_by_sess[sess.session_id] = br
                    for turn in sess.turns:
                        await ct.repo.append_message(
                            node_id=br.id,
                            tree_id=tree.id,
                            role=turn.role,
                            content=f"[{turn.date_time}] {turn.content}",
                            mark_dirty=True,
                        )
                    t_m = await tree.model()
                    n_m = await br.model()
                    await ct.staleness.ensure_fresh_summary(
                        t_m, n_m, force=True, new_status="COMPLETED", session_date=sess.date_time
                    )
                    await ct.rag.index_node_chunks(br.id)
                    m_added = await asyncio.to_thread(mem0_h.add_session, sess)
                    imp_note = f" + `node_dependencies` edge from {sess.imports}" if sess.imports else ""
                    steps.append(
                        TraceStep(
                            step=step_no,
                            operation=f"Create `{sess.session_id}` ({sess.title})",
                            chatree_action=f"Spawned clean branch `{br.id}`{imp_note}; committed v1 summary",
                            mem0_action=f"`mem.add()` stored {len(m_added)} flat fact(s) (no causal links)",
                        )
                    )

            b_stale_final = await node_by_sess["branch_b_cache"].stale_dependencies()
            c_stale_final = await node_by_sess["branch_c_mobile"].stale_dependencies()
            stale_flagged = int(bool(b_stale_final)) + int(bool(c_stale_final))

            q = spec.queries[0]
            step_no += 1
            branch_c = node_by_sess["branch_c_mobile"]
            c_turn = await tree.chat(q.question, node_id=branch_c.id)
            c_ans_txt = c_turn.assistant_message.content if c_turn.assistant_message is not None else ""
            c_sc = score_probe_answer(q, c_ans_txt)

            # Mem0 top-4 semantic search on indirect mobile release query (no 'JWT'/'auth' keywords)
            m_hits = await asyncio.to_thread(mem0_h.search, q.question, 4)
            m_ev = "\n".join(f"- {h}" for h in m_hits)
            m_ans_txt = _TraceRecordingSimLLM._answer_probe2_from_context(m_ev)
            m_sc = score_probe_answer(q, m_ans_txt)
            m_found_rev = "10m" in m_ev.lower() or "ed25519" in m_ev.lower()

            steps.append(
                TraceStep(
                    step=step_no,
                    operation="Indirect release-safety query from Branch C (never mentions JWT/auth)",
                    chatree_action=(
                        f"Envelope injected `STALE [branch_a material 2h]` -> agent queried A -> "
                        f"{'PASS' if c_sc['passed'] else 'FAIL'} (`{c_ans_txt[:95]}`)"
                    ),
                    mem0_action=(
                        f"Top-4 vector search retrieved {len(m_hits)} fact(s) "
                        f"({'hit A v2' if m_found_rev else 'missed unlinked A v2'}) -> "
                        f"{'PASS' if m_sc['passed'] else 'FAIL'} (`{m_ans_txt[:95]}`)"
                    ),
                )
            )

            mem0_all = await asyncio.to_thread(mem0_h.get_all_memories)
            return ProbeTraceComparison(
                probe_id=spec.probe_id,
                title=spec.title,
                mem0_trace_backend=mem0_h.backend_name,
                chatree_passed=int(c_sc["passed"]) + (1 if stale_flagged == 2 else 0),
                chatree_total=2,  # 1: proactive structural stale graph (2/2), 2: indirect downstream query
                mem0_passed=int(m_sc["passed"]),
                mem0_total=2,
                chatree_extra_metrics={
                    "stale_dependents_flagged": stale_flagged,
                    "branch_b_hops": [d.hops for d in b_stale_final],
                    "branch_c_hops": [d.hops for d in c_stale_final],
                    "stale_notices_injected": c_turn.ephemeral_notices,
                },
                mem0_extra_metrics={
                    "stale_dependents_flagged": 0,
                    "mem0_http_calls": len(mem0_h.captured_calls),
                    "stored_memories_count": len(mem0_all),
                    "top_k_retrieved": m_hits,
                },
                steps=steps,
                query_outcomes=[
                    {
                        "query_id": "p2_structural_graph",
                        "question": "Proactive write-time detection of stale downstream branches B (1-hop) & C (2-hop)",
                        "chatree_answer": f"Flagged {stale_flagged}/2 (B hops={[d.hops for d in b_stale_final]}, C hops={[d.hops for d in c_stale_final]})",
                        "chatree_passed": stale_flagged == 2,
                        "mem0_answer": "0/2 (no causal dependency graph in flat vector store)",
                        "mem0_passed": False,
                    },
                    {
                        "query_id": q.query_id,
                        "question": q.question,
                        "chatree_answer": c_ans_txt,
                        "chatree_passed": c_sc["passed"],
                        "mem0_retrieved": m_hits,
                        "mem0_answer": m_ans_txt,
                        "mem0_passed": m_sc["passed"],
                    },
                ],
            )
    finally:
        mem0_h.close()


async def run_zero_token_simulation_proof(
    *,
    num_prefix_turns: int = 16,
    model: str = "deepseek/deepseek-chat",
    prefer_real_mem0_proxy: bool = True,
    prefer_fastembed_materiality: bool | None = None,
) -> ZeroTokenSimulationReport:
    """
    Execute the complete Part A zero-token structural simulation & framework execution trace
    (0 external API tokens, $0.00).
    """
    prefix_rep = await simulate_prefix_kv_cache_stability(num_turns=num_prefix_turns)
    scaling_rep = simulate_write_vs_query_scaling(model=model)

    p1 = await _trace_probe1_or_3_zero_token(
        build_probe1_verbatim_recovery(), prefer_real_mem0_proxy=prefer_real_mem0_proxy
    )
    p2 = await _trace_probe2_zero_token(
        build_probe2_transitive_staleness(), prefer_real_mem0_proxy=prefer_real_mem0_proxy
    )
    p3 = await _trace_probe1_or_3_zero_token(
        build_probe3_silent_quarantine(), prefer_real_mem0_proxy=prefer_real_mem0_proxy
    )
    use_fastembed = (
        prefer_real_mem0_proxy
        if prefer_fastembed_materiality is None
        else prefer_fastembed_materiality
    )
    mat_rep = await evaluate_materiality_classifier(prefer_fastembed=use_fastembed)
    return ZeroTokenSimulationReport(
        mem0_trace_backend=p1.mem0_trace_backend,
        prefix_stability=prefix_rep,
        scaling=scaling_rep,
        probe_traces=[p1, p2, p3],
        materiality_eval=mat_rep,
    )


# =============================================================================
# Part B — Ultra-Lean Live Head-to-Head Micro-Diagnostic Proof
# =============================================================================


class LiveProbeQueryOutcome(BaseModel):
    probe_id: str
    query_id: str
    system_name: str
    question: str
    answer: str
    passed: bool
    matched_groups: list[str] = Field(default_factory=list)
    missed_groups: list[list[str]] = Field(default_factory=list)
    leaked_tokens: list[str] = Field(default_factory=list)
    warning_ok: bool = True
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiveProbeSystemSummary(BaseModel):
    probe_id: str
    probe_title: str
    system_name: str
    passed: int
    total: int
    accuracy: float
    leakage_count: int = 0
    structural_stale_detected: int | None = None
    structural_stale_total: int | None = None
    query_tokens_in: int = 0
    query_tokens_out: int = 0
    query_cache_read_tokens: int = 0
    ingest_tokens_in: int = 0
    ingest_tokens_out: int = 0
    query_usd: float = 0.0
    ingest_usd: float = 0.0
    total_usd: float = 0.0
    ingest_seconds: float = 0.0
    mean_latency_ms: float = 0.0
    stored_state_preview: list[str] = Field(default_factory=list)


class LiveMicroDiagnosticReport(BaseModel):
    proof_type: Literal["live_head_to_head_micro_diagnostics"] = "live_head_to_head_micro_diagnostics"
    model: str
    embed_model: str
    reranker_name: str
    summaries: list[LiveProbeSystemSummary]
    outcomes: list[LiveProbeQueryOutcome]
    total_wall_seconds: float = 0.0

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"# Track 8B: Live Head-to-Head Micro-Diagnostic Proof (`{self.model}`)",
            "",
            (
                f"> Real unmodified `Chatree` vs. real unmodified `mem0ai` (`Mem0OfficialAdapter`) "
                f"on Probes 1-3 (`embed={self.embed_model}`, `reranker={self.reranker_name}`, "
                f"wall time `{self.total_wall_seconds:.1f}s`)."
            ),
            "",
            "| Probe | System | Score | Leakage | Structural Stale | Query / Ingest / Total USD | Ingest s | Mean Query ms | KV Cache % |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for s in self.summaries:
            stale_str = (
                f"{s.structural_stale_detected}/{s.structural_stale_total}"
                if s.structural_stale_total is not None
                else "—"
            )
            cache_pct = (
                (s.query_cache_read_tokens / max(1, s.query_tokens_in)) * 100.0
                if s.query_tokens_in > 0
                else 0.0
            )
            lines.append(
                f"| {s.probe_title} | `{s.system_name}` | **{s.passed}/{s.total} ({s.accuracy * 100:.0f}%)** "
                f"| {s.leakage_count} | {stale_str} "
                f"| ${s.query_usd:.4f} / ${s.ingest_usd:.4f} / **${s.total_usd:.4f}** "
                f"| {s.ingest_seconds:.1f}s | {s.mean_latency_ms:.0f} | {cache_pct:.1f}% |"
            )

        # Overall rollup across all 3 probes
        by_sys: dict[str, list[LiveProbeSystemSummary]] = {}
        for s in self.summaries:
            by_sys.setdefault(s.system_name, []).append(s)
        lines.extend(
            [
                "",
                "## Overall Micro-Diagnostic Rollup (Probes 1-3 Combined)",
                "",
                "| System | Combined Score | Dead-End Leaks | Total USD (Query + Ingest) | Total Ingest s |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for sys_name, rows in by_sys.items():
            tp = sum(r.passed for r in rows)
            tt = sum(r.total for r in rows)
            lk = sum(r.leakage_count for r in rows)
            usd = sum(r.total_usd for r in rows)
            ing = sum(r.ingest_seconds for r in rows)
            lines.append(
                f"| `{sys_name}` | **{tp}/{tt} ({(tp / max(1, tt)) * 100:.1f}%)** "
                f"| {lk} | **${usd:.4f}** | {ing:.1f}s |"
            )

        lines.extend(
            [
                "",
                (
                    "> *Cost accounting note:* `chatree_live` reports exact API token usage from the provider. "
                    "`mem0_official` ingest USD reflects `Mem0OfficialAdapter`'s conservative lower-bound estimator "
                    "(`chars // 4 + 350`) because upstream `mem0ai.add()` does not return prompt token counts; "
                    "as verified in Track 8A, upstream `mem0ai==2.2.1` actually sends a `33,653`-char (~8,413-token) "
                    "`ADDITIVE_EXTRACTION_PROMPT` on every `add()` call."
                ),
            ]
        )

        lines.extend(["", "## Per-Query Verbatim Evidence & Stored State Audit", ""])
        for probe_id in dict.fromkeys(s.probe_id for s in self.summaries):
            p_rows = [s for s in self.summaries if s.probe_id == probe_id]
            lines.append(f"### {p_rows[0].probe_title}")
            lines.append("")
            for s in p_rows:
                lines.append(f"- **`{s.system_name}` stored state ({len(s.stored_state_preview)} items):**")
                for item in s.stored_state_preview[:10]:
                    lines.append(f"  - {item}")
            lines.append("")
            lines.append("| Query | System | Result | Answer Excerpt | Notes |")
            lines.append("|---|---|---|---|---|")
            for o in self.outcomes:
                if o.probe_id != probe_id:
                    continue
                res = "PASS" if o.passed else "FAIL"
                exc = " ".join(o.answer.split())[:160].replace("|", "\\|")
                notes_parts: list[str] = []
                if o.leaked_tokens:
                    notes_parts.append(f"LEAKED={o.leaked_tokens}")
                if o.missed_groups:
                    notes_parts.append(f"MISSED={[g[0] for g in o.missed_groups]}")
                if not o.warning_ok:
                    notes_parts.append("NO_STALE_WARNING")
                lines.append(
                    f"| `{o.query_id}` | `{o.system_name}` | **{res}** | {exc} | {', '.join(notes_parts) or 'OK'} |"
                )
            lines.append("")
        return "\n".join(lines)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "probe_id",
            "probe_title",
            "system_name",
            "passed",
            "total",
            "accuracy",
            "leakage_count",
            "structural_stale_detected",
            "structural_stale_total",
            "query_tokens_in",
            "query_tokens_out",
            "query_cache_read_tokens",
            "ingest_tokens_in",
            "ingest_tokens_out",
            "query_usd",
            "ingest_usd",
            "total_usd",
            "ingest_seconds",
            "mean_latency_ms",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.summaries:
                row = s.model_dump()
                row.pop("stored_state_preview", None)
                writer.writerow(row)


def _snapshot_mem0_memories(adapter: Mem0OfficialAdapter) -> list[str]:
    mem = getattr(adapter, "_mem", None)
    if mem is None:
        return []
    try:
        raw = mem.get_all(filters={"user_id": adapter._user_id})
    except Exception:
        try:
            raw = mem.get_all(user_id=adapter._user_id)
        except Exception:
            return []
    items = raw.get("results", raw) if isinstance(raw, dict) else raw
    return [
        str(it.get("memory", it)) if isinstance(it, dict) else str(it)
        for it in (items or [])
    ]


async def _run_live_probe1_or_3(
    spec: MicroDiagnosticProbeSpec,
    *,
    model: str,
    llm: LLMProvider,
    embedder: Embedder,
    reranker: Reranker | None,
    api_base: str | None,
    cost_model: CostModel,
) -> tuple[list[LiveProbeSystemSummary], list[LiveProbeQueryOutcome]]:
    summaries: list[LiveProbeSystemSummary] = []
    outcomes: list[LiveProbeQueryOutcome] = []

    chatree_ad = ChatreeSystemAdapter(
        name="chatree_live",
        model=model,
        shadow_model=model,
        summary_model=model,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        max_ask_targets=4,
        generate_session_summaries=True,
        include_branch_summary_map=True,
        enable_recursive_hop=True,
        ingest_concurrency=3,
    )
    mem0_ad = Mem0OfficialAdapter(
        name="mem0_official",
        model=model,
        llm=llm,
        api_base=api_base,
        top_k=8,
    )

    for ad in (chatree_ad, mem0_ad):
        try:
            t_ing0 = time.perf_counter()
            await ad.reset(spec.probe_id)
            if isinstance(ad, ChatreeSystemAdapter):
                await ad.ingest_conversation(spec.sessions)
            else:
                for sess in spec.sessions:
                    await ad.start_session(sess.session_id, title=sess.title, date_time=sess.date_time)
                    for turn in sess.turns:
                        await ad.ingest_turn(
                            sess.session_id, turn.role, turn.content, date_time=turn.date_time or sess.date_time
                        )
                    await ad.finish_session(sess.session_id)
            ingest_s = round(time.perf_counter() - t_ing0, 2)
            ing_in, ing_out = ad.get_ingest_usage()

            if isinstance(ad, ChatreeSystemAdapter):
                assert ad._tree is not None
                nodes = await ad._tree.nodes()
                state_preview = [
                    f"{n.title} [{n.status} v{n.summary_version}]: {(n.summary or '(no summary)')[:140]}"
                    for n in nodes
                    if n.parent_id is not None
                ]
            else:
                state_preview = await asyncio.to_thread(_snapshot_mem0_memories, ad)

            q_in = q_out = q_cache = passed = leaks = 0
            latencies: list[float] = []
            for q in spec.queries:
                ans = await ad.answer_query(q.question)
                sc = score_probe_answer(q, ans.answer)
                passed += int(sc["passed"])
                leaks += int(bool(sc["leaked_tokens"]))
                q_in += ans.tokens_in
                q_out += ans.tokens_out
                q_cache += ans.cache_read_tokens
                latencies.append(ans.latency_ms)
                outcomes.append(
                    LiveProbeQueryOutcome(
                        probe_id=spec.probe_id,
                        query_id=q.query_id,
                        system_name=ad.name,
                        question=q.question,
                        answer=ans.answer,
                        passed=bool(sc["passed"]),
                        matched_groups=list(sc["matched_groups"]),
                        missed_groups=list(sc["missed_groups"]),
                        leaked_tokens=list(sc["leaked_tokens"]),
                        warning_ok=bool(sc["warning_ok"]),
                        tokens_in=ans.tokens_in,
                        tokens_out=ans.tokens_out,
                        cache_read_tokens=ans.cache_read_tokens,
                        latency_ms=ans.latency_ms,
                        metadata=dict(ans.metadata),
                    )
                )

            q_usd = cost_model.estimate_usd(model, tokens_in=q_in, tokens_out=q_out, cache_read_tokens=q_cache)
            i_usd = cost_model.estimate_usd(model, tokens_in=ing_in, tokens_out=ing_out, cache_read_tokens=0)
            summaries.append(
                LiveProbeSystemSummary(
                    probe_id=spec.probe_id,
                    probe_title=spec.title,
                    system_name=ad.name,
                    passed=passed,
                    total=len(spec.queries),
                    accuracy=round(passed / max(1, len(spec.queries)), 4),
                    leakage_count=leaks,
                    query_tokens_in=q_in,
                    query_tokens_out=q_out,
                    query_cache_read_tokens=q_cache,
                    ingest_tokens_in=ing_in,
                    ingest_tokens_out=ing_out,
                    query_usd=round(q_usd, 6),
                    ingest_usd=round(i_usd, 6),
                    total_usd=round(q_usd + i_usd, 6),
                    ingest_seconds=ingest_s,
                    mean_latency_ms=round(sum(latencies) / max(1, len(latencies)), 1),
                    stored_state_preview=state_preview,
                )
            )
        finally:
            await ad.close()

    return summaries, outcomes


async def _run_live_probe2(
    spec: MicroDiagnosticProbeSpec,
    *,
    model: str,
    llm: LLMProvider,
    embedder: Embedder,
    reranker: Reranker | None,
    api_base: str | None,
    cost_model: CostModel,
) -> tuple[list[LiveProbeSystemSummary], list[LiveProbeQueryOutcome]]:
    summaries: list[LiveProbeSystemSummary] = []
    outcomes: list[LiveProbeQueryOutcome] = []
    q = spec.queries[0]

    # 1. Real interactive Chatree with `spawn(..., imports=...)` -> transitive BFS -> Branch C `tree.chat()`
    cfg = ChatreeConfig(
        db_path=":memory:",
        default_model=model,
        shadow_model=model,
        summary_model=model,
        auto_commit_handoffs=True,
        guardrails=GuardrailPolicy(auto_title_enabled=False),
    )
    t_ing0 = time.perf_counter()
    c_ing_in = c_ing_out = 0
    async with Chatree.in_memory(config=cfg, llm=llm, embedder=embedder, reranker=reranker) as ct:
        tree = await ct.create_tree("Probe 2 Live Transitive Staleness", default_model=model)
        root = await tree.root()
        node_by_sess: dict[str, Any] = {}
        for sess in spec.sessions:
            if sess.is_revision_of:
                target_h = node_by_sess[sess.is_revision_of]
                for turn in sess.turns:
                    await ct.repo.append_message(
                        node_id=target_h.id,
                        tree_id=tree.id,
                        role=turn.role,
                        content=f"[{turn.date_time}] {turn.content}",
                        mark_dirty=True,
                    )
                t_m = await tree.model()
                n_m = await target_h.model()
                _, _, tin, tout, _ = await ct.staleness.ensure_fresh_summary(
                    t_m, n_m, force=True, new_status="COMPLETED", session_date=sess.date_time
                )
                c_ing_in += tin
                c_ing_out += tout
            else:
                imp_ids = [node_by_sess[s].id for s in sess.imports if s in node_by_sess]
                br = await root.spawn(
                    goal=sess.title,
                    title=sess.title,
                    mode="clean",
                    imports=imp_ids or None,
                    switch_head=False,
                )
                node_by_sess[sess.session_id] = br
                for turn in sess.turns:
                    await ct.repo.append_message(
                        node_id=br.id,
                        tree_id=tree.id,
                        role=turn.role,
                        content=f"[{turn.date_time}] {turn.content}",
                        mark_dirty=True,
                    )
                t_m = await tree.model()
                n_m = await br.model()
                _, _, tin, tout, _ = await ct.staleness.ensure_fresh_summary(
                    t_m, n_m, force=True, new_status="COMPLETED", session_date=sess.date_time
                )
                await ct.rag.index_node_chunks(br.id)
                c_ing_in += tin
                c_ing_out += tout

        c_ingest_s = round(time.perf_counter() - t_ing0, 2)
        b_stale = await node_by_sess["branch_b_cache"].stale_dependencies()
        c_stale = await node_by_sess["branch_c_mobile"].stale_dependencies()
        stale_detected = int(bool(b_stale)) + int(bool(c_stale))

        nodes = await tree.nodes()
        c_state_preview = [
            f"{n.title} [v{n.summary_version}, stale={len(n.stale_dependencies)} "
            f"hops={[d.hops for d in n.stale_dependencies]}]: {(n.summary or '')[:130]}"
            for n in nodes
            if n.parent_id is not None
        ]

        m_before = await tree.metrics()
        t_q0 = time.perf_counter()
        c_turn = await tree.chat(q.question, node_id=node_by_sess["branch_c_mobile"].id)
        c_q_ms = round((time.perf_counter() - t_q0) * 1000.0, 1)
        m_after = await tree.metrics()

        c_q_in = max(0, m_after.total_tokens_in - m_before.total_tokens_in)
        c_q_out = max(0, m_after.total_tokens_out - m_before.total_tokens_out)
        c_q_cache = max(0, m_after.total_cache_read_tokens - m_before.total_cache_read_tokens)
        c_live_ans = c_turn.assistant_message.content if c_turn.assistant_message is not None else ""
        c_sc = score_probe_answer(q, c_live_ans)

        outcomes.append(
            LiveProbeQueryOutcome(
                probe_id=spec.probe_id,
                query_id="p2_structural_graph",
                system_name="chatree_live",
                question="Proactive write-time detection of stale downstream branches B (1-hop) & C (2-hop)",
                answer=f"Flagged {stale_detected}/2 downstream branches stale (B hops={[d.hops for d in b_stale]}, C hops={[d.hops for d in c_stale]}).",
                passed=(stale_detected == 2),
                tokens_in=0,
                tokens_out=0,
                latency_ms=0.0,
            )
        )
        outcomes.append(
            LiveProbeQueryOutcome(
                probe_id=spec.probe_id,
                query_id=q.query_id,
                system_name="chatree_live",
                question=q.question,
                answer=c_live_ans,
                passed=bool(c_sc["passed"]),
                matched_groups=list(c_sc["matched_groups"]),
                missed_groups=list(sc_g for sc_g in c_sc["missed_groups"]),
                leaked_tokens=list(c_sc["leaked_tokens"]),
                warning_ok=bool(c_sc["warning_ok"]),
                tokens_in=c_q_in,
                tokens_out=c_q_out,
                cache_read_tokens=c_q_cache,
                latency_ms=c_q_ms,
                metadata={"staleness_notices": c_turn.ephemeral_notices},
            )
        )
        c_passed = int(stale_detected == 2) + int(bool(c_sc["passed"]))
        c_q_usd = cost_model.estimate_usd(model, tokens_in=c_q_in, tokens_out=c_q_out, cache_read_tokens=c_q_cache)
        c_i_usd = cost_model.estimate_usd(model, tokens_in=c_ing_in, tokens_out=c_ing_out, cache_read_tokens=0)
        summaries.append(
            LiveProbeSystemSummary(
                probe_id=spec.probe_id,
                probe_title=spec.title,
                system_name="chatree_live",
                passed=c_passed,
                total=2,
                accuracy=round(c_passed / 2.0, 4),
                structural_stale_detected=stale_detected,
                structural_stale_total=2,
                query_tokens_in=c_q_in,
                query_tokens_out=c_q_out,
                query_cache_read_tokens=c_q_cache,
                ingest_tokens_in=c_ing_in,
                ingest_tokens_out=c_ing_out,
                query_usd=round(c_q_usd, 6),
                ingest_usd=round(c_i_usd, 6),
                total_usd=round(c_q_usd + c_i_usd, 6),
                ingest_seconds=c_ingest_s,
                mean_latency_ms=c_q_ms,
                stored_state_preview=c_state_preview,
            )
        )

    # 2. Real unmodified Mem0 (`mem0ai`) ingesting the exact same 4 sessions
    mem0_ad = Mem0OfficialAdapter(name="mem0_official", model=model, llm=llm, api_base=api_base, top_k=8)
    try:
        t_m0 = time.perf_counter()
        await mem0_ad.reset(spec.probe_id)
        for sess in spec.sessions:
            await mem0_ad.start_session(sess.session_id, title=sess.title, date_time=sess.date_time)
            for turn in sess.turns:
                await mem0_ad.ingest_turn(
                    sess.session_id, turn.role, turn.content, date_time=turn.date_time or sess.date_time
                )
            await mem0_ad.finish_session(sess.session_id)
        m_ingest_s = round(time.perf_counter() - t_m0, 2)
        m_ing_in, m_ing_out = mem0_ad.get_ingest_usage()
        m_state_preview = await asyncio.to_thread(_snapshot_mem0_memories, mem0_ad)

        m_ans = await mem0_ad.answer_query(q.question)
        m_sc = score_probe_answer(q, m_ans.answer)

        outcomes.append(
            LiveProbeQueryOutcome(
                probe_id=spec.probe_id,
                query_id="p2_structural_graph",
                system_name="mem0_official",
                question="Proactive write-time detection of stale downstream branches B (1-hop) & C (2-hop)",
                answer="0/2 — Mem0 flat vector store maintains no causal dependency edges or stale flags.",
                passed=False,
                tokens_in=0,
                tokens_out=0,
                latency_ms=0.0,
            )
        )
        outcomes.append(
            LiveProbeQueryOutcome(
                probe_id=spec.probe_id,
                query_id=q.query_id,
                system_name="mem0_official",
                question=q.question,
                answer=m_ans.answer,
                passed=bool(m_sc["passed"]),
                matched_groups=list(m_sc["matched_groups"]),
                missed_groups=list(m_sc["missed_groups"]),
                leaked_tokens=list(m_sc["leaked_tokens"]),
                warning_ok=bool(m_sc["warning_ok"]),
                tokens_in=m_ans.tokens_in,
                tokens_out=m_ans.tokens_out,
                cache_read_tokens=m_ans.cache_read_tokens,
                latency_ms=m_ans.latency_ms,
                metadata=dict(m_ans.metadata),
            )
        )
        m_passed = int(bool(m_sc["passed"]))
        m_q_usd = cost_model.estimate_usd(
            model, tokens_in=m_ans.tokens_in, tokens_out=m_ans.tokens_out, cache_read_tokens=m_ans.cache_read_tokens
        )
        m_i_usd = cost_model.estimate_usd(model, tokens_in=m_ing_in, tokens_out=m_ing_out, cache_read_tokens=0)
        summaries.append(
            LiveProbeSystemSummary(
                probe_id=spec.probe_id,
                probe_title=spec.title,
                system_name="mem0_official",
                passed=m_passed,
                total=2,
                accuracy=round(m_passed / 2.0, 4),
                structural_stale_detected=0,
                structural_stale_total=2,
                query_tokens_in=m_ans.tokens_in,
                query_tokens_out=m_ans.tokens_out,
                query_cache_read_tokens=m_ans.cache_read_tokens,
                ingest_tokens_in=m_ing_in,
                ingest_tokens_out=m_ing_out,
                query_usd=round(m_q_usd, 6),
                ingest_usd=round(m_i_usd, 6),
                total_usd=round(m_q_usd + m_i_usd, 6),
                ingest_seconds=m_ingest_s,
                mean_latency_ms=m_ans.latency_ms,
                stored_state_preview=m_state_preview,
            )
        )
    finally:
        await mem0_ad.close()

    return summaries, outcomes


async def run_live_micro_diagnostics(
    *,
    model: str,
    llm: LLMProvider,
    embedder: Embedder,
    reranker: Reranker | None = None,
    embed_model: str = "local/BAAI/bge-small-en-v1.5",
    reranker_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
    api_base: str | None = None,
) -> LiveMicroDiagnosticReport:
    """
    Execute Part B: Ultra-lean live head-to-head micro-diagnostic proof (Probes 1-3)
    comparing real `Chatree` against real unmodified `mem0ai` (`Mem0OfficialAdapter`).
    """
    t0 = time.perf_counter()
    cost_model = CostModel()
    all_summaries: list[LiveProbeSystemSummary] = []
    all_outcomes: list[LiveProbeQueryOutcome] = []

    s1, o1 = await _run_live_probe1_or_3(
        build_probe1_verbatim_recovery(),
        model=model,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        api_base=api_base,
        cost_model=cost_model,
    )
    all_summaries.extend(s1)
    all_outcomes.extend(o1)

    s2, o2 = await _run_live_probe2(
        build_probe2_transitive_staleness(),
        model=model,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        api_base=api_base,
        cost_model=cost_model,
    )
    all_summaries.extend(s2)
    all_outcomes.extend(o2)

    s3, o3 = await _run_live_probe1_or_3(
        build_probe3_silent_quarantine(),
        model=model,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        api_base=api_base,
        cost_model=cost_model,
    )
    all_summaries.extend(s3)
    all_outcomes.extend(o3)

    return LiveMicroDiagnosticReport(
        model=model,
        embed_model=embed_model,
        reranker_name=reranker_name if reranker is not None else "none",
        summaries=all_summaries,
        outcomes=all_outcomes,
        total_wall_seconds=round(time.perf_counter() - t0, 1),
    )
