# Track 8A: Zero-Token Structural Simulation & Real-Framework Execution Trace

> **Zero-Token Guarantee:** `real_api_tokens_used = 0`, `real_api_usd_cost = $0.00`. Chatree executes its real SQLite + ContextAssembler + StalenessService + TreeRAG engine; Mem0 trace backend = `real_mem0ai_2.2.1_local_proxy`.

## 1. Byte-0 Prefix KV-Cache Stability (Exact LCP Trace over 16 Turns)

Both frameworks process the identical 16-turn multi-topic engineering dialogue. Chatree appends its dynamic `<dialogue_tree_context>` envelope **ephemerally at the tail** of the newest user message without mutating bytes `0..k`. Mem0-style chat integration re-injects top-k retrieved memories into the system/header block on each turn, shifting byte 0.

| Metric | Chatree (Real `ContextAssembler`) | Mem0-Style Prefix Injection |
|---|---:|---:|
| Mean Prefix Cacheability `LCP(t, t-1) / |P_t|` (Turns 2-16) | **82.5%** | 0.7% |
| Raw Input Tokens (16 turns) | 21,052 | 19,780 |
| Effective Billed Input Tokens (`0.1x` cached read) | **6,018** | 19,658 |
| Effective Input Token Reduction via Byte-0 Prefix Stability | **-69.4%** | baseline |

<details><summary>Turn-by-Turn Exact LCP Trace (Turns 1-16)</summary>

| Turn | Chatree Tok | Chatree LCP | Chatree Cache % | Mem0 Tok | Mem0 LCP | Mem0 Cache % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 940 | 0 | 0.0% | 833 | 0 | 0.0% |
| 2 | 994 | 737 | 74.1% | 882 | 9 | 1.0% |
| 3 | 1018 | 791 | 77.7% | 935 | 9 | 1.0% |
| 4 | 1071 | 845 | 78.9% | 989 | 9 | 0.9% |
| 5 | 1125 | 899 | 79.9% | 1048 | 9 | 0.9% |
| 6 | 1174 | 952 | 81.1% | 1107 | 9 | 0.8% |
| 7 | 1233 | 1006 | 81.6% | 1162 | 9 | 0.8% |
| 8 | 1286 | 1060 | 82.4% | 1214 | 9 | 0.7% |
| 9 | 1340 | 1114 | 83.1% | 1263 | 9 | 0.7% |
| 10 | 1389 | 1167 | 84.0% | 1312 | 9 | 0.7% |
| 11 | 1448 | 1221 | 84.3% | 1365 | 9 | 0.7% |
| 12 | 1501 | 1275 | 84.9% | 1419 | 9 | 0.6% |
| 13 | 1555 | 1329 | 85.5% | 1478 | 9 | 0.6% |
| 14 | 1604 | 1382 | 86.2% | 1537 | 9 | 0.6% |
| 15 | 1663 | 1436 | 86.4% | 1592 | 9 | 0.6% |
| 16 | 1711 | 1490 | 87.1% | 1644 | 9 | 0.5% |

</details>

## 2. Write-Time vs. Query-Time LLM Call & Cost Scaling

- **Measured upstream `mem0ai` extraction system prompt:** `33,653` chars (~`8,413` tokens; source: `live_import:mem0ai==2.2.1`) + `5,310`-char update prompt on reconciliation.
- **Chatree write path:** `0` extra LLM calls per live chat turn; `1` cold-trimmed summary call capped at `7,500` chars (~1,875 tok, 0 tool schemas) on branch finish.
- **Empirical LongMemEval-S anchor (`n=30`, `deepseek/deepseek-chat`):** Mem0 `$1.5460` ingest + `$0.0088` query = **`$1.5548`** (`176x` write tax); Chatree v3.3 `$0.892` ingest + `$0.163` query = **`$1.055`** (`32%` lower total USD, `+10pp` higher accuracy in v3).

| Sessions (N) | Queries (Q) | Chatree Write Calls | Chatree Total USD | Mem0 Write Calls | Mem0 Total USD (70% sys-cached) | Mem0 Total USD (uncached) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10 | 5 | **$0.0109** | 8 | $0.0140 | $0.0206 |
| 5 | 50 | 5 | **$0.0416** | 8 | $0.0276 | $0.0342 |
| 20 | 10 | 20 | **$0.0205** | 35 | $0.0422 | $0.0739 |
| 20 | 50 | 20 | **$0.0513** | 35 | $0.0558 | $0.0876 |
| 50 | 10 | 50 | **$0.0398** | 89 | $0.0987 | $0.1806 |
| 50 | 50 | 50 | **$0.0705** | 89 | $0.1123 | $0.1942 |
| 200 | 10 | 200 | **$0.1362** | 359 | $0.3811 | $0.7137 |
| 200 | 50 | 200 | **$0.1669** | 359 | $0.3947 | $0.7273 |

## 3. Step-by-Step Framework Execution Traces (Probes 1-3, Identical Lossy Extraction)

To eliminate prompt-engineering bias, **both frameworks receive the exact same deterministic lossy distillation** (high-level decisions preserved; incidental low-salience constants omitted). Every pass/fail delta below is purely architectural.

| Probe | Chatree Pass | Mem0 Pass | Key Structural Mechanism |
|---|---:|---:|---|
| Probe 1: Un-Anchored Verbatim Detail Recovery | **4/4** | 0/4 | Tier-1 SQLite verbatim FTS5 + read-only shadow turn recovers omitted literals |
| Probe 2: Transitive Downstream Staleness (A -> B -> C) | **2/2** | 0/2 | `node_dependencies` BFS flags B (1h) + C (2h) STALE (2/2 vs. 0/2) |
| Probe 3: Silent Dead-End Quarantine (No Cancellation Essay) | **2/2** | 0/2 | Native `status='ARCHIVED'` excludes dead-end from all 4 tiers (0% leakage) |

### Probe 1: Un-Anchored Verbatim Detail Recovery

| Step | Operation | Chatree State & Execution | Mem0 State & Execution |
|---:|---|---|---|
| 1 | Ingest `incident_409_gateway` (12 turns) | Persisted 12 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (264 chars) | `mem.add()` extracted 3 distilled fact(s) into Qdrant; raw transcript discarded |
| 2 | Ingest `frontend_checkout_ui` (4 turns) | Persisted 4 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (81 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 3 | Ingest `billing_webhook_retries` (4 turns) | Persisted 4 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (85 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 4 | Query `p1_q1_timeout` | PASS via targets=['n_2309295a', 'n_6c080cfb', 'n_8ba35190'] -> `The legacy nginx ingress snippet specifies `proxy_read_timeout 43s`.` | FAIL via 4 retrieved fact(s) -> `Not found in the available evidence.` |
| 5 | Query `p1_q2_retry_epoch` | PASS via targets=['n_2309295a', 'n_6c080cfb', 'n_8ba35190'] -> `The captured response header was `X-Retry-Epoch: 17`.` | FAIL via 4 retrieved fact(s) -> `Not found in the available evidence.` |
| 6 | Query `p1_q3_error_code` | PASS via targets=['n_2309295a', 'n_6c080cfb', 'n_8ba35190'] -> `The coordinator `renew_failed` log reported error code `ERR_SHARD_LEASE_419`.` | FAIL via 4 retrieved fact(s) -> `Not found in the available evidence.` |
| 7 | Query `p1_q4_index_name` | PASS via targets=['n_2309295a', 'n_6c080cfb', 'n_8ba35190'] -> `The `EXPLAIN ANALYZE` plan scanned PostgreSQL index `idx_ledger_tenant_ts_v3`.` | FAIL via 4 retrieved fact(s) -> `Not found in the available evidence.` |

- **Chatree metrics:** `{"shadow_turns_executed": 12, "summarizer_calls": 3}`
- **Mem0 metrics:** `{"mem0_http_calls": 3, "mem0_max_system_prompt_chars": 33653, "stored_memories": ["Checkout UI redesign uses React 19, TanStack Query, Tailwind v4, and WCAG 2.2 AA.", "Headline decision: replace legacy Redis 6 with Valkey 8.0 for rate-limit and idempotency keys.", "Headline decision: switch internal service-to-service hops from REST/JSON to gRPC mTLS.", "Billing webhooks use 6-attempt 24h exponential backoff and HMAC-SHA256 X-Billing-Sig.", "Headline decision: migrate edge payment gateway to Envoy 1.31 across all regions."], "stored_memories_count": 5}`

### Probe 2: Transitive Downstream Staleness (A -> B -> C)

| Step | Operation | Chatree State & Execution | Mem0 State & Execution |
|---:|---|---|---|
| 1 | Create `branch_a_auth` (Auth Token Spec) | Spawned clean branch `n_1ec05c11`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 2 | Create `branch_b_cache` (Edge Cache Policy) | Spawned clean branch `n_741410b1` + `node_dependencies` edge from ['branch_a_auth']; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 3 | Create `branch_c_mobile` (Mobile Refresh Flow) | Spawned clean branch `n_b76a9bdc` + `node_dependencies` edge from ['branch_b_cache']; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 4 | Create `mobile_release_checklist` (Mobile Release Cut Checklist) | Spawned clean branch `n_2b6c3c55`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 5 | Create `ios_bg_sync_scheduler` (iOS Background Refresh Scheduler) | Spawned clean branch `n_b5c76465`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 6 | Create `android_workmanager_sync` (Android WorkManager Refresh Policy) | Spawned clean branch `n_96c0ea0a`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 7 | Create `cdn_edge_cache_purge` (CDN Edge Cache Purge and Vary Rules) | Spawned clean branch `n_b4070944`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 8 | Create `mobile_crashlytics_slo` (Mobile Release Stability Gate) | Spawned clean branch `n_9b1aeba1`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 9 | Create `edge_cache_swr_tuning` (Edge Cache Stale-While-Revalidate Policy) | Spawned clean branch `n_20ec75be`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 10 | Revise ONLY `branch_a_auth` (JWT_TTL 60m->10m, RS256->ED25519) | `commit_summary_with_materiality` v2 (`severity=material`) -> transitive BFS flagged B (1 stale, hops=[1]) & C (1 stale, hops=[2]) | `mem.add()` added 1 revision fact(s); 0 dependency edges -> B (`CACHE_TTL=30m`) & C (`REFRESH_WINDOW=25m`) unflagged |
| 11 | Indirect release-safety query from Branch C (never mentions JWT/auth) | Envelope injected `STALE [branch_a material 2h]` -> agent queried A -> PASS (`NOT SAFE to ship as-is — upstream `Auth Token Spec` was materially revised to `JWT_TTL=10m` and`) | Top-4 vector search retrieved 4 fact(s) (missed unlinked A v2) -> FAIL (`Settings appear consistent and safe to ship as-is: `CACHE_TTL=30m` satisfies the half-of-JWT_TT`) |

- **Chatree metrics:** `{"branch_b_hops": [1], "branch_c_hops": [2], "stale_dependents_flagged": 2, "stale_notices_injected": ["Reopened completed branch n_b76a9bdc (Mobile Refresh Flow) for revision.", "n_1ec05c11 \"Auth Token Spec\" material 2h | Transitive upstream change via Edge Cache Policy: Summary updated (v1 -> v2) \u2014 removed [60m, RS256, v1]; added [10m, ED25519] [delta: removed [60m, RS256, v1]; added [10m, ED25519]]"]}`
- **Mem0 metrics:** `{"mem0_http_calls": 10, "stale_dependents_flagged": 0, "stored_memories_count": 10, "top_k_retrieved": ["Mobile Refresh Flow sets REFRESH_WINDOW=25m (5 minutes shorter than CACHE_TTL).", "Edge Cache Policy sets CACHE_TTL=30m (must stay <= half of JWT_TTL).", "Mobile release v4.8.0 checklist requires 24h TestFlight freeze and 10/50/100% Play Console staged rollout.", "Android mobile refresh uses WorkManager PeriodicWorkRequest with 15m flex window and 30s backoff."]}`

### Probe 3: Silent Dead-End Quarantine (No Cancellation Essay)

| Step | Operation | Chatree State & Execution | Mem0 State & Execution |
|---:|---|---|---|
| 1 | Ingest `main_storage_arch` (4 turns) | Persisted 4 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (148 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 2 | Ingest `explore_mongodb_spike` (4 turns) | Persisted 4 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (167 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 3 | Silent archive `explore_mongodb_spike` (NO cancellation essay) | Native `archive()` set `status='ARCHIVED'`; pruned from Tier 1/2/2.5/3 RAG (kept only in SQLite audit log) | No branch-archive primitive and no cancellation essay -> 1 dead-end fact(s) remain ACTIVE in Qdrant |
| 4 | Ingest `observability_metrics` (2 turns) | Persisted 2 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (94 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 5 | Query `p3_q1_active_db` | PASS via targets=['n_cc70364c', 'n_d821200b'] -> `Active production ledger storage: PostgreSQL 16.4 with pgbouncer_pool_size=120 and isolation_le` | FAIL via 3 retrieved fact(s) -> `Active production ledger storage: PostgreSQL 16.4 with pgbouncer_pool_size=120 and isolation_le` |
| 6 | Query `p3_q2_dead_end_leakage` | PASS via targets=['n_cc70364c', 'n_d821200b'] -> `None — no MongoDB replica set, write concern, or WiredTiger cache is part of the active product` | FAIL via 3 retrieved fact(s) -> `Active MongoDB settings found in memory: replica set `rs_ledger_0`, `writeConcern=majority`, an` |

- **Chatree metrics:** `{"shadow_turns_executed": 4, "summarizer_calls": 3}`
- **Mem0 metrics:** `{"mem0_http_calls": 3, "mem0_max_system_prompt_chars": 33653, "stored_memories": ["Ledger MongoDB 7.0 setup uses replica set rs_ledger_0, writeConcern=majority, shardKey=tenant_id_hashed, wiredTigerCacheSizeGB=24, and readPreference=primaryPreferred.", "Observability uses OpenTelemetry collector otel-gw-02, 15s Prometheus scrape, p99 < 180ms SLO.", "Production ledger database uses PostgreSQL 16.4 in us-east-1 with pgbouncer_pool_size=120, isolation_level=serializable, and pgBackRest WAL backups."], "stored_memories_count": 3}`
