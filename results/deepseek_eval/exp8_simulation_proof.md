# Track 8A: Zero-Token Structural Simulation & Real-Framework Execution Trace

> **Zero-Token Guarantee:** `real_api_tokens_used = 0`, `real_api_usd_cost = $0.00`. Chatree executes its real SQLite + ContextAssembler + StalenessService + TreeRAG engine; Mem0 trace backend = `real_mem0ai_2.2.1_local_proxy`.

## 1. Prompt Prefix KV-Cache Stability (Exact 3-Way LCP Trace over 16 Turns)

All three layouts process the identical 16-turn multi-topic engineering dialogue with the exact same base system prompt. Chatree appends its dynamic `<dialogue_tree_context>` block **ephemerally at the tail** of the newest user turn, keeping both the system prompt and all prior turns immutable. We compare against two Mem0 system-prompt injection layouts: (a) **End-of-System Injection** (appending retrieved memories after the static system prompt, which preserves the system prompt in cache but invalidates all prior turns on every top-k change), and (b) **Byte-0 Prepend** (prepending memories before the system prompt). *Note: A Mem0 integration that injects memories inside the newest user turn (tail injection) can match static-turn cacheability; this simulation evaluates system-prompt injection layouts.*

| Metric | Chatree (Tail-Only Envelope) | Mem0 (End-of-System Injection) | Mem0 (Byte-0 Prepend) |
|---|---:|---:|---:|
| Mean Prefix Cacheability `LCP(t, t-1) / |P_t|` (Turns 2-16) | **82.5%** | 61.0% | 0.7% |
| Raw Input Tokens (16 turns) | 21,052 | 19,780 | 19,780 |
| Effective Billed Input Tokens (`0.1x` cached read) | **6,018** | 9,750 | 19,658 |
| Effective Input Token Reduction (Chatree vs. Layout) | baseline | **-38.3%** | **-69.4%** |

<details><summary>Turn-by-Turn Exact LCP Trace (Turns 1-16)</summary>

| Turn | Chatree Tok | Chatree Cache % | Mem0 End-of-Sys Cache % | Mem0 Byte-0 Cache % |
|---:|---:|---:|---:|---:|
| 1 | 940 | 0.0% | 0.0% | 0.0% |
| 2 | 994 | 74.1% | 84.2% | 1.0% |
| 3 | 1018 | 77.7% | 79.5% | 1.0% |
| 4 | 1071 | 78.9% | 75.1% | 0.9% |
| 5 | 1125 | 79.9% | 70.9% | 0.9% |
| 6 | 1174 | 81.1% | 67.1% | 0.8% |
| 7 | 1233 | 81.6% | 63.9% | 0.8% |
| 8 | 1286 | 82.4% | 61.2% | 0.7% |
| 9 | 1340 | 83.1% | 58.8% | 0.7% |
| 10 | 1389 | 84.0% | 56.6% | 0.7% |
| 11 | 1448 | 84.3% | 54.4% | 0.7% |
| 12 | 1501 | 84.9% | 52.4% | 0.6% |
| 13 | 1555 | 85.5% | 50.3% | 0.6% |
| 14 | 1604 | 86.2% | 48.3% | 0.6% |
| 15 | 1663 | 86.4% | 46.7% | 0.6% |
| 16 | 1711 | 87.1% | 45.2% | 0.5% |

</details>

## 2. Write-Time vs. Query-Time LLM Call & Cost Scaling

- **Measured upstream `mem0ai` extraction system prompt:** `33,653` chars (~`8,413` tokens; source: `live_import:mem0ai==2.2.1`) + `5,310`-char update prompt on reconciliation.
- **Chatree write path:** `0` extra LLM calls per live chat turn; `1` cold-trimmed summary call capped at `7,500` chars (~1,875 tok, 0 tool schemas) on branch finish.
- **Empirical LongMemEval-S anchor (`n=30`, `deepseek/deepseek-chat`):** Mem0 `$1.5460` ingest (character-heuristic estimate prior to v0.2.0 client metering) + `$0.0088` query = **`$1.5548`** (`176x` write tax); Chatree `$0.892` ingest + `$0.163` query = **`$1.055`** (`32%` lower total USD).

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

## 3. Controlled Lossy-Distillation Stress Traces (Probes 1-3)

To isolate pure architectural recovery guarantees under distillation loss, **both frameworks receive the exact same deliberately lossy distillation** (high-level decisions kept; incidental low-salience constants omitted). Note the honest contrast with live Track 8B on the short 12-turn Probe 1 dialogue, where DeepSeek+Mem0 emitted 13 granular facts and passed 4/4 — demonstrating that flat distilled stores rely on extraction verbosity rather than a structural verbatim fallback.

| Probe | Chatree Pass | Mem0 Pass | Key Structural Mechanism |
|---|---:|---:|---|
| Probe 1: Un-Anchored Verbatim Detail Recovery | **4/4** | 0/4 | Verbatim SQLite log + FTS5 + read-only lookup recovers omitted constants |
| Probe 2: Transitive Downstream Staleness (A -> B -> C) | **2/2** | 0/2 | `node_dependencies` BFS flags B (1h) + C (2h) STALE (2/2 vs. 0/2) |
| Probe 3: Silent Dead-End Quarantine (No Cancellation Essay) | **2/2** | 0/2 | Native `status='ARCHIVED'` excludes dead-end from all search levels (0% leakage) |

### Probe 1: Un-Anchored Verbatim Detail Recovery

| Step | Operation | Chatree State & Execution | Mem0 State & Execution |
|---:|---|---|---|
| 1 | Ingest `incident_409_gateway` (12 turns) | Persisted 12 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (264 chars) | `mem.add()` extracted 3 distilled fact(s) into Qdrant; raw transcript discarded |
| 2 | Ingest `frontend_checkout_ui` (4 turns) | Persisted 4 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (81 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 3 | Ingest `billing_webhook_retries` (4 turns) | Persisted 4 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (85 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 4 | Query `p1_q1_timeout` | PASS via targets=['n_1abe8d01', 'n_5bf1e0c9', 'n_8c57acb5'] -> `The legacy nginx ingress snippet specifies `proxy_read_timeout 43s`.` | FAIL via 4 retrieved fact(s) -> `Not found in the available evidence.` |
| 5 | Query `p1_q2_retry_epoch` | PASS via targets=['n_1abe8d01', 'n_5bf1e0c9', 'n_8c57acb5'] -> `The captured response header was `X-Retry-Epoch: 17`.` | FAIL via 4 retrieved fact(s) -> `Not found in the available evidence.` |
| 6 | Query `p1_q3_error_code` | PASS via targets=['n_1abe8d01', 'n_5bf1e0c9', 'n_8c57acb5'] -> `The coordinator `renew_failed` log reported error code `ERR_SHARD_LEASE_419`.` | FAIL via 4 retrieved fact(s) -> `Not found in the available evidence.` |
| 7 | Query `p1_q4_index_name` | PASS via targets=['n_1abe8d01', 'n_5bf1e0c9', 'n_8c57acb5'] -> `The `EXPLAIN ANALYZE` plan scanned PostgreSQL index `idx_ledger_tenant_ts_v3`.` | FAIL via 4 retrieved fact(s) -> `Not found in the available evidence.` |

- **Chatree metrics:** `{"shadow_turns_executed": 12, "summarizer_calls": 3}`
- **Mem0 metrics:** `{"mem0_http_calls": 3, "mem0_max_system_prompt_chars": 33653, "stored_memories": ["Checkout UI redesign uses React 19, TanStack Query, Tailwind v4, and WCAG 2.2 AA.", "Headline decision: switch internal service-to-service hops from REST/JSON to gRPC mTLS.", "Headline decision: migrate edge payment gateway to Envoy 1.31 across all regions.", "Billing webhooks use 6-attempt 24h exponential backoff and HMAC-SHA256 X-Billing-Sig.", "Headline decision: replace legacy Redis 6 with Valkey 8.0 for rate-limit and idempotency keys."], "stored_memories_count": 5}`

### Probe 2: Transitive Downstream Staleness (A -> B -> C)

| Step | Operation | Chatree State & Execution | Mem0 State & Execution |
|---:|---|---|---|
| 1 | Create `branch_a_auth` (Auth Token Spec) | Spawned clean branch `n_933926d5`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 2 | Create `branch_b_cache` (Edge Cache Policy) | Spawned clean branch `n_2cc7e3e8` + `node_dependencies` edge from ['branch_a_auth']; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 3 | Create `branch_c_mobile` (Mobile Refresh Flow) | Spawned clean branch `n_e6751fe2` + `node_dependencies` edge from ['branch_b_cache']; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 4 | Create `mobile_release_checklist` (Mobile Release Cut Checklist) | Spawned clean branch `n_add86a32`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 5 | Create `ios_bg_sync_scheduler` (iOS Background Refresh Scheduler) | Spawned clean branch `n_ce1974b9`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 6 | Create `android_workmanager_sync` (Android WorkManager Refresh Policy) | Spawned clean branch `n_80ccad3a`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 7 | Create `cdn_edge_cache_purge` (CDN Edge Cache Purge and Vary Rules) | Spawned clean branch `n_a47ee261`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 8 | Create `mobile_crashlytics_slo` (Mobile Release Stability Gate) | Spawned clean branch `n_790fd3cc`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 9 | Create `edge_cache_swr_tuning` (Edge Cache Stale-While-Revalidate Policy) | Spawned clean branch `n_fea2ccd3`; committed v1 summary | `mem.add()` stored 1 flat fact(s) (no causal links) |
| 10 | Revise ONLY `branch_a_auth` (JWT_TTL 60m->10m, RS256->ED25519) | `commit_summary_with_materiality` v2 (`severity=material`) -> transitive BFS flagged B (1 stale, hops=[1]) & C (1 stale, hops=[2]) | `mem.add()` added 1 revision fact(s); 0 dependency edges -> B (`CACHE_TTL=30m`) & C (`REFRESH_WINDOW=25m`) unflagged |
| 11 | Indirect release-safety query from Branch C (never mentions JWT/auth) | Envelope injected `STALE [branch_a material 2h]` -> agent queried A -> PASS (`NOT SAFE to ship as-is — upstream `Auth Token Spec` was materially revised to `JWT_TTL=10m` and`) | Top-4 vector search retrieved 4 fact(s) (missed unlinked A v2) -> FAIL (`Settings appear consistent and safe to ship as-is: `CACHE_TTL=30m` satisfies the half-of-JWT_TT`) |

- **Chatree metrics:** `{"branch_b_hops": [1], "branch_c_hops": [2], "stale_dependents_flagged": 2, "stale_notices_injected": ["Reopened completed branch n_e6751fe2 (Mobile Refresh Flow) for revision.", "n_933926d5 \"Auth Token Spec\" material 2h | Transitive upstream change via Edge Cache Policy: Summary updated (v1 -> v2) \u2014 removed [60m, RS256, v1]; added [10m, ED25519] [delta: removed [60m, RS256, v1]; added [10m, ED25519]]"]}`
- **Mem0 metrics:** `{"mem0_http_calls": 10, "stale_dependents_flagged": 0, "stored_memories_count": 10, "top_k_retrieved": ["Mobile Refresh Flow sets REFRESH_WINDOW=25m (5 minutes shorter than CACHE_TTL).", "Edge Cache Policy sets CACHE_TTL=30m (must stay <= half of JWT_TTL).", "Mobile release v4.8.0 checklist requires 24h TestFlight freeze and 10/50/100% Play Console staged rollout.", "Android mobile refresh uses WorkManager PeriodicWorkRequest with 15m flex window and 30s backoff."]}`

### Probe 3: Silent Dead-End Quarantine (No Cancellation Essay)

| Step | Operation | Chatree State & Execution | Mem0 State & Execution |
|---:|---|---|---|
| 1 | Ingest `main_storage_arch` (4 turns) | Persisted 4 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (148 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 2 | Ingest `explore_mongodb_spike` (4 turns) | Persisted 4 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (167 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 3 | Silent archive `explore_mongodb_spike` (NO cancellation essay) | Native `archive()` set `status='ARCHIVED'`; pruned from Tier 1/2/2.5/3 RAG (kept only in SQLite audit log) | No branch-archive primitive and no cancellation essay -> 1 dead-end fact(s) remain ACTIVE in Qdrant |
| 4 | Ingest `observability_metrics` (2 turns) | Persisted 2 verbatim turns in SQLite + FTS5; lossy Tier-2 summary (94 chars) | `mem.add()` extracted 1 distilled fact(s) into Qdrant; raw transcript discarded |
| 5 | Query `p3_q1_active_db` | PASS via targets=['n_75b38e5c', 'n_ac644a29'] -> `Active production ledger storage: PostgreSQL 16.4 with pgbouncer_pool_size=120 and isolation_le` | FAIL via 3 retrieved fact(s) -> `Active production ledger storage: PostgreSQL 16.4 with pgbouncer_pool_size=120 and isolation_le` |
| 6 | Query `p3_q2_dead_end_leakage` | PASS via targets=['n_75b38e5c', 'n_ac644a29'] -> `None — no MongoDB replica set, write concern, or WiredTiger cache is part of the active product` | FAIL via 3 retrieved fact(s) -> `Active MongoDB settings found in memory: replica set `rs_ledger_0`, `writeConcern=majority`, an` |

- **Chatree metrics:** `{"shadow_turns_executed": 4, "summarizer_calls": 3}`
- **Mem0 metrics:** `{"mem0_http_calls": 3, "mem0_max_system_prompt_chars": 33653, "stored_memories": ["Observability uses OpenTelemetry collector otel-gw-02, 15s Prometheus scrape, p99 < 180ms SLO.", "Ledger MongoDB 7.0 setup uses replica set rs_ledger_0, writeConcern=majority, shardKey=tenant_id_hashed, wiredTigerCacheSizeGB=24, and readPreference=primaryPreferred.", "Production ledger database uses PostgreSQL 16.4 in us-east-1 with pgbouncer_pool_size=120, isolation_level=serializable, and pgBackRest WAL backups."], "stored_memories_count": 3}`

## 4. Zero-Token Materiality Classifier Benchmark (20 Branch-Revision Pairs)

Evaluates `StalenessService.classify_change_materiality()` (`embedder=FastEmbedEmbedder(BAAI/bge-small-en-v1.5)`, `cosine_threshold=0.90`) on 20 paired branch summaries: 10 true material revisions (4 identifier/numeric, 3 polarity flips, 3 subtle non-alphanumeric semantic changes) and 10 true cosmetic rewordings.

| Metric | Score |
|---|---:|
| Precision (material) | **90.9%** (TP=10, FP=1) |
| Recall (material) | **100.0%** (TP=10, FN=0) |
| F1 (material) | **95.2%** |
| Overall Accuracy | **95.0%** (19/20) |

| Category | Expected | Correct / Total | Recall / Specificity |
|---|---|---:|---:|
| `identifier_numeric` | `material` | 4/4 | 100.0% |
| `polarity_flip` | `material` | 3/3 | 100.0% |
| `semantic_non_alphanumeric` | `material` | 3/3 | 100.0% |
| `cosmetic_rewording` | `cosmetic` | 9/10 | 90.0% |
