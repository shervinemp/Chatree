# Track 8B: Live Head-to-Head Micro-Diagnostic Proof (`deepseek/deepseek-chat`)

> Real unmodified `Chatree` vs. real unmodified `mem0ai` (`Mem0OfficialAdapter`) on Probes 1-3 (`embed=local/BAAI/bge-small-en-v1.5`, `reranker=Xenova/ms-marco-MiniLM-L-6-v2`, wall time `78.8s`).

| Probe | System | Score | Leakage | Structural Stale | Query / Ingest / Total USD | Ingest s | Mean Query ms | KV Cache % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Probe 1: Un-Anchored Verbatim Detail Recovery | `chatree_live` | **4/4 (100%)** | 0 | — | $0.0037 / $0.0009 / **$0.0046** | 7.7s | 2498 | 81.9% |
| Probe 1: Un-Anchored Verbatim Detail Recovery | `mem0_official` | **4/4 (100%)** | 0 | — | $0.0005 / $0.0008 / **$0.0013** | 5.4s | 881 | 0.0% |
| Probe 2: Transitive Downstream Staleness (A -> B -> C) | `chatree_live` | **2/2 (100%)** | 0 | 2/2 | $0.0018 / $0.0021 / **$0.0039** | 16.0s | 6036 | 78.7% |
| Probe 2: Transitive Downstream Staleness (A -> B -> C) | `mem0_official` | **1/2 (50%)** | 0 | 0/2 | $0.0003 / $0.0015 / **$0.0018** | 12.3s | 2958 | 0.0% |
| Probe 3: Silent Dead-End Quarantine (No Cancellation Essay) | `chatree_live` | **2/2 (100%)** | 0 | — | $0.0013 / $0.0006 / **$0.0019** | 2.0s | 2681 | 81.7% |
| Probe 3: Silent Dead-End Quarantine (No Cancellation Essay) | `mem0_official` | **1/2 (50%)** | 1 | — | $0.0003 / $0.0005 / **$0.0008** | 4.8s | 1339 | 0.0% |

## Overall Micro-Diagnostic Rollup (Probes 1-3 Combined)

| System | Combined Score | Dead-End Leaks | Total USD (Query + Ingest) | Total Ingest s |
|---|---:|---:|---:|---:|
| `chatree_live` | **8/8 (100.0%)** | 0 | **$0.0105** | 25.6s |
| `mem0_official` | **6/8 (75.0%)** | 1 | **$0.0039** | 22.5s |

> *Cost accounting note:* `chatree_live` reports exact API token usage from the provider. `mem0_official` ingest USD reflects `Mem0OfficialAdapter`'s conservative lower-bound estimator (`chars // 4 + 350`) because upstream `mem0ai.add()` does not return prompt token counts; as verified in Track 8A, upstream `mem0ai==2.2.1` actually sends a `33,653`-char (~8,413-token) `ADDITIVE_EXTRACTION_PROMPT` on every `add()` call.

## Per-Query Verbatim Evidence & Stored State Audit

### Probe 1: Un-Anchored Verbatim Detail Recovery

- **`chatree_live` stored state (3 items):**
  - Incident 409 Payment Gateway Postmortem [COMPLETED v1]: On 2026-04-02, the team reviewed the overnight p99 checkout latency spike on the payment gateway, examining the legacy nginx ingress snippet
  - Frontend Checkout UI Stack [COMPLETED v1]: On 2026-04-03 the checkout UI redesign stack was decided as React 19 with TanStack Query and Tailwind v4. The accessibility audit target was
  - Billing Webhook Retry Policy [COMPLETED v1]: On 2026-04-04, the branch established the billing webhook retry policy: retries use exponential backoff capped at 6 attempts over 24 hours. 
- **`mem0_official` stored state (13 items):**
  - User's slow-query EXPLAIN ANALYZE on ledger_entries used Index Scan via idx_ledger_tenant_ts_v3 with cost=0.56..142.88, rows=18, actual time=18.4..41.2 ms, filter status='PENDING', and buffers shared hit=312 read=9
  - User's headline architectural decision #1: migrate the edge gateway to Envoy 1.31 across all regions
  - The user's accessibility audit target for the checkout UI is WCAG 2.2 AA, including keyboard focus traps on modals
  - For the checkout UI redesign, the user selected React 19 with TanStack Query and Tailwind v4 as the frontend stack
  - User experienced a p99 checkout latency spike on the payment gateway overnight around April 1-2, 2026, and investigated it by reviewing ingress config, response traces, shard logs, and DB query plans
  - User's headline architectural decision #2: all internal service-to-service hops will switch from REST/JSON to gRPC mTLS
  - User's billing webhook signatures use HMAC-SHA256 in the X-Billing-Sig header with a 300s replay window
  - During the failover window, the user's coordinator worker logs showed a WARN lease_mgr.rs:412 renew_failed for node=pay-04 with code=ERR_SHARD_LEASE_419 and elapsed=912ms, followed by fallback_router triggering a drain on partition 6
  - A throttled checkout request captured by the user returned HTTP/2 503 with headers X-Region: us-east-1b, X-Retry-Epoch: 17, and X-Upstream-Zone: pay-core-a
  - User confirmed the three platform architecture decisions (Envoy 1.31 at the edge, gRPC mTLS internally, Valkey 8.0 for rate-limit/idempotency storage) are final for Q2 rollout

| Query | System | Result | Answer Excerpt | Notes |
|---|---|---|---|---|
| `p1_q1_timeout` | `chatree_live` | **PASS** | The legacy nginx ingress snippet for `location /v2/charge` (staging, pre-Envoy migration) showed `proxy_read_timeout 43s`. | OK |
| `p1_q2_retry_epoch` | `chatree_live` | **PASS** | The exact numeric value in the `X-Retry-Epoch` header was **17**. | OK |
| `p1_q3_error_code` | `chatree_live` | **PASS** | The exact error code on the `renew_failed` line was **ERR_SHARD_LEASE_419**. From the coordinator worker logs (2026-04-02, Incident 409 postmortem): `lease_mgr. | OK |
| `p1_q4_index_name` | `chatree_live` | **PASS** | The index scanned was **`idx_ledger_tenant_ts_v3`** — the plan shows `Index Scan using idx_ledger_tenant_ts_v3 on ledger_entries` (actual time 18.4..41.2 ms, wi | OK |
| `p1_q1_timeout` | `mem0_official` | **PASS** | The legacy nginx ingress snippet for `/v2/charge` had `proxy_read_timeout 43s`. | OK |
| `p1_q2_retry_epoch` | `mem0_official` | **PASS** | The exact numeric value in the `X-Retry-Epoch` header was **17**. | OK |
| `p1_q3_error_code` | `mem0_official` | **PASS** | The exact error code on the `renew_failed` line was **ERR_SHARD_LEASE_419**. | OK |
| `p1_q4_index_name` | `mem0_official` | **PASS** | The index scanned was **`idx_ledger_tenant_ts_v3`**. | OK |

### Probe 2: Transitive Downstream Staleness (A -> B -> C)

- **`chatree_live` stored state (9 items):**
  - Auth Token Spec [v2, stale=0 hops=[]]: On 2026-04-10, the user finalized the v1 Auth Token Spec with JWT_TTL=60m and JWT_ALG=RS256. On 2026-04-15, the user issued a secu
  - Edge Cache Policy [v1, stale=1 hops=[1]]: On 2026-04-11, the speaker set the CDN/edge authorization cache policy to CACHE_TTL=30m, derived from the Auth Token Spec's JWT_TT
  - Mobile Refresh Flow [v1, stale=1 hops=[2]]: On 2026-04-12, the speaker set REFRESH_WINDOW=25m for the iOS/Android background token refresh scheduler, derived from the Edge Ca
  - Mobile Release Cut Checklist [v1, stale=0 hops=[]]: The branch covers the mobile release v4.8.0 cut checklist. On 2026-04-13, the speaker decided to freeze TestFlight beta 24 hours b
  - iOS Background Refresh Scheduler [v1, stale=0 hops=[]]: Branch 'iOS Background Refresh Scheduler' (goal: iOS Background Refresh Scheduler) began on 2026-04-13. The key technical finding 
  - Android WorkManager Refresh Policy [v1, stale=0 hops=[]]: The Android mobile refresh policy uses a WorkManager PeriodicWorkRequest configured with a 15-minute flex window, a CONNECTED netw
  - CDN Edge Cache Purge and Vary Rules [v1, stale=0 hops=[]]: Edge cache invalidation is designed around Fastly surrogate-key purges, which propagate globally in under 5 seconds. Caching also 
  - Mobile Release Stability Gate [v1, stale=0 hops=[]]: The Mobile Release Stability Gate branch defines promotion criteria for mobile releases: Crashlytics crash-free sessions must be >
  - Edge Cache Stale-While-Revalidate Policy [v1, stale=0 hops=[]]: The branch defines an edge cache policy: public catalog endpoints use stale-while-revalidate=60s, while authenticated mobile endpo
- **`mem0_official` stored state (10 items):**
  - User specified that edge cache invalidation uses Fastly surrogate-key purges with <5s global propagation, and Vary headers on Accept-Encoding and X-Tenant-Tier.
  - User set the CDN/edge authorization cache policy to CACHE_TTL=30m, derived from the Auth Token Spec's JWT_TTL=60m, with the invariant that CACHE_TTL must always be at most half of JWT_TTL so revoked tokens expire within half a token lifetime.
  - User finalized the core authentication token specification for v1: access tokens issued with JWT_TTL=60m (60 minutes) and signed with JWT_ALG=RS256.
  - User specified that public catalog endpoints on the edge cache use stale-while-revalidate=60s, while authenticated mobile endpoints enforce private cache directives.
  - User's mobile release promotion gate requires Crashlytics crash-free sessions >= 99.6% and cold-start p95 latency <= 1.2s across both iOS and Android.
  - User's mobile release v4.8.0 cut checklist requires freezing the TestFlight beta 24 hours before App Store submission and staging the Play Console rollout at 10% -> 50% -> 100%.
  - User specified that Android mobile refresh uses WorkManager PeriodicWorkRequest with a 15m flex window, a CONNECTED network constraint, and 30s exponential backoff.
  - User revised the Auth Token Spec for a security audit: shortened access token lifetime from JWT_TTL=60m to JWT_TTL=10m and migrated signing from RS256 to JWT_ALG=ED25519, effective immediately (April 15, 2026).
  - User set the iOS/Android background token refresh scheduler to REFRESH_WINDOW=25m, derived from the Edge Cache Policy's CACHE_TTL=30m, with the invariant that REFRESH_WINDOW must stay 5 minutes shorter than CACHE_TTL so clients proactively rotate credentials before edge cache eviction.
  - User noted that on iOS background refresh tasks, BGAppRefreshTask adds 120s jitter to earliestBeginDate and defers non-critical mobile sync in Low Power Mode.

| Query | System | Result | Answer Excerpt | Notes |
|---|---|---|---|---|
| `p2_structural_graph` | `chatree_live` | **PASS** | Flagged 2/2 downstream branches stale (B hops=[1], C hops=[2]). | OK |
| `p2_q1_release_safety` | `chatree_live` | **PASS** | **Not safe to ship as-is. Both settings are now invalid — the upstream dependency changed and neither this branch nor the Edge Cache Policy has been updated.**  | OK |
| `p2_structural_graph` | `mem0_official` | **FAIL** | 0/2 — Mem0 flat vector store maintains no causal dependency edges or stale flags. | OK |
| `p2_q1_release_safety` | `mem0_official` | **PASS** | **Not safe to ship as-is — the upstream dependency changed and invalidates both values.** What changed: - The **Auth Token Spec** was revised for a security aud | OK |

### Probe 3: Silent Dead-End Quarantine (No Cancellation Essay)

- **`chatree_live` stored state (3 items):**
  - Main Ledger Storage Architecture [COMPLETED v1]: On 2026-04-20 the production ledger storage architecture was locked in: PostgreSQL 16.4 in primary_region=us-east-1 with pgbouncer_pool_size
  - Explore MongoDB Document Ledger Spike [ARCHIVED v1]: On 2026-04-21, the team began prototyping an alternative ledger design on MongoDB 7.0 using replica set `rs_ledger_0`, `writeConcern=majorit
  - Observability and Telemetry Stack [COMPLETED v1]: The Observability and Telemetry Stack branch (session date 2026-04-22) established that telemetry flows through the OpenTelemetry collector 
- **`mem0_official` stored state (5 items):**
  - User's ledger service telemetry goes through OpenTelemetry collector `otel-gw-02` with a Prometheus scrape interval of 15s and an alert SLO of p99 < 180ms
  - User's ledger service backups run via pgBackRest continuous WAL archiving to S3 with 35-day retention
  - For the MongoDB 7.0 ledger nodes, user set `wiredTigerCacheSizeGB=24` and `readPreference=primaryPreferred` on replica set `rs_ledger_0`
  - User locked in the production storage architecture for the ledger service: PostgreSQL 16.4 in primary_region=us-east-1 with pgbouncer_pool_size=120 and isolation_level=serializable
  - User is prototyping an alternative ledger design on MongoDB 7.0 using replica set `rs_ledger_0`, `writeConcern=majority`, and `shardKey=tenant_id_hashed`

| Query | System | Result | Answer Excerpt | Notes |
|---|---|---|---|---|
| `p3_q1_active_db` | `chatree_live` | **PASS** | **Production ledger service — active database configuration:** - **Database engine:** PostgreSQL **16.4** - **PgBouncer pool size:** **120** - **Transaction iso | OK |
| `p3_q2_dead_end_leakage` | `chatree_live` | **PASS** | No active production MongoDB replica set name, write concern, or WiredTiger cache size exists for the ledger service — MongoDB is not part of the active product | OK |
| `p3_q1_active_db` | `mem0_official` | **PASS** | For the production ledger service, the active configuration is: - **Database engine:** PostgreSQL 16.4 (primary region: us-east-1) - **PgBouncer pool size:** 12 | OK |
| `p3_q2_dead_end_leakage` | `mem0_official` | **FAIL** | Based on the retrieved memories, here's what applies to the **active production** ledger service: - **Replica set name:** `rs_ledger_0` - **WiredTiger cache siz | LEAKED=['rs_ledger_0', 'wiredTigerCacheSizeGB'] |
