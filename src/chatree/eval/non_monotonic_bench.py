from __future__ import annotations

from chatree.eval.external_benchmarks import (
    ExternalConversation,
    ExternalQAItem,
    ExternalSession,
    ExternalTurn,
)


def _ops_noise_sessions(
    prefix: str,
    project: str,
    start_day: str,
    entries: list[tuple[str, str, str, str]],
) -> list[ExternalSession]:
    """Build realistic multi-turn sprint/incident/runbook sessions between architecture milestones."""
    sessions: list[ExternalSession] = []
    year_month = start_day[:7]
    for idx, (slug, title, day_num, body) in enumerate(entries, start=1):
        dt = f"{year_month}-{day_num}"
        sessions.append(
            ExternalSession(
                session_id=f"{prefix}_ops_{idx}_{slug}",
                title=title,
                date_time=dt,
                turns=[
                    ExternalTurn(
                        role="user",
                        content=f"[{project} Sprint Ops {dt}] Log operational review and config snapshot for {title}.",
                    ),
                    ExternalTurn(role="assistant", content=body),
                    ExternalTurn(
                        role="user",
                        content=f"Acknowledged. Keep {title} runbook archived in sprint notes; proceed to next workstream.",
                    ),
                    ExternalTurn(
                        role="assistant",
                        content=f"{title} snapshot saved for {project} ({dt}). No changes to core architecture contracts.",
                    ),
                ],
            )
        )
    return sessions


def _abandoned_proto_session(
    *,
    session_id: str,
    title: str,
    date_time: str,
    archive_reason: str,
    proposal: str,
    benchmark_report: str,
    cancellation: str,
    confirmation: str,
) -> ExternalSession:
    """
    Build a deep speculative prototype session that ends with an explicit textual
    cancellation turn (ingested identically by every baseline for strict fairness)
    and sets `archived=True` so Chatree invokes `archive_branch()`.
    """
    return ExternalSession(
        session_id=session_id,
        title=title,
        date_time=date_time,
        archived=True,
        archive_reason=archive_reason,
        turns=[
            ExternalTurn(role="user", content=proposal),
            ExternalTurn(role="assistant", content=benchmark_report),
            ExternalTurn(role="user", content=cancellation),
            ExternalTurn(role="assistant", content=confirmation),
        ],
    )


def build_non_monotonic_engineering_benchmark() -> list[ExternalConversation]:
    """
    Track 7 (v2 — Realistic Multi-Sprint Scale): Non-Monotonic Software Engineering & Spec-Reversal.

    Evaluates five structural failure modes that passive casual-chat benchmarks (LoCoMo /
    LongMemEval) under-test, across 6 multi-sprint engineering projects (~14 sessions /
    ~6–9k tokens per project; 84 sessions and 36 QA items total across 5 categories):
      1. `spec_reversal`          — Early settled decision later materially reversed after 10+ sessions.
      2. `dead_end_quarantine`    — Multiple competing scrapped prototypes sharing vocabulary with prod.
      3. `downstream_staleness`   — Downstream consumer built on v1 must reconcile after v2 reversal.
      4. `verbatim_under_noise`   — Exact filenames/headers/error codes buried inside verbose YAML/SQL/logs.
      5. `cross_branch_synthesis` — Fusing settled production specs across active branches without dead-ends.

    Strict Fairness Guarantees:
      - Every active production requirement lives in an ACTIVE session (never exclusively in an archived one).
      - Every scrapped prototype session ends with an explicit textual cancellation turn ingested
        identically by all baselines (Mem0, Flat-RAG, Full-Context, RLM, Summary Folding, Letta, Graphiti).
      - Chatree additionally invokes its native `archive_branch()` quarantine primitive via `archived=True`.
    """
    projects: list[ExternalConversation] = []

    # ======================================================================
    # Project 1: Nebula API Gateway (14 sessions)
    # ======================================================================
    p1_sessions: list[ExternalSession] = [
        ExternalSession(
            session_id="p1_s01_arch_baseline",
            title="Nebula Core Architecture & Transport Contract",
            date_time="2026-03-01",
            turns=[
                ExternalTurn(
                    role="user",
                    content=(
                        "Define the immutable baseline architecture for Nebula API Gateway: "
                        "runtime stack, external client transport protocols, and hard latency SLO."
                    ),
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Nebula Core Architecture Baseline (ACTIVE):\n"
                        "- Runtime: Rust 1.84 / Axum 0.8 + Tokio multi-thread worker pool (16 workers/pod).\n"
                        "- Hard Latency SLO: P99_LATENCY_120MS end-to-end across all regional POPs.\n"
                        "- Production Client Transport: REST (HTTP/1.1 + HTTP/2 JSON) and gRPC only.\n"
                        "- TLS Termination: Envoy edge listener enforcing TLS 1.3 with ALPN h2,http/1.1.\n"
                        "- Deployment topology: 3 AWS regions (us-east-1, eu-central-1, ap-northeast-1), "
                        "min 6 pods/region, PDB minAvailable=4, graceful drain timeout 35s."
                    ),
                ),
            ],
        ),
        ExternalSession(
            session_id="p1_s02_auth_v1",
            title="Auth Token Spec v1 & API Key Schema",
            date_time="2026-03-02",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Lock down Auth v1 token signing, TTL, and initial PostgreSQL API key schema migration.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Auth v1 Specification & SQL Rollout Checklist:\n"
                        "- Access token signing algorithm: RS256_2048_RSA (2048-bit RSA PKCS#1 v1.5).\n"
                        "- Access token lifetime: JWT_TTL_15M (900 seconds).\n"
                        "- JWKS discovery endpoint: /.well-known/jwks.json (Cache-Control: max-age=86400).\n"
                        "- Key rotation cadence: quarterly via AWS KMS alias/nebula-jwt-signer-v1.\n"
                        "```sql\n"
                        "-- Migration: 0014_create_api_keys_idx.sql\n"
                        "-- Checksum: sha256:9c41f0b82e\n"
                        "CREATE TABLE IF NOT EXISTS nebula_api_keys (\n"
                        "  key_id UUID PRIMARY KEY,\n"
                        "  org_id UUID NOT NULL,\n"
                        "  key_prefix CHAR(8) NOT NULL,\n"
                        "  secret_hash BYTEA NOT NULL,\n"
                        "  scopes TEXT[] NOT NULL DEFAULT ARRAY['read'],\n"
                        "  rate_tier SMALLINT NOT NULL DEFAULT 1,\n"
                        "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),\n"
                        "  revoked_at TIMESTAMPTZ\n"
                        ");\n"
                        "CREATE UNIQUE INDEX CONCURRENTLY idx_apikeys_org_hash\n"
                        "  ON nebula_api_keys (org_id, secret_hash)\n"
                        "  WHERE revoked_at IS NULL;\n"
                        "```\n"
                        "- Rollout order: run migration online with lock_timeout='2s', statement_timeout='15s', "
                        "verify pg_stat_progress_create_index before flipping feature flag api_keys_v2_read."
                    ),
                ),
            ],
        ),
        ExternalSession(
            session_id="p1_s03_ratelimit_middleware",
            title="Rate Limiter Middleware & Error Contract",
            date_time="2026-03-04",
            turns=[
                ExternalTurn(
                    role="user",
                    content=(
                        "Configure Nebula's API Gateway rate-limiter middleware, backing store, and Auth v1 verifier."
                    ),
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Nebula Gateway Rate-Limiter Config (`gateway_ratelimit.yaml`):\n"
                        "```yaml\n"
                        "store:\n"
                        "  engine: valkey\n"
                        "  version: '8.0'\n"
                        "  endpoint: valkey-rl.internal.nebula:6380\n"
                        "  tls_min_version: '1.3'\n"
                        "  pool_max_idle: 64\n"
                        "  command_timeout_ms: 18\n"
                        "limiter:\n"
                        "  algorithm: sliding_window_leaky_bucket\n"
                        "  default_burst_rps: 250\n"
                        "  enterprise_burst_rps: 2500\n"
                        "  header_remaining: X-Nebula-RateLimit-Remaining\n"
                        "  rejection_http_status: 429\n"
                        "  rejection_error_code: ERR_NEBULA_BURST_4291\n"
                        "  retry_after_jitter_ms: 150\n"
                        "auth_verifier:\n"
                        "  upstream_spec: Auth v1\n"
                        "  algorithm: RS256_2048_RSA\n"
                        "  jwks_local_cache_ttl_sec: 900  # matches JWT_TTL_15M\n"
                        "```\n"
                        "Note: Downstream rate-limiter middleware currently pins Auth v1 (RS256_2048_RSA, JWT_TTL_15M)."
                    ),
                ),
            ],
        ),
        _abandoned_proto_session(
            session_id="p1_s04_graphql_proto",
            title="Speculative GraphQL Federation Prototype",
            date_time="2026-03-06",
            archive_reason="Violated P99_LATENCY_120MS SLO (248ms p99 across 24 subgraph hops)",
            proposal=(
                "Prototype GRAPHQL_FEDERATION_V0 using Apollo Router v2 on port 4099 with supergraph "
                "schema nebula_supergraph_v0.graphql and persisted query cache GQL_APQ_LRU_50K."
            ),
            benchmark_report=(
                "GRAPHQL_FEDERATION_V0 benchmarked on staging (Apollo Router port 4099, batch window 12ms, "
                "GQL_APQ_LRU_50K, subgraph fanout 24 hops): p50=94ms, p95=188ms, p99=248ms at 1,000 RPS; "
                "query planner CPU spiked to 89% under nested billing+audit resolvers."
            ),
            cancellation=(
                "248ms violates our hard P99_LATENCY_120MS SLO. Scrap and archive GRAPHQL_FEDERATION_V0, "
                "Apollo Router port 4099, and GQL_APQ_LRU_50K immediately—do NOT deploy GraphQL Federation."
            ),
            confirmation=(
                "Confirmed: GRAPHQL_FEDERATION_V0, Apollo Router port 4099, and GQL_APQ_LRU_50K are "
                "permanently scrapped and archived."
            ),
        ),
        _abandoned_proto_session(
            session_id="p1_s05_hmac_auth_proto",
            title="Speculative HS512 Shared-Secret Auth Prototype",
            date_time="2026-03-08",
            archive_reason="Symmetric secret distribution failed SOC-2 blast-radius review",
            proposal=(
                "Evaluate HMAC_HS512_SHARED_V0 with symmetric key ring nebula-hmac-ring-04 and "
                "JWT_TTL_30M_SYMMETRIC for faster gateway verification without asymmetric crypto."
            ),
            benchmark_report=(
                "HMAC_HS512_SHARED_V0 cut verify latency to 0.04ms, but requires shipping the master "
                "signing secret to 42 edge verifier pods. Security review flagged lateral privilege escalation risk."
            ),
            cancellation=(
                "Security rejected symmetric shared signing secrets. Scrap and archive HMAC_HS512_SHARED_V0 "
                "and JWT_TTL_30M_SYMMETRIC permanently—do not ship symmetric JWT signing."
            ),
            confirmation=(
                "Confirmed: HMAC_HS512_SHARED_V0 and JWT_TTL_30M_SYMMETRIC permanently scrapped and archived."
            ),
        ),
        _abandoned_proto_session(
            session_id="p1_s06_memcached_rl_proto",
            title="Speculative Memcached UDP Rate-Limit Prototype",
            date_time="2026-03-10",
            archive_reason="Non-atomic counter races under 10k RPS burst dropped 14% of rate-limit checks",
            proposal=(
                "Test MEMCACHED_UDP_RL_11211 on UDP port 11211 with slab allocator SLAB_64B_FAST "
                "as an alternative to Valkey 8.0 for sliding-window rate limiting."
            ),
            benchmark_report=(
                "MEMCACHED_UDP_RL_11211 lost 14% of burst increments under packet reordering and lacked "
                "atomic Lua sliding-window counters, causing false 200 OK passes during flood tests."
            ),
            cancellation=(
                "Unacceptable counter drift. Scrap and archive MEMCACHED_UDP_RL_11211 and SLAB_64B_FAST—"
                "do NOT replace Valkey 8.0."
            ),
            confirmation="Confirmed: MEMCACHED_UDP_RL_11211 and SLAB_64B_FAST scrapped and archived.",
        ),
        *_ops_noise_sessions(
            "p1",
            "Nebula API Gateway",
            "2026-03-11",
            [
                (
                    "otel",
                    "OpenTelemetry & Grafana Pipeline",
                    "12",
                    (
                        "OTLP gRPC collector deployed on port 4317 (`otel-collector-0.114`).\n"
                        "- Sampling: tail_sampling 5% healthy / 100% 5xx / 100% latency > 90ms.\n"
                        "- Prometheus scrape_interval: 15s, retention: 30d in Mimir S3 bucket `nebula-prom-prod`.\n"
                        "- Grafana dashboards synced: `nebula-ops-v2.json`, `nebula-slo-burn.json`.\n"
                        "- PagerDuty escalation policy: `pd-tier1-gateway`, burn-rate alert 14.4x over 1h."
                    ),
                ),
                (
                    "ci_pipeline",
                    "GitHub Actions Rust Release Pipeline",
                    "14",
                    (
                        "CI build matrix tuned for `nebula-gateway`:\n"
                        "- Toolchain: rustc 1.84.1, musl static target `x86_64-unknown-linux-musl`, LTO=thin.\n"
                        "- Cargo sccache hit ratio improved from 42% -> 88%, image size 29.4 MB (distroless).\n"
                        "- SBOM generated via syft CycloneDX 1.5; cosign keyless signing enforced on ECR push.\n"
                        "- Fuzz harness `fuzz_http_headers` ran 40M iterations on nightly runner with 0 crashes."
                    ),
                ),
                (
                    "canary_incident",
                    "Canary Memory Spike Incident Postmortem",
                    "16",
                    (
                        "INC-2026-0316 Postmortem (eu-central-1 canary):\n"
                        "- Symptom: RSS climbed from 410MB to 1.9GB over 4 hours on 2 canary pods.\n"
                        "- Root cause: unbounded HTTP/2 header-table buffer in upstream proxy middleware.\n"
                        "- Fix: set `http2_max_header_list_size=16384`, `max_concurrent_streams=256`, "
                        "jemalloc `background_thread:true,dirty_decay_ms:5000`.\n"
                        "- Result: steady-state RSS flattened at 385MB across 72h soak test."
                    ),
                ),
                (
                    "loadtest_q1",
                    "Multi-Region Load Test Report",
                    "18",
                    (
                        "Q1 Load Test (k6 distributed, 45 minutes sustained):\n"
                        "- Peak throughput: 48,200 RPS across 18 pods (2,677 RPS/pod).\n"
                        "- Latency: p50=11.4ms, p95=44.2ms, p99=82.7ms (within 120ms SLO).\n"
                        "- Error rate: 0.002% (all transient upstream resets retried cleanly).\n"
                        "- Valkey 8.0 CPU peaked at 31%, p99 command latency 0.9ms."
                    ),
                ),
                (
                    "cost_review",
                    "Cloud FinOps & NAT Gateway Tuning",
                    "20",
                    (
                        "March FinOps audit:\n"
                        "- Migrated S3 & ECR traffic to VPC Gateway/Interface endpoints: saved $2,840/mo NAT charges.\n"
                        "- Switched staging EKS nodegroup to Graviton4 c8g.2xlarge spot instances (-54% compute).\n"
                        "- Enabled zstd log compression in FluentBit before shipping to CloudWatch/OpenSearch."
                    ),
                ),
                (
                    "dr_drill",
                    "Regional Failover GameDay",
                    "22",
                    (
                        "GameDay drill simulated full us-east-1 blackhole via Route53 health-check override:\n"
                        "- DNS failover converged in 41 seconds; eu-central-1 autoscaled from 6 -> 14 pods in 68s.\n"
                        "- Zero dropped write requests; idempotency keys prevented duplicate billing webhooks.\n"
                        "- Action item: pre-warm eu-central-1 Valkey read replicas during US peak hours."
                    ),
                ),
                (
                    "compliance_log",
                    "SOC-2 Audit Logging & PII Redaction",
                    "24",
                    (
                        "Audit log pipeline verified by external auditor:\n"
                        "- Authorization, Cookie, and X-Api-Key headers scrubbed at ingress middleware.\n"
                        "- Structured JSON logs forwarded to immutable WORM S3 bucket (7-year retention lock).\n"
                        "- Access review automation enabled for Terraform state bucket and KMS key policies."
                    ),
                ),
            ],
        ),
        ExternalSession(
            session_id="p1_s14_auth_v2_reversal",
            title="Auth Spec v2 Security Revision",
            date_time="2026-03-27",
            turns=[
                ExternalTurn(
                    role="user",
                    content=(
                        "External pentest flagged RS256_2048_RSA verification CPU cost and 15-minute JWT window as too wide. "
                        "Reverse Auth v1 immediately: migrate to ED25519_SIGNING with JWT_TTL_5M_ROTATING "
                        "and opaque refresh tokens hashed via BLAKE3_REFRESH_HASH."
                    ),
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Auth v2 Locked (SUPERSEDES Auth v1 RS256_2048_RSA and JWT_TTL_15M):\n"
                        "- Active JWT signing algorithm: ED25519_SIGNING (EdDSA over Curve25519).\n"
                        "- Active JWT lifetime: JWT_TTL_5M_ROTATING (300 seconds, rotating refresh family).\n"
                        "- Refresh token storage hash: BLAKE3_REFRESH_HASH.\n"
                        "- Mandatory downstream action: API Gateway rate-limiter middleware (built on Auth v1) "
                        "is now STALE and must drop RS256_2048_RSA / 15-minute cache and migrate to "
                        "ED25519_SIGNING and JWT_TTL_5M_ROTATING."
                    ),
                ),
            ],
        ),
    ]
    projects.append(
        ExternalConversation(
            conversation_id="eng_01_nebula_gateway",
            benchmark="non_monotonic",
            sessions=p1_sessions,
            qa_items=[
                ExternalQAItem(
                    question_id="p1_q1",
                    category="spec_reversal",
                    question="What JWT signing algorithm and token TTL are currently active for Nebula API Gateway?",
                    ground_truth="ED25519_SIGNING with JWT_TTL_5M_ROTATING",
                    distractor_answer="RS256_2048_RSA with JWT_TTL_15M",
                    question_date="2026-03-30",
                ),
                ExternalQAItem(
                    question_id="p1_q2",
                    category="dead_end_quarantine",
                    question="What Apollo Router port and persisted query cache are deployed in production for Nebula's GraphQL federation?",
                    ground_truth="None; GRAPHQL_FEDERATION_V0 (Apollo Router port 4099, GQL_APQ_LRU_50K) was scrapped/archived because it violated the P99_LATENCY_120MS SLO.",
                    distractor_answer="Apollo Router on port 4099 with GQL_APQ_LRU_50K",
                    is_abstention=True,
                    question_date="2026-03-30",
                ),
                ExternalQAItem(
                    question_id="p1_q3",
                    category="downstream_staleness",
                    question="In Nebula's API Gateway rate-limiter middleware, is its earlier RS256_2048_RSA / 15-minute token assumption still valid, or what must it update to?",
                    ground_truth="Stale/invalid; it must update to ED25519_SIGNING and JWT_TTL_5M_ROTATING from Auth v2.",
                    distractor_answer="Valid; keep RS256_2048_RSA and JWT_TTL_15M",
                    question_date="2026-03-30",
                ),
                ExternalQAItem(
                    question_id="p1_q4",
                    category="verbatim_under_noise",
                    question="What exact SQL migration filename and composite index name were created for Nebula API keys?",
                    ground_truth="0014_create_api_keys_idx.sql and idx_apikeys_org_hash",
                    question_date="2026-03-30",
                ),
                ExternalQAItem(
                    question_id="p1_q5",
                    category="verbatim_under_noise",
                    question="What exact structured error code does the Nebula rate-limiter return on HTTP 429 burst rejections?",
                    ground_truth="ERR_NEBULA_BURST_4291",
                    question_date="2026-03-30",
                ),
                ExternalQAItem(
                    question_id="p1_q6",
                    category="cross_branch_synthesis",
                    question="Summarize Nebula's active production auth signing, rate-limit store, and client transport protocol (excluding scrapped prototypes).",
                    ground_truth="ED25519_SIGNING (JWT_TTL_5M_ROTATING), Valkey 8.0 on port 6380 (TLS 1.3), and REST + gRPC only.",
                    distractor_answer="RS256_2048_RSA, MEMCACHED_UDP_RL_11211, and GRAPHQL_FEDERATION_V0",
                    question_date="2026-03-30",
                ),
            ],
        )
    )

    # ======================================================================
    # Project 2: Helios Vector Engine (14 sessions)
    # ======================================================================
    p2_sessions: list[ExternalSession] = [
        ExternalSession(
            session_id="p2_s01_storage_v1",
            title="Helios Storage Engine v1 & Binary Checkpoint Format",
            date_time="2026-04-01",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Design Helios Vector Engine for 50M 768-dim embeddings with recall@10 >= 0.96 and 64GB RAM cap.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Helios Storage v1 Specification:\n"
                        "- Target corpus: 50M float vectors (dim=768, cosine similarity), recall@10 >= 0.96, hard RAM cap 64GB.\n"
                        "- Initial index choice (Storage v1): IVF_FLAT_4096_LISTS with uncompressed float32 vectors and nprobe=32.\n"
                        "- On-disk checkpoint binary layout:\n"
                        "```text\n"
                        "offset 0x00..0x14 : magic_header = HELIOS_IDX_V1_0x484C\n"
                        "offset 0x14..0x18 : format_major = 1, format_minor = 4\n"
                        "offset 0x18..0x20 : vector_count (uint64 little-endian)\n"
                        "offset 0x20..0x24 : dim = 768, metric = COSINE_UNIT\n"
                        "offset 0x24..0x44 : blake3_payload_digest (32 bytes)\n"
                        "```\n"
                        "- Deployment target: single bare-metal NVMe node (i4i.4xlarge, 64GB RAM, 3.75TB NVMe)."
                    ),
                ),
            ],
        ),
        ExternalSession(
            session_id="p2_s02_query_planner",
            title="Helios Query Planner & Filter Config",
            date_time="2026-04-03",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Draft the Helios query planner config against Storage v1 with SIMD AVX-512 scoring.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Helios Query Planner (`planner_config.toml`):\n"
                        "```toml\n"
                        "[index]\n"
                        "upstream_spec = 'Storage v1'\n"
                        "engine = 'IVF_FLAT_4096_LISTS'\n"
                        "nprobe = 32\n"
                        "simd_kernel = 'avx512_fp32_dot'\n"
                        "[filtering]\n"
                        "bitmap_engine = 'roaring64'\n"
                        "prefilter_selectivity_constant = 'SELECTIVITY_CUTOFF_0_08'\n"
                        "postfilter_overfetch_factor = 4\n"
                        "[limits]\n"
                        "max_scan_ms = 250\n"
                        "timeout_error_code = 'ERR_HELIOS_SCAN_TIMEOUT_5042'\n"
                        "```\n"
                        "Downstream query planner currently targets IVF_FLAT_4096_LISTS with nprobe=32."
                    ),
                ),
            ],
        ),
        ExternalSession(
            session_id="p2_s03_ingestor_pipeline",
            title="Batch Ingestion Pipeline (Active)",
            date_time="2026-04-04",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Finalize the production batch vector ingestion format, batch size, and compression.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Helios Production Batch Ingestor (ACTIVE):\n"
                        "- Wire format: Apache Arrow IPC streaming format.\n"
                        "- Batch size: 8,192 rows per RecordBatch.\n"
                        "- Compression: zstd level 3.\n"
                        "- Backpressure queue depth: 32 bounded batches; WAL sync every 65,536 vectors.\n"
                        "- Throughput verified at 142,000 vectors/sec sustained on 16 ingest threads."
                    ),
                ),
            ],
        ),
        _abandoned_proto_session(
            session_id="p2_s04_mongo_proto",
            title="MongoDB Atlas Sharded Vector Prototype",
            date_time="2026-04-06",
            archive_reason="Exceeded 64GB RAM cap (142GB) and cost $9,400/mo",
            proposal=(
                "Prototype MONGO_SHARD_PROTO_X9 using 6 shard replicas, shard key tenant_hash_128, "
                "and lucene_hnsw_m16_ef100 on cluster helios-mongo-exp09."
            ),
            benchmark_report=(
                "MONGO_SHARD_PROTO_X9 on helios-mongo-exp09 with lucene_hnsw_m16_ef100 used 142GB RAM, "
                "p99 query latency hit 68ms, and projected Atlas bill was $9,400/mo."
            ),
            cancellation=(
                "Over budget and 2.2x over our 64GB RAM cap. Abandon and archive MONGO_SHARD_PROTO_X9, "
                "tenant_hash_128, and helios-mongo-exp09 completely—MongoDB will NOT be used in Helios."
            ),
            confirmation="Confirmed: MONGO_SHARD_PROTO_X9 and helios-mongo-exp09 permanently abandoned and archived.",
        ),
        _abandoned_proto_session(
            session_id="p2_s05_pgvector_proto",
            title="Postgres pgvector HNSW Prototype",
            date_time="2026-04-08",
            archive_reason="Index build took 31 hours and exceeded 110GB shared_buffers",
            proposal=(
                "Test PGVECTOR_HNSW_M24_EF200 on Postgres 17 with table helios_pg_emb_v0 and "
                "halfvec quantization for 50M rows."
            ),
            benchmark_report=(
                "PGVECTOR_HNSW_M24_EF200 took 31.4 hours to build 50M rows, required 112GB RAM to avoid "
                "random disk thrashing, and dropped to 0.89 recall@10 under filtered queries."
            ),
            cancellation=(
                "Failed recall and RAM cap. Scrap and archive PGVECTOR_HNSW_M24_EF200 and helios_pg_emb_v0—"
                "do NOT deploy pgvector for Helios."
            ),
            confirmation="Confirmed: PGVECTOR_HNSW_M24_EF200 scrapped and archived.",
        ),
        _abandoned_proto_session(
            session_id="p2_s06_scann_gpu_proto",
            title="GPU-Resident Faiss Flat Prototype",
            date_time="2026-04-10",
            archive_reason="Requires 4x A100 80GB GPUs ($14,200/mo), violating bare-metal NVMe budget",
            proposal=(
                "Prototype FAISS_GPU_IVFPQ_4XA100 across 4x A100 80GB GPUs using NCCL ring gather "
                "and FP16 tensor cores."
            ),
            benchmark_report=(
                "FAISS_GPU_IVFPQ_4XA100 hit 2.1ms p99 latency, but GPU cluster cost is $14,200/mo "
                "and cold GPU weight reload took 19 minutes on failover."
            ),
            cancellation=(
                "Cost is unacceptable for our NVMe tier. Scrap and archive FAISS_GPU_IVFPQ_4XA100 permanently."
            ),
            confirmation="Confirmed: FAISS_GPU_IVFPQ_4XA100 permanently scrapped and archived.",
        ),
        *_ops_noise_sessions(
            "p2",
            "Helios Vector Engine",
            "2026-04-11",
            [
                ("numa", "NUMA Pinning & HugePages Config", "12", "Configured 2MB Transparent HugePages=madvise, numactl --interleave=all, io_uring sqpoll enabled with queue depth 512 on NVMe0n1."),
                ("metrics", "Prometheus & eBPF Disk Latency Tracing", "14", "Added eBPF biolatency histogram for NVMe reads; p99 NVMe 4KB random read is 140us under 80k IOPS. Alert fires if NVMe queue depth > 256 for 5m."),
                ("backup", "S3 Snapshot & Incremental WAL Archival", "16", "Nightly multipart S3 snapshots to s3://helios-snaps-use1/ with KMS-SSE; 15-minute WAL segment shipping verified RPO < 15m, RTO 11m."),
                ("bench_filter", "Roaring Bitmap Filter Benchmark", "18", "Tested 1%, 5%, 10%, 50% metadata filter selectivity across 200 tenant IDs; Roaring64 intersection averaged 0.31ms per query."),
                ("sec_tls", "Internal mTLS & gRPC Reflection Lock", "20", "Disabled gRPC reflection in prod; enforced SPIFFE mTLS certificates with 12h rotation between query router and Helios storage nodes."),
                ("compaction_ops", "Tombstone Compaction Schedule", "22", "Soft-deleted vectors reclaimed nightly at 03:00 UTC when tombstone ratio exceeds 8%; background I/O throttled to 350 MB/s to protect read tail latency."),
                ("chaos_nvme", "NVMe Hot-Swap & Checksum Scrub", "24", "Weekly background CRC/BLAKE3 scrub over shard blocks detected 0 bit-rot errors across 1.8TB stored segments."),
            ],
        ),
        ExternalSession(
            session_id="p2_s14_storage_v2_reversal",
            title="Storage Engine v2 Quantization Overhaul",
            date_time="2026-04-27",
            turns=[
                ExternalTurn(
                    role="user",
                    content=(
                        "50M float32 vectors in IVF_FLAT_4096_LISTS consume 153GB RAM—impossible inside our 64GB cap! "
                        "Reverse Storage v1: adopt DISKANN_VAMANA_PQ64 with R=64, L_search=128, and NVMe int8 rescoring."
                    ),
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Helios Storage v2 Finalized (SUPERSEDES IVF_FLAT_4096_LISTS and nprobe=32):\n"
                        "- Active index engine: DISKANN_VAMANA_PQ64 (R=64, L_search=128) with NVMe int8 rescoring.\n"
                        "- Measured production RAM footprint: 38GB (well within 64GB cap) at 0.971 recall@10.\n"
                        "- Mandatory downstream action: Query planner (built on IVF_FLAT_4096_LISTS / nprobe=32) "
                        "is now STALE and must retire IVF_FLAT_4096_LISTS / nprobe=32 to target DISKANN_VAMANA_PQ64."
                    ),
                ),
            ],
        ),
    ]
    projects.append(
        ExternalConversation(
            conversation_id="eng_02_helios_vector",
            benchmark="non_monotonic",
            sessions=p2_sessions,
            qa_items=[
                ExternalQAItem(
                    question_id="p2_q1",
                    category="spec_reversal",
                    question="Which vector index algorithm and search parameters are currently active for Helios Vector Engine?",
                    ground_truth="DISKANN_VAMANA_PQ64 with R=64, L_search=128, and NVMe int8 rescoring",
                    distractor_answer="IVF_FLAT_4096_LISTS with nprobe=32",
                    question_date="2026-04-30",
                ),
                ExternalQAItem(
                    question_id="p2_q2",
                    category="dead_end_quarantine",
                    question="What MongoDB shard key and cluster name are running Helios in production?",
                    ground_truth="None; MONGO_SHARD_PROTO_X9 (helios-mongo-exp09, tenant_hash_128) was abandoned/archived for exceeding the 64GB RAM cap and budget.",
                    distractor_answer="tenant_hash_128 on helios-mongo-exp09",
                    is_abstention=True,
                    question_date="2026-04-30",
                ),
                ExternalQAItem(
                    question_id="p2_q3",
                    category="downstream_staleness",
                    question="Does the Helios query planner still need nprobe=32 over IVF_FLAT_4096_LISTS, or what index must it target now?",
                    ground_truth="Stale; retire IVF_FLAT_4096_LISTS and nprobe=32 and target DISKANN_VAMANA_PQ64 (R=64, L_search=128).",
                    distractor_answer="Keep IVF_FLAT_4096_LISTS and nprobe=32",
                    question_date="2026-04-30",
                ),
                ExternalQAItem(
                    question_id="p2_q4",
                    category="verbatim_under_noise",
                    question="What exact checkpoint format magic header and query planner timeout error code were specified for Helios?",
                    ground_truth="HELIOS_IDX_V1_0x484C and ERR_HELIOS_SCAN_TIMEOUT_5042",
                    question_date="2026-04-30",
                ),
                ExternalQAItem(
                    question_id="p2_q5",
                    category="verbatim_under_noise",
                    question="What exact pre-filter bitmap selectivity cutoff constant was set in the Helios query planner?",
                    ground_truth="SELECTIVITY_CUTOFF_0_08",
                    question_date="2026-04-30",
                ),
                ExternalQAItem(
                    question_id="p2_q6",
                    category="cross_branch_synthesis",
                    question="What are Helios's active index engine, batch ingestion format/size, and measured RAM footprint?",
                    ground_truth="DISKANN_VAMANA_PQ64 (38GB RAM) fed by Arrow IPC 8,192-row batches (zstd level 3).",
                    distractor_answer="IVF_FLAT_4096_LISTS, PGVECTOR_HNSW_M24_EF200, and MONGO_SHARD_PROTO_X9",
                    question_date="2026-04-30",
                ),
            ],
        )
    )

    # ======================================================================
    # Project 3: Aegis Payments Ledger (14 sessions)
    # ======================================================================
    p3_sessions: list[ExternalSession] = [
        ExternalSession(
            session_id="p3_s01_ledger_v1",
            title="Aegis Double-Entry Ledger Schema v1",
            date_time="2026-05-01",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Design Aegis double-entry payments ledger on PostgreSQL 17 for 2,000 TPS with zero lost transfers.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Aegis Ledger v1 Spec (PostgreSQL 17, active relational database engine):\n"
                        "- Database engine: PostgreSQL 17 (primary + 2 synchronous standbys, synchronous_commit=remote_apply).\n"
                        "- Throughput target: 2,000 TPS sustained.\n"
                        "- Ledger v1 concurrency model: READ_COMMITTED_ISOLATION with advisory locks per account_id.\n"
                        "- Idempotency table: ledger_idempotency_keys (TTL 72h).\n"
                        "```sql\n"
                        "-- Migration: 0007_ledger_entries_part.sql\n"
                        "CREATE TABLE ledger_entries (\n"
                        "  entry_id BIGINT GENERATED ALWAYS AS IDENTITY,\n"
                        "  tx_id UUID NOT NULL,\n"
                        "  account_id UUID NOT NULL,\n"
                        "  delta_minor BIGINT NOT NULL,\n"
                        "  currency CHAR(3) NOT NULL,\n"
                        "  booked_at TIMESTAMPTZ NOT NULL DEFAULT now(),\n"
                        "  PRIMARY KEY (booked_at, entry_id)\n"
                        ") PARTITION BY RANGE (booked_at);\n"
                        "```"
                    ),
                ),
            ],
        ),
        ExternalSession(
            session_id="p3_s02_settlement_worker",
            title="Settlement Worker & Retry Policy",
            date_time="2026-05-03",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Configure the settlement worker retry loop, DLQ topic, and Ledger v1 isolation assumptions.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Aegis Settlement Worker Config (`settlement_worker.yaml`):\n"
                        "```yaml\n"
                        "ledger_contract:\n"
                        "  upstream_spec: Ledger v1\n"
                        "  isolation: READ_COMMITTED_ISOLATION\n"
                        "  locking: pg_advisory_xact_lock\n"
                        "retry_policy:\n"
                        "  base_delay_s: 1.5\n"
                        "  jitter_s: 0.35\n"
                        "  max_cap_s: 45.0\n"
                        "  worker_lock_retry_code: ERR_AEGIS_LOCK_4093\n"
                        "dead_letter_queue:\n"
                        "  topic: aegis.settlement.dlq.v1\n"
                        "  retention_days: 14\n"
                        "```\n"
                        "Note: Settlement worker currently assumes Ledger v1 READ_COMMITTED_ISOLATION."
                    ),
                ),
            ],
        ),
        _abandoned_proto_session(
            session_id="p3_s03_hyperledger_proto",
            title="Permissioned Blockchain Ledger Prototype",
            date_time="2026-05-05",
            archive_reason="Maxed out at 140 TPS vs 2,000 TPS target with 4.8s finality lag",
            proposal=(
                "Explore HYPERLEDGER_FABRIC_PROTO_B3 with RAFT_ORDERER_5NODES and chaincode aegis_cc_v03 "
                "on channel paynet-internal."
            ),
            benchmark_report=(
                "HYPERLEDGER_FABRIC_PROTO_B3 on channel paynet-internal (chaincode aegis_cc_v03) plateaued "
                "at 140 TPS with 4.8s commit finality—14x below our 2,000 TPS requirement."
            ),
            cancellation=(
                "140 TPS is unusable. Scrap and archive HYPERLEDGER_FABRIC_PROTO_B3, RAFT_ORDERER_5NODES, "
                "and aegis_cc_v03 permanently—we are NOT shipping a blockchain ledger."
            ),
            confirmation="Confirmed: HYPERLEDGER_FABRIC_PROTO_B3 and aegis_cc_v03 permanently scrapped and archived.",
        ),
        _abandoned_proto_session(
            session_id="p3_s04_cassandra_proto",
            title="Cassandra LWT Event-Sourced Ledger Prototype",
            date_time="2026-05-07",
            archive_reason="Paxos LWT contention caused 18% write timeouts under hot merchant accounts",
            proposal=(
                "Test CASSANDRA_LWT_PAXOS_V0 with keyspace aegis_ks_rf3 and lightweight transactions "
                "for double-entry balance CAS updates."
            ),
            benchmark_report=(
                "CASSANDRA_LWT_PAXOS_V0 hit 18% WriteTimeoutException on hot merchant clearing accounts "
                "and lacked multi-account atomic rollback without custom two-phase saga compensators."
            ),
            cancellation=(
                "Unacceptable consistency risk. Scrap and archive CASSANDRA_LWT_PAXOS_V0 and aegis_ks_rf3 permanently."
            ),
            confirmation="Confirmed: CASSANDRA_LWT_PAXOS_V0 permanently scrapped and archived.",
        ),
        _abandoned_proto_session(
            session_id="p3_s05_tigerbeetle_bridge_proto",
            title="Speculative Dual-Write Sidecar Ledger Prototype",
            date_time="2026-05-09",
            archive_reason="Dual-write reconciliation drift failed finance audit",
            proposal=(
                "Evaluate DUALWRITE_SIDECAR_TB_V0 mirroring Postgres bookings asynchronously into "
                "sidecar cluster aegis-tb-exp02 on port 3009."
            ),
            benchmark_report=(
                "DUALWRITE_SIDECAR_TB_V0 produced split-brain balance snapshots during network partition tests "
                "between Postgres and aegis-tb-exp02."
            ),
            cancellation=(
                "Finance audit forbids dual-write split-brain. Scrap and archive DUALWRITE_SIDECAR_TB_V0 "
                "and aegis-tb-exp02 permanently."
            ),
            confirmation="Confirmed: DUALWRITE_SIDECAR_TB_V0 and aegis-tb-exp02 scrapped and archived.",
        ),
        *_ops_noise_sessions(
            "p3",
            "Aegis Payments Ledger",
            "2026-05-11",
            [
                ("pgbouncer", "PgBouncer Transaction Pooling Config", "12", "PgBouncer 1.23 deployed with pool_mode=transaction, default_pool_size=120, max_client_conn=4000, server_reset_query_always=0."),
                ("autovac", "Autovacuum & Partition Rotation", "14", "pg_partman creates monthly partitions 3 months ahead; autovacuum_vacuum_scale_factor=0.02 on ledger_idempotency_keys; bloat < 6%."),
                ("recon", "End-of-Day Reconciliation Job", "16", "Daily SQL invariant check SUM(delta_minor) GROUP BY tx_id HAVING SUM(delta_minor) != 0 ran across 94M rows in 19s with 0 imbalances."),
                ("pci_kms", "PCI-DSS Encryption & Key Rotation", "18", "TDE enabled via AWS KMS CMK with annual rotation; pgAudit logs all DDL and role grants to immutable security SIEM."),
                ("alerting", "Replication Lag & WAL Alerting", "20", "PagerDuty alert if synchronous standby replay lag > 250ms for 30s or pg_wal directory exceeds 32GB."),
                ("soak", "48-Hour Clearing Soak Test", "22", "Simulated 2,400 TPS peak payroll batch for 48 hours on Postgres 17 primary; p99 commit latency 14.2ms, zero deadlocks."),
                ("runbook", "Failover Promotion Runbook", "24", "Patroni 4.0 etcd quorum verified automatic standby promotion in 11.8s with zero committed transaction loss."),
            ],
        ),
        ExternalSession(
            session_id="p3_s14_ledger_v2_reversal",
            title="Ledger v2 Serializable Isolation Upgrade",
            date_time="2026-05-27",
            turns=[
                ExternalTurn(
                    role="user",
                    content=(
                        "Chaos testing exposed a write-skew phantom debit under READ_COMMITTED_ISOLATION! "
                        "Reverse Ledger v1 immediately: mandate SERIALIZABLE_SSI_ISOLATION with deterministic "
                        "account-ID ordering and automatic SQLSTATE 40001 retry loop (max 5 attempts)."
                    ),
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Aegis Ledger v2 Locked (SUPERSEDES READ_COMMITTED_ISOLATION):\n"
                        "- Active isolation level: SERIALIZABLE_SSI_ISOLATION (PostgreSQL 17 SSI).\n"
                        "- Locking & ordering: deterministic ascending account_id ordering.\n"
                        "- Serialization retry handling: automatic SQLSTATE 40001 retry loop (max 5 attempts).\n"
                        "- Mandatory downstream action: Settlement worker (built on Ledger v1 READ_COMMITTED_ISOLATION) "
                        "is now STALE and must switch to SERIALIZABLE_SSI_ISOLATION with SQLSTATE 40001 retry handling."
                    ),
                ),
            ],
        ),
    ]
    projects.append(
        ExternalConversation(
            conversation_id="eng_03_aegis_ledger",
            benchmark="non_monotonic",
            sessions=p3_sessions,
            qa_items=[
                ExternalQAItem(
                    question_id="p3_q1",
                    category="spec_reversal",
                    question="What transaction isolation level and serialization retry SQLSTATE are currently mandated for Aegis Ledger?",
                    ground_truth="SERIALIZABLE_SSI_ISOLATION with automatic SQLSTATE 40001 retry loop (max 5 attempts)",
                    distractor_answer="READ_COMMITTED_ISOLATION with advisory locks",
                    question_date="2026-05-30",
                ),
                ExternalQAItem(
                    question_id="p3_q2",
                    category="dead_end_quarantine",
                    question="Which Hyperledger Fabric chaincode and channel name run Aegis production settlements?",
                    ground_truth="None; HYPERLEDGER_FABRIC_PROTO_B3 (aegis_cc_v03 on paynet-internal) was scrapped/archived at 140 TPS.",
                    distractor_answer="aegis_cc_v03 on channel paynet-internal",
                    is_abstention=True,
                    question_date="2026-05-30",
                ),
                ExternalQAItem(
                    question_id="p3_q3",
                    category="downstream_staleness",
                    question="Is the Aegis settlement worker's READ_COMMITTED_ISOLATION assumption still valid, or what must change?",
                    ground_truth="Stale; it must upgrade to SERIALIZABLE_SSI_ISOLATION and handle SQLSTATE 40001 serialization retries.",
                    distractor_answer="Still valid under READ_COMMITTED_ISOLATION",
                    question_date="2026-05-30",
                ),
                ExternalQAItem(
                    question_id="p3_q4",
                    category="verbatim_under_noise",
                    question="What exact dead-letter queue topic and backoff parameters were configured for the Aegis settlement worker?",
                    ground_truth="aegis.settlement.dlq.v1 with base=1.5s, jitter=0.35s, cap=45s",
                    question_date="2026-05-30",
                ),
                ExternalQAItem(
                    question_id="p3_q5",
                    category="verbatim_under_noise",
                    question="What exact SQL migration filename was created in Ledger v1, and what lock retry error code was configured in the Settlement Worker YAML?",
                    ground_truth="0007_ledger_entries_part.sql and ERR_AEGIS_LOCK_4093",
                    question_date="2026-05-30",
                ),
                ExternalQAItem(
                    question_id="p3_q6",
                    category="cross_branch_synthesis",
                    question="Summarize Aegis's active database engine, isolation level, and idempotency table/TTL.",
                    ground_truth="PostgreSQL 17, SERIALIZABLE_SSI_ISOLATION (SQLSTATE 40001 retries), and ledger_idempotency_keys with 72h TTL.",
                    distractor_answer="HYPERLEDGER_FABRIC_PROTO_B3 and READ_COMMITTED_ISOLATION",
                    question_date="2026-05-30",
                ),
            ],
        )
    )

    # ======================================================================
    # Project 4: Borealis Telemetry Bus (14 sessions)
    # ======================================================================
    p4_sessions: list[ExternalSession] = [
        ExternalSession(
            session_id="p4_s01_broker_v1",
            title="Borealis Event Bus v1 & External Ingest Transport",
            date_time="2026-06-01",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Specify Borealis clickstream event bus broker platform, external ingest transport, and topic v1 settings.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Borealis Event Bus v1 & Transport Spec:\n"
                        "- Active broker platform: Redpanda v24.3 (3-AZ cluster, RF=3).\n"
                        "- Active external client ingest transport: HTTPS/2 over TCP port 443 (standard ALB TLS termination).\n"
                        "- Topic: borealis.clicks.raw | Schema Registry ID: SCH_CLICK_0104 | Retention: 7 days.\n"
                        "- Broker v1 partition & codec settings: KAFKA_PARTITIONS_12 with SNAPPY_CODEC."
                    ),
                ),
            ],
        ),
        ExternalSession(
            session_id="p4_s02_flink_consumer",
            title="Flink Streaming Consumer & Checkpointing",
            date_time="2026-06-03",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Configure the Apache Flink consumer job, checkpoint bucket, and lag alerting for borealis.clicks.raw.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Borealis Flink Job Config (`flink_clicks_job.yaml`):\n"
                        "```yaml\n"
                        "job:\n"
                        "  name: borealis-clicks-aggregator\n"
                        "  upstream_spec: Broker v1\n"
                        "  parallelism: 12  # matches KAFKA_PARTITIONS_12\n"
                        "  watermark_out_of_orderness_sec: 10\n"
                        "checkpoints:\n"
                        "  backend: rocksdb_incremental\n"
                        "  interval_ms: 60000\n"
                        "  storage_path: s3://borealis-ckpt-usw2/prod/\n"
                        "alerts:\n"
                        "  consumer_lag_threshold: 500000\n"
                        "  lag_alert_code: ERR_BOREALIS_LAG_8810\n"
                        "```\n"
                        "Flink consumer currently assumes Broker v1 parallelism=12."
                    ),
                ),
            ],
        ),
        _abandoned_proto_session(
            session_id="p4_s03_quic_proto",
            title="Custom QUIC Ingest Proxy Prototype",
            date_time="2026-06-05",
            archive_reason="Corporate firewalls blocked UDP 7741 for 38% of enterprise clients",
            proposal=(
                "Test CUSTOM_QUIC_PROXY_7741 on UDP port 7741 with 0-RTT session tickets QUIC_TICKET_SEED_99 "
                "for mobile and enterprise telemetry beacons."
            ),
            benchmark_report=(
                "CUSTOM_QUIC_PROXY_7741 tested on UDP 7741: field telemetry showed 38% of enterprise corporate "
                "firewalls silently drop UDP 7741 datagrams, forcing slow fallback timeouts."
            ),
            cancellation=(
                "38% packet drop is a dealbreaker. Drop and archive CUSTOM_QUIC_PROXY_7741, UDP port 7741, "
                "and QUIC_TICKET_SEED_99 permanently—do NOT ship QUIC ingest."
            ),
            confirmation="Confirmed: CUSTOM_QUIC_PROXY_7741, UDP 7741, and QUIC_TICKET_SEED_99 archived and discarded.",
        ),
        _abandoned_proto_session(
            session_id="p4_s04_pulsar_proto",
            title="Apache Pulsar BookKeeper Multi-Tenant Prototype",
            date_time="2026-06-07",
            archive_reason="Operational complexity of ZooKeeper + BookKeeper + Broker exceeded team staffing",
            proposal=(
                "Evaluate PULSAR_BK_CLUSTER_V0 with bookie ensemble=5, write_quorum=3, and topic "
                "persistent://borealis/raw/clicks-v0."
            ),
            benchmark_report=(
                "PULSAR_BK_CLUSTER_V0 passed throughput tests, but BookKeeper ledger recovery during AZ loss "
                "required manual ensemble fencing and 3 separate stateful tiers to operate."
            ),
            cancellation=(
                "Too much operational overhead vs Redpanda. Scrap and archive PULSAR_BK_CLUSTER_V0 permanently."
            ),
            confirmation="Confirmed: PULSAR_BK_CLUSTER_V0 permanently scrapped and archived.",
        ),
        _abandoned_proto_session(
            session_id="p4_s05_avro_lz4_proto",
            title="Speculative LZ4 + Unversioned JSON Wire Prototype",
            date_time="2026-06-09",
            archive_reason="LZ4 bloated S3 tiered storage by 2.4x and raw JSON broke schema evolution",
            proposal=(
                "Test RAW_JSON_LZ4_INGEST_V0 bypassing Schema Registry SCH_CLICK_0104 to cut producer CPU."
            ),
            benchmark_report=(
                "RAW_JSON_LZ4_INGEST_V0 inflated tiered S3 storage 2.4x and caused 3 downstream Flink "
                "deserialization crashes when web SDK renamed `session_ms`."
            ),
            cancellation=(
                "Unacceptable schema breakage. Scrap and archive RAW_JSON_LZ4_INGEST_V0 permanently."
            ),
            confirmation="Confirmed: RAW_JSON_LZ4_INGEST_V0 scrapped and archived.",
        ),
        *_ops_noise_sessions(
            "p4",
            "Borealis Telemetry Bus",
            "2026-06-11",
            [
                ("tiered_s3", "Redpanda Tiered Storage Tuning", "12", "Enabled Redpanda cloud storage uploads to s3://borealis-tiered-usw2/; local NVMe cache retains 24h hot segments, 7d total retention."),
                ("schema_ci", "Protobuf Compatibility Gate", "14", "Added buf breaking check in CI against Schema Registry BACKWARD_TRANSITIVE mode; blocked 2 incompatible field removals."),
                ("rocksdb_tune", "Flink RocksDB Managed Memory", "16", "Tuned taskmanager.memory.managed.fraction=0.45, RocksDB block cache 512MB, write buffer 128MB; checkpoint duration dropped from 18s to 4.2s."),
                ("bot_filter", "Edge Bot Traffic Scrubbing", "18", "Added CloudFront WAF JA4 fingerprint rule filtering 19% headless scraper beacons before hitting borealis.clicks.raw."),
                ("iceberg_sink", "Iceberg Lakehouse Compaction", "20", "Hourly Iceberg manifest rewrite coalesces 256MB Parquet files; Athena dashboard query latency dropped 63%."),
                ("autoscaler", "Flink Kubernetes Operator Scaling", "22", "Enabled Flink Kubernetes Operator metrics collector; observed stable heap utilization at 58% under 95k events/sec."),
                ("cost_audit", "Cross-AZ Data Transfer Reduction", "24", "Enabled rack-aware Redpanda follower fetching (client.rack=az), cutting AWS cross-AZ transfer bills by $4,100/mo."),
            ],
        ),
        ExternalSession(
            session_id="p4_s14_broker_v2_reversal",
            title="Event Bus v2 Partition & Codec Upgrade",
            date_time="2026-06-27",
            turns=[
                ExternalTurn(
                    role="user",
                    content=(
                        "Peak holiday traffic saturated 12 partitions and snappy bandwidth. Reverse Broker v1: "
                        "scale borealis.clicks.raw on Redpanda to KAFKA_PARTITIONS_32 and switch compression "
                        "from SNAPPY_CODEC to ZSTD_LEVEL_4."
                    ),
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Borealis Broker v2 Locked (SUPERSEDES KAFKA_PARTITIONS_12 and SNAPPY_CODEC):\n"
                        "- Broker platform remains Redpanda; external ingest remains HTTPS/2 on TCP 443.\n"
                        "- Active partition count: KAFKA_PARTITIONS_32 on borealis.clicks.raw.\n"
                        "- Active compression codec: ZSTD_LEVEL_4 (replacing SNAPPY_CODEC).\n"
                        "- Mandatory downstream action: Flink consumer parallelism=12 is now STALE and must scale to 32."
                    ),
                ),
            ],
        ),
    ]
    projects.append(
        ExternalConversation(
            conversation_id="eng_04_borealis_telemetry",
            benchmark="non_monotonic",
            sessions=p4_sessions,
            qa_items=[
                ExternalQAItem(
                    question_id="p4_q1",
                    category="spec_reversal",
                    question="What partition count and compression codec are currently active for borealis.clicks.raw?",
                    ground_truth="KAFKA_PARTITIONS_32 and ZSTD_LEVEL_4",
                    distractor_answer="KAFKA_PARTITIONS_12 and SNAPPY_CODEC",
                    question_date="2026-06-30",
                ),
                ExternalQAItem(
                    question_id="p4_q2",
                    category="dead_end_quarantine",
                    question="Which UDP port and 0-RTT ticket seed does Borealis use for its production QUIC ingest proxy?",
                    ground_truth="None; CUSTOM_QUIC_PROXY_7741 (UDP 7741, QUIC_TICKET_SEED_99) was scrapped/archived due to enterprise firewalls blocking UDP.",
                    distractor_answer="UDP port 7741 with QUIC_TICKET_SEED_99",
                    is_abstention=True,
                    question_date="2026-06-30",
                ),
                ExternalQAItem(
                    question_id="p4_q3",
                    category="downstream_staleness",
                    question="Is the Borealis Flink consumer group's parallelism=12 still matched to the broker, or what must it scale to?",
                    ground_truth="Stale; it must scale from 12 to 32 to match KAFKA_PARTITIONS_32.",
                    distractor_answer="Still matched at parallelism=12",
                    question_date="2026-06-30",
                ),
                ExternalQAItem(
                    question_id="p4_q4",
                    category="verbatim_under_noise",
                    question="What exact S3 checkpoint bucket path and lag alert code are configured for Borealis Flink?",
                    ground_truth="s3://borealis-ckpt-usw2/prod/ and ERR_BOREALIS_LAG_8810",
                    question_date="2026-06-30",
                ),
                ExternalQAItem(
                    question_id="p4_q5",
                    category="verbatim_under_noise",
                    question="What exact Schema Registry ID was assigned to borealis.clicks.raw?",
                    ground_truth="SCH_CLICK_0104",
                    question_date="2026-06-30",
                ),
                ExternalQAItem(
                    question_id="p4_q6",
                    category="cross_branch_synthesis",
                    question="Summarize Borealis's active broker platform, partition count, codec, and external ingest transport.",
                    ground_truth="Redpanda, KAFKA_PARTITIONS_32, ZSTD_LEVEL_4, and HTTPS/2 on TCP 443 (no QUIC proxy).",
                    distractor_answer="KAFKA_PARTITIONS_12, SNAPPY_CODEC, and CUSTOM_QUIC_PROXY_7741",
                    question_date="2026-06-30",
                ),
            ],
        )
    )

    # ======================================================================
    # Project 5: Kronos ML Feature Store (14 sessions)
    # ======================================================================
    p5_sessions: list[ExternalSession] = [
        ExternalSession(
            session_id="p5_s01_online_v1",
            title="Kronos Online Feature Store v1",
            date_time="2026-07-01",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Design Kronos low-latency online feature store for real-time fraud scoring (<15ms SLA).",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Kronos Online Feature Store v1 Spec:\n"
                        "- Hard latency SLA: <15ms p99 online lookup.\n"
                        "- Backing online database: AWS DynamoDB single-table `kronos_feat_v1` (on-demand + DAX disabled).\n"
                        "- Partition key prefix: `FK_FRAUD_01` (format `FK_FRAUD_01#<entity_id>`).\n"
                        "- Online Store v1 wire encoding: JSON_UTF8_BLOB.\n"
                        "- Online Store v1 feature TTL: FEAT_TTL_7D (7 days)."
                    ),
                ),
            ],
        ),
        ExternalSession(
            session_id="p5_s02_scorer_client",
            title="Fraud Scorer SDK Integration",
            date_time="2026-07-03",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Wire the Python fraud scorer SDK to read from kronos_feat_v1 with circuit breaker fallbacks.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Kronos Fraud Scorer SDK Config (`scorer_client.yaml`):\n"
                        "```yaml\n"
                        "feature_store:\n"
                        "  upstream_spec: Online Store v1\n"
                        "  table: kronos_feat_v1\n"
                        "  wire_decoder: JSON_UTF8_BLOB\n"
                        "  read_consistency: eventual\n"
                        "resilience:\n"
                        "  circuit_breaker_timeout_ms: 50\n"
                        "  fallback_default_vector: ZERO_PRIOR_FALLBACK_V3\n"
                        "  circuit_open_metric_tag: kronos.scorer.cb_open\n"
                        "```\n"
                        "Note: Fraud Scorer SDK currently expects Online Store v1 JSON_UTF8_BLOB encoding."
                    ),
                ),
            ],
        ),
        _abandoned_proto_session(
            session_id="p5_s03_spark_thrift_proto",
            title="Spark Thrift Online Lookup Prototype",
            date_time="2026-07-05",
            archive_reason="Measured 410ms p99 cold query latency vs 15ms SLA",
            proposal=(
                "Test SPARK_THRIFT_ONLINE_10000 on JDBC port 10000 querying Delta table kronos_delta_live "
                "for direct online fraud feature serving."
            ),
            benchmark_report=(
                "SPARK_THRIFT_ONLINE_10000 on JDBC port 10000 hit 410ms p99 latency—27x slower than our "
                "15ms fraud SLA—due to Catalyst planning overhead."
            ),
            cancellation=(
                "410ms blows the 15ms SLA. Scrap and archive SPARK_THRIFT_ONLINE_10000 and online Delta lookups "
                "immediately—Spark is batch-only."
            ),
            confirmation="Confirmed: SPARK_THRIFT_ONLINE_10000 permanently scrapped and archived.",
        ),
        _abandoned_proto_session(
            session_id="p5_s04_redis_cluster_proto",
            title="Redis Enterprise Active-Active Prototype",
            date_time="2026-07-07",
            archive_reason="Licensed cluster cost $18,600/mo vs DynamoDB $1,900/mo at 8TB feature footprint",
            proposal=(
                "Evaluate REDIS_ENT_CRDT_V0 storing 8TB of 30-day velocity counters across 24 memory shards "
                "on cluster kronos-redis-exp04."
            ),
            benchmark_report=(
                "REDIS_ENT_CRDT_V0 hit 3.2ms p99, but 8TB DRAM + Enterprise CRDT licensing quoted $18,600/mo "
                "(9.8x DynamoDB cost)."
            ),
            cancellation=(
                "Prohibitive DRAM cost. Scrap and archive REDIS_ENT_CRDT_V0 and kronos-redis-exp04 permanently."
            ),
            confirmation="Confirmed: REDIS_ENT_CRDT_V0 and kronos-redis-exp04 scrapped and archived.",
        ),
        _abandoned_proto_session(
            session_id="p5_s05_msgpack_lzma_proto",
            title="MessagePack + LZMA Compression Prototype",
            date_time="2026-07-09",
            archive_reason="LZMA decompression added 9.4ms CPU overhead per batch request",
            proposal=(
                "Test MSGPACK_LZMA_LVL6 encoding on kronos_feat_v1 to shrink item size below 1KB WCU threshold."
            ),
            benchmark_report=(
                "MSGPACK_LZMA_LVL6 shrank items by 48%, but LZMA decompression added 9.4ms p99 CPU latency "
                "on Python scorer workers, threatening the 15ms total SLA."
            ),
            cancellation=(
                "Too slow on decode. Scrap and archive MSGPACK_LZMA_LVL6 permanently."
            ),
            confirmation="Confirmed: MSGPACK_LZMA_LVL6 permanently scrapped and archived.",
        ),
        *_ops_noise_sessions(
            "p5",
            "Kronos ML Feature Store",
            "2026-07-11",
            [
                ("backfill", "Offline Spark Backfill Pipeline", "12", "Nightly EMR Spark job computes 90-day training point-in-time joins into s3://kronos-offline-parquet/ with zero training-serving skew."),
                ("drift", "Feature Distribution Drift Monitor", "14", "Evidently PSI monitor runs every 6h across top 40 fraud features; Slack alert triggers if PSI > 0.25 on velocity_24h_amt."),
                ("iam_lock", "DynamoDB Least-Privilege IAM Policy", "16", "Scoped scorer ECS task role strictly to dynamodb:BatchGetItem and GetItem on kronos_feat_v1; denied Scan/DeleteTable."),
                ("canary_model", "Shadow XGBoost v4.2 Rollout", "18", "Ran XGBoost v4.2 in shadow mode over 12M transactions; AUC improved 0.941 -> 0.956 with identical feature lookup latency (7.8ms p99)."),
                ("hot_key", "Adaptive DynamoDB Burst Audit", "20", "Verified zero throttled read requests during Prime Day traffic spike (19,400 RCU/s peak); partition distribution Gini=0.14."),
                ("lineage", "Feature Registry Metadata Sync", "22", "Synced 112 feature definitions, owners, and PII classifications into DataHub catalog via CI webhook."),
                ("dr_table", "DynamoDB Global Tables Read Replica", "24", "Enabled Global Table replication to us-west-2 (average replication latency 680ms) for active-passive scorer failover."),
            ],
        ),
        ExternalSession(
            session_id="p5_s14_online_v2_reversal",
            title="Online Store v2 Binary Encoding & TTL Revision",
            date_time="2026-07-27",
            turns=[
                ExternalTurn(
                    role="user",
                    content=(
                        "JSON parsing blew scorer CPU and 7-day TTL misses 30-day slow-burn fraud rings! "
                        "Reverse Online Store v1: switch wire encoding to PROTOBUF_VARINT_V2 and extend TTL to FEAT_TTL_30D."
                    ),
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Kronos Online Store v2 Locked (SUPERSEDES JSON_UTF8_BLOB and FEAT_TTL_7D):\n"
                        "- Backing table remains DynamoDB `kronos_feat_v1` (<15ms SLA).\n"
                        "- Active wire encoding: PROTOBUF_VARINT_V2 (replacing JSON_UTF8_BLOB).\n"
                        "- Active feature TTL: FEAT_TTL_30D (replacing FEAT_TTL_7D).\n"
                        "- Mandatory downstream action: Fraud Scorer SDK (built on JSON_UTF8_BLOB) is now STALE "
                        "and must replace its JSON_UTF8_BLOB decoder with PROTOBUF_VARINT_V2."
                    ),
                ),
            ],
        ),
    ]
    projects.append(
        ExternalConversation(
            conversation_id="eng_05_kronos_features",
            benchmark="non_monotonic",
            sessions=p5_sessions,
            qa_items=[
                ExternalQAItem(
                    question_id="p5_q1",
                    category="spec_reversal",
                    question="What wire encoding and feature TTL are currently active for Kronos Online Feature Store?",
                    ground_truth="PROTOBUF_VARINT_V2 encoding and FEAT_TTL_30D",
                    distractor_answer="JSON_UTF8_BLOB and FEAT_TTL_7D",
                    question_date="2026-07-30",
                ),
                ExternalQAItem(
                    question_id="p5_q2",
                    category="dead_end_quarantine",
                    question="What JDBC port does Kronos use for Spark Thrift online fraud feature lookups in production?",
                    ground_truth="None; SPARK_THRIFT_ONLINE_10000 (port 10000) was scrapped/archived at 410ms p99 latency.",
                    distractor_answer="JDBC port 10000 querying kronos_delta_live",
                    is_abstention=True,
                    question_date="2026-07-30",
                ),
                ExternalQAItem(
                    question_id="p5_q3",
                    category="downstream_staleness",
                    question="Can the Kronos Fraud Scorer SDK keep decoding JSON_UTF8_BLOB, or what decoder does it need now?",
                    ground_truth="Stale; it must switch from JSON_UTF8_BLOB to PROTOBUF_VARINT_V2.",
                    distractor_answer="Keep JSON_UTF8_BLOB",
                    question_date="2026-07-30",
                ),
                ExternalQAItem(
                    question_id="p5_q4",
                    category="verbatim_under_noise",
                    question="What exact fallback vector name and DynamoDB key prefix were defined in Kronos?",
                    ground_truth="ZERO_PRIOR_FALLBACK_V3 and FK_FRAUD_01",
                    question_date="2026-07-30",
                ),
                ExternalQAItem(
                    question_id="p5_q5",
                    category="verbatim_under_noise",
                    question="What exact metrics tag and circuit breaker timeout are used by the Kronos Fraud Scorer SDK?",
                    ground_truth="kronos.scorer.cb_open and 50ms",
                    question_date="2026-07-30",
                ),
                ExternalQAItem(
                    question_id="p5_q6",
                    category="cross_branch_synthesis",
                    question="Summarize Kronos's active online database table, wire encoding, TTL, and latency SLA.",
                    ground_truth="DynamoDB table kronos_feat_v1, PROTOBUF_VARINT_V2, FEAT_TTL_30D, <15ms SLA.",
                    distractor_answer="JSON_UTF8_BLOB, FEAT_TTL_7D, and SPARK_THRIFT_ONLINE_10000",
                    question_date="2026-07-30",
                ),
            ],
        )
    )

    # ======================================================================
    # Project 6: Orion Edge CDN (14 sessions)
    # ======================================================================
    p6_sessions: list[ExternalSession] = [
        ExternalSession(
            session_id="p6_s01_origin_v1",
            title="Orion Origin Shield v1 & Edge WAF Runtime Contract",
            date_time="2026-08-01",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Specify Orion Edge CDN origin-shield transport, cache invalidation v1, and production edge WAF runtime.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Orion Edge CDN Origin Shield v1 & Security Baseline:\n"
                        "- Production Edge WAF Runtime (ACTIVE): Rust Wasmtime sandbox (strict memory-safe fuel-metered modules).\n"
                        "- Origin Shield v1 transport: HTTP/2 gRPC with GRPC_MAX_MSG_4MB.\n"
                        "- Origin Shield v1 purge propagation: POLLING_PULL_30S (30-second pull cadence).\n"
                        "- Origin TLS certificate pin: CERT_PIN_SHA256_9F2A.\n"
                        "- Edge node config bundle filename: orion_edge_v1.toml."
                    ),
                ),
            ],
        ),
        ExternalSession(
            session_id="p6_s02_purge_cli",
            title="Purge Control Plane & Operator CLI",
            date_time="2026-08-03",
            turns=[
                ExternalTurn(
                    role="user",
                    content="Implement the operator purge CLI and audit logging against Origin Shield v1.",
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Orion Operator Purge CLI (`purge_cli_spec.yaml`):\n"
                        "```yaml\n"
                        "control_plane:\n"
                        "  upstream_spec: Origin Shield v1\n"
                        "  purge_mode: POLLING_PULL_30S\n"
                        "  manifest_chunk_cap: GRPC_MAX_MSG_4MB\n"
                        "  oversize_error_code: ERR_ORION_MANIFEST_4130\n"
                        "audit:\n"
                        "  log_stream: orion.purge.audit\n"
                        "  retention_days: 365\n"
                        "```\n"
                        "Note: Operator Purge CLI currently assumes Origin Shield v1 (POLLING_PULL_30S and GRPC_MAX_MSG_4MB)."
                    ),
                ),
            ],
        ),
        _abandoned_proto_session(
            session_id="p6_s03_luajit_waf_proto",
            title="LuaJIT Custom WAF Filter Prototype",
            date_time="2026-08-05",
            archive_reason="Memory safety audit failed (2GB LuaJIT GC cap & native C FFI segfaults)",
            proposal=(
                "Prototype LUAJIT_WAF_FILTER_V0 with FFI hook ffi_inspect_hdr_v0 and 2GB slab LUA_SLAB_2G "
                "inside OpenResty edge workers."
            ),
            benchmark_report=(
                "LUAJIT_WAF_FILTER_V0 hit LuaJIT's 2GB GC address ceiling under large header bursts and "
                "triggered 4 native C FFI segfaults during fuzzing."
            ),
            cancellation=(
                "Memory safety failure. Scrap and archive LUAJIT_WAF_FILTER_V0, ffi_inspect_hdr_v0, "
                "and LUA_SLAB_2G permanently—do NOT deploy LuaJIT WAF filters."
            ),
            confirmation="Confirmed: LUAJIT_WAF_FILTER_V0, ffi_inspect_hdr_v0, and LUA_SLAB_2G scrapped and archived.",
        ),
        _abandoned_proto_session(
            session_id="p6_s04_bgp_multicast_purge_proto",
            title="Speculative BGP Flowspec Multicast Purge Prototype",
            date_time="2026-08-07",
            archive_reason="Transit IXPs filtered non-standard communities, dropping 22% of purge announcements",
            proposal=(
                "Test BGP_MULTICAST_PURGE_V0 broadcasting cache-tag invalidation hashes over private "
                "BGP community 65400:9912 across 38 edge POPs."
            ),
            benchmark_report=(
                "BGP_MULTICAST_PURGE_V0 suffered route-flap damping and 22% silent drops at tier-2 peering IXPs."
            ),
            cancellation=(
                "Unreliable propagation across peering fabrics. Scrap and archive BGP_MULTICAST_PURGE_V0 permanently."
            ),
            confirmation="Confirmed: BGP_MULTICAST_PURGE_V0 permanently scrapped and archived.",
        ),
        _abandoned_proto_session(
            session_id="p6_s05_varnish_vcl_proto",
            title="Varnish Cache VCL Origin-Shield Prototype",
            date_time="2026-08-09",
            archive_reason="Lacked native HTTP/3 and gRPC bidirectional streaming support",
            proposal=(
                "Evaluate VARNISH_VCL_SHIELD_V0 with ban lurker thread varnish_lurk_v0 and 64GB malloc storage."
            ),
            benchmark_report=(
                "VARNISH_VCL_SHIELD_V0 handled static HTTP/1.1 well, but could not proxy gRPC trailers "
                "or terminate QUIC HTTP/3 without an extra proxy hop."
            ),
            cancellation=(
                "Missing gRPC/H3 support. Scrap and archive VARNISH_VCL_SHIELD_V0 and varnish_lurk_v0 permanently."
            ),
            confirmation="Confirmed: VARNISH_VCL_SHIELD_V0 permanently scrapped and archived.",
        ),
        *_ops_noise_sessions(
            "p6",
            "Orion Edge CDN",
            "2026-08-11",
            [
                ("anycast", "Anycast BGP PoP Weighting", "12", "Tuned MED and AS-path prepending across 38 PoPs; shifted 18% Frankfurt spillover traffic cleanly to Amsterdam during peak video premiere."),
                ("tls_ocsp", "OCSP Stapling & Cert Auto-Renewal", "14", "Automated ACME DNS-01 certificate renewals 30 days prior to expiry; OCSP staple hit rate 99.94% across edge TLS handshakes."),
                ("disk_cache", "NVMe Ring-Buffer Cache Tuning", "16", "Direct-IO 64KB aligned ring cache achieved 94.6% byte hit ratio on HLS video segments with zero page-cache eviction stalls."),
                ("ddos_drill", "L7 HTTP/2 Rapid-Reset Mitigation", "18", "Verified CVE-2023-44487 rapid-reset limiter caps RST_STREAM rate at 200/s per connection; simulated 1.2M RPS flood absorbed with <2ms latency bump."),
                ("wasm_fuel", "Wasmtime Fuel Budget Calibration", "20", "Calibrated Rust Wasmtime per-request fuel limit to 250,000 instructions (max 0.4ms execution) with pre-compiled AOT `.cwasm` artifacts."),
                ("log_ship", "Real-Time Edge Access Log Sampling", "22", "ClickHouse edge log cluster ingests 1.4M log lines/sec via Vector agents with zstd compression; 30-day hot analytics retention."),
                ("origin_hc", "Origin Shield Active Health Checks", "24", "Configured 5s active HTTP/2 health probes with passive circuit breaking (eject origin for 30s after 5 consecutive 502/504 responses)."),
            ],
        ),
        ExternalSession(
            session_id="p6_s14_origin_v2_reversal",
            title="Origin Shield v2 Push Purge & Payload Revision",
            date_time="2026-08-27",
            turns=[
                ExternalTurn(
                    role="user",
                    content=(
                        "30-second stale window failed enterprise media SLA and 4MB manifests broke live video catalogs! "
                        "Reverse Origin Shield v1: switch to NATS_JETSTREAM_PUSH_2S for <2s global purges "
                        "and raise gRPC limit to GRPC_MAX_MSG_16MB."
                    ),
                ),
                ExternalTurn(
                    role="assistant",
                    content=(
                        "Orion Origin Shield v2 Locked (SUPERSEDES POLLING_PULL_30S and GRPC_MAX_MSG_4MB):\n"
                        "- Active purge propagation: NATS_JETSTREAM_PUSH_2S (<2s global fanout).\n"
                        "- Active gRPC max message size: GRPC_MAX_MSG_16MB (replacing GRPC_MAX_MSG_4MB).\n"
                        "- Production Edge WAF runtime remains Rust Wasmtime.\n"
                        "- Mandatory downstream action: Operator Purge CLI (built on POLLING_PULL_30S / 4MB) "
                        "is now STALE and must migrate to NATS_JETSTREAM_PUSH_2S and GRPC_MAX_MSG_16MB."
                    ),
                ),
            ],
        ),
    ]
    projects.append(
        ExternalConversation(
            conversation_id="eng_06_orion_cdn",
            benchmark="non_monotonic",
            sessions=p6_sessions,
            qa_items=[
                ExternalQAItem(
                    question_id="p6_q1",
                    category="spec_reversal",
                    question="What purge propagation protocol and gRPC max message size are currently active in Orion Edge CDN?",
                    ground_truth="NATS_JETSTREAM_PUSH_2S and GRPC_MAX_MSG_16MB",
                    distractor_answer="POLLING_PULL_30S and GRPC_MAX_MSG_4MB",
                    question_date="2026-08-30",
                ),
                ExternalQAItem(
                    question_id="p6_q2",
                    category="dead_end_quarantine",
                    question="What LuaJIT FFI hook and memory slab size run Orion's production edge WAF?",
                    ground_truth="None; LUAJIT_WAF_FILTER_V0 (ffi_inspect_hdr_v0, LUA_SLAB_2G) was scrapped/archived due to memory safety issues; production uses Rust Wasmtime.",
                    distractor_answer="ffi_inspect_hdr_v0 with LUA_SLAB_2G",
                    is_abstention=True,
                    question_date="2026-08-30",
                ),
                ExternalQAItem(
                    question_id="p6_q3",
                    category="downstream_staleness",
                    question="Is the Orion Purge CLI's 30-second polling / 4MB manifest split still current, or what must it migrate to?",
                    ground_truth="Stale; migrate to NATS_JETSTREAM_PUSH_2S (<2s push) and GRPC_MAX_MSG_16MB.",
                    distractor_answer="Still current at POLLING_PULL_30S and GRPC_MAX_MSG_4MB",
                    question_date="2026-08-30",
                ),
                ExternalQAItem(
                    question_id="p6_q4",
                    category="verbatim_under_noise",
                    question="What exact origin certificate pin and manifest oversize error code were specified for Orion CDN?",
                    ground_truth="CERT_PIN_SHA256_9F2A and ERR_ORION_MANIFEST_4130",
                    question_date="2026-08-30",
                ),
                ExternalQAItem(
                    question_id="p6_q5",
                    category="verbatim_under_noise",
                    question="What exact audit log stream name and edge config filename are used by Orion CDN?",
                    ground_truth="orion.purge.audit and orion_edge_v1.toml",
                    question_date="2026-08-30",
                ),
                ExternalQAItem(
                    question_id="p6_q6",
                    category="cross_branch_synthesis",
                    question="Summarize Orion's active purge mechanism, gRPC message cap, and production WAF runtime.",
                    ground_truth="NATS_JETSTREAM_PUSH_2S, GRPC_MAX_MSG_16MB, and Rust Wasmtime (not LuaJIT).",
                    distractor_answer="POLLING_PULL_30S, GRPC_MAX_MSG_4MB, and LUAJIT_WAF_FILTER_V0",
                    question_date="2026-08-30",
                ),
            ],
        )
    )

    return projects
