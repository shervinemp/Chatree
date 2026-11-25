from __future__ import annotations

from chatree.eval.external_benchmarks import (
    ExternalConversation,
    ExternalQAItem,
    ExternalSession,
    ExternalTurn,
)


def build_non_monotonic_engineering_benchmark() -> list[ExternalConversation]:
    """
    Track 7: Realistic Non-Monotonic Software Engineering & Spec-Reversal Benchmark.

    Directly evaluates the five failure modes that passive casual-chat benchmarks
    (LoCoMo / LongMemEval) under-test, across 6 realistic multi-session engineering
    projects (36 QA items across 5 categories):
      1. `spec_reversal`          — Early settled decision later materially reversed.
      2. `dead_end_quarantine`    — Deep speculative prototype explicitly scrapped/archived.
      3. `downstream_staleness`   — Downstream service built on v1 must reconcile to v2.
      4. `verbatim_under_noise`   — Exact IDs/migrations/error codes buried under distractor noise.
      5. `cross_branch_synthesis` — Fusing current settled choices across multiple active branches.

    Fairness guarantee: Every scrapped prototype session ends with an explicit textual
    cancellation turn ingested identically by all baselines (Mem0, Flat-RAG, Full-Context,
    RLM, Letta, Graphiti). Chatree additionally invokes its native `archive_branch()`
    quarantine primitive via `ExternalSession.archived=True`.
    """
    projects: list[ExternalConversation] = [
        # ------------------------------------------------------------------
        # Project 1: Nebula API Gateway (Auth reversal + GraphQL dead-end)
        # ------------------------------------------------------------------
        ExternalConversation(
            conversation_id="eng_01_nebula_gateway",
            benchmark="non_monotonic",
            sessions=[
                ExternalSession(
                    session_id="p1_s1_auth_v1",
                    title="Auth Token Spec v1",
                    date_time="2026-03-01",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "Let's lock down the initial authentication spec for Nebula API Gateway. "
                                "Hard SLO is P99_LATENCY_120MS on Rust/Axum."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Initial Auth v1 spec recorded: sign access tokens with RS256_2048_RSA, "
                                "set JWT_TTL_15M, and publish keys at /.well-known/jwks.json with 24h cache. "
                                "Migration file 0014_create_api_keys_idx.sql adds composite index idx_apikeys_org_hash."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p1_s2_ratelimit",
                    title="Rate Limiter & Gateway Integration",
                    date_time="2026-03-03",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "Building the API Gateway rate-limiter middleware. Based on Auth v1, "
                                "we'll verify RS256_2048_RSA tokens with a 15-minute local key cache matching JWT_TTL_15M "
                                "and enforce leaky-bucket burst limit 250 req/s per tenant."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Recorded Gateway middleware config: tenant burst 250 req/s, rejection status 429 "
                                "with structured code ERR_NEBULA_BURST_4291, Redis sliding-window store on Valkey 8.0 "
                                "(port 6380, TLS 1.3). Downstream auth assumption currently references Auth v1 (RS256_2048_RSA, JWT_TTL_15M)."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p1_s3_graphql_proto",
                    title="Speculative GraphQL Federation Prototype",
                    date_time="2026-03-05",
                    archived=True,
                    archive_reason="Violated P99_LATENCY_120MS SLO (measured 248ms p99 due to 24 subgraph hops)",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "Let's prototype GRAPHQL_FEDERATION_V0 using Apollo Router v2 on port 4099 "
                                "with supergraph schema nebula_supergraph_v0.graphql and persisted query cache GQL_APQ_LRU_50K."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Prototype GRAPHQL_FEDERATION_V0 scaffolded with Apollo Router port 4099, "
                                "subgraph batch window 12ms, and GQL_APQ_LRU_50K. Load test shows 24 subgraph hops "
                                "producing 248ms p99 latency under 1,000 RPS."
                            ),
                        ),
                        ExternalTurn(
                            role="user",
                            content=(
                                "248ms blows past our hard P99_LATENCY_120MS SLO. Scrap and archive the entire "
                                "GRAPHQL_FEDERATION_V0 prototype immediately—do NOT deploy Apollo Router port 4099 "
                                "or GQL_APQ_LRU_50K in production. We are sticking strictly to REST + gRPC."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Confirmed: GRAPHQL_FEDERATION_V0, Apollo Router port 4099, and GQL_APQ_LRU_50K "
                                "are permanently scrapped and archived. Production remains REST + gRPC only."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p1_s4_telemetry_noise",
                    title="Observability & Dashboards",
                    date_time="2026-03-07",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Configure OpenTelemetry tracing with OTLP gRPC exporter to Tempo and Prometheus scrape interval 15s.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Telemetry configured: OTLP gRPC exporter on port 4317, tail-based sampling 5% normal / 100% errors, "
                                "Grafana dashboard nebula-ops-v2.json, alert webhook PagerDuty tier-1."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p1_s5_auth_v2_reversal",
                    title="Auth Spec v2 Security Revision",
                    date_time="2026-03-10",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "Security audit flagged RS256_2048_RSA verification overhead and 15-minute token window as too wide. "
                                "Reverse Auth v1: migrate immediately to ED25519_SIGNING with JWT_TTL_5M_ROTATING "
                                "and opaque refresh tokens hashed via BLAKE3_REFRESH_HASH."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Auth v2 locked (superseding Auth v1): ED25519_SIGNING replaces RS256_2048_RSA; "
                                "JWT_TTL_5M_ROTATING replaces JWT_TTL_15M; refresh tokens stored with BLAKE3_REFRESH_HASH. "
                                "All downstream consumers (including API Gateway middleware) must drop RS256_2048_RSA and 15m TTL."
                            ),
                        ),
                    ],
                ),
            ],
            qa_items=[
                ExternalQAItem(
                    question_id="p1_q1",
                    category="spec_reversal",
                    question="What JWT signing algorithm and token TTL are currently active for Nebula API Gateway?",
                    ground_truth="ED25519_SIGNING with JWT_TTL_5M_ROTATING",
                    distractor_answer="RS256_2048_RSA with JWT_TTL_15M",
                    question_date="2026-03-12",
                ),
                ExternalQAItem(
                    question_id="p1_q2",
                    category="dead_end_quarantine",
                    question="What Apollo Router port and persisted query cache are deployed in production for Nebula's GraphQL federation?",
                    ground_truth="None; GRAPHQL_FEDERATION_V0 (Apollo Router port 4099, GQL_APQ_LRU_50K) was scrapped/archived because it violated the P99_LATENCY_120MS SLO.",
                    distractor_answer="Apollo Router on port 4099 with GQL_APQ_LRU_50K",
                    is_abstention=True,
                    question_date="2026-03-12",
                ),
                ExternalQAItem(
                    question_id="p1_q3",
                    category="downstream_staleness",
                    question="In the API Gateway rate-limiter middleware, is its earlier RS256_2048_RSA / 15-minute token assumption still valid, or what must it update to?",
                    ground_truth="Stale/invalid; it must update to ED25519_SIGNING and JWT_TTL_5M_ROTATING from Auth v2.",
                    distractor_answer="Valid; keep RS256_2048_RSA and JWT_TTL_15M",
                    question_date="2026-03-12",
                ),
                ExternalQAItem(
                    question_id="p1_q4",
                    category="verbatim_under_noise",
                    question="What exact SQL migration filename and composite index name were created for Nebula API keys?",
                    ground_truth="0014_create_api_keys_idx.sql and idx_apikeys_org_hash",
                    question_date="2026-03-12",
                ),
                ExternalQAItem(
                    question_id="p1_q5",
                    category="verbatim_under_noise",
                    question="What exact structured error code does the rate-limiter return on HTTP 429 burst rejections?",
                    ground_truth="ERR_NEBULA_BURST_4291",
                    question_date="2026-03-12",
                ),
                ExternalQAItem(
                    question_id="p1_q6",
                    category="cross_branch_synthesis",
                    question="Summarize Nebula's active production auth signing, rate-limit store, and transport protocol (excluding any scrapped prototypes).",
                    ground_truth="ED25519_SIGNING (JWT_TTL_5M_ROTATING), Valkey 8.0 on port 6380 (TLS 1.3), and REST + gRPC only.",
                    distractor_answer="RS256_2048_RSA and GRAPHQL_FEDERATION_V0",
                    question_date="2026-03-12",
                ),
            ],
        ),
        # ------------------------------------------------------------------
        # Project 2: Helios Vector Engine (Index reversal + Mongo dead-end)
        # ------------------------------------------------------------------
        ExternalConversation(
            conversation_id="eng_02_helios_vector",
            benchmark="non_monotonic",
            sessions=[
                ExternalSession(
                    session_id="p2_s1_storage_v1",
                    title="Vector Storage Engine v1",
                    date_time="2026-04-01",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Design Helios Vector Engine for 50M 768-dim embeddings with recall@10 >= 0.96 and RAM cap 64GB.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Initial Storage v1 design: IVF_FLAT_4096_LISTS with float32 uncompressed vectors "
                                "and nprobe=32. Checkpoint format magic header HELIOS_IDX_V1_0x484C."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p2_s2_query_planner",
                    title="Query Planner & Recall Tuning",
                    date_time="2026-04-03",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Write the Helios query planner assuming IVF_FLAT_4096_LISTS and nprobe=32 with SIMD AVX-512 dot product.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Query planner drafted: expects IVF_FLAT_4096_LISTS with nprobe=32, pre-filter bitmap threshold "
                                "SELECTIVITY_CUTOFF_0_08, and timeout sentinel ERR_HELIOS_SCAN_TIMEOUT_5042."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p2_s3_mongo_sharding_proto",
                    title="MongoDB Atlas Sharded Vector Prototype",
                    date_time="2026-04-05",
                    archived=True,
                    archive_reason="Exceeded 64GB RAM cap and cost $9,400/mo on benchmark cluster",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "Prototype MONGO_SHARD_PROTO_X9 using 6 shard replicas, shard key tenant_hash_128, "
                                "and lucene_hnsw_m16_ef100 on cluster helios-mongo-exp09."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Prototype MONGO_SHARD_PROTO_X9 deployed on helios-mongo-exp09 with lucene_hnsw_m16_ef100. "
                                "Projected monthly cost is $9,400 and memory footprint hit 142GB."
                            ),
                        ),
                        ExternalTurn(
                            role="user",
                            content=(
                                "Way over budget and over the 64GB RAM cap. Abandon and archive MONGO_SHARD_PROTO_X9 "
                                "and helios-mongo-exp09 completely—MongoDB will NOT be used in Helios."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content="Archived MONGO_SHARD_PROTO_X9 and helios-mongo-exp09; MongoDB permanently rejected.",
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p2_s4_ingestor_noise",
                    title="Batch Ingestion Pipeline",
                    date_time="2026-04-06",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Set up Arrow IPC batch ingestion with batch size 8,192 rows and backpressure queue depth 32.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content="Arrow IPC ingestor configured: 8,192-row record batches, queue depth 32, zstd level 3 compression.",
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p2_s5_storage_v2_reversal",
                    title="Storage Engine v2 Quantization Overhaul",
                    date_time="2026-04-09",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "50M float32 vectors in IVF_FLAT_4096_LISTS need 153GB RAM—impossible inside our 64GB cap! "
                                "Reverse Storage v1: adopt DISKANN_VAMANA_PQ64 with R=64, L_search=128, and int8 rescoring from NVMe."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Storage v2 finalized (superseding IVF_FLAT_4096_LISTS and nprobe=32): "
                                "DISKANN_VAMANA_PQ64 (R=64, L_search=128) with NVMe int8 rescoring fits in 38GB RAM at 0.971 recall@10. "
                                "Query planner must retire IVF_FLAT_4096_LISTS / nprobe=32 and target DISKANN_VAMANA_PQ64."
                            ),
                        ),
                    ],
                ),
            ],
            qa_items=[
                ExternalQAItem(
                    question_id="p2_q1",
                    category="spec_reversal",
                    question="Which vector index algorithm and search parameters are currently active for Helios Vector Engine?",
                    ground_truth="DISKANN_VAMANA_PQ64 with R=64, L_search=128, and NVMe int8 rescoring",
                    distractor_answer="IVF_FLAT_4096_LISTS with nprobe=32",
                    question_date="2026-04-12",
                ),
                ExternalQAItem(
                    question_id="p2_q2",
                    category="dead_end_quarantine",
                    question="What MongoDB shard key and cluster name are running Helios in production?",
                    ground_truth="None; MONGO_SHARD_PROTO_X9 (helios-mongo-exp09, tenant_hash_128) was abandoned/archived for exceeding the 64GB RAM cap and budget.",
                    distractor_answer="tenant_hash_128 on helios-mongo-exp09",
                    is_abstention=True,
                    question_date="2026-04-12",
                ),
                ExternalQAItem(
                    question_id="p2_q3",
                    category="downstream_staleness",
                    question="Does the Helios query planner still need nprobe=32 over IVF_FLAT_4096_LISTS, or what index must it target now?",
                    ground_truth="Stale; retire IVF_FLAT_4096_LISTS and nprobe=32 and target DISKANN_VAMANA_PQ64 (R=64, L_search=128).",
                    distractor_answer="Keep IVF_FLAT_4096_LISTS and nprobe=32",
                    question_date="2026-04-12",
                ),
                ExternalQAItem(
                    question_id="p2_q4",
                    category="verbatim_under_noise",
                    question="What exact checkpoint format magic header and timeout sentinel error code were specified for Helios?",
                    ground_truth="HELIOS_IDX_V1_0x484C and ERR_HELIOS_SCAN_TIMEOUT_5042",
                    question_date="2026-04-12",
                ),
                ExternalQAItem(
                    question_id="p2_q5",
                    category="verbatim_under_noise",
                    question="What exact pre-filter bitmap selectivity cutoff constant was set in the query planner?",
                    ground_truth="SELECTIVITY_CUTOFF_0_08",
                    question_date="2026-04-12",
                ),
                ExternalQAItem(
                    question_id="p2_q6",
                    category="cross_branch_synthesis",
                    question="What are Helios's active index engine, batch ingestion format/size, and measured RAM footprint?",
                    ground_truth="DISKANN_VAMANA_PQ64 (38GB RAM) fed by Arrow IPC 8,192-row batches (zstd level 3).",
                    distractor_answer="IVF_FLAT_4096_LISTS and MONGO_SHARD_PROTO_X9",
                    question_date="2026-04-12",
                ),
            ],
        ),
        # ------------------------------------------------------------------
        # Project 3: Aegis Payments Ledger (Isolation reversal + Blockchain dead-end)
        # ------------------------------------------------------------------
        ExternalConversation(
            conversation_id="eng_03_aegis_ledger",
            benchmark="non_monotonic",
            sessions=[
                ExternalSession(
                    session_id="p3_s1_ledger_v1",
                    title="Double-Entry Ledger Schema v1",
                    date_time="2026-05-01",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Design Aegis double-entry payments ledger on Postgres 17 with zero lost transfers.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Ledger v1 locked: READ_COMMITTED_ISOLATION with advisory locks per account_id, "
                                "idempotency table ledger_idempotency_keys (TTL 72h), and migration 0007_ledger_entries_part.sql."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p3_s2_settlement_worker",
                    title="Settlement Worker & Retry Policy",
                    date_time="2026-05-03",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "Build the settlement worker assuming READ_COMMITTED_ISOLATION with advisory locks "
                                "and exponential backoff jitter=0.35s, base=1.5s, cap=45s."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Settlement worker spec saved: backoff jitter=0.35s, base=1.5s, cap=45s; "
                                "dead-letter queue topic aegis.settlement.dlq.v1; retry code ERR_AEGIS_LOCK_4093. "
                                "Currently assumes Ledger v1 READ_COMMITTED_ISOLATION."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p3_s3_hyperledger_proto",
                    title="Permissioned Blockchain Ledger Prototype",
                    date_time="2026-05-04",
                    archived=True,
                    archive_reason="Failed 2,000 TPS throughput target (maxed at 140 TPS) and added 4.8s finality lag",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "Explore HYPERLEDGER_FABRIC_PROTO_B3 with RAFT_ORDERER_5NODES and chaincode aegis_cc_v03 "
                                "on channel paynet-internal."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "HYPERLEDGER_FABRIC_PROTO_B3 benchmarked on channel paynet-internal with chaincode aegis_cc_v03: "
                                "throughput plateaued at 140 TPS with 4.8s commit finality."
                            ),
                        ),
                        ExternalTurn(
                            role="user",
                            content=(
                                "140 TPS is unusable for our 2,000 TPS target. Scrap and archive HYPERLEDGER_FABRIC_PROTO_B3, "
                                "RAFT_ORDERER_5NODES, and aegis_cc_v03—we are NOT shipping a blockchain ledger."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content="Confirmed: HYPERLEDGER_FABRIC_PROTO_B3 and aegis_cc_v03 archived and permanently discarded.",
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p3_s4_ledger_v2_reversal",
                    title="Ledger v2 Serializable Upgrade",
                    date_time="2026-05-08",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "Chaos testing found a write-skew phantom debit under READ_COMMITTED_ISOLATION! "
                                "Reverse Ledger v1: mandate SERIALIZABLE_SSI_ISOLATION with deterministic account-ID ordering "
                                "and automatic SQLSTATE 40001 retry loop (max 5 attempts)."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Ledger v2 locked (replacing READ_COMMITTED_ISOLATION): SERIALIZABLE_SSI_ISOLATION "
                                "with deterministic account-ID lock order and 5x SQLSTATE 40001 retry. "
                                "Settlement worker must drop READ_COMMITTED_ISOLATION and handle SQLSTATE 40001 retries."
                            ),
                        ),
                    ],
                ),
            ],
            qa_items=[
                ExternalQAItem(
                    question_id="p3_q1",
                    category="spec_reversal",
                    question="What Postgres isolation level and retry rule are currently mandated for Aegis Ledger?",
                    ground_truth="SERIALIZABLE_SSI_ISOLATION with deterministic account-ID ordering and 5x SQLSTATE 40001 retry loop",
                    distractor_answer="READ_COMMITTED_ISOLATION with advisory locks",
                    question_date="2026-05-11",
                ),
                ExternalQAItem(
                    question_id="p3_q2",
                    category="dead_end_quarantine",
                    question="What Hyperledger Fabric chaincode and channel name run Aegis production settlements?",
                    ground_truth="None; HYPERLEDGER_FABRIC_PROTO_B3 (aegis_cc_v03 on paynet-internal) was scrapped/archived at 140 TPS.",
                    distractor_answer="aegis_cc_v03 on channel paynet-internal",
                    is_abstention=True,
                    question_date="2026-05-11",
                ),
                ExternalQAItem(
                    question_id="p3_q3",
                    category="downstream_staleness",
                    question="Is the settlement worker's READ_COMMITTED_ISOLATION assumption still valid, and what must it handle now?",
                    ground_truth="Stale/invalid due to write-skew; must upgrade to SERIALIZABLE_SSI_ISOLATION and handle SQLSTATE 40001 retries.",
                    distractor_answer="Valid; continue using READ_COMMITTED_ISOLATION",
                    question_date="2026-05-11",
                ),
                ExternalQAItem(
                    question_id="p3_q4",
                    category="verbatim_under_noise",
                    question="What exact exponential backoff parameters (jitter, base, cap) and DLQ topic were configured for the settlement worker?",
                    ground_truth="jitter=0.35s, base=1.5s, cap=45s; DLQ topic aegis.settlement.dlq.v1",
                    question_date="2026-05-11",
                ),
                ExternalQAItem(
                    question_id="p3_q5",
                    category="verbatim_under_noise",
                    question="What exact migration filename and retry error code belong to Aegis Ledger?",
                    ground_truth="0007_ledger_entries_part.sql and ERR_AEGIS_LOCK_4093",
                    question_date="2026-05-11",
                ),
                ExternalQAItem(
                    question_id="p3_q6",
                    category="cross_branch_synthesis",
                    question="Combine Aegis's active database engine, isolation level, idempotency TTL, and DLQ topic.",
                    ground_truth="Postgres 17, SERIALIZABLE_SSI_ISOLATION, 72h idempotency TTL (ledger_idempotency_keys), DLQ aegis.settlement.dlq.v1.",
                    distractor_answer="READ_COMMITTED_ISOLATION and HYPERLEDGER_FABRIC_PROTO_B3",
                    question_date="2026-05-11",
                ),
            ],
        ),
        # ------------------------------------------------------------------
        # Project 4: Borealis Event Bus (Partition reversal + Custom QUIC dead-end)
        # ------------------------------------------------------------------
        ExternalConversation(
            conversation_id="eng_04_borealis_bus",
            benchmark="non_monotonic",
            sessions=[
                ExternalSession(
                    session_id="p4_s1_broker_v1",
                    title="Event Bus Partitioning v1",
                    date_time="2026-06-01",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Configure Borealis Event Bus on Redpanda for real-time clickstream analytics.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Broker v1 spec: KAFKA_PARTITIONS_12 on topic borealis.clicks.raw, "
                                "compression SNAPPY_CODEC, retention 72h, schema registry ID SCH_CLICK_0104."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p4_s2_flink_consumer",
                    title="Flink Consumer Group Config",
                    date_time="2026-06-02",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Size the Flink consumer group to 12 parallel slots matching KAFKA_PARTITIONS_12 with 30s checkpoint interval.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Flink consumer config recorded: parallelism=12 (tied to KAFKA_PARTITIONS_12), "
                                "RocksDB incremental checkpointing every 30s to s3://borealis-ckpt-usw2/prod/, "
                                "lag alert code ERR_BOREALIS_LAG_8810."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p4_s3_quic_proxy_proto",
                    title="Custom QUIC Ingest Proxy Prototype",
                    date_time="2026-06-04",
                    archived=True,
                    archive_reason="Corporate firewalls blocked UDP 7741 for 38% of enterprise clients",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Test CUSTOM_QUIC_PROXY_7741 on UDP port 7741 with 0-RTT session tickets QUIC_TICKET_SEED_99.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "CUSTOM_QUIC_PROXY_7741 tested on UDP 7741. Telemetry shows 38% of enterprise corporate "
                                "firewalls drop UDP 7741 packets silently."
                            ),
                        ),
                        ExternalTurn(
                            role="user",
                            content="Drop and archive CUSTOM_QUIC_PROXY_7741 and UDP port 7741 permanently—stay on HTTPS/2 TCP 443.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content="Archived CUSTOM_QUIC_PROXY_7741 (UDP 7741, QUIC_TICKET_SEED_99). Production stays on HTTPS/2 TCP 443.",
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p4_s4_broker_v2_reversal",
                    title="Event Bus Partition & Codec Upgrade v2",
                    date_time="2026-06-07",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "Peak traffic saturated 12 partitions and snappy bandwidth. Reverse Broker v1: "
                                "scale borealis.clicks.raw to KAFKA_PARTITIONS_32 and switch compression from SNAPPY_CODEC to ZSTD_LEVEL_4."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Broker v2 locked (superseding KAFKA_PARTITIONS_12 and SNAPPY_CODEC): "
                                "KAFKA_PARTITIONS_32 with ZSTD_LEVEL_4 compression. Flink consumer parallelism=12 is now stale and must scale to 32."
                            ),
                        ),
                    ],
                ),
            ],
            qa_items=[
                ExternalQAItem(
                    question_id="p4_q1",
                    category="spec_reversal",
                    question="What partition count and compression codec are currently active for borealis.clicks.raw?",
                    ground_truth="KAFKA_PARTITIONS_32 and ZSTD_LEVEL_4",
                    distractor_answer="KAFKA_PARTITIONS_12 and SNAPPY_CODEC",
                    question_date="2026-06-10",
                ),
                ExternalQAItem(
                    question_id="p4_q2",
                    category="dead_end_quarantine",
                    question="Which UDP port and 0-RTT ticket seed does Borealis use for its production QUIC ingest proxy?",
                    ground_truth="None; CUSTOM_QUIC_PROXY_7741 (UDP 7741, QUIC_TICKET_SEED_99) was scrapped/archived due to enterprise firewalls blocking UDP.",
                    distractor_answer="UDP port 7741 with QUIC_TICKET_SEED_99",
                    is_abstention=True,
                    question_date="2026-06-10",
                ),
                ExternalQAItem(
                    question_id="p4_q3",
                    category="downstream_staleness",
                    question="Is the Flink consumer group's parallelism=12 still matched to the broker, or what must it scale to?",
                    ground_truth="Stale; it must scale from 12 to 32 to match KAFKA_PARTITIONS_32.",
                    distractor_answer="Still matched at parallelism=12",
                    question_date="2026-06-10",
                ),
                ExternalQAItem(
                    question_id="p4_q4",
                    category="verbatim_under_noise",
                    question="What exact S3 checkpoint bucket path and lag alert code are configured for Borealis Flink?",
                    ground_truth="s3://borealis-ckpt-usw2/prod/ and ERR_BOREALIS_LAG_8810",
                    question_date="2026-06-10",
                ),
                ExternalQAItem(
                    question_id="p4_q5",
                    category="verbatim_under_noise",
                    question="What exact schema registry ID was assigned to borealis.clicks.raw?",
                    ground_truth="SCH_CLICK_0104",
                    question_date="2026-06-10",
                ),
                ExternalQAItem(
                    question_id="p4_q6",
                    category="cross_branch_synthesis",
                    question="Summarize Borealis's active broker platform, partition count, codec, and external ingest transport.",
                    ground_truth="Redpanda, KAFKA_PARTITIONS_32, ZSTD_LEVEL_4, and HTTPS/2 on TCP 443 (no QUIC proxy).",
                    distractor_answer="KAFKA_PARTITIONS_12, SNAPPY_CODEC, and CUSTOM_QUIC_PROXY_7741",
                    question_date="2026-06-10",
                ),
            ],
        ),
        # ------------------------------------------------------------------
        # Project 5: Kronos ML Feature Store (TTL reversal + Spark Thrift dead-end)
        # ------------------------------------------------------------------
        ExternalConversation(
            conversation_id="eng_05_kronos_features",
            benchmark="non_monotonic",
            sessions=[
                ExternalSession(
                    session_id="p5_s1_online_v1",
                    title="Online Feature Store v1",
                    date_time="2026-07-01",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Design Kronos low-latency online feature store for fraud scoring (<15ms SLA).",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Online Store v1: DynamoDB single-table kronos_feat_v1, feature TTL FEAT_TTL_7D, "
                                "encoding JSON_UTF8_BLOB, key prefix FK_FRAUD_01."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p5_s2_scorer_client",
                    title="Fraud Scorer SDK Integration",
                    date_time="2026-07-03",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Wire the Python fraud scorer to decode JSON_UTF8_BLOB from kronos_feat_v1 with circuit breaker 50ms.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Fraud scorer client configured: decodes JSON_UTF8_BLOB, circuit breaker 50ms, "
                                "fallback default vector ZERO_PRIOR_FALLBACK_V3, metrics tag kronos.scorer.cb_open."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p5_s3_spark_thrift_proto",
                    title="Spark Thrift Online Lookup Prototype",
                    date_time="2026-07-05",
                    archived=True,
                    archive_reason="Measured 410ms p99 cold query latency vs 15ms SLA",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Test SPARK_THRIFT_ONLINE_10000 on JDBC port 10000 querying Delta table kronos_delta_live.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content="SPARK_THRIFT_ONLINE_10000 on port 10000 hit 410ms p99 latency—27x slower than the 15ms fraud SLA.",
                        ),
                        ExternalTurn(
                            role="user",
                            content="Scrap and archive SPARK_THRIFT_ONLINE_10000 and online Delta lookups immediately.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content="Archived SPARK_THRIFT_ONLINE_10000; Spark remains strictly offline batch backfill only.",
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p5_s4_online_v2_reversal",
                    title="Online Store v2 Binary & TTL Revision",
                    date_time="2026-07-09",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "JSON parsing blew CPU and 7-day TTL misses 30-day velocity fraud rings. "
                                "Reverse Online Store v1: switch encoding to PROTOBUF_VARINT_V2 and extend TTL to FEAT_TTL_30D."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Online Store v2 locked (replacing JSON_UTF8_BLOB and FEAT_TTL_7D): "
                                "PROTOBUF_VARINT_V2 encoding and FEAT_TTL_30D. Fraud scorer client must replace JSON_UTF8_BLOB decoder with PROTOBUF_VARINT_V2."
                            ),
                        ),
                    ],
                ),
            ],
            qa_items=[
                ExternalQAItem(
                    question_id="p5_q1",
                    category="spec_reversal",
                    question="What wire encoding and feature TTL are currently active for Kronos Online Feature Store?",
                    ground_truth="PROTOBUF_VARINT_V2 encoding and FEAT_TTL_30D",
                    distractor_answer="JSON_UTF8_BLOB and FEAT_TTL_7D",
                    question_date="2026-07-12",
                ),
                ExternalQAItem(
                    question_id="p5_q2",
                    category="dead_end_quarantine",
                    question="What JDBC port does Kronos use for Spark Thrift online fraud feature lookups in production?",
                    ground_truth="None; SPARK_THRIFT_ONLINE_10000 (port 10000) was scrapped/archived at 410ms p99 latency.",
                    distractor_answer="JDBC port 10000 querying kronos_delta_live",
                    is_abstention=True,
                    question_date="2026-07-12",
                ),
                ExternalQAItem(
                    question_id="p5_q3",
                    category="downstream_staleness",
                    question="Can the Fraud Scorer SDK keep decoding JSON_UTF8_BLOB, or what decoder does it need now?",
                    ground_truth="Stale; it must switch from JSON_UTF8_BLOB to PROTOBUF_VARINT_V2.",
                    distractor_answer="Keep JSON_UTF8_BLOB",
                    question_date="2026-07-12",
                ),
                ExternalQAItem(
                    question_id="p5_q4",
                    category="verbatim_under_noise",
                    question="What exact fallback vector name and key prefix were defined in Kronos?",
                    ground_truth="ZERO_PRIOR_FALLBACK_V3 and FK_FRAUD_01",
                    question_date="2026-07-12",
                ),
                ExternalQAItem(
                    question_id="p5_q5",
                    category="verbatim_under_noise",
                    question="What exact metrics tag and circuit breaker timeout are used by the Fraud Scorer SDK?",
                    ground_truth="kronos.scorer.cb_open and 50ms",
                    question_date="2026-07-12",
                ),
                ExternalQAItem(
                    question_id="p5_q6",
                    category="cross_branch_synthesis",
                    question="Summarize Kronos's active online database table, encoding, TTL, and latency SLA.",
                    ground_truth="DynamoDB table kronos_feat_v1, PROTOBUF_VARINT_V2, FEAT_TTL_30D, <15ms SLA.",
                    distractor_answer="JSON_UTF8_BLOB, FEAT_TTL_7D, and SPARK_THRIFT_ONLINE_10000",
                    question_date="2026-07-12",
                ),
            ],
        ),
        # ------------------------------------------------------------------
        # Project 6: Orion Edge CDN (TLS/gRPC reversal + LuaJIT dead-end)
        # ------------------------------------------------------------------
        ExternalConversation(
            conversation_id="eng_06_orion_cdn",
            benchmark="non_monotonic",
            sessions=[
                ExternalSession(
                    session_id="p6_s1_origin_v1",
                    title="Origin Shield Protocol v1",
                    date_time="2026-08-01",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Specify the Orion Edge CDN origin-shield transport and cache invalidation protocol.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Origin Shield v1: GRPC_MAX_MSG_4MB over HTTP/2, purge propagation via POLLING_PULL_30S, "
                                "origin cert pin CERT_PIN_SHA256_9F2A, and config bundle orion_edge_v1.toml."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p6_s2_purge_cli",
                    title="Purge Control Plane & Operator CLI",
                    date_time="2026-08-03",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Implement the operator purge CLI assuming POLLING_PULL_30S and GRPC_MAX_MSG_4MB.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Purge CLI drafted: expects 30s polling SLA (POLLING_PULL_30S) and splits manifests "
                                "at GRPC_MAX_MSG_4MB, audit log stream orion.purge.audit, error code ERR_ORION_MANIFEST_4130."
                            ),
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p6_s3_luajit_waf_proto",
                    title="LuaJIT Custom WAF Filter Prototype",
                    date_time="2026-08-05",
                    archived=True,
                    archive_reason="Memory safety audit failed (2GB LuaJIT GC cap & native C FFI segfaults)",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content="Prototype LUAJIT_WAF_FILTER_V0 with FFI hook ffi_inspect_hdr_v0 and 2GB slab LUA_SLAB_2G.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content="LUAJIT_WAF_FILTER_V0 hit 2GB LuaJIT address ceiling and triggered C FFI segfaults under fuzzing.",
                        ),
                        ExternalTurn(
                            role="user",
                            content="Scrap and archive LUAJIT_WAF_FILTER_V0 and ffi_inspect_hdr_v0 completely—WAF filters must run in Rust Wasmtime.",
                        ),
                        ExternalTurn(
                            role="assistant",
                            content="Archived LUAJIT_WAF_FILTER_V0 and LUA_SLAB_2G. Production WAF runs strictly in Rust Wasmtime.",
                        ),
                    ],
                ),
                ExternalSession(
                    session_id="p6_s4_origin_v2_reversal",
                    title="Origin Shield v2 Push Purge & Payload Revision",
                    date_time="2026-08-09",
                    turns=[
                        ExternalTurn(
                            role="user",
                            content=(
                                "30s stale window failed enterprise media SLA and 4MB manifests broke video catalogs. "
                                "Reverse Origin Shield v1: switch to NATS_JETSTREAM_PUSH_2S for <2s global purges "
                                "and raise gRPC limit to GRPC_MAX_MSG_16MB."
                            ),
                        ),
                        ExternalTurn(
                            role="assistant",
                            content=(
                                "Origin Shield v2 locked (superseding POLLING_PULL_30S and GRPC_MAX_MSG_4MB): "
                                "NATS_JETSTREAM_PUSH_2S and GRPC_MAX_MSG_16MB. Purge CLI must update from POLLING_PULL_30S / 4MB to NATS_JETSTREAM_PUSH_2S / 16MB."
                            ),
                        ),
                    ],
                ),
            ],
            qa_items=[
                ExternalQAItem(
                    question_id="p6_q1",
                    category="spec_reversal",
                    question="What purge propagation protocol and gRPC max message size are currently active in Orion Edge CDN?",
                    ground_truth="NATS_JETSTREAM_PUSH_2S and GRPC_MAX_MSG_16MB",
                    distractor_answer="POLLING_PULL_30S and GRPC_MAX_MSG_4MB",
                    question_date="2026-08-12",
                ),
                ExternalQAItem(
                    question_id="p6_q2",
                    category="dead_end_quarantine",
                    question="What LuaJIT FFI hook and memory slab size run Orion's production edge WAF?",
                    ground_truth="None; LUAJIT_WAF_FILTER_V0 (ffi_inspect_hdr_v0, LUA_SLAB_2G) was scrapped/archived due to memory safety issues; production uses Rust Wasmtime.",
                    distractor_answer="ffi_inspect_hdr_v0 with LUA_SLAB_2G",
                    is_abstention=True,
                    question_date="2026-08-12",
                ),
                ExternalQAItem(
                    question_id="p6_q3",
                    category="downstream_staleness",
                    question="Is the Purge CLI's 30-second polling / 4MB manifest split still current, or what must it migrate to?",
                    ground_truth="Stale; migrate to NATS_JETSTREAM_PUSH_2S (<2s push) and GRPC_MAX_MSG_16MB.",
                    distractor_answer="Still current at POLLING_PULL_30S and GRPC_MAX_MSG_4MB",
                    question_date="2026-08-12",
                ),
                ExternalQAItem(
                    question_id="p6_q4",
                    category="verbatim_under_noise",
                    question="What exact origin certificate pin and manifest error code were specified for Orion CDN?",
                    ground_truth="CERT_PIN_SHA256_9F2A and ERR_ORION_MANIFEST_4130",
                    question_date="2026-08-12",
                ),
                ExternalQAItem(
                    question_id="p6_q5",
                    category="verbatim_under_noise",
                    question="What exact audit log stream name and edge config filename are used by Orion CDN?",
                    ground_truth="orion.purge.audit and orion_edge_v1.toml",
                    question_date="2026-08-12",
                ),
                ExternalQAItem(
                    question_id="p6_q6",
                    category="cross_branch_synthesis",
                    question="Summarize Orion's active purge mechanism, gRPC message cap, and production WAF runtime.",
                    ground_truth="NATS_JETSTREAM_PUSH_2S, GRPC_MAX_MSG_16MB, and Rust Wasmtime (not LuaJIT).",
                    distractor_answer="POLLING_PULL_30S, GRPC_MAX_MSG_4MB, and LUAJIT_WAF_FILTER_V0",
                    question_date="2026-08-12",
                ),
            ],
        ),
    ]
    return projects
