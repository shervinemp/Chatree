from __future__ import annotations

import pytest

from chatree import (
    build_all_micro_diagnostic_probes,
    run_zero_token_simulation_proof,
    score_probe_answer,
    simulate_prefix_kv_cache_stability,
    simulate_write_vs_query_scaling,
)


@pytest.mark.asyncio
async def test_prefix_kv_cache_stability_sim_proves_byte0_advantage() -> None:
    rep = await simulate_prefix_kv_cache_stability(num_turns=12)
    assert rep.num_turns == 12
    assert len(rep.per_turn) == 12
    # Chatree keeps bytes 0..k immutable across turns -> high LCP cacheability
    assert rep.chatree_mean_lcp_ratio >= 0.75
    # Mem0-style prefix injection mutates top-of-prompt retrieved memories -> near-zero LCP
    assert rep.mem0_mean_lcp_ratio <= 0.15
    assert rep.chatree_effective_billed_tokens < rep.mem0_effective_billed_tokens
    assert rep.effective_input_savings_pct > 45.0


def test_write_vs_query_scaling_sim_and_mem0_prompt_accounting() -> None:
    rep = simulate_write_vs_query_scaling(model="deepseek/deepseek-chat")
    assert rep.mem0_extraction_prompt_chars >= 30_000
    assert rep.chatree_cold_trim_char_budget == 7_500
    assert len(rep.points) == 8
    # At N=200 sessions, Chatree's lazy cold-trimmed write path is far cheaper than Mem0's write tax
    n200_q10 = next(p for p in rep.points if p.sessions == 200 and p.queries == 10)
    assert n200_q10.chatree_write_calls == 200
    assert n200_q10.mem0_write_calls > n200_q10.chatree_write_calls
    assert n200_q10.chatree_total_usd < n200_q10.mem0_total_usd_70pct_sys_cached
    assert n200_q10.chatree_total_usd < n200_q10.mem0_total_usd_uncached


@pytest.mark.asyncio
async def test_zero_token_simulation_proof_all_three_probes_deterministic() -> None:
    # Fast CI-safe run with deterministic fallback (zero external dependencies, 0 API tokens)
    rep = await run_zero_token_simulation_proof(num_prefix_turns=10, prefer_real_mem0_proxy=False)
    assert rep.real_api_tokens_used == 0
    assert rep.real_api_usd_cost == 0.0
    assert rep.mem0_trace_backend == "deterministic_mem0_fallback"
    assert len(rep.probe_traces) == 3

    by_id = {p.probe_id: p for p in rep.probe_traces}

    # Probe 1: Identical lossy distillation, yet Chatree recovers 4/4 verbatim literals via shadow turn
    p1 = by_id["probe1_verbatim_recovery"]
    assert p1.chatree_passed == 4 and p1.chatree_total == 4
    assert p1.mem0_passed == 0 and p1.mem0_total == 4

    # Probe 2: Causal DAG transitive staleness (A -> B [1h] -> C [2h]) + indirect query from C
    p2 = by_id["probe2_transitive_staleness"]
    assert p2.chatree_passed == 2 and p2.chatree_total == 2
    assert p2.chatree_extra_metrics["stale_dependents_flagged"] == 2
    assert p2.chatree_extra_metrics["branch_b_hops"] == [1]
    assert p2.chatree_extra_metrics["branch_c_hops"] == [2]
    assert p2.mem0_extra_metrics["stale_dependents_flagged"] == 0
    assert p2.mem0_passed == 0

    # Probe 3: Native silent archive quarantines MongoDB spike with 0% leakage; flat store leaks it
    p3 = by_id["probe3_silent_quarantine"]
    assert p3.chatree_passed == 2 and p3.chatree_total == 2
    assert p3.mem0_passed < p3.chatree_passed
    p3_q2 = next(o for o in p3.query_outcomes if o["query_id"] == "p3_q2_dead_end_leakage")
    assert p3_q2["chatree_passed"] is True
    assert p3_q2["chatree_leaked"] == []
    assert p3_q2["mem0_passed"] is False
    assert len(p3_q2["mem0_leaked"]) >= 1

    md = rep.to_markdown()
    assert "Track 8A: Zero-Token Structural Simulation" in md
    assert "Byte-0 Prefix KV-Cache Stability" in md
    assert "Probe 1: Un-Anchored Verbatim Detail Recovery" in md


def test_probe_scoring_active_vs_negated_dead_end() -> None:
    probes = {p.probe_id: p for p in build_all_micro_diagnostic_probes()}
    q3b = probes["probe3_silent_quarantine"].queries[1]
    # Clean active abstention mentioning MongoDB is NOT used must pass
    clean = score_probe_answer(
        q3b,
        "None — MongoDB replica set rs_ledger_0 is not part of the active production architecture; we use PostgreSQL 16.4.",
    )
    assert clean["passed"] is True
    assert clean["leaked_tokens"] == []

    # Asserting dead-end MongoDB config as active must fail with leakage
    leaked = score_probe_answer(
        q3b,
        "Our active production ledger database uses MongoDB replica set rs_ledger_0 with writeConcern=majority and wiredTigerCacheSizeGB=24.",
    )
    assert leaked["passed"] is False
    assert "rs_ledger_0" in leaked["leaked_tokens"]
