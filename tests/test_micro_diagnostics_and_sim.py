from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chatree import (
    DEFAULT_MATERIALITY_CASES,
    Mem0OfficialAdapter,
    build_all_micro_diagnostic_probes,
    evaluate_materiality_classifier,
    run_zero_token_simulation_proof,
    score_probe_answer,
    simulate_prefix_kv_cache_stability,
    simulate_write_vs_query_scaling,
)


@pytest.mark.asyncio
async def test_prefix_kv_cache_stability_sim_proves_three_way_prefix_advantage() -> None:
    rep = await simulate_prefix_kv_cache_stability(num_turns=12)
    assert rep.num_turns == 12
    assert len(rep.per_turn) == 12
    # Chatree keeps system prompt + all prior turns immutable -> highest LCP cacheability
    assert rep.chatree_mean_lcp_ratio >= 0.75
    # End-of-system-prompt Mem0 keeps static system prompt cached, but invalidates prior turns
    assert rep.mem0_mean_lcp_ratio < rep.mem0_end_of_sys_mean_lcp_ratio < rep.chatree_mean_lcp_ratio
    # Byte-0 prepend mutates top-of-prompt retrieved memories -> near-zero LCP
    assert rep.mem0_mean_lcp_ratio <= 0.15
    assert rep.chatree_effective_billed_tokens < rep.mem0_end_of_sys_effective_billed_tokens
    assert rep.mem0_end_of_sys_effective_billed_tokens < rep.mem0_effective_billed_tokens
    assert rep.effective_input_savings_vs_end_of_sys_pct > 25.0
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
async def test_materiality_classifier_zero_token_benchmark() -> None:
    assert len(DEFAULT_MATERIALITY_CASES) == 20
    rep = await evaluate_materiality_classifier(prefer_fastembed=False)
    assert rep.total_cases == 20
    assert rep.tp + rep.fp + rep.tn + rep.fn == 20
    # Identifier/numeric and polarity categories are deterministically caught at 100% recall
    assert rep.category_breakdown["identifier_numeric"]["correct"] == 4
    assert rep.category_breakdown["polarity_flip"]["correct"] == 3
    assert 0.0 <= rep.precision <= 1.0
    assert 0.0 <= rep.recall <= 1.0
    assert 0.0 <= rep.f1 <= 1.0


@pytest.mark.asyncio
async def test_mem0_official_adapter_client_level_token_metering() -> None:
    # Verify exact metering of DeepSeek (`prompt_cache_hit_tokens`) and OpenAI
    # (`prompt_tokens_details.cached_tokens`) usage objects with zero heuristic double-counting.
    adapter = Mem0OfficialAdapter(model="deepseek/deepseek-chat")

    class _FakeCompletions:
        def __init__(self) -> None:
            self.call_idx = 0

        def create(self, **kwargs: Any) -> Any:
            del kwargs
            self.call_idx += 1
            if self.call_idx == 1:
                usage = SimpleNamespace(
                    prompt_tokens=8400,
                    completion_tokens=120,
                    prompt_cache_hit_tokens=6000,
                )
            else:
                usage = SimpleNamespace(
                    prompt_tokens=1900,
                    completion_tokens=65,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=1100),
                )
            return SimpleNamespace(usage=usage)

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    class _FakeMemory:
        def __init__(self) -> None:
            self.llm = SimpleNamespace(client=fake_client)

        def add(self, msgs: list[dict[str, str]], user_id: str) -> dict[str, Any]:
            del msgs, user_id
            # Simulate Mem0's two internal write-path calls (extract + update)
            self.llm.client.chat.completions.create(model="deepseek-chat", messages=[])
            self.llm.client.chat.completions.create(model="deepseek-chat", messages=[])
            return {"results": [{"memory": "User prefers PostgreSQL 16.4"}]}

    adapter._mem = _FakeMemory()
    assert adapter._install_llm_usage_meter() is True
    await adapter.start_session("s1")
    await adapter.ingest_turn("s1", "user", "We standardized on PostgreSQL 16.4.")
    await adapter.finish_session("s1")

    assert adapter.get_ingest_usage() == (10300, 185)
    info = adapter.get_ingest_metering_info()
    assert info["metered"] is True
    assert info["llm_calls"] == 2
    assert info["cache_read_tokens"] == 7100

    # Verify fallback heuristic activates ONLY when underlying client cannot be metered
    fallback_adapter = Mem0OfficialAdapter(model="ollama/llama3")

    class _UnmeteredMemory:
        def add(self, msgs: list[dict[str, str]], user_id: str) -> dict[str, Any]:
            del msgs, user_id
            return {"results": [{"memory": "Fallback memory"}]}

    fallback_adapter._mem = _UnmeteredMemory()
    assert fallback_adapter._install_llm_usage_meter() is False
    await fallback_adapter.start_session("s2")
    await fallback_adapter.ingest_turn("s2", "user", "Fallback test message content.")
    await fallback_adapter.finish_session("s2")
    fb_in, fb_out = fallback_adapter.get_ingest_usage()
    assert fb_in >= 200 and fb_out >= 40
    assert fallback_adapter.get_ingest_metering_info()["metered"] is False


@pytest.mark.asyncio
async def test_zero_token_simulation_proof_all_three_probes_deterministic() -> None:
    # Fast CI-safe run with deterministic fallback (zero external dependencies, 0 API tokens)
    rep = await run_zero_token_simulation_proof(num_prefix_turns=10, prefer_real_mem0_proxy=False)
    assert rep.real_api_tokens_used == 0
    assert rep.real_api_usd_cost == 0.0
    assert rep.mem0_trace_backend == "deterministic_mem0_fallback"
    assert len(rep.probe_traces) == 3
    assert rep.materiality_eval is not None
    assert rep.materiality_eval.total_cases == 20

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
    assert "Prompt Prefix KV-Cache Stability" in md
    assert "End-of-System Injection" in md
    assert "Zero-Token Materiality Classifier Benchmark" in md
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
