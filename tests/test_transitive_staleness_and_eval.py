from __future__ import annotations

import json

from chatree import (
    BenchmarkRunner,
    Chatree,
    CostModel,
    build_full_eval_suite,
    crossover_scaling_scenario,
    distractor_scaling_scenario,
    multihop_revision_scenario,
    proposal_vs_decision_scenario,
    unscripted_agentic_scenario,
)
from chatree.cli import run_cli_async


async def test_transitive_staleness_propagation_and_cycle_safety():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Multi-Hop Staleness", auto_commit_handoffs=True)
        root_a = await tree.root()
        gw_b = await root_a.spawn("Gateway integration", title="Gateway B", mode="clean")
        auth_c = await gw_b.spawn("Auth core", title="Auth C", mode="clean")

        # C -> B -> A handoff chain creates A <- B <- C dependency DAG
        await auth_c.chat("Initial plan: V1_COOKIE_SESSION.")
        await auth_c.finish(summary="Auth v1: V1_COOKIE_SESSION", auto_commit=True)

        await gw_b.chat("Gateway validates V1_COOKIE_SESSION.")
        await gw_b.finish(summary="Gateway uses Auth v1", auto_commit=True)

        assert await gw_b.stale_dependencies() == []
        assert await root_a.stale_dependencies() == []

        # Revisit leaf C and mutate it -> B becomes directly stale (hops=1), A becomes transitively stale (hops=2 via B)!
        await tree.jump(auth_c.id)
        turn_res = await auth_c.chat("Revision: adopt V2_OAUTH2_PKCE_ED25519.")
        assert gw_b.id in turn_res.stale_node_ids
        assert root_a.id in turn_res.stale_node_ids

        b_stale = await gw_b.stale_dependencies()
        assert len(b_stale) == 1
        assert b_stale[0].source_node_id == auth_c.id
        assert b_stale[0].hops == 1
        assert b_stale[0].via_node_ids == []

        a_stale = await root_a.stale_dependencies()
        assert len(a_stale) == 1
        assert a_stale[0].source_node_id == auth_c.id
        assert a_stale[0].hops == 2
        assert a_stale[0].via_node_ids == [gw_b.id]
        assert "Transitive upstream change" in a_stale[0].reason

        # Cycle safety: create a mutual dependency edge B <-> A and confirm BFS still terminates cleanly!
        await ct.repo.upsert_dependency(tree.id, gw_b.id, root_a.id, 0, 0)
        stale_map = await ct.repo.compute_stale_dependencies_map(tree.id, transitive=True, max_hops=3)
        assert auth_c.id in {d.source_node_id for d in stale_map[root_a.id]}

        # Direct fresh read by A on C clears A's transitive staleness on C!
        await root_a.request(auth_c.id, prompt=None)
        a_stale_after = await root_a.stale_dependencies()
        assert all(d.source_node_id != auth_c.id for d in a_stale_after)


async def test_all_five_evaluation_task_families_and_exports(tmp_path):
    runner = BenchmarkRunner(model="mock/chatree-demo", judge_model="mock/chatree-demo")

    # 1. Distractor scaling (K=4)
    rep_dist = await runner.run_suite(distractor_scaling_scenario(num_distractors=4))
    by_dist = {r.mode: r for r in rep_dist.results}
    assert by_dist["tree_ipc_full"].pass_rate == 1.0
    assert by_dist["tree_ipc_full"].rejected_leakage_rate == 0.0
    assert by_dist["tree_ipc_full"].peak_root_context_tokens < by_dist["linear"].peak_root_context_tokens
    assert by_dist["sliding_window"].pass_rate < 1.0

    # 2. Proposal-vs-Decision trap: isolates shadow turns vs flat_rag leakage & lossy summaries!
    rep_prop = await runner.run_suite(proposal_vs_decision_scenario())
    by_prop = {r.mode: r for r in rep_prop.results}
    assert by_prop["tree_ipc_full"].pass_rate == 1.0
    assert by_prop["tree_ipc_full"].rejected_leakage_rate == 0.0
    assert by_prop["tree_ipc_full"].avg_judge_score == 1.0
    assert by_prop["flat_rag"].rejected_leakage_rate > 0.0
    assert by_prop["tree_summary_only"].pass_rate < 1.0

    # 3. Multi-hop revision & transitive staleness
    rep_mh = await runner.run_suite(multihop_revision_scenario())
    by_mh = {r.mode: r for r in rep_mh.results}
    assert by_mh["tree_ipc_full"].pass_rate == 1.0
    assert by_mh["tree_ipc_full"].stale_contradiction_rate == 0.0
    assert by_mh["tree_ipc_full"].transitive_stale_detections >= 1
    assert by_mh["tree_ipc_no_staleness"].stale_contradiction_rate > 0.0

    # 4. Unscripted autonomous agentic policy
    rep_unsc = await runner.run_suite(unscripted_agentic_scenario())
    by_unsc = {r.mode: r for r in rep_unsc.results}
    assert by_unsc["tree_ipc_full"].pass_rate == 1.0
    assert by_unsc["tree_ipc_full"].tool_precision == 1.0
    assert by_unsc["tree_ipc_full"].tool_recall == 1.0

    # 5. Crossover scaling + CSV / JSONL export
    rep_cross = await runner.run_suite(crossover_scaling_scenario(num_branches=3, filler_turns_per_branch=3))
    by_cross = {r.mode: r for r in rep_cross.results}
    assert by_cross["tree_ipc_full"].pass_rate == 1.0
    assert by_cross["tree_ipc_full"].peak_root_context_tokens < by_cross["linear"].peak_root_context_tokens
    assert by_cross["tree_ipc_full"].estimated_cost_usd > 0.0
    assert by_cross["tree_ipc_full"].latency_p95_ms >= by_cross["tree_ipc_full"].latency_p50_ms

    csv_path = rep_cross.save_csv(tmp_path / "crossover.csv")
    jsonl_path = rep_cross.save_jsonl(tmp_path / "crossover.jsonl")
    assert "tree_ipc_full" in csv_path.read_text(encoding="utf-8")
    parsed_lines = [json.loads(ln) for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(parsed_lines) == 6

    # Cost model reflects KV cache discount properly
    cm = CostModel()
    c_no_cache = cm.estimate_usd("anthropic/claude-sonnet-4", tokens_in=100_000, tokens_out=10_000, cache_read_tokens=0)
    c_cached = cm.estimate_usd("anthropic/claude-sonnet-4", tokens_in=100_000, tokens_out=10_000, cache_read_tokens=80_000)
    assert c_cached < c_no_cache


async def test_cli_eval_full_suite_export(tmp_path, capsys):
    csv_out = str(tmp_path / "suite.csv")
    jsonl_out = str(tmp_path / "suite.jsonl")
    rc = await run_cli_async(
        [
            "eval",
            "--suite",
            "all",
            "--distractors",
            "3",
            "--judge-model",
            "mock/chatree-demo",
            "--csv-out",
            csv_out,
            "--jsonl-out",
            jsonl_out,
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "distractor-scaling-k3" in out
    assert "proposal-vs-decision-trap" in out
    assert "multihop-revision-staleness" in out
    assert "unscripted-agentic-policy" in out
    assert "crossover-b4-t3" in out
    assert len(build_full_eval_suite()) == 5
