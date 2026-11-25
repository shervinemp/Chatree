from __future__ import annotations

from chatree import AblationConfig, Chatree, ChatreeConfig, GuardrailPolicy
from chatree.engine.guardrails import TraceContext


async def test_cycle_hop_and_token_budget_guardrails():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Guardrails")
        root = await tree.root()
        child_b = await root.spawn("Child B", title="B", mode="clean")

        # Mutual recursion A -> B -> A deterministically blocked
        trace = TraceContext.start(root.id, max_hop_depth=4, token_budget=32_000)
        res = await ct.ipc.request(
            await tree.model(),
            await root.model(),
            child_b.id,
            prompt=f"Verify with root trigger_cycle:{root.id}",
            trace_ctx=trace,
        )
        assert res.tool_output.get("nested_guardrail_warning", {}).get("error") == "CYCLE_DETECTED"
        assert any(e.status == "rejected_cycle" for e in await tree.ipc_log())

        # Max hop depth rejection
        res_hop = await ct.ipc.request(
            await tree.model(),
            await root.model(),
            child_b.id,
            prompt="Question",
            trace_ctx=TraceContext.start(root.id, max_hop_depth=0, token_budget=10_000),
        )
        assert res_hop.tool_output["error"] == "MAX_HOP_DEPTH_EXCEEDED"

        # Token budget rejection
        res_budget = await ct.ipc.request(
            await tree.model(),
            await root.model(),
            child_b.id,
            prompt="Question",
            trace_ctx=TraceContext.start(root.id, max_hop_depth=3, token_budget=100),
        )
        assert res_budget.tool_output["error"] == "TOKEN_BUDGET_EXCEEDED"


async def test_reflog_revisitation_triggers_and_clears_staleness():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Staleness Flow", auto_commit_handoffs=True)
        root = await tree.root()
        child = await root.spawn("Auth", title="Auth", mode="clean")
        await child.chat("Use session cookies.")
        await child.finish(summary="Conclusion v1: session cookies.", auto_commit=True)

        assert await root.stale_dependencies() == []

        # Jump back via reflog and revise conclusion
        await tree.jump(child.id, note="Revise auth")
        res = await child.chat("Switch to OAuth2 PKCE + Ed25519.")
        assert root.id in res.stale_node_ids
        assert (await child.model()).status == "ACTIVE"

        stale = await root.stale_dependencies()
        assert len(stale) == 1 and stale[0].source_node_id == child.id

        # Refresh via request(prompt=None) -> summary v2 -> staleness clears!
        refreshed = await root.request(child.id, prompt=None)
        assert refreshed["summary_version"] == 2
        assert await root.stale_dependencies() == []


async def test_reflog_back_forward_and_ablation_switches():
    cfg = ChatreeConfig(
        db_path=":memory:",
        ablations=AblationConfig(enable_shadow_turns=False, enable_staleness_notices=False),
        guardrails=GuardrailPolicy(max_hop_depth=3, token_budget=32_000),
    )
    async with Chatree.in_memory(config=cfg) as ct:
        tree = await ct.create_tree("Ablation & Reflog")
        root = await tree.root()
        a = await root.spawn("A", title="A", mode="clean")
        b = await a.spawn("B", title="B", mode="clean")
        await b.chat("Hidden secret token SECRET_XYZ_99.")

        # Back/forward stepping
        assert (await tree.back()).id == a.id
        assert (await tree.back()).id == root.id
        assert (await tree.forward()).id == a.id
        assert (await tree.jump("HEAD@{0}")).id == b.id

        # Because enable_shadow_turns=False, prompt request degrades cleanly to summary-only!
        out = await root.request(b.id, prompt="What secret token is in B?")
        assert out["ablation_shadow_disabled"] is True
        assert out["mode"] in ("refreshed_summary", "cached_summary")
