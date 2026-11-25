from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from chatree import ArtifactInput, Chatree, ChatreeConfig, GuardrailPolicy
from chatree.engine.guardrails import TraceContext
from chatree.tools import TOOL_ASK_TREE
from chatree.types import LLMResponse


class _CompactionChainStubLLM:
    """Stub LLM that returns NOT_FOUND_IN_BRANCH when a detail is absent from the node's own turns."""

    def __init__(self) -> None:
        self.shadow_targets: list[str] = []

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last = str(messages[-1].get("content", ""))
        if "[Chatree Compactor]" in last:
            return LLMResponse(
                content=(
                    "1. OBJECTIVE: Continue gateway sprint.\n"
                    "2. LOCKED DECISIONS & VERBATIM ANCHORS: ED25519_SIGNING.\n"
                    "3. VERIFIED FACTS & ARTIFACTS: none.\n"
                    "4. REJECTED / SUPERSEDED: none.\n"
                    "5. OPEN NEXT STEPS: deploy."
                ),
                model_used=model,
                tokens_in=80,
                tokens_out=40,
            )
        if "[Shadow Query |" in last:
            body = " ".join(str(m.get("content", "")) for m in messages[1:-1])
            self.shadow_targets.append(last[:80])
            if "DIAG_PORT_9481" in body:
                return LLMResponse(
                    content="Found in verbatim history: diagnostic metrics port is DIAG_PORT_9481.",
                    model_used=model,
                    tokens_in=70,
                    tokens_out=20,
                )
            return LLMResponse(
                content="NOT_FOUND_IN_BRANCH",
                model_used=model,
                tokens_in=70,
                tokens_out=8,
            )
        return LLMResponse(content="OK", model_used=model, tokens_in=40, tokens_out=5)

    async def stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        resp = await self.complete(model=model, messages=messages, tools=tools, temperature=temperature)
        yield {"type": "token", "content": resp.content}
        yield {"type": "done", "response": resp}


@pytest.mark.asyncio
async def test_compaction_chain_transparent_shadow_lookup() -> None:
    stub = _CompactionChainStubLLM()
    cfg = ChatreeConfig(
        db_path=":memory:",
        default_model="stub/compact-test",
        guardrails=GuardrailPolicy(auto_title_enabled=False),
    )
    async with Chatree.in_memory(config=cfg, llm=stub) as ct:
        tree = await ct.create_tree("Compaction Chain Tree", default_model="stub/compact-test")
        b1 = await tree.root()
        await ct.repo.append_message(
            node_id=b1.id,
            tree_id=tree.id,
            role="user",
            content="Early un-anchored lab note: diagnostic metrics run on DIAG_PORT_9481.",
        )
        await ct.repo.append_message(
            node_id=b1.id,
            tree_id=tree.id,
            role="assistant",
            content="Recorded DIAG_PORT_9481.",
        )
        # Compact twice cleanly: B1 -> B2 -> B3 (B3 prefix only holds the compacted handoff, not DIAG_PORT_9481)
        b2, _ = await b1.compact(keep_recent_turns=0, reason="Gen 1 -> Gen 2")
        await ct.repo.append_message(
            node_id=b2.id, tree_id=tree.id, role="user", content="Working on rollout phase 2."
        )
        b3, _ = await b2.compact(keep_recent_turns=0, reason="Gen 2 -> Gen 3")

        # Sibling caller queries ONLY the tip B3 for the un-anchored Gen-1 token
        sibling = await b1.spawn("Verify diagnostics", title="Ops Audit", mode="clean", switch_head=False)
        out = await sibling.request(b3.id, prompt="What is DIAG_PORT_9481?")
        assert out["found_in_branch"] is True
        assert "DIAG_PORT_9481" in out["answer"]
        assert out.get("resolved_via_compaction_ancestor") == b1.id

        # Second identical query on B3 hits shadow_cache in 0 LLM calls
        calls_before = len(stub.shadow_targets)
        out_cached = await sibling.request(b3.id, prompt="What is DIAG_PORT_9481?")
        assert out_cached["cache_hit"] is True
        assert "DIAG_PORT_9481" in out_cached["answer"]
        assert len(stub.shadow_targets) == calls_before


@pytest.mark.asyncio
async def test_multi_branch_context_imports_and_auto_staleness() -> None:
    cfg = ChatreeConfig(
        db_path=":memory:",
        default_model="mock/chatree-demo",
        auto_commit_handoffs=True,
        guardrails=GuardrailPolicy(auto_title_enabled=False),
    )
    async with Chatree.in_memory(config=cfg) as ct:
        tree = await ct.create_tree("Diamond Convergence Tree")
        root = await tree.root()

        auth = await root.spawn("Design JWT auth", title="Auth Design", mode="clean", switch_head=False)
        await auth.chat("Settled on ED25519_SIGNING with JWT_TTL_5M.")
        await auth.finish(
            summary="Settled on ED25519_SIGNING and JWT_TTL_5M.",
            artifacts=[ArtifactInput(slug="auth.md", title="Auth Spec", content="# Auth\nED25519_SIGNING")],
            auto_commit=True,
            return_to_parent=False,
        )

        db = await root.spawn("Design Postgres", title="DB Schema", mode="clean", switch_head=False)
        await db.chat("Use POSTGRES_16 with POOL_SIZE_64.")
        await db.finish(
            summary="Use POSTGRES_16 with POOL_SIZE_64.",
            auto_commit=True,
            return_to_parent=False,
        )

        dead = await root.spawn("Try GraphQL", title="GraphQL Dead End", mode="clean", switch_head=False)
        await dead.chat("GRAPHQL_FEDERATION_V0.")
        await dead.archive(reason="Violates latency SLO", switch_to_parent=False)

        # Spawn Integration importing Auth + DB + archived Dead End (archived must be skipped!)
        integ = await root.spawn(
            "Build gateway integration",
            title="Integration Sprint",
            mode="clean",
            imports=[auth.id, db.id, dead.id],
            switch_head=False,
        )
        integ_m = await integ.model()
        prefix = await ct.context.build_static_prefix(integ_m)
        prefix_blob = "\n".join(str(m.get("content", "")) for m in prefix)
        assert "ED25519_SIGNING" in prefix_blob
        assert "auth.md@v1" in prefix_blob
        assert "POSTGRES_16" in prefix_blob
        assert "GRAPHQL_FEDERATION_V0" not in prefix_blob

        # Initially fresh (0 stale deps)
        assert await integ.stale_dependencies() == []

        # Material revision on Auth automatically flags Integration as STALE!
        auth_m = await auth.model()
        await ct.staleness.commit_summary_with_materiality(
            auth_m, "Revised: switched to RS256_LEGACY and JWT_TTL_15M.", status="COMPLETED"
        )
        stale_after = await integ.stale_dependencies()
        assert any(d.source_node_id == auth.id and d.severity == "material" for d in stale_after)


@pytest.mark.asyncio
async def test_search_and_query_deep_tool_dispatch() -> None:
    cfg = ChatreeConfig(
        db_path=":memory:",
        default_model="mock/chatree-demo",
        guardrails=GuardrailPolicy(auto_title_enabled=False),
    )
    async with Chatree.in_memory(config=cfg) as ct:
        tree = await ct.create_tree("Deep Tool Tree")
        root = await tree.root()
        ch = await root.spawn("Cache config", title="Cache Branch", mode="clean", switch_head=False)
        await ct.repo.append_message(
            node_id=ch.id, tree_id=tree.id, role="user", content="Cache uses VALKEY_CLUSTER_7."
        )
        await ct.repo.commit_node_summary(ch.id, "Cache uses VALKEY_CLUSTER_7.", status="COMPLETED")
        await ct.rag.index_tree(tree.id)

        t_m = await tree.model()
        r_m = await root.model()
        ctx = TraceContext.start(r_m.id, max_hop_depth=3, token_budget=64_000)

        # Fast 1-hop default (`deep=False`)
        res_fast = await ct.ipc.execute_tool(
            t_m, r_m, TOOL_ASK_TREE, {"query": "What cache store?", "deep": False}, ctx
        )
        assert res_fast.tool_output.get("deep") is None
        assert ch.id in (res_fast.tool_output.get("targets_queried") or [])

        # Deep multi-hop dispatch (`deep=True`) routes through TreeQueryEngine.deep_ask()
        res_deep = await ct.ipc.execute_tool(
            t_m, r_m, TOOL_ASK_TREE, {"query": "What cache store?", "deep": True, "max_targets": 2}, ctx
        )
        assert res_deep.tool_output.get("deep") is True
        assert "VALKEY_CLUSTER_7" in str(res_deep.tool_output.get("answer") or "")
        assert res_deep.ipc_event is not None
        assert res_deep.ipc_event.payload.get("deep") is True


@pytest.mark.asyncio
async def test_bg_summary_min_dirty_messages_guard() -> None:
    cfg = ChatreeConfig(
        db_path=":memory:",
        default_model="mock/chatree-demo",
        auto_background_summarize_on_switch=True,
        guardrails=GuardrailPolicy(auto_title_enabled=False, bg_summary_min_dirty_messages=2),
    )
    async with Chatree.in_memory(config=cfg) as ct:
        tree = await ct.create_tree("BG Guard Tree")
        root = await tree.root()
        c1 = await root.spawn("Quick peek", title="Peek Branch", mode="clean", switch_head=True)
        # Only 1 message (< min_dirty_messages=2): switching away must NOT fire a summary
        await ct.repo.append_message(
            node_id=c1.id, tree_id=tree.id, role="user", content="Single quick note.", mark_dirty=True
        )
        await tree.jump(root.id)
        await ct.wait_background_summaries()
        assert (await c1.model()).summary is None

        # Add a 2nd message (>= 2): switching away triggers background summary cleanly
        await tree.jump(c1.id)
        await ct.repo.append_message(
            node_id=c1.id, tree_id=tree.id, role="assistant", content="Second turn recorded.", mark_dirty=True
        )
        await tree.jump(root.id)
        await ct.wait_background_summaries()
        assert (await c1.model()).summary is not None
