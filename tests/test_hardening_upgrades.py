from __future__ import annotations

import asyncio
from pathlib import Path

from chatree import (
    ArtifactInput,
    Chatree,
    ChatreeConfig,
    DefaultModelRouter,
    GitWorktreeSandbox,
    GuardrailPolicy,
    LiteLLMProvider,
    SearchHit,
)


async def _git(cwd: Path, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    assert proc.returncode == 0, f"git {args} failed: {err.decode()}"
    return out.decode().strip()


async def test_shadow_turn_memoization_cache_and_invalidation():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Shadow Cache Test")
        root = await tree.root()
        child = await root.spawn("Cache Policy", title="Cache", mode="clean")
        await child.chat("Final decision: use REDIS_TTL_300S sliding window.")

        # 1st shadow query -> LLM executes (cache_hit=False)
        r1 = await root.request(child.id, prompt="What Redis TTL did we choose?")
        assert r1["mode"] == "shadow_turn"
        assert r1["cache_hit"] is False
        assert "REDIS_TTL_300S" in r1["answer"]

        # 2nd normalized-identical query -> 0-token shadow cache hit!
        r2 = await root.request(child.id, prompt="  what redis ttl   did we choose? ")
        assert r2["mode"] == "cached_shadow_turn"
        assert r2["cache_hit"] is True
        assert r2["answer"] == r1["answer"]
        metrics = await tree.metrics()
        assert metrics.shadow_cache_hits == 1
        latest_ipc = (await tree.ipc_log(limit=1))[0]
        assert latest_ipc.tokens_in == 0 and latest_ipc.tokens_out == 0

        # Mutate child -> last_msg_seq advances -> shadow cache automatically invalidates!
        await child.chat("Revision: increase TTL to REDIS_TTL_900S.")
        r3 = await root.request(child.id, prompt="What Redis TTL did we choose?")
        assert r3["mode"] == "shadow_turn"
        assert r3["cache_hit"] is False
        assert "REDIS_TTL_900S" in r3["answer"]


async def test_provider_kv_cache_control_and_threshold_shadow_router():
    messages = [
        {"role": "system", "content": "Static system prompt."},
        {"role": "user", "content": "Inherited turn 1"},
        {"role": "assistant", "content": "Inherited reply 1"},
        {"role": "user", "content": "<chatree_context>...</chatree_context>\nNewest turn"},
    ]
    anthropic_wire = LiteLLMProvider.apply_cache_control("anthropic/claude-sonnet-4", messages)
    assert isinstance(anthropic_wire[0]["content"], list)
    assert anthropic_wire[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # Boundary message right before newest turn is cached; newest ephemeral user turn is NOT
    assert isinstance(anthropic_wire[2]["content"], list)
    assert anthropic_wire[2]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert isinstance(anthropic_wire[3]["content"], str)

    # Non-Anthropic models remain standard string content
    openai_wire = LiteLLMProvider.apply_cache_control("openai/gpt-4o", messages)
    assert openai_wire == messages

    # Threshold-based shadow routing: cold/short branch -> fast model; warm/long branch -> inherit
    router = DefaultModelRouter(
        fast_shadow_model="gemini/gemini-2.5-flash", warm_cache_token_threshold=4_000
    )
    async with Chatree.in_memory(router=router) as ct:
        tree = await ct.create_tree("Routing", default_model="anthropic/claude-sonnet-4")
        tm = await tree.model()
        root = await tree.root()
        cold_node = await root.model()
        assert router.resolve_model(tm, cold_node, "shadow") == "gemini/gemini-2.5-flash"
        await ct.repo.increment_node_usage(root.id, tokens_in=6_000, tokens_out=400)
        warm_node = await root.model()
        assert router.resolve_model(tm, warm_node, "shadow") == "anthropic/claude-sonnet-4"


async def test_contextual_rag_reranker_speculative_pointers_ask_tree_and_spawn_hysteresis():
    class ReverseBoostReranker:
        async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
            del query
            # Boost hits whose snippet contains ZSTD_LEVEL_19 to rank #1
            return sorted(
                hits,
                key=lambda h: (1 if "ZSTD_LEVEL_19" in h.snippet else 0, h.score),
                reverse=True,
            )[:top_k]

    async with Chatree.in_memory(reranker=ReverseBoostReranker()) as ct:
        tree = await ct.create_tree("RAG Upgrades", auto_commit_handoffs=True)
        root = await tree.root()
        comp = await root.spawn("Compression", title="Compression", mode="clean")
        await comp.chat("Final decision: compress blobs using ZSTD_LEVEL_19 streaming frames.")
        await comp.finish(
            summary="Use ZSTD compression for payload storage.",
            artifacts=[
                ArtifactInput(
                    slug="codec.json",
                    title="Codec Config",
                    kind="json",
                    content='{"codec": "ZSTD_LEVEL_19"}',
                )
            ],
            auto_commit=True,
        )

        # 1. Contextualized chunk header includes full branch path & goal
        chunks = await tree.search("ZSTD_LEVEL_19", scope="all", granularities={"chunk"})
        assert len(chunks) >= 1
        assert "[Path: RAG Upgrades > Compression" in chunks[0].snippet

        # 2. Tier-3 rollup includes structured Key Decisions & Identifiers from leaf + artifacts!
        root_model = await root.model()
        assert root_model.rollup_summary is not None
        assert "ZSTD_LEVEL_19" in root_model.rollup_summary
        assert "codec.json@v1" in root_model.rollup_summary

        # 3. Pre-turn speculative branch pointers appear in ephemeral envelope for root
        hits = await tree.search("ZSTD_LEVEL_19 compression", scope="all", top_k=3)
        env, _, _ = await ct.context.build_ephemeral_envelope(
            await tree.model(), root_model, speculative_hits=hits
        )
        assert "RELATED [" in env
        assert comp.id in env

        # 4. Single-hop compound ask_tree() fuses search + shadow turn in one call!
        ask_out = await tree.ask("What exact ZSTD level did Compression pick?", caller_id=root.id)
        assert comp.id in ask_out["targets_queried"]
        assert any("ZSTD_LEVEL_19" in str(a.get("answer", "")) for a in ask_out["answers"])

    # 5. Spawn hysteresis: suggest mode emits branch_suggested without creating a node!
    suggested_events: list[dict] = []
    cfg_suggest = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(spawn_policy="suggest"),
    )
    async with Chatree.in_memory(config=cfg_suggest) as ct:
        ct.hooks.register("branch_suggested", lambda p: suggested_events.append(p))
        tree = await ct.create_tree("Suggest Policy")
        root = await tree.root()
        await root.chat("Please branch off to explore GraphQL caching")
        assert len(suggested_events) == 1
        assert len(await tree.nodes()) == 1  # no node auto-created!

    # 6. Spawn cap: max_active_children=1 blocks 2nd autonomous/model spawn
    cfg_cap = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(spawn_policy="auto", max_active_children=1),
    )
    async with Chatree.in_memory(config=cfg_cap) as ct:
        tree = await ct.create_tree("Cap Policy")
        root = await tree.root()
        await root.spawn("Active child 1", mode="clean", switch_head=False)
        res = await root.chat("Please branch off to explore a second tangent")
        assert any(
            ev.status == "rejected_policy"
            and ev.response.get("error") == "MAX_ACTIVE_CHILDREN_EXCEEDED"
            for ev in res.ipc_events
        )


async def test_materiality_gated_staleness_cosmetic_vs_material():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Materiality Test", auto_commit_handoffs=True)
        root_a = await tree.root()
        gw_b = await root_a.spawn("Gateway B", title="Gateway B", mode="clean")
        auth_c = await gw_b.spawn("Auth C", title="Auth C", mode="clean")

        await auth_c.chat("Initial decision: use HS256_LEGACY signing tokens with 15m TTL.")
        await auth_c.finish(
            summary="Auth C uses HS256_LEGACY signing tokens with 15m TTL for API requests.",
            auto_commit=True,
        )
        await gw_b.chat("Gateway B integrates Auth C.")
        await gw_b.finish(summary="Gateway B integrates Auth C v1.", auto_commit=True)

        # Case 1: Cosmetic edit on C (punctuation/wording clarification, same HS256_LEGACY decision)
        c_node = await auth_c.model()
        c_cosmetic = await ct.staleness.commit_summary_with_materiality(
            c_node,
            "Auth C uses HS256_LEGACY signing tokens with 15m TTL for API requests!",
            status="COMPLETED",
        )
        assert c_cosmetic.summary_version == 1  # major stays v1!
        assert c_cosmetic.summary_minor_version == 1  # minor bumped v1.1!
        assert c_cosmetic.last_change_severity == "cosmetic"

        # Transitive consumer A is NOT spammed by a cosmetic edit on C!
        assert await root_a.stale_dependencies() == []

        # Case 2: Material revision on C (HS256_LEGACY -> ED25519_ROTATING_24H)
        c_material = await ct.staleness.commit_summary_with_materiality(
            c_cosmetic,
            "Auth C switched to ED25519_ROTATING_24H asymmetric signing tokens with 15m TTL.",
            status="COMPLETED",
        )
        assert c_material.summary_version == 2
        assert c_material.summary_minor_version == 0
        assert c_material.last_change_severity == "material"
        assert "HS256_LEGACY" in c_material.last_change_delta
        assert "ED25519_ROTATING_24H" in c_material.last_change_delta

        # Now BOTH B (1-hop) and A (2-hop transitive) receive material staleness + delta!
        a_stale = await root_a.stale_dependencies()
        assert len(a_stale) == 1
        assert a_stale[0].hops == 2
        assert a_stale[0].severity == "material"
        assert "ED25519_ROTATING_24H" in a_stale[0].change_delta


async def test_git_worktree_sandbox_parallel_isolation_and_dual_commit_merge(tmp_path):
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    await _git(repo, "init")
    (repo / "config.py").write_text("AUTH_MODE = 'legacy'\n", encoding="utf-8")
    await _git(repo, "add", "-A")
    await _git(repo, "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-m", "initial")

    sandbox = GitWorktreeSandbox(repo, worktrees_dir=tmp_path / "worktrees")
    async with Chatree.in_memory(sandbox=sandbox) as ct:
        tree = await ct.create_tree("Coding Sandbox", auto_commit_handoffs=False)
        root = await tree.root()

        # Spawn two parallel read_write branches -> isolated Git worktrees!
        opt_a = await root.spawn("Implement Ed25519", title="Option A", mode="clean", permission="read_write")
        opt_b = await root.spawn("Implement OAuth2", title="Option B", mode="clean", permission="read_write")

        ws_a = tree.workspace(opt_a.id)
        ws_b = tree.workspace(opt_b.id)
        assert ws_a is not None and ws_b is not None
        assert ws_a.path != ws_b.path

        # Edit same file concurrently in both worktrees without clobbering each other or main repo!
        (ws_a.path / "config.py").write_text("AUTH_MODE = 'ed25519'\n", encoding="utf-8")
        (ws_b.path / "config.py").write_text("AUTH_MODE = 'oauth2'\n", encoding="utf-8")
        assert (repo / "config.py").read_text(encoding="utf-8") == "AUTH_MODE = 'legacy'\n"

        # Finishing opt_a auto-attaches its Git diff artifact to the PR handoff!
        fin = await opt_a.finish(summary="Implemented Ed25519 config.", auto_commit=False)
        mb = fin["handoff_pending"]
        assert mb is not None
        assert any(r.kind == "diff" for r in mb.artifact_refs)

        # Promoting opt_a merges its worktree commit into main repo & cleans up BOTH worktrees!
        await tree.promote_child(opt_a.id, archive_siblings=True)
        assert "AUTH_MODE = 'ed25519'" in (repo / "config.py").read_text(encoding="utf-8")
        assert tree.workspace(opt_a.id) is None
        assert tree.workspace(opt_b.id) is None
        assert not ws_a.path.exists()
        assert not ws_b.path.exists()
