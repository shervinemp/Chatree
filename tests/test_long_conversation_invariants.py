from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from chatree import Chatree, ChatreeConfig
from chatree.config import GuardrailPolicy, resolve_model_context_window
from chatree.eval.ablations import AblationMatrixRunner
from chatree.eval.runner import BenchmarkRunner
from chatree.eval.scenarios import MultiTangentScenario, ScenarioStep
from chatree.eval.scoring import ExactFactScorer, is_token_asserted_as_active
from chatree.providers.mock import MockLLMProvider, estimate_messages_tokens, estimate_tokens
from chatree.types import LLMResponse, LLMStreamEvent


class _RecordingMockLLM(MockLLMProvider):
    """Wraps MockLLMProvider to record every prompt, tool schema list, and token estimate."""

    def __init__(self, *, verbose_compactor_chars: int = 0) -> None:
        super().__init__(stream_delay_s=0.0)
        self.calls: list[dict[str, Any]] = []
        self.verbose_compactor_chars = verbose_compactor_chars

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> LLMResponse:
        msg_tok = estimate_messages_tokens(messages)
        tool_tok = estimate_tokens(json.dumps(tools)) if tools else 0
        self.calls.append(
            {
                "kind": "complete",
                "model": model,
                "messages": messages,
                "tools": tools,
                "msg_tokens": msg_tok,
                "tool_tokens": tool_tok,
                "total_tokens": msg_tok + tool_tok,
                "kwargs": dict(kwargs),
            }
        )
        if (
            self.verbose_compactor_chars > 0
            and messages
            and "[Chatree Compactor]" in str(messages[-1].get("content") or "")
        ):
            bloated = (
                "1. OBJECTIVE: Long session.\n"
                "2. LOCKED DECISIONS & VERBATIM ANCHORS: ANCHOR_ALPHA_1, ANCHOR_BETA_2.\n"
                + ("Historical verbatim detail line repeated across generations. " * (self.verbose_compactor_chars // 40))
                + "\n5. OPEN NEXT STEPS: Continue."
            )
            return LLMResponse(
                content=bloated,
                model_used=model,
                tokens_in=msg_tok,
                tokens_out=estimate_tokens(bloated),
                cache_read_tokens=0,
                latency_ms=5,
            )
        return await super().complete(model, messages, tools=tools, temperature=temperature)

    async def stream_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamEvent]:
        msg_tok = estimate_messages_tokens(messages)
        tool_tok = estimate_tokens(json.dumps(tools)) if tools else 0
        self.calls.append(
            {
                "kind": "stream",
                "model": model,
                "messages": messages,
                "tools": tools,
                "msg_tokens": msg_tok,
                "tool_tokens": tool_tok,
                "total_tokens": msg_tok + tool_tok,
                "kwargs": dict(kwargs),
            }
        )
        async for ev in super().stream_complete(model, messages, tools=tools, temperature=temperature):
            yield ev


def test_phase0_scorer_handles_multiline_lists_tables_and_full_answer() -> None:
    answer = (
        "### Active Configuration\n"
        "- Primary database: `PostgreSQL 16` with `pgvector`.\n"
        "- Auth signing: `ED25519_JWKS_24H`.\n\n"
        "### Excluded / Retired:\n"
        "- `JWT_TTL_15M`\n"
        "- `REDIS_CLUSTER_V1`\n"
        "- `RS256_LEGACY_KEY`\n"
        "- `MONGO_SHARD_PROTO`\n\n"
        "### Decision Audit Table\n"
        "| Option | Status | Notes |\n"
        "| --- | --- | --- |\n"
        "| `SQLITE_SYNC_OLD` | Retired | Replaced before launch |\n"
        "| `KAFKA_TEST_CLUSTER` | Abandoned | Too heavy for single-node |\n"
        + (" trailing explanation" * 30)
    )
    # Multi-item bullet list: bullet #1 through bullet #4 under Excluded/Retired must NOT be flagged active
    for retired_tok in (
        "JWT_TTL_15M",
        "REDIS_CLUSTER_V1",
        "RS256_LEGACY_KEY",
        "MONGO_SHARD_PROTO",
        "SQLITE_SYNC_OLD",
        "KAFKA_TEST_CLUSTER",
    ):
        assert not is_token_asserted_as_active(answer, retired_tok), retired_tok

    # Active tokens ARE still detected as active
    assert is_token_asserted_as_active(answer, "PostgreSQL 16")
    assert is_token_asserted_as_active(answer, "ED25519_JWKS_24H")

    # If a token appears retired first AND actively claimed later, it MUST still be flagged active
    mixed = (
        "### Excluded / Retired:\n"
        "- `MONGO_SHARD_PROTO`\n\n"
        "### Active Configuration\n"
        "- Deploy `MONGO_SHARD_PROTO` in production.\n"
    )
    assert is_token_asserted_as_active(mixed, "MONGO_SHARD_PROTO")

    scorer = ExactFactScorer()
    res = scorer.score(
        query="What is active?",
        target_branch="root",
        answer_text=answer,
        expected_keywords=["PostgreSQL 16", "ED25519_JWKS_24H"],
        rejected_keywords=["MONGO_SHARD_PROTO", "KAFKA_TEST_CLUSTER"],
        stale_keywords=["JWT_TTL_15M", "SQLITE_SYNC_OLD"],
    )
    assert res.passed is True
    assert res.rejected_leakage is False
    assert res.stale_contradiction is False
    assert len(res.answer_preview) == 240
    assert res.full_answer == answer


@pytest.mark.asyncio
async def test_phase0_ablation_and_runner_fairness() -> None:
    mini = MultiTangentScenario(
        name="mini_fairness",
        description="Verify summary-only synthesizes through LLM and no-staleness keeps tools.",
        steps=[
            ScenarioStep(kind="chat", branch="root", content="Let's design the auth service."),
            ScenarioStep(kind="spawn", branch="auth", parent="root", title="Auth", goal="Auth design"),
            ScenarioStep(
                kind="chat",
                branch="auth",
                content="Initial proposal: JWT_TTL_15M with secret rotation.",
            ),
            ScenarioStep(kind="finish", branch="auth", content="Use JWT_TTL_15M for access tokens."),
            ScenarioStep(
                kind="revisit",
                branch="auth",
                content="Revision: switch to JWT_TTL_5M_ROTATING immediately.",
            ),
            ScenarioStep(
                kind="verify_query",
                target_branch="auth",
                content="What is the current JWT TTL?",
                expected_keywords=["JWT_TTL_5M_ROTATING"],
                stale_keywords=["JWT_TTL_15M"],
                is_post_revision=True,
            ),
        ],
    )

    rec_llm = _RecordingMockLLM()
    ab_runner = AblationMatrixRunner(model="mock/chatree-demo", llm=rec_llm)
    await ab_runner.run_ablation_matrix(
        scenarios=[mini], preset_names=["no_shadow_turns_folding", "no_staleness"]
    )
    # 0.2: no_shadow_turns_folding must invoke LLM synthesis with [Branch Evidence]
    assert any(
        "[Branch Evidence]" in str(c["messages"][-1].get("content") or "")
        for c in rec_llm.calls
        if c["kind"] == "complete"
    )
    # 0.3: no_staleness must keep tools enabled on the verify_query turn (not strip tools)
    verify_stream_calls = [
        c
        for c in rec_llm.calls
        if c["kind"] == "stream"
        and "What is the current JWT TTL?" in str(c["messages"][-1].get("content") or "")
    ]
    assert verify_stream_calls and all(bool(c["tools"]) for c in verify_stream_calls)

    rec_llm2 = _RecordingMockLLM()
    ev_runner = BenchmarkRunner(model="mock/chatree-demo", llm=rec_llm2)
    await ev_runner.run_suite(
        mini, modes=["tree_summary_only", "tree_ipc_no_staleness"]
    )
    assert any(
        "[Branch Evidence]" in str(c["messages"][-1].get("content") or "")
        for c in rec_llm2.calls
        if c["kind"] == "complete"
    )
    ev_verify_streams = [
        c
        for c in rec_llm2.calls
        if c["kind"] == "stream"
        and "What is the current JWT TTL?" in str(c["messages"][-1].get("content") or "")
    ]
    assert ev_verify_streams and all(bool(c["tools"]) for c in ev_verify_streams)


@pytest.mark.asyncio
async def test_phase1_autonomous_spawn_cannot_bypass_guardrails() -> None:
    # 1. spawn_policy='manual' blocks BOTH autonomous=False and autonomous=True model spawns
    cfg_manual = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(spawn_policy="manual"),
    )
    async with Chatree.in_memory(config=cfg_manual, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Manual Policy")
        tm = await tree.model()
        root = await (await tree.root()).model()
        res_auto = await ct.ipc.spawn_child(
            tree=tm,
            parent=root,
            goal="Sneaky autonomous worker",
            autonomous=True,
            initiated_by_model=True,
        )
        assert res_auto.tool_output.get("error") == "SPAWN_POLICY_MANUAL"
        assert len(await tree.nodes()) == 1

    # 2. spawn_policy='suggest' turns autonomous=True model spawns into suggestions (no child created)
    cfg_suggest = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(spawn_policy="suggest"),
    )
    async with Chatree.in_memory(config=cfg_suggest, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Suggest Policy")
        tm = await tree.model()
        root = await (await tree.root()).model()
        res_sug = await ct.ipc.spawn_child(
            tree=tm,
            parent=root,
            goal="Suggested autonomous worker",
            autonomous=True,
            initiated_by_model=True,
        )
        assert res_sug.tool_output.get("status") == "suggested"
        assert len(await tree.nodes()) == 1

    # 3. allow_autonomous_spawns=False blocks autonomous=True even when spawn_policy='auto'
    cfg_no_auto = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(spawn_policy="auto", allow_autonomous_spawns=False),
    )
    async with Chatree.in_memory(config=cfg_no_auto, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("No Autonomous")
        tm = await tree.model()
        root = await (await tree.root()).model()
        res_dis = await ct.ipc.spawn_child(
            tree=tm,
            parent=root,
            goal="Blocked autonomous worker",
            autonomous=True,
            initiated_by_model=True,
        )
        assert res_dis.tool_output.get("error") == "AUTONOMOUS_SPAWN_DISABLED"

    # 4. Explicit max_active_children=1 also blocks 2nd autonomous=True spawn
    cfg_cap = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(spawn_policy="auto", max_active_children=1),
    )
    async with Chatree.in_memory(config=cfg_cap, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Explicit Cap")
        await tree.spawn("First child", title="Child 1", switch_head=False)
        tm = await tree.model()
        root = await (await tree.root()).model()
        res_cap = await ct.ipc.spawn_child(
            tree=tm,
            parent=root,
            goal="Bypass attempt",
            autonomous=True,
            initiated_by_model=True,
        )
        assert res_cap.tool_output.get("error") == "MAX_ACTIVE_CHILDREN_EXCEEDED"


@pytest.mark.asyncio
async def test_phase1_unlimited_chat_topics_and_bounded_subagents() -> None:
    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(spawn_policy="auto", max_active_subagents=3),
    )
    async with Chatree.in_memory(config=cfg, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Many Topics")
        tm = await tree.model()
        root = await (await tree.root()).model()

        # Model can open 25+ interactive chat topics from one parent with default max_active_children=None
        for i in range(25):
            res = await ct.ipc.spawn_child(
                tree=tm,
                parent=root,
                goal=f"Topic {i}",
                title=f"Topic {i}",
                autonomous=False,
                switch_head=False,
                initiated_by_model=True,
            )
            assert "child_id" in res.tool_output, res.tool_output

        # Active subagents remain strictly capped at max_active_subagents=3
        for j in range(3):
            res_sub = await ct.ipc.spawn_child(
                tree=tm,
                parent=root,
                goal=f"Active subagent {j}",
                title=f"Worker {j}",
                node_kind="subagent",
                autonomous=False,
                switch_head=False,
                initiated_by_model=True,
            )
            assert "child_id" in res_sub.tool_output

        res_sub_over = await ct.ipc.spawn_child(
            tree=tm,
            parent=root,
            goal="4th active subagent",
            title="Worker 4",
            node_kind="subagent",
            autonomous=False,
            switch_head=False,
            initiated_by_model=True,
        )
        assert res_sub_over.tool_output.get("error") == "MAX_ACTIVE_SUBAGENTS_EXCEEDED"


@pytest.mark.asyncio
async def test_phase1_newest_ordering_omitted_counts_steady_state_ids_and_paging() -> None:
    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Directory Scale")
        root = await tree.root()
        topic_ids: list[str] = []
        for i in range(25):
            ch = await tree.spawn(f"Work on topic {i}", title=f"Topic {i:02d}", switch_head=False)
            await ct.repo.commit_node_summary(ch.id, f"Summary for Topic {i:02d}")
            topic_ids.append(ch.id)

        # Touch Topic 24 last so it has the newest updated_at
        newest_id = topic_ids[-1]
        await ct.repo.commit_node_summary(newest_id, "Latest updated Topic 24 summary")

        tm = await tree.model()
        rm = await root.model()
        env_full, _, _ = await ct.context.build_ephemeral_envelope(
            tm, rm, compact_steady_state=False
        )
        # Newest topic MUST appear in both TREE children and BRANCH SUMMARIES (not dropped behind oldest 8!)
        assert newest_id in env_full
        assert "(+15 more)" in env_full  # 25 children, 10 shown -> +15 more
        assert "(+17 more summarized branches" in env_full  # 25 summaries, 8 shown -> +17 more

        # Steady-state envelope keeps branch IDs + short titles visible!
        env_steady, _, _ = await ct.context.build_ephemeral_envelope(
            tm, rm, compact_steady_state=True
        )
        assert "BRANCH SUMMARIES: 25 unchanged" in env_steady
        assert newest_id in env_steady
        assert "Topic 24" in env_steady
        assert "(+17 more)" in env_steady

        # Paged wildcard search (`query="*"`) and SDK `list_branches()` reach all 26 nodes
        seen_tool_ids: list[str] = []
        offset = 0
        while True:
            res = await ct.ipc.search_tree(
                tree=tm, caller=rm, query="*", scope="all", top_k=10, offset=offset
            )
            out = res.tool_output
            assert out["mode"] == "list"
            assert out["total_branches"] == 26
            page_ids = [b["node_id"] for b in out["branches"]]
            seen_tool_ids.extend(page_ids)
            if not out["has_more"]:
                break
            offset += len(page_ids)
        assert len(seen_tool_ids) == 26
        assert len(set(seen_tool_ids)) == 26

        sdk_page = await tree.list_branches(offset=0, limit=5)
        assert sdk_page["total_branches"] == 26
        assert sdk_page["has_more"] is True
        assert sdk_page["branches"][0]["node_id"] == newest_id


@pytest.mark.asyncio
async def test_phase2_handoff_pinned_block_trimming_and_whole_request_window() -> None:
    # 1. Model context window lookup covers local models, cloud models, and custom overrides
    assert resolve_model_context_window("ollama/llama3.1") == 8_192
    assert resolve_model_context_window("deepseek/deepseek-chat") == 128_000
    assert (
        resolve_model_context_window(
            "ollama/llama3.1", overrides={"ollama/llama3.1": 32_768}, configured_max=120_000
        )
        == 32_768
    )

    # 2. Handoff preservation + block-aligned hysteresis trimming
    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(
            max_branch_context_tokens=1_400,
            auto_compact_enabled=False,
        ),
    )
    async with Chatree.in_memory(config=cfg, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Trim Test")
        cont = await tree.spawn(
            "Continue work",
            title="Cont Session",
            mode="clean",
            summary_brief="ANCHOR_PINNED_777: Always use Ed25519.",
        )
        for i in range(18):
            await ct.repo.append_message(
                cont.id, tree.id, "user", f"Turn {i} filler " + ("token " * 32)
            )
            await ct.repo.append_message(
                cont.id, tree.id, "assistant", f"Ack {i} " + ("reply " * 20)
            )

        prefix_a = await ct.context.build_static_prefix(await cont.model())
        # Brief/handoff MUST remain pinned at payload[1..2]
        assert "ANCHOR_PINNED_777" in prefix_a[1]["content"]
        assert "Chatree Context Guard" in prefix_a[3]["content"]

        # Append one small turn inside the 25% headroom -> trimmed prefix head+marker+first kept msg
        # must stay byte-for-byte identical (no per-turn KV cache thrash!)
        await ct.repo.append_message(cont.id, tree.id, "user", "Small follow-up question.")
        prefix_b = await ct.context.build_static_prefix(await cont.model())
        assert prefix_b[: len(prefix_a)] == prefix_a

    # 3. Whole-request budget guard: NO provider call may exceed the resolved model context window
    rec = _RecordingMockLLM()
    cfg_small = ChatreeConfig(
        db_path=":memory:",
        default_model="mock/tiny-4k",
        guardrails=GuardrailPolicy(
            max_branch_context_tokens=120_000,
            model_context_windows={"mock/tiny-4k": 4_096},
            reply_reserve_tokens=400,
            auto_compact_enabled=False,
        ),
    )
    async with Chatree.in_memory(config=cfg_small, llm=rec) as ct:
        tree = await ct.create_tree("Strict Window", default_model="mock/tiny-4k")
        root = await tree.root()
        for i in range(16):
            await ct.repo.append_message(
                root.id, tree.id, "user", f"History {i}: " + ("context " * 90)
            )
            await ct.repo.append_message(
                root.id, tree.id, "assistant", f"Response {i}: " + ("answer " * 90)
            )
        giant_prompt = "Summarize everything: " + ("detail " * 900)
        await tree.chat(giant_prompt)
        assert rec.calls
        for c in rec.calls:
            assert c["total_tokens"] <= 4_096, f"Exceeded 4096-token model window: {c['total_tokens']}"


@pytest.mark.asyncio
async def test_phase2_bounded_stale_and_mailbox_envelopes() -> None:
    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(max_stale_notices=8, max_mailbox_notices=6),
    )
    async with Chatree.in_memory(config=cfg, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Bounded Notices")
        root = await tree.root()
        for i in range(40):
            ch = await tree.spawn(f"Dep {i}", title=f"Dep {i}", switch_head=False)
            await ct.repo.commit_node_summary(ch.id, f"Initial summary {i}")
            await ct.repo.upsert_dependency(
                tree_id=tree.id,
                consumer_node_id=root.id,
                source_node_id=ch.id,
                seen_summary_version=1,
                seen_msg_seq=0,
            )
            # Revise summary -> creates 40 stale dependencies on root!
            await ct.repo.commit_node_summary(
                ch.id, f"Revised summary {i} with new value {i + 100}", severity="material"
            )
            if i < 15:
                await ct.repo.create_mailbox_item(
                    tree_id=tree.id,
                    from_node_id=ch.id,
                    to_node_id=root.id,
                    kind="message",
                    status="unread",
                    summary=f"Background deliverable {i}",
                    artifact_refs=[],
                    payload={"sender_title": f"Dep {i}"},
                )

        tm = await tree.model()
        rm = await root.model()
        unread = await ct.repo.list_mailbox_items(tree.id, to_node_id=root.id, status="unread")
        env, _, _ = await ct.context.build_ephemeral_envelope(tm, rm, unread_mailbox=unread)
        assert "(+32 more stale dependencies)" in env
        assert "(+9 more unread mailbox items)" in env
        assert estimate_tokens(env) < 1_500


@pytest.mark.asyncio
async def test_phase2_bounded_compaction_handoff_across_generations() -> None:
    rec = _RecordingMockLLM(verbose_compactor_chars=2_800)
    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(
            compaction_handoff_max_chars=900,
            auto_compact_enabled=False,
        ),
    )
    async with Chatree.in_memory(config=cfg, llm=rec) as ct:
        tree = await ct.create_tree("Compaction Cap")
        cur = await tree.root()
        for gen in range(10):
            await ct.repo.append_message(cur.id, tree.id, "user", f"Gen {gen} work on GEN_{gen}_KEY.")
            await ct.repo.append_message(cur.id, tree.id, "assistant", f"Recorded GEN_{gen}_KEY.")
            nxt, info = await tree.compact(node_id=cur.id, keep_recent_turns=0)
            src_node = await ct.repo.get_node(cur.id)
            assert src_node is not None and src_node.summary is not None
            assert len(src_node.summary) <= 900, f"Gen {gen} handoff grew to {len(src_node.summary)} chars"
            cur = nxt


@pytest.mark.asyncio
async def test_phase2_overbudget_shadow_query_recovers_turn1_middle_and_final() -> None:
    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(
            max_branch_context_tokens=3_200,
            auto_compact_enabled=False,
        ),
    )
    async with Chatree.in_memory(config=cfg, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Overbudget Shadow")
        root = await tree.root()
        worker = await tree.spawn("Long investigation", title="Long Worker", mode="clean")

        await ct.repo.append_message(
            worker.id, tree.id, "user", "Turn 1 locked fact: TURN1_TOKEN_ALPHA is OMEGA_111."
        )
        await ct.repo.append_message(
            worker.id, tree.id, "assistant", "Confirmed TURN1_TOKEN_ALPHA = OMEGA_111."
        )
        for i in range(2, 50):
            if i == 25:
                await ct.repo.append_message(
                    worker.id, tree.id, "user", "Turn 25 locked fact: MID_TOKEN_BETA is SIGMA_222."
                )
                await ct.repo.append_message(
                    worker.id, tree.id, "assistant", "Confirmed MID_TOKEN_BETA = SIGMA_222."
                )
            else:
                await ct.repo.append_message(
                    worker.id, tree.id, "user", f"Filler discussion {i}: " + ("padding " * 60)
                )
                await ct.repo.append_message(
                    worker.id, tree.id, "assistant", f"Filler reply {i}: " + ("notes " * 45)
                )
        await ct.repo.append_message(
            worker.id, tree.id, "user", "Turn 50 locked fact: FINAL_TOKEN_GAMMA is ZETA_333."
        )
        await ct.repo.append_message(
            worker.id, tree.id, "assistant", "Confirmed FINAL_TOKEN_GAMMA = ZETA_333."
        )
        await ct.rag.index_node_chunks(worker.id)

        # Verify assembled shadow messages mark Partial View + include Searched Excerpts for Turn 1
        assembled = await ct.context.assemble_for_shadow_query(
            await worker.model(),
            await root.model(),
            "What is TURN1_TOKEN_ALPHA?",
            rag=ct.rag,
        )
        joined = "\n".join(str(m.get("content") or "") for m in assembled)
        assert "Partial View" in joined
        assert "Searched Excerpts" in joined
        assert "OMEGA_111" in joined

        # Verify end-to-end shadow queries recover Turn 1, Turn 15, AND Turn 30
        r1 = await tree.request(worker.id, prompt="What is TURN1_TOKEN_ALPHA?", caller_id=root.id)
        r15 = await tree.request(worker.id, prompt="What is MID_TOKEN_BETA?", caller_id=root.id)
        r30 = await tree.request(worker.id, prompt="What is FINAL_TOKEN_GAMMA?", caller_id=root.id)
        assert r1["found_in_branch"] is True and "OMEGA_111" in r1["answer"]
        assert r15["found_in_branch"] is True and "SIGMA_222" in r15["answer"]
        assert r30["found_in_branch"] is True and "ZETA_333" in r30["answer"]


@pytest.mark.asyncio
async def test_phase3_incremental_o1_chunk_indexing_and_exact_equivalence() -> None:
    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(auto_compact_enabled=False),
    )
    async with Chatree.in_memory(config=cfg, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Incremental Indexing")
        root = await tree.root()

        # Warm up 10 turns and populate vector cache via search
        for i in range(10):
            await ct.repo.append_message(
                root.id, tree.id, "user", f"Turn {i} topic alpha_{i}: " + ("word " * 55)
            )
            await ct.repo.append_message(
                root.id, tree.id, "assistant", f"Reply {i}: " + ("resp " * 45)
            )
            await ct.rag.index_node_chunks(root.id)

        await ct.rag.search(tree.id, "alpha_3", top_k=3)
        misses_before = ct.repo.vector_cache_stats()["misses"]

        # Grow to 80 turns while measuring chunk rows written & median SQLite changes per turn
        chunk_writes_late: list[int] = []
        changes_late: list[int] = []
        for i in range(10, 80):
            await ct.repo.append_message(
                root.id, tree.id, "user", f"Turn {i} topic alpha_{i}: " + ("word " * 55)
            )
            await ct.repo.append_message(
                root.id, tree.id, "assistant", f"Reply {i}: " + ("resp " * 45)
            )
            before_changes = ct.repo.conn.total_changes
            await ct.rag.index_node_chunks(root.id)
            delta = ct.repo.conn.total_changes - before_changes
            if i >= 70:
                chunk_writes_late.append(ct.repo.last_chunk_write_count)
                changes_late.append(delta)

        # O(1) invariant: incremental indexing writes ONLY 1-2 tail chunks per turn (NOT all 25+!)
        total_chunks = len(await ct.repo.list_node_chunk_embeddings(root.id))
        assert total_chunks >= 25
        assert max(chunk_writes_late) <= 2, f"Expected O(1) tail chunk writes, got {chunk_writes_late}"
        median_changes = sorted(changes_late)[len(changes_late) // 2]
        assert median_changes <= 20, f"Expected flat O(1) SQLite delta, got {changes_late}"

        # Vector cache stayed hot (patched in-place, zero cache misses on search!)
        # Turn 75 user message is seq = 75*2 + 1 = 151
        hits_after = await ct.rag.search(tree.id, "alpha_75", top_k=3)
        assert hits_after
        assert any(
            (h.seq_range is not None and h.seq_range[0] <= 151 <= h.seq_range[1])
            or "alpha_75" in h.snippet
            for h in hits_after
        ), hits_after
        assert ct.repo.vector_cache_stats()["misses"] == misses_before

        # Exact equivalence check: incremental chunks MUST match a fresh full re-chunk byte-for-byte
        inc_rows = [
            (c_idx, s_start, s_end, content)
            for c_idx, s_start, s_end, content, _blob in await ct.repo.list_node_chunk_embeddings(
                root.id
            )
        ]
        # Force full rebuild by clearing chunk rows
        async with ct.repo.db.write_transaction():
            await ct.repo.conn.execute(
                "DELETE FROM embeddings_fts WHERE node_id = ? AND granularity = 'chunk'", (root.id,)
            )
            await ct.repo.conn.execute(
                "DELETE FROM embeddings WHERE node_id = ? AND granularity = 'chunk'", (root.id,)
            )
        await ct.rag.index_node_chunks(root.id)
        full_rows = [
            (c_idx, s_start, s_end, content)
            for c_idx, s_start, s_end, content, _blob in await ct.repo.list_node_chunk_embeddings(
                root.id
            )
        ]
        assert inc_rows == full_rows
