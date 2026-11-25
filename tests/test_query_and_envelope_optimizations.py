from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from chatree import Chatree, ChatreeConfig, ChatreeSystemAdapter, GuardrailPolicy
from chatree.engine.query import (
    classify_query_intent,
    extract_comparison_subqueries,
    parse_escalation_directive,
)
from chatree.engine.staleness import _compact_cold_prefix_messages, enrich_date_with_weekday
from chatree.eval.external_benchmarks import ExternalSession, ExternalTurn
from chatree.storage.repository import SQLiteRepository
from chatree.types import LLMResponse


class _FusedQueryStubLLM:
    """Deterministic non-mock stub LLM to test selective Hop-2 skipping vs. escalation + cold summary."""

    def __init__(self, *, escalate_candidates: list[str] | None = None) -> None:
        self.escalate_candidates = list(escalate_candidates or [])
        self.plan_calls = 0
        self.synth_calls = 0
        self.shadow_calls: list[str] = []
        self.summary_tools_seen: list[bool] = []

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
        sys_msg = str(messages[0].get("content", "")) if messages else ""
        if "[Shadow Query |" in last:
            self.shadow_calls.append(last)
            body = " ".join(str(m.get("content", "")) for m in messages[1:-1])
            return LLMResponse(
                content=f"Shadow found: {body[:160]}",
                model_used=model,
                tokens_in=60,
                tokens_out=20,
            )
        if "[Chatree Summarizer]" in last:
            self.summary_tools_seen.append(bool(tools))
            return LLMResponse(
                content=(
                    "TITLE: Stub Session Topic\n"
                    "Summary of branch.\n"
                    'FACTS_JSON: [{"fact": "Alice visited Rome.", "entities": ["Alice", "Rome"], "resolved_date": "2025-04-01"}]'
                ),
                model_used=model,
                tokens_in=50,
                tokens_out=30,
            )
        if "root coordinator of a dialogue tree" in sys_msg:
            self.plan_calls += 1
            unq_section = last.split("Unqueried Branch Summary Map:", 1)[-1]
            pick = next((cid for cid in self.escalate_candidates if cid in unq_section), None)
            if pick is not None:
                return LLMResponse(
                    content=f'{{"need_more": true, "target_ids": ["{pick}"], "followup_query": ""}}',
                    model_used=model,
                    tokens_in=120,
                    tokens_out=15,
                )
            return LLMResponse(
                content='{"need_more": false, "target_ids": [], "followup_query": ""}',
                model_used=model,
                tokens_in=120,
                tokens_out=12,
            )
        self.synth_calls += 1
        return LLMResponse(
            content=f"Synthesized final answer (plan={self.plan_calls}, synth={self.synth_calls}).",
            model_used=model,
            tokens_in=140,
            tokens_out=25,
        )

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


def test_weekday_anchoring_and_query_intent() -> None:
    assert enrich_date_with_weekday("2023-05-25") == "2023-05-25 (Thursday)"
    assert enrich_date_with_weekday("1:56 pm on 8 May, 2023") == "1:56 pm on 8 May, 2023 (Monday)"
    # Idempotent when weekday already present
    assert enrich_date_with_weekday("2023/05/20 (Sat) 02:21") == "2023/05/20 (Sat) 02:21"
    assert enrich_date_with_weekday("unknown date") == "unknown date"

    intent_cnt = classify_query_intent("How many model kits have I bought altogether?")
    assert intent_cnt["is_counting"] is True
    assert intent_cnt["is_update"] is False

    # Elapsed-time duration must NOT inflate counting fan-out, whereas cumulative days spent DOES
    intent_dur = classify_query_intent("How many weeks ago did I receive the crystal chandelier?")
    assert intent_dur["is_counting"] is False
    assert intent_dur["is_temporal_compare"] is True

    intent_sum_days = classify_query_intent("How many days did I spend on camping trips?")
    assert intent_sum_days["is_counting"] is True

    intent_upd = classify_query_intent("Where does Rachel currently live after her recent relocation?")
    assert intent_upd["is_update"] is True

    subs = extract_comparison_subqueries(
        "[Current date: 2023-06-01] How many days passed between my visit to MoMA and my trip to Ancient Art?"
    )
    assert len(subs) == 2
    assert "MoMA" in subs[0] and "Ancient Art" in subs[1]
    # Must not over-split normal non-comparison conjunctions
    assert extract_comparison_subqueries("Tell me about research and development plans") == []

    esc_ok, tids, fq = parse_escalation_directive(
        'NEED_MORE_BRANCHES: ["n_a", "n_invalid", "n_b"]', {"n_a", "n_b", "n_c"}
    )
    assert esc_ok is True
    assert tids == ["n_a", "n_b"]
    assert fq == ""

    esc_no, tids_no, _ = parse_escalation_directive("Final direct answer: 42.", {"n_a"})
    assert esc_no is False and tids_no == []


@pytest.mark.asyncio
async def test_selective_hop2_skip_escalation_and_cold_summary_tools() -> None:
    # Case 1: Clean single-hop query with >= 2 Hop-1 hits SKIPS the redundant Hop-2 planner call!
    stub_direct = _FusedQueryStubLLM(escalate_candidates=None)
    cfg = ChatreeConfig(db_path=":memory:", default_model="stub/test-model")
    async with Chatree.in_memory(config=cfg, llm=stub_direct) as ct:
        tree = await ct.create_tree("Direct Query Test", default_model="stub/test-model")
        root = await tree.root()
        cold_child = None
        for idx in range(1, 5):
            ch = await root.spawn(
                goal=f"Topic {idx} | Date: 2025-05-0{idx}",
                title=f"Branch {idx}",
                mode="clean",
                switch_head=False,
            )
            if idx == 1:
                cold_child = ch
            await ct.repo.append_message(
                node_id=ch.id, tree_id=tree.id, role="user", content=f"Fact alpha-{idx} on port 800{idx}."
            )
            await ct.repo.commit_node_summary(ch.id, f"Branch {idx} covers alpha-{idx}.", status="COMPLETED")
            await ct.rag.index_node_chunks(ch.id)
            await ct.rag.index_node_summary(ch.id)

        # Verify cold-branch summary omits unused 9 tool schemas (`tools=None`), while warm branch keeps them
        assert cold_child is not None
        tm = await tree.model()
        cm = await cold_child.model()
        assert cm.tokens_in == 0
        await ct.staleness.ensure_fresh_summary(tm, cm, force=True)
        assert stub_direct.summary_tools_seen[-1] is False
        await ct.repo.increment_node_usage(cm.id, 400, 80, 0)
        cm_warm = await cold_child.model()
        await ct.staleness.ensure_fresh_summary(tm, cm_warm, force=True)
        assert stub_direct.summary_tools_seen[-1] is True

        out = await ct.query_engine.deep_ask(
            tree=await tree.model(),
            caller=await root.model(),
            query="What port is used for alpha-1?",
            max_targets=2,
            enable_recursive_hop=True,
            include_branch_summary_map=True,
        )
        assert stub_direct.plan_calls == 0
        assert stub_direct.synth_calls == 1
        assert "Synthesized final answer" in out["answer"]

    # Case 2: Low Hop-1 hits (<2) or broad query triggers dedicated Hop-2 planner + shadow + synthesis
    stub_esc = _FusedQueryStubLLM()
    async with Chatree.in_memory(config=cfg, llm=stub_esc) as ct:
        tree = await ct.create_tree("Escalation Query Test", default_model="stub/test-model")
        root = await tree.root()
        ch1 = await root.spawn(goal="Alpha | Date: 2025-05-01", title="Alpha Branch", mode="clean", switch_head=False)
        ch2 = await root.spawn(goal="Omega | Date: 2025-05-05", title="Omega Branch", mode="clean", switch_head=False)
        await ct.repo.append_message(node_id=ch1.id, tree_id=tree.id, role="user", content="Alpha secret is ALPHA_99.")
        await ct.repo.append_message(node_id=ch2.id, tree_id=tree.id, role="user", content="Omega secret is OMEGA_77.")
        await ct.repo.commit_node_summary(ch1.id, "Alpha branch discusses ALPHA_99.", status="COMPLETED")
        await ct.repo.commit_node_summary(ch2.id, "Omega branch discusses OMEGA_77.", status="COMPLETED")
        await ct.rag.index_tree(tree.id)

        stub_esc.escalate_candidates = [ch1.id, ch2.id]
        out2 = await ct.query_engine.deep_ask(
            tree=await tree.model(),
            caller=await root.model(),
            query="Which secret would you recommend from Alpha or Omega?",
            max_targets=1,
            enable_recursive_hop=True,
            include_branch_summary_map=True,
        )
        assert stub_esc.plan_calls == 1
        assert stub_esc.synth_calls == 1
        assert {ch1.id, ch2.id}.issubset(set(out2["targets_queried"]))


@pytest.mark.asyncio
async def test_hybrid_summary_map_scales_with_directory() -> None:
    cfg = ChatreeConfig(db_path=":memory:", default_model="mock/chatree-demo")
    async with Chatree.in_memory(config=cfg) as ct:
        tree = await ct.create_tree("Large Tree Map")
        root = await tree.root()
        nids: list[str] = []
        titles: dict[str, str] = {}
        dates: dict[str, str] = {}
        summaries: dict[str, str] = {}
        for i in range(30):
            ch = await root.spawn(goal=f"Topic {i}", title=f"Branch {i:02d}", mode="clean", switch_head=False)
            nids.append(ch.id)
            titles[ch.id] = f"Branch {i:02d}"
            dates[ch.id] = f"2025-01-{(i % 28) + 1:02d}"
            summaries[ch.id] = f"Detailed summary text for branch {i:02d} about project module {i}."
            await ct.repo.commit_node_summary(ch.id, summaries[ch.id], status="COMPLETED")
            await ct.rag.index_node_summary(ch.id)

        smap = await ct.query_engine._build_hybrid_summary_map(
            tree_id=tree.id,
            caller=await root.model(),
            query="project module 29 latest update",
            scope="all",
            ordered_nids=nids,
            queried_ids={nids[0]},
            titles=titles,
            dates=dates,
            summaries=summaries,
            is_update=True,
            full_threshold=20,
            top_unqueried=12,
        )
        assert "[Other Branches Directory" in smap
        # Every single one of the 30 branch IDs remains visible (either full summary or directory line)
        for nid in nids:
            assert nid in smap


@pytest.mark.asyncio
async def test_parallel_ingest_conversation_and_shadow_hash_normalization() -> None:
    # Exact shadow-cache prompt normalization ignores case, whitespace, and trailing punctuation
    assert SQLiteRepository.hash_shadow_prompt("What JWT TTL did we pick?") == (
        SQLiteRepository.hash_shadow_prompt("  what   jwt ttl did we pick  ")
    )

    adapter = ChatreeSystemAdapter(
        name="chatree_parallel_test",
        model="mock/chatree-demo",
        generate_session_summaries=True,
        ingest_concurrency=3,
    )
    await adapter.reset("conv_par")
    sessions = [
        ExternalSession(
            session_id=f"s{i}",
            title=f"Session {i}",
            date_time=f"2025-05-0{i}",
            turns=[ExternalTurn(role="user", content=f"Active fact K{i}=VAL_{i}.")],
            archived=(i == 4),
            archive_reason="scrapped prototype" if i == 4 else "",
        )
        for i in range(1, 5)
    ]
    await adapter.ingest_conversation(sessions)
    assert adapter._ct is not None and adapter._tree is not None
    nodes = await adapter._ct.repo.list_nodes(adapter._tree.id)
    archived = [n for n in nodes if n.status == "ARCHIVED"]
    completed = [n for n in nodes if n.status == "COMPLETED"]
    assert len(archived) == 1
    assert len(completed) == 3
    await adapter.close()


@pytest.mark.asyncio
async def test_conservative_steady_state_envelope() -> None:
    cfg = ChatreeConfig(
        db_path=":memory:",
        default_model="mock/chatree-demo",
        guardrails=GuardrailPolicy(compact_steady_state_envelope=True, auto_title_enabled=False),
    )
    captured_envelopes: list[str] = []

    async with Chatree.in_memory(config=cfg) as ct:
        tree = await ct.create_tree("Steady Envelope Tree")
        root = await tree.root()
        ch = await root.spawn(goal="Auth design", title="Auth Branch", mode="clean", switch_head=False)
        await ct.repo.commit_node_summary(ch.id, "Settled on ED25519 signing.", status="COMPLETED")

        orig_build = ct.context.build_ephemeral_envelope

        async def _spy(*args: Any, **kwargs: Any) -> tuple[str, list[str], list[str]]:
            env, cids, notices = await orig_build(*args, **kwargs)
            captured_envelopes.append(env)
            return env, cids, notices

        ct.context.build_ephemeral_envelope = _spy  # type: ignore[method-assign]

        # Turn 1 on root: full BRANCH SUMMARIES shown
        await root.chat("First question on root.")
        assert 'BRANCH SUMMARIES [id "title"' in captured_envelopes[-1]
        assert "TREE [" in captured_envelopes[-1]

        # Turn 2 on same root with zero topology/summary changes: compact 1-line BRANCH SUMMARIES,
        # while TREE stays fully intact!
        await root.chat("Follow-up question on root.")
        assert "BRANCH SUMMARIES: 1 unchanged" in captured_envelopes[-1]
        assert "TREE [" in captured_envelopes[-1]

        # Mutate sibling summary -> Turn 3 restores full BRANCH SUMMARIES automatically
        await ct.repo.commit_node_summary(ch.id, "Revised to RS256 signing.", status="COMPLETED")
        await root.chat("Third question after sibling change.")
        assert 'BRANCH SUMMARIES [id "title"' in captured_envelopes[-1]


def test_role_balanced_proportional_cold_trimming() -> None:
    # Case 1: Verbose assistant essays (~20k chars) + late user turn #12 -> 100% of user turns kept,
    # and ALL assistant turns represented proportionally (no chronological `break` dropping late turns!).
    long_session: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for i in range(7):
        long_session.append({"role": "user", "content": f"User turn #{i}: important preference_{i}."})
        long_session.append({"role": "assistant", "content": f"Asst#{i} start " + ("filler " * 420) + f" END_{i}"})
    compacted = _compact_cold_prefix_messages(long_session, char_budget=7_500)
    assert len(compacted) == 14
    blob = "\n".join(m["content"] for m in compacted)
    for i in range(7):
        assert f"preference_{i}" in blob
        assert f"END_{i}" in blob
    assert sum(len(m["content"]) for m in compacted) <= 7_900

    # Case 2: Giant 10k pasted log in one user turn -> user paste trimmed to <=50% so assistant stays intact!
    giant_paste: list[dict[str, Any]] = [
        {"role": "user", "content": "HEADER_LOG " + ("stackline\n" * 1000) + " TAIL_ERR"},
        {"role": "assistant", "content": "Root cause is VALKEY_TIMEOUT_500MS in pool.rs."},
    ]
    comp2 = _compact_cold_prefix_messages(giant_paste, char_budget=7_500)
    assert len(comp2) == 2
    assert "HEADER_LOG" in comp2[0]["content"] and "TAIL_ERR" in comp2[0]["content"]
    assert "VALKEY_TIMEOUT_500MS" in comp2[1]["content"]


@pytest.mark.asyncio
async def test_hop2_runs_both_chosen_tids_and_followup_query() -> None:
    class _DualHop2Stub(_FusedQueryStubLLM):
        def __init__(self, target_id: str, followup_query: str) -> None:
            super().__init__()
            self._tid = target_id
            self._fq = followup_query

        async def complete(
            self,
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.0,
            tool_choice: Any = None,
            **kwargs: Any,
        ) -> LLMResponse:
            sys_msg = str(messages[0].get("content", "")) if messages else ""
            if "root coordinator of a dialogue tree" in sys_msg:
                self.plan_calls += 1
                return LLMResponse(
                    content=f'{{"need_more": true, "target_ids": ["{self._tid}"], "followup_query": "{self._fq}"}}',
                    model_used=model,
                    tokens_in=100,
                    tokens_out=20,
                )
            return await super().complete(
                model=model, messages=messages, tools=tools, temperature=temperature, tool_choice=tool_choice
            )

    cfg = ChatreeConfig(db_path=":memory:", default_model="stub/dual-hop2")
    stub = _DualHop2Stub("", "customized phone case")
    async with Chatree.in_memory(config=cfg, llm=stub) as ct:
        tree = await ct.create_tree("Dual Hop2 Tree", default_model="stub/dual-hop2")
        root = await tree.root()
        b1 = await root.spawn(goal="Nursery | Date: 2023-02-05", title="Nursery", mode="clean", switch_head=False)
        b2 = await root.spawn(goal="Shower | Date: 2023-02-10", title="Baby Shower", mode="clean", switch_head=False)
        b3 = await root.spawn(goal="Gifts | Date: 2023-02-20", title="Sibling Gifts", mode="clean", switch_head=False)
        await ct.repo.append_message(node_id=b1.id, tree_id=tree.id, role="user", content="Helped prepare nursery.")
        await ct.repo.append_message(node_id=b2.id, tree_id=tree.id, role="user", content="Picked baby shower gifts.")
        await ct.repo.append_message(node_id=b3.id, tree_id=tree.id, role="user", content="Ordered customized phone case.")
        await ct.repo.commit_node_summary(b1.id, "Nursery prep.", status="COMPLETED")
        await ct.repo.commit_node_summary(b2.id, "Baby shower.", status="COMPLETED")
        await ct.repo.commit_node_summary(b3.id, "Ordered customized phone case for friend.", status="COMPLETED")
        await ct.rag.index_tree(tree.id)

        stub._tid = b2.id
        out = await ct.query_engine.deep_ask(
            tree=await tree.model(),
            caller=await root.model(),
            query="Which happened first: nursery or baby shower?",
            max_targets=1,
            enable_recursive_hop=True,
            include_branch_summary_map=True,
        )
        assert stub.plan_calls == 1
        # Both explicit `target_ids` (b2) AND `followup_query` hit (b3) were queried in Hop-2!
        assert b2.id in out["targets_queried"]
        assert b3.id in out["targets_queried"]

