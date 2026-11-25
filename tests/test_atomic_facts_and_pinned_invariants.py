from __future__ import annotations

from typing import Any

import pytest

from chatree import Chatree, ChatreeConfig, GuardrailPolicy, MockLLMProvider
from chatree.engine.staleness import parse_summary_and_facts
from chatree.types import LLMResponse


@pytest.mark.asyncio
async def test_tier25_atomic_temporal_facts_quarantine_and_piggybacked_parsing() -> None:
    """
    Verify Tier-2.5 Provenance-Linked Atomic Temporal Facts:
      1. `parse_summary_and_facts` handles `FACTS_JSON: [...]`, JSON objects, trailing prose,
         and malformed JSON gracefully without ever losing the summary.
      2. Piggybacked single-call extraction stores facts strictly scoped to the branch (append-only
         across branches; editing one branch only supersedes THAT branch's old facts!).
      3. Entity-boosted hybrid retrieval surfaces cross-branch dated atomic facts (`granularity='fact'`).
      4. Archiving a dead-end branch immediately quarantines its atomic facts from active search!
    """
    # 1. Parser robustness (including trailing prose like Mem0's Extra data failure!)
    s1, f1 = parse_summary_and_facts(
        "Melanie talked about running a charity race.\n"
        'FACTS_JSON: [{"fact": "Melanie ran a charity race on 2023-05-20.", '
        '"entities": ["Melanie"], "resolved_date": "2023-05-20"}]\nExtra trailing note!'
    )
    assert "Melanie talked about" in s1 and "FACTS_JSON" not in s1
    assert len(f1) == 1 and f1[0]["resolved_date"] == "2023-05-20"

    s2, f2 = parse_summary_and_facts("Plain text summary with broken FACTS_JSON: [{bad json")
    assert s2 == "Plain text summary with broken"
    assert f2 == []

    # Truncated JSON array salvage (e.g. model hit max_tokens mid-fact)
    s3, f3 = parse_summary_and_facts(
        "Session covered two events.\n"
        'FACTS_JSON: [{"fact": "Alice moved to Austin on 2023-06-01.", "entities": ["Alice"], '
        '"resolved_date": "2023-06-01"}, {"fact": "Bob started pian'
    )
    assert s3 == "Session covered two events."
    assert len(f3) == 1 and f3[0]["resolved_date"] == "2023-06-01"

    # 2. Piggybacked extraction in StalenessService.ensure_fresh_summary (0 extra LLM calls!)
    class _FactExtractingMockLLM(MockLLMProvider):
        def __init__(self) -> None:
            super().__init__(stream_delay_s=0.0)
            self.summary_calls = 0

        async def complete(
            self,
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.2,
            **kwargs: Any,
        ) -> LLMResponse:
            if any("[Chatree Summarizer]" in str(m.get("content", "")) for m in messages):
                self.summary_calls += 1
                body = (
                    "Melanie completed a 5K mental health charity race.\n"
                    'FACTS_JSON: [{"fact": "Melanie ran a 5K charity race on 2023-05-20.", '
                    '"entities": ["Melanie", "charity_race"], "resolved_date": "2023-05-20"}]'
                )
                return LLMResponse(content=body, model_used=model, tokens_in=80, tokens_out=40)
            return await super().complete(model, messages, tools=tools, temperature=temperature, **kwargs)

    spy = _FactExtractingMockLLM()
    async with Chatree.in_memory(llm=spy) as ct:
        tree = await ct.create_tree("Atomic Facts Tree")
        root = await tree.root()

        b_mel = await root.spawn("Session 1", title="Melanie Chat", mode="clean")
        await b_mel.chat("[2023-05-25] Melanie: I ran a 5K charity race last Saturday!")
        # Single summary call piggybacks atomic dated fact extraction!
        res = await root.request(b_mel.id, prompt=None)
        assert spy.summary_calls == 1
        assert "FACTS_JSON" not in res["summary"]

        facts_mel = await tree.facts(node_id=b_mel.id)
        assert len(facts_mel) == 1
        assert facts_mel[0].resolved_date == "2023-05-20"
        assert "Melanie" in facts_mel[0].entities

        # Add a dead-end speculative branch with its own atomic fact, then archive it
        b_dead = await root.spawn("Dead End", title="GraphQL Prototype", mode="clean")
        await tree.add_facts(
            b_dead.id,
            [{"fact": "GRAPHQL_FEDERATION_V0 prototype uses 24 hops.", "entities": ["GraphQL"]}],
        )
        # Before archiving, fact is searchable
        hits_before = await tree.search("GRAPHQL_FEDERATION_V0", granularities={"fact"}, top_k=5)
        assert any(h.node_id == b_dead.id for h in hits_before)

        # Archive dead-end branch -> quarantined from active facts & default search!
        await ct.repo.update_node(b_dead.id, status="ARCHIVED")
        assert all(f.node_id != b_dead.id for f in await tree.facts())
        hits_after = await tree.search("GRAPHQL_FEDERATION_V0", granularities={"fact"}, top_k=5)
        assert all(h.node_id != b_dead.id for h in hits_after)

        # Active dated fact in Melanie branch stays searchable & unaffected (append-only across branches!)
        hits_mel = await tree.search("Melanie 2023-05-20 charity race", granularities={"fact"}, top_k=5)
        assert hits_mel and hits_mel[0].node_id == b_mel.id
        assert hits_mel[0].granularity == "fact"


@pytest.mark.asyncio
async def test_pinned_invariants_ephemeral_kv_safe_and_permission_gated() -> None:
    """
    Verify Letta-inspired bounded Tree PINNED Invariants:
      1. Rendered in `<dialogue_tree_context>` (`PINNED [key=value]: ...`) for every child branch.
      2. Zero KV-cache impact: mutating PINNED invariants keeps `build_static_prefix()` byte-identical!
      3. Enforces hard cap (`max_pinned_invariants`) while allowing in-place key updates.
      4. Enforces `read_write` branch permission (blocks `read_only` speculative tangents & shadow turns!).
    """
    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(max_pinned_invariants=2, max_pinned_value_chars=80),
    )
    async with Chatree.in_memory(config=cfg, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Pinned Invariants Test")
        root = await tree.root()
        clean_child = await root.spawn(
            "Auth Design", title="Auth Design", mode="clean", permission="read_only"
        )
        await clean_child.chat("Drafting JWT rotation.")

        static_before = await ct.context.build_static_prefix(await clean_child.model())

        # Root (read_write) pins two invariants via IPC tool
        tm = await tree.model()
        rm = await root.model()
        p1 = await ct.ipc.pin_invariant(tm, rm, key="slo", value="P99_LATENCY_120MS")
        assert p1.tool_output["ok"] is True
        p2 = await ct.ipc.pin_invariant(tm, rm, key="runtime", value="Rust / Axum")
        assert p2.tool_output["ok"] is True

        # 3rd distinct key hits bounded cap (max_pinned_invariants=2), while updating existing key succeeds!
        p3 = await ct.ipc.pin_invariant(tm, rm, key="extra", value="too many")
        assert p3.tool_output["error"] == "INVARIANT_ERROR"
        p_upd = await ct.ipc.pin_invariant(tm, rm, key="slo", value="P99_LATENCY_100MS")
        assert p_upd.tool_output["ok"] is True

        # read_only speculative child is deterministically BLOCKED from mutating global invariants!
        cm = await clean_child.model()
        p_ro = await ct.ipc.pin_invariant(tm, cm, key="graphql", value="GRAPHQL_FEDERATION_V0")
        assert p_ro.tool_output["error"] == "PERMISSION_DENIED"

        # Clean child sees PINNED invariants in ephemeral envelope while static KV prefix stayed byte-identical!
        static_after = await ct.context.build_static_prefix(await clean_child.model())
        assert static_after == static_before
        env, _, _ = await ct.context.build_ephemeral_envelope(tm, cm)
        assert "PINNED [key=value]:" in env
        assert "slo=P99_LATENCY_100MS" in env
        assert "runtime=Rust / Axum" in env

        # Unpin works cleanly via SDK
        assert await tree.unpin("runtime") is True
        assert len(await tree.invariants()) == 1
