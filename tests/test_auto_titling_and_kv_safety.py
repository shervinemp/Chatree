from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chatree import (
    Chatree,
    ChatreeConfig,
    DialogueDirectoryStore,
    LLMResponse,
    MockLLMProvider,
)
from chatree.engine.staleness import parse_title_summary_and_facts
from chatree.types import is_placeholder_title, sanitize_branch_title


def test_placeholder_detection_sanitization_and_parser() -> None:
    """Placeholder detector, title sanitizer, and TITLE+SUMMARY+FACTS parser must be strict and clean."""
    for ph in ("", "New Chat", "Untitled", "Untitled Chat", "Child Branch", "Session 14", "Session 1 (2023/05/08)", "Branch 3"):
        assert is_placeholder_title(ph), f"Expected placeholder: {ph!r}"
    for real in ("Nebula Auth v2", "Valkey TLS Pool Exhaustion", "Q4 Budget Review"):
        assert not is_placeholder_title(real), f"Should not be placeholder: {real!r}"

    assert sanitize_branch_title('Title: "Discussion about Nebula Auth v2 — Ed25519 Migration."') == (
        "Nebula Auth v2 — Ed25519 Migration"
    )
    clamped = sanitize_branch_title(
        "One Two Three Four Five Six Seven Eight Nine Ten Eleven", max_words=8
    )
    assert len(clamped.split()) <= 8

    # Prose + TITLE + FACTS_JSON
    raw = (
        "TITLE: Nebula Auth v2 — Ed25519 Migration\n"
        "SUMMARY: Migrated JWT signing from RS256 to ED25519_SIGNING with 5m TTL.\n"
        'FACTS_JSON: [{"fact": "Auth uses ED25519_SIGNING", "entities": ["Nebula"], "resolved_date": "2026-03-27"}]'
    )
    title, summary, facts = parse_title_summary_and_facts(raw)
    assert title == "Nebula Auth v2 — Ed25519 Migration"
    assert "ED25519_SIGNING" in summary and "TITLE:" not in summary and "FACTS_JSON" not in summary
    assert len(facts) == 1 and facts[0]["resolved_date"] == "2026-03-27"

    # JSON object format
    raw_json = '{"title": "Valkey 8.0 Pool Fix", "summary": "Raised max idle to 64.", "facts": []}'
    jt, js, jf = parse_title_summary_and_facts(raw_json)
    assert jt == "Valkey 8.0 Pool Fix"
    assert js == "Raised max idle to 64."
    assert jf == []


@pytest.mark.asyncio
async def test_turn1_auto_title_updates_root_and_rag_without_blocking() -> None:
    """Placeholder root titled 'New Chat' should auto-title after Turn 1 and index into RAG."""
    titled_events: list[dict[str, Any]] = []
    cfg = ChatreeConfig(db_path=":memory:", default_model="mock/chatree-demo")
    async with Chatree.in_memory(config=cfg) as ct:
        ct.hooks.register("node_titled", lambda payload: titled_events.append(payload))
        tree = await ct.create_tree("New Chat")
        root = await tree.root()
        rm = await root.model()
        assert rm.title_source == "placeholder"
        assert rm.brief_title == "New Chat"

        await root.chat("Investigate VALKEY_POOL_6380 TLS handshake exhaustion in Nebula Gateway.")
        await ct.wait_background_summaries()

        updated_root = await root.model()
        updated_tree = await ct.repo.get_tree(tree.id)
        assert updated_tree is not None
        assert updated_root.title_source == "auto"
        assert updated_root.title != "New Chat"
        assert "VALKEY_POOL_6380" in updated_root.title
        assert updated_tree.title == updated_root.title
        assert len(titled_events) == 1 and titled_events[0]["title"] == updated_root.title

        # Both Tier-1 chunk (with semantic [Path: ...]) and Tier-2 summary must be searchable
        hits = await tree.search("VALKEY_POOL_6380", top_k=6)
        grans = {h.granularity for h in hits if h.node_id == root.id}
        assert "chunk" in grans and "summary" in grans


@pytest.mark.asyncio
async def test_re_titling_preserves_byte_identical_static_prefix_kv_cache() -> None:
    """Mutating node.title on clean/slice branches must NEVER alter build_static_prefix()."""
    cfg = ChatreeConfig(db_path=":memory:", default_model="mock/chatree-demo")
    async with Chatree.in_memory(config=cfg) as ct:
        tree = await ct.create_tree("Nebula Platform")
        root = await tree.root()
        await root.chat("Root context established for Nebula Platform.")

        clean_child = await root.spawn(
            "Investigate incident",
            title="Session 7",
            mode="clean",
            summary_brief="Initial brief for incident triage.",
        )
        slice_child = await root.spawn(
            "Investigate cache",
            title="Session 8",
            mode="slice",
            summary_brief="Initial slice brief.",
        )

        clean_m0 = await clean_child.model()
        slice_m0 = await slice_child.model()
        assert clean_m0.title_source == "placeholder" and clean_m0.brief_title == "Session 7"
        assert slice_m0.title_source == "placeholder" and slice_m0.brief_title == "Session 8"

        clean_prefix_before = await ct.context.build_static_prefix(clean_m0)
        slice_prefix_before = await ct.context.build_static_prefix(slice_m0)

        await clean_child.chat("Root cause is DISKANN_PQ64_OVERFLOW in NVMe rescoring queue.")
        await slice_child.chat("Cache miss storm traced to VALKEY_EVICT_9001.")
        await ct.wait_background_summaries()

        clean_m1 = await clean_child.model()
        slice_m1 = await slice_child.model()
        assert clean_m1.title != "Session 7" and clean_m1.title_source == "auto"
        assert slice_m1.title != "Session 8" and slice_m1.title_source == "auto"
        assert clean_m1.brief_title == "Session 7"
        assert slice_m1.brief_title == "Session 8"

        # Static prefix opening brief messages MUST stay byte-identical (0% KV-cache invalidation!)
        clean_prefix_after = await ct.context.build_static_prefix(clean_m1)
        slice_prefix_after = await ct.context.build_static_prefix(slice_m1)
        assert clean_prefix_after[: len(clean_prefix_before)] == clean_prefix_before
        assert slice_prefix_after[: len(slice_prefix_before)] == slice_prefix_before

        # Meanwhile the ephemeral tail envelope immediately surfaces the new semantic titles
        tree_m = await ct.repo.get_tree(tree.id)
        root_m = await root.model()
        assert tree_m is not None
        env, _, _ = await ct.context.build_ephemeral_envelope(tree_m, root_m)
        assert clean_m1.title in env and slice_m1.title in env


@pytest.mark.asyncio
async def test_user_sovereignty_exit_refinement_and_portable_roundtrip(tmp_path: Path) -> None:
    """
    1. User-chosen titles (`title_source=='user'`) are NEVER overwritten on Turn 1 or exit summary.
    2. Auto-titled branches refine on material exit drift (piggybacked TITLE:, 0 extra LLM calls),
       not on cosmetic edits.
    3. Portable `.dialogue/` export/import preserves `title_source` and frozen `brief_title`.
    """

    class ControlledSummaryLLM(MockLLMProvider):
        def __init__(self) -> None:
            super().__init__(stream_delay_s=0.0)
            self.summary_calls = 0
            self.next_summary_payload = ""

        async def complete(
            self,
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.2,
            **kwargs: Any,
        ) -> LLMResponse:
            last = str(messages[-1].get("content") or "") if messages else ""
            if "[Chatree Summarizer]" in last and self.next_summary_payload:
                self.summary_calls += 1
                return LLMResponse(
                    content=self.next_summary_payload,
                    model_used=model,
                    tokens_in=120,
                    tokens_out=40,
                )
            return await super().complete(
                model=model, messages=messages, tools=tools, temperature=temperature, **kwargs
            )

    llm = ControlledSummaryLLM()
    cfg = ChatreeConfig(db_path=":memory:", default_model="mock/chatree-demo")
    async with Chatree.in_memory(config=cfg, llm=llm) as ct:
        tree = await ct.create_tree("My Custom Architecture")
        root = await tree.root()
        assert (await root.model()).title_source == "user"

        # 1. User-titled branch stays untouched across Turn 1 and summary refresh
        user_branch = await root.spawn("Auth work", title="My Custom Auth Branch", mode="clean")
        await user_branch.chat("Switched to ED25519_SIGNING and JWT_TTL_5M.")
        await ct.wait_background_summaries()
        llm.next_summary_payload = (
            "TITLE: Attempted Overwrite Title\n"
            "SUMMARY: Auth uses ED25519_SIGNING with JWT_TTL_5M."
        )
        t_model = await ct.repo.get_tree(tree.id)
        assert t_model is not None
        ub_m, *_ = await ct.staleness.ensure_fresh_summary(t_model, await user_branch.model(), force=True)
        assert ub_m.title == "My Custom Auth Branch"
        assert ub_m.title_source == "user"

        # 2. Placeholder branch upgrades on Turn 1, ignores cosmetic drift on exit, refines on material drift
        auto_branch = await root.spawn("General triage", title="Session 12", mode="clean")
        await auto_branch.chat("Initial RS256_2048_RSA token latency check.")
        await ct.wait_background_summaries()
        ab_after_t1 = await auto_branch.model()
        assert ab_after_t1.title_source == "auto"

        # First summary establishes v1.0 baseline (piggybacked TITLE:, 0 extra calls)
        baseline_summary = (
            "Audited RS256_2048_RSA token verification latency across regional gateway workers with "
            "15m JWT lifetime, standard JWKS caching headers, and baseline throughput profiling."
        )
        llm.next_summary_payload = (
            f"TITLE: RS256 Token Latency Audit\nSUMMARY: {baseline_summary}"
        )
        ab_v1, *_ = await ct.staleness.ensure_fresh_summary(t_model, ab_after_t1, force=True)
        assert ab_v1.title == "RS256 Token Latency Audit"
        calls_after_v1 = llm.summary_calls

        # Cosmetic tweak: identical decisions/identifiers -> summary_minor_version bumps, title MUST NOT flicker
        await ct.repo.append_message(
            node_id=auto_branch.id, tree_id=tree.id, role="user", content="minor formatting note", mark_dirty=True
        )
        llm.next_summary_payload = (
            f"TITLE: Flickering Cosmetic Title\nSUMMARY: {baseline_summary} Confirmed."
        )
        ab_cosmetic, *_ = await ct.staleness.ensure_fresh_summary(t_model, await auto_branch.model(), force=True)
        assert ab_cosmetic.last_change_severity == "cosmetic"
        assert ab_cosmetic.title == "RS256 Token Latency Audit"

        # Material scope shift: new identifiers -> major v2.0 + piggybacked title refines cleanly
        await ct.repo.append_message(
            node_id=auto_branch.id,
            tree_id=tree.id,
            role="user",
            content="Reverse RS256_2048_RSA completely: adopt ED25519_SIGNING and BLAKE3_REFRESH_HASH.",
            mark_dirty=True,
        )
        llm.next_summary_payload = (
            "TITLE: Ed25519 Signing & Blake3 Migration\n"
            "SUMMARY: Replaced RS256 with ED25519_SIGNING and BLAKE3_REFRESH_HASH."
        )
        ab_mat, *_ = await ct.staleness.ensure_fresh_summary(t_model, await auto_branch.model(), force=True)
        assert ab_mat.last_change_severity == "material"
        assert ab_mat.title == "Ed25519 Signing & Blake3 Migration"
        assert ab_mat.brief_title == "Session 12"
        assert llm.summary_calls == calls_after_v1 + 2  # Exactly 1 call per summary, 0 extra title calls!

        # 3. Portable .dialogue/ round-trip preserves title_source and brief_title
        export_dir = tmp_path / "portable_titles"
        await DialogueDirectoryStore.export_tree(ct.repo, tree.id, export_dir)

    async with Chatree.in_memory(config=cfg) as ct2:
        restored = await DialogueDirectoryStore.import_tree(ct2.repo, export_dir, rag=ct2.rag)
        nodes = {n.id: n for n in await ct2.repo.list_nodes(restored.id)}
        assert nodes[auto_branch.id].title == "Ed25519 Signing & Blake3 Migration"
        assert nodes[auto_branch.id].title_source == "auto"
        assert nodes[auto_branch.id].brief_title == "Session 12"
        assert nodes[user_branch.id].title_source == "user"
