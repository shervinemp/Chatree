from __future__ import annotations

import json
from pathlib import Path

from httpx import ASGITransport, AsyncClient
import pytest

from chatree import (
    CallableLLMProvider,
    Chatree,
    ChatreeSystemAdapter,
    HeuristicTokenCounter,
    LLMResponse,
    StorageBackend,
    TokenCounter,
    TreeQueryEngine,
    build_token_counter,
    get_default_token_counter,
    set_default_token_counter,
)
from chatree.cli import run_cli_async
from chatree.mcp_server import ChatreeMCPServer
from chatree.providers.mock import estimate_messages_tokens, estimate_tokens
from chatree.server.app import create_app


async def test_portable_dialogue_roundtrip_preserves_facts_and_invariants(tmp_path: Path):
    """Portable `.dialogue/` v3 export/import must preserve Tier-2.5 facts and PINNED invariants."""
    async with Chatree.in_memory() as ct1:
        tree = await ct1.create_tree("Portable v3 Parity", goal="Verify lossless export/import")
        root = await tree.root()
        await root.chat("Kick off architecture planning.")
        auth = await tree.spawn("Design JWT auth signing", title="Auth Design")
        await auth.chat("Decision: migrate JWT signing to Ed25519 on 2026-03-14.")

        await tree.add_facts(
            auth.id,
            [
                {
                    "fact": "JWT signing migrated to Ed25519 on 2026-03-14",
                    "resolved_date": "2026-03-14",
                    "entities": ["JWT", "Ed25519"],
                }
            ],
            source_msg_seq=2,
        )
        pinned = await tree.pin("jwt_algo", "Ed25519", source_node_id=auth.id)
        assert pinned.source_node_id == auth.id

        export_dir = await ct1.export_directory(tree.id, tmp_path / "portable_tree")
        manifest = json.loads((export_dir / "tree.json").read_text(encoding="utf-8"))
        assert manifest["format_version"] == 3
        assert (export_dir / "facts.jsonl").exists()
        assert (export_dir / "invariants.json").exists()

    async with Chatree.in_memory() as ct2:
        restored = await ct2.import_directory(export_dir)
        restored_facts = await restored.facts()
        assert len(restored_facts) == 1
        assert restored_facts[0].node_id == auth.id
        assert "Ed25519" in restored_facts[0].fact_text
        assert restored_facts[0].resolved_date == "2026-03-14"
        assert restored_facts[0].entities == ["JWT", "Ed25519"]

        restored_invs = await restored.invariants()
        assert len(restored_invs) == 1
        assert restored_invs[0].key == "jwt_algo"
        assert restored_invs[0].value == "Ed25519"
        assert restored_invs[0].source_node_id == auth.id
        assert restored_invs[0].source_summary_version == pinned.source_summary_version
        assert restored_invs[0].source_msg_seq == pinned.source_msg_seq

        fact_hits = await restored.search("Ed25519 JWT signing", granularities={"fact"})
        assert any(h.node_id == auth.id and h.granularity == "fact" for h in fact_hits)


async def test_cli_archive_pin_unpin_invariants_facts_and_deep_ask(tmp_path: Path, capsys):
    """CLI must expose full parity for pin/unpin/invariants/facts/archive and ask --deep."""
    db_file = str(tmp_path / "cli_parity.db")
    async with Chatree.open(db_file) as ct:
        tree = await ct.create_tree("CLI Parity Tree", goal="Test CLI parity commands")
        branch = await tree.spawn("Payment retry policy", title="Payments", permission="read_write")
        await branch.chat("Set payment webhook retries cap to 5 with exponential backoff.")
        await tree.add_facts(
            branch.id,
            [{"fact": "Payment webhook retry cap is 5", "category": "decision", "entities": ["payment"]}],
        )
        tree_id = tree.id
        branch_id = branch.id

    rc = await run_cli_async(
        ["--db", db_file, "pin", tree_id, "retry_cap", "5", "--source-node", branch_id]
    )
    assert rc == 0
    pin_out = json.loads(capsys.readouterr().out)
    assert pin_out["key"] == "retry_cap"
    assert pin_out["value"] == "5"
    assert pin_out["source_node_id"] == branch_id

    rc = await run_cli_async(["--db", db_file, "invariants", tree_id])
    assert rc == 0
    invs_out = json.loads(capsys.readouterr().out)
    assert len(invs_out) == 1 and invs_out[0]["key"] == "retry_cap"

    rc = await run_cli_async(["--db", db_file, "facts", tree_id])
    assert rc == 0
    facts_out = json.loads(capsys.readouterr().out)
    assert len(facts_out) == 1 and "retry cap is 5" in facts_out[0]["fact_text"]

    rc = await run_cli_async(
        ["--db", db_file, "ask", tree_id, "What is the payment retry cap?", "--deep"]
    )
    assert rc == 0
    ask_out = json.loads(capsys.readouterr().out)
    assert ask_out["deep"] is True
    assert ask_out["facts_used"] >= 1
    assert isinstance(ask_out["answer"], str)

    rc = await run_cli_async(
        ["--db", db_file, "archive", tree_id, "--node", branch_id, "--reason", "superseded"]
    )
    assert rc == 0
    arch_out = json.loads(capsys.readouterr().out)
    assert arch_out["ok"] is True
    assert arch_out["archived_node_id"] == branch_id
    assert arch_out["status"] == "ARCHIVED"

    rc = await run_cli_async(["--db", db_file, "unpin", tree_id, "retry_cap"])
    assert rc == 0
    unpin_out = json.loads(capsys.readouterr().out)
    assert unpin_out["removed"] is True


async def test_mcp_server_compact_archive_pin_facts_and_deep_ask():
    """MCP server must expose compact, archive, pin_invariant, list_facts, and deep search_and_query."""
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("MCP Parity Tree")
        b1 = await tree.spawn("Caching layer", title="Caching", permission="read_write")
        await b1.chat("Use Valkey cluster with 60s TTL for session cache.")
        await b1.chat("Confirm memory limit 4GB per shard.")
        await tree.add_facts(
            b1.id,
            [{"fact": "Session cache uses Valkey with 60s TTL", "entities": ["Valkey"]}],
        )

        srv = ChatreeMCPServer(ct)
        tools_resp = await srv.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
        assert {
            "dialogue_tree_compact_branch",
            "dialogue_tree_archive_branch",
            "dialogue_tree_pin_invariant",
            "dialogue_tree_list_facts",
        }.issubset(tool_names)

        async def _call(call_id: int, name: str, arguments: dict) -> dict | list:
            resp = await srv.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                }
            )
            assert resp is not None and "result" in resp
            assert resp["result"]["isError"] is False
            return json.loads(resp["result"]["content"][0]["text"])

        pin_res = await _call(
            2,
            "dialogue_tree_pin_invariant",
            {"tree_id": tree.id, "action": "pin", "key": "cache_engine", "value": "Valkey"},
        )
        assert pin_res["ok"] is True

        inv_list = await _call(
            3, "dialogue_tree_pin_invariant", {"tree_id": tree.id, "action": "list"}
        )
        assert any(i["key"] == "cache_engine" for i in inv_list)

        facts_res = await _call(4, "dialogue_tree_list_facts", {"tree_id": tree.id})
        assert len(facts_res) == 1 and "Valkey" in facts_res[0]["fact_text"]

        deep_res = await _call(
            5,
            "dialogue_tree_search_and_query",
            {"tree_id": tree.id, "query": "What cache engine and TTL?", "deep": True},
        )
        assert deep_res["deep"] is True
        assert deep_res["facts_used"] >= 1

        comp_res = await _call(
            6,
            "dialogue_tree_compact_branch",
            {"tree_id": tree.id, "node_id": b1.id, "keep_recent_turns": 1, "reason": "trim"},
        )
        assert comp_res["compacted"] is True
        cont_id = comp_res["to_node_id"]

        arch_res = await _call(
            7,
            "dialogue_tree_archive_branch",
            {"tree_id": tree.id, "node_id": cont_id, "reason": "done"},
        )
        assert arch_res["status"] == "ARCHIVED"


async def test_server_api_compact_archive_invariants_facts_deep_ask_and_settings(tmp_path: Path):
    """Modular FastAPI routes must support compact, archive, invariants, facts, deep ask & llm_provider."""
    app = create_app(db_path=tmp_path / "server_parity.db")
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            cr = await ac.post("/api/trees", json={"title": "REST Parity"})
            assert cr.status_code == 201
            snap = cr.json()
            tree_id = snap["tree"]["id"]
            root_id = snap["tree"]["root_node_id"]

            sp = await ac.post(
                f"/api/nodes/{root_id}/spawn",
                json={"goal": "Investigate queues", "title": "Queues", "tool_permission": "read_write"},
            )
            assert sp.status_code == 200
            child_id = sp.json()["child"]["id"]

            # Add facts & list facts
            af = await ac.post(
                f"/api/trees/{tree_id}/facts",
                json={
                    "node_id": child_id,
                    "facts": [{"fact": "Queue broker selected: NATS JetStream", "entities": ["NATS"]}],
                },
            )
            assert af.status_code == 201
            assert len(af.json()) == 1

            lf = await ac.get(f"/api/trees/{tree_id}/facts")
            assert lf.status_code == 200
            assert any("NATS JetStream" in f["fact_text"] for f in lf.json())

            # Pin, list, delete invariants
            pi = await ac.post(
                f"/api/trees/{tree_id}/invariants",
                json={"key": "broker", "value": "NATS JetStream", "source_node_id": child_id},
            )
            assert pi.status_code == 201
            assert pi.json()["key"] == "broker"

            li = await ac.get(f"/api/trees/{tree_id}/invariants")
            assert li.status_code == 200 and len(li.json()) == 1

            # Deep ask via unified TreeQueryEngine
            da = await ac.post(
                f"/api/nodes/{root_id}/ask",
                json={"query": "Which queue broker was selected?", "deep": True},
            )
            assert da.status_code == 200
            assert da.json()["result"]["deep"] is True
            assert da.json()["result"]["facts_used"] >= 1

            # Compact & Archive
            cm = await ac.post(
                f"/api/nodes/{child_id}/compact",
                json={"keep_recent_turns": 0, "reason": "rest compact"},
            )
            assert cm.status_code == 200
            assert cm.json()["compacted"] is True
            cont_id = cm.json()["to_node_id"]

            ar = await ac.post(
                f"/api/nodes/{cont_id}/archive",
                json={"reason": "dead end", "switch_to_parent": True},
            )
            assert ar.status_code == 200
            assert ar.json()["status"] == "ARCHIVED"

            # Unpin invariant
            di = await ac.delete(f"/api/trees/{tree_id}/invariants/broker")
            assert di.status_code == 200 and di.json()["removed"] is True

            # Update llm_provider setting and verify persistence
            st = await ac.put("/api/settings", json={"llm_provider": "openai_compat"})
            assert st.status_code == 200
            assert st.json()["llm_provider"] == "openai_compat"
            st_get = await ac.get("/api/settings")
            assert st_get.status_code == 200
            assert st_get.json()["llm_provider"] == "openai_compat"


async def test_vector_matrix_cache_hit_and_write_invalidation():
    """Per-tree NumPy vector matrix cache must hit on repeated searches and invalidate on writes."""
    async with Chatree.in_memory() as ct:
        # Exercise the cached NumPy matrix fallback path deterministically even when sqlite-vec is installed
        ct.db.has_sqlite_vec = False
        tree = await ct.create_tree("Vector Cache Test")
        b1 = await tree.spawn("Database tuning", title="Postgres Tuning")
        await b1.chat("Enable pg_stat_statements and set shared_buffers to 8GB.")

        ct.repo.invalidate_vector_cache()
        s0 = ct.repo.vector_cache_stats()
        assert s0["cached_trees"] == 0

        hits1 = await tree.search("pg_stat_statements shared_buffers")
        assert len(hits1) >= 1
        s1 = ct.repo.vector_cache_stats()
        # First vector query populates the cache (at least 1 miss)
        assert s1["cached_trees"] == 1
        assert s1["misses"] >= 1

        # Second identical search must reuse the cached matrix (hit count increases, misses unchanged)
        hits2 = await tree.search("pg_stat_statements shared_buffers")
        assert len(hits2) >= 1
        s2 = ct.repo.vector_cache_stats()
        assert s2["hits"] > s1["hits"]
        assert s2["misses"] == s1["misses"]

        # Mutating embeddings (adding a brand new fact) must invalidate the cached tree matrix
        await tree.add_facts(
            b1.id,
            [{"fact": "Vacuum cost limit raised to 2000 for night batch", "entities": ["vacuum"]}],
        )
        assert ct.repo.vector_cache_stats()["cached_trees"] == 0

        # Subsequent search repopulates cache and finds the newly indexed fact
        hits3 = await tree.search("vacuum cost limit 2000", granularities={"fact"})
        assert any("2000" in h.snippet for h in hits3)
        s3 = ct.repo.vector_cache_stats()
        assert s3["misses"] > s2["misses"]


async def test_pluggable_token_counter_and_storage_backend_protocol():
    """Pluggable TokenCounter and expanded StorageBackend Protocol must satisfy runtime checks."""

    class DoubleWordCounter:
        name = "double-word"

        def count_text(self, text: str, *, model: str | None = None) -> int:
            return max(1, len((text or "").split()) * 2) if text else 0

        def count_messages(self, messages: list[dict], *, model: str | None = None) -> int:
            return sum(self.count_text(str(m.get("content") or "")) for m in messages)

    heur = HeuristicTokenCounter()
    assert isinstance(heur, TokenCounter)
    assert isinstance(build_token_counter(prefer_tiktoken=False), TokenCounter)
    assert isinstance(build_token_counter(prefer_tiktoken=True), TokenCounter)

    prev = get_default_token_counter()
    custom = DoubleWordCounter()
    assert isinstance(custom, TokenCounter)
    try:
        set_default_token_counter(custom)
        assert estimate_tokens("one two three") == 6
        assert estimate_messages_tokens([{"role": "user", "content": "alpha beta"}]) == 4
    finally:
        set_default_token_counter(prev)

    async with Chatree.in_memory() as ct:
        assert isinstance(ct.repo, StorageBackend)


async def test_unified_tree_query_engine_live_and_adapter_parity():
    """Live `tree.ask(..., deep=True)` and `ChatreeSystemAdapter` must share `TreeQueryEngine.deep_ask`."""
    calls: list[str] = []

    async def fake_llm(model: str, messages: list[dict], tools, temperature: float) -> LLMResponse:
        sys_msg = str(messages[0].get("content") or "") if messages else ""
        user_msg = str(messages[-1].get("content") or "") if messages else ""
        if "NOT_FOUND_IN_BRANCH" in user_msg or "read-only shadow" in user_msg.lower():
            calls.append("shadow")
            if "Redis" in str(messages) or "Valkey" in str(messages):
                return LLMResponse(content="The team switched from Redis to Valkey on May 10, 2026.", tokens_in=40, tokens_out=14)
            return LLMResponse(content="NOT_FOUND_IN_BRANCH", tokens_in=30, tokens_out=4)
        if "second parallel read-only shadow drill-down" in sys_msg.lower():
            calls.append("hop2_plan")
            return LLMResponse(content='{"need_more": false, "target_ids": [], "followup_query": ""}', tokens_in=35, tokens_out=12)
        if "Synthesize one direct, concise, accurate answer" in sys_msg:
            calls.append("synthesis")
            assert "May 10, 2026" in user_msg
            return LLMResponse(content="Valkey (switched from Redis on May 10, 2026).", tokens_in=55, tokens_out=10)
        calls.append("other")
        return LLMResponse(content="Summary: Cache migration to Valkey on May 10, 2026.", tokens_in=30, tokens_out=10)

    provider = CallableLLMProvider(fake_llm)
    async with Chatree.in_memory(llm=provider) as ct:
        assert isinstance(ct.query_engine, TreeQueryEngine)
        tree = await ct.create_tree("Unified Engine", default_model="custom/test-synth")
        root = await tree.root()
        b = await tree.spawn("Cache migration | Date: 2026-05-10", title="Cache Migration")
        await b.chat("We switched from Redis to Valkey on May 10, 2026.")
        await tree.add_facts(
            b.id,
            [{"fact": "Switched from Redis to Valkey", "resolved_date": "2026-05-10", "entities": ["Redis", "Valkey"]}],
        )
        out = await tree.ask(
            "Which cache backend are we using now?", caller_id=root.id, deep=True
        )
        assert out["deep"] is True
        assert out["facts_used"] == 1
        assert "Valkey" in out["answer"]
        assert "shadow" in calls
        assert "synthesis" in calls

    # Verify ChatreeSystemAdapter delegates through the exact same TreeQueryEngine
    adapter = ChatreeSystemAdapter(model="custom/test-synth", llm=provider)
    await adapter.reset("conv_unified_1")
    await adapter.start_session("s1", title="Cache Migration", date_time="2026-05-10")
    await adapter.ingest_turn(
        "s1",
        "user",
        "We switched from Redis to Valkey on May 10, 2026.",
        date_time="2026-05-10",
    )
    await adapter.ingest_turn(
        "s1",
        "assistant",
        "Great, Valkey is now the primary cache.",
        date_time="2026-05-10",
    )
    await adapter.finish_session("s1")
    ans = await adapter.answer_query("Which cache backend is used?")
    await adapter.close()
    assert "Valkey" in ans.answer
    assert "facts_used" in ans.metadata


def test_modular_package_backward_compatible_imports():
    """All pre-refactor import paths must continue resolving cleanly alongside modular subpackages."""
    from chatree.engine.ipc import IPCRouter, IpcExecutionResult
    from chatree.engine.ipc.branching import IpcBranchingMixin
    from chatree.engine.ipc.lifecycle import IpcLifecycleMixin
    from chatree.engine.ipc.shadow import IpcShadowMixin
    from chatree.eval.adapters import (
        ChatreeSystemAdapter as LegacyChatreeAdapter,
        FlatHybridRAGAdapter,
        FullContextLinearAdapter,
        Mem0OfficialAdapter,
        MemorySystemAdapter,
        RLMOfficialAdapter,
    )
    from chatree.server.app import AskTreePayload, TreeSnapshotResponse, create_app as legacy_create_app
    from chatree.storage.repos import SearchVectorsMixin, TreesNodesMessagesMixin
    from chatree.storage.repository import SQLiteRepository, cosine_similarity, new_id

    assert issubclass(IPCRouter, (IpcBranchingMixin, IpcShadowMixin, IpcLifecycleMixin))
    assert issubclass(SQLiteRepository, (TreesNodesMessagesMixin, SearchVectorsMixin))
    assert IpcExecutionResult is not None
    assert LegacyChatreeAdapter is ChatreeSystemAdapter
    assert all(
        cls is not None
        for cls in (
            MemorySystemAdapter,
            FullContextLinearAdapter,
            FlatHybridRAGAdapter,
            Mem0OfficialAdapter,
            RLMOfficialAdapter,
            AskTreePayload,
            TreeSnapshotResponse,
            legacy_create_app,
            cosine_similarity,
            new_id,
        )
    )
