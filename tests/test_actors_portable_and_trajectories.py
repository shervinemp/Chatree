from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from chatree import (
    ArtifactInput,
    BenchmarkRunner,
    Chatree,
    ChatreeConfig,
    DialogueDirectoryStore,
    MockLLMProvider,
    StorageBackend,
    TrajectoryExporter,
    default_multi_tangent_scenario,
)
from chatree.cli import run_cli_async
from chatree.mcp_server import ChatreeMCPServer
from chatree.server.app import create_app


@pytest.mark.asyncio
async def test_background_actors_and_unread_mailbox_injection() -> None:
    started_events: list[dict] = []
    completed_events: list[dict] = []

    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        ct.on("background_started")(lambda ev: started_events.append(ev))
        ct.on("background_completed")(lambda ev: completed_events.append(ev))

        tree = await ct.create_tree("Async Research", default_model="mock/chatree-demo")
        root = await tree.root()
        await root.chat("We are designing a low-latency distributed cache.")

        # Spawn two concurrent non-blocking background sub-agents while staying on root
        child1, job1 = await root.spawn_background(
            "Investigate Redis Cluster slot migration edge cases",
            title="Redis Slot Audit",
            mode="clean",
            summary_brief="Focus on MOVED/ASK redirects during resharding.",
        )
        child2, job2 = await root.spawn_background(
            "Evaluate consistent hashing ring replication factor",
            title="Hash Ring Sizing",
            mode="fork",
        )

        # HEAD stays on root so the user can keep chatting immediately
        assert (await tree.head()).id == root.id
        concurrent_res = await root.chat("Keep drafting the main cache interface while agents run.")
        assert concurrent_res.content

        # Wait for both background actors to complete
        jobs = await tree.wait_background(timeout=15.0)
        assert len(jobs) == 2
        assert {j.job_id for j in jobs} == {job1.job_id, job2.job_id}
        assert all(j.status == "completed" for j in jobs)
        assert all(j.turns_executed >= 1 and j.mailbox_id for j in jobs)
        assert len(started_events) == 2
        assert len(completed_events) == 2

        # Both child nodes committed COMPLETED summaries and enqueued unread parent mailbox items
        assert (await child1.model()).status == "COMPLETED"
        assert (await child2.model()).status == "COMPLETED"
        unread = await tree.unread_mailbox(root.id)
        assert len(unread) == 2
        assert {m.from_node_id for m in unread} == {child1.id, child2.id}

        # Next turn on parent automatically injects unread deliveries into ephemeral tail and drains them
        next_turn = await root.chat("Now synthesize the cache design with any incoming findings.")
        notice_blob = "\n".join(next_turn.ephemeral_notices)
        assert "Redis Slot Audit" in notice_blob
        assert "Hash Ring Sizing" in notice_blob
        assert await tree.unread_mailbox(root.id) == []


@pytest.mark.asyncio
async def test_storage_backend_protocol_and_portable_dialogue_roundtrip(tmp_path: Path) -> None:
    export_dir = tmp_path / ".dialogue"

    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        assert isinstance(ct.repo, StorageBackend)

        tree = await ct.create_tree("Portable Auth Tree", default_model="mock/chatree-demo")
        root = await tree.root()
        await root.chat("Let's architect API authentication.")

        child = await root.spawn("Token rotation protocol", title="Token Rotation", mode="fork")
        await child.chat("We settled on Ed25519 15-minute access tokens and Argon2id refresh hashes.")
        await child.finish(
            summary="Ed25519 15m JWT access tokens + Argon2id-hashed opaque refresh tokens.",
            artifacts=[
                ArtifactInput(
                    slug="docs/auth.md",
                    title="Auth Spec v1",
                    kind="markdown",
                    content="# Auth Spec\nEd25519 15m access tokens.",
                    summary="Initial auth spec",
                )
            ],
            auto_commit=True,
        )
        # Bump artifact version with slash in slug to verify filename sanitization on Windows & POSIX
        await tree.add_artifact(
            "docs/auth.md",
            "Auth Spec v2",
            "# Auth Spec v2\nEd25519 15m + Argon2id refresh rotation.",
            created_by_node_id=child.id,
        )
        await root.request(child.id, prompt="Which signature algorithm and TTL did we select?")

        orig_nodes = await tree.nodes()
        orig_reflog = await tree.reflog()
        orig_ipc = await tree.ipc_log()
        orig_arts = await tree.artifacts()

        written = await tree.export_directory(export_dir)
        assert written == export_dir.resolve()
        assert (export_dir / "tree.json").exists()
        assert (export_dir / "reflog.jsonl").exists()
        assert (export_dir / "ipc_events.jsonl").exists()
        assert (export_dir / "dependencies.jsonl").exists()
        assert (export_dir / "nodes" / f"{child.id}.md").exists()
        assert "Argon2id" in (export_dir / "nodes" / f"{child.id}.md").read_text(encoding="utf-8")
        assert (export_dir / "artifacts" / "docs-auth.md@v2.md").exists()

    # Import into a brand-new isolated database and verify lossless restoration + live RAG search
    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct2:
        restored = await ct2.import_directory(export_dir)
        assert restored.id == tree.id
        assert len(await restored.nodes()) == len(orig_nodes)
        assert len(await restored.reflog()) == len(orig_reflog)
        assert len(await restored.ipc_log()) == len(orig_ipc)

        restored_arts = await restored.artifacts()
        assert len(restored_arts) == len(orig_arts) == 2
        assert {a.slug for a in restored_arts} == {"docs-auth.md"}
        assert any("Argon2id" in a.content for a in restored_arts)

        hits = await restored.search("Ed25519 Argon2id refresh", scope="all", top_k=5)
        assert hits
        assert any(h.node_id == child.id for h in hits)

        # Verify overwrite=False guard
        with pytest.raises(ValueError, match="already exists"):
            await DialogueDirectoryStore.import_tree(ct2.repo, export_dir, overwrite=False)


@pytest.mark.asyncio
async def test_trajectory_exporter_sft_and_grpo(tmp_path: Path) -> None:
    async with Chatree.in_memory(llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Trajectory Tree", default_model="mock/chatree-demo")
        root = await tree.root()
        await root.chat("Let's plan our database architecture.")
        # Trigger mock LLM tool call (`spawn_child`) to produce an assistant tool_calls turn
        await root.chat("Spawn child to benchmark Postgres B-tree vs BRIN indexes.")

        all_examples = await tree.export_sft_examples(only_tool_turns=False)
        tool_examples = await tree.export_sft_examples(only_tool_turns=True)
        assert len(all_examples) >= len(tool_examples) >= 1

        for ex in tool_examples:
            assert ex["metadata"]["has_tool_calls"] is True
            assert ex["metadata"]["tool_names"]
            assert ex["messages"][-1]["role"] == "assistant"
            assert ex["messages"][-1].get("tool_calls")
            assert ex["messages"][0]["role"] == "system"
            assert len(ex["tools"]) >= 5

        sft_path = await tree.export_sft_jsonl(tmp_path / "sft.jsonl", only_tool_turns=True)
        lines = [json.loads(ln) for ln in sft_path.read_text(encoding="utf-8").splitlines() if ln]
        assert len(lines) == len(tool_examples)

    # RL/GRPO episode export from benchmark report
    runner = BenchmarkRunner(model="mock/chatree-demo")
    report = await runner.run_suite(default_multi_tangent_scenario())
    episodes = TrajectoryExporter.export_grpo_episodes(report)
    assert len(episodes) == len(report.results)
    by_mode = {ep["mode"]: ep for ep in episodes}
    assert "tree_ipc_full" in by_mode
    assert "tree_summary_only" in by_mode
    assert by_mode["tree_ipc_full"]["reward"] > by_mode["tree_summary_only"]["reward"]

    grpo_path = TrajectoryExporter.save_jsonl(episodes, tmp_path / "grpo.jsonl")
    loaded = [json.loads(ln) for ln in grpo_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(loaded) == len(episodes)


@pytest.mark.asyncio
async def test_cli_mcp_and_server_new_endpoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db1 = tmp_path / "cli1.db"
    db2 = tmp_path / "cli2.db"
    dump_dir = tmp_path / "tree_dump"
    sft_out = tmp_path / "cli_sft.jsonl"
    grpo_out = tmp_path / "cli_grpo.jsonl"

    # 1) CLI: init -> spawn -> chat -> ask -> dump -> load -> trajectories -> eval --grpo-out
    assert await run_cli_async(["--db", str(db1), "init", "CLI Feature Test"]) == 0
    tree_id = json.loads(capsys.readouterr().out)["id"]

    assert (
        await run_cli_async(
            [
                "--db",
                str(db1),
                "spawn",
                tree_id,
                "Choose rate-limiter algorithm",
                "--title",
                "Rate Limiter",
                "--mode",
                "clean",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        await run_cli_async(
            [
                "--db",
                str(db1),
                "chat",
                tree_id,
                "We chose Token Bucket at 250 req/sec per tenant in Redis.",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # Jump back to root and run compound `chatree ask`
    assert await run_cli_async(["--db", str(db1), "jump", tree_id, "back"]) == 0
    capsys.readouterr()

    assert (
        await run_cli_async(["--db", str(db1), "ask", tree_id, "What rate-limiter algorithm and limit?"])
        == 0
    )
    ask_payload = json.loads(capsys.readouterr().out)
    assert len(ask_payload["targets_queried"]) >= 1
    assert "250" in ask_payload["answers"][0]["answer"] or "Token Bucket" in ask_payload["answers"][0]["answer"]

    # Dump & load into a second DB
    assert await run_cli_async(["--db", str(db1), "dump", tree_id, str(dump_dir)]) == 0
    capsys.readouterr()
    assert (dump_dir / "tree.json").exists()

    assert await run_cli_async(["--db", str(db2), "load", str(dump_dir)]) == 0
    loaded_tree = json.loads(capsys.readouterr().out)
    assert loaded_tree["id"] == tree_id

    # Trajectories & eval --grpo-out
    assert (
        await run_cli_async(["--db", str(db1), "trajectories", tree_id, "--out", str(sft_out)]) == 0
    )
    capsys.readouterr()
    assert sft_out.exists() and len(sft_out.read_text(encoding="utf-8").splitlines()) >= 1

    assert (
        await run_cli_async(
            ["--db", str(db1), "eval", "--suite", "default", "--grpo-out", str(grpo_out)]
        )
        == 0
    )
    capsys.readouterr()
    assert grpo_out.exists() and len(grpo_out.read_text(encoding="utf-8").splitlines()) >= 5

    # 2) MCP: verify dialogue_tree_search_and_query tool listing + execution
    async with Chatree.open(db1, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        mcp = ChatreeMCPServer(ct)
        list_resp = await mcp.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert list_resp is not None
        tool_names = [t["name"] for t in list_resp["result"]["tools"]]
        assert "dialogue_tree_search_and_query" in tool_names

        call_resp = await mcp.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "dialogue_tree_search_and_query",
                    "arguments": {
                        "tree_id": tree_id,
                        "query": "Token Bucket rate-limiter",
                        "max_targets": 2,
                    },
                },
            }
        )
        assert call_resp is not None
        mcp_ask = json.loads(call_resp["result"]["content"][0]["text"])
        assert len(mcp_ask["targets_queried"]) >= 1
        assert mcp_ask["answers"]

        # 3) Optional FastAPI server: /ask, /spawn-background, and /jobs
        app = create_app(client=ct)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                tm = await (await ct.get_tree(tree_id)).model()
                assert tm.root_node_id is not None

                r_ask = await client.post(
                    f"/api/nodes/{tm.root_node_id}/ask",
                    json={"query": "Token Bucket per tenant", "scope": "all", "max_targets": 2},
                )
                assert r_ask.status_code == 200
                assert len(r_ask.json()["result"]["targets_queried"]) >= 1

                r_bg = await client.post(
                    f"/api/nodes/{tm.root_node_id}/spawn-background",
                    json={
                        "goal": "Check Redis burst window behavior",
                        "title": "Burst Check",
                        "context_mode": "clean",
                    },
                )
                assert r_bg.status_code == 202
                job_id = r_bg.json()["job"]["job_id"]

                await ct.actors.wait(job_id=job_id, timeout=15.0)

                r_jobs = await client.get(f"/api/trees/{tree_id}/jobs")
                assert r_jobs.status_code == 200
                jobs = r_jobs.json()
                assert any(j["job_id"] == job_id and j["status"] == "completed" for j in jobs)
