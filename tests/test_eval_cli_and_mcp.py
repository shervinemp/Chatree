from __future__ import annotations

import json

from chatree import BenchmarkRunner, Chatree, default_multi_tangent_scenario
from chatree.cli import run_cli_async
from chatree.mcp_server import ChatreeMCPServer


async def test_multi_tangent_ablation_benchmark_runner():
    runner = BenchmarkRunner(model="mock/chatree-demo")
    report = await runner.run_suite(default_multi_tangent_scenario())
    by_mode = {r.mode: r for r in report.results}

    # Full Chatree Tree-IPC passes BOTH fine-grained shadow query + post-reflog revision query!
    assert by_mode["tree_ipc_full"].pass_rate == 1.0
    assert by_mode["tree_ipc_full"].shadow_turns >= 2
    assert by_mode["tree_ipc_full"].stale_detections >= 1

    # Summary-only ablation misses the fine-grained HNSW_EF_64 dropped from lossy summary!
    assert by_mode["tree_summary_only"].pass_rate < 1.0

    # No-staleness ablation misses the revised ED25519_ROTATING_24H conclusion!
    assert by_mode["tree_ipc_no_staleness"].pass_rate < 1.0

    # Tree-IPC keeps peak root context leaner than linear chat!
    assert by_mode["tree_ipc_full"].peak_root_context_tokens < by_mode["linear"].peak_root_context_tokens

    md_table = report.to_markdown_table()
    assert "tree_ipc_full" in md_table
    assert "linear" in md_table


async def test_cli_end_to_end(tmp_path, capsys):
    db_file = str(tmp_path / "cli.db")

    assert await run_cli_async(["--db", db_file, "init", "CLI Tree", "--goal", "Test CLI"]) == 0
    created = json.loads(capsys.readouterr().out)
    tree_id = created["id"]

    assert await run_cli_async(["--db", db_file, "spawn", tree_id, "Explore caching", "--title", "Cache"]) == 0
    child = json.loads(capsys.readouterr().out)
    child_id = child["id"]

    assert await run_cli_async(["--db", db_file, "chat", tree_id, "Use RedisCluster99 for cache."]) == 0
    _ = capsys.readouterr()

    assert await run_cli_async(["--db", db_file, "search", tree_id, "RedisCluster99"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert any(h["node_id"] == child_id for h in hits)

    assert await run_cli_async(
        ["--db", db_file, "request", tree_id, child_id, "--prompt", "Which cache engine?"]
    ) == 0
    req_out = json.loads(capsys.readouterr().out)
    assert "RedisCluster99" in req_out["answer"]

    out_md = tmp_path / "adr.md"
    assert await run_cli_async(["--db", db_file, "export", tree_id, "--out", str(out_md)]) == 0
    _ = capsys.readouterr()
    assert "# CLI Tree" in out_md.read_text(encoding="utf-8")

    assert await run_cli_async(["--db", db_file, "metrics", tree_id]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["node_count"] == 2

    assert await run_cli_async(["eval"]) == 0
    eval_out = capsys.readouterr().out
    assert "tree_ipc_full" in eval_out


async def test_mcp_jsonrpc_server():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("MCP Design Tree")
        root = await tree.root()
        child = await root.spawn("Storage details", title="Storage", mode="clean")
        await child.chat("We store embeddings in SQLite WAL with MCP_VECTOR_TOKEN_42.")
        await child.finish(summary="SQLite WAL embeddings store.", auto_commit=True)

        mcp = ChatreeMCPServer(ct)

        init_resp = await mcp.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert init_resp is not None and init_resp["result"]["serverInfo"]["name"] == "chatree"

        list_resp = await mcp.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert list_resp is not None
        tool_names = {t["name"] for t in list_resp["result"]["tools"]}
        assert {
            "dialogue_tree_list_trees",
            "dialogue_tree_inspect_tree",
            "dialogue_tree_search_branches",
            "dialogue_tree_query_branch",
            "dialogue_tree_search_and_query",
            "dialogue_tree_export_markdown",
        } <= tool_names

        search_resp = await mcp.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "dialogue_tree_search_branches",
                    "arguments": {"tree_id": tree.id, "query": "MCP_VECTOR_TOKEN_42"},
                },
            }
        )
        assert search_resp is not None
        hits = json.loads(search_resp["result"]["content"][0]["text"])
        assert any(h["node_id"] == child.id for h in hits)

        req_resp = await mcp.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "dialogue_tree_query_branch",
                    "arguments": {
                        "tree_id": tree.id,
                        "target_node_id": child.id,
                        "prompt": "What token did we store?",
                    },
                },
            }
        )
        assert req_resp is not None
        ans = json.loads(req_resp["result"]["content"][0]["text"])
        assert "MCP_VECTOR_TOKEN_42" in ans["answer"]
