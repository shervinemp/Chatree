from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chatree import (
    AdapterAnswer,
    Chatree,
    ChatreeInteractiveAdapter,
    ExternalBenchmarkRunner,
    ExternalConversation,
    ExternalQAItem,
    ExternalQAResult,
    ExternalSession,
    ExternalTurn,
    HashEmbedder,
    MockLLMProvider,
    cohens_kappa,
    compare_checkpoint_judges,
)
from chatree.cli import run_cli_async
from chatree.mcp_server import ChatreeMCPServer


def test_cohens_kappa_and_checkpoint_judge_agreement(tmp_path: Path) -> None:
    # Perfect agreement
    perf = cohens_kappa([True, False, True, True], [True, False, True, True])
    assert perf.kappa == 1.0
    assert perf.observed_agreement == 1.0
    assert perf.disagreements == 0
    assert perf.interpretation == "almost_perfect"

    # Partial disagreement: 8/10 agree -> po = 0.8, pa1 = 0.6, pb1 = 0.6, pe = 0.52 -> kappa = 0.5833
    a = [True, True, True, True, True, True, False, False, False, False]
    b = [True, True, True, True, True, False, True, False, False, False]
    partial = cohens_kappa(a, b)
    assert partial.n == 10
    assert partial.observed_agreement == 0.8
    assert partial.disagreements == 2
    assert partial.a_only_positive == 1
    assert partial.b_only_positive == 1
    assert abs(partial.kappa - 0.5833) < 1e-3
    assert partial.interpretation == "moderate"

    # Checkpoint-level inter-judge agreement report ($0.00)
    ckpt_a = tmp_path / "judge_a.ckpt.jsonl"
    ckpt_b = tmp_path / "judge_b.ckpt.jsonl"

    def _write_ckpt(path: Path, verdicts: list[tuple[str, str, bool]]) -> None:
        lines = []
        for sys_name, qid, ok in verdicts:
            rec = ExternalQAResult(
                conversation_id="c1",
                question_id=qid,
                category="multi_hop",
                question=f"Question {qid}",
                ground_truth="ans",
                prediction="ans",
                correct=ok,
                f1=1.0 if ok else 0.0,
                bleu1=1.0 if ok else 0.0,
            )
            lines.append(json.dumps({"system_name": sys_name, "result": rec.model_dump()}))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_ckpt(
        ckpt_a,
        [
            ("chatree_full", "q1", True),
            ("chatree_full", "q2", False),
            ("chatree_full", "q3", True),
            ("mem0_official", "q1", True),
            ("mem0_official", "q2", False),
            ("mem0_official", "q3", False),
        ],
    )
    _write_ckpt(
        ckpt_b,
        [
            ("chatree_full", "q1", True),
            ("chatree_full", "q2", False),
            ("chatree_full", "q3", True),
            ("mem0_official", "q1", True),
            ("mem0_official", "q2", True),
            ("mem0_official", "q3", False),
        ],
    )

    rep = compare_checkpoint_judges(ckpt_a, ckpt_b, label_a="deepseek", label_b="gpt4o")
    assert rep.overall.n == 6
    assert rep.overall.disagreements == 1
    assert rep.per_system["chatree_full"].kappa == 1.0
    assert rep.per_system["mem0_official"].disagreements == 1
    md = rep.to_markdown()
    assert "Inter-Judge Agreement Audit" in md
    assert "`chatree_full`" in md


@pytest.mark.asyncio
async def test_external_runner_discounts_metered_ingest_cache_tokens() -> None:
    class _MeteredAdapter:
        def __init__(self, *, name: str, cache_read: int) -> None:
            self._name = name
            self._cache_read = cache_read

        @property
        def name(self) -> str:
            return self._name

        def clone(self) -> _MeteredAdapter:
            return _MeteredAdapter(name=self._name, cache_read=self._cache_read)

        async def reset(self, conversation_id: str) -> None:
            del conversation_id

        async def start_session(
            self, session_id: str, *, title: str = "", date_time: str = "", parent_session_id: str | None = None
        ) -> None:
            del session_id, title, date_time, parent_session_id

        async def ingest_turn(
            self, session_id: str, role: str, content: str, *, date_time: str = ""
        ) -> None:
            del session_id, role, content, date_time

        async def finish_session(self, session_id: str, *, summary: str | None = None) -> None:
            del session_id, summary

        def get_ingest_metering_info(self) -> dict[str, Any]:
            return {
                "tokens_in": 100_000,
                "tokens_out": 2_000,
                "cache_read_tokens": self._cache_read,
                "llm_calls": 4,
                "metered": True,
            }

        async def answer_query(self, question: str, *, date_time: str = "") -> AdapterAnswer:
            del question, date_time
            return AdapterAnswer(answer="PostgreSQL 16.4", tokens_in=500, tokens_out=20)

        async def close(self) -> None:
            return None

    conv = ExternalConversation(
        benchmark="custom",
        conversation_id="c1",
        sessions=[
            ExternalSession(
                session_id="s1",
                title="DB",
                turns=[ExternalTurn(role="user", content="We use PostgreSQL 16.4.")],
            )
        ],
        qa_items=[
            ExternalQAItem(
                question_id="q1",
                category="single_hop",
                question="What DB?",
                ground_truth="PostgreSQL 16.4",
            )
        ],
    )
    runner = ExternalBenchmarkRunner(model="deepseek/deepseek-chat", judge_model="mock/offline")
    rep = await runner.evaluate(
        [conv],
        [
            _MeteredAdapter(name="cached_sys", cache_read=80_000),
            _MeteredAdapter(name="uncached_sys", cache_read=0),
        ],
        reference_system="cached_sys",
    )
    by_name = {r.system_name: r for r in rep.results}
    assert by_name["cached_sys"].ingest_usage_metered is True
    assert by_name["cached_sys"].ingest_cache_read_tokens == 80_000
    assert by_name["cached_sys"].ingest_cost_usd < by_name["uncached_sys"].ingest_cost_usd


@pytest.mark.asyncio
async def test_chatree_interactive_adapter_isolates_queries_offline() -> None:
    adapter = ChatreeInteractiveAdapter(
        name="chatree_interactive_test",
        model="mock/chatree-demo",
        llm=MockLLMProvider(stream_delay_s=0.0),
        embedder=HashEmbedder(),
        generate_session_summaries=False,
    )
    conv = ExternalConversation(
        benchmark="custom",
        conversation_id="conv_live_agent",
        sessions=[
            ExternalSession(
                session_id="s_auth",
                title="Auth Spec",
                date_time="2026-01-10",
                turns=[
                    ExternalTurn(role="user", content="We selected JWT_TTL=10m with ED25519 signing."),
                    ExternalTurn(role="assistant", content="Confirmed JWT_TTL=10m and ED25519."),
                ],
            )
        ],
        qa_items=[
            ExternalQAItem(
                question_id="q1",
                category="single_hop",
                question="What JWT TTL did we choose?",
                ground_truth="10m",
            ),
            ExternalQAItem(
                question_id="q2",
                category="single_hop",
                question="What signing algorithm is active?",
                ground_truth="ED25519",
            ),
        ],
    )
    try:
        await adapter.reset(conv.conversation_id)
        await adapter.ingest_conversation(conv.sessions)
        info = adapter.get_ingest_metering_info()
        assert info["metered"] is True

        a1 = await adapter.answer_query(conv.qa_items[0].question)
        a2 = await adapter.answer_query(conv.qa_items[1].question)
        assert a1.answer and a2.answer
        assert a1.metadata["mode"] == "interactive_orchestrator"
        # Verify ephemeral query branches were archived cleanly after each QA
        assert adapter._tree is not None
        archived_evals = [
            n for n in await adapter._tree.nodes() if n.title == "Eval Query" and n.status == "ARCHIVED"
        ]
        assert len(archived_evals) == 2
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_mcp_and_cli_phase3_branch_operations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 1. MCP server new tools: create_branch (with imports), check_staleness, compare_siblings, promote_branch
    async with Chatree.in_memory(embedder=HashEmbedder()) as ct:
        tree = await ct.create_tree("MCP Phase3 Ops")
        root = await tree.root()
        mcp = ChatreeMCPServer(ct)

        tools_resp = await mcp.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert tools_resp is not None
        tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
        assert {
            "dialogue_tree_create_branch",
            "dialogue_tree_check_staleness",
            "dialogue_tree_compare_siblings",
            "dialogue_tree_promote_branch",
        }.issubset(tool_names)

        async def _call(tool: str, arguments: dict[str, Any]) -> Any:
            resp = await mcp.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": arguments},
                }
            )
            assert resp is not None and "result" in resp
            return json.loads(resp["result"]["content"][0]["text"])

        br_a = await _call(
            "dialogue_tree_create_branch",
            {"tree_id": tree.id, "goal": "Auth Spec", "title": "Auth", "mode": "clean"},
        )
        await ct.repo.commit_node_summary(br_a["id"], "JWT_TTL=60m with RS256", status="COMPLETED")

        br_b = await _call(
            "dialogue_tree_create_branch",
            {
                "tree_id": tree.id,
                "goal": "Edge Cache",
                "title": "Cache",
                "mode": "clean",
                "imports": [br_a["id"]],
            },
        )
        await ct.repo.commit_node_summary(br_b["id"], "CACHE_TTL=30m for Envoy", status="COMPLETED")

        # Revise branch A materially -> check_staleness must report branch B stale
        br_a_model = await ct.repo.get_node(br_a["id"])
        assert br_a_model is not None
        await ct.staleness.commit_summary_with_materiality(
            br_a_model, "JWT_TTL=10m with ED25519", status="COMPLETED"
        )
        stale_res = await _call("dialogue_tree_check_staleness", {"tree_id": tree.id})
        assert stale_res["stale_branch_count"] == 1
        assert stale_res["stale_branches"][0]["node_id"] == br_b["id"]

        # Compare siblings & promote winner
        cmp_res = await _call(
            "dialogue_tree_compare_siblings",
            {"tree_id": tree.id, "parent_id": root.id},
        )
        assert len(cmp_res["siblings"]) == 2

        prom_res = await _call(
            "dialogue_tree_promote_branch",
            {"tree_id": tree.id, "child_id": br_a["id"], "archive_siblings": True},
        )
        assert prom_res["id"] == root.id
        br_b_after = await ct.repo.get_node(br_b["id"])
        assert br_b_after is not None and br_b_after.status == "ARCHIVED"

    # 2. CLI parity: spawn --import, stale, compare, fork-edit, extract, promote
    db_file = str(tmp_path / "cli_phase3.db")
    assert await run_cli_async(["--db", db_file, "init", "CLI Tree"]) == 0
    tree_data = json.loads(capsys.readouterr().out)
    tid = tree_data["id"]
    root_id = tree_data["root_node_id"]

    assert await run_cli_async(["--db", db_file, "chat", tid, "Turn 1 on root"]) == 0
    capsys.readouterr()
    assert await run_cli_async(["--db", db_file, "chat", tid, "Turn 2 tangent on root"]) == 0
    capsys.readouterr()

    assert (
        await run_cli_async(
            ["--db", db_file, "fork-edit", tid, root_id, "--at-seq", "2", "--title", "Forked Edit"]
        )
        == 0
    )
    forked = json.loads(capsys.readouterr().out)
    assert forked["context_mode"] == "fork"

    assert (
        await run_cli_async(
            [
                "--db",
                db_file,
                "extract",
                tid,
                root_id,
                "--start",
                "1",
                "--end",
                "2",
                "--title",
                "Extracted Tangent",
            ]
        )
        == 0
    )
    ext = json.loads(capsys.readouterr().out)
    assert ext["child"]["title"] == "Extracted Tangent"

    assert (
        await run_cli_async(
            [
                "--db",
                db_file,
                "spawn",
                tid,
                "Downstream Cache",
                "--parent",
                root_id,
                "--mode",
                "clean",
                "--import",
                ext["child"]["id"],
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert await run_cli_async(["--db", db_file, "stale", tid]) == 0
    stale_cli = json.loads(capsys.readouterr().out)
    assert stale_cli["tree_id"] == tid

    assert await run_cli_async(["--db", db_file, "compare", tid, root_id]) == 0
    cmp_cli = json.loads(capsys.readouterr().out)
    assert len(cmp_cli["siblings"]) >= 2

    assert await run_cli_async(["--db", db_file, "promote", tid, ext["child"]["id"]]) == 0
    prom_cli = json.loads(capsys.readouterr().out)
    assert prom_cli["id"] == root_id
