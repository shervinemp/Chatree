from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Sequence

from chatree.config import ChatreeConfig
from chatree.eval import (
    AblationMatrixRunner,
    BenchmarkRunner,
    ChatreeSystemAdapter,
    ExternalBenchmarkRunner,
    FlatHybridRAGAdapter,
    FullContextLinearAdapter,
    JumpAroundBenchmarkRunner,
    LoCoMoDataset,
    LongMemEvalDataset,
    TrajectoryExporter,
    build_full_eval_suite,
    crossover_scaling_scenario,
    default_multi_tangent_scenario,
    distractor_scaling_scenario,
    multihop_revision_scenario,
    proposal_vs_decision_scenario,
    unscripted_agentic_scenario,
    write_run_manifest,
)
from chatree.providers.embeddings import build_embedder
from chatree.providers.factory import build_llm_provider
from chatree.providers.rerankers import FastEmbedCrossEncoderReranker, LLMReranker
from chatree.sdk.client import Chatree


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatree",
        description="Chatree — Hierarchical LLM Dialogue Tree Framework CLI",
    )
    parser.add_argument(
        "--db",
        default="./data/chatree.db",
        help="Path to local SQLite database (default: ./data/chatree.db)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("trees", help="List all dialogue trees")

    p_init = sub.add_parser("init", help="Create a new dialogue tree")
    p_init.add_argument("title", help="Title for the dialogue tree")
    p_init.add_argument("--goal", default="", help="Root goal")
    p_init.add_argument("--model", default=None, help="Default model ID")

    p_chat = sub.add_parser("chat", help="Send a turn to HEAD (or --node)")
    p_chat.add_argument("tree_id")
    p_chat.add_argument("message")
    p_chat.add_argument("--node", default=None)

    p_spawn = sub.add_parser("spawn", help="Spawn a child branch")
    p_spawn.add_argument("tree_id")
    p_spawn.add_argument("goal")
    p_spawn.add_argument("--title", default=None)
    p_spawn.add_argument("--mode", choices=["fork", "slice", "clean"], default="fork")
    p_spawn.add_argument("--parent", default=None)
    p_spawn.add_argument("--permission", choices=["read_only", "read_write"], default=None)

    p_search = sub.add_parser("search", help="Run 3-tier hybrid search across a tree")
    p_search.add_argument("tree_id")
    p_search.add_argument("query")
    p_search.add_argument(
        "--scope", choices=["all", "ancestors", "subtree", "siblings"], default="all"
    )
    p_search.add_argument("--node", default=None)
    p_search.add_argument("--top-k", type=int, default=5)

    p_req = sub.add_parser("request", help="Query a node (cached summary or read-only shadow turn)")
    p_req.add_argument("tree_id")
    p_req.add_argument("target_node_id")
    p_req.add_argument("--prompt", default=None)
    p_req.add_argument("--caller", default=None)
    p_req.add_argument("--mutate", action="store_true")

    p_ask = sub.add_parser(
        "ask", help="Single-hop compound retrieve-then-ask across relevant branches"
    )
    p_ask.add_argument("tree_id")
    p_ask.add_argument("query")
    p_ask.add_argument(
        "--scope", choices=["all", "ancestors", "subtree", "siblings"], default="all"
    )
    p_ask.add_argument("--max-targets", type=int, default=2)
    p_ask.add_argument("--caller", default=None)
    p_ask.add_argument(
        "--deep",
        action="store_true",
        help="Use TreeQueryEngine multi-hop synthesis (Hop-1 + Tier-2.5 facts + Hop-2 drill-down)",
    )

    p_reflog = sub.add_parser("reflog", help="Show Git-style reflog history")
    p_reflog.add_argument("tree_id")

    p_jump = sub.add_parser("jump", help="Jump HEAD to node ID, HEAD@{n}, back, or forward")
    p_jump.add_argument("tree_id")
    p_jump.add_argument("target")

    p_compact = sub.add_parser(
        "compact",
        help="Non-destructively compact a branch into a linked continuation child session",
    )
    p_compact.add_argument("tree_id")
    p_compact.add_argument("--node", default=None, help="Branch node ID (defaults to HEAD)")
    p_compact.add_argument("--keep-turns", type=int, default=4, help="Recent turns to keep verbatim (0=clean)")
    p_compact.add_argument("--title", default=None, help="Optional continuation title")
    p_compact.add_argument("--reason", default="", help="Optional compaction note")

    p_archive = sub.add_parser(
        "archive",
        help="Archive a non-root dead-end branch to quarantine it from active search",
    )
    p_archive.add_argument("tree_id")
    p_archive.add_argument("--node", default=None, help="Branch node ID to archive (defaults to HEAD)")
    p_archive.add_argument("--reason", default="", help="Reason for archiving")
    p_archive.add_argument("--no-switch", action="store_true", help="Do not switch HEAD back to parent")

    p_pin = sub.add_parser("pin", help="Pin or update a tree-wide invariant")
    p_pin.add_argument("tree_id")
    p_pin.add_argument("key")
    p_pin.add_argument("value")
    p_pin.add_argument("--source-node", default=None)

    p_unpin = sub.add_parser("unpin", help="Unpin a tree-wide invariant by key")
    p_unpin.add_argument("tree_id")
    p_unpin.add_argument("key")

    p_invs = sub.add_parser("invariants", help="List pinned tree-wide invariants")
    p_invs.add_argument("tree_id")

    p_facts = sub.add_parser("facts", help="List Tier-2.5 provenance-linked atomic temporal facts")
    p_facts.add_argument("tree_id")
    p_facts.add_argument("--node", default=None)
    p_facts.add_argument("--include-archived", action="store_true")

    p_export = sub.add_parser("export", help="Compile tree active path & artifacts to Markdown")
    p_export.add_argument("tree_id")
    p_export.add_argument("--leaf", default=None)
    p_export.add_argument("--out", default=None, help="Output file path (prints to stdout if omitted)")

    p_dump = sub.add_parser("dump", help="Export tree into a portable git-greppable .dialogue/ directory")
    p_dump.add_argument("tree_id")
    p_dump.add_argument("out_dir")

    p_load = sub.add_parser("load", help="Import a portable .dialogue/ directory into the database")
    p_load.add_argument("in_dir")
    p_load.add_argument("--no-overwrite", action="store_true")

    p_traj = sub.add_parser(
        "trajectories", help="Export SFT tool-calling training trajectories as JSONL"
    )
    p_traj.add_argument("tree_id")
    p_traj.add_argument("--out", required=True, help="Output JSONL file path")
    p_traj.add_argument("--only-tool-turns", action="store_true")

    p_metrics = sub.add_parser("metrics", help="Print JSON tree metrics & token/cache economics")
    p_metrics.add_argument("tree_id")

    p_eval = sub.add_parser("eval", help="Run Chatree multi-family evaluation, ablations & external benchmarks")
    p_eval.add_argument(
        "--suite",
        choices=["default", "all", "distractor", "proposal", "multihop", "unscripted", "crossover"],
        default="default",
    )
    p_eval.add_argument("--model", default="mock/chatree-demo")
    p_eval.add_argument(
        "--provider",
        choices=["auto", "litellm", "openai_compat", "mock"],
        default="auto",
        help="LLM backend driver (auto uses litellm if installed, else zero-dep openai_compat)",
    )
    p_eval.add_argument("--api-base", default=None, help="Optional OpenAI-compatible API base URL (Ollama, vLLM, SGLang, proxy)")
    p_eval.add_argument("--shadow-model", default="inherit")
    p_eval.add_argument("--judge-model", default=None)
    p_eval.add_argument("--embed-model", default="hash/local-128", help="Embedding model (hash/local-128, local/BAAI/bge-small-en-v1.5, litellm/...)")
    p_eval.add_argument("--reranker", choices=["none", "local", "llm"], default="none")
    p_eval.add_argument("--distractors", type=int, default=4)
    p_eval.add_argument("--oracle-routing", action="store_true")
    p_eval.add_argument("--ablation-matrix", action="store_true", help="Run full systematic ablation matrix with 95%% CIs & McNemar p-values")
    p_eval.add_argument("--crossover-ks", default=None, help="Comma-separated K sweep (e.g. '2,4,8') for scaling crossover curve")
    p_eval.add_argument("--jump-around", action="store_true", help="Run Track 5 live multi-turn jump-around chatbot benchmark")
    p_eval.add_argument("--include-rlm", action="store_true", help="Include MIT rlm_official baseline in --jump-around eval")
    p_eval.add_argument("--locomo", default=None, help="Path to official LoCoMo JSON file (locomo10.json)")
    p_eval.add_argument("--longmemeval", default=None, help="Path to official LongMemEval JSON file (longmemeval_s_cleaned.json)")
    p_eval.add_argument("--max-items", type=int, default=None, help="Cap conversations/items for external benchmarks")
    p_eval.add_argument("--checkpoint", default=None, help="JSONL checkpoint file for crash-safe resume")
    p_eval.add_argument("--concurrency", type=int, default=4, help="Max concurrent conversations for external benchmarks")
    p_eval.add_argument("--dry-run", action="store_true", help="Estimate API calls, tokens, and USD without spending credits")
    p_eval.add_argument("--csv-out", default=None)
    p_eval.add_argument("--jsonl-out", default=None)
    p_eval.add_argument("--latex-out", default=None, help="Export camera-ready LaTeX booktabs table")
    p_eval.add_argument("--manifest-out", default=None, help="Save reproducibility run_manifest.json")
    p_eval.add_argument("--grpo-out", default=None, help="Export RL/GRPO reward episodes JSONL")

    sub.add_parser("mcp", help="Run Chatree MCP server over stdio")

    p_serve = sub.add_parser("serve", help="Start optional FastAPI server (requires chatree[server])")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8001)

    return parser


async def run_cli_async(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval":
        embedder = build_embedder(model=args.embed_model, api_base=args.api_base)
        llm_inst = (
            None
            if args.model.startswith("mock/") and args.provider in ("auto", "mock")
            else build_llm_provider(args.provider, model=args.model, api_base=args.api_base)
        )
        reranker = None
        if args.reranker == "local":
            reranker = FastEmbedCrossEncoderReranker()
        elif args.reranker == "llm":
            from chatree.providers.mock import MockLLMProvider

            reranker = LLMReranker(llm_inst or MockLLMProvider(stream_delay_s=0.0), args.model)

        if args.locomo or args.longmemeval:
            convs = (
                LoCoMoDataset.from_file(args.locomo, max_conversations=args.max_items)
                if args.locomo
                else LongMemEvalDataset.from_file(args.longmemeval, max_items=args.max_items)
            )
            adapters = [
                ChatreeSystemAdapter(
                    model=args.model,
                    shadow_model=args.shadow_model,
                    llm=llm_inst,
                    embedder=embedder,
                    reranker=reranker,
                ),
                FullContextLinearAdapter(model=args.model, llm=llm_inst),
                FlatHybridRAGAdapter(
                    model=args.model, llm=llm_inst, embedder=embedder, reranker=reranker
                ),
            ]
            ext_runner = ExternalBenchmarkRunner(
                model=args.model,
                judge_model=args.judge_model or args.model,
                judge_llm=llm_inst,
                max_concurrency=args.concurrency,
            )
            if args.dry_run:
                est = ext_runner.estimate_cost_dry_run(convs, adapters)
                print(est.to_summary())
                return 0
            ext_rep = await ext_runner.evaluate(
                convs,
                adapters,
                reference_system="chatree_full",
                checkpoint_path=args.checkpoint,
            )
            if args.csv_out:
                ext_rep.save_csv(args.csv_out)
            if args.latex_out:
                Path(args.latex_out).write_text(ext_rep.to_latex_table(), encoding="utf-8")
            if args.manifest_out:
                ext_rep.save_manifest(
                    args.manifest_out,
                    judge_model=args.judge_model or args.model,
                    embed_model=args.embed_model,
                    reranker_name=args.reranker,
                )
            print(ext_rep.to_markdown_table())
            return 0

        if args.jump_around:
            ja_runner = JumpAroundBenchmarkRunner(
                model=args.model,
                shadow_model=args.shadow_model,
                judge_model=args.judge_model,
                llm=llm_inst,
                embedder=embedder,
                reranker=reranker,
            )
            ja_systems = ["chatree_live", "linear", "sliding_window", "flat_rag"]
            if args.include_rlm:
                ja_systems.append("rlm_official")
            ja_rep = await ja_runner.run(systems=ja_systems)  # type: ignore[arg-type]
            if args.csv_out:
                ja_rep.save_csv(args.csv_out)
            if args.jsonl_out:
                ja_rep.save_json(args.jsonl_out)
            if args.latex_out:
                Path(args.latex_out).write_text(ja_rep.to_latex_table(), encoding="utf-8")
            if args.manifest_out:
                write_run_manifest(
                    args.manifest_out,
                    benchmark="jump_around_live",
                    model=args.model,
                    judge_model=args.judge_model or "",
                    embed_model=args.embed_model,
                    reranker_name=args.reranker,
                    total_estimated_cost_usd=round(sum(r.estimated_cost_usd for r in ja_rep.results), 6),
                    systems=[r.system for r in ja_rep.results],
                )
            print(ja_rep.to_markdown_table())
            return 0

        if args.ablation_matrix:
            abl_runner = AblationMatrixRunner(
                model=args.model, shadow_model=args.shadow_model, llm=llm_inst, embedder=embedder
            )
            abl_rep = await abl_runner.run_ablation_matrix()
            if args.csv_out:
                abl_rep.save_csv(args.csv_out)
            if args.latex_out:
                Path(args.latex_out).write_text(abl_rep.to_latex_table(), encoding="utf-8")
            if args.manifest_out:
                write_run_manifest(
                    args.manifest_out,
                    benchmark="ablation_matrix",
                    model=args.model,
                    embed_model=args.embed_model,
                    reranker_name=args.reranker,
                    total_estimated_cost_usd=round(sum(r.estimated_cost_usd for r in abl_rep.rows), 6),
                    systems=[r.ablation_name for r in abl_rep.rows],
                )
            print(abl_rep.to_markdown_table())
            return 0

        if args.crossover_ks:
            ks = [int(x.strip()) for x in args.crossover_ks.split(",") if x.strip()]
            abl_runner = AblationMatrixRunner(
                model=args.model, shadow_model=args.shadow_model, llm=llm_inst, embedder=embedder
            )
            cross_rep = await abl_runner.run_crossover_sweep(ks=ks)
            if args.latex_out:
                Path(args.latex_out).write_text(cross_rep.to_latex_table(), encoding="utf-8")
            print(cross_rep.to_markdown_table())
            return 0

        runner = BenchmarkRunner(
            model=args.model,
            shadow_model=args.shadow_model,
            judge_model=args.judge_model,
            oracle_routing=args.oracle_routing,
            llm=llm_inst,
            embedder=embedder,
        )
        if args.suite == "all":
            suite_report = await runner.run_scenarios(
                build_full_eval_suite(num_distractors=args.distractors)
            )
            if args.csv_out:
                suite_report.save_csv(args.csv_out)
            if args.jsonl_out:
                suite_report.save_jsonl(args.jsonl_out)
            if args.grpo_out:
                TrajectoryExporter.save_jsonl(
                    TrajectoryExporter.export_grpo_episodes(suite_report), args.grpo_out
                )
            print(suite_report.to_markdown_table())
            return 0

        scenario_map = {
            "default": default_multi_tangent_scenario(),
            "distractor": distractor_scaling_scenario(num_distractors=args.distractors),
            "proposal": proposal_vs_decision_scenario(),
            "multihop": multihop_revision_scenario(),
            "unscripted": unscripted_agentic_scenario(),
            "crossover": crossover_scaling_scenario(),
        }
        report = await runner.run_suite(scenario_map[args.suite])
        if args.csv_out:
            report.save_csv(args.csv_out)
        if args.jsonl_out:
            report.save_jsonl(args.jsonl_out)
        if args.grpo_out:
            TrajectoryExporter.save_jsonl(
                TrajectoryExporter.export_grpo_episodes(report), args.grpo_out
            )
        print(report.to_markdown_table())
        return 0

    if args.command == "mcp":
        from chatree.mcp_server import run_stdio_server

        await run_stdio_server(db_path=args.db)
        return 0

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            print("Install server dependencies first: uv sync --extra server", file=sys.stderr)
            return 1
        from chatree.server.app import create_app

        app = create_app(db_path=args.db)
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        return 0

    cfg = ChatreeConfig(db_path=args.db)
    async with Chatree.open(args.db, config=cfg) as ct:
        if args.command == "trees":
            trees = await ct.list_trees()
            print(json.dumps([t.model_dump() for t in trees], indent=2))
            return 0

        if args.command == "init":
            tree = await ct.create_tree(args.title, goal=args.goal, default_model=args.model)
            tm = await tree.model()
            print(json.dumps(tm.model_dump(), indent=2))
            return 0

        if args.command == "load":
            restored = await ct.import_directory(args.in_dir, overwrite=not args.no_overwrite)
            tm = await restored.model()
            print(json.dumps(tm.model_dump(), indent=2))
            return 0

        tree = await ct.get_tree(args.tree_id)

        if args.command == "chat":
            res = await tree.chat(args.message, node_id=args.node)
            print(res.content)
            return 0

        if args.command == "spawn":
            child = await tree.spawn(
                args.goal,
                parent_id=args.parent,
                title=args.title,
                mode=args.mode,
                permission=args.permission,
            )
            cm = await child.model()
            print(json.dumps(cm.model_dump(), indent=2))
            return 0

        if args.command == "search":
            hits = await tree.search(
                args.query, scope=args.scope, node_id=args.node, top_k=args.top_k
            )
            print(json.dumps([h.model_dump() for h in hits], indent=2))
            return 0

        if args.command == "request":
            out = await tree.request(
                args.target_node_id,
                prompt=args.prompt,
                mutate=args.mutate,
                caller_id=args.caller,
            )
            print(json.dumps(out, indent=2))
            return 0

        if args.command == "ask":
            out = await tree.ask(
                args.query,
                scope=args.scope,
                max_targets=args.max_targets,
                caller_id=args.caller,
                deep=args.deep,
            )
            print(json.dumps(out, indent=2))
            return 0

        if args.command == "reflog":
            entries = await tree.reflog()
            for e in entries:
                print(f"{e.head_ref}\t{e.action}\t-> {e.to_node_id}\t{e.note}")
            return 0

        if args.command == "jump":
            dest = await tree.jump(args.target)
            dm = await dest.model()
            print(json.dumps({"head_node_id": dm.id, "title": dm.title}, indent=2))
            return 0

        if args.command == "compact":
            _, info = await tree.compact(
                node_id=args.node,
                keep_recent_turns=args.keep_turns,
                title=args.title,
                reason=args.reason,
            )
            print(json.dumps(info, indent=2))
            return 0

        if args.command == "archive":
            info = await tree.archive(
                node_id=args.node,
                reason=args.reason,
                switch_to_parent=not args.no_switch,
            )
            print(json.dumps(info, indent=2))
            return 0

        if args.command == "pin":
            inv = await tree.pin(args.key, args.value, source_node_id=args.source_node)
            print(json.dumps(inv.model_dump(), indent=2))
            return 0

        if args.command == "unpin":
            removed = await tree.unpin(args.key)
            print(json.dumps({"key": args.key, "removed": removed}, indent=2))
            return 0

        if args.command == "invariants":
            invs = await tree.invariants()
            print(json.dumps([i.model_dump() for i in invs], indent=2))
            return 0

        if args.command == "facts":
            facts = await tree.facts(
                node_id=args.node,
                exclude_archived=not args.include_archived,
            )
            print(json.dumps([f.model_dump() for f in facts], indent=2))
            return 0

        if args.command == "export":
            md = await tree.compile_markdown(leaf_node_id=args.leaf)
            if args.out:
                Path(args.out).write_text(md, encoding="utf-8")
                print(f"Exported Markdown ADR to {args.out}")
            else:
                print(md)
            return 0

        if args.command == "dump":
            out_dir = await tree.export_directory(args.out_dir)
            print(json.dumps({"tree_id": tree.id, "exported_dir": str(out_dir)}, indent=2))
            return 0

        if args.command == "trajectories":
            out_path = await tree.export_sft_jsonl(
                args.out, only_tool_turns=args.only_tool_turns
            )
            print(json.dumps({"tree_id": tree.id, "jsonl_path": str(out_path)}, indent=2))
            return 0

        if args.command == "metrics":
            m = await tree.metrics()
            print(json.dumps(m.model_dump(), indent=2))
            return 0

    return 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(asyncio.run(run_cli_async(argv)))
