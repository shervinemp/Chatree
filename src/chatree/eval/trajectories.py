from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chatree.engine.context import ContextAssembler
from chatree.storage.repository import SQLiteRepository
from chatree.tools import COMPOUND_ASK_TOOL_DEFINITION, CORE_TOOL_DEFINITIONS, ToolRegistry

if TYPE_CHECKING:
    from chatree.eval.runner import BenchmarkReport, ModeBenchmarkResult, SuiteBenchmarkReport


class TrajectoryExporter:
    """
    Exports Chatree dialogue trees and benchmark runs into fine-tuning datasets for
    post-training smaller/open models (SFT & RL/GRPO, e.g. FoldGRPO-style reward shaping)
    to learn *when* to call `dialogue_tree_create_branch`, `dialogue_tree_search_branches`,
    `dialogue_tree_query_branch`, `dialogue_tree_search_and_query`, and `dialogue_tree_send_to_branch`.
    """

    @staticmethod
    async def export_sft_examples(
        repo: SQLiteRepository,
        context: ContextAssembler,
        tree_id: str,
        *,
        only_tool_turns: bool = False,
        include_context_envelope: bool = True,
        tools_registry: ToolRegistry | None = None,
    ) -> list[dict[str, Any]]:
        """
        Emit OpenAI/TRL-compatible SFT training examples (`messages`, `tools`, `metadata`)
        reconstructing both the static prefix AND the `<dialogue_tree_context>` ephemeral envelope
        on the preceding user message to prevent train/serve distribution skew.
        """
        tree = await repo.get_tree(tree_id)
        if tree is None:
            raise ValueError(f"Tree not found: {tree_id}")
        nodes = await repo.list_nodes(tree_id)
        examples: list[dict[str, Any]] = []
        reg = tools_registry or ToolRegistry()

        for node in nodes:
            msgs = await repo.list_messages(node.id)
            envelope = ""
            if include_context_envelope:
                envelope, _, _ = await context.build_ephemeral_envelope(tree, node)
            node_tools = (
                reg.schemas_for_node(
                    node.tool_permission, include_core=True, include_compound_ask=True
                )
                if tools_registry is not None
                else [*CORE_TOOL_DEFINITIONS, COMPOUND_ASK_TOOL_DEFINITION]
            )
            for m in msgs:
                if m.role != "assistant":
                    continue
                has_tools = bool(m.tool_calls)
                if only_tool_turns and not has_tools:
                    continue
                prefix = await context.build_static_prefix(node, exclude_last_seq=m.seq)
                # Inject <dialogue_tree_context> onto the latest user message preceding this assistant turn
                if include_context_envelope and envelope:
                    for idx in range(len(prefix) - 1, 0, -1):
                        if prefix[idx].get("role") == "user":
                            u_content = str(prefix[idx].get("content") or "")
                            if "<dialogue_tree_context>" not in u_content:
                                prefix[idx] = {
                                    **prefix[idx],
                                    "content": f"{envelope}\n\n{u_content}",
                                }
                            break
                target_msg: dict[str, Any] = {"role": "assistant", "content": m.content}
                if m.tool_calls:
                    target_msg["tool_calls"] = m.tool_calls
                examples.append(
                    {
                        "messages": [*prefix, target_msg],
                        "tools": node_tools,
                        "metadata": {
                            "tree_id": tree.id,
                            "node_id": node.id,
                            "node_title": node.title,
                            "context_mode": node.context_mode,
                            "tool_permission": node.tool_permission,
                            "seq": m.seq,
                            "has_tool_calls": has_tools,
                            "include_context_envelope": include_context_envelope,
                            "tool_names": [
                                tc["function"]["name"]
                                for tc in (m.tool_calls or [])
                                if isinstance(tc, dict) and "function" in tc
                            ],
                            "model_used": m.model_used,
                        },
                    }
                )
        return examples

    @staticmethod
    def export_grpo_episodes(
        report: BenchmarkReport | SuiteBenchmarkReport,
        *,
        leakage_penalty: float = 0.5,
        stale_penalty: float = 0.6,
        token_cost_weight: float = 0.05,
    ) -> list[dict[str, Any]]:
        """
        Convert benchmark evaluation reports into RL / GRPO training episodes with composite
        scalar rewards balancing task accuracy against rejected-proposal leakage,
        stale-decision contradictions, and token cost.
        """
        all_results: list[ModeBenchmarkResult] = []
        if hasattr(report, "reports"):
            for rep in report.reports:  # type: ignore[union-attr]
                all_results.extend(rep.results)
        else:
            all_results.extend(report.results)  # type: ignore[union-attr]

        episodes: list[dict[str, Any]] = []
        for r in all_results:
            tok_norm = (r.total_tokens_in + r.total_tokens_out) / 10_000.0
            reward = round(
                r.pass_rate
                - leakage_penalty * r.rejected_leakage_rate
                - stale_penalty * r.stale_contradiction_rate
                - token_cost_weight * tok_norm,
                4,
            )
            rollouts = [
                {
                    "prompt": [{"role": "user", "content": q.query}],
                    "completion": [{"role": "assistant", "content": q.answer_preview}],
                    "reward": round(
                        (1.0 if q.passed else 0.0)
                        - (leakage_penalty if q.rejected_leakage else 0.0)
                        - (stale_penalty if q.stale_contradiction else 0.0),
                        4,
                    ),
                    "target_branch": q.target_branch,
                    "passed": q.passed,
                }
                for q in r.query_results
            ]
            episodes.append(
                {
                    "scenario": r.scenario_name,
                    "family": r.family,
                    "mode": r.mode,
                    "reward": reward,
                    "rollouts": rollouts,
                    "components": {
                        "strict_pass_rate": r.pass_rate,
                        "recall_rate": r.recall_rate,
                        "rejected_leakage_rate": r.rejected_leakage_rate,
                        "stale_contradiction_rate": r.stale_contradiction_rate,
                        "peak_root_context_tokens": r.peak_root_context_tokens,
                        "total_tokens": r.total_tokens_in + r.total_tokens_out,
                        "cache_hit_ratio": r.cache_hit_ratio,
                        "shadow_turns": r.shadow_turns,
                        "guardrail_blocks": r.guardrail_blocks,
                    },
                    "queries": [q.model_dump() for q in r.query_results],
                }
            )
        return episodes

    @staticmethod
    def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "\n".join(json.dumps(rec) for rec in records) + ("\n" if records else ""),
            encoding="utf-8",
        )
        return p
