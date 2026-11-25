from __future__ import annotations

import inspect
import json
import re
from typing import Any

from chatree.config import AblationConfig, GuardrailPolicy
from chatree.engine.context import ContextAssembler
from chatree.engine.guardrails import TraceContext
from chatree.engine.rag import TreeRAGService, extract_key_identifiers
from chatree.protocols import Embedder, LLMProvider, ModelRouter
from chatree.storage.repository import SQLiteRepository, cosine_similarity
from chatree.tools import ToolRegistry
from chatree.types import ChangeSeverity, NodeModel, NodeStatus, TreeModel


_POLARITY_TOKENS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "without",
        "disable",
        "disabled",
        "forbid",
        "forbidden",
        "deny",
        "denied",
        "reject",
        "rejected",
        "drop",
        "dropped",
        "remove",
        "removed",
        "enable",
        "enabled",
        "allow",
        "allowed",
        "require",
        "required",
        "must",
        "stateless",
        "stateful",
        "sync",
        "synchronous",
        "async",
        "asynchronous",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "public",
        "private",
    }
)


def _extract_polarity_tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b[A-Za-z]+\b", text) if w.lower() in _POLARITY_TOKENS}


def parse_summary_and_facts(raw_text: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Gracefully extract `(clean_summary, atomic_facts)` from a piggybacked summary response.
    Supports either:
      1) Prose summary followed by `FACTS_JSON: [...]` (preferred: zero JSON noise in summary!)
      2) Full JSON object `{"summary": "...", "facts": [...]}`
      3) Plain prose fallback (returns `(raw_text.strip(), [])` — never drops a summary on malformed JSON).
    """
    txt = (raw_text or "").strip()
    if not txt:
        return "", []

    # Format 1: Prose + `FACTS_JSON: [...]`
    marker_m = re.search(r"FACTS_JSON\s*:\s*", txt, flags=re.IGNORECASE)
    if marker_m:
        summary_part = txt[: marker_m.start()].strip()
        tail = txt[marker_m.end() :].strip()
        tail = re.sub(r"^```(?:json)?\s*", "", tail, flags=re.IGNORECASE)
        arr_start = tail.find("[")
        if arr_start != -1:
            decoder = json.JSONDecoder()
            try:
                parsed, _ = decoder.raw_decode(tail[arr_start:])
                if isinstance(parsed, list):
                    facts = [f for f in parsed if isinstance(f, dict) and (f.get("fact") or f.get("fact_text"))]
                    return summary_part or txt, facts
            except Exception:
                # Truncation-tolerant fallback: salvage every complete `{...}` fact object before cutoff
                salvaged: list[dict[str, Any]] = []
                idx = arr_start + 1
                n = len(tail)
                while idx < n:
                    brace = tail.find("{", idx)
                    if brace == -1:
                        break
                    try:
                        obj, end_offset = decoder.raw_decode(tail[brace:])
                        if isinstance(obj, dict) and (obj.get("fact") or obj.get("fact_text")):
                            salvaged.append(obj)
                        idx = brace + max(1, end_offset)
                    except Exception:
                        break
                if salvaged:
                    return summary_part or txt, salvaged
        return summary_part or txt, []

    # Format 2: JSON object `{"summary": "...", "facts": [...]}`
    obj_start = txt.find("{")
    if obj_start != -1 and '"summary"' in txt[obj_start : obj_start + 80]:
        try:
            parsed_obj, _ = json.JSONDecoder().raw_decode(txt[obj_start:])
            if isinstance(parsed_obj, dict) and isinstance(parsed_obj.get("summary"), str):
                raw_facts = parsed_obj.get("facts") or []
                facts = [
                    f
                    for f in raw_facts
                    if isinstance(f, dict) and (f.get("fact") or f.get("fact_text"))
                ]
                return parsed_obj["summary"].strip(), facts
        except Exception:
            pass

    return txt, []


class StalenessService:
    """
    Manages versioned node summaries, materiality-gated cosmetic vs. material change
    classification (major v2.0 vs. minor v1.1 with structured 1-line deltas), lazy dirty
    refreshes, and Tier-2/3 RAG propagation.
    """

    def __init__(
        self,
        repo: SQLiteRepository,
        context: ContextAssembler,
        llm: LLMProvider,
        router: ModelRouter,
        rag: TreeRAGService | None = None,
        *,
        embedder: Embedder | None = None,
        guardrails: GuardrailPolicy | None = None,
        ablations: AblationConfig | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.repo = repo
        self.context = context
        self.llm = llm
        self.router = router
        self.rag = rag
        self.embedder = embedder or (rag.embedder if rag is not None else None)
        self.guardrails = guardrails or GuardrailPolicy()
        self.ablations = ablations or AblationConfig()
        self.tool_registry = tool_registry

    async def classify_change_materiality(
        self, old_summary: str | None, new_summary: str
    ) -> tuple[ChangeSeverity, str]:
        """
        Compare `old_summary` vs. `new_summary` to separate cosmetic edits from material
        decision revisions:
          - If no prior summary exists or materiality gating is disabled -> `material`.
          - If key spec identifiers changed (`added` or `removed`) OR cosine similarity drops
            below `guardrails.materiality_cosine_threshold` -> `material` with structured delta.
          - Otherwise -> `cosmetic` (bumps minor version only, avoiding transitive alert fatigue).
        """
        if not old_summary or not old_summary.strip() or not self.ablations.enable_materiality_gating:
            return "material", ""

        old_norm = " ".join(old_summary.strip().split())
        new_norm = " ".join(new_summary.strip().split())
        if old_norm == new_norm:
            return "cosmetic", "Identical summary text"

        old_ids = set(extract_key_identifiers(old_summary, limit=30))
        new_ids = set(extract_key_identifiers(new_summary, limit=30))
        added = sorted(new_ids - old_ids)
        removed = sorted(old_ids - new_ids)

        old_pol = _extract_polarity_tokens(old_summary)
        new_pol = _extract_polarity_tokens(new_summary)
        pol_added = sorted(new_pol - old_pol)
        pol_removed = sorted(old_pol - new_pol)

        sim = 0.0
        if self.embedder is not None:
            vecs = await self.embedder.embed([old_norm, new_norm])
            if len(vecs) == 2:
                sim = cosine_similarity(vecs[0], vecs[1])

        if added or removed or pol_added or pol_removed or sim < self.guardrails.materiality_cosine_threshold:
            parts: list[str] = []
            if removed:
                parts.append(f"removed [{', '.join(removed[:4])}]")
            if added:
                parts.append(f"added [{', '.join(added[:4])}]")
            if pol_removed or pol_added:
                parts.append(
                    f"polarity shifted ({', '.join(pol_removed[:3]) or '∅'} -> {', '.join(pol_added[:3]) or '∅'})"
                )
            if not parts:
                parts.append(f"summary shifted (sim={sim:.2f})")
            return "material", "; ".join(parts)

        return "cosmetic", f"Cosmetic clarification (sim={sim:.2f}, decisions unchanged)"

    async def commit_summary_with_materiality(
        self,
        node: NodeModel,
        new_summary: str,
        *,
        status: NodeStatus | None = None,
    ) -> NodeModel:
        severity, delta = await self.classify_change_materiality(node.summary, new_summary)
        updated = await self.repo.commit_node_summary(
            node.id,
            new_summary.strip() or "(No summary content)",
            status=status,
            severity=severity,
            change_delta=delta,
        )
        if self.rag is not None:
            await self.rag.index_node_summary(updated.id)
            await self.rag.propagate_rollups_upward(updated.id)
        return updated

    async def ensure_fresh_summary(
        self,
        tree: TreeModel,
        node: NodeModel,
        trace_ctx: TraceContext | None = None,
        force: bool = False,
        new_status: NodeStatus | None = None,
    ) -> tuple[NodeModel, bool, int, int, int]:
        """Return `(node, was_regenerated, tokens_in, tokens_out, latency_ms)`."""
        if not force and node.summary is not None and not node.is_dirty:
            if new_status and new_status != node.status:
                updated = await self.repo.update_node(node.id, status=new_status)
                assert updated is not None
                return updated, False, 0, 0, 0
            return node, False, 0, 0, 0

        prefix = await self.context.build_static_prefix(node)
        facts_suffix = (
            '\nOptionally append one final line `FACTS_JSON: [{"fact": "atomic dated fact", '
            '"entities": ["Entity"], "resolved_date": "YYYY-MM-DD"}]` with up to 8 key atomic facts '
            "(resolving any relative dates to absolute calendar dates when session dates are present)."
            if self.ablations.enable_atomic_facts
            else ""
        )
        summarizer_prompt = (
            "[Chatree Summarizer] Produce a concise, high-signal summary of this branch "
            f"('{node.title}', goal: '{node.goal or 'general discussion'}'). "
            "Preserve exact decisions, constraints, file paths, schemas, and open questions. "
            f"Reply in plain text only (do not call any tools).{facts_suffix}"
        )
        messages = [*prefix, {"role": "user", "content": summarizer_prompt}]
        model = self.router.resolve_model(tree, node, "summary")
        branch_chat_model = self.router.resolve_model(tree, node, "chat")

        # Preserve byte-0 provider KV prefix cache when summarizing on the branch's warm model
        # by keeping the canonical tool schemas with `tool_choice="none"` (falling back cleanly
        # to a tool-free system prompt if the provider rejects `tool_choice="none"`).
        summary_tools = (
            self.tool_registry.schemas_for_node(
                node.tool_permission,
                include_core=True,
                include_compound_ask=self.guardrails.enable_compound_ask_tool,
                is_shadow=False,
            )
            if (
                self.tool_registry is not None
                and self.guardrails.preserve_kv_cache_tool_schemas
                and model == branch_chat_model
            )
            else None
        )
        complete_params = inspect.signature(self.llm.complete).parameters
        supports_tool_choice = "tool_choice" in complete_params
        extra_kw: dict[str, Any] = {}
        if self.ablations.enable_atomic_facts and "max_tokens" in complete_params:
            extra_kw["max_tokens"] = 900
        resp = None
        if summary_tools and supports_tool_choice:
            try:
                resp = await self.llm.complete(
                    model=model,
                    messages=messages,
                    tools=summary_tools,
                    temperature=0.1,
                    tool_choice="none",  # type: ignore[call-arg]
                    **extra_kw,
                )
            except Exception:
                resp = None
        if resp is None:
            tool_free_msgs = list(messages)
            if tool_free_msgs and tool_free_msgs[0].get("role") == "system":
                tool_free_msgs[0] = {
                    "role": "system",
                    "content": (
                        "You are a concise technical summarizer for a conversation branch. "
                        "Reply in plain text only; no tools are available."
                    ),
                }
            resp = await self.llm.complete(
                model=model, messages=tool_free_msgs, tools=None, temperature=0.1, **extra_kw
            )

        if trace_ctx is not None:
            trace_ctx.ledger.record(resp.tokens_in, resp.tokens_out)
        await self.repo.increment_node_usage(
            node.id, resp.tokens_in, resp.tokens_out, resp.cache_read_tokens
        )
        clean_summary, extracted_facts = parse_summary_and_facts(resp.content)
        updated = await self.commit_summary_with_materiality(
            node, clean_summary or resp.content, status=new_status
        )
        if extracted_facts and self.ablations.enable_atomic_facts:
            await self.repo.replace_branch_facts(
                tree.id, updated.id, extracted_facts, source_msg_seq=updated.last_msg_seq
            )
            if self.rag is not None:
                await self.rag.index_node_facts(updated.id)
        return updated, True, resp.tokens_in, resp.tokens_out, resp.latency_ms

    async def generate_compaction_handoff(
        self,
        tree: TreeModel,
        node: NodeModel,
        *,
        trace_ctx: TraceContext | None = None,
        reason: str = "",
    ) -> tuple[NodeModel, str, int, int, int]:
        """
        Generate a SOTA structured anchor-preserving compaction handoff on `node`'s warm KV prefix
        (preserving canonical `tools` + `tool_choice="none"` for ~95% KV-cache hit rate), commit it
        as `node`'s Tier-2 summary (marking `node` COMPLETED while leaving its verbatim messages
        100% untouched), and return `(updated_source_node, handoff_text, tokens_in, tokens_out, latency_ms)`.
        """
        prefix = await self.context.build_static_prefix(node)
        prior_anchors = (
            "\nThis branch already continues an earlier compacted session — MERGE and preserve "
            "all still-active locked decisions, identifiers, and ancestor IDs from the prior brief "
            "while updating anything revised in this segment."
            if node.summary_brief and "[Compacted from ancestor" in node.summary_brief
            else ""
        )
        reason_line = f"\nCompaction trigger: {reason}" if reason else ""
        compactor_prompt = (
            "[Chatree Compactor] Produce a dense, structured continuation handoff for a fresh "
            f"child session continuing branch '{node.title}' (`{node.id}`, goal: '{node.goal or 'general'}')."
            f"{prior_anchors}{reason_line}\n"
            "Use these exact 5 compact sections (reply in plain text only, do not call tools):\n"
            "1. OBJECTIVE: Current task & scope.\n"
            "2. LOCKED DECISIONS & VERBATIM ANCHORS: Exact identifiers, config constants, file paths, schemas, numbers, and settled choices (preserve verbatim — never paraphrase identifiers).\n"
            "3. VERIFIED FACTS & ARTIFACTS: Key findings, artifact refs (`slug@v#`), and relevant ancestor branch IDs.\n"
            "4. REJECTED / SUPERSEDED: Explicitly abandoned or retired proposals so they never resurface.\n"
            "5. OPEN NEXT STEPS: Unresolved questions and immediate next action."
        )
        messages = [*prefix, {"role": "user", "content": compactor_prompt}]
        model = self.router.resolve_model(tree, node, "summary")
        branch_chat_model = self.router.resolve_model(tree, node, "chat")

        summary_tools = (
            self.tool_registry.schemas_for_node(
                node.tool_permission,
                include_core=True,
                include_compound_ask=self.guardrails.enable_compound_ask_tool,
                is_shadow=False,
            )
            if (
                self.tool_registry is not None
                and self.guardrails.preserve_kv_cache_tool_schemas
                and model == branch_chat_model
            )
            else None
        )
        supports_tool_choice = "tool_choice" in inspect.signature(self.llm.complete).parameters
        resp = None
        if summary_tools and supports_tool_choice:
            try:
                resp = await self.llm.complete(
                    model=model,
                    messages=messages,
                    tools=summary_tools,
                    temperature=0.1,
                    tool_choice="none",  # type: ignore[call-arg]
                )
            except Exception:
                resp = None
        if resp is None:
            tool_free_msgs = list(messages)
            if tool_free_msgs and tool_free_msgs[0].get("role") == "system":
                tool_free_msgs[0] = {
                    "role": "system",
                    "content": (
                        "You are a precision technical compactor creating a structured continuation "
                        "handoff for a new session. Preserve exact identifiers verbatim; reply in plain text only."
                    ),
                }
            resp = await self.llm.complete(
                model=model, messages=tool_free_msgs, tools=None, temperature=0.1
            )

        if trace_ctx is not None:
            trace_ctx.ledger.record(
                resp.tokens_in,
                resp.tokens_out,
                cache_read_tokens=resp.cache_read_tokens,
                cache_read_weight=self.guardrails.cache_read_budget_weight,
            )
        await self.repo.increment_node_usage(
            node.id, resp.tokens_in, resp.tokens_out, resp.cache_read_tokens
        )
        handoff_text = resp.content.strip() or f"Structured handoff from branch '{node.title}'."
        updated = await self.commit_summary_with_materiality(
            node, handoff_text, status="COMPLETED"
        )
        return updated, handoff_text, resp.tokens_in, resp.tokens_out, resp.latency_ms

    async def get_affected_consumers(self, source_node_id: str) -> list[str]:
        return await self.repo.list_consumers_of_source(
            source_node_id,
            transitive=self.ablations.enable_transitive_staleness,
            max_hops=self.guardrails.max_staleness_hops,
        )
