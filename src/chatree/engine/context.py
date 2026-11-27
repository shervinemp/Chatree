from __future__ import annotations

import json
import re
from typing import Any

from chatree.config import AblationConfig, GuardrailPolicy, resolve_model_context_window
from chatree.providers.mock import estimate_messages_tokens, estimate_tokens
from chatree.storage.repository import SQLiteRepository
from chatree.tools import STATIC_SYSTEM_PROMPT
from chatree.types import (
    DEFAULT_SESSION_ID,
    MailboxItemModel,
    MessageModel,
    NodeModel,
    SearchHit,
    TreeModel,
)


def _escape_envelope_text(text: str) -> str:
    """Prevent untrusted child/mailbox content from breaking out of <dialogue_tree_context> XML tags."""
    if not text:
        return ""
    return (
        text.replace("</dialogue_tree_context>", "&lt;/dialogue_tree_context&gt;")
        .replace("<dialogue_tree_context>", "&lt;dialogue_tree_context&gt;")
        .replace("</chatree_context>", "&lt;/chatree_context&gt;")
        .replace("<chatree_context>", "&lt;chatree_context&gt;")
    )


def _message_to_openai(msg: MessageModel) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.role == "assistant" and msg.tool_calls:
        out["tool_calls"] = msg.tool_calls
    if msg.role == "tool":
        if msg.tool_call_id:
            out["tool_call_id"] = msg.tool_call_id
        if msg.tool_name:
            out["name"] = msg.tool_name
    return out


def _strip_dangling_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    open_ids: set[str] = set()
    last_valid_len = 0
    for idx, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id"):
                    open_ids.add(str(tc["id"]))
        elif role == "tool" and m.get("tool_call_id"):
            open_ids.discard(str(m["tool_call_id"]))
        if not open_ids:
            last_valid_len = idx + 1
    return messages[:last_valid_len]


class ContextAssembler:
    """
    Constructs prefix-cache-safe prompts by strictly separating:
      1) Static Cacheable Prefix (system prompt + copy-on-write fork chain + committed messages)
      2) Ephemeral Tail Envelope (dynamic orientation, reflog cues, staleness alerts — never persisted)
    """

    def __init__(
        self,
        repo: SQLiteRepository,
        *,
        system_prompt: str = STATIC_SYSTEM_PROMPT,
        ablations: AblationConfig | None = None,
        guardrails: GuardrailPolicy | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        self.repo = repo
        self.system_prompt = system_prompt
        self.ablations = ablations or AblationConfig()
        self.guardrails = guardrails or GuardrailPolicy()
        self.max_context_tokens = (
            max_context_tokens
            if max_context_tokens is not None
            else (guardrails.max_branch_context_tokens if guardrails is not None else None)
        )
        # Per-(node_id, budget) hysteresis drop watermark so block-aligned trims keep the
        # trimmed prefix byte-for-byte identical across subsequent turns until the next block!
        self._trim_cutoffs: dict[tuple[str, int], int] = {}

    async def collect_inherited_messages(self, node: NodeModel) -> list[MessageModel]:
        if node.context_mode in ("root", "clean") or not node.parent_id:
            return []

        if node.context_mode == "fork":
            chain: list[tuple[NodeModel, int]] = []
            curr = node
            while curr.context_mode == "fork" and curr.parent_id:
                parent = await self.repo.get_node(curr.parent_id)
                if parent is None:
                    break
                chain.append((parent, curr.spawn_msg_seq))
                curr = parent
            chain.reverse()

            inherited: list[MessageModel] = []
            for anc_node, cutoff_seq in chain:
                if cutoff_seq > 0:
                    inherited.extend(
                        await self.repo.list_messages(anc_node.id, up_to_seq=cutoff_seq)
                    )
            return inherited

        if node.context_mode == "slice":
            parent = await self.repo.get_node(node.parent_id)
            if parent is None or node.spawn_msg_seq <= 0:
                return []
            parent_msgs = await self.repo.list_messages(parent.id, up_to_seq=node.spawn_msg_seq)
            return parent_msgs[-node.slice_turn_count :]

        return []

    async def _build_raw_static_prefix(
        self,
        node: NodeModel,
        exclude_last_seq: int | None = None,
    ) -> tuple[list[dict[str, Any]], int, list[MessageModel]]:
        """Assemble the untrimmed static prefix and return `(payload, pinned_head_len, own_msgs)`."""
        payload: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        head_len = 1

        inherited = await self.collect_inherited_messages(node)
        frozen_title = node.brief_title or node.title
        if node.context_mode == "fork":
            if inherited:
                payload.extend(
                    _strip_dangling_tool_calls([_message_to_openai(m) for m in inherited])
                )
        elif node.context_mode == "slice":
            brief_lines = [f"[Branch Slice Brief] Goal: {node.goal or frozen_title}"]
            if node.summary_brief:
                brief_lines.append(f"Brief: {node.summary_brief}")
            if inherited:
                brief_lines.append("\nRecent parent turns:")
                for m in inherited:
                    if m.role in ("user", "assistant") and m.content.strip():
                        brief_lines.append(f"- {m.role}: {m.content}")
            payload.append({"role": "user", "content": "\n".join(brief_lines)})
            payload.append(
                {"role": "assistant", "content": f"Understood. Ready to work on branch '{frozen_title}'."}
            )
            head_len = 3
        elif node.context_mode == "clean":
            if node.goal or node.summary_brief:
                brief_lines = [f"[Clean Branch Brief] Goal: {node.goal or frozen_title}"]
                if node.summary_brief:
                    brief_lines.append(f"Brief: {node.summary_brief}")
                payload.append({"role": "user", "content": "\n".join(brief_lines)})
                payload.append(
                    {"role": "assistant", "content": f"Understood. Starting clean branch '{frozen_title}'."}
                )
                head_len = 3

        own_msgs = await self.repo.list_messages(node.id)
        for m in own_msgs:
            if exclude_last_seq is not None and m.seq >= exclude_last_seq:
                continue
            payload.append(_message_to_openai(m))

        return payload, head_len, own_msgs

    def _trim_body_for_budget(
        self,
        body: list[dict[str, Any]],
        target_budget: int,
    ) -> list[dict[str, Any]]:
        kept_rev: list[dict[str, Any]] = []
        used = 0
        for msg in reversed(body):
            mt = estimate_messages_tokens([msg])
            if kept_rev and used + mt > target_budget:
                break
            kept_rev.append(msg)
            used += mt
        kept = list(reversed(kept_rev))
        while kept and kept[0].get("role") == "tool":
            kept.pop(0)
        return _strip_dangling_tool_calls(kept)

    async def build_static_prefix(
        self,
        node: NodeModel,
        exclude_last_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        payload, head_len, _own_msgs = await self._build_raw_static_prefix(
            node, exclude_last_seq=exclude_last_seq
        )

        # Handoff-preserving, block-aligned context-window overflow guard:
        #   - Pins `head` (system prompt + compaction/slice/clean handoff pair when it fits)
        #   - Trims in hysteresis blocks down to 75% watermark so the trimmed prefix stays
        #     byte-for-byte identical across subsequent turns instead of invalidating KV cache
        #     on every single turn!
        if self.max_context_tokens is not None and len(payload) > head_len + 1:
            budget_cap = self.max_context_tokens
            head = payload[:head_len]
            sys_tok = estimate_messages_tokens([payload[0]])
            brief_tok = estimate_messages_tokens(payload[1:head_len]) if head_len > 1 else 0
            if head_len > 1 and brief_tok > max(96, (budget_cap - sys_tok) // 2):
                head = [payload[0]]
                head_len = 1
            body = payload[head_len:]
            head_tok = estimate_messages_tokens(head)
            body_budget = max(128, budget_cap - head_tok - 64)

            cache_key = (node.id, budget_cap)
            saved_drop = self._trim_cutoffs.get(cache_key, 0)
            if saved_drop >= len(body):
                saved_drop = 0
                self._trim_cutoffs.pop(cache_key, None)

            candidate_body = body[saved_drop:] if saved_drop > 0 else body
            if saved_drop > 0:
                while candidate_body and candidate_body[0].get("role") == "tool":
                    candidate_body = candidate_body[1:]
                candidate_body = _strip_dangling_tool_calls(candidate_body)

            total_tok = head_tok + estimate_messages_tokens(candidate_body) + (64 if saved_drop > 0 else 0)
            if total_tok > budget_cap:
                # Hysteresis: shed down to 75% of available body budget so the next several turns
                # reuse the exact same trimmed prefix with 0 KV-cache invalidation.
                watermark_budget = max(96, int(body_budget * 0.75))
                kept = self._trim_body_for_budget(body, watermark_budget)
                dropped = max(0, len(body) - len(kept))
                self._trim_cutoffs[cache_key] = dropped
                marker = {
                    "role": "user",
                    "content": (
                        f"[Chatree Context Guard: omitted {dropped} earlier message(s) to stay "
                        f"within max_branch_context_tokens={budget_cap}. "
                        f"Use dialogue_tree_search_branches() if older details are needed.]"
                    ),
                }
                payload = [*head, marker, *kept]
            elif saved_drop > 0:
                dropped = max(0, len(body) - len(candidate_body))
                marker = {
                    "role": "user",
                    "content": (
                        f"[Chatree Context Guard: omitted {dropped} earlier message(s) to stay "
                        f"within max_branch_context_tokens={budget_cap}. "
                        f"Use dialogue_tree_search_branches() if older details are needed.]"
                    ),
                }
                payload = [*head, marker, *candidate_body]
        return payload

    @staticmethod
    def _slim_envelope_in_user_content(content: str) -> str:
        """Drop optional envelope sections (BRANCH SUMMARIES / RELATED / FACTS / SESSIONS) first when tight."""
        start_tag = "<dialogue_tree_context>"
        end_tag = "</dialogue_tree_context>"
        s_idx = content.find(start_tag)
        e_idx = content.find(end_tag)
        if s_idx == -1 or e_idx == -1 or e_idx <= s_idx:
            return content
        env_text = content[s_idx : e_idx + len(end_tag)]
        kept_lines: list[str] = []
        skip_section = False
        for ln in env_text.splitlines():
            if ln.startswith(("BRANCH SUMMARIES", "RELATED [", "FACTS [", "SESSIONS [", "SUBAGENTS [")):
                skip_section = True
                continue
            if ln.startswith(("ORIENTATION:", "STALE [", "MAILBOX [", "</dialogue_tree_context>")):
                skip_section = False
            if skip_section and ln.startswith("- "):
                continue
            kept_lines.append(ln)
        slim_env = "\n".join(kept_lines)
        return content[:s_idx] + slim_env + content[e_idx + len(end_tag) :]

    @staticmethod
    def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        cur_tok = estimate_tokens(text)
        if cur_tok <= max_tokens:
            return text
        marker = "\n...[truncated to fit model context window]...\n"
        chars_per_tok = max(1.5, len(text) / max(1, cur_tok))
        max_chars = max(len(marker) + 24, int(max_tokens * chars_per_tok * 0.90))
        out = text
        for _ in range(6):
            avail = max(16, max_chars - len(marker))
            head_n = int(avail * 0.70)
            tail_n = max(0, avail - head_n)
            out = (
                text[:head_n].rstrip()
                + marker
                + (text[-tail_n:].lstrip() if tail_n > 0 else "")
            )
            if estimate_tokens(out) <= max_tokens or max_chars <= len(marker) + 24:
                break
            max_chars = max(len(marker) + 24, int(max_chars * 0.78))
        return out

    @classmethod
    def _truncate_user_preserving_envelope(cls, text: str, max_tokens: int) -> str:
        """
        When a single user turn exceeds `max_tokens`, preserve `<dialogue_tree_context>`
        (including `PINNED` invariants, `TREE`, `STALE`, and `MAILBOX`) intact and truncate
        only the user message body whenever the envelope fits inside `max_tokens`.
        """
        start_tag = "<dialogue_tree_context>"
        end_tag = "</dialogue_tree_context>"
        s_idx = text.find(start_tag)
        e_idx = text.find(end_tag)
        if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
            env_end = e_idx + len(end_tag)
            env_block = text[:env_end]
            body = text[env_end:]
            env_tok = estimate_tokens(env_block)
            if env_tok < max_tokens - 32 and body.strip():
                body_budget = max(16, max_tokens - env_tok - 4)
                return env_block + "\n\n" + cls._truncate_text_to_tokens(body.lstrip(), body_budget)
        return cls._truncate_text_to_tokens(text, max_tokens)

    def fit_messages_to_budget(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens_override: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Whole-request context-window enforcer accounting for:
          prefix + ephemeral envelope + newest user message + trailing tool messages
          + tool schemas + reply reserve.
        Returns `messages` untouched when already within budget (0 KV-cache impact).
        """
        if not messages:
            return messages
        configured_cap = (
            max_tokens_override
            if max_tokens_override is not None
            else self.max_context_tokens
        )
        window = resolve_model_context_window(
            model,
            overrides=self.guardrails.model_context_windows,
            configured_max=configured_cap,
        )
        tool_tokens = estimate_tokens(json.dumps(tools)) if tools else 0
        reserve = min(self.guardrails.reply_reserve_tokens, max(64, window // 5))
        if window - tool_tokens - reserve >= 96:
            input_budget = window - tool_tokens - reserve
        elif window - tool_tokens >= 96:
            input_budget = window - tool_tokens
        else:
            # Tiny synthetic cap smaller than serialized tool schemas alone
            input_budget = max(96, window - min(tool_tokens, window // 2) - min(reserve, window // 5))

        if estimate_messages_tokens(messages) <= input_budget:
            return messages

        fitted = [dict(m) for m in messages]

        # 1. Identify pinned head (system + optional slice/clean brief pair) and active tail
        head_end = 1 if fitted[0].get("role") == "system" else 0
        if (
            len(fitted) >= head_end + 2
            and fitted[head_end].get("role") == "user"
            and fitted[head_end + 1].get("role") == "assistant"
            and any(
                tag in str(fitted[head_end].get("content") or "")
                for tag in ("[Branch Slice Brief]", "[Clean Branch Brief]")
            )
        ):
            if estimate_messages_tokens(fitted[: head_end + 2]) < int(input_budget * 0.55):
                head_end += 2

        # Active tail: newest user message (+ immediately preceding [Searched Excerpts] if present)
        # + any trailing assistant(tool_calls)/tool messages
        tail_start = len(fitted) - 1
        while tail_start > head_end and fitted[tail_start].get("role") != "user":
            tail_start -= 1
        if (
            tail_start - 1 >= head_end
            and "[Searched Excerpts" in str(fitted[tail_start - 1].get("content") or "")
        ):
            tail_start -= 1
        if tail_start < head_end:
            tail_start = head_end

        # 2. Slim optional sections inside ephemeral envelope before touching static prefix
        if fitted[tail_start].get("role") == "user":
            orig_u = str(fitted[tail_start].get("content") or "")
            slim_u = self._slim_envelope_in_user_content(orig_u)
            if slim_u != orig_u:
                fitted[tail_start] = {**fitted[tail_start], "content": slim_u}
                if estimate_messages_tokens(fitted) <= input_budget:
                    return fitted

        # 2b. On tight windows where the static system prompt alone consumes >50% of input_budget,
        #     trim the system prompt first so active tail/excerpts are not starved.
        if fitted[0].get("role") == "system":
            sys_tok = estimate_messages_tokens([fitted[0]])
            if sys_tok > int(input_budget * 0.50):
                fitted[0] = {
                    **fitted[0],
                    "content": self._truncate_text_to_tokens(
                        str(fitted[0].get("content") or ""), max(48, int(input_budget * 0.35))
                    ),
                }
                if estimate_messages_tokens(fitted) <= input_budget:
                    return fitted

        # 3. Drop oldest middle turns between `head` and active `tail`
        head = fitted[:head_end]
        middle = fitted[head_end:tail_start]
        tail = fitted[tail_start:]
        if middle:
            fixed_tok = estimate_messages_tokens(head) + estimate_messages_tokens(tail) + 48
            mid_budget = max(0, input_budget - fixed_tok)
            if mid_budget <= 48:
                kept_mid: list[dict[str, Any]] = []
            else:
                kept_mid = self._trim_body_for_budget(middle, mid_budget)
            dropped = len(middle) - len(kept_mid)
            if dropped > 0:
                marker = {
                    "role": "user",
                    "content": (
                        f"[Chatree Context Guard: omitted {dropped} earlier message(s) to fit "
                        f"model context window ({window} tok).]"
                    ),
                }
                fitted = [*head, marker, *kept_mid, *tail]
            else:
                fitted = [*head, *kept_mid, *tail]

        if estimate_messages_tokens(fitted) <= input_budget:
            return fitted

        # 4. If even head + tail exceed budget (e.g. massive user prompt or giant tool output),
        #    drop oldest non-excerpt/non-final messages first, then proportionally trim largest msgs.
        while len(fitted) > 2 and estimate_messages_tokens(fitted) > input_budget:
            drop_idx = next(
                (
                    i
                    for i in range(1, len(fitted) - 1)
                    if "[Searched Excerpts" not in str(fitted[i].get("content") or "")
                ),
                None,
            )
            if drop_idx is None:
                break
            if fitted[drop_idx].get("role") == "assistant" and fitted[drop_idx].get("tool_calls"):
                del fitted[drop_idx]
                while drop_idx < len(fitted) - 1 and fitted[drop_idx].get("role") == "tool":
                    del fitted[drop_idx]
            else:
                del fitted[drop_idx]

        if estimate_messages_tokens(fitted) > input_budget:
            # Truncate system prompt only if necessary, then truncate the largest remaining message(s)
            sys_tok = estimate_messages_tokens([fitted[0]]) if fitted[0].get("role") == "system" else 0
            if sys_tok > int(input_budget * 0.45) and fitted[0].get("role") == "system":
                fitted[0] = {
                    **fitted[0],
                    "content": self._truncate_text_to_tokens(
                        str(fitted[0].get("content") or ""), max(48, int(input_budget * 0.35))
                    ),
                }
            # Trim remaining messages from largest down until strictly within input_budget
            for _ in range(8):
                cur_tok = estimate_messages_tokens(fitted)
                if cur_tok <= input_budget:
                    break
                excess = cur_tok - input_budget + 16
                # Pick largest non-system message (or message 0 if single message)
                cand_indices = list(range(1, len(fitted))) if len(fitted) > 1 else [0]
                largest_idx = max(
                    cand_indices,
                    key=lambda idx: estimate_messages_tokens([fitted[idx]]),
                )
                msg_tok = estimate_messages_tokens([fitted[largest_idx]])
                target_msg_tok = max(32, msg_tok - excess)
                raw_txt = str(fitted[largest_idx].get("content") or "")
                trimmed_txt = (
                    self._truncate_user_preserving_envelope(raw_txt, target_msg_tok)
                    if fitted[largest_idx].get("role") == "user"
                    else self._truncate_text_to_tokens(raw_txt, target_msg_tok)
                )
                fitted[largest_idx] = {
                    **fitted[largest_idx],
                    "content": trimmed_txt,
                }
        return fitted

    async def build_ephemeral_envelope(
        self,
        tree: TreeModel,
        node: NodeModel,
        *,
        session_id: str = DEFAULT_SESSION_ID,
        speculative_hits: list[SearchHit] | None = None,
        fact_hits: list[SearchHit] | None = None,
        unread_mailbox: list[MailboxItemModel] | None = None,
        compact_steady_state: bool = False,
    ) -> tuple[str, list[str], list[str]]:
        if not self.ablations.enable_ephemeral_cues:
            return "", [], []

        children = await self.repo.get_children(node.id)
        all_nodes = await self.repo.list_nodes(tree.id)
        nodes_by_id = {n.id: n for n in all_nodes}
        parent_node = nodes_by_id.get(node.parent_id) if node.parent_id else None
        siblings = [
            n
            for n in all_nodes
            if n.parent_id == node.parent_id and n.id != node.id and node.parent_id is not None
        ]
        # Order siblings & children newest-updated first (prioritizing interactive chat over subagents)
        siblings.sort(key=lambda n: n.updated_at, reverse=True)
        siblings.sort(key=lambda n: 0 if n.node_kind == "chat" else 1)
        children.sort(key=lambda n: n.updated_at, reverse=True)
        children.sort(key=lambda n: 0 if n.node_kind == "chat" else 1)

        reflog_entries = await self.repo.list_reflog(tree.id, limit=24, session_id=session_id)
        invariants = await self.repo.list_tree_invariants(tree.id)
        cues = await self.repo.get_unconsumed_orientation_cues(
            tree.id, node.id, session_id=session_id
        )
        all_sessions = (
            await self.repo.list_sessions(tree.id)
            if hasattr(self.repo, "list_sessions")
            else []
        )
        other_sessions = [s for s in all_sessions if s.id != session_id]
        running_jobs = (
            [
                j
                for j in await self.repo.list_background_jobs(tree.id, limit=20)
                if j.status == "running"
            ]
            if hasattr(self.repo, "list_background_jobs")
            else []
        )
        stale_deps = list(node.stale_dependencies) if self.ablations.enable_staleness_notices else []
        if not self.ablations.enable_transitive_staleness:
            stale_deps = [d for d in stale_deps if d.hops == 1]

        self_kind_tag = f" kind={node.node_kind}" if node.node_kind != "chat" else ""
        lines: list[str] = [
            "<dialogue_tree_context>",
            (
                f'branch={node.id} "{_escape_envelope_text(node.title)}" | '
                f"status={node.status}{self_kind_tag} mode={node.context_mode} depth={node.depth} "
                f"perm={node.tool_permission} parent={node.parent_id or 'NONE'} path=\"{node.path}\""
            ),
        ]
        if node.goal:
            lines.append(f"goal={_escape_envelope_text(node.goal)}")
        if node.context_mode == "fork" and node.summary_brief:
            lines.append(f"brief={_escape_envelope_text(node.summary_brief)}")
        notices: list[str] = []
        cue_ids: list[str] = []
        if invariants:
            inv_strs: list[str] = []
            max_pins = getattr(self.guardrails, "max_pinned_invariants", 8)
            for inv in invariants[:max_pins]:
                stale_tag = ""
                if inv.source_node_id and self.ablations.enable_staleness_notices:
                    src_node = nodes_by_id.get(inv.source_node_id)
                    if src_node is not None:
                        if src_node.status == "ARCHIVED":
                            stale_tag = f" [STALE: source {src_node.id} archived]"
                        elif src_node.summary_version > inv.source_summary_version:
                            stale_tag = f" [STALE: source {src_node.id} revised to v{src_node.summary_version}]"
                        elif src_node.is_dirty and src_node.last_msg_seq > inv.source_msg_seq:
                            stale_tag = f" [STALE: source {src_node.id} has uncommitted edits]"
                if stale_tag:
                    notices.append(f"PINNED {inv.key}{stale_tag}")
                src_tag = (
                    f" [src:{inv.source_node_id}]"
                    if (inv.source_node_id and not stale_tag)
                    else ""
                )
                inv_strs.append(
                    f"{_escape_envelope_text(inv.key)}={_escape_envelope_text(inv.value)}{src_tag}{stale_tag}"
                )
            lines.append("PINNED [key=value]: " + "; ".join(inv_strs))

        # 1. Structural TREE topology (parent, newest siblings, newest children + explicit omitted counts)
        tree_parts: list[str] = []
        shown_in_tree_ids: set[str] = set()
        if parent_node is not None:
            shown_in_tree_ids.add(parent_node.id)
            tree_parts.append(
                f'parent={parent_node.id} "{_escape_envelope_text(parent_node.title)}"'
            )
        if siblings:
            shown_sibs = siblings[:8]
            shown_in_tree_ids.update(s.id for s in shown_sibs)
            sib_strs = [
                f'{s.id} "{_escape_envelope_text(s.title)}"'
                f'{" [subagent]" if s.node_kind == "subagent" else ""} '
                f'{s.status} v{s.summary_version}{"*" if s.is_dirty else ""}'
                for s in shown_sibs
            ]
            sib_more = f"; (+{len(siblings) - 8} more)" if len(siblings) > 8 else ""
            tree_parts.append("siblings: " + "; ".join(sib_strs) + sib_more)
        if children:
            shown_children = children[:10]
            shown_in_tree_ids.update(c.id for c in shown_children)
            child_strs = [
                f'{c.id} "{_escape_envelope_text(c.title)}"'
                f'{" [subagent]" if c.node_kind == "subagent" else ""} '
                f'{c.status} v{c.summary_version}{"*" if c.is_dirty else ""}'
                for c in shown_children
            ]
            child_more = f"; (+{len(children) - 10} more)" if len(children) > 10 else ""
            tree_parts.append("children: " + "; ".join(child_strs) + child_more)
        if tree_parts:
            lines.append('TREE [id "title" ([subagent]) status v# (*=dirty)]: ' + " | ".join(tree_parts))

        # 2. Temporal REFLOG visit stack (HEAD-only, MRU-deduplicated while preserving exact HEAD@{k} indices!)
        reflog_rank: dict[str, int] = {}
        for idx, entry in enumerate(reflog_entries):
            if entry.to_node_id not in reflog_rank:
                reflog_rank[entry.to_node_id] = idx
        if len(reflog_entries) > 1:
            reflog_parts: list[str] = []
            seen_reflog_nodes: set[str] = set()
            for idx, entry in enumerate(reflog_entries):
                if len(reflog_parts) >= 5:
                    break
                if entry.to_node_id in seen_reflog_nodes:
                    continue
                seen_reflog_nodes.add(entry.to_node_id)
                dest = nodes_by_id.get(entry.to_node_id)
                dest_title = _escape_envelope_text(dest.title) if dest else entry.to_node_id
                reflog_parts.append(f'@{{{idx}}}={entry.to_node_id} "{dest_title}" ({entry.action})')
            if len(reflog_parts) > 1:
                lines.append(
                    'REFLOG [HEAD@{k}=id "title" (action) -> call dialogue_tree_switch_branch]: '
                    + "; ".join(reflog_parts)
                )

        # 3. Compact Tier-2 BRANCH SUMMARIES overview:
        #    Ordered by: (a) relevant to current turn (speculative_hits), (b) recently visited (reflog MRU),
        #    (c) interactive chat over subagent, (d) most recently updated (`updated_at` DESC).
        spec_rank: dict[str, int] = {}
        if speculative_hits:
            for idx, sh in enumerate(speculative_hits):
                if sh.node_id not in spec_rank:
                    spec_rank[sh.node_id] = idx
        summarized_others = [
            n
            for n in all_nodes
            if n.id != node.id and n.status != "ARCHIVED" and n.summary and n.summary.strip()
        ]
        # Stable multi-pass sort: newest updated_at first, then chat before subagent,
        # then reflog MRU, then speculative relevance to current turn!
        summarized_others.sort(key=lambda n: n.updated_at, reverse=True)
        summarized_others.sort(
            key=lambda n: (
                0 if n.id in spec_rank else (1 if n.id in reflog_rank else 2),
                spec_rank.get(n.id, 999),
                reflog_rank.get(n.id, 999),
                0 if n.node_kind == "chat" else 1,
            )
        )
        if summarized_others:
            if compact_steady_state:
                # Keep every top-8 branch ID visible without duplicating IDs already listed in TREE
                # right above (preserving steady-state KV-cache efficiency on small trees while
                # surfacing cousin/deep branch IDs and omitted counts on large trees).
                extra_others = [o for o in summarized_others[:8] if o.id not in shown_in_tree_ids]
                more_tag = (
                    f" (+{len(summarized_others) - 8} more)"
                    if len(summarized_others) > 8
                    else ""
                )
                if extra_others:
                    extra_roster = "; ".join(
                        f'{o.id} "{_escape_envelope_text(o.title[:36])}" v{o.summary_version}'
                        for o in extra_others
                    )
                    lines.append(
                        f"BRANCH SUMMARIES: {len(summarized_others)} unchanged "
                        f"[+{extra_roster}{more_tag}] "
                        "(call dialogue_tree_query_branch(id) for 0-token cached summary)"
                    )
                else:
                    lines.append(
                        f"BRANCH SUMMARIES: {len(summarized_others)} unchanged{more_tag} "
                        "(IDs in TREE above; call dialogue_tree_query_branch(id) for 0-token cached summary)"
                    )
            else:
                lines.append('BRANCH SUMMARIES [id "title" status v# | summary]:')
                for other in summarized_others[:8]:
                    compact = _escape_envelope_text(" ".join((other.summary or "").split())[:140])
                    kind_tag = " [subagent]" if other.node_kind == "subagent" else ""
                    lines.append(
                        f'- {other.id} "{_escape_envelope_text(other.title)}"{kind_tag} '
                        f"{other.status} v{other.summary_version} | {compact}"
                    )
                if len(summarized_others) > 8:
                    lines.append(
                        f"- ... (+{len(summarized_others) - 8} more summarized branches; "
                        "call dialogue_tree_search_branches(query='*', offset=8) to browse)"
                    )

        if cues:
            lines.append("ORIENTATION:")
            for c in reversed(cues):
                safe_cue = _escape_envelope_text(c.orientation_cue)
                lines.append(f"- {safe_cue}")
                notices.append(c.orientation_cue)
                cue_ids.append(c.id)

        if stale_deps:
            # Cap STALE notices to prevent unbounded envelope growth on high fan-in hubs
            stale_deps.sort(
                key=lambda d: (
                    0 if d.severity == "material" else 1,
                    d.hops,
                    -d.current_summary_version,
                    d.source_node_id,
                )
            )
            max_stale = getattr(self.guardrails, "max_stale_notices", 8)
            shown_stale = stale_deps[:max_stale]
            lines.append(
                'STALE [id "title" severity hops | reason -> call dialogue_tree_query_branch(id)]:'
            )
            for dep in shown_stale:
                delta_tag = (
                    f" [delta: {_escape_envelope_text(dep.change_delta)}]"
                    if dep.change_delta
                    else ""
                )
                row = (
                    f'{dep.source_node_id} "{_escape_envelope_text(dep.source_title)}" '
                    f"{dep.severity} {dep.hops}h | {_escape_envelope_text(dep.reason)}{delta_tag}"
                )
                lines.append(f"- {row}")
                notices.append(row)
            if len(stale_deps) > max_stale:
                extra_s = f"... (+{len(stale_deps) - max_stale} more stale dependencies)"
                lines.append(f"- {extra_s}")
                notices.append(extra_s)

        if speculative_hits and self.ablations.enable_speculative_pointers:
            # Factored pointer table: action declared ONCE in header, compact data rows below
            seen_nodes: set[str] = set()
            pointer_lines: list[str] = []
            for h in speculative_hits:
                if h.node_id == node.id or h.node_id in seen_nodes:
                    continue
                seen_nodes.add(h.node_id)
                one_line = _escape_envelope_text(" ".join((h.summary or h.snippet).split())[:140])
                kind_tag = " [subagent]" if h.node_kind == "subagent" else ""
                pointer_lines.append(
                    f'- {h.node_id} "{_escape_envelope_text(h.node_title)}"{kind_tag} '
                    f"{h.node_status} v{h.summary_version} {h.score:.3f} | {one_line}"
                )
            if pointer_lines:
                lines.append(
                    'RELATED [id "title" status v# score | summary -> query_branch (or switch_branch for chat topics)]:'
                )
                lines.extend(pointer_lines)

        if fact_hits and self.ablations.enable_atomic_facts:
            fact_lines: list[str] = []
            seen_facts: set[str] = set()
            for fh in fact_hits:
                if fh.node_id == node.id or fh.node_status == "ARCHIVED":
                    continue
                body = fh.snippet.strip()
                if "Fact:" in body:
                    body = body.split("Fact:", 1)[1].strip()
                body = _escape_envelope_text(" ".join(body.split())[:160])
                dedup_key = f"{fh.node_id}:{body.lower()}"
                if body and dedup_key not in seen_facts:
                    seen_facts.add(dedup_key)
                    fact_lines.append(
                        f'- {fh.node_id} "{_escape_envelope_text(fh.node_title)}" | {body}'
                    )
                if len(fact_lines) >= 4:
                    break
            if fact_lines:
                lines.append('FACTS [id "title" | dated atomic fact (verify via query_branch if critical)]:')
                lines.extend(fact_lines)

        if other_sessions:
            sess_strs: list[str] = []
            for osess in other_sessions[:6]:
                head_n = nodes_by_id.get(osess.head_node_id) if osess.head_node_id else None
                head_lbl = (
                    f'{head_n.id} "{_escape_envelope_text(head_n.title)}"'
                    if head_n
                    else (osess.head_node_id or "NONE")
                )
                same_tag = " (SAME NODE)" if osess.head_node_id == node.id else ""
                sess_strs.append(
                    f'"{_escape_envelope_text(osess.title)}" ({osess.id}, {osess.kind}) at {head_lbl}{same_tag}'
                )
            lines.append("SESSIONS [other visitors on this tree]: " + "; ".join(sess_strs))

        if running_jobs:
            job_strs = [
                f'{j.child_node_id} "{_escape_envelope_text(j.title)}" (parent={j.parent_node_id})'
                for j in running_jobs[:6]
            ]
            lines.append(
                "SUBAGENTS [running -> results arrive in MAILBOX; do not duplicate-spawn]: "
                + "; ".join(job_strs)
            )

        if unread_mailbox:
            max_mb = getattr(self.guardrails, "max_mailbox_notices", 6)
            shown_mb = unread_mailbox[-max_mb:]
            lines.append('MAILBOX [from_id "title" | summary | artifacts]:')
            for mb in shown_mb:
                sender_title = _escape_envelope_text(
                    str(mb.payload.get("sender_title") or mb.from_node_id)
                )
                art_str = (
                    " | "
                    + ", ".join(_escape_envelope_text(r.ref_label) for r in mb.artifact_refs)
                    if mb.artifact_refs
                    else ""
                )
                safe_summary = _escape_envelope_text(mb.summary[:260])
                row = f'{mb.from_node_id} "{sender_title}" | {safe_summary}{art_str}'
                lines.append(f"- {row}")
                notices.append(row)
            if len(unread_mailbox) > max_mb:
                extra_mb = f"... (+{len(unread_mailbox) - max_mb} more unread mailbox items)"
                lines.append(f"- {extra_mb}")
                notices.append(extra_mb)

        lines.append("</dialogue_tree_context>")
        return "\n".join(lines), cue_ids, notices

    async def assemble_for_active_turn(
        self,
        tree: TreeModel,
        node: NodeModel,
        newest_user_msg: MessageModel,
        trailing_tool_messages: list[MessageModel] | None = None,
        speculative_hits: list[SearchHit] | None = None,
        fact_hits: list[SearchHit] | None = None,
        unread_mailbox: list[MailboxItemModel] | None = None,
        session_id: str = DEFAULT_SESSION_ID,
        compact_steady_state: bool = False,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        prefix = await self.build_static_prefix(node, exclude_last_seq=newest_user_msg.seq)
        envelope, cue_ids, notices = await self.build_ephemeral_envelope(
            tree,
            node,
            session_id=session_id,
            speculative_hits=speculative_hits,
            fact_hits=fact_hits,
            unread_mailbox=unread_mailbox,
            compact_steady_state=compact_steady_state,
        )
        combined_user = (
            f"{envelope}\n\n{newest_user_msg.content}" if envelope else newest_user_msg.content
        )
        messages = [*prefix, {"role": "user", "content": combined_user}]
        if trailing_tool_messages:
            for m in trailing_tool_messages:
                messages.append(_message_to_openai(m))
        if model is not None or self.max_context_tokens is not None:
            messages = self.fit_messages_to_budget(messages, model=model, tools=tools)
        return messages, cue_ids, notices

    async def assemble_for_shadow_query(
        self,
        target_node: NodeModel,
        caller_node: NodeModel,
        prompt: str,
        *,
        rag: Any | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        raw_payload, head_len, own_msgs = await self._build_raw_static_prefix(target_node)
        window = resolve_model_context_window(
            model,
            overrides=self.guardrails.model_context_windows,
            configured_max=self.max_context_tokens,
        )
        reserve = min(self.guardrails.reply_reserve_tokens, max(128, window // 4))

        shadow_user = (
            f'[Shadow Query | target={target_node.id} "{_escape_envelope_text(target_node.title)}" '
            f'| from={caller_node.id} "{_escape_envelope_text(caller_node.title)}" '
            f"| read-only, no state-changing tools | "
            f'You ARE branch {target_node.id} ("{_escape_envelope_text(target_node.title)}"). '
            f"Answer strictly from THIS branch's own conversation turns above (do not confuse "
            f"quoted tool outputs from other branches with your own identity, and do not call "
            f"tools unless explicitly asked). Extract concisely any relevant facts, exact "
            f"identifiers/tokens, dates, latest settled decisions, or relevant user "
            f"preferences/traits/clues from THIS branch (omit superseded or rejected earlier "
            f"proposals unless explicitly asked); reply with exactly NOT_FOUND_IN_BRANCH ONLY "
            f"if THIS branch has no relevant information or clues for the query]\n"
            f"Query: {prompt}"
        )

        # Fast path: branch fits completely inside the model context window (0 overhead, 100% KV-safe!)
        if estimate_messages_tokens(raw_payload) + estimate_tokens(shadow_user) + 64 <= max(
            256, window - reserve
        ):
            prefix = _strip_dangling_tool_calls(raw_payload)
            return [*prefix, {"role": "user", "content": shadow_user}]

        # Over-budget fallback: retrieve matching earlier turns strictly inside `target_node`
        # (hybrid RAG chunks + lexical identifier/term overlap) with +-1 neighbor context so
        # shadow Q&A never drops Turn 1 or falsely claims NOT_FOUND_IN_BRANCH.
        head = raw_payload[:head_len]
        if head_len > 1 and estimate_messages_tokens(head) > max(128, int(window * 0.45)):
            head = [raw_payload[0]]
            head_len = 1

        matched_seqs: list[int] = []
        if rag is not None and getattr(getattr(rag, "cfg", None), "enabled", False):
            try:
                hits = await rag.search(
                    tree_id=target_node.tree_id,
                    query=prompt,
                    scope="subtree",
                    current_node=target_node,
                    top_k=8,
                    granularities={"chunk", "fact"},
                    use_reranker=False,
                )
                for h in hits:
                    if h.node_id == target_node.id and h.seq_range is not None:
                        matched_seqs.extend(range(h.seq_range[0], h.seq_range[1] + 1))
                    elif h.node_id == target_node.id and h.matched_seq is not None:
                        matched_seqs.append(h.matched_seq)
            except Exception:
                pass

        # Deterministic lexical overlap scan over `own_msgs` catches exact tokens/numbers even
        # when FTS tokenization splits symbols or before chunks are refreshed.
        q_stop = {
            "what", "which", "when", "where", "about", "this", "that", "from", "with", "does",
            "were", "have", "been", "into", "your", "their", "branch", "turn", "turns", "exact",
            "confirm", "tell", "show", "give", "value", "current", "final",
        }
        q_terms = {
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_.-]{3,}", prompt)
            if w.lower() not in q_stop
        }
        scored_msgs: list[tuple[float, int]] = []
        for idx, m in enumerate(own_msgs):
            if m.role not in ("user", "assistant") or not m.content.strip():
                continue
            low_c = m.content.lower()
            overlap = sum(2.0 if "_" in t or any(ch.isdigit() for ch in t) else 1.0 for t in q_terms if t in low_c)
            if overlap > 0:
                scored_msgs.append((overlap, idx))
        scored_msgs.sort(key=lambda pair: (-pair[0], pair[1]))

        direct_indices: list[int] = []
        neighbor_indices: list[int] = []
        seq_to_idx = {m.seq: i for i, m in enumerate(own_msgs)}
        for _score, idx in scored_msgs[:4]:
            if idx not in direct_indices:
                direct_indices.append(idx)
            for nb in (idx - 1, idx + 1):
                if (
                    0 <= nb < len(own_msgs)
                    and own_msgs[nb].role in ("user", "assistant")
                    and nb not in neighbor_indices
                ):
                    neighbor_indices.append(nb)
        for s in matched_seqs[:10]:
            found_idx = seq_to_idx.get(s)
            if found_idx is not None and found_idx not in neighbor_indices:
                neighbor_indices.append(found_idx)

        direct_set = set(direct_indices)
        excerpt_budget = max(160, int((window - reserve) * 0.35))
        chosen_direct: list[tuple[int, str]] = []
        chosen_neighbors: list[tuple[int, str]] = []
        used_ex = 0
        # Place direct query-matching turns FIRST (chronological) and surrounding neighbor turns
        # after them so tight truncation NEVER clips direct evidence behind filler neighbors.
        for idx in [*direct_indices, *(n for n in neighbor_indices if n not in direct_set)]:
            m = own_msgs[idx]
            txt = " ".join(m.content.split())
            is_direct = idx in direct_set
            max_len = 500 if is_direct else 180
            if len(txt) > max_len:
                txt = txt[:max_len] + "..."
            ln = f"- [{m.role} #{m.seq}] {txt}"
            lt = estimate_tokens(ln)
            if (chosen_direct or chosen_neighbors) and used_ex + lt > excerpt_budget:
                if is_direct:
                    continue
                break
            if is_direct:
                chosen_direct.append((idx, ln))
            else:
                chosen_neighbors.append((idx, ln))
            used_ex += lt
        chosen_direct.sort(key=lambda pair: pair[0])
        chosen_neighbors.sort(key=lambda pair: pair[0])
        excerpt_lines = [ln for _, ln in [*chosen_direct, *chosen_neighbors]]

        body = raw_payload[head_len:]
        partial_shadow_user = (
            f'[Shadow Query | Partial View | target={target_node.id} "{_escape_envelope_text(target_node.title)}" '
            f'| from={caller_node.id} "{_escape_envelope_text(caller_node.title)}" '
            f"| read-only, no state-changing tools | "
            f"Target branch exceeded context window; relevant searched earlier excerpts + recent turns "
            f"are shown above. Answer strictly from these excerpts and recent turns; do NOT treat "
            f"omitted earlier turns as absent — reply NOT_FOUND_IN_BRANCH ONLY if no relevant "
            f"information appears in either the searched excerpts or recent turns.]\n"
            f"Query: {prompt}"
        )

        head_and_q_tok = (
            estimate_messages_tokens(head) + estimate_tokens(partial_shadow_user) + used_ex + 96
        )
        recent_budget = max(0, window - reserve - head_and_q_tok)
        recent_kept = (
            self._trim_body_for_budget(body, recent_budget) if recent_budget >= 64 else []
        )

        assembled: list[dict[str, Any]] = [*head]
        if target_node.summary and target_node.summary.strip():
            assembled.append(
                {
                    "role": "user",
                    "content": (
                        f"[Branch Summary Snapshot for {target_node.id} (v{target_node.summary_version})]\n"
                        f"{target_node.summary[:800]}"
                    ),
                }
            )
        assembled.extend(recent_kept)
        if excerpt_lines:
            assembled.append(
                {
                    "role": "user",
                    "content": (
                        f"[Searched Excerpts from earlier turns of branch {target_node.id} matching query]\n"
                        + "\n".join(excerpt_lines)
                    ),
                }
            )
        assembled.append({"role": "user", "content": partial_shadow_user})
        return self.fit_messages_to_budget(assembled, model=model, tools=None)

    async def get_fork_breakpoint_index(self, node: NodeModel) -> int | None:
        """
        Return the message index of the last inherited parent message for `fork` branches
        so Anthropic prompt caching can pin a 4th `cache_control` breakpoint at the shared trunk.
        """
        if node.context_mode != "fork" or not node.parent_id:
            return None
        inherited = await self.collect_inherited_messages(node)
        cleaned = _strip_dangling_tool_calls([_message_to_openai(m) for m in inherited])
        # Index 0 is the system prompt, so inherited messages occupy indices 1..len(cleaned)
        return len(cleaned) if cleaned else None

    async def estimate_node_context_tokens(self, node: NodeModel) -> int:
        return estimate_messages_tokens(await self.build_static_prefix(node))
