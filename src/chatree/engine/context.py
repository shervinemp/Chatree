from __future__ import annotations

from typing import Any

from chatree.config import AblationConfig
from chatree.providers.mock import estimate_messages_tokens
from chatree.storage.repository import SQLiteRepository
from chatree.tools import STATIC_SYSTEM_PROMPT
from chatree.types import MailboxItemModel, MessageModel, NodeModel, SearchHit, TreeModel


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
        max_context_tokens: int | None = None,
    ) -> None:
        self.repo = repo
        self.system_prompt = system_prompt
        self.ablations = ablations or AblationConfig()
        self.max_context_tokens = max_context_tokens

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

    async def build_static_prefix(
        self,
        node: NodeModel,
        exclude_last_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        inherited = await self.collect_inherited_messages(node)
        if node.context_mode == "fork":
            if inherited:
                payload.extend(
                    _strip_dangling_tool_calls([_message_to_openai(m) for m in inherited])
                )
        elif node.context_mode == "slice":
            brief_lines = [f"[Branch Slice Brief] Goal: {node.goal or node.title}"]
            if node.summary_brief:
                brief_lines.append(f"Brief: {node.summary_brief}")
            if inherited:
                brief_lines.append("\nRecent parent turns:")
                for m in inherited:
                    if m.role in ("user", "assistant") and m.content.strip():
                        brief_lines.append(f"- {m.role}: {m.content}")
            payload.append({"role": "user", "content": "\n".join(brief_lines)})
            payload.append(
                {"role": "assistant", "content": f"Understood. Ready to work on branch '{node.title}'."}
            )
        elif node.context_mode == "clean":
            if node.goal or node.summary_brief:
                brief_lines = [f"[Clean Branch Brief] Goal: {node.goal or node.title}"]
                if node.summary_brief:
                    brief_lines.append(f"Brief: {node.summary_brief}")
                payload.append({"role": "user", "content": "\n".join(brief_lines)})
                payload.append(
                    {"role": "assistant", "content": f"Understood. Starting clean branch '{node.title}'."}
                )

        own_msgs = await self.repo.list_messages(node.id)
        for m in own_msgs:
            if exclude_last_seq is not None and m.seq >= exclude_last_seq:
                continue
            payload.append(_message_to_openai(m))

        # Per-branch context-window overflow guard: keep system prompt + most recent valid turns
        if self.max_context_tokens is not None and len(payload) > 3:
            total_tok = estimate_messages_tokens(payload)
            if total_tok > self.max_context_tokens:
                head = [payload[0]]
                body = payload[1:]
                # Keep newest messages fitting within budget, preserving tool-call pairing
                kept_rev: list[dict[str, Any]] = []
                budget = max(128, self.max_context_tokens - estimate_messages_tokens(head) - 64)
                used = 0
                for msg in reversed(body):
                    mt = estimate_messages_tokens([msg])
                    if kept_rev and used + mt > budget:
                        break
                    kept_rev.append(msg)
                    used += mt
                kept = list(reversed(kept_rev))
                # Never start trimmed history on a dangling `tool` response whose call was dropped
                while kept and kept[0].get("role") == "tool":
                    kept.pop(0)
                kept = _strip_dangling_tool_calls(kept)
                dropped = len(body) - len(kept)
                marker = {
                    "role": "user",
                    "content": (
                        f"[Chatree Context Guard: omitted {dropped} earlier message(s) to stay "
                        f"within max_branch_context_tokens={self.max_context_tokens}. "
                        f"Use dialogue_tree_search_branches() if older details are needed.]"
                    ),
                }
                payload = [*head, marker, *kept]
        return payload

    async def build_ephemeral_envelope(
        self,
        tree: TreeModel,
        node: NodeModel,
        *,
        speculative_hits: list[SearchHit] | None = None,
        fact_hits: list[SearchHit] | None = None,
        unread_mailbox: list[MailboxItemModel] | None = None,
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
        reflog_entries = await self.repo.list_reflog(tree.id, limit=12)
        invariants = await self.repo.list_tree_invariants(tree.id)
        cues = await self.repo.get_unconsumed_orientation_cues(tree.id, node.id)
        stale_deps = node.stale_dependencies if self.ablations.enable_staleness_notices else []
        if not self.ablations.enable_transitive_staleness:
            stale_deps = [d for d in stale_deps if d.hops == 1]

        lines: list[str] = [
            "<dialogue_tree_context>",
            (
                f'branch={node.id} "{_escape_envelope_text(node.title)}" | '
                f"status={node.status} mode={node.context_mode} depth={node.depth} "
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
            for inv in invariants[:8]:
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
                inv_strs.append(
                    f"{_escape_envelope_text(inv.key)}={_escape_envelope_text(inv.value)}{stale_tag}"
                )
            lines.append("PINNED [key=value]: " + "; ".join(inv_strs))

        # 1. Structural TREE topology (parent, siblings, children — factored headers, zero bloat)
        tree_parts: list[str] = []
        if parent_node is not None:
            tree_parts.append(
                f'parent={parent_node.id} "{_escape_envelope_text(parent_node.title)}"'
            )
        if siblings:
            sib_strs = [
                f'{s.id} "{_escape_envelope_text(s.title)}" {s.status} v{s.summary_version}{"*" if s.is_dirty else ""}'
                for s in siblings[:8]
            ]
            tree_parts.append('siblings: ' + "; ".join(sib_strs))
        if children:
            child_strs = [
                f'{c.id} "{_escape_envelope_text(c.title)}" {c.status} v{c.summary_version}{"*" if c.is_dirty else ""}'
                for c in children[:10]
            ]
            tree_parts.append('children: ' + "; ".join(child_strs))
        if tree_parts:
            lines.append('TREE [id "title" status v# (*=dirty)]: ' + " | ".join(tree_parts))

        # 2. Temporal REFLOG visit stack (explicitly distinct from structural TREE!)
        if len(reflog_entries) > 1:
            reflog_parts: list[str] = []
            seen_refs = 0
            for idx, entry in enumerate(reflog_entries):
                if seen_refs >= 5:
                    break
                dest = nodes_by_id.get(entry.to_node_id)
                dest_title = _escape_envelope_text(dest.title) if dest else entry.to_node_id
                reflog_parts.append(f'@{{{idx}}}={entry.to_node_id} "{dest_title}" ({entry.action})')
                seen_refs += 1
            if reflog_parts:
                lines.append(
                    'REFLOG [HEAD@{k}=id "title" (action) -> call dialogue_tree_switch_branch]: '
                    + "; ".join(reflog_parts)
                )

        # 3. Compact Tier-2 BRANCH SUMMARIES overview for other non-archived summarized branches
        summarized_others = [
            n
            for n in all_nodes
            if n.id != node.id and n.status != "ARCHIVED" and n.summary and n.summary.strip()
        ]
        if summarized_others:
            lines.append('BRANCH SUMMARIES [id "title" status v# | summary]:')
            for other in summarized_others[:8]:
                compact = _escape_envelope_text(" ".join((other.summary or "").split())[:140])
                lines.append(
                    f'- {other.id} "{_escape_envelope_text(other.title)}" '
                    f"{other.status} v{other.summary_version} | {compact}"
                )

        if cues:
            lines.append("ORIENTATION:")
            for c in reversed(cues):
                safe_cue = _escape_envelope_text(c.orientation_cue)
                lines.append(f"- {safe_cue}")
                notices.append(c.orientation_cue)
                cue_ids.append(c.id)

        if stale_deps:
            lines.append(
                'STALE [id "title" severity hops | reason -> call dialogue_tree_query_branch(id)]:'
            )
            for dep in stale_deps:
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

        if speculative_hits and self.ablations.enable_speculative_pointers:
            # Factored pointer table: action declared ONCE in header, compact data rows below
            seen_nodes: set[str] = set()
            pointer_lines: list[str] = []
            for h in speculative_hits:
                if h.node_id == node.id or h.node_id in seen_nodes:
                    continue
                seen_nodes.add(h.node_id)
                one_line = _escape_envelope_text(" ".join((h.summary or h.snippet).split())[:140])
                pointer_lines.append(
                    f'- {h.node_id} "{_escape_envelope_text(h.node_title)}" '
                    f"{h.node_status} v{h.summary_version} {h.score:.3f} | {one_line}"
                )
            if pointer_lines:
                lines.append(
                    'RELATED [id "title" status v# score | summary -> call dialogue_tree_query_branch(id, prompt)]:'
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

        if unread_mailbox:
            lines.append('MAILBOX [from_id "title" | summary | artifacts]:')
            for mb in unread_mailbox:
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
    ) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        prefix = await self.build_static_prefix(node, exclude_last_seq=newest_user_msg.seq)
        envelope, cue_ids, notices = await self.build_ephemeral_envelope(
            tree,
            node,
            speculative_hits=speculative_hits,
            fact_hits=fact_hits,
            unread_mailbox=unread_mailbox,
        )
        combined_user = (
            f"{envelope}\n\n{newest_user_msg.content}" if envelope else newest_user_msg.content
        )
        messages = [*prefix, {"role": "user", "content": combined_user}]
        if trailing_tool_messages:
            for m in trailing_tool_messages:
                messages.append(_message_to_openai(m))
        return messages, cue_ids, notices

    async def assemble_for_shadow_query(
        self,
        target_node: NodeModel,
        caller_node: NodeModel,
        prompt: str,
    ) -> list[dict[str, Any]]:
        prefix = _strip_dangling_tool_calls(await self.build_static_prefix(target_node))
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
        return [*prefix, {"role": "user", "content": shadow_user}]

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
