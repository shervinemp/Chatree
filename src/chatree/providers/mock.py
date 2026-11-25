from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncIterator

from chatree.providers.tokenizer import get_default_token_counter
from chatree.tools import (
    TOOL_ASK_TREE,
    TOOL_COMPACT_BRANCH,
    TOOL_REFLOG_JUMP,
    TOOL_REQUEST,
    TOOL_SEARCH_TREE,
    TOOL_SEND,
    TOOL_SPAWN_CHILD,
)
from chatree.types import LLMResponse, LLMStreamEvent, LLMToolCall


def estimate_tokens(text: str, *, model: str | None = None) -> int:
    if not text:
        return 0
    return get_default_token_counter().count_text(text, model=model)


def estimate_messages_tokens(
    messages: list[dict[str, Any]], *, model: str | None = None
) -> int:
    return get_default_token_counter().count_messages(messages, model=model)


class MockLLMProvider:
    """
    Deterministic zero-key LLM provider exercising every Chatree primitive:
    spawn_child, send(finish=True), cached/shadow request(), parallel multi-request fan-out,
    search_tree, reflog_jump, cycle detection hooks, and custom registered tools.
    """

    def __init__(self, *, stream_delay_s: float = 0.005) -> None:
        self.stream_delay_s = stream_delay_s

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        *,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        fork_breakpoint_index: int | None = None,
    ) -> LLMResponse:
        del temperature, max_tokens, fork_breakpoint_index
        if tool_choice == "none":
            tools = None
        t0 = time.perf_counter()
        tokens_in = estimate_messages_tokens(messages)
        cache_read = int(tokens_in * 0.65) if len(messages) > 2 else 0

        last_msg = messages[-1] if messages else {"role": "user", "content": ""}
        last_role = last_msg.get("role", "user")
        last_content = str(last_msg.get("content") or "")

        ctx_match = re.search(
            r"<dialogue_tree_context>(.*?)</dialogue_tree_context>", last_content, re.DOTALL
        )
        ctx_block = ctx_match.group(1) if ctx_match else ""
        user_text = re.sub(
            r"<dialogue_tree_context>.*?</dialogue_tree_context>\s*",
            "",
            last_content,
            flags=re.DOTALL,
        ).strip()

        # 1a. Internal Structured Compactor prompt (non-destructive continuation handoff)
        if "[Chatree Compactor]" in last_content:
            anchors: list[str] = []
            snippets: list[str] = []
            for m in messages:
                if (
                    m.get("role") in ("user", "assistant")
                    and "[Chatree Compactor]" not in str(m.get("content", ""))
                    and str(m.get("content", "")).strip()
                ):
                    txt = " ".join(str(m.get("content", "")).split())
                    anchors.extend(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", txt))
                    snippets.append(f"- {m['role']}: {txt[:140]}")
            dedup_anchors = list(dict.fromkeys(a for a in anchors if a not in {"OBJECTIVE", "LOCKED", "VERIFIED", "REJECTED", "OPEN", "DEMO", "MODE"}))[:16]
            handoff = (
                "1. OBJECTIVE: Continue active workstream seamlessly.\n"
                f"2. LOCKED DECISIONS & VERBATIM ANCHORS: {', '.join(dedup_anchors) or 'None recorded'}.\n"
                "3. VERIFIED FACTS & ARTIFACTS:\n"
                + ("\n".join(snippets[-5:]) or "- No prior turns.")
                + "\n4. REJECTED / SUPERSEDED: Superseded proposals archived in ancestor branch.\n"
                "5. OPEN NEXT STEPS: Proceed with incoming user request in continuation session."
            )
            return LLMResponse(
                content=handoff,
                model_used=model,
                tokens_in=tokens_in,
                tokens_out=estimate_tokens(handoff),
                cache_read_tokens=cache_read,
                latency_ms=max(8, int((time.perf_counter() - t0) * 1000)),
            )

        # 1. Internal summarizer / subtree rollup prompt
        if "[Chatree Summarizer]" in last_content:
            history_snippets: list[str] = []
            for m in messages:
                if (
                    m.get("role") in ("user", "assistant")
                    and "[Chatree Summarizer]" not in str(m.get("content", ""))
                    and str(m.get("content", "")).strip()
                ):
                    raw_line = " ".join(str(m.get("content", "")).split())
                    raw_line = raw_line.replace("**", "").replace("*", "")
                    if len(raw_line) > 150:
                        raw_line = raw_line[:150].rsplit(" ", 1)[0] + "…"
                    history_snippets.append(f"- {m['role']}: {raw_line}")
            bullets = "\n".join(history_snippets[-6:]) or "- No dialogue turns recorded yet."
            summary = f"Concise branch summary of key points and conclusions:\n{bullets}"
            return LLMResponse(
                content=summary,
                model_used=model,
                tokens_in=tokens_in,
                tokens_out=estimate_tokens(summary),
                cache_read_tokens=cache_read,
                latency_ms=max(8, int((time.perf_counter() - t0) * 1000)),
            )

        # 1b. Internal Eval Judge prompt (offline deterministic judge fallback)
        if "[Chatree Eval Judge]" in last_content:
            exp_m = re.search(r"EXPECTED_FACTS:\s*(.*)", last_content)
            forb_m = re.search(r"FORBIDDEN_FACTS:\s*(.*)", last_content)
            ans_m = re.search(r"CANDIDATE_ANSWER:\s*(.*)", last_content, re.DOTALL)
            expected = [x.strip() for x in (exp_m.group(1) if exp_m else "").split(",") if x.strip()]
            forbidden = [x.strip() for x in (forb_m.group(1) if forb_m else "").split(",") if x.strip()]
            cand = (ans_m.group(1) if ans_m else "").strip()
            exp_hits = sum(1 for k in expected if k.lower() in cand.lower())
            forb_hits = [k for k in forbidden if k.lower() in cand.lower()]
            recall = (exp_hits / len(expected)) if expected else 1.0
            score = max(0.0, recall - (0.5 if forb_hits else 0.0))
            payload = {
                "score": round(score, 4),
                "factual_match": exp_hits == len(expected),
                "rejected_leakage": bool(forb_hits),
                "rationale": f"Matched {exp_hits}/{len(expected)} expected facts; forbidden matches: {forb_hits}",
            }
            content = json.dumps(payload)
            return LLMResponse(
                content=content,
                model_used=model,
                tokens_in=tokens_in,
                tokens_out=estimate_tokens(content),
                cache_read_tokens=cache_read,
                latency_ms=8,
            )

        # 2. Read-only shadow turn
        if "[Shadow Query |" in last_content or "[IPC Shadow Query" in last_content:
            q_match = re.search(r"Query:\s*(.*)", last_content, re.DOTALL)
            question = q_match.group(1).strip() if q_match else last_content
            cycle_match = re.search(r"trigger_cycle:(n_[a-f0-9]+)", question)
            if cycle_match and tools:
                tc = LLMToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=TOOL_REQUEST,
                    arguments={"target_id": cycle_match.group(1), "prompt": "Recursive echo test"},
                )
                return LLMResponse(
                    content="",
                    tool_calls=[tc],
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=28,
                    cache_read_tokens=cache_read,
                    latency_ms=10,
                )

            clean_question = re.sub(r"\[Current date:[^\]]*\]", "", question).strip()
            q_words = {
                w.lower()
                for w in re.findall(r"[A-Za-z0-9_]{3,}", clean_question)
                if w.lower() not in {"what", "which", "about", "this", "that", "from", "with", "decide", "decided", "did", "the", "was", "were", "are", "for", "does", "date"}
            }
            # Chronologically aware extraction: prefer latest settled/revised decisions over
            # superseded earlier proposals (reflecting full-arc conversational understanding!)
            history = [
                (idx, m)
                for idx, m in enumerate(messages[:-1])
                if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()
            ]
            has_revision_or_final = any(
                any(
                    marker in str(m.get("content") or "").lower()
                    for marker in ("final decision", "settled decision", "revision:", "final settled")
                )
                for _, m in history
            )
            scored_lines: list[tuple[float, str]] = []
            total_hist = max(1, len(history))
            for pos, (idx, m) in enumerate(history):
                txt = str(m.get("content") or "").strip()
                t_low = txt.lower()
                if has_revision_or_final and (
                    "initial proposal:" in t_low or "initial plan:" in t_low
                ) and "final" not in t_low and "revision:" not in t_low:
                    continue
                overlap = sum(1 for w in q_words if w in t_low)
                recency_bonus = (pos + 1) / total_hist
                decision_bonus = (
                    2.5
                    if any(
                        mk in t_low
                        for mk in ("final decision", "settled decision", "revision:", "final settled")
                    )
                    else 0.0
                )
                # Strip rejected clause prefixes if a line explicitly marks `| Final decision:`
                clean_txt = txt
                if "Final decision:" in txt:
                    clean_txt = txt.split("Final decision:", 1)[1].strip()
                elif "Revision:" in txt:
                    clean_txt = txt.split("Revision:", 1)[1].strip()
                scored_lines.append((overlap + decision_bonus + recency_bonus, f"[{m['role']}] {clean_txt[:800]}"))
            scored_lines.sort(key=lambda x: x[0], reverse=True)
            # Explicit NOT_FOUND_IN_BRANCH abstention when a specific factual query has zero overlap
            generic_q = {
                "compare", "decision", "decisions", "summary", "status", "details", "picked",
                "selected", "chosen", "exact", "token", "store", "policy", "branch", "confirm",
                "current", "final", "config", "algorithm", "limit", "parameter", "params",
                "design", "engine", "system", "plan", "architecture", "recent", "revisions",
            }
            concrete_q = {
                w for w in q_words if w not in generic_q and not re.fullmatch(r"n_[a-f0-9]{8}", w)
            }
            hist_blob = " ".join(
                str(m.get("content") or "").lower()
                for _, m in history
                if not str(m.get("content") or "").startswith("[Clean Branch Brief]")
            )
            if (
                len(concrete_q) >= 2
                and not any(w in hist_blob for w in concrete_q)
                and not has_revision_or_final
                and "NOT_FOUND_IN_BRANCH" in last_content
            ):
                return LLMResponse(
                    content="NOT_FOUND_IN_BRANCH",
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=4,
                    cache_read_tokens=cache_read,
                    latency_ms=10,
                )
            evidence = "\n".join(line for _, line in scored_lines[:2]) or "(No prior messages in target branch.)"
            ans = f'Shadow-turn answer from target branch context for "{question}":\n{evidence}'
            return LLMResponse(
                content=ans,
                model_used=model,
                tokens_in=tokens_in,
                tokens_out=estimate_tokens(ans),
                cache_read_tokens=cache_read,
                latency_ms=12,
            )

        # 2b. Flat-RAG / Sliding-Window / Multi-Branch Evidence synthesis from provided prompt
        if "[Branch Evidence]" in last_content or "[Verified Shadow Evidence]" in last_content:
            q_match = re.search(r"Question:\s*(.+)", last_content, re.DOTALL)
            question = q_match.group(1).strip() if q_match else last_content
            ev_block = last_content.split("Question:", 1)[0].split("[Branch Evidence]", 1)[-1].strip()
            blocks = [b.strip() for b in re.split(r"\n(?=\[Branch:)", ev_block) if b.strip()]
            # Prefer later-dated / revised branch blocks when reconciling multi-session evidence
            def _block_key(blk: str) -> tuple[str, int]:
                dates = re.findall(r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b", blk)
                rev_bonus = 1 if any(k in blk.lower() for k in ("revision:", "final decision", "migrated", "moved")) else 0
                return (max(dates) if dates else "", rev_bonus)
            blocks.sort(key=_block_key, reverse=True)
            best_block = blocks[0] if blocks else "(No branch evidence.)"
            ans = f'Synthesized branch answer for "{question[:120]}":\n{best_block[:380]}'
            return LLMResponse(
                content=ans,
                model_used=model,
                tokens_in=tokens_in,
                tokens_out=estimate_tokens(ans),
                cache_read_tokens=cache_read,
                latency_ms=10,
            )

        if "[RAG Context]" in last_content:
            q_match = re.search(r"Question:\s*(.+)", last_content, re.DOTALL)
            question = q_match.group(1).strip() if q_match else last_content
            # Extract snippets provided in the RAG block without access to branch chronology
            chunks_block = last_content.split("Question:", 1)[0]
            snippet_lines = [
                ln.strip()
                for ln in chunks_block.splitlines()
                if ln.strip() and not ln.strip().startswith("[RAG Context]")
            ]
            ans = (
                f'Synthesized RAG answer for "{question[:120]}":\n'
                + ("\n".join(snippet_lines[:3]) or "(No relevant chunks retrieved.)")
            )
            return LLMResponse(
                content=ans,
                model_used=model,
                tokens_in=tokens_in,
                tokens_out=estimate_tokens(ans),
                cache_read_tokens=cache_read,
                latency_ms=10,
            )

        if "[Sliding Window Context]" in last_content:
            q_match = re.search(r"Question:\s*(.+)", last_content, re.DOTALL)
            question = q_match.group(1).strip() if q_match else last_content
            q_terms = {w for w in re.findall(r"[a-z0-9_]{3,}", question.lower())}
            # Score only the rolling summary + visible window messages actually passed in `messages`
            scored_win: list[tuple[float, str]] = []
            for idx, m in enumerate(messages[:-1]):
                if m.get("role") not in ("user", "assistant"):
                    continue
                txt = str(m.get("content") or "").strip()
                if not txt:
                    continue
                m_terms = {w for w in re.findall(r"[a-z0-9_]{3,}", txt.lower())}
                overlap = float(len(q_terms & m_terms))
                scored_win.append((overlap + idx * 0.05, txt[:260]))
            scored_win.sort(key=lambda x: x[0], reverse=True)
            best = "\n".join(t for sc, t in scored_win[:2] if sc > 0.5) or "(Not found in sliding window.)"
            ans = f'Sliding-window answer for "{question[:120]}":\n{best}'
            return LLMResponse(
                content=ans,
                model_used=model,
                tokens_in=tokens_in,
                tokens_out=estimate_tokens(ans),
                cache_read_tokens=cache_read,
                latency_ms=10,
            )

        # 3. Tool-result synthesis (supports one or multiple parallel tool results!)
        if last_role == "tool":
            recent_tool_msgs: list[dict[str, Any]] = []
            for m in reversed(messages):
                if m.get("role") == "tool":
                    recent_tool_msgs.append(m)
                else:
                    break
            recent_tool_msgs.reverse()

            sections: list[str] = []
            for tm in recent_tool_msgs:
                tname = tm.get("name") or tm.get("tool_name") or "tool"
                raw_txt = str(tm.get("content") or "{}")
                try:
                    parsed = json.loads(raw_txt)
                except json.JSONDecodeError:
                    parsed = {"raw": raw_txt}

                if tname == TOOL_SPAWN_CHILD:
                    cid = parsed.get("child_id", "child")
                    ctitle = parsed.get("title", cid)
                    mode = parsed.get("context_mode", "fork")
                    if parsed.get("autonomous_result"):
                        sections.append(
                            f"Spawned autonomous child **{ctitle}** (`{cid}`, `{mode}` mode):\n\n{parsed['autonomous_result']}"
                        )
                    else:
                        sections.append(
                            f"Opened child branch **{ctitle}** (`{cid}`, `{mode}` mode). HEAD switched into the new branch."
                        )
                elif tname == TOOL_REQUEST:
                    if parsed.get("error"):
                        sections.append(
                            f"IPC guardrail blocked `{TOOL_REQUEST}()`: **{parsed['error']}** — {parsed.get('detail', '')}."
                        )
                    else:
                        src = parsed.get("target_title") or parsed.get("target_id")
                        mode = parsed.get("mode", "summary")
                        body = parsed.get("answer") or parsed.get("summary") or "(empty)"
                        sections.append(
                            f"Queried **{src}** (`{parsed.get('target_id')}`, mode=`{mode}`, v{parsed.get('summary_version', 0)}):\n\n{body}"
                        )
                elif tname == TOOL_SEND:
                    if parsed.get("finish"):
                        status = parsed.get("handoff_status", "pending_review")
                        if status == "pending_review":
                            sections.append(
                                f"Prepared PR-style handoff to parent `{parsed.get('target_id')}` "
                                f"(summary v{parsed.get('summary_version', 1)})."
                            )
                        else:
                            sections.append(
                                f"Completed branch and committed summary v{parsed.get('summary_version', 1)} "
                                f"to parent `{parsed.get('target_id')}`."
                            )
                    else:
                        sections.append(f"Delivered message to `{parsed.get('target_id')}`.")
                elif tname == TOOL_SEARCH_TREE:
                    hits = parsed.get("hits", [])
                    if not hits:
                        sections.append("No matching nodes or snippets found across the requested scope.")
                    else:
                        lines = [
                            f"- **{h['node_title']}** (`{h['node_id']}`, {h.get('granularity', 'chunk')}): {h['snippet']}"
                            for h in hits
                        ]
                        sections.append("Tree search results:\n" + "\n".join(lines))
                elif tname == TOOL_ASK_TREE:
                    answers = parsed.get("found_answers") or parsed.get("answers") or []
                    if not answers:
                        sections.append("No matching branch answers found.")
                    else:
                        lines = [
                            f"- **{a.get('target_title', a.get('target_id'))}**: {a.get('answer') or a.get('summary') or ''}"
                            for a in answers
                        ]
                        sections.append("Compound ask_tree results:\n" + "\n".join(lines))
                elif tname == TOOL_REFLOG_JUMP:
                    dest_label = parsed.get("to_node_title") or parsed.get("to_node_id")
                    base_line = (
                        f"Switched HEAD to **{dest_label}** (`{parsed.get('to_node_id')}`, "
                        f"{parsed.get('head_ref', 'HEAD@{0}')}, status={parsed.get('dest_status', 'ACTIVE')})."
                    )
                    if parsed.get("follow_up_response"):
                        base_line += f"\n\nDestination response:\n{parsed['follow_up_response']}"
                    elif parsed.get("dest_summary"):
                        base_line += f"\n\nDestination summary: {parsed['dest_summary']}"
                    sections.append(base_line)
                elif tname == TOOL_COMPACT_BRANCH:
                    sections.append(
                        f"Compacted **{parsed.get('from_node_title')}** (`{parsed.get('from_node_id')}`) "
                        f"into continuation branch **{parsed.get('to_node_title')}** (`{parsed.get('to_node_id')}`, "
                        f"{parsed.get('tokens_before')}→{parsed.get('tokens_after')} tok). "
                        f"Previous branch stays verbatim and queryable."
                    )
                else:
                    if parsed.get("error"):
                        sections.append(
                            f"Custom tool `{tname}` blocked/failed: **{parsed['error']}** — {parsed.get('detail', '')}"
                        )
                    else:
                        sections.append(f"Custom tool `{tname}` returned: `{json.dumps(parsed)}`")

            content = "\n\n".join(sections)
            return LLMResponse(
                content=content,
                model_used=model,
                tokens_in=tokens_in,
                tokens_out=estimate_tokens(content),
                cache_read_tokens=cache_read,
                latency_ms=10,
            )

        # 4. Interactive user turn -> emit tool call(s) when requested
        low = user_text.lower()
        available_names = {
            t["function"]["name"] for t in (tools or []) if isinstance(t, dict) and "function" in t
        }
        if tools:
            # Explicit custom tool trigger: `use tool <name> <json_args>`
            custom_match = re.search(r"use tool\s+([A-Za-z0-9_-]+)(?:\s+(\{.*\}))?", user_text)
            if custom_match:
                cname = custom_match.group(1)
                cargs_raw = custom_match.group(2) or "{}"
                try:
                    cargs = json.loads(cargs_raw)
                except json.JSONDecodeError:
                    cargs = {}
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=cname, arguments=cargs)
                    ],
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=24,
                    cache_read_tokens=cache_read,
                    latency_ms=10,
                )

            # Multi-node parallel fan-out request trigger when 2+ node IDs are mentioned
            all_node_ids = list(dict.fromkeys(re.findall(r"\b(n_[a-f0-9]{8})\b", user_text)))
            if (
                all_node_ids
                and TOOL_REQUEST in available_names
                and any(
                    k in low
                    for k in ("request", "ask", "query", "check", "shadow", "summary", "refresh", "compare")
                )
            ):
                tcs: list[LLMToolCall] = []
                for nid in all_node_ids:
                    args: dict[str, Any] = {"target_id": nid}
                    if "summary only" not in low and "cached summary" not in low:
                        args["prompt"] = user_text
                    tcs.append(
                        LLMToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=TOOL_REQUEST, arguments=args)
                    )
                return LLMResponse(
                    content="",
                    tool_calls=tcs,
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=24 * len(tcs),
                    cache_read_tokens=cache_read,
                    latency_ms=12,
                )

            # Explicit non-destructive compact branch trigger
            if TOOL_COMPACT_BRANCH in available_names and any(
                k in low
                for k in (
                    "compact branch",
                    "compact this branch",
                    "compact session",
                    "dialogue_tree_compact_branch",
                )
            ):
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=TOOL_COMPACT_BRANCH,
                            arguments={"keep_recent_turns": 2, "reason": user_text[:80]},
                        )
                    ],
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=26,
                    cache_read_tokens=cache_read,
                    latency_ms=10,
                )

            # Explicit branch-switch / reflog jump trigger
            if TOOL_REFLOG_JUMP in available_names and any(
                k in low
                for k in (
                    "switch branch",
                    "switch back to",
                    "jump to head@{",
                    "jump back to",
                    "dialogue_tree_switch_branch",
                )
            ):
                jump_target: str | None = None
                href_m = re.search(r"(HEAD@\{\d+\})", user_text, flags=re.IGNORECASE)
                nid_m = re.search(r"\b(n_[a-f0-9]{8})\b", user_text)
                if nid_m:
                    jump_target = nid_m.group(1)
                elif href_m:
                    jump_target = href_m.group(1)
                else:
                    # Resolve named branch or "root" from TREE / REFLOG / BRANCH SUMMARIES in ctx_block
                    ctx_nodes = re.findall(r'(n_[a-f0-9]{8})\s+"([^"]+)"', ctx_block)
                    parent_m = re.search(r"parent=(n_[a-f0-9]{8})", ctx_block)
                    if "root" in low:
                        # Prefer root (earliest ancestor / parent) if present
                        for cid, ctitle in ctx_nodes:
                            if any(w in ctitle.lower() for w in ("nebula", "root")):
                                jump_target = cid
                                break
                        if not jump_target and parent_m:
                            jump_target = parent_m.group(1)
                    if not jump_target:
                        for cid, ctitle in ctx_nodes:
                            title_words = [
                                w for w in re.findall(r"[a-z0-9]+", ctitle.lower()) if len(w) >= 4
                            ]
                            if title_words and any(w in low for w in title_words):
                                jump_target = cid
                                break
                    if not jump_target:
                        jump_target = "HEAD@{1}"

                jump_args: dict[str, Any] = {"target": jump_target, "note": user_text[:80]}
                follow_m = re.search(
                    r"\band\s+(?:immediately\s+)?(revise|update|set|confirm|record|change)\b.*",
                    user_text,
                    flags=re.IGNORECASE,
                )
                if follow_m:
                    jump_args["follow_up_prompt"] = follow_m.group(0).lstrip("and ").strip()

                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=TOOL_REFLOG_JUMP,
                            arguments=jump_args,
                        )
                    ],
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=30,
                    cache_read_tokens=cache_read,
                    latency_ms=10,
                )

            # Compound search-and-query trigger (for cross-branch synthesis / verification)
            if TOOL_ASK_TREE in available_names and any(
                k in low
                for k in (
                    "search and query",
                    "assemble the final production configuration",
                    "quick final verification",
                    "dialogue_tree_search_and_query",
                )
            ):
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=TOOL_ASK_TREE,
                            arguments={"query": user_text, "scope": "all", "max_targets": 4},
                        )
                    ],
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=28,
                    cache_read_tokens=cache_read,
                    latency_ms=10,
                )

            if TOOL_SEARCH_TREE in available_names and any(
                k in low
                for k in (
                    "search tree",
                    "search_tree",
                    "search branches",
                    "dialogue_tree_search_branches",
                    "find in tree",
                    "look up across branches",
                )
            ):
                cleaned_q = (
                    re.sub(
                        r"(dialogue_tree_search_branches|search[_\s](tree|branches)|find in tree|look up across branches)[:\s]*",
                        "",
                        user_text,
                        flags=re.IGNORECASE,
                    ).strip()
                    or user_text
                )
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=TOOL_SEARCH_TREE,
                            arguments={"query": cleaned_q, "scope": "all", "top_k": 5},
                        )
                    ],
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=24,
                    cache_read_tokens=cache_read,
                    latency_ms=10,
                )

            if TOOL_SPAWN_CHILD in available_names and any(
                k in low
                for k in (
                    "spawn child",
                    "spawn_child",
                    "create branch",
                    "create a new child branch",
                    "create a clean child branch",
                    "create a clean side branch",
                    "dialogue_tree_create_branch",
                    "spin off a child",
                    "start a tangent",
                    "branch off",
                )
            ):
                mode = "clean" if "clean" in low else ("slice" if "slice" in low else "fork")
                autonomous = "autonomous" in low
                titled_m = re.search(r"titled\s+['\"]([^'\"]+)['\"]", user_text, flags=re.IGNORECASE)
                goal = (
                    re.sub(
                        r"^(please\s+)?(dialogue_tree_create_branch|create[_\s]branch|spawn[_\s]child|spin off a child|start a tangent|branch off)(\s+(to|for|about|into|on))?\s*",
                        "",
                        user_text,
                        flags=re.IGNORECASE,
                    ).strip()
                    or "Explore tangent topic"
                )
                short_title = (
                    titled_m.group(1).strip()
                    if titled_m
                    else (" ".join(goal.split()[:5]).rstrip(".,?!") or "Tangent Branch")
                )
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=TOOL_SPAWN_CHILD,
                            arguments={
                                "title": short_title,
                                "goal": goal,
                                "context_mode": mode,
                                "summary_brief": f"Investigate: {goal}",
                                "autonomous": autonomous,
                            },
                        )
                    ],
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=34,
                    cache_read_tokens=cache_read,
                    latency_ms=12,
                )

            parent_match = re.search(r"parent=(n_[a-f0-9]{8})", ctx_block)
            if (
                parent_match
                and TOOL_SEND in available_names
                and any(
                    k in low
                    for k in ("finish branch", "close branch", "wrap up and return", "send finish", "merge back to parent")
                )
            ):
                parent_id = parent_match.group(1)
                return LLMResponse(
                    content="",
                    tool_calls=[
                        LLMToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=TOOL_SEND,
                            arguments={
                                "target_id": parent_id,
                                "content": f"Branch concluded: {user_text}",
                                "artifacts": [
                                    {
                                        "slug": "branch-decision",
                                        "title": "Branch Conclusion Notes",
                                        "kind": "markdown",
                                        "content": f"# Decision\n\n{user_text}\n",
                                        "summary": "Key conclusion captured at branch handoff.",
                                    }
                                ],
                                "finish": True,
                            },
                        )
                    ],
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=42,
                    cache_read_tokens=cache_read,
                    latency_ms=12,
                )

        # Default conversational response — if the user asks a factual question over prior
        # turns in this node's own prompt, synthesize from the top matching turns in `messages`.
        node_match = re.search(r'branch=(n_[a-f0-9]{8})\s+"([^"]+)"', ctx_block)
        node_label = f"{node_match.group(2)} (`{node_match.group(1)}`)" if node_match else "this branch"
        stale_warn = ""
        if "STALE [" in ctx_block:
            stale_warn = (
                "\n\n> **Note:** A staleness notice appeared in `<dialogue_tree_context>` — "
                f"call `{TOOL_REQUEST}(target_id)` to pull refreshed state."
            )

        is_question = user_text.strip().endswith("?") or low.startswith(("confirm ", "what ", "which "))
        prior_conv = [
            m
            for m in messages[:-1]
            if m.get("role") in ("user", "assistant")
            and str(m.get("content") or "").strip()
            and not str(m.get("content") or "").startswith("Understood.")
        ]
        if is_question and prior_conv:
            q_terms = {
                w
                for w in re.findall(r"[a-z0-9_]{3,}", user_text.lower())
                if w not in {"what", "which", "exact", "did", "the", "for", "and", "our", "confirm", "current", "final", "settle"}
            }
            scored_prev: list[tuple[float, str]] = []
            for idx, m in enumerate(prior_conv):
                txt = str(m.get("content") or "").strip()
                m_terms = {w for w in re.findall(r"[a-z0-9_]{3,}", txt.lower())}
                overlap = float(len(q_terms & m_terms))
                if overlap > 0:
                    scored_prev.append((overlap + idx * 0.02, f"[{m['role']}] {txt[:260]}"))
            if scored_prev:
                scored_prev.sort(key=lambda x: x[0], reverse=True)
                top_lines = "\n".join(line for _, line in scored_prev[:2])
                reply = f"Based on **{node_label}** context:\n{top_lines}{stale_warn}"
                return LLMResponse(
                    content=reply,
                    model_used=model,
                    tokens_in=tokens_in,
                    tokens_out=estimate_tokens(reply),
                    cache_read_tokens=cache_read,
                    latency_ms=max(10, int((time.perf_counter() - t0) * 1000)),
                )

        reply = (
            f"**[Demo Mode · `{model}`]** Recorded in **{node_label}**:\n\n"
            f'Noted: *"{user_text[:280]}"*.{stale_warn}'
        )
        return LLMResponse(
            content=reply,
            model_used=model,
            tokens_in=tokens_in,
            tokens_out=estimate_tokens(reply),
            cache_read_tokens=cache_read,
            latency_ms=max(10, int((time.perf_counter() - t0) * 1000)),
        )

    async def stream_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[LLMStreamEvent]:
        resp = await self.complete(model, messages, tools=tools, temperature=temperature)
        if resp.content:
            for chunk in re.findall(r"\S+\s*", resp.content):
                yield LLMStreamEvent(kind="delta", delta=chunk)
                if self.stream_delay_s > 0:
                    await asyncio.sleep(self.stream_delay_s)
        yield LLMStreamEvent(kind="done", response=resp)
