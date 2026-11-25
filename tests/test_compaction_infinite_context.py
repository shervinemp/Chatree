from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chatree import Chatree, ChatreeConfig, GuardrailPolicy, MockLLMProvider
from chatree.cli import run_cli_async
from chatree.types import LLMResponse


class _CompactorKVSpyLLM(MockLLMProvider):
    """Verifies non-destructive compaction reuses canonical `tools` + `tool_choice='none'`."""

    def __init__(self) -> None:
        super().__init__(stream_delay_s=0.0)
        self.compactor_calls: list[dict[str, Any]] = []

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
        if any("[Chatree Compactor]" in str(m.get("content", "")) for m in messages):
            self.compactor_calls.append(
                {"tools": tools, "tool_choice": tool_choice, "n_messages": len(messages)}
            )
        return await super().complete(
            model,
            messages,
            tools=tools,
            temperature=temperature,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            fork_breakpoint_index=fork_breakpoint_index,
        )


@pytest.mark.asyncio
async def test_non_destructive_compaction_chain_and_infinite_ancestor_recall() -> None:
    """
    Verify 3-generation non-destructive compaction chain (B1 -> B2 -> B3):
      - B1's verbatim messages and static KV prefix stay 100% UNTOUCHED after compaction.
      - Compaction preserves canonical `tools` + `tool_choice='none'` (byte-0 KV cache hit!).
      - B2 and B3 inherit compact structured handoffs + recent turns (bounded O(1) context!).
      - From B3 (two compactions later), lossless shadow query / ancestor search retrieves exact
        verbatim secrets buried way back in B1!
      - Revisiting & mutating archived B1 triggers STALE notice on B2 and transitively on B3!
    """
    spy = _CompactorKVSpyLLM()
    compacted_events: list[dict[str, Any]] = []

    async with Chatree.in_memory(llm=spy) as ct:
        @ct.on("branch_compacted")
        def _on_compact(ev: dict[str, Any]) -> None:
            compacted_events.append(ev)

        tree = await ct.create_tree("Infinite Context Project")
        b1 = await tree.root()
        await b1.chat("Early turn 1: encryption vault cipher is AES256_GCM_KMS_V1 with salt rotation 90d.")
        for i in range(6):
            await b1.chat(f"Working note #{i}: " + ("telemetry pipeline details " * 18))

        b1_msgs_before = await b1.messages()
        b1_prefix_before = await ct.context.build_static_prefix(await b1.model())
        b1_tokens_before = await ct.context.estimate_node_context_tokens(await b1.model())

        # Generation 1 -> Generation 2 compaction
        b2, info1 = await b1.compact(keep_recent_turns=2, reason="Hit context budget")
        assert info1["ok"] is True
        assert info1["from_node_id"] == b1.id
        assert info1["to_node_id"] == b2.id
        assert info1["tokens_after"] < info1["tokens_before"]
        assert (await b2.model()).title == "Infinite Context Project (cont. 2)"
        assert (await b2.model()).parent_id == b1.id
        assert (await b2.model()).tool_permission == "read_write"  # Inherited writable perm!
        assert (await tree.head()).id == b2.id
        assert len(compacted_events) == 1

        # B1's verbatim messages and static KV prefix MUST remain 100% untouched!
        assert await b1.messages() == b1_msgs_before
        assert await ct.context.build_static_prefix(await b1.model()) == b1_prefix_before
        assert (await b1.model()).status == "COMPLETED"
        assert (await b1.model()).summary is not None
        assert "OBJECTIVE" in ((await b1.model()).summary or "")

        # Compactor preserved canonical tool schemas + tool_choice="none" for warm KV-cache reuse!
        assert spy.compactor_calls
        assert spy.compactor_calls[-1]["tools"] is not None
        assert spy.compactor_calls[-1]["tool_choice"] == "none"

        # Add more turns in B2, then compact B2 -> B3 (3-generation infinite chain!)
        for i in range(6):
            await b2.chat(f"Phase 2 note #{i}: " + ("queue throughput tuning " * 18))
        b3, info2 = await b2.compact(keep_recent_turns=2)
        assert (await b3.model()).title == "Infinite Context Project (cont. 3)"
        assert (await b3.model()).parent_id == b2.id
        assert info2["tokens_after"] < b1_tokens_before  # Working context stays strictly bounded!

        # From B3, query way back into untouched B1 via ancestor search & lossless shadow turn!
        ask_out = await tree.ask(
            "What encryption vault cipher and salt rotation did we set in early turn 1?",
            scope="ancestors",
            caller_id=b3.id,
        )
        combined = " ".join(str(a.get("answer", "")) for a in ask_out.get("answers", []))
        assert "AES256_GCM_KMS_V1" in combined

        # Direct shadow turn against untouched B1 also works verbatim
        direct = await b3.request(b1.id, prompt="What exact encryption vault cipher was set?")
        assert direct["mode"] in ("shadow_turn", "cached_shadow_turn")
        assert "AES256_GCM_KMS_V1" in direct["answer"]

        # If someone later reopens and revises B1, staleness propagates to B2 -> B3!
        await b1.chat("Revision: upgraded vault cipher to CHACHA20_POLY1305_V2.")
        b2_stale = await b2.stale_dependencies()
        b3_stale = await b3.stale_dependencies()
        assert any(d.source_node_id == b1.id for d in b2_stale)
        assert any(d.source_node_id == b1.id for d in b3_stale)


@pytest.mark.asyncio
async def test_auto_compact_threshold_trigger_and_tool_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    Verify:
      1. `auto_compact_enabled=True` automatically splits a branch when `live_tokens >= threshold`
         BEFORE appending the incoming prompt, landing the new prompt in the continuation node!
      2. Model-invoked `dialogue_tree_compact_branch` tool works interactively.
      3. CLI `chatree compact` works cleanly end-to-end.
    """
    cfg = ChatreeConfig(
        db_path=":memory:",
        guardrails=GuardrailPolicy(
            auto_compact_enabled=True,
            auto_compact_threshold_tokens=320,
            auto_compact_keep_recent_turns=2,
            auto_compact_min_messages=4,
        ),
    )
    async with Chatree.in_memory(config=cfg, llm=MockLLMProvider(stream_delay_s=0.0)) as ct:
        tree = await ct.create_tree("Auto Split Demo")
        root = await tree.root()

        # Grow root beyond 320 tokens (>= 4 messages)
        await root.chat("Early secret anchor: LEGACY_TOKEN_ALPHA_777. " + ("context filler " * 30))
        await root.chat("Second long design turn. " + ("architecture discussion " * 30))

        # Next turn on HEAD automatically triggers non-destructive compaction!
        res = await tree.chat("Now let's work on rate limiting in the fresh session.")
        assert res.compacted_from_node_id == root.id
        assert res.node_id != root.id
        assert res.head_node_id == res.node_id

        # New prompt landed in continuation node; old root stays untouched with LEGACY_TOKEN_ALPHA_777
        new_node = tree.node(res.node_id)
        assert (await new_node.model()).parent_id == root.id
        new_msgs = await new_node.messages()
        assert any("rate limiting" in m.content for m in new_msgs)
        root_msgs = await root.messages()
        assert not any("rate limiting" in m.content for m in root_msgs)
        assert any("LEGACY_TOKEN_ALPHA_777" in m.content for m in root_msgs)

        # Model-triggered `dialogue_tree_compact_branch` tool also works
        tool_res = await tree.chat("Please compact this branch into a clean continuation.")
        assert tool_res.head_node_id != new_node.id
        assert (await tree.node(tool_res.head_node_id).model()).parent_id == new_node.id

    # CLI `chatree compact` check
    db_file = tmp_path / "compact_cli.db"
    assert await run_cli_async(["--db", str(db_file), "init", "CLI Compact Tree"]) == 0
    tree_id = json.loads(capsys.readouterr().out)["id"]
    assert await run_cli_async(["--db", str(db_file), "chat", tree_id, "Anchor TOKEN_CLI_999"]) == 0
    capsys.readouterr()
    assert (
        await run_cli_async(
            ["--db", str(db_file), "compact", tree_id, "--keep-turns", "1", "--reason", "CLI test"]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["compacted"] is True
    assert out["to_node_title"] == "CLI Compact Tree (cont. 2)"
