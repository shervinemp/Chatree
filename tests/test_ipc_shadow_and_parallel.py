from __future__ import annotations

import asyncio
import time

from chatree import ArtifactInput, Chatree, LLMResponse, LLMStreamEvent, LLMToolCall, MockLLMProvider


class SlowParallelProvider(MockLLMProvider):
    """Adds a 60ms delay on shadow turns to prove `asyncio.gather` parallel IPC fan-out."""

    async def complete(self, model, messages, tools=None, temperature=0.2) -> LLMResponse:
        last = str(messages[-1].get("content") or "") if messages else ""
        if "[Shadow Query |" in last or "[IPC Shadow Query" in last:
            await asyncio.sleep(0.06)
        return await super().complete(model, messages, tools=tools, temperature=temperature)

    async def stream_complete(self, model, messages, tools=None, temperature=0.2):
        resp = await self.complete(model, messages, tools=tools, temperature=temperature)
        yield LLMStreamEvent(kind="done", response=resp)


async def test_shadow_turn_is_read_only_and_cached_summary_zero_cost():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("DB Architecture")
        root = await tree.root()
        child = await root.spawn("Schema design", title="DB Schema", mode="clean")
        await ct.repo.append_message(
            child.id, tree.id, "user", "Use pgvector HNSW with m=16 ef_construction=64."
        )
        await ct.repo.append_message(
            child.id, tree.id, "assistant", "Locked in embeddings_v2 using HNSW m=16."
        )

        # Read-only shadow turn
        ans = await root.request(child.id, prompt="What HNSW params did we choose?")
        assert ans["mode"] == "shadow_turn"
        assert ans["mutated"] is False
        assert "HNSW" in ans["answer"]
        assert len(await child.messages()) == 2  # untouched!

        # Cached summary: 1st refreshes (dirty), 2nd is 0-token cache hit!
        s1 = await root.request(child.id, prompt=None)
        assert s1["mode"] == "refreshed_summary"
        assert s1["cache_hit"] is False
        assert s1["summary_version"] == 1

        s2 = await root.request(child.id, prompt=None)
        assert s2["mode"] == "cached_summary"
        assert s2["cache_hit"] is True
        events = await tree.ipc_log()
        latest_summary_ev = next(e for e in events if e.event_type == "request_summary")
        assert latest_summary_ev.tokens_in == 0 and latest_summary_ev.tokens_out == 0


async def test_parallel_ipc_fanout_executes_concurrently():
    async with Chatree.in_memory(llm=SlowParallelProvider(stream_delay_s=0)) as ct:
        tree = await ct.create_tree("Parallel Fan-Out")
        root = await tree.root()

        b1 = await root.spawn("Branch 1", title="Auth", mode="clean", switch_head=False)
        b2 = await root.spawn("Branch 2", title="Cache", mode="clean", switch_head=False)
        b3 = await root.spawn("Branch 3", title="Queue", mode="clean", switch_head=False)
        await ct.repo.append_message(b1.id, tree.id, "user", "Auth uses Ed25519.")
        await ct.repo.append_message(b2.id, tree.id, "user", "Cache uses Redis LRU.")
        await ct.repo.append_message(b3.id, tree.id, "user", "Queue uses NATS JetStream.")

        t0 = time.perf_counter()
        # Mock provider detects all 3 node IDs and emits 3 parallel request() tool calls!
        res = await root.chat(f"Compare {b1.id}, {b2.id}, and {b3.id} decisions")
        elapsed = time.perf_counter() - t0

        assert len(res.ipc_events) == 3
        assert "Ed25519" in res.content
        assert "Redis" in res.content
        assert "JetStream" in res.content
        # 3 * 60ms = 180ms sequential; parallel asyncio.gather finishes well under 155ms!
        assert elapsed < 0.155, f"Expected parallel fan-out < 0.155s, got {elapsed:.3f}s"


async def test_editable_pr_handoff_and_versioned_artifacts():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("PR Handoff", auto_commit_handoffs=False)
        root = await tree.root()
        child = await root.spawn("SQL schema", title="SQL", mode="clean")

        fin = await child.finish(
            summary="Draft summary missing index.",
            artifacts=[
                ArtifactInput(
                    slug="schema.sql",
                    title="Initial Schema",
                    kind="sql",
                    content="CREATE TABLE users (id TEXT PRIMARY KEY);",
                )
            ],
        )
        mb = fin["handoff_pending"]
        assert mb is not None and mb.status == "pending_review"
        assert len(await root.messages()) == 0

        committed_mb, _ = await tree.commit_handoff(
            mb.id,
            summary="Final summary with unique email index.",
            artifacts=[
                ArtifactInput(
                    slug="schema.sql",
                    title="Final Schema",
                    kind="sql",
                    content="CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT UNIQUE);",
                )
            ],
        )
        assert committed_mb.status == "delivered"
        root_msgs = await root.messages()
        assert len(root_msgs) == 1
        assert "schema.sql@v2" in root_msgs[0].content
        assert [a.version for a in await tree.artifacts()] == [2, 1]
