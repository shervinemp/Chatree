from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from chatree.providers.openai_compat import CallableLLMProvider
from chatree.sdk.client import Chatree


@pytest.mark.asyncio
async def test_node_kind_and_head_only_reflog_invariant() -> None:
    """
    Verify Architecture #3 core invariants:
      - Interactive spawns (`switch_head=True`) create `node_kind='chat'` and enter `reflog`.
      - Sync sub-agents (`autonomous=True`), async `spawn_background`, and `switch_head=False`
        spawns NEVER pollute `reflog` (strict HEAD-only reflog), while ALL spawns log to `ipc_events`.
      - `reflog` Back/Forward steps strictly across interactive chat branches.
    """
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Unified Actor Tree")
        root = await tree.root()
        await root.chat("Hard constraint: LATENCY_BUDGET_50MS.")

        # 1. Interactive tangent -> chat + HEAD moves + enters reflog
        auth = await root.spawn("Design JWT rotation", title="Auth Design", mode="clean")
        assert (await auth.model()).node_kind == "chat"
        assert (await tree.model()).head_node_id == auth.id
        reflog_after_auth = await tree.reflog()
        assert [e.to_node_id for e in reflog_after_auth[:2]] == [auth.id, root.id]

        # 2. Synchronous autonomous sub-agent -> subagent, HEAD stays on auth, reflog unchanged
        worker = await auth.spawn(
            "Verify ED25519 key size",
            title="Graph RAG Probe",
            mode="clean",
            autonomous=True,
        )
        worker_m = await worker.model()
        assert worker_m.node_kind == "subagent"
        assert worker_m.status == "COMPLETED"
        assert (await tree.model()).head_node_id == auth.id
        assert len(await tree.reflog()) == len(reflog_after_auth)

        # 3. Non-HEAD pre-created chat branch -> chat, HEAD unchanged, reflog unchanged
        queued = await auth.spawn(
            "Draft billing webhook", title="Billing Queued", mode="clean", switch_head=False
        )
        assert (await queued.model()).node_kind == "chat"
        assert (await tree.model()).head_node_id == auth.id
        assert len(await tree.reflog()) == len(reflog_after_auth)

        # 4. Async background sub-agent -> subagent, HEAD unchanged, reflog unchanged, logged in ipc_events
        bg_node, job = await auth.spawn_background(
            "Audit rate-limit token bucket", title="Rate Limit Auditor", mode="clean"
        )
        await ct.actors.wait(job.job_id, timeout=10.0)
        bg_m = await bg_node.model()
        assert bg_m.node_kind == "subagent"
        assert bg_m.status == "COMPLETED"
        assert (await tree.model()).head_node_id == auth.id
        assert len(await tree.reflog()) == len(reflog_after_auth)

        # All spawns (interactive + autonomous + queued + background) appear in ipc_events
        spawn_events = [e for e in await tree.ipc_log() if e.event_type == "spawn"]
        spawned_ids = {e.to_node_id for e in spawn_events}
        assert {auth.id, worker.id, queued.id, bg_node.id}.issubset(spawned_ids)

        # Single-table projections cleanly separate active chat topics vs sub-agents
        active_chat_ids = {n.id for n in await tree.active_chat_nodes()}
        subagent_ids = {n.id for n in await tree.subagents()}
        assert {root.id, auth.id, queued.id} == active_chat_ids
        assert {worker.id, bg_node.id} == subagent_ids

        # Pressing Back steps directly from Auth -> Root (never into any subagent!)
        back_node = await tree.jump("back")
        assert back_node.id == root.id


@pytest.mark.asyncio
async def test_subagent_auto_promotes_on_human_chat_but_not_shadow_query() -> None:
    """Shadow queries and passive inspection keep `node_kind='subagent'`; interactive chat auto-promotes to `'chat'`."""
    promoted_events: list[dict[str, Any]] = []

    async with Chatree.in_memory() as ct:
        ct.hooks.register("subagent_promoted", lambda payload: promoted_events.append(payload))
        tree = await ct.create_tree("Promotion Test")
        root = await tree.root()

        sub = await root.spawn(
            "Summarize token refresh invariants: TOKEN_ROTATION_60S",
            title="Token Worker",
            mode="clean",
            autonomous=True,
        )
        assert (await sub.model()).node_kind == "subagent"

        # Read-only shadow query must NOT promote subagent
        shadow = await root.request(sub.id, prompt="What token rotation setting is mentioned?")
        assert "answer" in shadow
        assert (await sub.model()).node_kind == "subagent"
        assert not promoted_events

        # Explicit jump to inspect without sending a chat message keeps it as subagent
        await tree.jump(sub.id, note="Inspect worker transcript")
        assert (await sub.model()).node_kind == "subagent"
        assert not promoted_events

        # Sending an interactive human chat turn inside it auto-promotes it to a first-class chat branch
        await sub.chat("Let's expand this into our full auth refresh spec.")
        assert (await sub.model()).node_kind == "chat"
        assert len(promoted_events) == 1
        assert promoted_events[0]["node_id"] == sub.id


@pytest.mark.asyncio
async def test_mru_deduplicated_reflog_envelope_and_running_subagent_indicator() -> None:
    """
    Verify `<dialogue_tree_context>`:
      - Deduplicates ping-pong REFLOG repeats while keeping valid exact `HEAD@{k}` indices.
      - Tags `[subagent]` nodes in TREE & BRANCH SUMMARIES.
      - Surfaces `SUBAGENTS [running` ONLY while a background worker is executing (0 tokens once done).
    """
    gate = asyncio.Event()

    async def slow_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        last_txt = str(messages[-1].get("content") or "")
        if "Autonomous Background Sub-Agent Task" in last_txt:
            await gate.wait()
            return "GRAPH_EDGE_VERIFIED: ServiceA -> ServiceB"
        return "Acknowledged."

    llm = CallableLLMProvider(slow_fn)
    async with Chatree.in_memory(llm=llm) as ct:
        tree = await ct.create_tree("Envelope Polish")
        root = await tree.root()
        auth = await root.spawn("Auth tangent", title="Auth", mode="clean")
        cache = await root.spawn("Cache tangent", title="Cache", mode="clean")

        # Ping-pong Root <-> Auth multiple times before jumping to Cache
        await tree.jump(root.id)
        await tree.jump(auth.id)
        await tree.jump(root.id)
        await tree.jump(auth.id)

        # Spawn an in-flight slow background sub-agent under Auth
        bg_node, job = await auth.spawn_background("Trace graph edges", title="Graph Worker", mode="clean")

        t_model = await tree.model()
        auth_model = await auth.model()
        env_running, _, _ = await ct.context.build_ephemeral_envelope(t_model, auth_model)

        # TREE must tag the subagent child
        assert f'{bg_node.id} "Graph Worker" [subagent]' in env_running
        # SUBAGENTS line must appear while running
        assert "SUBAGENTS [running" in env_running
        assert bg_node.id in env_running

        # REFLOG must deduplicate ping-pong repeats and include Cache without wasting slots!
        reflog_line = next(ln for ln in env_running.splitlines() if ln.startswith("REFLOG "))
        assert reflog_line.count(auth.id) == 1
        assert reflog_line.count(root.id) == 1
        assert cache.id in reflog_line

        # Release background job -> SUBAGENTS line vanishes cleanly (0 tokens)
        gate.set()
        await ct.actors.wait(job.job_id, timeout=10.0)
        env_after, _, _ = await ct.context.build_ephemeral_envelope(
            await tree.model(), await auth.model()
        )
        assert "SUBAGENTS [running" not in env_after


@pytest.mark.asyncio
async def test_supervisor_cascade_cancels_running_subagents_on_archive_and_portable_preserves_node_kind(
    tmp_path: Path,
) -> None:
    """
    Verify:
      - Archiving a branch cancels any in-flight background sub-agents under it (supervisor cascade).
      - `.dialogue/` v3 export/import losslessly preserves `node_kind` (`chat` vs `subagent`).
    """
    started = asyncio.Event()
    never_release = asyncio.Event()

    async def blocking_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        last_txt = str(messages[-1].get("content") or "")
        if "Autonomous Background Sub-Agent Task" in last_txt:
            started.set()
            await never_release.wait()
            return "Should never complete"
        return "Done."

    async with Chatree.in_memory(llm=CallableLLMProvider(blocking_fn)) as ct:
        tree = await ct.create_tree("Cascade & Portable")
        root = await tree.root()
        proto = await root.spawn("Speculative GraphQL prototype", title="GraphQL Proto", mode="clean")
        bg_node, job = await proto.spawn_background(
            "Benchmark 24-hop GraphQL fanout", title="GraphQL Bench", mode="clean"
        )
        await asyncio.wait_for(started.wait(), timeout=5.0)
        assert ct.actors.get_job(job.job_id) is not None
        assert ct.actors.get_job(job.job_id).status == "running"  # type: ignore[union-attr]

        # Archiving parent branch cascades cancellation immediately to its running background sub-agent
        arch_res = await proto.archive(reason="Violates latency budget")
        assert arch_res["ok"] is True
        assert job.job_id in arch_res["cancelled_background_jobs"]
        cancelled_job = ct.actors.get_job(job.job_id)
        assert cancelled_job is not None and cancelled_job.status == "cancelled"
        assert (await tree.model()).head_node_id == root.id

        # Export & import into a brand-new SQLite store -> node_kind preserved losslessly
        export_dir = tmp_path / "portable_actor_tree"
        await ct.export_directory(tree.id, export_dir)

        async with Chatree.in_memory() as ct2:
            restored = await ct2.import_directory(export_dir)
            restored_by_id = {n.id: n for n in await restored.nodes()}
            assert restored_by_id[root.id].node_kind == "chat"
            assert restored_by_id[proto.id].node_kind == "chat"
            assert restored_by_id[bg_node.id].node_kind == "subagent"
