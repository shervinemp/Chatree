from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from chatree import DEFAULT_SESSION_ID, Chatree
from chatree.server.app import create_app


@pytest.mark.asyncio
async def test_multi_session_visitors_independent_heads_reflog_and_back_forward() -> None:
    """
    Multiple `TreeSession` visitors walking one shared `DialogueTree`:
      - Each session maintains its own independent `HEAD`, `reflog`, and `Back`/`Forward`.
      - All sessions share the exact same underlying nodes, artifacts, and shadow-query provenance.
      - `tree.reflog(session_id=None)` exposes the unified causal walk across all visitors.
    """
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Multi-Session Platform")
        root = await tree.root()
        await root.chat("Platform anchor: SLO_P99_100MS.")

        backend = tree.session(DEFAULT_SESSION_ID)
        frontend = await tree.create_session("Frontend Tab", start_node_id=root.id)

        sessions = await tree.sessions()
        assert {s.id for s in sessions} == {DEFAULT_SESSION_ID, frontend.id}

        # Backend walks Root -> Auth -> Cache
        auth = await backend.spawn("Design JWT rotation", title="Auth Design", mode="clean")
        await auth.chat("Settled on ED25519_SESSION_KEY and JWT_TTL_5M.")
        cache = await backend.spawn("Design Redis cache", title="Cache Design", mode="clean")
        await cache.chat("Use VALKEY_CLUSTER_V8.")

        # Frontend walks Root -> UI (must NOT touch backend's HEAD or reflog!)
        ui = await frontend.spawn("Build Graph Explorer", title="UI Design", mode="clean")
        await ui.chat("Render tree outline with React 19.")

        assert (await backend.head()).id == cache.id
        assert (await frontend.head()).id == ui.id
        # Primary default session stays synced with TreeModel.head_node_id
        assert (await tree.model()).head_node_id == cache.id

        # Each visitor's reflog contains strictly its own walk
        backend_rf = await backend.reflog()
        frontend_rf = await frontend.reflog()
        assert [e.to_node_id for e in backend_rf] == [cache.id, auth.id, root.id]
        assert [e.to_node_id for e in frontend_rf] == [ui.id, root.id]

        # Back/Forward stays 100% isolated per visitor
        assert (await backend.back()).id == auth.id
        assert (await frontend.head()).id == ui.id
        assert (await frontend.back()).id == root.id
        assert (await backend.head()).id == auth.id
        assert (await backend.forward()).id == cache.id

        # Frontend can shadow-query Backend's branch across the shared tree without moving either HEAD
        shadow = await frontend.request(auth.id, prompt="What JWT TTL did Auth settle on?")
        assert "JWT_TTL_5M" in shadow["answer"]
        assert (await frontend.head()).id == root.id
        assert (await backend.head()).id == cache.id

        # Tree-wide reflog view (`session_id=None`) returns causal union of both visitors
        all_rf = await tree.reflog(session_id=None)
        assert {e.session_id for e in all_rf} == {DEFAULT_SESSION_ID, frontend.id}
        assert len(all_rf) == len(backend_rf) + len(frontend_rf)


@pytest.mark.asyncio
async def test_session_scoped_orientation_cues_and_other_visitors_envelope() -> None:
    """
    Verify:
      - `SESSIONS [other visitors` is omitted (0 tokens) in single-session trees.
      - Appears when multiple visitors share the tree (with `(SAME NODE)` tag when overlapping).
      - Orientation cues are strictly session-scoped so Session B never consumes Session A's cues.
    """
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Session Cues")
        root = await tree.root()

        # Single-session tree: 0 extra tokens for SESSIONS line
        env_single, _, _ = await ct.context.build_ephemeral_envelope(
            await tree.model(), await root.model(), session_id=DEFAULT_SESSION_ID
        )
        assert "SESSIONS [other visitors" not in env_single

        # Add a second visitor standing on the same node -> SESSIONS line surfaces with (SAME NODE)
        agent_sess = await tree.create_session(
            "Agent Walker", kind="agent", start_node_id=root.id
        )
        env_multi, _, _ = await ct.context.build_ephemeral_envelope(
            await tree.model(), await root.model(), session_id=DEFAULT_SESSION_ID
        )
        assert "SESSIONS [other visitors" in env_multi
        assert '"Agent Walker"' in env_multi
        assert "(SAME NODE)" in env_multi

        # Spawn a branch in default session (creates an unconsumed orientation cue for default)
        branch = await tree.spawn("Design API", title="API", mode="clean")
        default_cues = await ct.repo.get_unconsumed_orientation_cues(
            tree.id, branch.id, session_id=DEFAULT_SESSION_ID
        )
        agent_cues = await ct.repo.get_unconsumed_orientation_cues(
            tree.id, branch.id, session_id=agent_sess.id
        )
        assert len(default_cues) == 1
        assert len(agent_cues) == 0


@pytest.mark.asyncio
async def test_archive_rescues_all_stranded_sessions_and_portable_roundtrips_sessions(
    tmp_path: Path,
) -> None:
    """
    Verify:
      - Archiving a branch moves ALL sessions currently standing on it cleanly back to parent.
      - `.dialogue/` v3 export/import losslessly preserves all sessions and their per-session reflogs.
    """
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Archive & Portable Sessions")
        root = await tree.root()
        proto = await tree.spawn("Dead-end prototype", title="Dead Proto", mode="clean")
        other = await tree.create_session("Reviewer", start_node_id=proto.id)
        assert (await tree.head()).id == proto.id
        assert (await other.head()).id == proto.id

        # Archiving proto moves BOTH default and Reviewer sessions cleanly back to root
        await tree.archive(proto.id, reason="Scrapped")
        assert (await tree.head()).id == root.id
        assert (await other.head()).id == root.id

        # Move Reviewer onto a fresh node before exporting so both sessions have distinct heads
        docs = await other.spawn("Write docs", title="Docs", mode="clean")
        assert (await other.head()).id == docs.id
        assert (await tree.head()).id == root.id

        export_dir = tmp_path / "multi_session_dialogue"
        await ct.export_directory(tree.id, export_dir)

        async with Chatree.in_memory() as ct2:
            restored = await ct2.import_directory(export_dir)
            restored_sessions = {s.id: s for s in await restored.sessions()}
            assert set(restored_sessions.keys()) == {DEFAULT_SESSION_ID, other.id}
            assert restored_sessions[DEFAULT_SESSION_ID].head_node_id == root.id
            assert restored_sessions[other.id].head_node_id == docs.id
            assert (await restored.session(other.id).reflog())[0].to_node_id == docs.id


@pytest.mark.asyncio
async def test_sessions_rest_api_endpoints() -> None:
    """End-to-end verification of REST session CRUD + session-scoped snapshot/jump/spawn."""
    async with Chatree.in_memory() as ct:
        app = create_app(client=ct)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r_tree = await client.post("/api/trees", json={"title": "API Sessions"})
                assert r_tree.status_code == 201
                snap0 = r_tree.json()
                tree_id = snap0["tree"]["id"]
                root_id = snap0["tree"]["root_node_id"]
                assert snap0["active_session_id"] == DEFAULT_SESSION_ID
                assert len(snap0["sessions"]) == 1

                # Create second visitor session
                r_s2 = await client.post(
                    f"/api/trees/{tree_id}/sessions",
                    json={"title": "Mobile Client", "kind": "human", "start_node_id": root_id},
                )
                assert r_s2.status_code == 201
                s2_id = r_s2.json()["id"]

                # Spawn a branch inside session 2 -> default session HEAD stays on root!
                r_sp = await client.post(
                    f"/api/nodes/{root_id}/spawn",
                    json={"goal": "iOS offline cache", "title": "iOS Cache", "session_id": s2_id},
                )
                assert r_sp.status_code == 200
                ios_id = r_sp.json()["child"]["id"]

                snap_default = (await client.get(f"/api/trees/{tree_id}?session_id=default")).json()
                snap_s2 = (await client.get(f"/api/trees/{tree_id}?session_id={s2_id}")).json()
                assert snap_default["tree"]["head_node_id"] == root_id
                assert len(snap_default["reflog"]) == 1
                assert len(snap_s2["reflog"]) == 2
                assert snap_s2["reflog"][0]["to_node_id"] == ios_id

                # Back in session 2 returns session 2 to root without touching default
                r_back = await client.post(
                    f"/api/trees/{tree_id}/reflog/jump",
                    json={"target": "back", "session_id": s2_id},
                )
                assert r_back.status_code == 200
                assert r_back.json()["head_node"]["id"] == root_id

                # Cannot delete default session, CAN delete s2
                assert (await client.delete(f"/api/trees/{tree_id}/sessions/default")).status_code == 400
                assert (await client.delete(f"/api/trees/{tree_id}/sessions/{s2_id}")).status_code == 200
