from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from chatree.server import create_app


def _parse_sse(raw_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        ev_name = "message"
        data_str = "{}"
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:") :].strip()
        events.append((ev_name, json.loads(data_str)))
    return events


@pytest.fixture
async def client(tmp_path):
    app = create_app(db_path=tmp_path / "server_e2e.db")
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


async def test_optional_server_api_sse_and_ergonomics(client: AsyncClient):
    h = await client.get("/api/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    create_res = await client.post("/api/trees", json={"title": "Server E2E", "goal": "Verify adapter"})
    assert create_res.status_code == 201
    snap = create_res.json()
    tree_id = snap["tree"]["id"]
    root_id = snap["tree"]["root_node_id"]

    # SSE turn spawning child
    sse_res = await client.post(
        f"/api/nodes/{root_id}/turns/stream",
        json={"content": "Please branch off to design the auth flow"},
    )
    assert sse_res.status_code == 200
    events = _parse_sse(sse_res.text)
    head_ev = next(data for name, data in events if name == "head_changed")
    child_id = head_ev["to_node_id"]

    # Turn in child + finish -> PR handoff commit
    await client.post(
        f"/api/nodes/{child_id}/turns/stream",
        json={"content": "Use Ed25519 JWT signing keys rotated daily."},
    )
    fin_res = await client.post(
        f"/api/nodes/{child_id}/finish",
        json={"summary": "Ed25519 daily rotation."},
    )
    assert fin_res.status_code == 200
    mb_id = fin_res.json()["handoff_pending"]["id"]

    commit_res = await client.post(
        f"/api/mailbox/{mb_id}/commit",
        json={"summary": "Ed25519 daily rotation with JWKS.", "return_head_to_target": True},
    )
    assert commit_res.status_code == 200

    # Metrics + Markdown export + Hybrid Search
    metrics_res = await client.get(f"/api/trees/{tree_id}/metrics")
    assert metrics_res.status_code == 200
    assert metrics_res.json()["node_count"] == 2

    export_res = await client.get(f"/api/trees/{tree_id}/export")
    assert export_res.status_code == 200
    assert "# Server E2E" in export_res.text

    search_res = await client.get(f"/api/trees/{tree_id}/search", params={"q": "Ed25519"})
    assert search_res.status_code == 200
    assert any(h["node_id"] == child_id for h in search_res.json())
