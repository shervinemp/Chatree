from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from chatree.sdk.client import Chatree
from chatree.server.dtos import (
    AddFactsRequest,
    CreateArtifactRequest,
    CreateTreeRequest,
    PinInvariantRequest,
    ReflogJumpRequest,
    TreeSnapshotResponse,
    UpdateSettingsRequest,
    UpdateTreeRequest,
    build_tree_snapshot,
    get_client,
)
from chatree.types import (
    ArtifactInput,
    ArtifactModel,
    BackgroundJobModel,
    BranchFactModel,
    IpcEventModel,
    ReflogEntryModel,
    RuntimeSettingsModel,
    SearchHit,
    SearchScope,
    TreeInvariantModel,
    TreeMetrics,
    TreeModel,
)

router = APIRouter()


@router.get("/api/trees", response_model=list[TreeModel])
async def list_trees(c: Chatree = Depends(get_client)) -> list[TreeModel]:
    return await c.list_trees()


@router.post("/api/trees", response_model=TreeSnapshotResponse, status_code=201)
async def create_tree(
    body: CreateTreeRequest, c: Chatree = Depends(get_client)
) -> TreeSnapshotResponse:
    dt = await c.create_tree(
        title=body.title,
        goal=body.goal,
        default_model=body.default_model,
        shadow_model=body.shadow_model,
        summary_model=body.summary_model,
        auto_commit_handoffs=body.auto_commit_handoffs,
    )
    return await build_tree_snapshot(c, dt.id)


@router.get("/api/trees/{tree_id}", response_model=TreeSnapshotResponse)
async def get_tree(tree_id: str, c: Chatree = Depends(get_client)) -> TreeSnapshotResponse:
    return await build_tree_snapshot(c, tree_id)


@router.patch("/api/trees/{tree_id}", response_model=TreeModel)
async def update_tree(
    tree_id: str, body: UpdateTreeRequest, c: Chatree = Depends(get_client)
) -> TreeModel:
    updated = await c.repo.update_tree(tree_id, **body.model_dump(exclude_none=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Tree not found")
    return updated


@router.delete("/api/trees/{tree_id}")
async def delete_tree(tree_id: str, c: Chatree = Depends(get_client)) -> dict[str, bool]:
    if not await c.delete_tree(tree_id):
        raise HTTPException(status_code=404, detail="Tree not found")
    return {"deleted": True}


@router.get("/api/trees/{tree_id}/metrics", response_model=TreeMetrics)
async def tree_metrics(tree_id: str, c: Chatree = Depends(get_client)) -> TreeMetrics:
    try:
        dt = await c.get_tree(tree_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return await dt.metrics()


@router.get("/api/trees/{tree_id}/jobs", response_model=list[BackgroundJobModel])
async def list_background_jobs(
    tree_id: str, c: Chatree = Depends(get_client)
) -> list[BackgroundJobModel]:
    if await c.repo.get_tree(tree_id) is None:
        raise HTTPException(status_code=404, detail="Tree not found")
    return await c.actors.list_jobs_persisted(tree_id=tree_id)


@router.get("/api/trees/{tree_id}/export", response_class=PlainTextResponse)
async def export_markdown(
    tree_id: str,
    leaf_node_id: str | None = Query(default=None),
    c: Chatree = Depends(get_client),
) -> str:
    try:
        dt = await c.get_tree(tree_id)
        return await dt.compile_markdown(leaf_node_id=leaf_node_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/trees/{tree_id}/ipc", response_model=list[IpcEventModel])
async def list_ipc(
    tree_id: str, limit: int = Query(default=100, ge=1, le=500), c: Chatree = Depends(get_client)
) -> list[IpcEventModel]:
    return await c.repo.list_ipc_events(tree_id, limit=limit)


@router.get("/api/trees/{tree_id}/search", response_model=list[SearchHit])
async def search_tree(
    tree_id: str,
    q: str = Query(..., min_length=1),
    scope: SearchScope = Query(default="all"),
    node_id: str | None = Query(default=None),
    top_k: int = Query(default=8, ge=1, le=30),
    c: Chatree = Depends(get_client),
) -> list[SearchHit]:
    curr = await c.repo.get_node(node_id) if node_id else None
    return await c.rag.search(tree_id=tree_id, query=q, scope=scope, current_node=curr, top_k=top_k)


@router.get("/api/trees/{tree_id}/reflog", response_model=list[ReflogEntryModel])
async def list_reflog(
    tree_id: str, limit: int = Query(default=100, ge=1, le=500), c: Chatree = Depends(get_client)
) -> list[ReflogEntryModel]:
    return await c.repo.list_reflog(tree_id, limit=limit)


@router.post("/api/trees/{tree_id}/reflog/jump")
async def jump_reflog(
    tree_id: str, body: ReflogJumpRequest, c: Chatree = Depends(get_client)
) -> dict[str, Any]:
    try:
        dest, entry, cursor = await c.reflog.jump(tree_id, body.target, note=body.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "head_node": dest.model_dump(),
        "reflog_entry": entry.model_dump() if entry else None,
        "reflog_cursor": cursor,
    }


@router.get("/api/trees/{tree_id}/artifacts", response_model=list[ArtifactModel])
async def list_artifacts(
    tree_id: str, c: Chatree = Depends(get_client)
) -> list[ArtifactModel]:
    return await c.repo.list_artifacts(tree_id)


@router.post("/api/trees/{tree_id}/artifacts", response_model=ArtifactModel, status_code=201)
async def create_artifact(
    tree_id: str, body: CreateArtifactRequest, c: Chatree = Depends(get_client)
) -> ArtifactModel:
    if await c.repo.get_tree(tree_id) is None:
        raise HTTPException(status_code=404, detail="Tree not found")
    return await c.repo.create_artifact(
        tree_id,
        ArtifactInput(
            slug=body.slug,
            title=body.title,
            kind=body.kind,
            content=body.content,
            summary=body.summary,
        ),
        created_by_node_id=body.created_by_node_id,
    )


# --- Tree-Level PINNED Invariants & Tier-2.5 Atomic Temporal Facts ---


@router.get("/api/trees/{tree_id}/invariants", response_model=list[TreeInvariantModel])
async def list_invariants(
    tree_id: str, c: Chatree = Depends(get_client)
) -> list[TreeInvariantModel]:
    if await c.repo.get_tree(tree_id) is None:
        raise HTTPException(status_code=404, detail="Tree not found")
    return await c.repo.list_tree_invariants(tree_id)


@router.post("/api/trees/{tree_id}/invariants", response_model=TreeInvariantModel, status_code=201)
async def pin_invariant(
    tree_id: str, body: PinInvariantRequest, c: Chatree = Depends(get_client)
) -> TreeInvariantModel:
    try:
        dt = await c.get_tree(tree_id)
        return await dt.pin(body.key, body.value, source_node_id=body.source_node_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/trees/{tree_id}/invariants/{key}")
async def unpin_invariant(
    tree_id: str, key: str, c: Chatree = Depends(get_client)
) -> dict[str, Any]:
    if await c.repo.get_tree(tree_id) is None:
        raise HTTPException(status_code=404, detail="Tree not found")
    removed = await c.repo.delete_tree_invariant(tree_id, key)
    return {"key": key, "removed": removed}


@router.get("/api/trees/{tree_id}/facts", response_model=list[BranchFactModel])
async def list_facts(
    tree_id: str,
    node_id: str | None = Query(default=None),
    include_archived: bool = Query(default=False),
    c: Chatree = Depends(get_client),
) -> list[BranchFactModel]:
    if await c.repo.get_tree(tree_id) is None:
        raise HTTPException(status_code=404, detail="Tree not found")
    return await c.repo.list_branch_facts(
        tree_id, node_id=node_id, active_only=True, exclude_archived=not include_archived
    )


@router.post("/api/trees/{tree_id}/facts", response_model=list[BranchFactModel], status_code=201)
async def add_facts(
    tree_id: str, body: AddFactsRequest, c: Chatree = Depends(get_client)
) -> list[BranchFactModel]:
    try:
        dt = await c.get_tree(tree_id)
        return await dt.add_facts(body.node_id, body.facts, source_msg_seq=body.source_msg_seq)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/settings", response_model=RuntimeSettingsModel)
async def get_settings(c: Chatree = Depends(get_client)) -> RuntimeSettingsModel:
    return await c.get_runtime_settings()


@router.put("/api/settings", response_model=RuntimeSettingsModel)
async def update_settings(
    body: UpdateSettingsRequest, c: Chatree = Depends(get_client)
) -> RuntimeSettingsModel:
    return await c.update_runtime_settings(body.model_dump(exclude_unset=True))
