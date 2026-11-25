from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
import hmac
from pydantic import BaseModel, Field

from chatree.config import ChatreeConfig, ToolPermission
from chatree.sdk.client import Chatree
from chatree.types import (
    ArtifactInput,
    ArtifactModel,
    BackgroundJobModel,
    BranchComparison,
    ChildContextMode,
    IpcEventModel,
    MailboxItemModel,
    MessageModel,
    NodeModel,
    NodeStatus,
    ReflogEntryModel,
    RuntimeSettingsModel,
    SearchHit,
    SearchScope,
    TreeMetrics,
    TreeModel,
)


# --- HTTP DTOs ---


class TreeSnapshotResponse(BaseModel):
    tree: TreeModel
    nodes: list[NodeModel]
    reflog: list[ReflogEntryModel]
    pending_handoffs: list[MailboxItemModel]
    artifacts: list[ArtifactModel]
    recent_ipc: list[IpcEventModel]


class NodeDetailResponse(BaseModel):
    node: NodeModel
    ancestors: list[NodeModel]
    inherited_messages: list[MessageModel]
    messages: list[MessageModel]
    pending_handoffs: list[MailboxItemModel]
    estimated_context_tokens: int


class CreateTreeRequest(BaseModel):
    title: str = Field(default="New Dialogue Tree", min_length=1, max_length=200)
    goal: str = ""
    default_model: str | None = None
    shadow_model: str | None = None
    summary_model: str | None = None
    auto_commit_handoffs: bool | None = None


class UpdateTreeRequest(BaseModel):
    title: str | None = None
    default_model: str | None = None
    shadow_model: str | None = None
    summary_model: str | None = None
    auto_commit_handoffs: bool | None = None


class CreateTurnRequest(BaseModel):
    content: str = Field(..., min_length=1)
    model_override: str | None = None
    max_hop_depth: int | None = Field(default=None, ge=1, le=16)
    token_budget: int | None = Field(default=None, ge=1000, le=500_000)


class SpawnNodeRequest(BaseModel):
    title: str | None = None
    goal: str = Field(..., min_length=1)
    context_mode: ChildContextMode = "fork"
    summary_brief: str = ""
    slice_turn_count: int = Field(default=6, ge=1, le=50)
    model_override: str | None = None
    tool_permission: ToolPermission | None = None
    switch_head: bool = True


class FinishNodeRequest(BaseModel):
    summary: str | None = None
    artifacts: list[ArtifactInput] = Field(default_factory=list)
    auto_commit: bool | None = None
    return_to_parent: bool = True


class UpdateNodeRequest(BaseModel):
    title: str | None = None
    goal: str | None = None
    model_override: str | None = None
    tool_permission: ToolPermission | None = None
    summary: str | None = None
    status: NodeStatus | None = None


class DirectRequestPayload(BaseModel):
    target_id: str
    prompt: str | None = None
    mutate: bool = False


class AskTreePayload(BaseModel):
    query: str = Field(..., min_length=1)
    scope: SearchScope = "all"
    max_targets: int = Field(default=2, ge=1, le=5)


class SpawnBackgroundRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    title: str | None = None
    context_mode: ChildContextMode = "clean"
    summary_brief: str = ""
    slice_turn_count: int = Field(default=6, ge=1, le=50)
    model_override: str | None = None
    tool_permission: ToolPermission | None = None
    max_rounds: int = Field(default=3, ge=1, le=6)
    token_budget: int | None = Field(default=None, ge=1000, le=500_000)


class CommitHandoffRequest(BaseModel):
    summary: str | None = None
    artifacts: list[ArtifactInput] | None = None
    return_head_to_target: bool = True


class ReflogJumpRequest(BaseModel):
    target: str
    note: str = ""


class CreateArtifactRequest(BaseModel):
    slug: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=200)
    kind: str = Field(default="markdown", max_length=40)
    content: str
    summary: str = ""
    created_by_node_id: str | None = None


class ExtractTangentRequest(BaseModel):
    seq_start: int = Field(..., ge=1)
    seq_end: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    goal: str = ""
    summary: str | None = None


class ForkEditRequest(BaseModel):
    at_seq: int = Field(..., ge=1)
    title: str | None = None
    switch_head: bool = True


class UpdateSettingsRequest(BaseModel):
    default_model: str | None = None
    shadow_model: str | None = None
    summary_model: str | None = None
    api_base: str | None = None
    max_hop_depth: int | None = Field(default=None, ge=1, le=16)
    token_budget: int | None = Field(default=None, ge=1000, le=500_000)
    auto_commit_handoffs: bool | None = None
    api_keys: dict[str, str] | None = None


def _get_client(request: Request) -> Chatree:
    return request.app.state.chatree


async def _snapshot(ct: Chatree, tree_id: str) -> TreeSnapshotResponse:
    tree = await ct.repo.get_tree(tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Tree not found")
    return TreeSnapshotResponse(
        tree=tree,
        nodes=await ct.repo.list_nodes(tree_id),
        reflog=await ct.repo.list_reflog(tree_id, limit=100),
        pending_handoffs=await ct.repo.list_mailbox_items(tree_id, status="pending_review"),
        artifacts=await ct.repo.list_artifacts(tree_id),
        recent_ipc=await ct.repo.list_ipc_events(tree_id, limit=100),
    )


def create_app(
    db_path: str | Path | None = None,
    *,
    config: ChatreeConfig | None = None,
    client: Chatree | None = None,
    api_token: str | None = None,
) -> FastAPI:
    """Create the optional FastAPI REST + SSE adapter wrapping a `Chatree` library instance."""
    ct = client or Chatree(db_path=db_path, config=config)
    owns_client = client is None
    effective_token = api_token if api_token is not None else ct.config.api_token

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if owns_client:
            await ct.startup()
        app.state.chatree = ct
        try:
            yield
        finally:
            if owns_client:
                await ct.close()

    app = FastAPI(
        title="Chatree Server",
        version="0.2.0",
        description="Optional HTTP/SSE adapter wrapping the Chatree dialogue-tree Python library.",
        lifespan=lifespan,
    )
    # Explicit origin allowlist only (never wildcard `*` with credentials on a local mutating API)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ct.config.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if effective_token:
        @app.middleware("http")
        async def _bearer_auth(request: Request, call_next: Any) -> Any:
            if (
                request.method == "OPTIONS"
                or request.url.path == "/api/health"
                or not request.url.path.startswith("/api/")
            ):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            prefix = "Bearer "
            supplied = auth[len(prefix) :] if auth.startswith(prefix) else ""
            if not supplied or not hmac.compare_digest(supplied, effective_token):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unauthorized: missing or invalid Bearer token."},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await call_next(request)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "chatree",
            "sqlite_vec": ct.db.has_sqlite_vec,
        }

    # --- Trees ---

    @app.get("/api/trees", response_model=list[TreeModel])
    async def list_trees(c: Chatree = Depends(_get_client)) -> list[TreeModel]:
        return await c.list_trees()

    @app.post("/api/trees", response_model=TreeSnapshotResponse, status_code=201)
    async def create_tree(
        body: CreateTreeRequest, c: Chatree = Depends(_get_client)
    ) -> TreeSnapshotResponse:
        dt = await c.create_tree(
            title=body.title,
            goal=body.goal,
            default_model=body.default_model,
            shadow_model=body.shadow_model,
            summary_model=body.summary_model,
            auto_commit_handoffs=body.auto_commit_handoffs,
        )
        return await _snapshot(c, dt.id)

    @app.get("/api/trees/{tree_id}", response_model=TreeSnapshotResponse)
    async def get_tree(tree_id: str, c: Chatree = Depends(_get_client)) -> TreeSnapshotResponse:
        return await _snapshot(c, tree_id)

    @app.patch("/api/trees/{tree_id}", response_model=TreeModel)
    async def update_tree(
        tree_id: str, body: UpdateTreeRequest, c: Chatree = Depends(_get_client)
    ) -> TreeModel:
        updated = await c.repo.update_tree(tree_id, **body.model_dump(exclude_none=True))
        if updated is None:
            raise HTTPException(status_code=404, detail="Tree not found")
        return updated

    @app.delete("/api/trees/{tree_id}")
    async def delete_tree(tree_id: str, c: Chatree = Depends(_get_client)) -> dict[str, bool]:
        if not await c.delete_tree(tree_id):
            raise HTTPException(status_code=404, detail="Tree not found")
        return {"deleted": True}

    @app.get("/api/trees/{tree_id}/metrics", response_model=TreeMetrics)
    async def tree_metrics(tree_id: str, c: Chatree = Depends(_get_client)) -> TreeMetrics:
        try:
            dt = await c.get_tree(tree_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return await dt.metrics()

    @app.get("/api/trees/{tree_id}/jobs", response_model=list[BackgroundJobModel])
    async def list_background_jobs(
        tree_id: str, c: Chatree = Depends(_get_client)
    ) -> list[BackgroundJobModel]:
        if await c.repo.get_tree(tree_id) is None:
            raise HTTPException(status_code=404, detail="Tree not found")
        return await c.actors.list_jobs_persisted(tree_id=tree_id)

    @app.get("/api/trees/{tree_id}/export", response_class=PlainTextResponse)
    async def export_markdown(
        tree_id: str,
        leaf_node_id: str | None = Query(default=None),
        c: Chatree = Depends(_get_client),
    ) -> str:
        try:
            dt = await c.get_tree(tree_id)
            return await dt.compile_markdown(leaf_node_id=leaf_node_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # --- Nodes & Streaming ---

    @app.get("/api/nodes/{node_id}", response_model=NodeDetailResponse)
    async def get_node(node_id: str, c: Chatree = Depends(_get_client)) -> NodeDetailResponse:
        node = await c.repo.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")
        return NodeDetailResponse(
            node=node,
            ancestors=await c.repo.get_ancestors(node),
            inherited_messages=await c.context.collect_inherited_messages(node),
            messages=await c.repo.list_messages(node.id),
            pending_handoffs=await c.repo.list_mailbox_items(
                node.tree_id, to_node_id=node.id, status="pending_review"
            ),
            estimated_context_tokens=await c.context.estimate_node_context_tokens(node),
        )

    @app.patch("/api/nodes/{node_id}", response_model=NodeModel)
    async def update_node(
        node_id: str, body: UpdateNodeRequest, c: Chatree = Depends(_get_client)
    ) -> NodeModel:
        node = await c.repo.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")
        data = body.model_dump(exclude_unset=True)
        if "summary" in data and data["summary"] is not None:
            updated = await c.repo.commit_node_summary(
                node_id, summary=data["summary"], status=data.get("status")
            )
            await c.rag.index_node_summary(node_id)
            await c.rag.propagate_rollups_upward(node_id)
            other = {k: v for k, v in data.items() if k not in ("summary", "status")}
            if other:
                updated = await c.repo.update_node(node_id, **other) or updated
            return updated
        updated = await c.repo.update_node(node_id, **data)
        assert updated is not None
        return updated

    @app.post("/api/nodes/{node_id}/turns/stream")
    async def stream_turn(
        node_id: str, body: CreateTurnRequest, c: Chatree = Depends(_get_client)
    ) -> StreamingResponse:
        if await c.repo.get_node(node_id) is None:
            raise HTTPException(status_code=404, detail="Node not found")
        rt = await c.get_runtime_settings()

        async def event_stream() -> AsyncIterator[str]:
            async for ev in c.orchestrator.run_turn_stream(
                node_id=node_id,
                user_content=body.content,
                max_hop_depth=body.max_hop_depth or rt.max_hop_depth,
                token_budget=body.token_budget or rt.token_budget,
                model_override=body.model_override,
            ):
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/nodes/{node_id}/spawn")
    async def spawn_child(
        node_id: str, body: SpawnNodeRequest, c: Chatree = Depends(_get_client)
    ) -> dict[str, Any]:
        parent = await c.repo.get_node(node_id)
        if parent is None:
            raise HTTPException(status_code=404, detail="Parent node not found")
        tree = await c.repo.get_tree(parent.tree_id)
        assert tree is not None
        res = await c.ipc.spawn_child(
            tree=tree,
            parent=parent,
            goal=body.goal,
            title=body.title,
            context_mode=body.context_mode,
            summary_brief=body.summary_brief,
            slice_turns=body.slice_turn_count,
            model_override=body.model_override,
            tool_permission=body.tool_permission,
            switch_head=body.switch_head,
        )
        child_id = res.tool_output.get("child_id")
        child = await c.repo.get_node(child_id) if child_id else None
        return {
            "child": child.model_dump() if child else None,
            "ipc_event": res.ipc_event.model_dump() if res.ipc_event else None,
            "head_changed": (
                {
                    "from_node_id": res.head_changed[0],
                    "to_node_id": res.head_changed[1],
                    "reflog_entry": res.head_changed[2].model_dump(),
                }
                if res.head_changed
                else None
            ),
        }

    @app.post("/api/nodes/{node_id}/spawn-background", status_code=202)
    async def spawn_background(
        node_id: str, body: SpawnBackgroundRequest, c: Chatree = Depends(_get_client)
    ) -> dict[str, Any]:
        parent = await c.repo.get_node(node_id)
        if parent is None:
            raise HTTPException(status_code=404, detail="Parent node not found")
        tree = await c.repo.get_tree(parent.tree_id)
        assert tree is not None
        child, job = await c.actors.spawn_background(
            tree=tree,
            parent=parent,
            goal=body.goal,
            title=body.title,
            context_mode=body.context_mode,
            summary_brief=body.summary_brief,
            slice_turns=body.slice_turn_count,
            model_override=body.model_override,
            tool_permission=body.tool_permission,
            max_rounds=body.max_rounds,
            token_budget=body.token_budget,
        )
        return {"child": child.model_dump(), "job": job.model_dump()}

    @app.post("/api/nodes/{node_id}/finish")
    async def finish_node(
        node_id: str, body: FinishNodeRequest, c: Chatree = Depends(_get_client)
    ) -> dict[str, Any]:
        node = await c.repo.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found")
        if not node.parent_id:
            raise HTTPException(status_code=400, detail="Cannot finish the root node.")
        dt = await c.get_tree(node.tree_id)
        fin = await dt.node(node_id).finish(
            summary=body.summary,
            artifacts=body.artifacts,
            auto_commit=body.auto_commit,
            return_to_parent=body.return_to_parent,
        )
        updated_node = await c.repo.get_node(node_id)
        updated_tree = await c.repo.get_tree(node.tree_id)
        return {
            "node": updated_node.model_dump() if updated_node else None,
            "result": fin["result"],
            "handoff_pending": fin["handoff_pending"].model_dump() if fin["handoff_pending"] else None,
            "ipc_event": fin["ipc_event"].model_dump() if fin["ipc_event"] else None,
            "head_changed": (
                {"from_node_id": node_id, "to_node_id": updated_tree.head_node_id}
                if updated_tree and updated_tree.head_node_id != node_id
                else None
            ),
        }

    @app.post("/api/nodes/{node_id}/dismiss-stale/{source_node_id}")
    async def dismiss_stale(
        node_id: str, source_node_id: str, c: Chatree = Depends(_get_client)
    ) -> dict[str, Any]:
        await c.repo.dismiss_dependency(node_id, source_node_id)
        node = await c.repo.get_node(node_id)
        return {"ok": True, "node": node.model_dump() if node else None}

    @app.post("/api/nodes/{node_id}/extract")
    async def extract_to_child(
        node_id: str, body: ExtractTangentRequest, c: Chatree = Depends(_get_client)
    ) -> dict[str, Any]:
        try:
            parent, child = await c.ops.extract_to_child(
                node_id,
                body.seq_start,
                body.seq_end,
                title=body.title,
                goal=body.goal,
                summary=body.summary,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"parent": parent.model_dump(), "child": child.model_dump()}

    @app.post("/api/nodes/{node_id}/fork-edit", response_model=NodeModel)
    async def fork_on_edit(
        node_id: str, body: ForkEditRequest, c: Chatree = Depends(_get_client)
    ) -> NodeModel:
        try:
            return await c.ops.fork_on_edit(
                node_id, body.at_seq, title=body.title, switch_head=body.switch_head
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/nodes/{node_id}/compare", response_model=BranchComparison)
    async def compare_siblings(
        node_id: str, c: Chatree = Depends(_get_client)
    ) -> BranchComparison:
        try:
            return await c.ops.compare_siblings(node_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/nodes/{node_id}/promote", response_model=NodeModel)
    async def promote_child(
        node_id: str,
        archive_siblings: bool = Query(default=True),
        c: Chatree = Depends(_get_client),
    ) -> NodeModel:
        try:
            return await c.ops.promote_child(node_id, archive_siblings=archive_siblings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- IPC, Mailbox, Search, Reflog, Artifacts, Settings ---

    @app.post("/api/nodes/{node_id}/request")
    async def direct_request(
        node_id: str, body: DirectRequestPayload, c: Chatree = Depends(_get_client)
    ) -> dict[str, Any]:
        caller = await c.repo.get_node(node_id)
        if caller is None:
            raise HTTPException(status_code=404, detail="Caller node not found")
        tree = await c.repo.get_tree(caller.tree_id)
        assert tree is not None
        res = await c.ipc.request(
            tree=tree, caller=caller, target_id=body.target_id, prompt=body.prompt, mutate=body.mutate
        )
        return {
            "result": res.tool_output,
            "ipc_event": res.ipc_event.model_dump() if res.ipc_event else None,
            "stale_node_ids": res.stale_node_ids,
        }

    @app.post("/api/nodes/{node_id}/ask")
    async def ask_tree(
        node_id: str, body: AskTreePayload, c: Chatree = Depends(_get_client)
    ) -> dict[str, Any]:
        caller = await c.repo.get_node(node_id)
        if caller is None:
            raise HTTPException(status_code=404, detail="Caller node not found")
        tree = await c.repo.get_tree(caller.tree_id)
        assert tree is not None
        res = await c.ipc.ask_tree(
            tree=tree,
            caller=caller,
            query=body.query,
            scope=body.scope,
            max_targets=body.max_targets,
        )
        return {
            "result": res.tool_output,
            "ipc_event": res.ipc_event.model_dump() if res.ipc_event else None,
        }

    @app.post("/api/mailbox/{mailbox_id}/commit")
    async def commit_handoff(
        mailbox_id: str, body: CommitHandoffRequest, c: Chatree = Depends(_get_client)
    ) -> dict[str, Any]:
        try:
            mb, target, rf = await c.ipc.commit_pending_handoff(
                mailbox_id=mailbox_id,
                edited_summary=body.summary,
                edited_artifacts=body.artifacts,
                return_head_to_target=body.return_head_to_target,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "mailbox_item": mb.model_dump(),
            "target_node": target.model_dump(),
            "reflog_entry": rf.model_dump() if rf else None,
        }

    @app.post("/api/mailbox/{mailbox_id}/dismiss")
    async def dismiss_handoff(
        mailbox_id: str, c: Chatree = Depends(_get_client)
    ) -> dict[str, Any]:
        updated = await c.repo.update_mailbox_item(mailbox_id, status="dismissed")
        if updated is None:
            raise HTTPException(status_code=404, detail="Mailbox item not found")
        return {"mailbox_item": updated.model_dump()}

    @app.get("/api/trees/{tree_id}/ipc", response_model=list[IpcEventModel])
    async def list_ipc(
        tree_id: str, limit: int = Query(default=100, ge=1, le=500), c: Chatree = Depends(_get_client)
    ) -> list[IpcEventModel]:
        return await c.repo.list_ipc_events(tree_id, limit=limit)

    @app.get("/api/trees/{tree_id}/search", response_model=list[SearchHit])
    async def search_tree(
        tree_id: str,
        q: str = Query(..., min_length=1),
        scope: SearchScope = Query(default="all"),
        node_id: str | None = Query(default=None),
        top_k: int = Query(default=8, ge=1, le=30),
        c: Chatree = Depends(_get_client),
    ) -> list[SearchHit]:
        curr = await c.repo.get_node(node_id) if node_id else None
        return await c.rag.search(tree_id=tree_id, query=q, scope=scope, current_node=curr, top_k=top_k)

    @app.get("/api/trees/{tree_id}/reflog", response_model=list[ReflogEntryModel])
    async def list_reflog(
        tree_id: str, limit: int = Query(default=100, ge=1, le=500), c: Chatree = Depends(_get_client)
    ) -> list[ReflogEntryModel]:
        return await c.repo.list_reflog(tree_id, limit=limit)

    @app.post("/api/trees/{tree_id}/reflog/jump")
    async def jump_reflog(
        tree_id: str, body: ReflogJumpRequest, c: Chatree = Depends(_get_client)
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

    @app.get("/api/trees/{tree_id}/artifacts", response_model=list[ArtifactModel])
    async def list_artifacts(
        tree_id: str, c: Chatree = Depends(_get_client)
    ) -> list[ArtifactModel]:
        return await c.repo.list_artifacts(tree_id)

    @app.post("/api/trees/{tree_id}/artifacts", response_model=ArtifactModel, status_code=201)
    async def create_artifact(
        tree_id: str, body: CreateArtifactRequest, c: Chatree = Depends(_get_client)
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

    @app.get("/api/settings", response_model=RuntimeSettingsModel)
    async def get_settings(c: Chatree = Depends(_get_client)) -> RuntimeSettingsModel:
        return await c.get_runtime_settings()

    @app.put("/api/settings", response_model=RuntimeSettingsModel)
    async def update_settings(
        body: UpdateSettingsRequest, c: Chatree = Depends(_get_client)
    ) -> RuntimeSettingsModel:
        return await c.update_runtime_settings(body.model_dump(exclude_unset=True))

    return app
