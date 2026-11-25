from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chatree.config import ChatreeConfig
from chatree.sdk.client import Chatree
from chatree.server.dtos import (  # noqa: F401 - re-exported for backward compatibility
    AddFactsRequest,
    ArchiveBranchRequest,
    AskTreePayload,
    CommitHandoffRequest,
    CompactBranchRequest,
    CreateArtifactRequest,
    CreateTreeRequest,
    CreateTurnRequest,
    DirectRequestPayload,
    ExtractTangentRequest,
    FinishNodeRequest,
    ForkEditRequest,
    NodeDetailResponse,
    PinInvariantRequest,
    ReflogJumpRequest,
    SpawnBackgroundRequest,
    SpawnNodeRequest,
    TreeSnapshotResponse,
    UpdateNodeRequest,
    UpdateSettingsRequest,
    UpdateTreeRequest,
)
from chatree.server.routes_nodes import router as nodes_router
from chatree.server.routes_trees import router as trees_router


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
            "sqlite_vec": bool(ct.db.has_sqlite_vec) if ct.db is not None else False,
        }

    app.include_router(trees_router)
    app.include_router(nodes_router)
    return app
