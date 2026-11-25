from __future__ import annotations

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from chatree.config import ToolPermission
from chatree.sdk.client import Chatree
from chatree.types import (
    ArtifactInput,
    ArtifactModel,
    ChildContextMode,
    IpcEventModel,
    MailboxItemModel,
    MessageModel,
    NodeModel,
    NodeStatus,
    ReflogEntryModel,
    SearchScope,
    TreeModel,
)


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
    deep: bool = False


class CompactBranchRequest(BaseModel):
    keep_recent_turns: int | None = Field(default=None, ge=0, le=24)
    title: str | None = None
    reason: str = ""
    switch_head: bool = True


class ArchiveBranchRequest(BaseModel):
    reason: str = ""
    switch_to_parent: bool = True


class PinInvariantRequest(BaseModel):
    key: str = Field(..., min_length=1, max_length=64)
    value: str = Field(..., min_length=1, max_length=240)
    source_node_id: str | None = None


class AddFactsRequest(BaseModel):
    node_id: str = Field(..., min_length=1)
    facts: list[dict] = Field(default_factory=list)
    source_msg_seq: int | None = None


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
    llm_provider: str | None = None
    default_model: str | None = None
    shadow_model: str | None = None
    summary_model: str | None = None
    api_base: str | None = None
    max_hop_depth: int | None = Field(default=None, ge=1, le=16)
    token_budget: int | None = Field(default=None, ge=1000, le=500_000)
    auto_commit_handoffs: bool | None = None
    api_keys: dict[str, str] | None = None


def get_client(request: Request) -> Chatree:
    client: Chatree = request.app.state.chatree
    return client


async def build_tree_snapshot(ct: Chatree, tree_id: str) -> TreeSnapshotResponse:
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
