from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from chatree.config import ToolPermission

ContextMode = Literal["root", "fork", "slice", "clean"]
ChildContextMode = Literal["fork", "slice", "clean"]
NodeStatus = Literal["ACTIVE", "COMPLETED", "ARCHIVED"]
NodeKind = Literal["chat", "subagent"]
SessionKind = Literal["human", "agent"]
DEFAULT_SESSION_ID = "default"
ReflogAction = Literal[
    "init",
    "spawn",
    "jump",
    "return",
    "close",
    "reopen",
    "fork_edit",
    "extract",
    "promote",
]
IpcEventType = Literal[
    "spawn",
    "send",
    "send_finish",
    "request_summary",
    "request_shadow",
    "request_mutate",
    "search",
    "reflog_jump",
    "custom_tool",
]
MailboxStatus = Literal["pending_review", "unread", "delivered", "dismissed"]
SearchScope = Literal["all", "ancestors", "subtree", "siblings"]
RagGranularity = Literal["chunk", "summary", "rollup", "fact"]


class ArtifactInput(BaseModel):
    slug: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=200)
    kind: str = Field(default="markdown", max_length=40)
    content: str
    summary: str = ""


class ArtifactRef(BaseModel):
    id: str
    slug: str
    version: int
    kind: str
    title: str
    summary: str = ""

    @property
    def ref_label(self) -> str:
        return f"{self.slug}@v{self.version}"


class ArtifactModel(BaseModel):
    id: str
    tree_id: str
    slug: str
    version: int
    kind: str
    title: str
    content: str
    summary: str
    created_by_node_id: str | None = None
    created_at: str


ChangeSeverity = Literal["cosmetic", "material"]


class StaleDependencyInfo(BaseModel):
    source_node_id: str
    source_title: str
    seen_summary_version: int
    current_summary_version: int
    seen_msg_seq: int
    current_msg_seq: int
    source_is_dirty: bool
    reason: str
    hops: int = 1
    via_node_ids: list[str] = Field(default_factory=list)
    severity: ChangeSeverity = "material"
    change_delta: str = ""
    artifact_slug: str | None = None


class TreeModel(BaseModel):
    id: str
    title: str
    root_node_id: str | None = None
    head_node_id: str | None = None
    default_model: str
    shadow_model: str
    summary_model: str
    auto_commit_handoffs: bool
    reflog_cursor: int = 0
    created_at: str
    updated_at: str


class TreeSessionModel(BaseModel):
    """Stateful visitor/cursor walking a DialogueTree with its own HEAD and reflog trail."""

    id: str
    tree_id: str
    title: str = "Main Session"
    kind: SessionKind = "human"
    head_node_id: str | None = None
    reflog_cursor: int = 0
    is_default: bool = False
    created_at: str
    updated_at: str


class NodeModel(BaseModel):
    id: str
    tree_id: str
    parent_id: str | None = None
    path: str
    depth: int
    title: str
    goal: str
    summary_brief: str
    context_mode: ContextMode
    spawn_msg_seq: int
    slice_turn_count: int
    status: NodeStatus
    node_kind: NodeKind = "chat"
    tool_permission: ToolPermission = "read_only"
    model_override: str | None = None
    summary: str | None = None
    summary_version: int
    summary_minor_version: int = 0
    last_change_severity: ChangeSeverity = "material"
    last_change_delta: str = ""
    rollup_summary: str | None = None
    is_dirty: bool
    last_msg_seq: int
    tokens_in: int
    tokens_out: int
    cache_read_tokens: int
    created_at: str
    updated_at: str
    stale_dependencies: list[StaleDependencyInfo] = Field(default_factory=list)
    child_count: int = 0
    pending_handoffs_count: int = 0


class MessageModel(BaseModel):
    id: str
    node_id: str
    tree_id: str
    seq: int
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    model_used: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class IpcEventModel(BaseModel):
    id: str
    tree_id: str
    trace_id: str
    hop_depth: int
    from_node_id: str
    to_node_id: str | None = None
    event_type: IpcEventType
    payload: dict[str, Any]
    response: dict[str, Any]
    status: str
    target_summary_version: int | None = None
    target_msg_seq: int | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    created_at: str


class ReflogEntryModel(BaseModel):
    id: str
    tree_id: str
    session_id: str = DEFAULT_SESSION_ID
    seq: int
    head_ref: str
    from_node_id: str | None = None
    to_node_id: str
    action: ReflogAction
    note: str
    orientation_cue: str
    cue_consumed: bool
    created_at: str


class MailboxItemModel(BaseModel):
    id: str
    tree_id: str
    to_node_id: str
    from_node_id: str
    ipc_event_id: str | None = None
    kind: Literal["handoff", "message"]
    status: MailboxStatus
    summary: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    delivered_at: str | None = None


class BackgroundJobModel(BaseModel):
    job_id: str
    tree_id: str
    parent_node_id: str
    child_node_id: str
    title: str
    goal: str
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    summary: str | None = None
    mailbox_id: str | None = None
    error: str | None = None
    turns_executed: int = 0
    tokens_used: int = 0
    created_at: str
    completed_at: str | None = None


class EmbeddingRecord(BaseModel):
    id: str
    tree_id: str
    node_id: str
    path: str
    granularity: RagGranularity
    chunk_index: int = 0
    seq_start: int | None = None
    seq_end: int | None = None
    content: str
    summary_version: int = 0
    updated_at: str


class BranchFactModel(BaseModel):
    """Tier-2.5 provenance-linked atomic temporal fact scoped to one branch (append-only across branches)."""

    id: str
    tree_id: str
    node_id: str
    fact_text: str
    entities: list[str] = Field(default_factory=list)
    resolved_date: str = ""
    source_msg_seq: int | None = None
    status: Literal["active", "superseded"] = "active"
    created_at: str


class TreeInvariantModel(BaseModel):
    """Tree-level pinned invariant surfaced in `<dialogue_tree_context>` (`PINNED [key=value]`)."""

    tree_id: str
    key: str
    value: str
    source_node_id: str | None = None
    source_summary_version: int = 0
    source_msg_seq: int = 0
    updated_at: str


class SearchHit(BaseModel):
    node_id: str
    node_title: str
    node_status: NodeStatus
    node_kind: NodeKind = "chat"
    path: str
    summary_version: int
    summary: str | None = None
    snippet: str
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    granularity: Literal["chunk", "summary", "rollup", "fact", "node", "message"] = "chunk"
    matched_seq: int | None = None
    seq_range: tuple[int, int] | None = None
    source: Literal["hybrid", "vector", "bm25", "node", "message"] = "hybrid"


class SiblingDiffEntry(BaseModel):
    node_id: str
    title: str
    status: NodeStatus
    summary_version: int
    summary: str
    artifact_refs: list[str] = Field(default_factory=list)
    tokens_total: int = 0
    message_count: int = 0


class BranchComparison(BaseModel):
    parent_id: str
    parent_title: str
    siblings: list[SiblingDiffEntry]
    shared_terms: list[str] = Field(default_factory=list)
    distinctive_terms_by_node: dict[str, list[str]] = Field(default_factory=dict)


class TreeMetrics(BaseModel):
    tree_id: str
    node_count: int
    active_nodes: int
    completed_nodes: int
    max_depth: int
    total_messages: int
    total_tokens_in: int
    total_tokens_out: int
    total_cache_read_tokens: int
    cache_hit_ratio: float
    ipc_total: int
    ipc_by_type: dict[str, int] = Field(default_factory=dict)
    shadow_turns: int = 0
    shadow_cache_hits: int = 0
    cached_summary_hits: int = 0
    guardrail_blocks: int = 0
    stale_edges_count: int = 0
    reflog_jumps: int = 0
    artifact_count: int = 0
    indexed_embeddings: int = 0


class TurnResult(BaseModel):
    trace_id: str
    node_id: str
    head_node_id: str
    session_id: str = DEFAULT_SESSION_ID
    user_message: MessageModel | None = None
    assistant_message: MessageModel | None = None
    content: str = ""
    ipc_events: list[IpcEventModel] = Field(default_factory=list)
    pending_handoffs: list[MailboxItemModel] = Field(default_factory=list)
    ephemeral_notices: list[str] = Field(default_factory=list)
    stale_node_ids: list[str] = Field(default_factory=list)
    compacted_from_node_id: str | None = None
    tokens_used: int = 0
    token_budget: int = 96_000
    error: dict[str, Any] | None = None


class RuntimeSettingsModel(BaseModel):
    llm_provider: str = "auto"
    default_model: str
    shadow_model: str
    summary_model: str
    api_base: str | None = None
    max_hop_depth: int
    token_budget: int
    auto_commit_handoffs: bool
    configured_providers: dict[str, bool] = Field(default_factory=dict)


# --- Low-level LLM Provider Dataclasses ---


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai_dict(self) -> dict[str, Any]:
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    model_used: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    latency_ms: int = 0


@dataclass
class LLMStreamEvent:
    kind: str  # "delta" | "done"
    delta: str = ""
    response: LLMResponse | None = None
