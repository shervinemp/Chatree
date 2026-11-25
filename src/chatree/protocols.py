from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Literal, Protocol, runtime_checkable

from chatree.types import (
    ArtifactModel,
    BackgroundJobModel,
    BranchFactModel,
    IpcEventModel,
    LLMResponse,
    LLMStreamEvent,
    MailboxItemModel,
    MessageModel,
    NodeModel,
    ReflogEntryModel,
    SearchHit,
    TreeInvariantModel,
    TreeModel,
)

ModelRole = Literal["chat", "shadow", "summary"]


@runtime_checkable
class LLMProvider(Protocol):
    """Pluggable LLM backend protocol (LiteLLM, OpenAI, Anthropic, local, or custom)."""

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse: ...

    def stream_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[LLMStreamEvent]: ...


@runtime_checkable
class Embedder(Protocol):
    """Pluggable vector embedding provider backing 3-tier hybrid Tree RAG."""

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Reranker(Protocol):
    """Optional post-retrieval reranker protocol (cross-encoder, Cohere, BGE, LLM reranker)."""

    async def rerank(
        self, query: str, hits: list[SearchHit], top_k: int
    ) -> list[SearchHit]: ...


@runtime_checkable
class TokenCounter(Protocol):
    """Pluggable token estimator (tiktoken, provider tokenizer, or calibrated heuristic)."""

    def count_text(self, text: str, *, model: str | None = None) -> int: ...

    def count_messages(self, messages: list[dict[str, Any]], *, model: str | None = None) -> int: ...


@runtime_checkable
class StorageBackend(Protocol):
    """
    Full pluggable persistence contract satisfied by `SQLiteRepository` and implementable
    for hosted Postgres + pgvector or custom enterprise stores (injectable via `Chatree(repo=...)`).
    """

    # Trees
    async def create_tree(
        self,
        title: str,
        goal: str,
        default_model: str,
        shadow_model: str,
        summary_model: str,
        auto_commit_handoffs: bool,
    ) -> tuple[TreeModel, NodeModel, ReflogEntryModel]: ...
    async def get_tree(self, tree_id: str) -> TreeModel | None: ...
    async def list_trees(self) -> list[TreeModel]: ...
    async def update_tree(self, tree_id: str, **fields: Any) -> TreeModel | None: ...
    async def delete_tree(self, tree_id: str) -> bool: ...

    # Nodes & messages
    async def create_child_node(self, *args: Any, **kwargs: Any) -> NodeModel: ...
    async def get_node(self, node_id: str) -> NodeModel | None: ...
    async def list_nodes(self, tree_id: str) -> list[NodeModel]: ...
    async def get_children(self, node_id: str) -> list[NodeModel]: ...
    async def get_ancestors(self, node: NodeModel) -> list[NodeModel]: ...
    async def update_node(self, node_id: str, **fields: Any) -> NodeModel | None: ...
    async def commit_node_summary(self, *args: Any, **kwargs: Any) -> NodeModel: ...
    async def append_message(self, *args: Any, **kwargs: Any) -> MessageModel: ...
    async def list_messages(self, node_id: str, up_to_seq: int | None = None) -> list[MessageModel]: ...

    # IPC, dependencies, reflog, artifacts, mailbox, jobs, settings
    async def create_ipc_event(self, *args: Any, **kwargs: Any) -> IpcEventModel: ...
    async def list_ipc_events(self, tree_id: str, limit: int = 200) -> list[IpcEventModel]: ...
    async def upsert_dependency(self, *args: Any, **kwargs: Any) -> None: ...
    async def list_dependencies(self, tree_id: str) -> list[dict[str, Any]]: ...
    async def append_reflog(self, *args: Any, **kwargs: Any) -> ReflogEntryModel: ...
    async def list_reflog(
        self, tree_id: str, limit: int = 100, *, session_id: str | None = "default"
    ) -> list[ReflogEntryModel]: ...
    async def create_artifact(self, *args: Any, **kwargs: Any) -> ArtifactModel: ...
    async def list_artifacts(self, tree_id: str) -> list[ArtifactModel]: ...
    async def create_mailbox_item(self, *args: Any, **kwargs: Any) -> MailboxItemModel: ...
    async def list_mailbox_items(
        self, tree_id: str, to_node_id: str | None = None, status: Any = None
    ) -> list[MailboxItemModel]: ...
    async def get_all_settings(self) -> dict[str, Any]: ...
    async def set_settings(self, values: dict[str, Any]) -> None: ...

    # Shadow memoization cache
    async def get_cached_shadow_answer(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None: ...
    async def put_cached_shadow_answer(self, *args: Any, **kwargs: Any) -> None: ...
    async def invalidate_shadow_cache(self, target_node_id: str) -> None: ...

    # Embeddings & background jobs
    async def replace_node_embeddings(self, *args: Any, **kwargs: Any) -> int: ...
    async def count_embeddings(self, tree_id: str) -> int: ...
    async def list_background_jobs(self, tree_id: str | None = None, *, limit: int = 50) -> list[BackgroundJobModel]: ...

    # Tier-2.5 atomic temporal facts & tree-level PINNED invariants
    async def replace_branch_facts(self, *args: Any, **kwargs: Any) -> list[BranchFactModel]: ...
    async def list_branch_facts(
        self,
        tree_id: str,
        *,
        node_id: str | None = None,
        active_only: bool = True,
        exclude_archived: bool = True,
    ) -> list[BranchFactModel]: ...
    async def upsert_tree_invariant(self, *args: Any, **kwargs: Any) -> TreeInvariantModel: ...
    async def delete_tree_invariant(self, tree_id: str, key: str) -> bool: ...
    async def list_tree_invariants(self, tree_id: str) -> list[TreeInvariantModel]: ...


@runtime_checkable
class ModelRouter(Protocol):
    """Determines which model handles a given node and role (chat, shadow, summary)."""

    def resolve_model(self, tree: TreeModel, node: NodeModel, role: ModelRole) -> str: ...


class DefaultModelRouter:
    """
    Default per-node / per-role model router with optional threshold-based fast shadow routing:
      - role="chat": node.model_override -> tree.default_model
      - role="shadow":
          * If tree.shadow_model is explicit (not 'inherit'/'auto'), use it.
          * If `fast_shadow_model` is configured and target node is short/cold
            (`tokens_in < warm_cache_token_threshold`), route to `fast_shadow_model` for low latency;
            otherwise inherit the target node's own model to maximize provider KV-cache hits!
      - role="summary": tree.summary_model (if not 'inherit') -> node chat model
      - Supports optional policy callback `override_fn(tree, node, role) -> str | None`.
    """

    def __init__(
        self,
        override_fn: Callable[[TreeModel, NodeModel, ModelRole], str | None] | None = None,
        *,
        fast_shadow_model: str | None = None,
        warm_cache_token_threshold: int = 4_000,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._override_fn = override_fn
        self.fast_shadow_model = fast_shadow_model
        self.warm_cache_token_threshold = warm_cache_token_threshold
        self.cache_ttl_seconds = cache_ttl_seconds

    def resolve_model(
        self,
        tree: TreeModel,
        node: NodeModel,
        role: ModelRole,
        *,
        context_tokens: int | None = None,
        cache_age_seconds: float | None = None,
    ) -> str:
        if self._override_fn is not None:
            custom = self._override_fn(tree, node, role)
            if custom:
                return custom
        base_model = node.model_override or tree.default_model
        if role == "shadow":
            if tree.shadow_model and tree.shadow_model not in ("inherit", "auto"):
                return tree.shadow_model
            if self.fast_shadow_model:
                # Prefer live estimated context prefix size over cumulative billing counter
                eff_tokens = context_tokens if context_tokens is not None else node.tokens_in
                cache_expired = (
                    cache_age_seconds is not None
                    and self.cache_ttl_seconds > 0
                    and cache_age_seconds > self.cache_ttl_seconds
                )
                if eff_tokens < self.warm_cache_token_threshold or cache_expired:
                    return self.fast_shadow_model
            return base_model
        if role == "summary":
            if tree.summary_model and tree.summary_model != "inherit":
                return tree.summary_model
            return base_model
        return base_model
