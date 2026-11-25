from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Callable, cast

from chatree.config import ChatreeConfig, ToolPermission
from chatree.engine.actors import BackgroundActorManager
from chatree.engine.context import ContextAssembler
from chatree.engine.guardrails import TraceContext
from chatree.engine.ipc import IPCRouter
from chatree.engine.operations import BranchOperations
from chatree.engine.orchestrator import TurnOrchestrator
from chatree.engine.query import TreeQueryEngine
from chatree.engine.rag import TreeRAGService
from chatree.engine.reflog import ReflogService
from chatree.engine.staleness import StalenessService
from chatree.hooks import HookBus, HookCallback, HookEventName
from chatree.protocols import (
    DefaultModelRouter,
    Embedder,
    LLMProvider,
    ModelRouter,
    Reranker,
    StorageBackend,
)
from chatree.providers.embeddings import build_embedder
from chatree.providers.factory import build_llm_provider
from chatree.sandbox import BranchSandbox, SandboxWorkspace
from chatree.storage.database import Database
from chatree.storage.portable import DialogueDirectoryStore
from chatree.storage.repository import SQLiteRepository
from chatree.tools import STATIC_SYSTEM_PROMPT, ToolRegistry, ToolSideEffect
from chatree.types import (
    ArtifactInput,
    ArtifactModel,
    BackgroundJobModel,
    BranchComparison,
    BranchFactModel,
    ChildContextMode,
    IpcEventModel,
    MailboxItemModel,
    MessageModel,
    NodeKind,
    NodeModel,
    NodeStatus,
    RagGranularity,
    ReflogEntryModel,
    RuntimeSettingsModel,
    SearchHit,
    SearchScope,
    StaleDependencyInfo,
    TreeInvariantModel,
    TreeMetrics,
    TreeModel,
    TurnResult,
)


class Chatree:
    """
    Primary entry point for embedding the Chatree hierarchical LLM dialogue tree framework.

    Example:
        async with Chatree.in_memory() as ct:
            tree = await ct.create_tree("Auth Design")
            res = await tree.chat("Let's use short-lived JWTs.")
            child = await tree.spawn("Design refresh token rotation", mode="fork")
            await child.chat("Use rotating opaque tokens stored hashed in SQLite.")
            await child.finish(auto_commit=True)
            ans = await tree.request(child.id, prompt="How are refresh tokens stored?")
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        config: ChatreeConfig | None = None,
        repo: StorageBackend | None = None,
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        router: ModelRouter | None = None,
        hooks: HookBus | None = None,
        tools: ToolRegistry | None = None,
        sandbox: BranchSandbox | None = None,
        system_prompt: str = STATIC_SYSTEM_PROMPT,
    ) -> None:
        self.config = config or ChatreeConfig()
        resolved = (
            db_path
            if db_path is not None
            else self.config.resolved_db_path()
        )
        self._owns_db = repo is None
        if repo is None:
            db = Database(resolved)
            self.db: Database | None = db
            self.repo: SQLiteRepository = SQLiteRepository(db)
        else:
            self.db = getattr(repo, "db", None)
            self.repo = cast(SQLiteRepository, repo)
        self.hooks = hooks or HookBus()
        self.tools = tools or ToolRegistry()
        self.sandbox = sandbox
        self.llm: LLMProvider = llm or build_llm_provider(
            self.config.llm_provider,
            model=self.config.default_model,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            extra_headers=self.config.extra_headers or None,
            enable_cache_control=self.config.ablations.enable_provider_cache_control,
        )
        self.embedder: Embedder = embedder or build_embedder(
            model=self.config.rag.embedding_model,
            dimensions=self.config.rag.embedding_dim,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
        )
        self.router: ModelRouter = router or DefaultModelRouter(
            warm_cache_token_threshold=self.config.guardrails.short_branch_shadow_threshold_tokens,
            cache_ttl_seconds=self.config.guardrails.warm_cache_ttl_seconds,
        )
        self.context = ContextAssembler(
            self.repo,
            system_prompt=system_prompt,
            ablations=self.config.ablations,
            max_context_tokens=self.config.guardrails.max_branch_context_tokens,
        )
        self.reflog = ReflogService(self.repo, hooks=self.hooks)
        self.rag = TreeRAGService(
            self.repo,
            self.embedder,
            rag_config=self.config.rag,
            ablations=self.config.ablations,
            reranker=reranker,
        )
        self._bg_summary_tasks: dict[str, asyncio.Task[str]] = {}
        self.staleness = StalenessService(
            self.repo,
            self.context,
            self.llm,
            self.router,
            rag=self.rag,
            embedder=self.embedder,
            guardrails=self.config.guardrails,
            ablations=self.config.ablations,
            tool_registry=self.tools,
        )
        self.ipc = IPCRouter(
            self.repo,
            self.context,
            self.llm,
            self.router,
            self.reflog,
            self.staleness,
            self.rag,
            self.tools,
            hooks=self.hooks,
            guardrails=self.config.guardrails,
            ablations=self.config.ablations,
            sandbox=self.sandbox,
            on_branch_switch_away=(
                self._schedule_bg_summary if self.config.auto_background_summarize_on_switch else None
            ),
            on_branch_archived=self._on_branch_archived,
        )
        self.query_engine = TreeQueryEngine(
            self.repo,
            self.ipc,
            self.rag,
            self.llm,
            self.router,
            ablations=self.config.ablations,
        )
        self.orchestrator = TurnOrchestrator(
            self.repo,
            self.context,
            self.llm,
            self.router,
            self.ipc,
            self.reflog,
            self.staleness,
            self.rag,
            self.tools,
            hooks=self.hooks,
            guardrails=self.config.guardrails,
        )
        self.ops = BranchOperations(
            self.repo, self.reflog, self.staleness, self.rag, hooks=self.hooks
        )
        self.actors = BackgroundActorManager(
            self.repo,
            self.context,
            self.llm,
            self.router,
            self.ipc,
            self.reflog,
            self.staleness,
            self.rag,
            self.tools,
            hooks=self.hooks,
            guardrails=self.config.guardrails,
        )

    @classmethod
    def open(
        cls,
        db_path: str | Path = "./data/chatree.db",
        **kwargs: Any,
    ) -> Chatree:
        return cls(db_path=db_path, **kwargs)

    @classmethod
    def in_memory(cls, **kwargs: Any) -> Chatree:
        return cls(db_path=":memory:", **kwargs)

    async def startup(self) -> Chatree:
        if self._owns_db and self.db is not None:
            await self.db.connect()
        if self.sandbox is not None:
            if hasattr(self.sandbox, "attach_repository"):
                self.sandbox.attach_repository(self.repo)
            if hasattr(self.sandbox, "hydrate"):
                await self.sandbox.hydrate()
        if hasattr(self.repo, "list_background_jobs"):
            await self.actors.hydrate()
        stored = await self.repo.get_all_settings()
        if "api_base" in stored and hasattr(self.llm, "api_base"):
            setattr(self.llm, "api_base", stored["api_base"])
        if (
            self.config.persist_api_keys_in_db
            and isinstance(stored.get("api_keys"), dict)
            and callable(getattr(self.llm, "set_runtime_keys", None))
        ):
            self.llm.set_runtime_keys(stored["api_keys"])  # type: ignore[attr-defined]
        return self

    def _schedule_bg_summary(self, tree_id: str, node_id: str) -> asyncio.Task[str]:
        """Schedule a non-blocking Tier-2 summary refresh for a paused/switched branch."""
        existing = self._bg_summary_tasks.get(node_id)
        if existing is not None and not existing.done():
            return existing

        async def _worker() -> str:
            try:
                tree = await self.repo.get_tree(tree_id)
                node = await self.repo.get_node(node_id)
                if tree is None or node is None or not node.is_dirty:
                    return node.summary or "" if node else ""
                updated, *_ = await self.staleness.ensure_fresh_summary(
                    tree, node, force=False, new_status=node.status
                )
                return updated.summary or ""
            except asyncio.CancelledError:
                raise
            except Exception:
                return ""
            finally:
                cur = self._bg_summary_tasks.get(node_id)
                if cur is asyncio.current_task():
                    self._bg_summary_tasks.pop(node_id, None)

        task: asyncio.Task[str] = asyncio.create_task(
            _worker(), name=f"chatree-bg-summary-{node_id}"
        )
        self._bg_summary_tasks[node_id] = task
        return task

    async def wait_background_summaries(self, *, timeout: float | None = 15.0) -> int:
        """Await any in-flight non-blocking background branch summaries (returns count awaited)."""
        pending = [t for t in self._bg_summary_tasks.values() if not t.done()]
        if not pending:
            return 0
        await asyncio.wait(pending, timeout=timeout)
        return len(pending)

    async def close(self) -> None:
        bg_tasks = list(self._bg_summary_tasks.values())
        for t in bg_tasks:
            if not t.done():
                t.cancel()
        if bg_tasks:
            await asyncio.gather(*bg_tasks, return_exceptions=True)
        self._bg_summary_tasks.clear()
        await self.actors.cancel_all()
        if self._owns_db and self.db is not None:
            await self.db.close()

    async def __aenter__(self) -> Chatree:
        await self.startup()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Extension ergonomics (custom tools + hooks)
    # ------------------------------------------------------------------

    def tool(
        self,
        *,
        name: str | None = None,
        namespace: str | None = None,
        separator: str = "_",
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        side_effect: ToolSideEffect = "read_only",
        allowed_in_shadow: bool = False,
        overwrite: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a custom domain tool (optionally namespaced, e.g. `namespace='ue5'`) with branch permission scoping."""
        return self.tools.tool(
            name=name,
            namespace=namespace,
            separator=separator,
            description=description,
            parameters=parameters,
            side_effect=side_effect,
            allowed_in_shadow=allowed_in_shadow,
            overwrite=overwrite,
        )

    def tool_namespace(self, namespace: str, *, separator: str = "_") -> Any:
        """Create a scoped namespace helper (e.g. `ue5 = ct.tool_namespace('ue5')`) for external tool packs."""
        return self.tools.namespace(namespace, separator=separator)

    def on(self, event: HookEventName | str) -> Callable[[HookCallback], HookCallback]:
        """Attach a sync or async lifecycle hook listener."""
        return self.hooks.on(event)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Tree lifecycle
    # ------------------------------------------------------------------

    async def create_tree(
        self,
        title: str,
        *,
        goal: str = "",
        default_model: str | None = None,
        shadow_model: str | None = None,
        summary_model: str | None = None,
        auto_commit_handoffs: bool | None = None,
    ) -> DialogueTree:
        rt = await self.get_runtime_settings()
        tree, root, _ = await self.repo.create_tree(
            title=title,
            goal=goal,
            default_model=default_model or rt.default_model,
            shadow_model=shadow_model or rt.shadow_model,
            summary_model=summary_model or rt.summary_model,
            auto_commit_handoffs=(
                auto_commit_handoffs
                if auto_commit_handoffs is not None
                else rt.auto_commit_handoffs
            ),
        )
        await self.rag.index_node_summary(root.id)
        return DialogueTree(self, tree.id)

    async def get_tree(self, tree_id: str) -> DialogueTree:
        tree = await self.repo.get_tree(tree_id)
        if tree is None:
            raise ValueError(f"Tree not found: {tree_id}")
        return DialogueTree(self, tree.id)

    async def list_trees(self) -> list[TreeModel]:
        return await self.repo.list_trees()

    async def _on_branch_archived(self, tree_id: str, node_id: str) -> list[str]:
        return await self.actors.cancel_for_subtree(
            tree_id, node_id, reason=f"Ancestor branch '{node_id}' archived"
        )

    async def delete_tree(self, tree_id: str) -> bool:
        await self.actors.cancel_for_subtree(tree_id, None, reason="Tree deleted")
        return await self.repo.delete_tree(tree_id)

    async def export_directory(self, tree_id: str, output_dir: str | Path) -> Path:
        """Export a tree into a portable, human-greppable `.dialogue/` directory."""
        return await DialogueDirectoryStore.export_tree(self.repo, tree_id, output_dir)

    async def import_directory(
        self, input_dir: str | Path, *, overwrite: bool = True
    ) -> DialogueTree:
        """Import a `.dialogue/` directory into this Chatree store and rebuild 3-tier RAG indexes."""
        restored = await DialogueDirectoryStore.import_tree(
            self.repo, input_dir, rag=self.rag, overwrite=overwrite
        )
        return DialogueTree(self, restored.id)

    # ------------------------------------------------------------------
    # Runtime settings persistence
    # ------------------------------------------------------------------

    async def get_runtime_settings(self) -> RuntimeSettingsModel:
        stored = await self.repo.get_all_settings()
        configured = (
            self.llm.configured_providers()  # type: ignore[attr-defined]
            if callable(getattr(self.llm, "configured_providers", None))
            else {"custom": True}
        )
        return RuntimeSettingsModel(
            llm_provider=stored.get("llm_provider", self.config.llm_provider),
            default_model=stored.get("default_model", self.config.default_model),
            shadow_model=stored.get("shadow_model", self.config.shadow_model),
            summary_model=stored.get("summary_model", self.config.summary_model),
            api_base=stored.get("api_base", self.config.api_base),
            max_hop_depth=int(stored.get("max_hop_depth", self.config.guardrails.max_hop_depth)),
            token_budget=int(stored.get("token_budget", self.config.guardrails.token_budget)),
            auto_commit_handoffs=bool(
                stored.get("auto_commit_handoffs", self.config.auto_commit_handoffs)
            ),
            configured_providers=configured,
        )

    async def update_runtime_settings(self, updates: dict[str, Any]) -> RuntimeSettingsModel:
        to_persist: dict[str, Any] = {}
        for key in (
            "llm_provider",
            "default_model",
            "shadow_model",
            "summary_model",
            "api_base",
            "max_hop_depth",
            "token_budget",
            "auto_commit_handoffs",
        ):
            if key in updates and updates[key] is not None:
                val = updates[key]
                if key == "llm_provider":
                    self.config.llm_provider = val
                elif key == "api_base":
                    val = val or None
                    if hasattr(self.llm, "api_base"):
                        setattr(self.llm, "api_base", val)
                elif key == "max_hop_depth":
                    self.config.guardrails.max_hop_depth = int(val)
                elif key == "token_budget":
                    self.config.guardrails.token_budget = int(val)
                to_persist[key] = val

        if isinstance(updates.get("api_keys"), dict) and updates["api_keys"]:
            clean_keys = {
                k: str(v).strip()
                for k, v in updates["api_keys"].items()
                if v and str(v).strip()
            }
            if callable(getattr(self.llm, "set_runtime_keys", None)):
                self.llm.set_runtime_keys(clean_keys)  # type: ignore[attr-defined]
            # Never persist plaintext API keys to SQLite unless explicitly opted in
            if self.config.persist_api_keys_in_db:
                stored = await self.repo.get_all_settings()
                existing = dict(stored.get("api_keys") or {})
                existing.update(clean_keys)
                to_persist["api_keys"] = existing

        if to_persist:
            await self.repo.set_settings(to_persist)
        return await self.get_runtime_settings()


class DialogueTree:
    """High-level ergonomic handle bound to a single Chatree dialogue tree."""

    def __init__(self, client: Chatree, tree_id: str) -> None:
        self._ct = client
        self.id = tree_id

    async def model(self) -> TreeModel:
        t = await self._ct.repo.get_tree(self.id)
        if t is None:
            raise ValueError(f"Tree not found: {self.id}")
        return t

    async def head(self) -> NodeHandle:
        t = await self.model()
        assert t.head_node_id is not None
        return NodeHandle(self, t.head_node_id)

    async def root(self) -> NodeHandle:
        t = await self.model()
        assert t.root_node_id is not None
        return NodeHandle(self, t.root_node_id)

    def node(self, node_id: str) -> NodeHandle:
        return NodeHandle(self, node_id)

    async def nodes(self) -> list[NodeModel]:
        return await self._ct.repo.list_nodes(self.id)

    async def active_chat_nodes(self) -> list[NodeModel]:
        """Return non-archived interactive chat branches (`node_kind == 'chat'`, `status == 'ACTIVE'`)."""
        return [
            n
            for n in await self.nodes()
            if n.node_kind == "chat" and n.status == "ACTIVE"
        ]

    async def subagents(self) -> list[NodeModel]:
        """Return all spawned sub-agent worker nodes (`node_kind == 'subagent'`) in this tree."""
        return [n for n in await self.nodes() if n.node_kind == "subagent"]

    async def chat(
        self,
        content: str,
        *,
        node_id: str | None = None,
        model_override: str | None = None,
        max_hop_depth: int | None = None,
        token_budget: int | None = None,
        include_core_tools: bool = True,
    ) -> TurnResult:
        target_id = node_id or (await self.model()).head_node_id
        assert target_id is not None
        return await self._ct.orchestrator.run_turn(
            target_id,
            content,
            model_override=model_override,
            max_hop_depth=max_hop_depth,
            token_budget=token_budget,
            include_core_tools=include_core_tools,
        )

    async def stream(
        self,
        content: str,
        *,
        node_id: str | None = None,
        model_override: str | None = None,
        max_hop_depth: int | None = None,
        token_budget: int | None = None,
        include_core_tools: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        target_id = node_id or (await self.model()).head_node_id
        assert target_id is not None
        async for ev in self._ct.orchestrator.run_turn_stream(
            target_id,
            content,
            model_override=model_override,
            max_hop_depth=max_hop_depth,
            token_budget=token_budget,
            include_core_tools=include_core_tools,
        ):
            yield ev

    async def spawn(
        self,
        goal: str,
        *,
        parent_id: str | None = None,
        title: str | None = None,
        mode: ChildContextMode = "fork",
        summary_brief: str = "",
        slice_turns: int = 6,
        autonomous: bool = False,
        model: str | None = None,
        permission: ToolPermission | None = None,
        node_kind: NodeKind | None = None,
        switch_head: bool = True,
    ) -> NodeHandle:
        t = await self.model()
        pid = parent_id or t.head_node_id or t.root_node_id
        assert pid is not None
        parent = await self._ct.repo.get_node(pid)
        if parent is None:
            raise ValueError(f"Parent node not found: {pid}")
        res = await self._ct.ipc.spawn_child(
            tree=t,
            parent=parent,
            goal=goal,
            title=title,
            context_mode=mode,
            summary_brief=summary_brief,
            slice_turns=slice_turns,
            autonomous=autonomous,
            model_override=model,
            tool_permission=permission,
            node_kind=node_kind,
            switch_head=switch_head,
        )
        if res.tool_output.get("error"):
            raise RuntimeError(f"{res.tool_output['error']}: {res.tool_output.get('detail')}")
        child_id = str(res.tool_output["child_id"])
        # Automatically provision an isolated sandbox workspace for writable child branches!
        if (
            self._ct.sandbox is not None
            and res.tool_output.get("tool_permission") == "read_write"
        ):
            await self._ct.sandbox.create_workspace(child_id, base_node_id=parent.id)
        return NodeHandle(self, child_id)

    def workspace(self, node_id: str) -> SandboxWorkspace | None:
        if self._ct.sandbox is None:
            return None
        return self._ct.sandbox.get_workspace(node_id)

    async def ask(
        self,
        query: str,
        *,
        scope: SearchScope = "all",
        max_targets: int = 2,
        caller_id: str | None = None,
        exclude_node_ids: set[str] | None = None,
        shadow_prompt: str | None = None,
        deep: bool = False,
        enable_recursive_hop: bool = True,
        include_branch_summary_map: bool = True,
        concise_synthesis: bool = True,
    ) -> dict[str, Any]:
        """
        Single-hop compound Retrieve-then-Ask fusing hybrid search + parallel shadow turns.
        Pass `deep=True` to execute the unified `TreeQueryEngine.deep_ask()` pipeline
        (Hop-1 shadows + Tier-2.5 atomic dated facts + optional Hop-2 drill-down + chronological synthesis).
        """
        t = await self.model()
        cid = caller_id or t.head_node_id or t.root_node_id
        assert cid is not None
        caller = await self._ct.repo.get_node(cid)
        assert caller is not None
        if deep:
            return await self._ct.query_engine.deep_ask(
                tree=t,
                caller=caller,
                query=query,
                scope=scope,
                max_targets=max_targets,
                enable_recursive_hop=enable_recursive_hop,
                include_branch_summary_map=include_branch_summary_map,
                concise_synthesis=concise_synthesis,
            )
        res = await self._ct.ipc.ask_tree(
            tree=t,
            caller=caller,
            query=query,
            scope=scope,
            max_targets=max_targets,
            exclude_node_ids=exclude_node_ids,
            shadow_prompt=shadow_prompt,
        )
        return res.tool_output

    async def request(
        self,
        target_id: str,
        *,
        prompt: str | None = None,
        mutate: bool = False,
        caller_id: str | None = None,
    ) -> dict[str, Any]:
        t = await self.model()
        cid = caller_id or t.head_node_id or t.root_node_id
        assert cid is not None
        caller = await self._ct.repo.get_node(cid)
        assert caller is not None
        # When an external SDK/CLI/MCP caller omits caller_id and queries HEAD/root directly,
        # start the trace from "external" so querying HEAD isn't rejected as a self-cycle
        # (while recursive loops initiated from inside target_id are still blocked).
        trace_ctx = None
        if caller_id is None and cid == target_id:
            trace_ctx = TraceContext.start(
                "external",
                self._ct.config.guardrails.max_hop_depth,
                self._ct.config.guardrails.token_budget,
            )
        res = await self._ct.ipc.request(
            tree=t,
            caller=caller,
            target_id=target_id,
            prompt=prompt,
            mutate=mutate,
            trace_ctx=trace_ctx,
        )
        return res.tool_output

    async def search(
        self,
        query: str,
        *,
        scope: SearchScope = "all",
        node_id: str | None = None,
        top_k: int = 5,
        status_filter: set[NodeStatus] | None = None,
        granularities: set[RagGranularity] | None = None,
    ) -> list[SearchHit]:
        t = await self.model()
        cid = node_id or t.head_node_id
        curr = await self._ct.repo.get_node(cid) if cid else None
        return await self._ct.rag.search(
            tree_id=self.id,
            query=query,
            scope=scope,
            current_node=curr,
            top_k=top_k,
            status_filter=status_filter,
            granularities=granularities,
        )

    async def jump(self, target: str, note: str = "") -> NodeHandle:
        prev_tree = await self.model()
        prev_head_id = prev_tree.head_node_id
        dest, _, _ = await self._ct.reflog.jump(self.id, target, note=note)
        if (
            self._ct.config.auto_background_summarize_on_switch
            and prev_head_id
            and prev_head_id != dest.id
        ):
            prev_node = await self._ct.repo.get_node(prev_head_id)
            if prev_node is not None and prev_node.parent_id is not None and prev_node.is_dirty:
                self._ct._schedule_bg_summary(self.id, prev_head_id)
        return NodeHandle(self, dest.id)

    def refresh_summary_background(self, node_id: str) -> asyncio.Task[str]:
        """Trigger a non-blocking background Tier-2 summary refresh for `node_id` without closing it."""
        return self._ct._schedule_bg_summary(self.id, node_id)

    async def compact(
        self,
        *,
        node_id: str | None = None,
        keep_recent_turns: int | None = None,
        title: str | None = None,
        reason: str = "",
        switch_head: bool = True,
    ) -> tuple[NodeHandle, dict[str, Any]]:
        """
        Non-destructively compact `node_id` (defaults to HEAD) into a linked continuation child
        branch while preserving the previous branch's verbatim messages & KV cache untouched.
        """
        t = await self.model()
        nid = node_id or t.head_node_id or t.root_node_id
        assert nid is not None
        src = await self._ct.repo.get_node(nid)
        if src is None:
            raise ValueError(f"Node not found: {nid}")
        res = await self._ct.ipc.compact_branch(
            tree=t,
            source=src,
            keep_recent_turns=keep_recent_turns,
            title=title,
            reason=reason,
            switch_head=switch_head,
        )
        return NodeHandle(self, str(res.tool_output["to_node_id"])), res.tool_output

    async def archive(
        self,
        node_id: str | None = None,
        *,
        reason: str = "",
        switch_to_parent: bool = True,
    ) -> dict[str, Any]:
        """Archive a non-root dead-end branch, quarantining its chunks/facts from default active search."""
        t = await self.model()
        nid = node_id or t.head_node_id
        if not nid:
            raise ValueError("No branch node specified or active.")
        caller_id = t.head_node_id or t.root_node_id or nid
        caller = await self._ct.repo.get_node(caller_id)
        if caller is None:
            raise ValueError(f"Node not found: {caller_id}")
        res = await self._ct.ipc.archive_branch(
            tree=t,
            caller=caller,
            target_id=nid,
            reason=reason,
            switch_to_parent=switch_to_parent,
        )
        return res.tool_output

    async def pin(
        self, key: str, value: str, *, source_node_id: str | None = None
    ) -> TreeInvariantModel:
        """Pin or update a bounded tree-wide invariant surfaced in every branch's `<dialogue_tree_context>`."""
        return await self._ct.repo.upsert_tree_invariant(
            self.id,
            key,
            value,
            source_node_id=source_node_id,
            max_items=self._ct.config.guardrails.max_pinned_invariants,
            max_value_chars=self._ct.config.guardrails.max_pinned_value_chars,
        )

    async def unpin(self, key: str) -> bool:
        """Remove a tree-wide pinned invariant by key."""
        return await self._ct.repo.delete_tree_invariant(self.id, key)

    async def invariants(self) -> list[TreeInvariantModel]:
        """List all pinned tree-wide invariants."""
        return await self._ct.repo.list_tree_invariants(self.id)

    async def facts(
        self,
        *,
        node_id: str | None = None,
        active_only: bool = True,
        exclude_archived: bool = True,
    ) -> list[BranchFactModel]:
        """List Tier-2.5 provenance-linked atomic temporal facts (excluding ARCHIVED dead ends by default)."""
        return await self._ct.repo.list_branch_facts(
            self.id,
            node_id=node_id,
            active_only=active_only,
            exclude_archived=exclude_archived,
        )

    async def add_facts(
        self, node_id: str, facts: list[dict[str, Any]], *, source_msg_seq: int | None = None
    ) -> list[BranchFactModel]:
        """Store and index Tier-2.5 atomic temporal facts scoped strictly to `node_id`."""
        created = await self._ct.repo.replace_branch_facts(
            self.id, node_id, facts, source_msg_seq=source_msg_seq
        )
        await self._ct.rag.index_node_facts(node_id)
        return created

    async def back(self) -> NodeHandle:
        return await self.jump("back")

    async def forward(self) -> NodeHandle:
        return await self.jump("forward")

    async def reflog(self, limit: int = 100) -> list[ReflogEntryModel]:
        return await self._ct.repo.list_reflog(self.id, limit=limit)

    async def ipc_log(self, limit: int = 100) -> list[IpcEventModel]:
        return await self._ct.repo.list_ipc_events(self.id, limit=limit)

    async def artifacts(self) -> list[ArtifactModel]:
        return await self._ct.repo.list_artifacts(self.id)

    async def add_artifact(
        self,
        slug: str,
        title: str,
        content: str,
        *,
        kind: str = "markdown",
        summary: str = "",
        created_by_node_id: str | None = None,
    ) -> ArtifactModel:
        return await self._ct.repo.create_artifact(
            self.id,
            ArtifactInput(slug=slug, title=title, kind=kind, content=content, summary=summary),
            created_by_node_id=created_by_node_id,
        )

    async def pending_handoffs(self) -> list[MailboxItemModel]:
        return await self._ct.repo.list_mailbox_items(self.id, status="pending_review")

    async def commit_handoff(
        self,
        mailbox_id: str,
        *,
        summary: str | None = None,
        artifacts: list[ArtifactInput] | None = None,
        return_head_to_target: bool = True,
    ) -> tuple[MailboxItemModel, NodeModel]:
        mb, target, _ = await self._ct.ipc.commit_pending_handoff(
            mailbox_id=mailbox_id,
            edited_summary=summary,
            edited_artifacts=artifacts,
            return_head_to_target=return_head_to_target,
        )
        return mb, target

    # ------------------------------------------------------------------
    # Branch ergonomics & compilation
    # ------------------------------------------------------------------

    async def extract_to_child(
        self,
        parent_node_id: str,
        seq_start: int,
        seq_end: int,
        *,
        title: str,
        goal: str = "",
        summary: str | None = None,
    ) -> tuple[NodeHandle, NodeHandle]:
        parent, child = await self._ct.ops.extract_to_child(
            parent_node_id, seq_start, seq_end, title=title, goal=goal, summary=summary
        )
        return NodeHandle(self, parent.id), NodeHandle(self, child.id)

    async def fork_on_edit(
        self,
        node_id: str,
        at_seq: int,
        *,
        title: str | None = None,
        switch_head: bool = True,
    ) -> NodeHandle:
        forked = await self._ct.ops.fork_on_edit(
            node_id, at_seq, title=title, switch_head=switch_head
        )
        return NodeHandle(self, forked.id)

    async def compare_siblings(
        self, parent_id: str, sibling_ids: list[str] | None = None
    ) -> BranchComparison:
        return await self._ct.ops.compare_siblings(parent_id, sibling_ids=sibling_ids)

    async def cherry_pick(
        self, source_node_id: str, target_node_id: str, excerpt: str, *, note: str = ""
    ) -> MessageModel:
        return await self._ct.ops.cherry_pick(source_node_id, target_node_id, excerpt, note=note)

    async def promote_child(
        self, child_id: str, *, archive_siblings: bool = True, switch_head_to_parent: bool = True
    ) -> NodeHandle:
        child_node = await self._ct.repo.get_node(child_id)
        if child_node is None or not child_node.parent_id:
            raise ValueError("Only a non-root child node can be promoted.")
        parent_before = await self._ct.repo.get_node(child_node.parent_id)
        if parent_before is None:
            raise ValueError(f"Parent node not found: {child_node.parent_id}")

        # Dual-commit atomicity: merge sandbox FIRST and abort WITHOUT deleting worktree on conflict!
        merged_sandbox = False
        if self._ct.sandbox is not None and self._ct.sandbox.get_workspace(child_id) is not None:
            parent_target = (
                parent_before.id
                if self._ct.sandbox.get_workspace(parent_before.id) is not None
                else None
            )
            merge_res = await self._ct.sandbox.merge_into(child_id, target_node_id=parent_target)
            if not merge_res.get("merged"):
                ws = self._ct.sandbox.get_workspace(child_id)
                raise RuntimeError(
                    f"Sandbox merge failed for branch '{child_id}': "
                    f"{merge_res.get('error') or merge_res.get('reason')}. "
                    f"Worktree and git branch preserved at {ws.path if ws else '(unknown)'}."
                )
            merged_sandbox = True

        parent = await self._ct.ops.promote_child(
            child_id,
            archive_siblings=archive_siblings,
            switch_head_to_parent=switch_head_to_parent,
        )

        # Only clean up worktrees AFTER both sandbox merge AND dialogue promotion succeeded!
        if self._ct.sandbox is not None:
            if merged_sandbox:
                await self._ct.sandbox.cleanup(child_id)
            if archive_siblings:
                for sib in await self._ct.repo.get_children(parent.id):
                    if sib.id != child_id and self._ct.sandbox.get_workspace(sib.id) is not None:
                        await self._ct.sandbox.cleanup(sib.id)
        return NodeHandle(self, parent.id)

    async def compile_markdown(
        self,
        *,
        leaf_node_id: str | None = None,
        include_transcript: bool = True,
        include_side_branches: bool = True,
        include_artifacts: bool = True,
    ) -> str:
        return await self._ct.ops.compile_markdown(
            self.id,
            leaf_node_id=leaf_node_id,
            include_transcript=include_transcript,
            include_side_branches=include_side_branches,
            include_artifacts=include_artifacts,
        )

    async def reindex_rag(self) -> dict[str, int]:
        return await self._ct.rag.index_tree(self.id)

    async def spawn_background(
        self,
        goal: str,
        *,
        parent_id: str | None = None,
        title: str | None = None,
        mode: ChildContextMode = "clean",
        summary_brief: str = "",
        slice_turns: int = 6,
        model: str | None = None,
        permission: ToolPermission | None = None,
        max_rounds: int = 3,
        token_budget: int | None = None,
    ) -> tuple[NodeHandle, BackgroundJobModel]:
        """Spawn a non-blocking async background sub-agent that delivers its result to parent's mailbox."""
        t = await self.model()
        pid = parent_id or t.head_node_id or t.root_node_id
        assert pid is not None
        parent = await self._ct.repo.get_node(pid)
        if parent is None:
            raise ValueError(f"Parent node not found: {pid}")
        child, job = await self._ct.actors.spawn_background(
            tree=t,
            parent=parent,
            goal=goal,
            title=title,
            context_mode=mode,
            summary_brief=summary_brief,
            slice_turns=slice_turns,
            model_override=model,
            tool_permission=permission,
            max_rounds=max_rounds,
            token_budget=token_budget,
        )
        return NodeHandle(self, child.id), job

    def background_jobs(self) -> list[BackgroundJobModel]:
        return self._ct.actors.list_jobs(tree_id=self.id)

    async def wait_background(
        self, job_id: str | None = None, *, timeout: float | None = 30.0
    ) -> list[BackgroundJobModel]:
        return await self._ct.actors.wait(job_id=job_id, timeout=timeout)

    async def unread_mailbox(self, node_id: str | None = None) -> list[MailboxItemModel]:
        return await self._ct.repo.list_mailbox_items(
            self.id, to_node_id=node_id, status="unread"
        )

    async def export_directory(self, output_dir: str | Path) -> Path:
        return await self._ct.export_directory(self.id, output_dir)

    async def export_sft_examples(
        self, *, only_tool_turns: bool = False, include_context_envelope: bool = True
    ) -> list[dict[str, Any]]:
        from chatree.eval.trajectories import TrajectoryExporter

        return await TrajectoryExporter.export_sft_examples(
            self._ct.repo,
            self._ct.context,
            self.id,
            only_tool_turns=only_tool_turns,
            include_context_envelope=include_context_envelope,
            tools_registry=self._ct.tools,
        )

    async def export_sft_jsonl(
        self, output_path: str | Path, *, only_tool_turns: bool = False
    ) -> Path:
        from chatree.eval.trajectories import TrajectoryExporter

        examples = await self.export_sft_examples(only_tool_turns=only_tool_turns)
        return TrajectoryExporter.save_jsonl(examples, output_path)

    async def metrics(self) -> TreeMetrics:
        nodes = await self._ct.repo.list_nodes(self.id)
        ipcs = await self._ct.repo.list_ipc_events(self.id, limit=10_000)
        reflog = await self._ct.repo.list_reflog(self.id, limit=10_000)
        arts = await self._ct.repo.list_artifacts(self.id)
        total_msgs = await self._ct.repo.count_tree_messages(self.id)
        emb_count = await self._ct.repo.count_embeddings(self.id)

        t_in = sum(n.tokens_in for n in nodes)
        t_out = sum(n.tokens_out for n in nodes)
        t_cache = sum(n.cache_read_tokens for n in nodes)
        stale_edges = sum(len(n.stale_dependencies) for n in nodes)

        ipc_by_type: dict[str, int] = {}
        shadow_turns = 0
        shadow_cache_hits = 0
        cached_hits = 0
        blocks = 0
        for e in ipcs:
            ipc_by_type[e.event_type] = ipc_by_type.get(e.event_type, 0) + 1
            if e.event_type == "request_shadow" and e.status == "ok":
                if e.response.get("cache_hit") is True:
                    shadow_cache_hits += 1
                else:
                    shadow_turns += 1
            if e.event_type == "request_summary" and e.response.get("cache_hit") is True:
                cached_hits += 1
            if e.status.startswith("rejected_"):
                blocks += 1

        return TreeMetrics(
            tree_id=self.id,
            node_count=len(nodes),
            active_nodes=sum(1 for n in nodes if n.status == "ACTIVE"),
            completed_nodes=sum(1 for n in nodes if n.status == "COMPLETED"),
            max_depth=max((n.depth for n in nodes), default=0),
            total_messages=total_msgs,
            total_tokens_in=t_in,
            total_tokens_out=t_out,
            total_cache_read_tokens=t_cache,
            cache_hit_ratio=round(t_cache / t_in, 4) if t_in > 0 else 0.0,
            ipc_total=len(ipcs),
            ipc_by_type=ipc_by_type,
            shadow_turns=shadow_turns,
            shadow_cache_hits=shadow_cache_hits,
            cached_summary_hits=cached_hits,
            guardrail_blocks=blocks,
            stale_edges_count=stale_edges,
            reflog_jumps=sum(1 for r in reflog if r.action in ("jump", "return", "reopen")),
            artifact_count=len(arts),
            indexed_embeddings=emb_count,
        )


class NodeHandle:
    """Ergonomic handle for interacting directly with a specific tree node."""

    def __init__(self, tree: DialogueTree, node_id: str) -> None:
        self.tree = tree
        self.id = node_id

    async def model(self) -> NodeModel:
        n = await self.tree._ct.repo.get_node(self.id)
        if n is None:
            raise ValueError(f"Node not found: {self.id}")
        return n

    async def promote_to_chat(self) -> NodeModel:
        """Promote a `subagent` worker node into a first-class interactive `chat` branch."""
        updated = await self.tree._ct.repo.update_node(self.id, node_kind="chat")
        if updated is None:
            raise ValueError(f"Node not found: {self.id}")
        return updated

    async def messages(self) -> list[MessageModel]:
        return await self.tree._ct.repo.list_messages(self.id)

    async def stale_dependencies(self) -> list[StaleDependencyInfo]:
        return (await self.model()).stale_dependencies

    async def chat(self, content: str, **kwargs: Any) -> TurnResult:
        return await self.tree.chat(content, node_id=self.id, **kwargs)

    async def spawn(self, goal: str, **kwargs: Any) -> NodeHandle:
        return await self.tree.spawn(goal, parent_id=self.id, **kwargs)

    async def spawn_background(
        self, goal: str, **kwargs: Any
    ) -> tuple[NodeHandle, BackgroundJobModel]:
        return await self.tree.spawn_background(goal, parent_id=self.id, **kwargs)

    async def compact(
        self,
        *,
        keep_recent_turns: int | None = None,
        title: str | None = None,
        reason: str = "",
        switch_head: bool = True,
    ) -> tuple[NodeHandle, dict[str, Any]]:
        """Non-destructively compact this branch into a linked continuation child branch."""
        return await self.tree.compact(
            node_id=self.id,
            keep_recent_turns=keep_recent_turns,
            title=title,
            reason=reason,
            switch_head=switch_head,
        )

    async def archive(
        self,
        *,
        reason: str = "",
        switch_to_parent: bool = True,
    ) -> dict[str, Any]:
        """Mark this non-root branch ARCHIVED to quarantine its turns/facts from active search."""
        return await self.tree.archive(
            node_id=self.id,
            reason=reason,
            switch_to_parent=switch_to_parent,
        )

    async def request(
        self, target_id: str, *, prompt: str | None = None, mutate: bool = False
    ) -> dict[str, Any]:
        return await self.tree.request(
            target_id, prompt=prompt, mutate=mutate, caller_id=self.id
        )

    async def finish(
        self,
        *,
        summary: str | None = None,
        artifacts: list[ArtifactInput] | None = None,
        auto_commit: bool | None = None,
        return_to_parent: bool = True,
    ) -> dict[str, Any]:
        node = await self.model()
        if not node.parent_id:
            raise ValueError("Cannot finish the root node into a parent.")
        t = await self.tree.model()
        summary_text = (summary or "").strip()
        if not summary_text:
            trace_ctx = TraceContext.start(
                node.id,
                self.tree._ct.config.guardrails.max_hop_depth,
                self.tree._ct.config.guardrails.token_budget,
            )
            refreshed, *_ = await self.tree._ct.staleness.ensure_fresh_summary(
                t, node, trace_ctx=trace_ctx, force=True
            )
            summary_text = refreshed.summary or f"Completed '{node.title}'."
            node = refreshed

        combined_artifacts = list(artifacts or [])
        if self.tree._ct.sandbox is not None and self.tree._ct.sandbox.get_workspace(node.id) is not None:
            diff_art = await self.tree._ct.sandbox.diff_artifact(
                node.id, title=f"Git Diff ({node.title})"
            )
            if diff_art is not None:
                combined_artifacts.append(diff_art)

        assert node.parent_id is not None
        res = await self.tree._ct.ipc.send(
            tree=t,
            sender=node,
            target_id=node.parent_id,
            content=summary_text,
            artifacts=combined_artifacts,
            finish=True,
            auto_commit_override=auto_commit,
            return_to_parent=return_to_parent,
        )
        return {
            "result": res.tool_output,
            "handoff_pending": res.handoff_pending,
            "ipc_event": res.ipc_event,
        }
