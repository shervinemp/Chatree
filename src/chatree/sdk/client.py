from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, cast

from chatree.config import ChatreeConfig
from chatree.engine.actors import BackgroundActorManager
from chatree.engine.context import ContextAssembler
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
from chatree.sandbox import BranchSandbox
from chatree.sdk.node import NodeHandle
from chatree.sdk.session import TreeSession
from chatree.sdk.tree import DialogueTree
from chatree.storage.database import Database
from chatree.storage.portable import DialogueDirectoryStore
from chatree.storage.repository import SQLiteRepository
from chatree.tools import STATIC_SYSTEM_PROMPT, ToolRegistry, ToolSideEffect
from chatree.types import (
    RuntimeSettingsModel,
    TreeModel,
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
            on_first_turn_title=self._schedule_bg_title,
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

    def _schedule_bg_title(self, tree_id: str, node_id: str) -> asyncio.Task[str]:
        """Schedule a non-blocking Turn-1 semantic title generation for a placeholder node."""
        task_key = f"title:{node_id}"
        existing = self._bg_summary_tasks.get(task_key)
        if existing is not None and not existing.done():
            return existing

        async def _worker() -> str:
            try:
                tree = await self.repo.get_tree(tree_id)
                node = await self.repo.get_node(node_id)
                if tree is None or node is None or node.title_source != "placeholder":
                    return node.title if node else ""
                prev_title = node.title
                updated = await self.staleness.maybe_generate_initial_title(tree, node)
                if updated.title != prev_title and self.hooks:
                    await self.hooks.emit(
                        "node_titled",
                        {
                            "tree_id": tree_id,
                            "node_id": node_id,
                            "old_title": prev_title,
                            "title": updated.title,
                            "title_source": updated.title_source,
                        },
                    )
                return updated.title
            except asyncio.CancelledError:
                raise
            except Exception:
                return ""
            finally:
                cur = self._bg_summary_tasks.get(task_key)
                if cur is asyncio.current_task():
                    self._bg_summary_tasks.pop(task_key, None)

        task: asyncio.Task[str] = asyncio.create_task(
            _worker(), name=f"chatree-bg-title-{node_id}"
        )
        self._bg_summary_tasks[task_key] = task
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

__all__ = ["Chatree", "DialogueTree", "NodeHandle", "TreeSession"]
