from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Literal

HookEventName = Literal[
    "turn_start",
    "turn_complete",
    "spawn",
    "branch_suggested",
    "ipc_before",
    "ipc_after",
    "handoff_created",
    "handoff_committed",
    "stale_propagated",
    "reflog_jump",
    "branch_extracted",
    "branch_promoted",
    "branch_compacted",
    "branch_archived",
    "node_titled",
    "subagent_promoted",
    "background_started",
    "background_completed",
    "background_failed",
    "custom_tool",
    "error",
]

HookCallback = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class HookBus:
    """
    Lightweight async/sync lifecycle event bus for observability, custom telemetry,
    audit logging, approval gates, and UI bridges.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self._listeners: dict[str, list[HookCallback]] = {}

    def register(self, event: HookEventName | Literal["*"], callback: HookCallback) -> HookCallback:
        self._listeners.setdefault(event, []).append(callback)
        return callback

    def on(self, event: HookEventName | Literal["*"]) -> Callable[[HookCallback], HookCallback]:
        def decorator(fn: HookCallback) -> HookCallback:
            return self.register(event, fn)

        return decorator

    def unregister(self, event: str, callback: HookCallback) -> None:
        listeners = self._listeners.get(event)
        if listeners and callback in listeners:
            listeners.remove(callback)

    async def emit(self, event: HookEventName, payload: dict[str, Any]) -> None:
        enriched = {"event": event, **payload}
        callbacks = [*self._listeners.get(event, []), *self._listeners.get("*", [])]
        for cb in callbacks:
            try:
                res = cb(enriched)
                if inspect.isawaitable(res):
                    await res
            except Exception:
                if self.strict:
                    raise
