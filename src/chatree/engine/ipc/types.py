from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chatree.types import IpcEventModel, MailboxItemModel, ReflogEntryModel


@dataclass
class IpcExecutionResult:
    tool_output: dict[str, Any]
    ipc_event: IpcEventModel | None = None
    head_changed: tuple[str, str, ReflogEntryModel] | None = None
    handoff_pending: MailboxItemModel | None = None
    branch_suggested: dict[str, Any] | None = None
    stale_node_ids: list[str] = field(default_factory=list)
