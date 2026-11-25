from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from chatree.sdk.client import Chatree
from chatree.server.dtos import (
    ArchiveBranchRequest,
    AskTreePayload,
    CommitHandoffRequest,
    CompactBranchRequest,
    CreateTurnRequest,
    DirectRequestPayload,
    ExtractTangentRequest,
    FinishNodeRequest,
    ForkEditRequest,
    NodeDetailResponse,
    SpawnBackgroundRequest,
    SpawnNodeRequest,
    UpdateNodeRequest,
    get_client,
)
from chatree.types import BranchComparison, NodeModel

router = APIRouter()


@router.get("/api/nodes/{node_id}", response_model=NodeDetailResponse)
async def get_node(node_id: str, c: Chatree = Depends(get_client)) -> NodeDetailResponse:
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


@router.patch("/api/nodes/{node_id}", response_model=NodeModel)
async def update_node(
    node_id: str, body: UpdateNodeRequest, c: Chatree = Depends(get_client)
) -> NodeModel:
    node = await c.repo.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    data = body.model_dump(exclude_unset=True)
    if "title" in data and data["title"] is not None:
        data["title_source"] = "user"
    if "summary" in data and data["summary"] is not None:
        updated = await c.repo.commit_node_summary(
            node_id, summary=data["summary"], status=data.get("status")
        )
        other = {k: v for k, v in data.items() if k not in ("summary", "status")}
        if other:
            updated = await c.repo.update_node(node_id, **other) or updated
        await c.rag.index_node_summary(node_id)
        await c.rag.propagate_rollups_upward(node_id)
        return updated
    updated_node = await c.repo.update_node(node_id, **data)
    assert updated_node is not None
    if "title" in data and data["title"] is not None:
        await c.rag.index_node_summary(node_id)
        await c.rag.propagate_rollups_upward(node_id)
    return updated_node


@router.post("/api/nodes/{node_id}/turns/stream")
async def stream_turn(
    node_id: str, body: CreateTurnRequest, c: Chatree = Depends(get_client)
) -> StreamingResponse:
    if await c.repo.get_node(node_id) is None:
        raise HTTPException(status_code=404, detail="Node not found")
    rt = await c.get_runtime_settings()

    async def event_stream() -> AsyncIterator[str]:
        async for ev in c.orchestrator.run_turn_stream(
            node_id=node_id,
            user_content=body.content,
            session_id=body.session_id,
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


@router.post("/api/nodes/{node_id}/spawn")
async def spawn_child(
    node_id: str, body: SpawnNodeRequest, c: Chatree = Depends(get_client)
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
        session_id=body.session_id,
        import_branch_ids=body.import_branch_ids or None,
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


@router.post("/api/nodes/{node_id}/spawn-background", status_code=202)
async def spawn_background(
    node_id: str, body: SpawnBackgroundRequest, c: Chatree = Depends(get_client)
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


@router.post("/api/nodes/{node_id}/finish")
async def finish_node(
    node_id: str, body: FinishNodeRequest, c: Chatree = Depends(get_client)
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
        session_id=body.session_id,
    )
    updated_node = await c.repo.get_node(node_id)
    updated_sess = await c.repo.get_session(node.tree_id, body.session_id)
    next_head_id = updated_sess.head_node_id if updated_sess else None
    return {
        "node": updated_node.model_dump() if updated_node else None,
        "result": fin["result"],
        "handoff_pending": fin["handoff_pending"].model_dump() if fin["handoff_pending"] else None,
        "ipc_event": fin["ipc_event"].model_dump() if fin["ipc_event"] else None,
        "head_changed": (
            {"from_node_id": node_id, "to_node_id": next_head_id}
            if next_head_id and next_head_id != node_id
            else None
        ),
    }


@router.post("/api/nodes/{node_id}/compact")
async def compact_node(
    node_id: str, body: CompactBranchRequest, c: Chatree = Depends(get_client)
) -> dict[str, Any]:
    node = await c.repo.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    dt = await c.get_tree(node.tree_id)
    _, info = await dt.compact(
        node_id=node_id,
        keep_recent_turns=body.keep_recent_turns,
        title=body.title,
        reason=body.reason,
        switch_head=body.switch_head,
        session_id=body.session_id,
    )
    return info


@router.post("/api/nodes/{node_id}/archive")
async def archive_node(
    node_id: str, body: ArchiveBranchRequest, c: Chatree = Depends(get_client)
) -> dict[str, Any]:
    node = await c.repo.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    dt = await c.get_tree(node.tree_id)
    out = await dt.archive(
        node_id=node_id,
        reason=body.reason,
        switch_to_parent=body.switch_to_parent,
        session_id=body.session_id,
    )
    if out.get("error"):
        raise HTTPException(status_code=400, detail=str(out.get("detail") or out["error"]))
    return out


@router.post("/api/nodes/{node_id}/dismiss-stale/{source_node_id}")
async def dismiss_stale(
    node_id: str, source_node_id: str, c: Chatree = Depends(get_client)
) -> dict[str, Any]:
    await c.repo.dismiss_dependency(node_id, source_node_id)
    node = await c.repo.get_node(node_id)
    return {"ok": True, "node": node.model_dump() if node else None}


@router.post("/api/nodes/{node_id}/extract")
async def extract_to_child(
    node_id: str, body: ExtractTangentRequest, c: Chatree = Depends(get_client)
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


@router.post("/api/nodes/{node_id}/fork-edit", response_model=NodeModel)
async def fork_on_edit(
    node_id: str, body: ForkEditRequest, c: Chatree = Depends(get_client)
) -> NodeModel:
    try:
        return await c.ops.fork_on_edit(
            node_id, body.at_seq, title=body.title, switch_head=body.switch_head
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/nodes/{node_id}/compare", response_model=BranchComparison)
async def compare_siblings(
    node_id: str, c: Chatree = Depends(get_client)
) -> BranchComparison:
    try:
        return await c.ops.compare_siblings(node_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/nodes/{node_id}/promote", response_model=NodeModel)
async def promote_child(
    node_id: str,
    archive_siblings: bool = Query(default=True),
    c: Chatree = Depends(get_client),
) -> NodeModel:
    try:
        return await c.ops.promote_child(node_id, archive_siblings=archive_siblings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/nodes/{node_id}/request")
async def direct_request(
    node_id: str, body: DirectRequestPayload, c: Chatree = Depends(get_client)
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


@router.post("/api/nodes/{node_id}/ask")
async def ask_tree(
    node_id: str, body: AskTreePayload, c: Chatree = Depends(get_client)
) -> dict[str, Any]:
    caller = await c.repo.get_node(node_id)
    if caller is None:
        raise HTTPException(status_code=404, detail="Caller node not found")
    tree = await c.repo.get_tree(caller.tree_id)
    assert tree is not None
    if body.deep:
        out = await c.query_engine.deep_ask(
            tree=tree, caller=caller, query=body.query, scope=body.scope, max_targets=body.max_targets
        )
        return {"result": out, "ipc_event": None}
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


@router.post("/api/mailbox/{mailbox_id}/commit")
async def commit_handoff(
    mailbox_id: str, body: CommitHandoffRequest, c: Chatree = Depends(get_client)
) -> dict[str, Any]:
    try:
        mb, target, rf = await c.ipc.commit_pending_handoff(
            mailbox_id=mailbox_id,
            edited_summary=body.summary,
            edited_artifacts=body.artifacts,
            return_head_to_target=body.return_head_to_target,
            session_id=body.session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "mailbox_item": mb.model_dump(),
        "target_node": target.model_dump(),
        "reflog_entry": rf.model_dump() if rf else None,
    }


@router.post("/api/mailbox/{mailbox_id}/dismiss")
async def dismiss_handoff(
    mailbox_id: str, c: Chatree = Depends(get_client)
) -> dict[str, Any]:
    updated = await c.repo.update_mailbox_item(mailbox_id, status="dismissed")
    if updated is None:
        raise HTTPException(status_code=404, detail="Mailbox item not found")
    return {"mailbox_item": updated.model_dump()}
