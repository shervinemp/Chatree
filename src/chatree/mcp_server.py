from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from chatree.sdk.client import Chatree


MCP_PROTOCOL_VERSION = "2024-11-05"

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "dialogue_tree_list_trees",
        "description": "List all dialogue trees stored in the local database.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "dialogue_tree_inspect_tree",
        "description": "Inspect a dialogue tree's branch hierarchy, statuses, summaries, and staleness.",
        "inputSchema": {
            "type": "object",
            "properties": {"tree_id": {"type": "string"}},
            "required": ["tree_id"],
        },
    },
    {
        "name": "dialogue_tree_search_branches",
        "description": "Search across a dialogue tree's branches (chunks, summaries, subtree rollups) for matching discussions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tree_id": {"type": "string"},
                "query": {"type": "string"},
                "scope": {
                    "type": "string",
                    "enum": ["all", "ancestors", "subtree", "siblings"],
                    "default": "all",
                },
                "node_id": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["tree_id", "query"],
        },
    },
    {
        "name": "dialogue_tree_query_branch",
        "description": (
            "Query a specific branch in a dialogue tree. Omit prompt for zero-cost cached summary; "
            "provide prompt for a read-only shadow turn against that branch's full context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tree_id": {"type": "string"},
                "target_node_id": {"type": "string"},
                "prompt": {"type": "string"},
                "caller_node_id": {"type": "string"},
            },
            "required": ["tree_id", "target_node_id"],
        },
    },
    {
        "name": "dialogue_tree_search_and_query",
        "description": (
            "Search a dialogue tree's branches and immediately run parallel read-only shadow queries "
            "against the top matching branches in one step."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tree_id": {"type": "string"},
                "query": {"type": "string"},
                "scope": {
                    "type": "string",
                    "enum": ["all", "ancestors", "subtree", "siblings"],
                    "default": "all",
                },
                "max_targets": {"type": "integer", "default": 2},
                "caller_node_id": {"type": "string"},
            },
            "required": ["tree_id", "query"],
        },
    },
    {
        "name": "dialogue_tree_export_markdown",
        "description": "Export a dialogue tree's active root-to-leaf path, resolved side branches, and artifacts as Markdown.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tree_id": {"type": "string"},
                "leaf_node_id": {"type": "string"},
            },
            "required": ["tree_id"],
        },
    },
]


class ChatreeMCPServer:
    """
    Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 server exposing Chatree trees
    to external coding agents and tools (`search_tree`, `request`, `compile_markdown`).
    """

    def __init__(self, client: Chatree) -> None:
        self.client = client

    async def handle_message(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        # Notifications have no id -> no response required
        if msg_id is None and method and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                result: Any = {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "chatree", "version": "0.2.0"},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": MCP_TOOLS}
            elif method == "tools/call":
                tool_name = str(params.get("name") or "")
                arguments = params.get("arguments") or {}
                payload = await self._call_tool(tool_name, arguments)
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": payload if isinstance(payload, str) else json.dumps(payload, indent=2),
                        }
                    ],
                    "isError": False,
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    async def _call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "dialogue_tree_list_trees":
            return [t.model_dump() for t in await self.client.list_trees()]

        if name == "dialogue_tree_inspect_tree":
            tree = await self.client.get_tree(str(args["tree_id"]))
            tm = await tree.model()
            nodes = await tree.nodes()
            return {"tree": tm.model_dump(), "nodes": [n.model_dump() for n in nodes]}

        if name == "dialogue_tree_search_branches":
            tree = await self.client.get_tree(str(args["tree_id"]))
            hits = await tree.search(
                query=str(args["query"]),
                scope=args.get("scope", "all"),
                node_id=args.get("node_id"),
                top_k=int(args.get("top_k", 5)),
            )
            return [h.model_dump() for h in hits]

        if name == "dialogue_tree_query_branch":
            tree = await self.client.get_tree(str(args["tree_id"]))
            return await tree.request(
                target_id=str(args["target_node_id"]),
                prompt=args.get("prompt"),
                mutate=False,
                caller_id=args.get("caller_node_id"),
            )

        if name == "dialogue_tree_search_and_query":
            tree = await self.client.get_tree(str(args["tree_id"]))
            return await tree.ask(
                query=str(args["query"]),
                scope=args.get("scope", "all"),
                max_targets=int(args.get("max_targets", 2)),
                caller_id=args.get("caller_node_id"),
            )

        if name == "dialogue_tree_export_markdown":
            tree = await self.client.get_tree(str(args["tree_id"]))
            return await tree.compile_markdown(leaf_node_id=args.get("leaf_node_id"))

        raise ValueError(f"Unknown MCP tool: {name}")


async def run_stdio_server(db_path: str | Path = "./data/chatree.db") -> None:
    async with Chatree.open(db_path) as ct:
        server = ChatreeMCPServer(ct)
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = await server.handle_message(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./data/chatree.db"
    asyncio.run(run_stdio_server(db_path=db_path))
