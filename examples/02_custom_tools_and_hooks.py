"""Registering custom domain tools with per-branch read_only / write scoping + lifecycle hooks."""

import asyncio
from chatree import Chatree, ToolCallContext


async def main() -> None:
    async with Chatree.in_memory() as ct:
        events: list[str] = []

        @ct.on("ipc_after")
        def log_ipc(payload: dict) -> None:
            events.append(f"{payload['tool_name']} (caller={payload['caller_id']})")

        @ct.tool(side_effect="read_only", allowed_in_shadow=True)
        def lookup_service_owner(service: str) -> dict:
            """Look up the owning engineering team for a microservice."""
            return {"service": service, "owner": "platform-security"}

        @ct.tool(side_effect="write")
        def deploy_config(env: str, ctx: ToolCallContext) -> dict:
            """Deploy configuration changes (write-scoped; blocked on read_only child branches!)."""
            return {"deployed": True, "env": env, "by_node": ctx.node.id}

        tree = await ct.create_tree("Tool Scoping Demo")
        root = await tree.root()

        # Root has read_write permission -> write tool succeeds
        res_root = await root.chat('use tool deploy_config {"env": "staging"}')
        print("Root (read_write):", res_root.content)

        # Spawned child defaults to read_only -> read_only tool succeeds, write tool is blocked!
        child = await root.spawn("Investigate config safely", title="Read-Only Audit")
        res_ro = await child.chat('use tool lookup_service_owner {"service": "auth"}')
        print("Child read_only tool:", res_ro.content)

        res_blocked = await child.chat('use tool deploy_config {"env": "prod"}')
        print("Child write tool blocked:", res_blocked.content)
        print("Captured lifecycle hooks:", events)


if __name__ == "__main__":
    asyncio.run(main())
