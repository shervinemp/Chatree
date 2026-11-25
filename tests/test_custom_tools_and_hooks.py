from __future__ import annotations

import re

import pytest

from chatree import (
    CORE_TOOL_NAMES,
    CORE_TOOL_NAMESPACE,
    Chatree,
    ToolCallContext,
    format_namespaced_tool,
)


async def test_custom_tools_permission_scoping_and_lifecycle_hooks():
    async with Chatree.in_memory() as ct:
        hook_events: list[str] = []

        @ct.on("turn_start")
        def on_start(_p):
            hook_events.append("turn_start")

        @ct.on("spawn")
        async def on_spawn(_p):
            hook_events.append("spawn")

        @ct.on("custom_tool")
        def on_custom(p):
            hook_events.append(f"custom:{p['tool_name']}:{p['status']}")

        @ct.on("turn_complete")
        def on_done(_p):
            hook_events.append("turn_complete")

        @ct.tool(side_effect="read_only", allowed_in_shadow=True)
        def get_quota(service: str) -> dict:
            """Return current quota for a service."""
            return {"service": service, "qps_limit": 500}

        @ct.tool(side_effect="write")
        def mutate_quota(service: str, qps_limit: int, ctx: ToolCallContext) -> dict:
            """Mutate service quota (write-scoped!)."""
            return {"updated": True, "service": service, "qps_limit": qps_limit, "node": ctx.node.id}

        tree = await ct.create_tree("Scoped Tools")
        root = await tree.root()

        # Root defaults to read_write -> write tool allowed
        r_root = await root.chat('use tool mutate_quota {"service": "auth", "qps_limit": 900}')
        assert "900" in r_root.content
        assert "custom:mutate_quota:ok" in hook_events

        # Default child has read_only permission -> read_only tool allowed, write tool BLOCKED!
        ro_child = await root.spawn("Audit branch", title="Audit", mode="clean")
        assert (await ro_child.model()).tool_permission == "read_only"

        r_ro_ok = await ro_child.chat('use tool get_quota {"service": "auth"}')
        assert "500" in r_ro_ok.content
        assert "custom:get_quota:ok" in hook_events

        r_ro_blocked = await ro_child.chat(
            'use tool mutate_quota {"service": "auth", "qps_limit": 1000}'
        )
        assert "PERMISSION_DENIED" in r_ro_blocked.content
        assert "custom:mutate_quota:rejected_permission" in hook_events

        # Explicitly write-enabled child can invoke write tool
        rw_child = await root.spawn(
            "Apply fix", title="Writer", mode="clean", permission="read_write"
        )
        assert (await rw_child.model()).tool_permission == "read_write"
        r_rw_ok = await rw_child.chat('use tool mutate_quota {"service": "auth", "qps_limit": 1200}')
        assert "1200" in r_rw_ok.content

        assert "spawn" in hook_events
        assert "turn_start" in hook_events
        assert "turn_complete" in hook_events


async def test_tool_namespacing_and_collision_freedom():
    # All core dialogue-tree primitives live cleanly under semantic `dialogue_tree_*` and are provider-safe
    assert CORE_TOOL_NAMESPACE == "dialogue_tree"
    assert CORE_TOOL_NAMES == {
        "dialogue_tree_create_branch",
        "dialogue_tree_send_to_branch",
        "dialogue_tree_query_branch",
        "dialogue_tree_search_branches",
        "dialogue_tree_switch_branch",
        "dialogue_tree_search_and_query",
        "dialogue_tree_compact_branch",
        "dialogue_tree_pin_invariant",
        "dialogue_tree_archive_branch",
    }
    for tname in CORE_TOOL_NAMES:
        assert tname.startswith("dialogue_tree_")
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tname)

    assert format_namespaced_tool("ue5", "spawn_actor") == "ue5_spawn_actor"
    assert format_namespaced_tool("ue5", "ue5_spawn_actor") == "ue5_spawn_actor"

    async with Chatree.in_memory() as ct:
        # 1. Bare external tools named `send` and `request` NO LONGER collide with Chatree!
        @ct.tool(name="send", side_effect="write")
        def slack_send(channel: str, text: str) -> dict:
            return {"channel": channel, "sent": text}

        @ct.tool(name="request", side_effect="read_only", allowed_in_shadow=True)
        def http_request(url: str) -> dict:
            return {"url": url, "status": 200}

        # 2. UE5 / Unity-style plugin namespaces isolate identical raw tool names cleanly
        ue5 = ct.tool_namespace("ue5")
        unity = ct.tool_namespace("unity")

        @ue5.tool(side_effect="write")
        def spawn_actor(blueprint: str) -> dict:
            return {"engine": "ue5", "blueprint": blueprint}

        @ct.tool(namespace="ue5", side_effect="read_only", allowed_in_shadow=True)
        def get_level() -> dict:
            return {"engine": "ue5", "level": "MainMap"}

        @unity.tool(side_effect="write")
        def spawn_actor_unity(prefab: str) -> dict:
            return {"engine": "unity", "prefab": prefab}

        # Registering same raw name `spawn_actor` in a different namespace succeeds cleanly
        unity.register(lambda prefab: {"engine": "unity", "prefab": prefab}, name="spawn_actor", side_effect="write")

        assert {s.name for s in ue5.list_tools()} == {"ue5_spawn_actor", "ue5_get_level"}
        assert {s.name for s in unity.list_tools()} == {"unity_spawn_actor_unity", "unity_spawn_actor"}

        # 3. Reserved `dialogue_tree` namespace and duplicate names are strictly protected
        with pytest.raises(ValueError, match="reserved"):
            ct.tool_namespace("dialogue_tree")
        with pytest.raises(ValueError, match="reserved"):
            ct.tools.register(lambda: {}, namespace="dialogue_tree", name="custom")
        with pytest.raises(ValueError, match="reserved"):
            ct.tools.register(lambda: {}, name="dialogue_tree_custom")
        with pytest.raises(ValueError, match="already registered"):
            ue5.register(lambda blueprint: {}, name="spawn_actor")
        with pytest.raises(ValueError, match="Invalid tool name"):
            ct.tools.register(lambda: {}, name="ue5.spawn_actor")

        # 4. Both `chatree_*` core tools AND bare/namespaced custom tools coexist in schemas & runtime!
        schemas = ct.tools.schemas_for_node("read_write", include_core=True, include_compound_ask=True)
        schema_names = {s["function"]["name"] for s in schemas}
        assert CORE_TOOL_NAMES.issubset(schema_names)
        assert {"send", "request", "ue5_spawn_actor", "ue5_get_level", "unity_spawn_actor"}.issubset(
            schema_names
        )

        tree = await ct.create_tree("Namespaced Tools")
        root = await tree.root()

        r_slack = await root.chat('use tool send {"channel": "#ops", "text": "deploy done"}')
        assert "deploy done" in r_slack.content
        r_http = await root.chat('use tool request {"url": "https://example.com"}')
        assert "200" in r_http.content
        r_ue5 = await root.chat('use tool ue5_spawn_actor {"blueprint": "BP_Player"}')
        assert "BP_Player" in r_ue5.content

        # Read-only child blocks write-scoped `ue5_spawn_actor` while allowing `ue5_get_level`
        ro_child = await root.spawn("Inspect scene", title="Scene Inspector", mode="clean")
        r_ro_ok = await ro_child.chat("use tool ue5_get_level {}")
        assert "MainMap" in r_ro_ok.content
        r_ro_deny = await ro_child.chat('use tool ue5_spawn_actor {"blueprint": "BP_Enemy"}')
        assert "PERMISSION_DENIED" in r_ro_deny.content
