from __future__ import annotations

from chatree import Chatree, DefaultModelRouter, STATIC_SYSTEM_PROMPT


async def test_sdk_cow_fork_prefix_and_isolation():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("System Design", goal="Design chatree")
        root = await tree.root()
        await ct.repo.append_message(root.id, tree.id, "user", "Let's use SQLite WAL.")
        await ct.repo.append_message(root.id, tree.id, "assistant", "Agreed, WAL mode is great.")

        child = await root.spawn("Design auth strategy", title="Auth", mode="fork")
        child_model = await child.model()
        assert child_model.spawn_msg_seq == 2

        # Parent receives unrelated message AFTER child forked
        await ct.repo.append_message(root.id, tree.id, "user", "Unrelated billing question.")
        await ct.repo.append_message(child.id, tree.id, "user", "Use short-lived Ed25519 JWTs.")

        grandchild = await child.spawn("Token rotation", title="Rotation", mode="fork")
        gc_prefix = await ct.context.build_static_prefix(await grandchild.model())
        assert gc_prefix[0] == {"role": "system", "content": STATIC_SYSTEM_PROMPT}
        contents = [m["content"] for m in gc_prefix[1:]]
        assert contents == [
            "Let's use SQLite WAL.",
            "Agreed, WAL mode is great.",
            "Use short-lived Ed25519 JWTs.",
        ]
        assert all("billing" not in c for c in contents)


async def test_fork_boundary_avoids_dangling_tool_calls():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Tool Safety")
        root = await tree.root()
        await ct.repo.append_message(root.id, tree.id, "user", "Branch into caching.")
        await ct.repo.append_message(
            root.id,
            tree.id,
            "assistant",
            "",
            tool_calls=[
                {
                    "id": "call_open_1",
                    "type": "function",
                    "function": {"name": "dialogue_tree_create_branch", "arguments": '{"goal": "Caching"}'},
                }
            ],
        )

        child = await root.spawn("Caching", mode="fork")
        cm = await child.model()
        assert cm.spawn_msg_seq == 1
        prefix = await ct.context.build_static_prefix(cm)
        assert len(prefix) == 2
        assert "tool_calls" not in prefix[-1]


async def test_static_prefix_immutable_across_dynamic_state_and_custom_router():
    routed_roles: list[str] = []

    def policy_router(_tree, _node, role):
        routed_roles.append(role)
        return "mock/chatree-demo"

    async with Chatree.in_memory(router=DefaultModelRouter(override_fn=policy_router)) as ct:
        tree = await ct.create_tree("Cache & Router")
        root = await tree.root()
        await root.chat("Turn 1 message")

        prefix_before = await ct.context.build_static_prefix(await root.model())
        child = await root.spawn("Side tangent", mode="clean")
        await tree.jump(root.id, note="Back to root")
        prefix_after = await ct.context.build_static_prefix(await root.model())

        # Static prefix must stay byte-for-byte identical for KV prefix cache hits!
        assert prefix_before == prefix_after
        env, cue_ids, _ = await ct.context.build_ephemeral_envelope(
            await tree.model(), await root.model()
        )
        assert child.id in env
        assert "ORIENTATION:" in env
        assert len(cue_ids) >= 1
        assert "chat" in routed_roles
