from __future__ import annotations

from chatree import ArtifactInput, Chatree


async def test_extract_to_child_refactors_turns_into_stub_and_child():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Refactor Test")
        root = await tree.root()
        await ct.repo.append_message(root.id, tree.id, "user", "Turn 1: overall API architecture.")
        await ct.repo.append_message(root.id, tree.id, "user", "Turn 2: tangent on Redis TTLs.")
        await ct.repo.append_message(root.id, tree.id, "assistant", "Turn 3: set 300s sliding TTL.")
        await ct.repo.append_message(root.id, tree.id, "user", "Turn 4: back to GraphQL schema.")

        updated_parent, child = await tree.extract_to_child(
            root.id,
            seq_start=2,
            seq_end=3,
            title="Redis TTL Tangent",
            summary="Use 300s sliding TTL for Redis cache keys.",
        )

        # Child contains the extracted 2 turns and is COMPLETED
        child_msgs = await child.messages()
        assert len(child_msgs) == 2
        assert "Redis TTLs" in child_msgs[0].content
        assert (await child.model()).status == "COMPLETED"

        # Parent collapsed 4 turns -> 3 turns (Turn 1, extracted stub, Turn 4)
        parent_msgs = await updated_parent.messages()
        assert len(parent_msgs) == 3
        assert "Extracted tangent" in parent_msgs[1].content
        assert child.id in parent_msgs[1].content
        assert "GraphQL schema" in parent_msgs[2].content


async def test_fork_on_edit_preserves_immutability():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Immutable Edit")
        root = await tree.root()
        await ct.repo.append_message(root.id, tree.id, "user", "Step 1: init project.")
        await ct.repo.append_message(root.id, tree.id, "assistant", "Step 2: initialized.")
        await ct.repo.append_message(root.id, tree.id, "user", "Step 3: use MySQL.")

        edited_branch = await tree.fork_on_edit(root.id, at_seq=3, title="Postgres Alternative")
        em = await edited_branch.model()
        assert em.spawn_msg_seq == 2
        assert em.context_mode == "fork"

        # Original root still keeps all 3 messages untouched!
        assert len(await root.messages()) == 3
        assert "MySQL" in (await root.messages())[2].content

        # Edited branch inherits only turns 1..2, NOT MySQL
        prefix = await ct.context.build_static_prefix(em)
        assert all("MySQL" not in m.get("content", "") for m in prefix)


async def test_compare_siblings_cherry_pick_promote_and_compile_markdown():
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Architecture Options", goal="Compare Option A vs B")
        root = await tree.root()

        opt_a = await root.spawn("Evaluate Postgres pgvector", title="Option A: Postgres", mode="clean")
        await opt_a.chat("Postgres pgvector supports ACID transactions and HNSW indexes.")
        await ct.repo.commit_node_summary(
            opt_a.id, "Postgres pgvector offers ACID transactions, SQL joins, and HNSW.", status="COMPLETED"
        )

        opt_b = await root.spawn("Evaluate SQLite sqlite-vec", title="Option B: SQLite", mode="clean")
        await opt_b.chat("SQLite sqlite-vec runs zero-config local-first embedded in a single file.")
        await ct.repo.commit_node_summary(
            opt_b.id, "SQLite sqlite-vec offers zero-config embedded local-first deployment.", status="COMPLETED"
        )
        await tree.add_artifact(
            "sqlite-ddl.sql",
            "SQLite Vector DDL",
            "CREATE TABLE embeddings (id TEXT PRIMARY KEY, vector_blob BLOB);",
            kind="sql",
            created_by_node_id=opt_b.id,
        )

        # 1. Compare siblings
        diff = await tree.compare_siblings(root.id)
        assert len(diff.siblings) == 2
        assert "sqlite-ddl.sql@v1" in next(s for s in diff.siblings if s.node_id == opt_b.id).artifact_refs
        assert "acid" in diff.distinctive_terms_by_node[opt_a.id] or "postgres" in diff.distinctive_terms_by_node[opt_a.id]

        # 2. Cherry-pick one insight from Option A into Option B
        cp_msg = await tree.cherry_pick(opt_a.id, opt_b.id, "Keep schema portable for future Postgres migration.")
        assert "Cherry-picked" in cp_msg.content

        # 3. Promote Option B as the winning branch -> archives Option A!
        await tree.promote_child(opt_b.id, archive_siblings=True)
        assert (await opt_a.model()).status == "ARCHIVED"
        assert any("Promoted Winning Branch" in m.content for m in await root.messages())

        # 4. Compile-to-document Markdown ADR export
        md = await tree.compile_markdown()
        assert "# Architecture Options" in md
        assert "Option B: SQLite" in md
        assert "sqlite-ddl.sql@v1" in md
        assert "CREATE TABLE embeddings" in md
