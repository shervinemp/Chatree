"""Quickstart showing Chatree used directly as an embeddable Python library."""

import asyncio

from chatree import ArtifactInput, Chatree


async def main() -> None:
    async with Chatree.in_memory() as ct:
        tree = await ct.create_tree("Backend Architecture", goal="Design auth + database")
        root = await tree.root()

        await root.chat("We are building a local-first AI tool.")

        # Spawn an isolated child tangent
        db_branch = await root.spawn(
            "Select SQLite indexing strategy",
            title="DB Indexing",
            mode="fork",
        )
        await db_branch.chat("Use SQLite WAL, FTS5, and sqlite-vec HNSW cosine similarity.")
        await db_branch.finish(
            summary="Use SQLite WAL + FTS5 + sqlite-vec.",
            artifacts=[
                ArtifactInput(
                    slug="schema.sql",
                    title="Embeddings Schema",
                    kind="sql",
                    content="CREATE TABLE embeddings (id TEXT PRIMARY KEY, vector_blob BLOB);",
                )
            ],
            auto_commit=True,
        )

        # Read-only shadow query against the closed child (0 mutation to child history)
        shadow = await root.request(
            db_branch.id, prompt="Which vector index and SQLite mode did we pick?"
        )
        print("Shadow answer:\n", shadow["answer"])

        # Hybrid 3-tier tree search
        hits = await tree.search("sqlite-vec cosine similarity", scope="all", top_k=3)
        print("\nTop RAG hits:", [(h.node_title, h.granularity, h.score) for h in hits])

        # Compile active path + artifacts to a shareable Markdown ADR
        print("\n--- Compiled Markdown ADR ---")
        print(await tree.compile_markdown(include_transcript=False))


if __name__ == "__main__":
    asyncio.run(main())
