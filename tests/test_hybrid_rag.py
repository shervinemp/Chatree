from __future__ import annotations

import importlib.util

from chatree import Chatree


async def test_three_tier_hybrid_rag_chunks_summaries_and_subtree_rollups():
    async with Chatree.in_memory() as ct:
        # Confirm native sqlite-vec loads cleanly whenever the optional `chatree[vec]` extra is installed
        if importlib.util.find_spec("sqlite_vec") is not None:
            assert ct.db is not None and ct.db.has_sqlite_vec is True

        tree = await ct.create_tree("Deep Tree RAG", auto_commit_handoffs=True)
        root = await tree.root()

        # Build a 3-level subtree: root -> security -> tokens -> revocation
        sec = await root.spawn("Security Architecture", title="Security", mode="clean")
        tokens = await sec.spawn("JWT Token Design", title="Tokens", mode="clean")
        revocation = await tokens.spawn("Token Revocation", title="Revocation", mode="clean")

        await revocation.chat(
            "Revoke compromised refresh tokens via a Redis BloomFilterRevocationList with 15m TTL."
        )
        await revocation.finish(
            summary="Compromised refresh tokens are revoked via Redis BloomFilterRevocationList.",
            auto_commit=True,
        )

        # Sibling branch under root with completely different topic
        billing = await root.spawn("Billing Stripe Webhooks", title="Billing", mode="clean")
        await billing.chat("Handle Stripe invoice.paid webhooks with idempotency keys.")
        await billing.finish(summary="Stripe webhook idempotency keys.", auto_commit=True)

        # 1. Tier-1 chunk hit with seq_range
        chunk_hits = await tree.search(
            "BloomFilterRevocationList Redis",
            scope="all",
            granularities={"chunk"},
            top_k=5,
        )
        assert len(chunk_hits) >= 1
        assert chunk_hits[0].node_id == revocation.id
        assert chunk_hits[0].granularity == "chunk"
        assert chunk_hits[0].seq_range is not None

        # 2. Tier-3 subtree rollup propagated all the way up to `tokens`, `sec`, and `root`!
        rollup_hits = await tree.search(
            "BloomFilterRevocationList",
            scope="all",
            granularities={"rollup"},
            top_k=5,
        )
        rollup_node_ids = {h.node_id for h in rollup_hits}
        assert tokens.id in rollup_node_ids
        assert sec.id in rollup_node_ids

        # 3. Scope filtering: from `billing`, siblings sees `sec` rollup (not `billing` itself!)
        sib_hits = await tree.search(
            "BloomFilterRevocationList", scope="siblings", node_id=billing.id, top_k=5
        )
        assert any(h.node_id == sec.id for h in sib_hits)
        assert all(h.node_id != billing.id for h in sib_hits)

        # 4. Portable cosine fallback produces identical top hit even when sqlite-vec disabled
        ct.db.has_sqlite_vec = False
        fallback_hits = await tree.search("Stripe invoice webhooks", scope="all", top_k=3)
        assert len(fallback_hits) >= 1
        assert fallback_hits[0].node_id == billing.id

        # 5. Punctuation-heavy query never breaks FTS5
        weird = await tree.search('C++: (BloomFilterRevocationList)? "quoted"!', scope="all")
        assert len(weird) >= 1
