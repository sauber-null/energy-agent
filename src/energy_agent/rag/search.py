"""Retrieval over the chunk store: vector, full-text, or both fused with Reciprocal Rank Fusion."""

from collections import defaultdict
from dataclasses import replace
from typing import Literal

import psycopg

from energy_agent.rag import store
from energy_agent.rag.embeddings import OllamaEmbedder
from energy_agent.rag.store import SearchResult

SearchMode = Literal["hybrid", "vector", "text"]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Score each id by sum(1 / (k + rank)) over all rankings it appears in.

    RRF only uses ranks, so it can combine cosine similarities and ts_rank scores,
    which live on different scales. k=60 is the value from the original paper.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1 / (k + rank)
    return scores


class Retriever:
    def __init__(self, conn: psycopg.Connection, embedder: OllamaEmbedder):
        self.conn = conn
        self.embedder = embedder

    def search(self, query: str, k: int = 5, mode: SearchMode = "hybrid") -> list[SearchResult]:
        if mode == "text":
            return store.text_search(self.conn, query, k)

        candidates = max(20, 4 * k)
        vector_hits = store.vector_search(self.conn, self.embedder.embed([query])[0], candidates)
        if mode == "vector":
            return vector_hits[:k]

        text_hits = store.text_search(self.conn, query, candidates)
        fused = reciprocal_rank_fusion(
            [[h.chunk_id for h in vector_hits], [h.chunk_id for h in text_hits]]
        )
        by_id = {h.chunk_id: h for h in text_hits + vector_hits}
        best = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [replace(by_id[chunk_id], score=fused[chunk_id]) for chunk_id in best]
