"""
agents/retrieval/retrieval_agent.py — Hybrid Retrieval Agent

Executes parallel Semantic (FAISS) and Keyword (BM25) searches,
then fuses results using Reciprocal Rank Fusion (RRF).

Supports intent-aware weighting:
  - ANSWER:       50/50 semantic/keyword
  - COMPARE:      60/40 semantic/keyword
  - GAP_ANALYSIS: 70/30 semantic/keyword
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from agents.ingestion.indexing_agent import IndexingAgent
from core.config import get_settings
from core.models import Chunk, ClassifiedQuery, QueryIntent, SearchResult

logger = logging.getLogger(__name__)


# ── Intent-to-weight mapping ──────────────────────────────────────────
_INTENT_WEIGHTS: dict[QueryIntent, tuple[float, float]] = {
    #                           (semantic, keyword)
    QueryIntent.ANSWER:       (0.50, 0.50),
    QueryIntent.COMPARE:      (0.60, 0.40),
    QueryIntent.GAP_ANALYSIS: (0.70, 0.30),
}


class RetrievalAgent:
    """
    Hybrid search engine combining FAISS semantic search and BM25 keyword
    search via Reciprocal Rank Fusion (RRF).

    Usage:
        indexer = IndexingAgent()
        indexer.load()

        retriever = RetrievalAgent(indexer)
        results = retriever.search(classified_query)
    """

    def __init__(
        self,
        indexing_agent: IndexingAgent,
        rrf_k: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        self._indexer = indexing_agent
        settings = get_settings()
        self._rrf_k = rrf_k or settings.rrf_k   # 60
        self._top_k = top_k or settings.top_k   # 10

    def search(
        self,
        query: ClassifiedQuery,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Execute hybrid search with RRF fusion.

        Args:
            query:  A ClassifiedQuery with intent and refined_query.
            top_k:  Override the default number of results.

        Returns:
            Top-K SearchResult objects sorted by fused RRF score.
        """
        k = top_k or self._top_k
        search_text = query.refined_query or query.original_query

        if not self._indexer.is_loaded:
            raise RuntimeError(
                "Indexes not loaded. Call IndexingAgent.load() first."
            )

        # ── 1. Semantic search (FAISS) ────────────────────────────────
        semantic_ranking = self._semantic_search(search_text)

        # ── 2. Keyword search (BM25) ─────────────────────────────────
        keyword_ranking = self._keyword_search(search_text)

        # ── 3. RRF Fusion ─────────────────────────────────────────────
        weights = _INTENT_WEIGHTS.get(query.intent, (0.5, 0.5))
        fused = self._rrf_fuse(
            semantic_ranking=semantic_ranking,
            keyword_ranking=keyword_ranking,
            semantic_weight=weights[0],
            keyword_weight=weights[1],
        )

        # ── 4. Build results ──────────────────────────────────────────
        results: list[SearchResult] = []
        for rank, (chunk_id, score) in enumerate(fused[:k]):
            chunk = self._indexer.chunk_map.get(chunk_id)
            if chunk is None:
                continue

            # Determine which sources contributed
            in_semantic = chunk_id in {cid for cid, _ in semantic_ranking}
            in_keyword = chunk_id in {cid for cid, _ in keyword_ranking}
            if in_semantic and in_keyword:
                source = "hybrid"
            elif in_semantic:
                source = "semantic"
            else:
                source = "keyword"

            results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=rank + 1,
                source=source,
            ))

        logger.info(
            f"Hybrid search for '{search_text[:60]}...' → "
            f"{len(results)} results (intent={query.intent.value}, "
            f"weights={weights})"
        )
        return results

    def search_simple(
        self,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Convenience method: search with a plain string (defaults to ANSWER intent).

        Args:
            query_text: Raw query string.
            top_k:      Number of results.

        Returns:
            Top-K SearchResult objects.
        """
        from core.models import ClassifiedQuery, QueryIntent

        classified = ClassifiedQuery(
            original_query=query_text,
            refined_query=query_text,
            intent=QueryIntent.ANSWER,
        )
        return self.search(classified, top_k=top_k)

    # ── Private: Individual search methods ────────────────────────────

    def _semantic_search(self, query: str) -> list[tuple[str, float]]:
        """
        Search FAISS index. Returns list of (chunk_id, similarity_score)
        sorted by descending similarity.
        """
        query_vec = self._indexer.encode_query(query)
        n_total = self._indexer.faiss_index.ntotal

        # Search for more than top_k to give RRF a richer pool
        search_k = min(n_total, self._top_k * 5)
        scores, indices = self._indexer.faiss_index.search(query_vec, search_k)

        ranking: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._indexer.chunk_ids):
                continue
            chunk_id = self._indexer.chunk_ids[idx]
            ranking.append((chunk_id, float(score)))

        return ranking

    def _keyword_search(self, query: str) -> list[tuple[str, float]]:
        """
        Search BM25 index. Returns list of (chunk_id, bm25_score)
        sorted by descending score.
        """
        tokenized_query = IndexingAgent._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._indexer.bm25.get_scores(tokenized_query)

        # Pair with chunk IDs and sort
        scored_pairs = [
            (self._indexer.chunk_ids[i], float(s))
            for i, s in enumerate(scores)
            if s > 0
        ]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        # Limit to top candidates for RRF
        return scored_pairs[: self._top_k * 5]

    # ── Private: Reciprocal Rank Fusion ───────────────────────────────

    def _rrf_fuse(
        self,
        semantic_ranking: list[tuple[str, float]],
        keyword_ranking: list[tuple[str, float]],
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
    ) -> list[tuple[str, float]]:
        """
        Merge two ranked lists using weighted Reciprocal Rank Fusion.

        RRF score for document d:
            Score(d) = w_s / (k + rank_semantic(d)) + w_k / (k + rank_keyword(d))

        Args:
            semantic_ranking: Ranked (chunk_id, score) from FAISS.
            keyword_ranking:  Ranked (chunk_id, score) from BM25.
            semantic_weight:  Weight for the semantic component.
            keyword_weight:   Weight for the keyword component.

        Returns:
            Fused list of (chunk_id, rrf_score) sorted descending.
        """
        rrf_scores: dict[str, float] = defaultdict(float)

        for rank, (chunk_id, _) in enumerate(semantic_ranking):
            rrf_scores[chunk_id] += semantic_weight / (self._rrf_k + rank + 1)

        for rank, (chunk_id, _) in enumerate(keyword_ranking):
            rrf_scores[chunk_id] += keyword_weight / (self._rrf_k + rank + 1)

        # Sort by fused score descending
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused
