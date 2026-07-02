"""
agents/analysis/gap_detection.py — Research Gap Detection Agent

Uses K-Means clustering on chunk embeddings to identify:
  - Topic clusters in the corpus
  - Sparse clusters (under-explored areas)
  - Gaps between clusters (research frontiers)

Auto-selects optimal k via Silhouette scoring.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from core.config import get_settings
from core.models import ClusterInfo, GapReport

logger = logging.getLogger(__name__)


class GapDetectionAgent:
    """
    Detects research gaps by clustering document chunk embeddings.

    Usage:
        from agents.ingestion.indexing_agent import IndexingAgent
        indexer = IndexingAgent(); indexer.load()

        agent = GapDetectionAgent()
        report = agent.detect(
            embeddings=indexer.get_all_embeddings(),
            chunk_ids=indexer.chunk_ids,
            chunk_texts=[indexer.chunk_map[c].text for c in indexer.chunk_ids],
        )
        print(report.gaps)
    """

    def __init__(
        self,
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
    ):
        s = get_settings()
        self._min_k = min_k or s.cluster_min_k  # 3
        self._max_k = max_k or s.cluster_max_k  # 15

    def detect(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        chunk_texts: list[str],
        llm_describe: Optional[object] = None,
    ) -> GapReport:
        """
        Run K-Means clustering and identify research gaps.

        Args:
            embeddings:   (N, D) array of chunk embeddings.
            chunk_ids:    Ordered chunk IDs matching the embeddings.
            chunk_texts:  Ordered chunk texts matching the embeddings.
            llm_describe: Optional callable(texts) -> str for cluster theme
                          generation. If None, uses keyword extraction.

        Returns:
            A GapReport with clusters, gaps, and silhouette score.
        """
        n = len(embeddings)
        if n < self._min_k:
            logger.warning(f"Only {n} chunks — too few for clustering.")
            return GapReport(optimal_k=1, silhouette_score=0.0)

        # ── 1. Find optimal k via Silhouette score ────────────────────
        max_k = min(self._max_k, n - 1)
        best_k, best_score, best_labels = self._find_optimal_k(
            embeddings, self._min_k, max_k
        )
        logger.info(f"Optimal k={best_k}, silhouette={best_score:.3f}")

        # ── 2. Build cluster info ─────────────────────────────────────
        clusters = self._build_clusters(
            best_k, best_labels, embeddings, chunk_ids, chunk_texts,
            llm_describe,
        )

        # ── 3. Identify gaps ─────────────────────────────────────────
        gaps = self._identify_gaps(clusters, embeddings, best_labels, best_k)

        return GapReport(
            clusters=clusters,
            gaps=gaps,
            optimal_k=best_k,
            silhouette_score=round(best_score, 4),
        )

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _find_optimal_k(
        embeddings: np.ndarray, min_k: int, max_k: int,
    ) -> tuple[int, float, np.ndarray]:
        """Iterate k values and pick the one with highest silhouette."""
        best_k = min_k
        best_score = -1.0
        best_labels = np.zeros(len(embeddings), dtype=int)

        for k in range(min_k, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
            labels = km.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        return best_k, best_score, best_labels

    def _build_clusters(
        self, k: int, labels: np.ndarray, embeddings: np.ndarray,
        chunk_ids: list[str], chunk_texts: list[str],
        llm_describe: Optional[object],
    ) -> list[ClusterInfo]:
        """Build ClusterInfo objects with themes and density."""
        clusters: list[ClusterInfo] = []

        for cid in range(k):
            mask = labels == cid
            indices = np.where(mask)[0]
            size = int(mask.sum())

            # Density: avg pairwise cosine similarity within cluster
            cluster_embs = embeddings[mask]
            if len(cluster_embs) > 1:
                norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
                normed = cluster_embs / (norms + 1e-9)
                sim_matrix = normed @ normed.T
                np.fill_diagonal(sim_matrix, 0)
                density = float(sim_matrix.sum() / (size * (size - 1)))
            else:
                density = 1.0

            # Representative chunks (3 closest to centroid)
            centroid = cluster_embs.mean(axis=0)
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            closest = np.argsort(dists)[:3]
            rep_ids = [chunk_ids[indices[i]] for i in closest]

            # Theme: use LLM if available, else keyword extraction
            rep_texts = [chunk_texts[indices[i]] for i in closest]
            if llm_describe and callable(llm_describe):
                try:
                    theme = llm_describe(rep_texts)
                except Exception:
                    theme = self._keyword_theme(rep_texts)
            else:
                theme = self._keyword_theme(rep_texts)

            clusters.append(ClusterInfo(
                cluster_id=cid,
                theme=theme,
                size=size,
                density=round(density, 4),
                representative_chunks=rep_ids,
            ))

        return clusters

    def _identify_gaps(
        self, clusters: list[ClusterInfo], embeddings: np.ndarray,
        labels: np.ndarray, k: int,
    ) -> list[str]:
        """Identify gaps: sparse clusters + inter-cluster voids."""
        gaps: list[str] = []

        # Sort clusters by density (ascending = most sparse first)
        sorted_clusters = sorted(clusters, key=lambda c: c.density)
        avg_density = np.mean([c.density for c in clusters])

        # Sparse clusters = under-explored
        for c in sorted_clusters:
            if c.density < avg_density * 0.7:
                gaps.append(
                    f"Under-explored area: '{c.theme}' "
                    f"(cluster {c.cluster_id}, density={c.density:.3f}, "
                    f"only {c.size} chunks) — this topic has low internal "
                    f"coherence, suggesting scattered or superficial coverage."
                )

        # Small clusters = niche/neglected topics
        avg_size = np.mean([c.size for c in clusters])
        for c in sorted_clusters:
            if c.size < avg_size * 0.4 and c.density >= avg_density * 0.7:
                gaps.append(
                    f"Niche topic: '{c.theme}' (cluster {c.cluster_id}, "
                    f"only {c.size} chunks) — this is a small but coherent "
                    f"area that may benefit from additional research."
                )

        # Inter-cluster gaps: large distances between centroids
        if k >= 2:
            centroids = []
            for cid in range(k):
                mask = labels == cid
                centroids.append(embeddings[mask].mean(axis=0))
            centroids = np.array(centroids)

            for i in range(k):
                for j in range(i + 1, k):
                    dist = float(np.linalg.norm(centroids[i] - centroids[j]))
                    if dist > np.median([
                        np.linalg.norm(centroids[a] - centroids[b])
                        for a in range(k) for b in range(a+1, k)
                    ]) * 1.3:
                        gaps.append(
                            f"Research frontier between '{clusters[i].theme}' "
                            f"and '{clusters[j].theme}' — these topics are "
                            f"distant in the embedding space, suggesting an "
                            f"unexplored intersection."
                        )

        if not gaps:
            gaps.append("No significant research gaps detected in the current corpus.")

        return gaps

    @staticmethod
    def _keyword_theme(texts: list[str], top_n: int = 5) -> str:
        """Extract a simple keyword-based theme from representative texts."""
        from collections import Counter
        import re

        words: list[str] = []
        stop = {"the","a","an","and","or","but","in","on","at","to","for","of",
                "with","by","from","is","are","was","were","be","been","this",
                "that","it","we","our","their","has","have","had","can","which",
                "as","not","no","also","using","used","based","than","more"}

        for text in texts:
            tokens = re.findall(r"[a-z]{3,}", text.lower())
            words.extend(t for t in tokens if t not in stop)

        common = Counter(words).most_common(top_n)
        return ", ".join(w for w, _ in common) if common else "general"
