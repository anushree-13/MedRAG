"""
agents/ingestion/indexing_agent.py — Dual Index Builder Agent

Builds and persists two search indexes from document chunks:
  1. FAISS (Semantic) — L2-normalized embeddings with inner-product search
  2. BM25 (Keyword)  — BM25Okapi from rank-bm25

Supports incremental indexing: new documents can be appended without
rebuilding the entire index from scratch.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from core.config import get_settings
from core.models import Chunk

logger = logging.getLogger(__name__)


# ── Simple English stop-words for BM25 tokenization ──────────────────
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "this", "that", "these", "those", "it", "its", "i", "we", "you",
    "he", "she", "they", "me", "us", "him", "her", "them", "my", "our",
    "your", "his", "their", "what", "which", "who", "whom", "where",
    "when", "how", "not", "no", "nor", "if", "then", "than", "so",
    "as", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "each", "all", "both", "such",
})

_WORD_RE = re.compile(r"[a-z0-9]+")


class IndexingAgent:
    """
    Builds dual FAISS + BM25 indexes from text chunks.

    Usage:
        agent = IndexingAgent()
        agent.build(chunks)

        # Later — load existing indexes:
        agent.load()
        faiss_index = agent.faiss_index
        bm25 = agent.bm25
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        index_dir: Optional[Path] = None,
    ):
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._index_dir = index_dir or settings.index_dir
        self._embedding_dim = settings.embedding_dim

        # Lazy-loaded resources
        self._model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: list[str] = []        # Ordered chunk IDs (index-aligned)
        self.chunk_map: dict[str, Chunk] = {} # chunk_id → Chunk object
        self._bm25_corpus: list[list[str]] = []

    # ── Public API ────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        """
        Build both FAISS and BM25 indexes from scratch, then persist.

        Args:
            chunks: List of Chunk objects to index.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return

        logger.info(f"Building indexes for {len(chunks)} chunks...")

        # ── 1. Prepare chunk registry ─────────────────────────────────
        self.chunk_ids = [c.id for c in chunks]
        self.chunk_map = {c.id: c for c in chunks}

        # ── 2. Build FAISS semantic index ─────────────────────────────
        model = self._get_model()
        texts = [c.text for c in chunks]

        logger.info("Encoding chunks with sentence-transformers...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,  # L2-normalize for cosine sim via IP
        )
        embeddings = np.array(embeddings, dtype="float32")

        self.faiss_index = faiss.IndexFlatIP(self._embedding_dim)
        self.faiss_index.add(embeddings)

        logger.info(
            f"FAISS index built: {self.faiss_index.ntotal} vectors "
            f"({self._embedding_dim}D)"
        )

        # ── 3. Build BM25 keyword index ──────────────────────────────
        self._bm25_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._bm25_corpus)

        logger.info(f"BM25 index built: {len(self._bm25_corpus)} documents")

        # ── 4. Persist everything ─────────────────────────────────────
        self._save()
        logger.info(f"Indexes saved to {self._index_dir}")

    def add(self, new_chunks: list[Chunk]) -> None:
        """
        Incrementally add new chunks to existing indexes.

        Args:
            new_chunks: Additional Chunk objects to add.
        """
        if not new_chunks:
            return

        # Load existing indexes if not in memory
        if self.faiss_index is None:
            self.load()

        # ── Update chunk registry ─────────────────────────────────────
        new_ids = [c.id for c in new_chunks]
        self.chunk_ids.extend(new_ids)
        self.chunk_map.update({c.id: c for c in new_chunks})

        # ── Append to FAISS ───────────────────────────────────────────
        model = self._get_model()
        new_texts = [c.text for c in new_chunks]
        new_embeddings = model.encode(
            new_texts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,
        )
        new_embeddings = np.array(new_embeddings, dtype="float32")
        self.faiss_index.add(new_embeddings)

        # ── Rebuild BM25 (BM25Okapi doesn't support incremental add) ─
        new_tokenized = [self._tokenize(c.text) for c in new_chunks]
        self._bm25_corpus.extend(new_tokenized)
        self.bm25 = BM25Okapi(self._bm25_corpus)

        # ── Persist ───────────────────────────────────────────────────
        self._save()
        logger.info(
            f"Added {len(new_chunks)} chunks. "
            f"Total: {self.faiss_index.ntotal} vectors"
        )

    def load(self) -> None:
        """Load persisted indexes from disk."""
        faiss_path = self._index_dir / "faiss.index"
        bm25_path = self._index_dir / "bm25.pkl"
        meta_path = self._index_dir / "chunk_meta.json"

        if not faiss_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found at {faiss_path}. Run build() first."
            )

        # FAISS
        self.faiss_index = faiss.read_index(str(faiss_path))
        logger.info(f"Loaded FAISS index: {self.faiss_index.ntotal} vectors")

        # BM25
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self._bm25_corpus = bm25_data["corpus"]
        logger.info(f"Loaded BM25 index: {len(self._bm25_corpus)} documents")

        # Chunk metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.chunk_ids = meta["chunk_ids"]
        self.chunk_map = {
            cid: Chunk(**cdata) for cid, cdata in meta["chunks"].items()
        }

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into a normalized embedding vector."""
        model = self._get_model()
        vec = model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(vec, dtype="float32")

    def get_all_embeddings(self) -> np.ndarray:
        """Reconstruct all embeddings from the FAISS index."""
        if self.faiss_index is None:
            raise RuntimeError("No FAISS index loaded. Call load() first.")
        n = self.faiss_index.ntotal
        return faiss.rev_swig_ptr(
            self.faiss_index.get_xb(), n * self._embedding_dim
        ).reshape(n, self._embedding_dim).copy()

    @property
    def is_loaded(self) -> bool:
        """Check if indexes are loaded and ready."""
        return (
            self.faiss_index is not None
            and self.bm25 is not None
            and len(self.chunk_ids) > 0
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25: lowercase, remove stop-words."""
        words = _WORD_RE.findall(text.lower())
        return [w for w in words if w not in _STOP_WORDS and len(w) > 1]

    def _save(self) -> None:
        """Persist both indexes and chunk metadata to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)

        # FAISS index
        faiss_path = self._index_dir / "faiss.index"
        faiss.write_index(self.faiss_index, str(faiss_path))

        # BM25 index + corpus
        bm25_path = self._index_dir / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "corpus": self._bm25_corpus},
                f,
            )

        # Chunk metadata (JSON for easy inspection)
        meta_path = self._index_dir / "chunk_meta.json"
        meta = {
            "chunk_ids": self.chunk_ids,
            "chunks": {
                cid: chunk.model_dump()
                for cid, chunk in self.chunk_map.items()
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
