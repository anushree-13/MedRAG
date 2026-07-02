"""
core/config.py — Central configuration for the Agentic Hybrid RAG system.

Loads environment variables from .env and exposes typed settings
via Pydantic BaseSettings. All paths, model names, and tunable
hyperparameters are defined here so every agent reads from one source.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

from pydantic_settings import BaseSettings
from pydantic import Field


# ── Project root (two levels up from this file) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from .env + defaults."""

    # ── API Keys ──────────────────────────────────────────────────────
    groq_api_key: str = Field(
        default="",
        description="Groq API key for LLaMA 3 inference",
    )

    # ── Model Configuration ───────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model for embeddings",
    )
    embedding_dim: int = Field(
        default=384,
        description="Dimension of the embedding vectors (must match model)",
    )
    llm_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model ID for text generation",
    )
    llm_temperature: float = Field(
        default=0.3,
        description="Sampling temperature for LLM generation",
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens in the LLM response",
    )

    # ── Chunking Hyperparameters ──────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        description="Maximum tokens per chunk",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Overlapping tokens between consecutive chunks",
    )

    # ── Retrieval Hyperparameters ─────────────────────────────────────
    rrf_k: int = Field(
        default=60,
        description="Smoothing constant for Reciprocal Rank Fusion",
    )
    top_k: int = Field(
        default=5,
        description="Number of chunks to return from hybrid search",
    )

    # ── Clustering Hyperparameters ────────────────────────────────────
    cluster_min_k: int = Field(
        default=3,
        description="Minimum number of clusters for K-Means gap detection",
    )
    cluster_max_k: int = Field(
        default=15,
        description="Maximum number of clusters for K-Means gap detection",
    )

    # ── Directory Paths (derived from project root) ───────────────────
    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        description="Root data directory",
    )

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def graph_dir(self) -> Path:
        return self.data_dir / "graphs"

    def ensure_directories(self) -> None:
        """Create all data sub-directories if they don't exist."""
        for d in [
            self.upload_dir,
            self.parsed_dir,
            self.chunks_dir,
            self.index_dir,
            self.graph_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    settings = Settings()
    settings.ensure_directories()

    # ── Dev-time key verification (masked) ────────────────────────────
    if settings.groq_api_key:
        masked = settings.groq_api_key[:6] + "..." + settings.groq_api_key[-4:]
        logger.info(f"GROQ_API_KEY loaded: {masked}")
        print(f"[config] [OK] GROQ_API_KEY loaded: {masked}")
    else:
        logger.warning("GROQ_API_KEY is empty -- LLM features will be disabled")
        print("[config] [!!] GROQ_API_KEY is NOT set in .env")

    return settings
