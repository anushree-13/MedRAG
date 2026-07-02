"""
core/models.py — Pydantic data models shared across all agents.

These models define the data contracts between agents so that each
phase can produce and consume typed, validated objects.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Helpers ───────────────────────────────────────────────────────────

def _new_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:12]


# ── Phase 1: Ingestion Models ────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadata extracted from a PDF document."""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    page_count: int = 0
    file_size_bytes: int = 0
    ingested_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


class Document(BaseModel):
    """A fully parsed PDF document."""
    id: str = Field(default_factory=_new_id)
    filename: str
    full_text: str
    pages: list[str] = Field(
        default_factory=list,
        description="Text content per page (index = page number)",
    )
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class Chunk(BaseModel):
    """A single text chunk from a document, ready for indexing."""
    id: str = Field(default_factory=_new_id)
    doc_id: str
    doc_filename: str = ""
    text: str
    page_num: int = 0
    chunk_index: int = 0
    token_count: int = 0
    start_char: int = 0
    end_char: int = 0


# ── Phase 2: Retrieval Models ────────────────────────────────────────

class QueryIntent(str, Enum):
    """Classified intent of a user query."""
    ANSWER = "answer"
    COMPARE = "compare"
    GAP_ANALYSIS = "gap_analysis"


class ClassifiedQuery(BaseModel):
    """A user query after intent classification."""
    original_query: str
    refined_query: str = ""
    intent: QueryIntent = QueryIntent.ANSWER
    entities: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """A single result from hybrid search."""
    chunk: Chunk
    score: float
    rank: int = 0
    source: str = "hybrid"  # "semantic", "keyword", or "hybrid"


# ── Phase 3: Analysis Models ─────────────────────────────────────────

class ExtractedEntity(BaseModel):
    """A structured entity extracted from text."""
    entity_type: str  # "dataset", "metric", "hardware", "method"
    name: str
    value: str = ""
    context: str = ""
    source_chunk_id: str = ""


class ReasoningStep(BaseModel):
    """A single step in the reasoning trace."""
    step_num: int
    action: str
    detail: str
    duration_ms: float = 0.0


class GeneratedResponse(BaseModel):
    """The LLM-generated response with full reasoning trace."""
    answer: str
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    sources_used: list[str] = Field(
        default_factory=list,
        description="Chunk IDs used to generate the answer",
    )
    confidence: float = 0.0
    intent: QueryIntent = QueryIntent.ANSWER


class ClusterInfo(BaseModel):
    """Information about a single topic cluster."""
    cluster_id: int
    theme: str = ""
    size: int = 0
    density: float = 0.0
    representative_chunks: list[str] = Field(default_factory=list)


class GapReport(BaseModel):
    """Result of K-Means gap detection analysis."""
    clusters: list[ClusterInfo] = Field(default_factory=list)
    gaps: list[str] = Field(
        default_factory=list,
        description="Natural-language descriptions of identified research gaps",
    )
    optimal_k: int = 0
    silhouette_score: float = 0.0


# ── Phase 4: Synthesis Models ─────────────────────────────────────────

class Triple(BaseModel):
    """A knowledge graph triple (Subject → Predicate → Object)."""
    subject: str
    predicate: str
    object: str
    source_chunk_id: str = ""
    subject_type: str = ""   # Paper, Method, Dataset, Metric, Result
    object_type: str = ""


class ValidationReport(BaseModel):
    """Output from the faithfulness & self-reflection check."""
    faithfulness_score: float = 0.0  # 0.0 – 1.0
    total_claims: int = 0
    faithful_claims: int = 0
    reflection_notes: list[str] = Field(default_factory=list)
    final_confidence: float = 0.0


# ── Agent Status (for UI) ────────────────────────────────────────────

class AgentState(str, Enum):
    """Possible states of an agent."""
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class AgentStatus(BaseModel):
    """Live status of a running agent (used by the sidebar monitor)."""
    name: str
    state: AgentState = AgentState.IDLE
    message: str = ""
    progress: float = 0.0  # 0.0 – 1.0
    error: str = ""
