"""
core/orchestrator.py — Pipeline Orchestrator

Chains all agents into three main workflows:
  1. ingest(files)  → parsing → chunking → indexing
  2. query(text)    → classify → retrieve → extract → generate → validate
  3. analyze()      → gap detection → knowledge graph → visualization

Provides status callbacks for real-time UI updates.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from core.config import get_settings
from core.models import (
    AgentState, AgentStatus, Chunk, ClassifiedQuery,
    ExtractedEntity, GapReport, GeneratedResponse,
    SearchResult, Triple, ValidationReport,
)

logger = logging.getLogger(__name__)

StatusCallback = Optional[Callable[[AgentStatus], None]]


class Orchestrator:
    """
    Central pipeline controller that chains all agents.

    Usage:
        orch = Orchestrator(on_status=update_ui)
        orch.ingest(uploaded_files)
        response, validation = orch.query("What is the accuracy of BERT?")
        gap_report = orch.analyze()
    """

    def __init__(self, on_status: StatusCallback = None):
        self._on_status = on_status or (lambda s: None)
        self._settings = get_settings()

        # Lazy-loaded agents
        self._parsing_agent = None
        self._chunking_agent = None
        self._indexing_agent = None
        self._query_classifier = None
        self._retrieval_agent = None
        self._extraction_agent = None
        self._llm_generator = None
        self._gap_agent = None
        self._kg_agent = None
        self._validation_agent = None
        self._viz_agent = None

        # State
        self._all_chunks: list[Chunk] = []
        self._all_entities: list[ExtractedEntity] = []
        self._all_triples: list[Triple] = []

    # ── Status helper ─────────────────────────────────────────────────

    def _emit(self, name: str, state: AgentState, msg: str = "", progress: float = 0.0):
        self._on_status(AgentStatus(
            name=name, state=state, message=msg, progress=progress,
        ))

    # ── Pipeline 1: Ingestion ─────────────────────────────────────────

    def ingest(self, file_paths: list[str | Path]) -> list[Chunk]:
        """
        Run the full ingestion pipeline: parse → chunk → index.

        Args:
            file_paths: List of paths to PDF files.

        Returns:
            All generated chunks.
        """
        from agents.ingestion.parsing_agent import ParsingAgent
        from agents.ingestion.chunking_agent import ChunkingAgent
        from agents.ingestion.indexing_agent import IndexingAgent

        if self._parsing_agent is None:
            self._parsing_agent = ParsingAgent()
        if self._chunking_agent is None:
            self._chunking_agent = ChunkingAgent()
        if self._indexing_agent is None:
            self._indexing_agent = IndexingAgent()

        # ── Parse ─────────────────────────────────────────────────────
        self._emit("Parsing Agent", AgentState.RUNNING, "Extracting text from PDFs...")
        documents = []
        for i, fp in enumerate(file_paths):
            try:
                doc = self._parsing_agent.parse(fp)
                documents.append(doc)
                self._emit(
                    "Parsing Agent", AgentState.RUNNING,
                    f"Parsed {doc.filename} ({doc.metadata.page_count} pages)",
                    progress=(i + 1) / len(file_paths),
                )
            except Exception as e:
                logger.error(f"Parse error: {e}")
                self._emit("Parsing Agent", AgentState.ERROR, str(e))

        self._emit("Parsing Agent", AgentState.DONE, f"Parsed {len(documents)} documents", 1.0)

        # ── Chunk ─────────────────────────────────────────────────────
        self._emit("Chunking Agent", AgentState.RUNNING, "Splitting into chunks...")
        chunks: list[Chunk] = []
        for i, doc in enumerate(documents):
            doc_chunks = self._chunking_agent.chunk(doc)
            chunks.extend(doc_chunks)
            self._emit(
                "Chunking Agent", AgentState.RUNNING,
                f"{doc.filename}: {len(doc_chunks)} chunks",
                progress=(i + 1) / len(documents),
            )

        self._all_chunks.extend(chunks)
        self._emit("Chunking Agent", AgentState.DONE, f"Created {len(chunks)} chunks", 1.0)

        # ── Index ─────────────────────────────────────────────────────
        self._emit("Indexing Agent", AgentState.RUNNING, "Building FAISS + BM25 indexes...")

        if self._indexing_agent.is_loaded:
            self._indexing_agent.add(chunks)
        else:
            self._indexing_agent.build(self._all_chunks)

        self._emit("Indexing Agent", AgentState.DONE, "Indexes built and saved", 1.0)

        return chunks

    # ── Pipeline 2: Query ─────────────────────────────────────────────

    def query(
        self, query_text: str, stream: bool = False,
    ) -> tuple[GeneratedResponse, ValidationReport, list[SearchResult]]:
        """
        Run the full query pipeline: classify → retrieve → generate → validate.

        Args:
            query_text: The user's question.
            stream:     If True, returns a streaming generator in the response.

        Returns:
            Tuple of (GeneratedResponse, ValidationReport, SearchResults).
        """
        from agents.retrieval.query_classifier import QueryClassifier
        from agents.retrieval.retrieval_agent import RetrievalAgent
        from agents.analysis.extraction_agent import ExtractionAgent
        from agents.analysis.llm_generator import LLMGenerator
        from agents.synthesis.validation_agent import ValidationAgent

        if self._indexing_agent is None or not self._indexing_agent.is_loaded:
            from agents.ingestion.indexing_agent import IndexingAgent
            self._indexing_agent = IndexingAgent()
            self._indexing_agent.load()

        if self._query_classifier is None:
            self._query_classifier = QueryClassifier()
        if self._retrieval_agent is None:
            self._retrieval_agent = RetrievalAgent(self._indexing_agent)
        if self._extraction_agent is None:
            self._extraction_agent = ExtractionAgent()
        if self._llm_generator is None:
            self._llm_generator = LLMGenerator()
        if self._validation_agent is None:
            self._validation_agent = ValidationAgent()

        # ── Classify ──────────────────────────────────────────────────
        self._emit("Query Classifier", AgentState.RUNNING, "Classifying intent...")
        classified = self._query_classifier.classify(query_text)
        self._emit(
            "Query Classifier", AgentState.DONE,
            f"Intent: {classified.intent.value}", 1.0,
        )

        # ── Retrieve ──────────────────────────────────────────────────
        self._emit("Retrieval Agent", AgentState.RUNNING, "Searching indexes...")
        results = self._retrieval_agent.search(classified)
        self._emit(
            "Retrieval Agent", AgentState.DONE,
            f"Found {len(results)} relevant chunks", 1.0,
        )

        # ── Extract entities ──────────────────────────────────────────
        self._emit("Extraction Agent", AgentState.RUNNING, "Extracting entities...")
        entities = self._extraction_agent.extract_batch(
            [r.chunk for r in results[:5]]
        )
        self._all_entities.extend(entities)
        self._emit(
            "Extraction Agent", AgentState.DONE,
            f"Extracted {len(entities)} entities", 1.0,
        )

        # ── Generate ─────────────────────────────────────────────────
        self._emit("LLM Generator", AgentState.RUNNING, "Generating response...")
        response = self._llm_generator.generate(classified, results)
        self._emit("LLM Generator", AgentState.DONE, "Response generated", 1.0)

        # ── Validate ──────────────────────────────────────────────────
        self._emit("Validation Agent", AgentState.RUNNING, "Checking faithfulness...")
        validation = self._validation_agent.validate(
            response.answer, results, response.confidence,
        )
        self._emit(
            "Validation Agent", AgentState.DONE,
            f"Faithfulness: {validation.faithfulness_score:.0%}", 1.0,
        )

        return response, validation, results

    # ── Pipeline 3: Analyze ───────────────────────────────────────────

    def analyze(self) -> tuple[GapReport, dict, list[ExtractedEntity]]:
        """
        Run deep analysis: gap detection + knowledge graph.

        Returns:
            Tuple of (GapReport, graph_stats, all_entities).
        """
        from agents.analysis.gap_detection import GapDetectionAgent
        from agents.synthesis.knowledge_graph import KnowledgeGraphAgent

        if self._indexing_agent is None or not self._indexing_agent.is_loaded:
            from agents.ingestion.indexing_agent import IndexingAgent
            self._indexing_agent = IndexingAgent()
            self._indexing_agent.load()

        if self._gap_agent is None:
            self._gap_agent = GapDetectionAgent()
        if self._kg_agent is None:
            self._kg_agent = KnowledgeGraphAgent()

        # ── Gap Detection ─────────────────────────────────────────────
        self._emit("Gap Detection", AgentState.RUNNING, "Clustering embeddings...")
        embeddings = self._indexing_agent.get_all_embeddings()
        chunk_ids = self._indexing_agent.chunk_ids
        chunk_texts = [
            self._indexing_agent.chunk_map[cid].text for cid in chunk_ids
        ]
        gap_report = self._gap_agent.detect(embeddings, chunk_ids, chunk_texts)
        self._emit("Gap Detection", AgentState.DONE, f"Found {len(gap_report.gaps)} gaps", 1.0)

        # ── Knowledge Graph ───────────────────────────────────────────
        self._emit("Knowledge Graph", AgentState.RUNNING, "Extracting triples...")

        # Use a sample of chunks for KG extraction (API rate limits)
        sample_chunks = [
            self._indexing_agent.chunk_map[cid]
            for cid in chunk_ids[:20]
        ]
        triples = self._kg_agent.extract_triples(sample_chunks)
        self._all_triples.extend(triples)

        G = self._kg_agent.build_graph(triples)
        self._kg_agent.save_graph(G)
        stats = self._kg_agent.get_stats(G)

        self._emit(
            "Knowledge Graph", AgentState.DONE,
            f"{stats['total_nodes']} nodes, {stats['total_edges']} edges", 1.0,
        )

        return gap_report, stats, self._all_entities

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def indexing_agent(self):
        return self._indexing_agent

    @property
    def kg_agent(self):
        return self._kg_agent

    @property
    def all_entities(self):
        return self._all_entities

    @property
    def all_chunks(self):
        return self._all_chunks
