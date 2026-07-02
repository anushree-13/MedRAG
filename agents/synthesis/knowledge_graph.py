"""
agents/synthesis/knowledge_graph.py — Knowledge Graph Agent

Extracts (Subject, Predicate, Object) triples from text chunks using
Groq LLM, builds a NetworkX DiGraph with typed nodes, and exports
to streamlit-agraph format for visualization.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import networkx as nx
from groq import Groq

from core.config import get_settings
from core.models import Chunk, Triple

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = (
    "Extract knowledge graph triples from the text below.\n"
    "Each triple: (Subject, Predicate, Object).\n"
    "Node types: Paper, Method, Dataset, Metric, Result.\n\n"
    "Respond ONLY with a JSON array:\n"
    '[{{"subject":"...","subject_type":"...","predicate":"...",'
    '"object":"...","object_type":"..."}}]\n\n'
    'Text:\n"""{text}"""'
)

# Node-type color mapping for visualization
NODE_COLORS = {
    "Paper": "#7C3AED",
    "Method": "#06B6D4",
    "Dataset": "#10B981",
    "Metric": "#F59E0B",
    "Result": "#EC4899",
    "Unknown": "#6B7280",
}


class KnowledgeGraphAgent:
    """
    Builds a knowledge graph from research paper chunks.

    Usage:
        agent = KnowledgeGraphAgent()
        triples = agent.extract_triples(chunks)
        G = agent.build_graph(triples)
        agraph_data = agent.to_agraph(G)
    """

    def __init__(self):
        s = get_settings()
        self._client: Optional[Groq] = None
        self._model = s.llm_model
        self._graph_dir = s.graph_dir
        if s.groq_api_key:
            self._client = Groq(api_key=s.groq_api_key)

    def extract_triples(self, chunks: list[Chunk]) -> list[Triple]:
        """Extract triples from chunks using LLM."""
        if not self._client:
            logger.warning("No Groq client — cannot extract triples.")
            return []

        all_triples: list[Triple] = []
        for chunk in chunks:
            try:
                triples = self._extract_from_chunk(chunk)
                all_triples.extend(triples)
            except Exception as e:
                logger.debug(f"Triple extraction failed for {chunk.id}: {e}")

        # Normalize entities
        all_triples = self._normalize(all_triples)
        logger.info(f"Extracted {len(all_triples)} triples from {len(chunks)} chunks")
        return all_triples

    def build_graph(self, triples: list[Triple]) -> nx.DiGraph:
        """Build a NetworkX DiGraph from triples."""
        G = nx.DiGraph()

        for t in triples:
            if not G.has_node(t.subject):
                G.add_node(t.subject, node_type=t.subject_type or "Unknown")
            if not G.has_node(t.object):
                G.add_node(t.object, node_type=t.object_type or "Unknown")
            G.add_edge(
                t.subject, t.object,
                label=t.predicate,
                source_chunk=t.source_chunk_id,
            )

        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def save_graph(self, G: nx.DiGraph, name: str = "knowledge_graph") -> Path:
        """Save graph as GraphML."""
        self._graph_dir.mkdir(parents=True, exist_ok=True)
        path = self._graph_dir / f"{name}.graphml"
        nx.write_graphml(G, str(path))
        logger.info(f"Graph saved to {path}")
        return path

    def load_graph(self, name: str = "knowledge_graph") -> nx.DiGraph:
        """Load a saved graph."""
        path = self._graph_dir / f"{name}.graphml"
        if not path.exists():
            raise FileNotFoundError(f"No graph at {path}")
        return nx.read_graphml(str(path))

    def to_agraph(self, G: nx.DiGraph) -> dict:
        """
        Convert NetworkX graph to streamlit-agraph compatible format.

        Returns:
            {"nodes": [...], "edges": [...]} for agraph rendering.
        """
        from streamlit_agraph import Node, Edge

        nodes = []
        for nid, data in G.nodes(data=True):
            ntype = data.get("node_type", "Unknown")
            nodes.append(Node(
                id=nid,
                label=nid,
                size=25 + G.degree(nid) * 3,
                color=NODE_COLORS.get(ntype, NODE_COLORS["Unknown"]),
                title=f"Type: {ntype}\nConnections: {G.degree(nid)}",
            ))

        edges = []
        for u, v, data in G.edges(data=True):
            edges.append(Edge(
                source=u,
                target=v,
                label=data.get("label", ""),
                color="#4B5563",
            ))

        return {"nodes": nodes, "edges": edges}

    def get_stats(self, G: nx.DiGraph) -> dict:
        """Get graph statistics."""
        type_counts: dict[str, int] = {}
        for _, data in G.nodes(data=True):
            t = data.get("node_type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        rel_counts: dict[str, int] = {}
        for _, _, data in G.edges(data=True):
            r = data.get("label", "unknown")
            rel_counts[r] = rel_counts.get(r, 0) + 1

        # Most connected nodes
        top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "node_types": type_counts,
            "relationship_types": rel_counts,
            "most_connected": [{"node": n, "degree": d} for n, d in top_nodes],
        }

    # ── Private ───────────────────────────────────────────────────────

    def _extract_from_chunk(self, chunk: Chunk) -> list[Triple]:
        """Extract triples from a single chunk via LLM."""
        prompt = _EXTRACT_PROMPT.format(text=chunk.text[:2000])
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=800,
        )
        raw = resp.choices[0].message.content.strip()

        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []

        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return []

        triples = []
        for item in items:
            if not isinstance(item, dict):
                continue
            s = item.get("subject", "").strip()
            p = item.get("predicate", "").strip()
            o = item.get("object", "").strip()
            if s and p and o:
                triples.append(Triple(
                    subject=s, predicate=p, object=o,
                    subject_type=item.get("subject_type", "Unknown"),
                    object_type=item.get("object_type", "Unknown"),
                    source_chunk_id=chunk.id,
                ))
        return triples

    @staticmethod
    def _normalize(triples: list[Triple]) -> list[Triple]:
        """Normalize entity names (lowercase, merge similar)."""
        # Build a canonical name map
        name_map: dict[str, str] = {}

        for t in triples:
            for name in [t.subject, t.object]:
                lower = name.lower().strip()
                if lower not in name_map:
                    name_map[lower] = name  # Keep first-seen casing

        # Apply normalization
        for t in triples:
            t.subject = name_map.get(t.subject.lower().strip(), t.subject)
            t.object = name_map.get(t.object.lower().strip(), t.object)

        return triples
