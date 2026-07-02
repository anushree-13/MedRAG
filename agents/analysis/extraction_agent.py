"""
agents/analysis/extraction_agent.py — Structured Data Extraction Agent

Extracts structured entities from text chunks using a two-tier approach:
  1. Regex patterns for high-confidence captures (metrics, hardware specs)
  2. LLM fallback via Groq for ambiguous/complex extractions

Entity types: datasets, metrics, hardware, methods.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from groq import Groq

from core.config import get_settings
from core.models import Chunk, ExtractedEntity

logger = logging.getLogger(__name__)


# ── Regex patterns for structured extraction ──────────────────────────

# Metrics: "accuracy of 95.3%", "F1-score: 0.89", "AUC = 0.94"
_METRIC_RE = re.compile(
    r"\b(accuracy|precision|recall|f1[\s\-]?score|auc[\s\-]?roc|auc|"
    r"specificity|sensitivity|bleu|rouge|mape|rmse|mae|"
    r"loss|perplexity|dice[\s\-]?score|iou|map|ndcg|mrr)"
    r"[\s:=]*(?:of\s+)?(\d+\.?\d*)\s*(%?)",
    re.IGNORECASE,
)

# Hardware: "NVIDIA A100", "Tesla V100", "RTX 3090", "TPU v4"
_HARDWARE_RE = re.compile(
    r"\b((?:nvidia\s+)?(?:a100|v100|a6000|h100|rtx\s*\d{4}|"
    r"tesla\s+\w+|geforce\s+\w+)|tpu\s*v\d|"
    r"\d+\s*(?:gb|tb)\s*(?:vram|gpu\s*memory|ram))",
    re.IGNORECASE,
)

# Training time: "trained for 24 hours", "training time: 3 days"
_TRAIN_TIME_RE = re.compile(
    r"\b(?:train(?:ing|ed)?)\s*(?:for|time|duration)?[\s:]*"
    r"(\d+\.?\d*)\s*(hours?|days?|epochs?|minutes?|hrs?)",
    re.IGNORECASE,
)

# Dataset names: typically capitalized, may include version numbers
_DATASET_RE = re.compile(
    r"\b(MIMIC[\s\-]?(?:III|IV|CXR)?|ImageNet|CIFAR[\s\-]?\d+|"
    r"MNIST|CheXpert|ChestX[\s\-]?ray\d*|PubMed[\s\-]?\w*|"
    r"COCO|SQuAD|GLUE|SuperGLUE|MS[\s\-]?MARCO|"
    r"(?:Open)?WebText|WikiText[\s\-]?\d*|"
    r"PhysioNet|eICU|TCGA|UK[\s\-]?Biobank)\b",
    re.IGNORECASE,
)


# ── LLM extraction prompt ────────────────────────────────────────────

_EXTRACT_PROMPT = """\
You are a research paper analysis agent. Extract ALL structured entities from the text below.

For each entity, provide:
- "type": one of "dataset", "metric", "hardware", "method"
- "name": the entity name
- "value": any associated numeric value (empty string if none)
- "context": a short phrase describing how it's used

Respond ONLY with a JSON array. Example:
[
  {{"type": "metric", "name": "accuracy", "value": "95.3%", "context": "model performance on test set"}},
  {{"type": "dataset", "name": "MIMIC-III", "value": "40K patients", "context": "training dataset"}}
]

Text:
\"\"\"{text}\"\"\"
"""


class ExtractionAgent:
    """
    Extracts structured entities (datasets, metrics, hardware, methods)
    from text chunks using regex + LLM.

    Usage:
        agent = ExtractionAgent()
        entities = agent.extract(chunk)
        entities = agent.extract_batch(chunks)
    """

    def __init__(self):
        settings = get_settings()
        self._client: Optional[Groq] = None
        self._model = settings.llm_model

        if settings.groq_api_key:
            try:
                self._client = Groq(api_key=settings.groq_api_key)
            except Exception as e:
                logger.warning(f"Groq client init failed: {e}")

    def extract(self, chunk: Chunk) -> list[ExtractedEntity]:
        """
        Extract structured entities from a single chunk.

        Uses regex first for high-confidence captures, then LLM
        for additional/ambiguous entities.

        Args:
            chunk: A text Chunk to analyze.

        Returns:
            List of ExtractedEntity objects found in the chunk.
        """
        entities: list[ExtractedEntity] = []

        # ── Tier 1: Regex extraction ──────────────────────────────────
        entities.extend(self._regex_extract(chunk))

        # ── Tier 2: LLM extraction (for what regex missed) ───────────
        if self._client:
            try:
                llm_entities = self._llm_extract(chunk)
                # Merge, avoiding duplicates
                existing_names = {
                    (e.entity_type, e.name.lower()) for e in entities
                }
                for e in llm_entities:
                    key = (e.entity_type, e.name.lower())
                    if key not in existing_names:
                        entities.append(e)
                        existing_names.add(key)
            except Exception as e:
                logger.debug(f"LLM extraction failed for chunk {chunk.id}: {e}")

        return entities

    def extract_batch(self, chunks: list[Chunk]) -> list[ExtractedEntity]:
        """Extract entities from multiple chunks."""
        all_entities: list[ExtractedEntity] = []
        for chunk in chunks:
            all_entities.extend(self.extract(chunk))
        return all_entities

    # ── Regex extraction ──────────────────────────────────────────────

    def _regex_extract(self, chunk: Chunk) -> list[ExtractedEntity]:
        """Apply all regex patterns to extract entities."""
        entities: list[ExtractedEntity] = []
        text = chunk.text

        # Metrics
        for match in _METRIC_RE.finditer(text):
            name = match.group(1).strip()
            value = match.group(2) + match.group(3)
            entities.append(ExtractedEntity(
                entity_type="metric",
                name=name,
                value=value,
                context=self._get_context(text, match.start(), match.end()),
                source_chunk_id=chunk.id,
            ))

        # Hardware
        for match in _HARDWARE_RE.finditer(text):
            entities.append(ExtractedEntity(
                entity_type="hardware",
                name=match.group(1).strip(),
                context=self._get_context(text, match.start(), match.end()),
                source_chunk_id=chunk.id,
            ))

        # Training time
        for match in _TRAIN_TIME_RE.finditer(text):
            entities.append(ExtractedEntity(
                entity_type="hardware",
                name="training_time",
                value=f"{match.group(1)} {match.group(2)}",
                context=self._get_context(text, match.start(), match.end()),
                source_chunk_id=chunk.id,
            ))

        # Datasets
        for match in _DATASET_RE.finditer(text):
            entities.append(ExtractedEntity(
                entity_type="dataset",
                name=match.group(1).strip(),
                context=self._get_context(text, match.start(), match.end()),
                source_chunk_id=chunk.id,
            ))

        return entities

    # ── LLM extraction ────────────────────────────────────────────────

    def _llm_extract(self, chunk: Chunk) -> list[ExtractedEntity]:
        """Use Groq LLM to extract entities the regex missed."""
        prompt = _EXTRACT_PROMPT.format(text=chunk.text[:2000])

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON array from response
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        entities: list[ExtractedEntity] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entity_type = item.get("type", "").lower()
            if entity_type not in ("dataset", "metric", "hardware", "method"):
                continue
            entities.append(ExtractedEntity(
                entity_type=entity_type,
                name=item.get("name", ""),
                value=str(item.get("value", "")),
                context=item.get("context", ""),
                source_chunk_id=chunk.id,
            ))

        return entities

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_context(text: str, start: int, end: int, window: int = 50) -> str:
        """Extract a context window around a match."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end].strip()
