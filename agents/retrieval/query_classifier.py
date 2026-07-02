"""
agents/retrieval/query_classifier.py — Query Intent Classifier

Detects whether the user wants:
  - ANSWER:       A direct factual answer from the papers.
  - COMPARE:      A comparison between methods, models, or results.
  - GAP_ANALYSIS: An exploration of under-researched areas.

Uses a lightweight LLM call to Groq for classification, with a
rule-based fallback when the API is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from groq import Groq

from core.config import get_settings
from core.models import ClassifiedQuery, QueryIntent

logger = logging.getLogger(__name__)


# ── Classification prompt ─────────────────────────────────────────────

_CLASSIFY_PROMPT = """\
You are a query intent classifier for a research paper analysis system.

Classify the user's query into exactly ONE of these intents:
- "answer": The user wants a direct factual answer or explanation from papers.
- "compare": The user wants to compare methods, models, datasets, or results.
- "gap_analysis": The user wants to find research gaps, unexplored areas, or future directions.

Also extract key entities (method names, dataset names, metrics, concepts) from the query.
Rewrite the query to be more precise for search if needed.

Respond ONLY with valid JSON in this exact format:
{
  "intent": "answer" | "compare" | "gap_analysis",
  "entities": ["entity1", "entity2"],
  "refined_query": "improved version of the query for search"
}

User query: "{query}"
"""

# ── Rule-based keyword patterns (fallback) ────────────────────────────

_COMPARE_KEYWORDS = re.compile(
    r"\b(compar|versus|vs\.?|differ|better|worse|advantage|disadvantage|"
    r"benchmark|outperform|against|superior|inferior|trade.?off)\b",
    re.IGNORECASE,
)

_GAP_KEYWORDS = re.compile(
    r"\b(gap|unexplored|under.?researched|future\s+(work|direction|research)|"
    r"limitation|missing|lack|open\s+(problem|question|issue)|"
    r"not\s+(been\s+)?(studied|explored|investigated)|opportunity)\b",
    re.IGNORECASE,
)


class QueryClassifier:
    """
    Classifies user queries by intent and extracts entities.

    Usage:
        classifier = QueryClassifier()
        result = classifier.classify("Compare BERT and GPT-4 on MIMIC-III")
        print(result.intent)    # QueryIntent.COMPARE
        print(result.entities)  # ["BERT", "GPT-4", "MIMIC-III"]
    """

    def __init__(self):
        settings = get_settings()
        self._client: Optional[Groq] = None
        self._model = settings.llm_model

        if settings.groq_api_key:
            try:
                self._client = Groq(api_key=settings.groq_api_key)
            except Exception as e:
                logger.warning(f"Groq client init failed: {e}. Using fallback.")

    def classify(self, query: str) -> ClassifiedQuery:
        """
        Classify a user query into intent + entities.

        Tries LLM-based classification first, falls back to rule-based
        keyword matching if the API is unavailable.

        Args:
            query: The raw user query string.

        Returns:
            A ClassifiedQuery with intent, entities, and refined query.
        """
        if not query.strip():
            return ClassifiedQuery(
                original_query=query,
                refined_query=query,
                intent=QueryIntent.ANSWER,
            )

        # Try LLM classification
        if self._client:
            try:
                return self._classify_with_llm(query)
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}. Using fallback.")

        # Fallback to rule-based
        return self._classify_with_rules(query)

    # ── LLM-based classification ──────────────────────────────────────

    def _classify_with_llm(self, query: str) -> ClassifiedQuery:
        """Use Groq LLM for intent classification."""
        prompt = _CLASSIFY_PROMPT.format(query=query)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in LLM response: {raw[:200]}")

        data = json.loads(json_match.group())

        intent_str = data.get("intent", "answer").lower()
        intent_map = {
            "answer": QueryIntent.ANSWER,
            "compare": QueryIntent.COMPARE,
            "gap_analysis": QueryIntent.GAP_ANALYSIS,
        }

        return ClassifiedQuery(
            original_query=query,
            refined_query=data.get("refined_query", query),
            intent=intent_map.get(intent_str, QueryIntent.ANSWER),
            entities=data.get("entities", []),
        )

    # ── Rule-based fallback ───────────────────────────────────────────

    def _classify_with_rules(self, query: str) -> ClassifiedQuery:
        """Classify using keyword pattern matching."""
        intent = QueryIntent.ANSWER

        if _COMPARE_KEYWORDS.search(query):
            intent = QueryIntent.COMPARE
        elif _GAP_KEYWORDS.search(query):
            intent = QueryIntent.GAP_ANALYSIS

        return ClassifiedQuery(
            original_query=query,
            refined_query=query,
            intent=intent,
            entities=self._extract_entities_simple(query),
        )

    @staticmethod
    def _extract_entities_simple(query: str) -> list[str]:
        """
        Simple entity extraction: capitalized multi-word phrases,
        acronyms, and quoted terms.
        """
        entities: list[str] = []

        # Quoted terms
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Acronyms (2+ uppercase letters)
        acronyms = re.findall(r"\b[A-Z]{2,}(?:-\d+)?\b", query)
        entities.extend(acronyms)

        # Capitalized phrases (Title Case words not at sentence start)
        words = query.split()
        for i, word in enumerate(words):
            clean = word.strip(".,;:!?()[]")
            if (
                i > 0
                and clean
                and clean[0].isupper()
                and not clean.isupper()
                and len(clean) > 2
            ):
                entities.append(clean)

        return list(dict.fromkeys(entities))  # Deduplicate preserving order
