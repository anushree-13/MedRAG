"""
agents/synthesis/validation_agent.py — Faithfulness & Self-Reflection Agent

Validates generated answers by:
  1. Decomposing the answer into atomic claims
  2. Checking each claim against source chunks (faithfulness)
  3. Running LLM self-reflection for logical errors
  4. Producing a ValidationReport with scores
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from groq import Groq

from core.config import get_settings
from core.models import SearchResult, ValidationReport

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = (
    "Decompose the following answer into individual atomic claims. "
    "Each claim should be a single, verifiable statement.\n\n"
    "Answer:\n\"\"\"{answer}\"\"\"\n\n"
    "Respond with a JSON array of strings: [\"claim1\", \"claim2\", ...]"
)

_VERIFY_PROMPT = (
    "Given the following context, determine if this claim is supported.\n\n"
    "Context:\n\"\"\"{context}\"\"\"\n\n"
    "Claim: \"{claim}\"\n\n"
    'Respond with JSON: {{"supported": true/false, "reason": "brief explanation"}}'
)

_REFLECT_PROMPT = (
    "Review the following answer for logical errors, unsupported claims, "
    "or contradictions.\n\nAnswer:\n\"\"\"{answer}\"\"\"\n\n"
    "Context used:\n\"\"\"{context}\"\"\"\n\n"
    "Respond with JSON:\n"
    '{{"issues": ["issue1", ...], "confidence_adjustment": -0.1 to 0.0, '
    '"overall_quality": "good/fair/poor"}}'
)


class ValidationAgent:
    """
    Validates RAG-generated answers for faithfulness and quality.

    Usage:
        agent = ValidationAgent()
        report = agent.validate(answer_text, search_results)
        print(report.faithfulness_score)
    """

    def __init__(self):
        s = get_settings()
        self._client: Optional[Groq] = None
        self._model = s.llm_model
        if s.groq_api_key:
            self._client = Groq(api_key=s.groq_api_key)

    def validate(
        self,
        answer: str,
        search_results: list[SearchResult],
        base_confidence: float = 0.5,
    ) -> ValidationReport:
        """
        Run full validation pipeline.

        Args:
            answer:          The generated answer text.
            search_results:  The source chunks used to generate it.
            base_confidence: Starting confidence from the generator.

        Returns:
            A ValidationReport with faithfulness score and reflection.
        """
        if not self._client:
            return ValidationReport(
                faithfulness_score=0.0,
                final_confidence=base_confidence,
                reflection_notes=["Validation unavailable — no API key"],
            )

        context = "\n\n".join(r.chunk.text for r in search_results)

        # ── 1. Decompose into claims ──────────────────────────────────
        claims = self._decompose(answer)

        if not claims:
            return ValidationReport(
                faithfulness_score=1.0,
                total_claims=0,
                faithful_claims=0,
                final_confidence=base_confidence,
                reflection_notes=["No verifiable claims found"],
            )

        # ── 2. Check faithfulness of each claim ──────────────────────
        faithful_count = 0
        for claim in claims:
            if self._verify_claim(claim, context):
                faithful_count += 1

        faithfulness = faithful_count / len(claims) if claims else 0.0

        # ── 3. Self-reflection ────────────────────────────────────────
        reflection_notes, confidence_adj = self._reflect(answer, context)

        final_confidence = min(
            max(base_confidence + confidence_adj, 0.0), 1.0
        )
        # Blend with faithfulness
        final_confidence = round(
            0.6 * faithfulness + 0.4 * final_confidence, 3
        )

        return ValidationReport(
            faithfulness_score=round(faithfulness, 3),
            total_claims=len(claims),
            faithful_claims=faithful_count,
            reflection_notes=reflection_notes,
            final_confidence=final_confidence,
        )

    # ── Private ───────────────────────────────────────────────────────

    def _decompose(self, answer: str) -> list[str]:
        """Decompose answer into atomic claims."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(answer=answer[:2000])}],
                temperature=0.1, max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception as e:
            logger.debug(f"Decomposition failed: {e}")
        return []

    def _verify_claim(self, claim: str, context: str) -> bool:
        """Check if a single claim is supported by the context."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _VERIFY_PROMPT.format(
                    context=context[:3000], claim=claim,
                )}],
                temperature=0.1, max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return data.get("supported", False)
        except Exception as e:
            logger.debug(f"Verification failed: {e}")
        return False

    def _reflect(self, answer: str, context: str) -> tuple[list[str], float]:
        """Run self-reflection on the answer."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": _REFLECT_PROMPT.format(
                    answer=answer[:2000], context=context[:2000],
                )}],
                temperature=0.2, max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                issues = data.get("issues", [])
                adj = float(data.get("confidence_adjustment", 0.0))
                return issues, max(min(adj, 0.0), -0.3)
        except Exception as e:
            logger.debug(f"Reflection failed: {e}")
        return [], 0.0
