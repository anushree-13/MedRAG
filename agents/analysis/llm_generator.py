"""
agents/analysis/llm_generator.py — LLM Response Generator

Uses LLaMA 3 via Groq to synthesize context-based responses with
structured reasoning traces. Supports streaming for real-time UI.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Generator, Optional

from groq import Groq

from core.config import get_settings
from core.models import (
    ClassifiedQuery, GeneratedResponse, QueryIntent,
    ReasoningStep, SearchResult,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert research analyst specializing in clinical and scientific "
    "literature. Provide precise, evidence-based answers grounded ONLY in the "
    "provided context. Cite sources by document filename and page number. "
    "If the context is insufficient, say so explicitly."
)

_ANSWER_TPL = (
    "Based on the following research paper excerpts, answer the question.\n\n"
    "CONTEXT:\n{context}\n\nQUESTION: {query}\n\n"
    "Provide: 1) A direct answer  2) Supporting evidence  3) Caveats."
)

_COMPARE_TPL = (
    "Based on the following excerpts, provide a detailed comparison.\n\n"
    "CONTEXT:\n{context}\n\nCOMPARISON REQUEST: {query}\n\n"
    "Structure: 1) Overview of each approach  2) Dimension-by-dimension comparison  "
    "3) Advantages/disadvantages  4) Recommendation."
)

_GAP_TPL = (
    "Based on the following excerpts, identify research gaps.\n\n"
    "CONTEXT:\n{context}\n\nANALYSIS REQUEST: {query}\n\n"
    "Structure: 1) Covered areas  2) Gaps  3) Contradictions  "
    "4) Future directions  5) Methodological limitations."
)

_TEMPLATES = {
    QueryIntent.ANSWER: _ANSWER_TPL,
    QueryIntent.COMPARE: _COMPARE_TPL,
    QueryIntent.GAP_ANALYSIS: _GAP_TPL,
}

_REASONING_SUFFIX = (
    "\n\nBefore answering, outline reasoning as JSON: "
    '[{"step":1,"action":"...","detail":"..."}]\n'
    "Format:\nREASONING:\n[json]\n\nANSWER:\n[answer]"
)


class LLMGenerator:
    """Generates context-grounded responses using LLaMA 3 via Groq."""

    def __init__(self):
        s = get_settings()
        self._model = s.llm_model
        self._temp = s.llm_temperature
        self._max_tok = s.llm_max_tokens
        self._client: Optional[Groq] = None
        if s.groq_api_key:
            self._client = Groq(api_key=s.groq_api_key)

    def generate(
        self, query: ClassifiedQuery, search_results: list[SearchResult],
    ) -> GeneratedResponse:
        """Generate a full response with reasoning trace."""
        if not self._client:
            return GeneratedResponse(
                answer="⚠️ Set GROQ_API_KEY in .env", confidence=0.0, intent=query.intent,
            )

        t0 = time.time()
        context = self._build_context(search_results)
        src_ids = [r.chunk.id for r in search_results]

        tpl = _TEMPLATES.get(query.intent, _ANSWER_TPL)
        prompt = tpl.format(context=context, query=query.original_query) + _REASONING_SUFFIX

        steps = [ReasoningStep(
            step_num=1, action="Retrieved context",
            detail=f"{len(search_results)} chunks from "
                   f"{len(set(r.chunk.doc_filename for r in search_results))} docs",
        )]

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temp, max_tokens=self._max_tok,
            )
            raw = resp.choices[0].message.content.strip()
            elapsed = (time.time() - t0) * 1000

            r_steps, answer = self._parse_response(raw)
            steps.append(ReasoningStep(
                step_num=2, action="Generated response",
                detail=f"Model: {self._model}", duration_ms=elapsed,
            ))
            steps.extend(r_steps)

            return GeneratedResponse(
                answer=answer, reasoning_steps=steps, sources_used=src_ids,
                confidence=self._estimate_confidence(search_results, answer),
                intent=query.intent,
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return GeneratedResponse(
                answer=f"⚠️ Generation failed: {e}", reasoning_steps=steps,
                sources_used=src_ids, confidence=0.0, intent=query.intent,
            )

    def generate_stream(
        self, query: ClassifiedQuery, search_results: list[SearchResult],
    ) -> Generator[str, None, None]:
        """Stream LLM response token-by-token."""
        if not self._client:
            yield "⚠️ Set GROQ_API_KEY in .env"
            return

        context = self._build_context(search_results)
        tpl = _TEMPLATES.get(query.intent, _ANSWER_TPL)
        prompt = tpl.format(context=context, query=query.original_query)

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temp, max_tokens=self._max_tok, stream=True,
            )
            for chunk in stream:
                d = chunk.choices[0].delta
                if d and d.content:
                    yield d.content
        except Exception as e:
            yield f"\n\n⚠️ Streaming error: {e}"

    # ── Constants ──────────────────────────────────────────────────────
    # Max characters in the assembled context block sent to the LLM.
    # 6 000 chars ≈ ~1 500 tokens — leaves headroom for system prompt,
    # reasoning suffix, and the model's own output within an 8K window.
    _MAX_CONTEXT_CHARS = 6000
    _MAX_CONTEXT_CHUNKS = 5

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_context(
        results: list[SearchResult],
        max_chars: int = 6000,
        max_chunks: int = 5,
    ) -> str:
        """Build a context string from search results, capped by size.

        Args:
            results:    Ranked search results (best first).
            max_chars:  Hard character cap on the assembled context.
            max_chunks: Maximum number of chunks to include.

        Returns:
            A truncated, source-annotated context string.
        """
        parts: list[str] = []
        total_len = 0

        for i, r in enumerate(results[:max_chunks], 1):
            c = r.chunk
            entry = (
                f"[Source {i}] (File: {c.doc_filename}, Page: {c.page_num+1}, "
                f"Relevance: {r.source})\n{c.text}"
            )
            if total_len + len(entry) > max_chars:
                # Include a truncated version of this chunk if there's room
                remaining = max_chars - total_len
                if remaining > 100:
                    parts.append(entry[:remaining] + " …[truncated]")
                break
            parts.append(entry)
            total_len += len(entry)

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _parse_response(raw: str) -> tuple[list[ReasoningStep], str]:
        steps: list[ReasoningStep] = []
        answer = raw
        if "REASONING:" in raw and "ANSWER:" in raw:
            parts = raw.split("ANSWER:", 1)
            rp = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else raw
            m = re.search(r"\[.*\]", rp, re.DOTALL)
            if m:
                try:
                    for i, s in enumerate(json.loads(m.group())):
                        steps.append(ReasoningStep(
                            step_num=i+3,
                            action=s.get("action", f"Step {i+1}"),
                            detail=s.get("detail", ""),
                        ))
                except json.JSONDecodeError:
                    pass
        return steps, answer

    @staticmethod
    def _estimate_confidence(results: list[SearchResult], answer: str) -> float:
        if not results:
            return 0.0
        src = min(len(results) / 5.0, 1.0)
        avg = sum(r.score for r in results) / len(results)
        sc = min(avg * 50, 1.0)
        ln = min(len(answer.split()) / 100.0, 1.0)
        return round(min(0.4*src + 0.4*sc + 0.2*ln, 1.0), 3)
