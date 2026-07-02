"""
agents/ingestion/chunking_agent.py — Text Chunking Agent

Splits parsed documents into token-aware chunks of 512 tokens with
64-token overlap. Uses NLTK sentence tokenization to ensure chunks
never break mid-sentence, preserving medical/clinical context.
"""

from __future__ import annotations

import logging
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize

from core.config import get_settings
from core.models import Document, Chunk

logger = logging.getLogger(__name__)

# Download the sentence tokenizer data (safe to call multiple times)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class ChunkingAgent:
    """
    Splits documents into token-aware, sentence-boundary-respecting chunks.

    The algorithm:
    1. Sentence-tokenize the full document text.
    2. Accumulate sentences until the token budget (chunk_size) is reached.
    3. Emit the chunk, then slide back by `chunk_overlap` tokens worth of
       sentences to create the overlap window.
    4. Repeat until all sentences are consumed.

    Usage:
        agent = ChunkingAgent()
        chunks = agent.chunk(document)
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size      # 512
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap  # 64

        # We use a simple whitespace tokenizer for token counting.
        # This closely approximates sub-word tokenizer counts for
        # sentence-transformers models while being much faster.
        # For exact counts, swap in the model's tokenizer.
        self._count_tokens = self._whitespace_token_count

    # ── Public API ────────────────────────────────────────────────────

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a Document into overlapping Chunks.

        Args:
            document: A parsed Document with full_text populated.

        Returns:
            List of Chunk objects with text, token counts, and positional
            metadata.
        """
        if not document.full_text.strip():
            logger.warning(
                f"Document '{document.filename}' has no text to chunk."
            )
            return []

        sentences = sent_tokenize(document.full_text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_token_count = 0
        chunk_index = 0
        char_offset = 0  # Running character offset in full_text

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If a single sentence exceeds chunk_size, force-split it
            if sentence_tokens > self.chunk_size:
                # Flush current buffer first
                if current_sentences:
                    chunk = self._create_chunk(
                        sentences=current_sentences,
                        doc=document,
                        chunk_index=chunk_index,
                        start_char=char_offset,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_offset += len(chunk.text)

                # Force-split the long sentence by tokens
                forced_chunks = self._force_split(
                    sentence, document, chunk_index, char_offset
                )
                chunks.extend(forced_chunks)
                chunk_index += len(forced_chunks)
                char_offset += len(sentence)
                current_sentences = []
                current_token_count = 0
                continue

            # Would adding this sentence exceed the budget?
            if current_token_count + sentence_tokens > self.chunk_size:
                # Emit current chunk
                chunk = self._create_chunk(
                    sentences=current_sentences,
                    doc=document,
                    chunk_index=chunk_index,
                    start_char=char_offset,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Slide back to create overlap
                current_sentences, current_token_count, char_offset = (
                    self._slide_back(
                        current_sentences, char_offset
                    )
                )

            current_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Flush remaining sentences
        if current_sentences:
            chunk = self._create_chunk(
                sentences=current_sentences,
                doc=document,
                chunk_index=chunk_index,
                start_char=char_offset,
            )
            chunks.append(chunk)

        logger.info(
            f"Chunked '{document.filename}': {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def chunk_multiple(self, documents: list[Document]) -> list[Chunk]:
        """Chunk a list of documents and return all chunks."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _whitespace_token_count(text: str) -> int:
        """Approximate token count using whitespace splitting."""
        return len(text.split())

    def _create_chunk(
        self,
        sentences: list[str],
        doc: Document,
        chunk_index: int,
        start_char: int,
    ) -> Chunk:
        """Build a Chunk from a list of sentences."""
        text = " ".join(sentences)
        token_count = self._count_tokens(text)

        # Estimate which page this chunk falls on
        page_num = self._estimate_page(start_char, doc)

        return Chunk(
            doc_id=doc.id,
            doc_filename=doc.filename,
            text=text,
            page_num=page_num,
            chunk_index=chunk_index,
            token_count=token_count,
            start_char=start_char,
            end_char=start_char + len(text),
        )

    def _slide_back(
        self,
        sentences: list[str],
        char_offset: int,
    ) -> tuple[list[str], int, int]:
        """
        Keep the last N sentences whose total tokens ≤ chunk_overlap
        to form the overlap window for the next chunk.
        """
        overlap_sentences: list[str] = []
        overlap_tokens = 0

        for sent in reversed(sentences):
            sent_tokens = self._count_tokens(sent)
            if overlap_tokens + sent_tokens > self.chunk_overlap:
                break
            overlap_sentences.insert(0, sent)
            overlap_tokens += sent_tokens

        # Advance char_offset past the non-overlapping portion
        non_overlap_text = " ".join(sentences[: len(sentences) - len(overlap_sentences)])
        new_char_offset = char_offset + len(non_overlap_text)

        return overlap_sentences, overlap_tokens, new_char_offset

    def _force_split(
        self,
        long_sentence: str,
        doc: Document,
        start_index: int,
        start_char: int,
    ) -> list[Chunk]:
        """Split an overly long sentence into token-budget chunks."""
        words = long_sentence.split()
        chunks: list[Chunk] = []
        idx = start_index

        for i in range(0, len(words), self.chunk_size):
            segment = " ".join(words[i : i + self.chunk_size])
            chunk = Chunk(
                doc_id=doc.id,
                doc_filename=doc.filename,
                text=segment,
                page_num=self._estimate_page(start_char, doc),
                chunk_index=idx,
                token_count=len(words[i : i + self.chunk_size]),
                start_char=start_char,
                end_char=start_char + len(segment),
            )
            chunks.append(chunk)
            idx += 1
            start_char += len(segment) + 1  # +1 for the space

        return chunks

    @staticmethod
    def _estimate_page(char_offset: int, doc: Document) -> int:
        """Estimate the page number for a given character offset."""
        if not doc.pages:
            return 0

        cumulative = 0
        for page_num, page_text in enumerate(doc.pages):
            cumulative += len(page_text) + 2  # +2 for "\n\n" join separator
            if cumulative > char_offset:
                return page_num

        return len(doc.pages) - 1
