"""
agents/ingestion/parsing_agent.py — PDF Parsing Agent

Extracts clean text from clinical/research PDFs using PyMuPDF (fitz).
Handles multi-column layouts, strips headers/footers/page numbers,
and captures document metadata (title, authors, page count).
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from core.models import Document, DocumentMetadata

logger = logging.getLogger(__name__)


# ── Regex patterns for cleaning ───────────────────────────────────────

# Page numbers: standalone digits, "Page X", "X of Y"
_PAGE_NUM_RE = re.compile(
    r"^\s*(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$",
    re.IGNORECASE,
)

# Common header/footer artifacts (running headers, copyright lines)
_HEADER_FOOTER_RE = re.compile(
    r"^\s*(?:"
    r"©.*|"
    r"all rights reserved.*|"
    r"doi\s*:\s*\S+|"
    r"http[s]?://\S+|"
    r"journal\s+of\s+.*|"
    r"proceedings\s+of\s+.*|"
    r"vol(?:ume)?\.?\s*\d+.*|"
    r"accepted\s+\d{1,2}\s+\w+\s+\d{4}|"
    r"published\s+\d{1,2}\s+\w+\s+\d{4}"
    r")\s*$",
    re.IGNORECASE,
)

# Excessive whitespace
_MULTI_SPACE_RE = re.compile(r"[ \t]{3,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


class ParsingAgent:
    """
    Extracts structured text from PDF files.

    Usage:
        agent = ParsingAgent()
        doc = agent.parse("path/to/paper.pdf")
        print(doc.metadata.title)
        print(doc.full_text[:500])
    """

    def parse(self, pdf_path: str | Path) -> Document:
        """
        Parse a PDF file and return a Document with cleaned text.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A Document with per-page text, full concatenated text,
            and extracted metadata.

        Raises:
            FileNotFoundError: If the PDF path does not exist.
            RuntimeError: If the PDF cannot be opened (corrupted / encrypted).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path.name}")

        try:
            pdf_doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(
                f"Cannot open PDF '{pdf_path.name}': {e}"
            ) from e

        pages: list[str] = []
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            raw_text = page.get_text("text")
            cleaned = self._clean_page(raw_text)
            pages.append(cleaned)

        full_text = "\n\n".join(pages)

        # Extract metadata from first page
        metadata = self._extract_metadata(
            pages=pages,
            pdf_doc=pdf_doc,
            pdf_path=pdf_path,
        )

        pdf_doc.close()

        doc = Document(
            filename=pdf_path.name,
            full_text=full_text,
            pages=pages,
            metadata=metadata,
        )

        logger.info(
            f"Parsed '{pdf_path.name}': {metadata.page_count} pages, "
            f"{len(full_text)} chars"
        )
        return doc

    # ── Private helpers ───────────────────────────────────────────────

    def _clean_page(self, raw_text: str) -> str:
        """Remove noise from a single page's extracted text."""
        lines = raw_text.split("\n")
        cleaned_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines (will consolidate later)
            if not stripped:
                cleaned_lines.append("")
                continue

            # Skip page numbers
            if _PAGE_NUM_RE.match(stripped):
                continue

            # Skip common header/footer lines
            if _HEADER_FOOTER_RE.match(stripped):
                continue

            # Collapse excessive inline whitespace (column artifacts)
            cleaned = _MULTI_SPACE_RE.sub("  ", line)
            cleaned_lines.append(cleaned)

        text = "\n".join(cleaned_lines)

        # Consolidate excessive blank lines
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)

        return text.strip()

    def _extract_metadata(
        self,
        pages: list[str],
        pdf_doc: fitz.Document,
        pdf_path: Path,
    ) -> DocumentMetadata:
        """Extract title, authors, and structural metadata."""
        page_count = len(pdf_doc)
        file_size = pdf_path.stat().st_size

        # Try PDF metadata first
        pdf_meta = pdf_doc.metadata or {}
        title = (pdf_meta.get("title") or "").strip()
        author_str = (pdf_meta.get("author") or "").strip()

        # Fallback: use first non-empty line of page 1 as title
        if not title and pages:
            first_lines = [
                l.strip() for l in pages[0].split("\n") if l.strip()
            ]
            if first_lines:
                # Title is usually the longest line in the top portion
                top_lines = first_lines[:5]
                title = max(top_lines, key=len)

        # Parse authors from PDF metadata or first page
        authors: list[str] = []
        if author_str:
            # Split on common delimiters: comma, semicolon, "and"
            authors = re.split(r"[;,]|\band\b", author_str)
            authors = [a.strip() for a in authors if a.strip()]

        return DocumentMetadata(
            title=title,
            authors=authors,
            page_count=page_count,
            file_size_bytes=file_size,
        )

    def parse_multiple(
        self, pdf_paths: list[str | Path]
    ) -> list[Document]:
        """Parse multiple PDFs and return a list of Documents."""
        documents: list[Document] = []
        for path in pdf_paths:
            try:
                doc = self.parse(path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to parse '{path}': {e}")
        return documents
