from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import fitz  # PyMuPDF
import re


@dataclass
class Reference:
    """
    Lightweight representation of a reference extracted from a PDF.

    Fields are best-effort and may be None if they cannot be parsed reliably.
    """

    index: int
    raw: str
    doi: Optional[str] = None
    year: Optional[int] = None


_DOI_REGEX = re.compile(
    r"\b10\.\d{4,9}/[^\s\"'>]+\b",
    re.IGNORECASE,
)

_YEAR_REGEX = re.compile(r"\b(19|20)\d{2}\b")


def _extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        parts: List[str] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            parts.append(text)
        return "\n".join(parts)
    finally:
        doc.close()


def _find_references_block(full_text: str) -> str:
    """
    Heuristically locate the references section of the manuscript.
    """
    # Look for a heading like "References" near the end of the document.
    lowered = full_text.lower()
    idx = lowered.rfind("\nreferences\n")
    if idx == -1:
        idx = lowered.rfind("\nreference\n")
    if idx == -1:
        # Fallback: return the last 20% of the text as a guess.
        start = int(len(full_text) * 0.8)
        return full_text[start:]
    return full_text[idx:]


def _split_references_block(block: str) -> List[str]:
    """
    Split a references block into individual reference strings.

    Strategy:
    - Split on lines that look like numbered references (e.g. "1. ", "12) ").
    """
    lines = block.splitlines()
    refs: List[str] = []
    current: List[str] = []

    def flush_current():
        if current:
            text = " ".join(part.strip() for part in current if part.strip())
            if text:
                refs.append(text)

    for line in lines:
        if re.match(r"^\s*\d+[\.\)]\s+", line):
            flush_current()
            current = [line]
        else:
            current.append(line)
    flush_current()
    return refs


def _parse_single_reference(raw: str, index: int) -> Reference:
    doi_match = _DOI_REGEX.search(raw)
    year_match = _YEAR_REGEX.search(raw)

    doi = doi_match.group(0) if doi_match else None
    year = int(year_match.group(0)) if year_match else None

    return Reference(index=index, raw=raw.strip(), doi=doi, year=year)


def parse(pdf_path: Union[str, Path]) -> List[Reference]:
    """
    Parse a PDF manuscript, extract the reference list, and return article info.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    List[Reference]
        One entry per reference, with:
        - index: reference order (1-based)
        - raw: full reference string as extracted
        - doi: best-effort DOI (if detected)
        - year: best-effort publication year (if detected)
    """
    path = Path(pdf_path)
    full_text = _extract_pdf_text(path)
    ref_block = _find_references_block(full_text)
    ref_strings = _split_references_block(ref_block)

    references: List[Reference] = []
    for i, ref_text in enumerate(ref_strings, start=1):
        references.append(_parse_single_reference(ref_text, i))

    return references

