from __future__ import annotations

import re
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_references(raw_text: str) -> List[Dict]:
    """
    Parse a raw references block into a list of dicts:
    {ref_num, title, doi, year}.

    Expects lines roughly like:
      1. Author et al. (2012) Title. Journal ... https://doi.org/10.xxxx/yyy

    Handles DOIs that are broken across lines with spaces by normalizing
    whitespace and joining wrapped URL fragments.
    """
    # Normalize whitespace and join lines so wrapped DOIs become contiguous.
    # Keep newline markers before reference numbers to preserve splitting.
    # First, collapse obvious soft-hyphen artifacts and zero-width spaces.
    cleaned = raw_text.replace("\u200b", "").replace("\u00ad", "")
    # Ensure each line break is kept, then split and re-join with single spaces.
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in cleaned.splitlines()]
    normalized = "\n".join(ln for ln in lines if ln)

    # Split into reference chunks based on leading ref numbers like "1." or "12. "
    parts = re.split(r"\n(?=\d+\s*[\.\)])", normalized)

    refs: List[Dict] = []

    doi_pattern = re.compile(r"(10\.\d{4,9}/[^\s\"'>]+)", re.IGNORECASE)
    year_pattern = re.compile(r"\b(19|20)\d{2}\b")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract reference number
        m_num = re.match(r"^(\d+)\s*[\.\)]\s*(.*)$", part)
        if not m_num:
            continue
        ref_num = int(m_num.group(1))
        rest = m_num.group(2).strip()

        # Extract DOI (if present)
        doi_match = doi_pattern.search(rest)
        doi = doi_match.group(1) if doi_match else None

        # Extract year (first 19xx/20xx)
        year_match = year_pattern.search(rest)
        year = int(year_match.group(0)) if year_match else None

        # Heuristic for title: take text between first ")" after year and next "."
        title = ""
        # Find pattern "(YEAR)" and take following segment as title
        m_year_paren = re.search(r"\(\s*(19|20)\d{2}\s*\)\s*(.+)", rest)
        if m_year_paren:
            after_year = m_year_paren.group(2)
            # Title until first period
            title = after_year.split(".")[0].strip()
        else:
            # Fallback: everything up to first period
            title = rest.split(".")[0].strip()

        refs.append(
            {
                "ref_num": ref_num,
                "title": title,
                "doi": doi,
                "year": year,
            }
        )

    return refs


def filter_references_by_pico(
    references: List[Dict],
    pico: str,
    must_include_dois: List[str],
) -> List[Dict]:
    """
    Filter references by TF-IDF cosine similarity to a PICO string.

    - references: list of dicts with at least {"ref_num", "title", "doi"}.
    - pico: PICO description string.
    - must_include_dois: DOIs that must always be included regardless of score.

    Prints each reference with its similarity score, then returns only those
    with score > 0.05 or whose DOI is in must_include_dois.
    """
    if not references:
        return []

    titles = [ref.get("title", "") or "" for ref in references]
    docs = [pico] + titles  # first doc is the PICO query

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs)

    # Cosine similarity of each title to the PICO text
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    must_include_set = {d.strip().lower() for d in must_include_dois if d}

    filtered: List[Dict] = []
    for ref, score in zip(references, sims):
        doi = (ref.get("doi") or "").strip().lower()
        include = score > 0.05 or (doi and doi in must_include_set)

        # Print for inspection
        print(
            f"Ref {ref.get('ref_num')}: score={score:.3f} | "
            f"DOI={ref.get('doi')} | title={ref.get('title')}"
        )

        if include:
            ref_with_score = dict(ref)
            ref_with_score["similarity"] = float(score)
            filtered.append(ref_with_score)

    return filtered

