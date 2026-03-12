from __future__ import annotations

import argparse
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gemini import pico_extractor
from pubmed import parse


def run(pdf_path: Path) -> None:
    # 1) Get SR's PICO via Gemini
    pico = pico_extractor(pdf_path)
    print("Step 1: PICO extracted from manuscript:")
    for key in ["population", "intervention", "comparator", "outcome"]:
        value = pico.get(key)
        print(f"- {key.capitalize()}: {value}")

    pico_text = " ".join(str(pico.get(k, "")) for k in ["population", "intervention", "comparator", "outcome"])

    # 2) Get all references from the SR
    refs = parse(pdf_path)
    print(f"\nStep 2: Parsed {len(refs)} references from {pdf_path}\n")

    # 3) Run TF-IDF similarity between PICO text and each reference TITLE only
    titles = []
    for ref in refs:
        raw = ref.raw or ""
        # Heuristic: title is the text after "(YEAR)" up to the next period.
        m_year_paren = re.search(r"\(\s*(19|20)\d{2}\s*\)\s*(.+)", raw)
        if m_year_paren:
            after_year = m_year_paren.group(2)
            title = after_year.split(".")[0].strip()
        else:
            # Fallback: everything up to first period
            title = raw.split(".")[0].strip()
        titles.append(title)

    docs = [pico_text] + titles  # first doc is the PICO query

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # Print all references with their similarity scores (no filtering)
    for ref, score in zip(refs, sims):
        line = (
            f"Ref {ref.index}: score={score:.3f} | "
            f"DOI={ref.doi} | Year={ref.year} | Reference={ref.raw}"
        )
        print(line)

    print(f"\nStep 3: Printed similarity scores for all {len(refs)} references")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="1) Extract SR PICO, 2) extract all references, 3) print TF-IDF similarity between PICO and each reference.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to the systematic review PDF.",
    )
    args = parser.parse_args()
    run(args.pdf)


if __name__ == "__main__":
    main()

