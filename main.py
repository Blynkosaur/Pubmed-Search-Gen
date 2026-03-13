from __future__ import annotations

import argparse
# import re
from pathlib import Path

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# from gemini import pico_extractor, extract_terms
from pubmed import parse, extract_doi_or_title, fetch_metadata_for_identifiers
# from pubmed import get_reference_texts, fetch_references_metadata, search_and_fetch_nbib





def run(pdf_path: Path) -> None:
    refs = parse(pdf_path)
    identifiers = [extract_doi_or_title(ref.raw) for ref in refs]
    print(f"Parsed {len(refs)} references from {pdf_path}\n")
    print("Fetching metadata from PubMed (bursts of 15, sleep 1 sec)...\n")
    metadata_list = fetch_metadata_for_identifiers(identifiers)
    for ident, meta in zip(identifiers, metadata_list):
        print(ident)
        if meta:
            print(f"  PMID: {meta.get('pmid')}")
            print(f"  Title: {meta.get('title', '')[:200]}{'...' if len(str(meta.get('title', ''))) > 200 else ''}")
            abstract = (meta.get("abstract") or "").strip()
            if abstract:
                print(f"  Abstract: {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
        else:
            print("  (no PubMed match)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse references from a systematic review PDF (parsing only).",
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

