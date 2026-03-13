from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gemini import (
    pico_extractor,
    get_pico_keywords,
    extract_terms,
    filter_terms_by_key_concepts,
    filter_extracted_terms,
)
from query_builder import build_query
from pubmed import parse as parse_pdf_references, fetch_metadata_for_identifiers


def _ref_cache_path(pdf_path: Path) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}_references.json"


def _doc_for_ref(rec: dict) -> str:
    """Use abstract if present, else mesh_terms as text, else title."""
    abstract = (rec.get("abstract") or "").strip()
    if abstract:
        return abstract
    mesh = rec.get("mesh_terms") or []
    if mesh:
        return " ".join(str(m) for m in mesh if m)
    return (rec.get("title") or "").strip()


def run(pdf_path: Path) -> None:
    pdf_path = Path(pdf_path)
    cache_path = _ref_cache_path(pdf_path)
    if not cache_path.exists():
        print(f"No references cache at {cache_path}. Building it from {pdf_path.name} (this may take a few minutes)...")

        # 1) Parse references from the PDF
        refs = parse_pdf_references(pdf_path)
        if not refs:
            print("No references could be parsed from the PDF; cannot build cache.")
            return

        # 2) Build identifiers: DOI if present, else PMID, else authors+title
        identifiers = []
        for r in refs:
            ident = (
                (getattr(r, "doi", None) or getattr(r, "pmid", None) or getattr(r, "title", None)) or ""
            ).strip()
            if ident:
                identifiers.append(ident)

        if not identifiers:
            print("No DOIs or titles found in parsed references; cannot build cache.")
            return

        # 3) Fetch PubMed metadata (pmid, doi, title, abstract, mesh_terms, etc.)
        metadata = fetch_metadata_for_identifiers(identifiers)

        # 4) Normalize into reference dicts expected by downstream code
        references = []
        for meta, ref in zip(metadata, refs):
            if isinstance(meta, dict) and meta:
                # Use metadata record as-is (includes title, abstract, mesh_terms)
                references.append(meta)
            else:
                # Fallback minimal record based on parsed reference
                title = (getattr(ref, "title", None) or "").strip()
                references.append(
                    {
                        "title": title,
                        "abstract": "",
                        "mesh_terms": [],
                    }
                )

        cache_path.write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(references)} reference records to {cache_path}\n")
    else:
        with cache_path.open("r", encoding="utf-8") as f:
            references = json.load(f)
        print(f"Loaded {len(references)} references from {cache_path}\n")

    pico = pico_extractor(pdf_path)
    key_concepts = get_pico_keywords(pico)
    pico_text = " ".join(str(pico.get(k, "")) for k in ("population", "intervention"))
    ref_docs = [_doc_for_ref(r) for r in references]
    docs = [pico_text] + ref_docs
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    THRESHOLD = 0.05
    kept = [(rec, s) for rec, s in zip(references, sims) if s >= THRESHOLD]

    print("PICO:")
    for key in ("population", "intervention"):
        print(f"  {key}: {pico.get(key, '')}")
    print("Key concepts (3 per facet):")
    for key in ("population", "intervention"):
        print(f"  {key}: {key_concepts.get(key, [])}")
    print(f"\nReferences with TF-IDF score >= {THRESHOLD}: {len(kept)} of {len(references)}\n")
    for rec, score in kept:
        title = (rec.get("title") or "").strip()
        print(f"{score:.4f}  {title}")

    filtered_refs = [rec for rec, _ in kept]
    terms = extract_terms(pico, filtered_refs)
    terms = filter_terms_by_key_concepts(terms, key_concepts)
    terms = filter_extracted_terms(terms, filtered_refs)
    query = build_query(terms)
    print("\nExtracted search terms (filtered):")
    print(json.dumps(terms, indent=2))
    print("\nPubMed query:")
    print(query)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PICO from a systematic review PDF.",
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

