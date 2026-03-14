from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gemini import (
    pico_extractor,
    get_pico_keywords,
    extract_terms,
    filter_terms_by_key_concepts,
    filter_extracted_terms,
    extract_freetext_terms,
    expand_terms_variants,
    extract_titles_from_references,
)
from query_builder import build_query
from pubmed import parse as parse_pdf_references
from openalex import load_or_build_citation_graph

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'>]+\b", re.IGNORECASE)


def _extract_sr_doi(pdf_path: Path) -> str | None:
    """Try to find the SR's own DOI from the first two pages of the PDF."""
    doc = fitz.open(pdf_path)
    try:
        text = ""
        for i in range(min(2, len(doc))):
            text += doc.load_page(i).get_text("text") or ""
    finally:
        doc.close()
    m = _DOI_RE.search(text)
    return m.group(0).strip().rstrip(".") if m else None


def _doc_for_ref(rec: dict) -> str:
    """Use abstract if present, else mesh as text, else title."""
    abstract = (rec.get("abstract") or "").strip()
    if abstract:
        return abstract
    mesh = rec.get("mesh") or rec.get("mesh_terms") or []
    if mesh:
        return " ".join(str(m) for m in mesh if m)
    return (rec.get("title") or "").strip()


def run(pdf_path: Path) -> None:
    pdf_path = Path(pdf_path)

    # 1) Parse references from the PDF
    print(f"Parsing references from {pdf_path.name} …")
    refs = parse_pdf_references(pdf_path)
    if not refs:
        print("No references parsed from the PDF.")
        return
    print(f"Parsed {len(refs)} references")

    # 2) Build seed info for OpenAlex: DOI when available, else Gemini-extracted title
    seed_refs: list[dict] = []
    needs_title_indices: list[int] = []
    needs_title_raws: list[str] = []

    for i, r in enumerate(refs):
        doi = (r.doi or "").strip() or None
        if doi:
            seed_refs.append({"doi": doi, "title": None})
        else:
            seed_refs.append({"doi": None, "title": None})
            needs_title_indices.append(i)
            needs_title_raws.append(r.raw or "")

    if needs_title_raws:
        print(
            f"Extracting titles via Gemini for {len(needs_title_raws)} "
            "references without DOI …"
        )
        gemini_titles = extract_titles_from_references(needs_title_raws)
        for idx, title in zip(needs_title_indices, gemini_titles):
            seed_refs[idx]["title"] = title

    seed_refs = [
        s for s in seed_refs if s.get("doi") or (s.get("title") or "").strip()
    ]
    if not seed_refs:
        print("No DOIs or titles found; cannot build citation graph.")
        return

    # 3) Extract the SR's own DOI so we can also seed papers that cite it
    sr_doi = _extract_sr_doi(pdf_path)
    if sr_doi:
        print(f"SR DOI detected: {sr_doi}")
    else:
        print("Could not detect SR DOI from PDF — skipping cited-by-SR seeds")

    # 4) Build / load citation graph via OpenAlex
    graph = load_or_build_citation_graph(seed_refs, pdf_path=pdf_path, sr_doi=sr_doi)

    hop0_count = sum(1 for n in graph.values() if n["hop"] == 0)
    hop1_count = sum(1 for n in graph.values() if n["hop"] == 1)
    print(f"\nCitation graph: {len(graph)} total nodes")
    print(f"  hop 0 (seeds): {hop0_count}")
    print(f"  hop 1 (neighbors): {hop1_count}")

    # ── 5) Build reference lists from graph ──────────────────────────────
    hop0_dois = {doi for doi, n in graph.items() if n["hop"] == 0}

    hop0_refs = [
        {"title": n["title"], "abstract": n["abstract"], "mesh_terms": n["mesh"]}
        for n in graph.values()
        if n["hop"] == 0
    ]

    HOP1_MIN_CONNECTIONS = 2
    hop1_filtered = []
    for doi, n in graph.items():
        if n["hop"] != 1:
            continue
        edges_to_hop0 = sum(
            1 for d in n.get("cited_by", []) + n.get("cites", [])
            if d in hop0_dois
        )
        if edges_to_hop0 >= HOP1_MIN_CONNECTIONS:
            hop1_filtered.append(
                {"title": n["title"], "abstract": n["abstract"], "mesh_terms": n["mesh"]}
            )

    all_refs = hop0_refs + hop1_filtered
    print(f"\nReferences for term extraction:")
    print(f"  hop 0: {len(hop0_refs)}")
    print(
        f"  hop 1 (≥{HOP1_MIN_CONNECTIONS} connections to hop-0): "
        f"{len(hop1_filtered)}"
    )

    # Keep only refs that have abstract or MeSH (title-only refs carry minimal signal)
    total_refs = len(all_refs)
    references = [
        r
        for r in all_refs
        if (r.get("abstract") or "").strip() or (r.get("mesh_terms") or [])
    ]
    print(f"  with abstract or MeSH: {len(references)} of {total_refs}")

    if not references:
        print("No references with abstract or MeSH — cannot extract terms.")
        return

    # ── 6) PICO extraction from the SR ───────────────────────────────────
    _FACETS = ("population", "intervention", "comparator", "outcome")

    pico = pico_extractor(pdf_path)
    key_concepts = get_pico_keywords(pico)

    print("\nPICO:")
    for key in _FACETS:
        print(f"  {key}: {pico.get(key, '')}")
    print("Key concepts (3 per facet):")
    for key in _FACETS:
        kw = key_concepts.get(key, [])
        if kw:
            print(f"  {key}: {kw}")

    # ── 7) TF-IDF similarity filter ──────────────────────────────────────
    pico_text = " ".join(str(pico.get(k, "")) for k in _FACETS)
    ref_docs = [_doc_for_ref(r) for r in references]
    docs = [pico_text] + ref_docs
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    THRESHOLD = 0.05
    kept = [(rec, s) for rec, s in zip(references, sims) if s >= THRESHOLD]
    print(
        f"\nReferences with TF-IDF ≥ {THRESHOLD}: {len(kept)} of {len(references)}"
    )
    for rec, score in kept:
        title = (rec.get("title") or "").strip()
        print(f"  {score:.4f}  {title[:90]}")

    # ── 8) Term extraction + filtering via Gemini ────────────────────────
    filtered_refs = [rec for rec, _ in kept]
    terms = extract_terms(pico, filtered_refs)
    terms = filter_terms_by_key_concepts(terms, key_concepts)
    terms = filter_extracted_terms(terms, filtered_refs)

    # ── 8b) Dedicated freetext call for recall ───────────────────────────
    print("Extracting additional freetext terms (recall-focused) …")
    extra_freetext = extract_freetext_terms(pico, filtered_refs)
    for facet in ("population", "intervention", "comparator", "outcome"):
        existing = list(terms.get(facet, {}).get("freetext") or [])
        added = extra_freetext.get(facet) or []
        terms.setdefault(facet, {"mesh": [], "freetext": []})
        terms[facet]["freetext"] = list(dict.fromkeys(existing + added))

    # ── 8c) Topic-anchored variant expansion ──────────────────────────────
    print("Expanding terms with synonyms and variants (within review scope) …")
    extra_variants = expand_terms_variants(terms, pico, key_concepts)
    for facet in ("population", "intervention", "comparator", "outcome"):
        existing = list(terms[facet].get("freetext") or [])
        added = extra_variants.get(facet) or []
        terms[facet]["freetext"] = list(dict.fromkeys(existing + added))

    # ── 9) Build PubMed query ────────────────────────────────────────────
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
