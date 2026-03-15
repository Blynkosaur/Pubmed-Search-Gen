from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf

from gemini import (
    pico_extractor,
    parse_prospero,
    classify_seed_mesh_terms,
    augment_seed_mesh_with_hop1,
    extract_terms_from_abstract,
    split_freetext_terms_by_pico,
    extract_titles_from_references,
    add_wildcards,
    clean_search_terms_for_pubmed,
    build_pubmed_query,
)
from pubmed import parse as parse_pdf_references
from openalex import find_doi_by_title, load_or_build_citation_graph
from src.recall_nbib_included_studies import get_n_random_studies

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'>]+\b", re.IGNORECASE)


def _extract_sr_doi(pdf_path: Path) -> str | None:
    """Try to find the SR's own DOI from the first two pages of the PDF."""
    doc = pymupdf.open(pdf_path)
    try:
        text = ""
        for i in range(min(2, len(doc))):
            text += doc.load_page(i).get_text("text") or ""
    finally:
        doc.close()
    m = _DOI_RE.search(text)
    return m.group(0).strip().rstrip(".") if m else None


def run(
    pdf_path: Path,
    xlsx_path: Path | None = None,
    n_seeds: int | None = None,
    prospero_path: Path | None = None,
) -> None:
    pdf_path = Path(pdf_path)

    # 1) Seed refs: from Excel (N random) or from PDF references
    if xlsx_path is not None and n_seeds is not None:
        xlsx_path = Path(xlsx_path)
        print(f"Loading {n_seeds} random seed studies from {xlsx_path.name} …")
        seed_refs = get_n_random_studies(xlsx_path, n_seeds)
        # If DOI is missing but title is present, try to find DOI via OpenAlex search
        for ref in seed_refs:
            if ref.get("doi") or not (ref.get("title") or "").strip():
                continue
            title = (ref.get("title") or "").strip()
            print(f"  Looking up DOI for: {title[:60]}…")
            found_doi = find_doi_by_title(title)
            if found_doi:
                ref["doi"] = found_doi
                print(f"  → Found DOI: {found_doi}")
            else:
                print(f"  → No DOI found on OpenAlex")
        n_before = len(seed_refs)
        seed_refs = [
            s for s in seed_refs if s.get("doi") or (s.get("title") or "").strip()
        ]
        n_dropped = n_before - len(seed_refs)
        if n_dropped:
            print(f"Dropped {n_dropped} study/studies with no DOI or title.")
        if not seed_refs:
            print("No studies with DOI or title in the spreadsheet.")
            return
        print(f"Using {len(seed_refs)} seed studies from Included Studies Excel")
    else:
        print(f"Parsing references from {pdf_path.name} …")
        refs = parse_pdf_references(pdf_path)
        if not refs:
            print("No references parsed from the PDF.")
            return
        print(f"Parsed {len(refs)} references")

        # Build seed info for OpenAlex: DOI when available, else Gemini-extracted title
        seed_refs = []
        needs_title_indices = []
        needs_title_raws = []
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

    hop1_refs = [
        {"title": n["title"], "abstract": n["abstract"], "mesh_terms": n["mesh"]}
        for n in graph.values()
        if n["hop"] == 1
    ]

    # Hop-1 with ≥2 connections to hop-0 (used for MeSH augmentation and abstract terms)
    HOP1_MIN_CONNECTIONS = 2
    hop1_connected_refs = []
    hop1_three_plus_refs = []  # ≥3 connections (for abstract term extraction)
    for doi, n in graph.items():
        if n["hop"] != 1:
            continue
        edges_to_hop0 = sum(
            1 for d in n.get("cited_by", []) + n.get("cites", [])
            if d in hop0_dois
        )
        if edges_to_hop0 >= HOP1_MIN_CONNECTIONS:
            ref = {"title": n["title"], "abstract": n["abstract"], "mesh_terms": n["mesh"]}
            hop1_connected_refs.append(ref)
            if edges_to_hop0 >= 3:
                hop1_three_plus_refs.append(ref)

    # MeSH for augmentation: only from hop-1 nodes with ≥2 connections
    hop1_mesh_set = set()
    for r in hop1_connected_refs:
        for m in r.get("mesh_terms") or r.get("mesh") or []:
            if m and str(m).strip():
                hop1_mesh_set.add(str(m).strip())

    all_refs = hop0_refs + hop1_refs
    print(f"\nReferences (all nodes, no filter):")
    print(f"  hop 0: {len(hop0_refs)}")
    print(f"  hop 1: {len(hop1_refs)}")
    print(f"  hop 1 (≥{HOP1_MIN_CONNECTIONS} connections to hop-0): {len(hop1_connected_refs)}")
    print(f"  hop 1 (≥3 connections, for abstract terms): {len(hop1_three_plus_refs)}")
    print(f"  hop 1 MeSH set (≥{HOP1_MIN_CONNECTIONS} conn only, for augmentation): {len(hop1_mesh_set)} unique terms")

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
    pico = pico_extractor(pdf_path)
    print("\nPICO:")

    # ── 6b) Optional PROSPERO: extract author-provided terms (priority, no classification) ─
    prospero_data = None
    if prospero_path is not None:
        prospero_path = Path(prospero_path)
        if prospero_path.exists():
            print(f"\nParsing PROSPERO registration: {prospero_path.name} …")
            prospero_data = parse_prospero(prospero_path)
            print("PROSPERO terms extracted (will be added with priority to final blocks).")
        else:
            print(f"\nPROSPERO path not found: {prospero_path}; skipping.")

    if pico.get("summary"):
        print(f"  summary: {pico.get('summary', '')}")
    for key in ("population", "intervention", "comparator", "outcome"):
        print(f"  {key}: {pico.get(key, '')}")

    # ── 7a) MeSH set from hop-0 seeds + Gemini classification ─────────────────
    seed_mesh_set = set()
    for r in hop0_refs:
        for m in r.get("mesh_terms") or r.get("mesh") or []:
            if m and str(m).strip():
                seed_mesh_set.add(str(m).strip())
    seed_mesh_list = sorted(seed_mesh_set)
    print(f"\nSeed papers MeSH set: {len(seed_mesh_list)} unique terms")
    classified = classify_seed_mesh_terms(seed_mesh_list, pico)
    if prospero_data:
        # PROSPERO MeSH into initial population/intervention sets (before augmentation)
        classified["population"] = list(classified["population"]) + list(prospero_data["mesh_terms_population"])
        classified["intervention"] = list(classified["intervention"]) + list(prospero_data["mesh_terms_intervention"])
    print("\nClassified seed MeSH — population:", classified["population"])
    print("Classified seed MeSH — intervention:", classified["intervention"])
    print("Classified seed MeSH — others (discarded):", classified["others"])

    # Augment intervention only with hop1 MeSH; keep population MeSH as seed (no hop-1)
    hop1_mesh_list = sorted(hop1_mesh_set)
    print(f"\nAugmenting intervention with relevant terms from hop1 ({len(hop1_mesh_list)} terms); population MeSH unchanged …")
    augmented = augment_seed_mesh_with_hop1(
        pico,
        classified["population"],
        classified["intervention"],
        hop1_mesh_list,
    )
    augmented["population"] = list(classified["population"])
    print("Population MeSH (seed only):", augmented["population"])
    print("Augmented intervention:", augmented["intervention"])

    # ── 7c) Free terms from hop-0 abstracts: one concurrent Gemini call per abstract ─
    hop0_abstracts = [(r.get("abstract") or "").strip() for r in hop0_refs]
    hop0_with_abstract = [a for a in hop0_abstracts if a]
    abstract_terms_set = set()
    if hop0_with_abstract:
        print(f"\nExtracting terms from {len(hop0_with_abstract)} hop-0 abstracts (concurrent, 5-8 terms each) …")
        with ThreadPoolExecutor(max_workers=min(len(hop0_with_abstract), 10)) as executor:
            futures = [executor.submit(extract_terms_from_abstract, ab, pico) for ab in hop0_with_abstract]
            for fut in as_completed(futures):
                try:
                    abstract_terms_set.update(fut.result())
                except Exception as e:
                    print(f"  Abstract extraction failed: {e}")
        print(f"Abstract-derived terms (hop-0): {len(abstract_terms_set)} unique terms")

    # ── 7c2) Free terms from hop-1 (≥3 connections) abstracts: concurrent ─
    hop1_three_abstracts = [(r.get("abstract") or "").strip() for r in hop1_three_plus_refs]
    hop1_three_with_abstract = [a for a in hop1_three_abstracts if a]
    if hop1_three_with_abstract:
        print(f"\nExtracting terms from {len(hop1_three_with_abstract)} hop-1 (≥3 conn) abstracts (concurrent) …")
        n_workers = min(len(hop1_three_with_abstract), 10)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(extract_terms_from_abstract, ab, pico) for ab in hop1_three_with_abstract]
            for fut in as_completed(futures):
                try:
                    abstract_terms_set.update(fut.result())
                except Exception as e:
                    print(f"  Abstract extraction failed: {e}")
        print(f"Abstract-derived terms (hop-0 + hop-1 ≥3): {len(abstract_terms_set)} unique terms")
    if prospero_data:
        # PROSPERO freetext into initial freetext set (before split)
        abstract_terms_set.update(prospero_data["population_terms"])
        abstract_terms_set.update(prospero_data["intervention_terms"])
        abstract_terms_set.update(prospero_data["search_terms"])
    print("Abstract-derived terms:", sorted(abstract_terms_set))

    # ── 7d) Split abstract-derived free terms into population vs intervention via Gemini ─
    all_freetext_set = abstract_terms_set
    split_freetext = {"population": [], "intervention": []}
    if all_freetext_set:
        print(f"\nCombined freetext terms: {len(all_freetext_set)} unique")
        print("Free text:", sorted(all_freetext_set))
        print("Splitting into population vs intervention (Gemini, PICO context) …")
        split_freetext = split_freetext_terms_by_pico(sorted(all_freetext_set), pico)
        print("Freetext — population:", split_freetext["population"])
        print("Freetext — intervention:", split_freetext["intervention"])

    # ── 7e) Final blocks: augmented + split_freetext, then PROSPERO terms with priority ─
    final_population_mesh = set(augmented["population"])
    final_population_freetext = set(split_freetext["population"])
    final_intervention_mesh = set(augmented["intervention"])
    final_intervention_freetext = set(split_freetext["intervention"])

    if prospero_data:
        # PROSPERO terms bypass classification; add directly to final blocks
        final_population_mesh.update(prospero_data["mesh_terms_population"])
        final_population_freetext.update(prospero_data["population_terms"])
        final_intervention_mesh.update(prospero_data["mesh_terms_intervention"])
        final_intervention_freetext.update(prospero_data["intervention_terms"])
        final_intervention_freetext.update(prospero_data["search_terms"])
        if prospero_data["full_query"]:
            print(f"\nPROSPERO full query found (reference only): {prospero_data['full_query'][:200]}…" if len(prospero_data["full_query"]) > 200 else f"\nPROSPERO full query found (reference only): {prospero_data['full_query']}")
        print(f"\nPROSPERO population terms added: {prospero_data['population_terms']}")
        print(f"PROSPERO intervention terms added: {prospero_data['intervention_terms']}")
        if prospero_data["search_terms"]:
            print(f"PROSPERO search terms added to intervention: {prospero_data['search_terms']}")
        if prospero_data["mesh_terms_population"] or prospero_data["mesh_terms_intervention"]:
            print(f"PROSPERO MeSH (population): {prospero_data['mesh_terms_population']}; (intervention): {prospero_data['mesh_terms_intervention']}")

    print("\nFinal population (MeSH):", sorted(final_population_mesh))
    print("Final population (freetext):", sorted(final_population_freetext))
    print("Final intervention (MeSH):", sorted(final_intervention_mesh))
    print("Final intervention (freetext):", sorted(final_intervention_freetext))

    # ── 7f) Add wildcards to freetext only (before cleaning) ─
    population_freetext_for_cleaning = sorted(final_population_freetext)
    intervention_freetext_for_cleaning = sorted(final_intervention_freetext)
    if population_freetext_for_cleaning:
        print("\nAdding wildcards to population freetext (Gemini) …")
        population_freetext_for_cleaning = add_wildcards(population_freetext_for_cleaning, pico)
    if intervention_freetext_for_cleaning:
        print("Adding wildcards to intervention freetext (Gemini) …")
        intervention_freetext_for_cleaning = add_wildcards(intervention_freetext_for_cleaning, pico)

    # ── 7g) Gemini cleaning: remove noise, keep only confident PICO-relevant terms ─
    print("\nCleaning term lists for PubMed (Gemini) …")
    cleaned = clean_search_terms_for_pubmed(
        pico,
        sorted(final_population_mesh),
        population_freetext_for_cleaning,
        sorted(final_intervention_mesh),
        intervention_freetext_for_cleaning,
    )
    final_population_mesh = set(cleaned["population_mesh"])
    final_population_freetext = set(cleaned["population_freetext"])
    final_intervention_mesh = set(cleaned["intervention_mesh"])
    final_intervention_freetext = set(cleaned["intervention_freetext"])
    print("Cleaned population (MeSH):", sorted(final_population_mesh))
    print("Cleaned population (freetext):", sorted(final_population_freetext))
    print("Cleaned intervention (MeSH):", sorted(final_intervention_mesh))
    print("Cleaned intervention (freetext):", sorted(final_intervention_freetext))

    # ── 7h) Build PubMed boolean query from all term sets ─
    print("\nBuilding PubMed boolean query (Gemini) …")
    pubmed_query = build_pubmed_query(
        sorted(final_population_mesh),
        sorted(final_population_freetext),
        sorted(final_intervention_mesh),
        sorted(final_intervention_freetext),
        pico,
    )
    print("\nPubMed query:")
    print(pubmed_query)


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
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="Path to Included Studies Excel file. If set, use --N random rows as seeds instead of PDF refs.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        metavar="N",
        help="Number of random seed studies to use from --xlsx. Requires --xlsx.",
    )
    parser.add_argument(
        "--prospero",
        type=Path,
        default=None,
        help="Optional path to PROSPERO registration PDF. If provided, author terms are extracted and added with priority to final blocks.",
    )
    args = parser.parse_args()
    if (args.xlsx is None) != (args.N is None):
        parser.error("--xlsx and --N must be given together.")
    run(args.pdf, args.xlsx, args.N, args.prospero)


if __name__ == "__main__":
    main()
