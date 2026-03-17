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
    extract_terms_from_seed_titles,
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
    hop2_count = sum(1 for n in graph.values() if n["hop"] == 2)
    hop3_count = sum(1 for n in graph.values() if n["hop"] == 3)
    print(f"\nCitation graph: {len(graph)} total nodes")
    print(f"  hop 0 (seeds): {hop0_count}")
    print(f"  hop 1 (papers that cite seeds): {hop1_count}")
    print(f"  hop 2 (top 30 refs of hop-1): {hop2_count}")
    print(f"  hop 3 (top 10 by connections to top 30): {hop3_count}")

    # ── 5) Build reference lists; pipeline uses hop 0 + top 30 (hop 2) + top 10 (hop 3) ─
    hop0_refs = [
        {"title": n["title"], "abstract": n["abstract"], "mesh_terms": n["mesh"]}
        for n in graph.values()
        if n["hop"] == 0
    ]

    hop2_refs = [
        {"title": n["title"], "abstract": n["abstract"], "mesh_terms": n["mesh"]}
        for n in graph.values()
        if n["hop"] == 2
    ]

    hop3_refs = [
        {"title": n["title"], "abstract": n["abstract"], "mesh_terms": n["mesh"]}
        for n in graph.values()
        if n["hop"] == 3
    ]

    # MeSH for augmentation: from hop-2 (top 30) + hop-3 (top 10)
    hop2_mesh_set = set()
    hop3_mesh_set = set()
    for r in hop2_refs:
        for m in r.get("mesh_terms") or r.get("mesh") or []:
            if m and str(m).strip():
                hop2_mesh_set.add(str(m).strip())
    for r in hop3_refs:
        for m in r.get("mesh_terms") or r.get("mesh") or []:
            if m and str(m).strip():
                hop3_mesh_set.add(str(m).strip())
    hop2_hop3_mesh_set = hop2_mesh_set | hop3_mesh_set

    all_refs = hop0_refs + hop2_refs + hop3_refs
    print(f"\nReferences used in pipeline (hop 0 + top 30 hop-2 + top 10 hop-3):")
    print(f"  hop 0: {len(hop0_refs)}")
    print(f"  hop 2 (top 30): {len(hop2_refs)}")
    if hop2_refs:
        for i, r in enumerate(hop2_refs[:10], 1):
            title = (r.get("title") or "")[:60]
            print(f"    {i}. {title}…" if len((r.get("title") or "")) > 60 else f"    {i}. {title}")
        if len(hop2_refs) > 10:
            print(f"    … and {len(hop2_refs) - 10} more")
    print(f"  hop 3 (top 10): {len(hop3_refs)}")
    if hop3_refs:
        for i, r in enumerate(hop3_refs, 1):
            title = (r.get("title") or "")[:60]
            print(f"    {i}. {title}…" if len((r.get("title") or "")) > 60 else f"    {i}. {title}")
    print(f"  MeSH set (hop-2 + hop-3, for augmentation): {len(hop2_hop3_mesh_set)} unique terms")

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

    # Augment intervention with MeSH from hop-2 (top 30) + hop-3 (top 10); keep population MeSH as seed
    hop2_hop3_mesh_list = sorted(hop2_hop3_mesh_set)
    print(f"\nAugmenting intervention with relevant terms from hop-2 + hop-3 ({len(hop2_hop3_mesh_list)} terms); population MeSH unchanged …")
    augmented = augment_seed_mesh_with_hop1(
        pico,
        classified["population"],
        classified["intervention"],
        hop2_hop3_mesh_list,
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

    # ── 7c1) Key phrases from hop-0 seed paper titles (MANDATORY in final query; protected from cleaning) ─
    hop0_titles = [(r.get("title") or "").strip() for r in hop0_refs if (r.get("title") or "").strip()]
    seed_title_population: set[str] = set()
    seed_title_intervention: set[str] = set()
    if hop0_titles:
        print(f"\nExtracting terms from {len(hop0_titles)} hop-0 seed paper titles (one Gemini call) …")
        title_terms = extract_terms_from_seed_titles(hop0_titles, pico)
        seed_title_population = set(title_terms.get("population") or [])
        seed_title_intervention = set(title_terms.get("intervention") or [])
        abstract_terms_set.update(seed_title_population)
        abstract_terms_set.update(seed_title_intervention)
        print(f"Seed title terms — population: {len(seed_title_population)}, intervention: {len(seed_title_intervention)} (mandatory in final query)")

    # ── 7c2) Free terms from hop-2 (top 30) + hop-3 (top 10) abstracts: concurrent ─
    hop2_hop3_refs = hop2_refs + hop3_refs
    hop2_hop3_abstracts = [(r.get("abstract") or "").strip() for r in hop2_hop3_refs]
    hop2_hop3_with_abstract = [a for a in hop2_hop3_abstracts if a]
    if hop2_hop3_with_abstract:
        print(f"\nExtracting terms from {len(hop2_hop3_with_abstract)} hop-2 + hop-3 abstracts (concurrent) …")
        n_workers = min(len(hop2_hop3_with_abstract), 10)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(extract_terms_from_abstract, ab, pico) for ab in hop2_hop3_with_abstract]
            for fut in as_completed(futures):
                try:
                    abstract_terms_set.update(fut.result())
                except Exception as e:
                    print(f"  Abstract extraction failed: {e}")
        print(f"Abstract-derived terms (hop-0 + hop-2 + hop-3): {len(abstract_terms_set)} unique terms")
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

    # Seed title terms are protected: get their wildcarded form and exclude from cleaning
    seed_title_population_wc: set[str] = set()
    seed_title_intervention_wc: set[str] = set()
    if seed_title_population:
        seed_title_population_wc = set(add_wildcards(sorted(seed_title_population), pico))
    if seed_title_intervention:
        seed_title_intervention_wc = set(add_wildcards(sorted(seed_title_intervention), pico))

    # ── 7g) Gemini cleaning: intervention only; seed_title terms are skipped and merged back after ─
    print("\nCleaning term lists for PubMed (Gemini); seed title terms are protected …")
    population_freetext_to_clean = [t for t in population_freetext_for_cleaning if t not in seed_title_population_wc]
    intervention_freetext_to_clean = [t for t in intervention_freetext_for_cleaning if t not in seed_title_intervention_wc]
    cleaned = clean_search_terms_for_pubmed(
        pico,
        sorted(final_population_mesh),
        population_freetext_to_clean,
        sorted(final_intervention_mesh),
        intervention_freetext_to_clean,
    )
    final_population_mesh = set(cleaned["population_mesh"])
    final_population_freetext = set(cleaned["population_freetext"]) | seed_title_population_wc
    final_intervention_mesh = set(cleaned["intervention_mesh"])
    final_intervention_freetext = set(cleaned["intervention_freetext"]) | seed_title_intervention_wc

    # Demographic hard ban for population (MeSH + freetext), after cleaning, before query building.
    banned_mesh_demo = {
        "humans",
        "male",
        "female",
        "adult",
        "young adult",
        "middle aged",
        "aged",
        "aged 80 and over",
        "adolescent",
        "child",
        "child preschool",
        "infant",
        "infant newborn",
        "pregnancy",
    }
    banned_freetext_bases = {
        "adult",
        "adults",
        "adult life",
        "child",
        "children",
        "children and adolescents",
        "adolescent",
        "adolescents",
        "adolescence",
        "infant",
        "infants",
        "infancy",
        "infants and toddlers",
        "toddlers",
        "neonatal",
        "neonate",
        "neonates",
        "pediatric",
        "paediatric",
        "childhood",
        "newborn",
        "newborns",
        "elderly",
        "geriatric",
        "young adult",
        "middle aged",
        "pregnant",
        "pregnancy",
    }

    # Seed population keywords from seed titles + PICO population, used to preserve disease-specific phrases.
    seed_keywords: set[str] = set()
    pop_text = (pico.get("population") or "").lower()
    for text in list(seed_title_population or []) + [pop_text]:
        for token in re.split(r"[^a-z0-9]+", text.lower()):
            token = token.strip()
            if len(token) >= 4:
                seed_keywords.add(token)

    def _is_demo_freetext(term: str) -> bool:
        base = term.rstrip("*").lower().strip()
        if not base:
            return False
        # Keep if any seed keyword appears in the term (disease-specific phrase).
        if any(kw in base for kw in seed_keywords):
            return False
        return any(base == b or base.startswith(b) for b in banned_freetext_bases)

    # Apply demographic ban to population
    before_mesh = len(final_population_mesh)
    final_population_mesh = {
        t
        for t in final_population_mesh
        if t.strip().lower() not in banned_mesh_demo
    }
    removed_mesh = before_mesh - len(final_population_mesh)

    before_free = len(final_population_freetext)
    final_population_freetext = {
        t
        for t in final_population_freetext
        if not _is_demo_freetext(t)
    }
    removed_free = before_free - len(final_population_freetext)
    if removed_mesh or removed_free:
        print(
            f"Demographic ban removed {removed_mesh} population MeSH term(s) and "
            f"{removed_free} population freetext term(s)."
        )

    # Apply the same demographic ban to intervention (no seed-keyword exception; demographics are never useful)
    before_int_mesh = len(final_intervention_mesh)
    final_intervention_mesh = {
        t
        for t in final_intervention_mesh
        if t.strip().lower() not in banned_mesh_demo
    }
    removed_int_mesh = before_int_mesh - len(final_intervention_mesh)

    before_int_free = len(final_intervention_freetext)
    final_intervention_freetext = {
        t
        for t in final_intervention_freetext
        if not any(
            t.rstrip("*").lower().strip().startswith(b) or t.rstrip("*").lower().strip() == b
            for b in banned_freetext_bases
        )
    }
    removed_int_free = before_int_free - len(final_intervention_freetext)
    if removed_int_mesh or removed_int_free:
        print(
            f"Demographic ban removed {removed_int_mesh} intervention MeSH term(s) and "
            f"{removed_int_free} intervention freetext term(s)."
        )

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
