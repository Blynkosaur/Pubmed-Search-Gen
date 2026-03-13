## Otto Take‑Home: Systematic Review Search Pipeline

This repo contains a prototype pipeline that reverse‑engineers a systematic review’s PICO and search strategy, builds a PubMed query, and evaluates recall of PubMed search result sets against the review’s own included studies.

The core flow is:

- **Input**: a systematic review PDF and, optionally, an Included Studies spreadsheet and NBIB search result files.
- **Processing**: Gemini + PubMed + TF‑IDF are used to extract PICO, derive search terms, and construct a PubMed query.
- **Output**: a PubMed Boolean query string and recall metrics for NBIB search sets.

---

## 1. Repository layout

- `main.py`  
  End‑to‑end driver for a single systematic review PDF: builds/loads a references cache, extracts PICO, filters references by similarity, asks Gemini for search terms, and builds a PubMed query.

- `gemini.py`  
  Functions that call Gemini (Google GenAI) to:
  - Extract the SR’s own PICO (`pico_extractor`).
  - Reduce PICO to 3 key concepts per facet (`get_pico_keywords`).
  - Extract search terms from the SR manuscript (`extract_terms_from_sr`).
  - Extract search terms from included references (`extract_terms`).
  - Merge and filter extracted terms based on key concepts and heuristic rules (`merge_terms`, `filter_terms_by_key_concepts`, `filter_extracted_terms`).

- `pubmed.py`  
  Helpers for parsing reference lists from PDFs and talking to NCBI E‑utilities:
  - Parse the PDF’s references into structured `Reference` objects (`parse`).
  - Fetch metadata (PMID, DOI, title, abstract, MeSH) for DOIs/titles (`fetch_references_metadata`, `fetch_metadata_for_identifiers`).
  - Turn a PubMed query into an NBIB record set (`search_and_fetch_nbib`).

- `query_builder.py`  
  Turns cleaned/filtered PICO term dictionaries into a final PubMed Boolean query string (`build_query`).

- `filter.py`  
  Utilities to parse and TF‑IDF‑filter raw reference lists (mainly useful for inspection / debugging).

- `src/recall_nbib_included_studies.py`  
  Standalone script to compute recall of NBIB/RIS search results vs an Included Studies Excel.

- `data/`  
  Example systematic reviews, included‑studies spreadsheets, NBIB search sets, and reference caches.

---

## 2. End‑to‑end pipeline from SR PDF to PubMed query

### 2.1. Step 1 – Build or load the references cache

For a given SR PDF (e.g. `data/151 - Moiz 2025/Moiz 2025.pdf`), the pipeline needs structured metadata (title, abstract, MeSH) for the references that the SR cites.

This metadata is cached in a JSON file next to the PDF:

- Cache path: `<pdf_directory>/<pdf_stem>_references.json`  
  e.g. `data/151 - Moiz 2025/Moiz 2025_references.json`

In `main.py`:

```12:29:/Users/bryanlin/OttoTakehome/main.py
def run(pdf_path: Path) -> None:
    pdf_path = Path(pdf_path)
    cache_path = _ref_cache_path(pdf_path)
    if not cache_path.exists():
        print(f"No references cache at {cache_path}. Building it from {pdf_path.name} (this may take a few minutes)...")

        # 1) Parse references from the PDF
        refs = parse_pdf_references(pdf_path)
        ...
        # 2) Build identifiers: DOI if present, else parsed title
        identifiers = [...]
        ...
        # 3) Fetch PubMed metadata for each identifier
        metadata = fetch_metadata_for_identifiers(identifiers)
        ...
        # 4) Normalize and save as JSON
        cache_path.write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with cache_path.open("r", encoding="utf-8") as f:
            references = json.load(f)
```

Under the hood:

- `pubmed.parse(pdf_path)`:
  - Reads the PDF.
  - Locates the References section.
  - Splits it into individual references.
  - Extracts a best‑effort DOI, year, and title for each.

- `pubmed.fetch_metadata_for_identifiers(identifiers)`:
  - For each DOI or title, calls PubMed’s ESearch + EFetch.
  - Returns metadata dicts with at least `pmid`, `doi`, `title`, `abstract`, and `mesh_terms`.

The resulting list of metadata dicts is written to `<pdf_stem>_references.json` and re‑used on subsequent runs.

### 2.2. Step 2 – Extract the SR’s PICO with Gemini

`gemini.pico_extractor(pdf_path)`:

- Reads the full text of the SR PDF.
- Sends a prompt to Gemini asking:
  - “Given this manuscript, what is the review’s own Population and Intervention?”
- Expects JSON:

```python
{"population": "...", "intervention": "..."}
```

This gives a concise machine‑readable PICO summary directly from the paper.

### 2.3. Step 3 – Compute PICO‑similarity scores for references (TF‑IDF)

In `main.run`:

- Build a PICO text string:

```12:39:/Users/bryanlin/OttoTakehome/main.py
pico = pico_extractor(pdf_path)
pico_text = " ".join(str(pico.get(k, "")) for k in ("population", "intervention"))
```

- Turn each reference into a “document”:

```12:18:/Users/bryanlin/OttoTakehome/main.py
def _doc_for_ref(rec: dict) -> str:
    abstract = (rec.get("abstract") or "").strip()
    if abstract:
        return abstract
    mesh = rec.get("mesh_terms") or []
    if mesh:
        return " ".join(str(m) for m in mesh if m)
    return (rec.get("title") or "").strip()
```

- Use TF‑IDF + cosine similarity:
  - Documents = `[pico_text] + ref_docs`.
  - `TfidfVectorizer(stop_words="english")`.
  - Compute cosine similarity between the PICO vector and each reference.

- Keep only PICO‑relevant references:
  - Threshold is `0.05` by default.
  - Those references become the context for search‑term extraction.

### 2.4. Step 4 – Extract search terms (MeSH + freetext) with Gemini

There are two main extraction passes in `gemini.py`:

1. **From the SR text itself** – `extract_terms_from_sr(pdf_path)`  
   - Gemini is asked to read the manuscript and locate where the authors describe their search strategy.
   - It extracts:
     - `study_design` (e.g. `randomized_controlled_trial`, `observational`, `systematic_review`, or `any`).
     - Population terms: `{"mesh": [...], "freetext": [...]}`
     - Intervention terms: same structure.

2. **From the filtered references** – `extract_terms(pico, references)`  
   - Gemini is given:
     - The PICO description.
     - Titles, abstracts, and MeSH terms from the most PICO‑relevant references.
   - It extracts MeSH and freetext terms for population and intervention, plus a study design label.

These two outputs are then merged:

- `merge_terms(sr_terms, ref_terms)`:
  - Unions `mesh` and `freetext` lists per facet (SR‑reported + reference‑derived).
  - De‑duplicates while preserving SR‑first order.
  - Picks a final `study_design`.

### 2.5. Step 5 – Narrow terms to key PICO concepts

To avoid over‑broad or off‑topic terms, the pipeline runs additional Gemini filters:

1. **Key concepts from PICO** – `get_pico_keywords(pico)`  
   - Asks Gemini for exactly **3 key concepts** for:
     - Population (who the patients are).
     - Intervention (what is being done).

2. **Filter by key concepts** – `filter_terms_by_key_concepts(terms, key_concepts)`  
   - Gemini is given:
     - The merged terms.
     - The 3 key concepts per facet.
   - It returns the same shape, but with only terms that clearly relate to those concepts per facet.

3. **Apply rule‑based filtering** – `filter_extracted_terms(terms, references)`  
   - Another Gemini pass with detailed rules:
     - Population = “who the patients are” (diagnoses/conditions/procedures that determine eligibility).
     - Intervention = core mechanism/modality (what makes the intervention distinct).
     - Prefer specific over broad.
     - Keep about 6–10 terms per list.
     - Prefer terms that appear frequently in the provided abstracts.
   - Returns a compact and focused set of terms for each facet.

At the end of this step, you have a small, high‑signal set of search terms for population and intervention plus a study design label.

### 2.6. Step 6 – Build the PubMed query

`query_builder.build_query(terms)`:

- Input: a dict like:

```python
{
  "study_design": "randomized_controlled_trial",
  "population": {"mesh": [...], "freetext": [...]},
  "intervention": {"mesh": [...], "freetext": [...]},
}
```

- For each facet:
  - MeSH → `"term"[MeSH Terms]`
  - Freetext:
    - Drops single‑word terms.
    - Keeps only multi‑word phrases.
    - Wraps as `"term"[Title/Abstract]`.

- Builds facet clauses and combines them:
  - If both population and intervention present: `(population_clause) AND (intervention_clause)`.
  - If only one facet has terms, uses that facet alone.

- If `study_design == "randomized_controlled_trial"`:
  - Adds an RCT filter: `"Randomized Controlled Trial"[pt]`.

The final query string is printed by `main.py` and can be pasted directly into PubMed.

---

## 3. Evaluating search recall with Included Studies + NBIB

The pipeline also supports measuring how well an NBIB search set retrieves a review’s own included studies.

Script: `src/recall_nbib_included_studies.py`

### 3.1. Inputs

- **Included studies Excel**  
  e.g. `data/151 - Moiz 2025/Moiz 2025 Included Studies.xlsx`
  - Columns (detected heuristically): DOI, PubMed ID, Title, Year.

- **NBIB or RIS search results**  
  e.g. `data/151 - Moiz 2025/pubmed-ObesityMeS-set.nbib`
  - Produced by running the generated PubMed query (or any other query) and exporting results as MEDLINE/NBIB.

### 3.2. Matching logic

For each included study:

1. Normalize:
   - DOI (strip protocol, lowercase, remove trailing slash).
   - PubMed ID (digit string).
   - Title (lowercase, remove punctuation, collapse whitespace).

2. Load bib records from `.nbib` / `.ris`:
   - Collect normalized DOI, PMID, title, and year per record.

3. Check if the included study is present in the bib:
   - DOI match OR
   - PMID match OR
   - Fuzzy title match (using `difflib.SequenceMatcher`) with optional year check.

### 3.3. Metrics

The script prints:

- `Included studies (Excel)` = total rows in the included‑studies file that have at least some identifier.
- `Included studies found in bib` = how many of those are matched in the NBIB/RIS.
- `Total studies in bib file(s)` = total records in the NBIB/RIS.
- `Recall %` = `(included found in bib) / (total included) * 100`.
- `Ratio (included in bib / total in bib)` = a rough measure of precision (how dense the included studies are in the bib set).

With `-l/--list`, it also prints each included study labeled as `FOUND` or `NOT FOUND`, with its DOI and PubMed ID, so you can see which ones are missed.

Example (Moiz 2025 + `pubmed-ObesityMeS-set.nbib`):

```text
Included studies (Excel):     26
Included studies found in bib: 23
Total studies in bib file(s):  667
Recall %:                     88.46%
Ratio (included in bib / total in bib): 0.0345
```

---

## 4. Typical workflow

For a new SR PDF:

1. **Generate the query**
   - Run:

```bash
python3 main.py --pdf "data/151 - Moiz 2025/Moiz 2025.pdf"
```

   - On first run, this:
     - Parses references and builds `Moiz 2025_references.json`.
     - Extracts PICO from the manuscript (Gemini).
     - TF‑IDF filters references by similarity to the PICO.
     - Extracts/filters PICO search terms (Gemini).
     - Prints a PubMed query.

2. **Run the query in PubMed and export results to NBIB**
   - Paste the query into PubMed.
   - Export results as MEDLINE (NBIB) to e.g. `data/151 - Moiz 2025/pubmed-ObesityMeS-set.nbib`.

3. **Evaluate recall vs Included Studies**
   - Run:

```bash
python3 src/recall_nbib_included_studies.py \
  "data/151 - Moiz 2025/Moiz 2025 Included Studies.xlsx" \
  "data/151 - Moiz 2025/pubmed-ObesityMeS-set.nbib"
```

   - Optionally add `-l` to see which included studies are missing from the NBIB set.

This gives a full loop from **manuscript PDF → PICO → PubMed query → NBIB results → recall metric** for each systematic review.

