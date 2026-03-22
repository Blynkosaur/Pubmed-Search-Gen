## Otto Take‑Home: Systematic Review Search Pipeline

For design notes, see the **docs** folder (e.g. *Strategies for Otto Take Home.pdf*).  
For **challenges and insights** from building the pipeline, see **[docs/SUBMISSION.md](SUBMISSION.md)**.

This repo contains a prototype pipeline that reverse‑engineers a systematic review’s PICO and search strategy, builds a PubMed query, and evaluates recall of PubMed search result sets against the review’s own included studies.

Overall, the structure is satisfactory to me , at this point it's just about tweaking prompts to filter better, or deciding how to extract terms

The core flow is:

- **Input**: a systematic review PDF and, optionally, an Included Studies spreadsheet (for seed studies), an N count, and a PROSPERO registration PDF.
- **Processing**: Seeds are chosen from the Excel or from PDF references; a citation graph is built via OpenAlex; Gemini extracts PICO, classifies and augments MeSH, extracts freetext terms from abstracts and seed titles, and splits them by PICO; terms are wildcarded, cleaned (with seed and PROSPERO terms protected), and age/race demographic terms are banned; population MeSH is merged into population freetext, and the PubMed query is built.
- **Output**: a PubMed Boolean query string and recall metrics for NBIB search sets.

---

## 1. Repository layout

- `main.py`  
  End‑to‑end driver: loads or builds seed references, builds the citation graph (OpenAlex), extracts PICO, runs the MeSH and freetext term pipeline, cleans (with protected terms), applies demographic bans, merges population MeSH into population freetext, and builds the PubMed query.

- `gemini.py`  
  Functions that call Gemini (Google GenAI) to:
  - Extract the SR’s PICO from the PDF (`pico_extractor`).
  - Parse PROSPERO registration PDF for author search terms (`parse_prospero`).
  - Classify seed-paper MeSH into population / intervention / others (`classify_seed_mesh_terms`).
  - Augment seed MeSH with relevant terms from hop-2/hop-3 papers (`augment_seed_mesh_with_hop1`), capped at 10 new terms total.
  - Extract search terms from abstracts (`extract_terms_from_abstract`) and from seed paper titles (`extract_terms_from_seed_titles`).
  - Split freetext terms into population vs intervention (`split_freetext_terms_by_pico`).
  - Add wildcards to freetext terms (`add_wildcards`).
  - Clean term lists for PubMed (`clean_search_terms_for_pubmed`).
  - Format the final PubMed query (`build_pubmed_query`).
  - Extract titles from reference strings when DOI is missing (`extract_titles_from_references`).

  For extraction steps (not cleaning or query formatting), the pipeline runs two concurrent Gemini calls and unions the results to improve recall.

- `pubmed.py`  
  Helpers for parsing reference lists from PDFs and talking to NCBI E‑utilities (parse references, fetch metadata by DOI/title, search and fetch NBIB).

- `openalex.py`  
  Citation graph: resolve DOIs/titles to OpenAlex works, fetch citing papers (hop 1), and select hop-2/hop-3 reference sets.

- `query_builder.py`  
  Utilities to turn PICO term dicts into PubMed Boolean strings (`build_query`, `build_query_two_blocks`); the main pipeline uses `gemini.build_pubmed_query` for the final query.

- `filter.py`  
  Utilities to parse and TF‑IDF‑filter raw reference lists (mainly for inspection / debugging).

- `src/recall_nbib_included_studies.py`  
  Standalone script to compute recall of NBIB/RIS search results vs an Included Studies Excel.

- `data/`  
  Example systematic reviews, included‑studies spreadsheets, NBIB search sets, and reference caches.

---

## 2. End‑to‑end pipeline from SR PDF to PubMed query

### 2.1. Step 1 – Seed studies

Seeds are chosen in one of two ways:

- **From Included Studies Excel** (when `--xlsx` and `--N` are provided):  
  `N` random rows are loaded from the Excel. For rows without DOI, OpenAlex is queried by title to try to find a DOI. Only rows with at least a DOI or title are kept.

- **From the SR PDF** (default):  
  References are parsed from the PDF. For each reference, DOI is used when present; otherwise Gemini extracts the title from the raw reference string. Only references with DOI or title are kept.

These seeds are used as **hop 0** in the citation graph.

### 2.2. Step 2 – Citation graph (OpenAlex)

- The pipeline builds or loads a citation graph for the seed DOIs/titles.
- **Hop 0**: seed papers.
- **Hop 1**: papers that cite any seed (from OpenAlex).
- **Hop 2**: top 30 references (by connection count) of hop-1 papers.
- **Hop 3**: top 10 papers by connections to those top-30 hop-2 refs.

The pipeline uses **hop 0 + hop 2 + hop 3** as the reference set for term extraction (only refs with abstract or MeSH are kept). The graph is cached as `citation_graph.json` next to the PDF.

### 2.3. Step 3 – PICO extraction

- `gemini.pico_extractor(pdf_path)` reads the SR PDF and asks Gemini for the review’s Population and Intervention (and comparator/outcome if present).
- Returns a dict with keys such as `summary`, `population`, `intervention`, `comparator`, `outcome`.

### 2.4. Step 4 – Optional PROSPERO

- If `--prospero` points to a PROSPERO registration PDF, `gemini.parse_prospero(prospero_path)` extracts author-provided search terms (population, intervention, MeSH, full query as reference).
- These terms are added with priority to the final population/intervention blocks and are **protected from cleaning** (excluded from the cleaning input and merged back after).

### 2.5. Step 5 – MeSH pipeline

1. **Seed MeSH**  
   All MeSH terms from hop-0 seed papers are collected. `classify_seed_mesh_terms(seed_mesh_list, pico)` (Gemini) classifies them into `population`, `intervention`, and `others`. If PROSPERO was parsed, its population and intervention MeSH are appended to the classified lists.

2. **Augmentation**  
   `augment_seed_mesh_with_hop1(pico, seed_population, seed_intervention, hop2_hop3_mesh_list)` (Gemini) adds relevant MeSH from hop-2 and hop-3 papers to the **intervention** list only (population MeSH remains seed-only, plus PROSPERO if present). Augmentation from hop-2/hop-3 is capped at **10 new MeSH terms total** across population and intervention; if more than 10 related-paper terms look relevant, only the 10 most specific and central ones are kept.

3. **Final MeSH blocks**
   - Population: seed (+ PROSPERO).
   - Intervention: augmented seed (+ PROSPERO), with at most 10 additional MeSH terms coming from hop-2/hop-3.

### 2.6. Step 6 – Freetext pipeline

1. **Abstracts**  
   Terms are extracted from hop-0 abstracts and from hop-2 + hop-3 abstracts via `extract_terms_from_abstract` (Gemini, concurrent over abstracts). All are merged into a single freetext pool.

2. **Seed paper titles**  
   `extract_terms_from_seed_titles(hop0_titles, pico)` (Gemini) extracts population and intervention phrases from seed titles. These are **mandatory** in the final query and **protected from cleaning** (excluded from cleaning input, merged back after). They are also added to the freetext pool.

3. **PROSPERO freetext**  
   If PROSPERO was parsed, its population terms, intervention terms, and search terms are added to the freetext pool.

4. **Split by PICO**  
   `split_freetext_terms_by_pico(all_freetext_set, pico)` (Gemini) assigns each freetext term to population or intervention (or drops it). Result is two lists: population freetext and intervention freetext.

5. **Final freetext blocks**  
   Combined with PROSPERO population/intervention/search terms as above.

### 2.7. Step 7 – Wildcards and cleaning

1. **Wildcards**  
   All population and intervention freetext terms are passed through `add_wildcards(..., pico)` (Gemini) to get PubMed-friendly forms (e.g. trailing `*` where appropriate).

2. **Protected terms**
   - **Seed title terms**: Their wildcarded forms are computed; these terms are excluded from the lists sent to cleaning and merged back after.
   - **PROSPERO terms**: PROSPERO MeSH are excluded from the mesh lists sent to cleaning; PROSPERO freetext (wildcarded) are excluded from the freetext lists sent to cleaning. All are merged back after cleaning.

3. **Cleaning**  
   `clean_search_terms_for_pubmed(...)` (Gemini) cleans and deduplicates the non-protected terms. Then final population/intervention MeSH and freetext are merged with protected sets.

### 2.8. Step 8 – Demographic ban

- Only **age- and race-related** descriptors are banned (not medical conditions like pregnancy or disability).
- A fixed list of age-related MeSH (e.g. Adult, Child, Adolescent, Aged) is removed from both population and intervention MeSH.
- A fixed list of age- and race-related freetext bases (e.g. adult, children, pediatric, elderly, racial, ethnic, Black, White, Asian, Hispanic) is applied:
  - **Population freetext**: A term is removed unless it contains a “seed keyword” (from PICO population and seed title population) that indicates disease-specific phrasing; otherwise if it matches a banned base it is dropped.
  - **Intervention freetext**: Any term that matches a banned age/race base is dropped.

### 2.9. Step 9 – Pre-query merge and query build

- **Population freetext** is augmented with **all population MeSH** (union, deduplicated), so population MeSH terms also appear in the population freetext set for the query.

- `gemini.build_pubmed_query(...)` (Gemini) formats the four term sets into a single PubMed Boolean string: two blocks (population AND intervention), MeSH with `[MeSH Terms]`, freetext with `[Title/Abstract]`. This step is formatting only; it does not add or remove terms.

The final query string is printed by `main.py`.

---

## 3. Evaluating search recall with Included Studies + NBIB

The pipeline supports measuring how well an NBIB search set retrieves a review’s included studies.

Script: `src/recall_nbib_included_studies.py`

### 3.1. Inputs

- **Included studies Excel**  
  e.g. `data/151 - Moiz 2025/Moiz 2025 Included Studies.xlsx`  
  Columns (detected heuristically): DOI, PubMed ID, Title, Year.

- **NBIB or RIS search results**  
  e.g. `data/151 - Moiz 2025/pubmed-ObesityMeS-set.nbib`  
  From running the generated PubMed query (or any query) and exporting as MEDLINE/NBIB.

### 3.2. Matching logic

For each included study: normalize DOI, PMID, and title; load bib records from `.nbib`/`.ris`; match by DOI, or PMID, or fuzzy title (and optional year).

### 3.3. Metrics

- Included studies (Excel), included found in bib, total studies in bib, Recall %, and ratio (included in bib / total in bib). With `-l/--list`, each included study is labeled FOUND or NOT FOUND.

---

## 4. Typical workflow

**Generate the query** (seeds from PDF references):

```bash
python3 main.py --pdf "data/151 - Moiz 2025/Moiz 2025.pdf"
```

**Generate the query** (seeds from Included Studies Excel, e.g. 5 random):

```bash
python3 main.py --pdf "data/151 - Moiz 2025/Moiz 2025.pdf" \
  --xlsx "data/151 - Moiz 2025/Moiz 2025 Included Studies.xlsx" --N 5
```

**With PROSPERO registration** (author terms added and protected from cleaning):

```bash
python3 main.py --pdf "data/151 - Moiz 2025/Moiz 2025.pdf" \
  --xlsx "data/151 - Moiz 2025/Moiz 2025 Included Studies.xlsx" --N 5 \
  --prospero "data/151 - Moiz 2025/Moiz 2025 PROSPERO.pdf"
```

On first run (without an existing citation graph), the pipeline will use OpenAlex to build the graph. PICO is extracted from the PDF, then the MeSH and freetext pipelines run (with optional PROSPERO), cleaning (seed and PROSPERO protected), demographic ban (age/race only), code-only population-mesh merge into population freetext, and query build. The PubMed query is printed.

**Run the query in PubMed** and export results to NBIB.

**Evaluate recall**:

```bash
python3 src/recall_nbib_included_studies.py \
  "data/151 - Moiz 2025/Moiz 2025 Included Studies.xlsx" \
  "data/151 - Moiz 2025/pubmed-ObesityMeS-set.nbib"
```

Use `-l` to list which included studies are missing from the NBIB set.

This gives a full loop: **manuscript PDF (and optional Excel/PROSPERO) → seeds → citation graph → PICO → MeSH + freetext → cleaning (protected terms) → demographic ban (age/race) → population mesh merged into population freetext → PubMed query → NBIB → recall**.

---

## 5. Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Set `GEMINI_API_KEY` in the environment (or in a `.env` file next to the project) for Gemini calls. The OpenAlex API is used without a key (rate-limited by request).
