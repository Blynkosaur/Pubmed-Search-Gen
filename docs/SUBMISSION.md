# Submission: Challenges and Insights

This document summarizes **challenges** and **insights** from developing the Otto Take‑Home systematic review search pipeline. It complements the [README](README.md) and the design notes in the [docs](docs/) folder (e.g. *Strategies for Otto Take Home.pdf*).

---

## Insights

### Term selection and recall

- **Precision vs recall**: To get a sensitive (high-recall) list, use **precise terms** (e.g. “renal failure” vs “kidney”). For good accuracy and recall together, include **all synonym terms** within PICO and keep terms precise so the query returns suitable results.
- **MeSH vs freetext**: MeSH terms are **very unpredictable** — a slight change can make the number of results explode. So: keep **MeSH terms quite specific** in the query and add **more variants and synonyms in freetext** to broaden safely. Basic idea: get as much context as possible, filter ideas, then broaden terms.
- **Graph-based context**: Using a citation graph (hop-0 seeds → hop-1 citing papers → hop-2/hop-3 co-citations) gives **more specific terms and their variants** and reduces over-reliance on MeSH alone, with less noise when combined with freetext.

### Pipeline design choices

1. **PROSPERO terms**: PROSPERO content is highly relevant; author-provided terms are **protected from cleaning** and merged back after the cleaning step.
2. **LLM unpredictability**: For critical extraction steps, we send **two identical Gemini calls** and **union the results** (e.g. call 1 → A, B; call 2 → A, C; final → A, B, C) to improve recall and stability.
3. **Demographic terms**: Demographic information (age, race/ethnicity) tends to bloat results without adding relevance; we apply a **hard ban** on those descriptors (medical conditions like pregnancy are allowed).
4. **Seed paper titles**: We **analyze seed paper titles** as a baseline and treat title-derived terms as **mandatory** in the final query (protected from cleaning).
5. **Population MeSH**: Population MeSH is usually already concise; we **do not augment** it with hop-2/hop-3, only with hop-0 seeds and PROSPERO. Augmentation from hop-2/hop-3 is limited to **intervention** and capped at **10 new MeSH terms** total.
6. **Broad single-word terms**: Broad terms are often single words; we **restrict or omit** generic single-word terms in splitting/cleaning unless they are specific medical/scientific terms.

---

## Challenges

### LLM behavior

- **Calibration**: Models are often **too harsh or too loose** on term extraction or cleaning, which required many **prompt tweaks** and iteration.
- **Instruction following**: Output does not always follow instructions exactly, so we use **post-processing** (e.g. hard caps, filtering, union of two calls) and **explicit prompt rules** (e.g. “return only valid JSON”, “do not add new terms”).
- **Case-by-case variation**: Some systematic reviews behave differently; **case-by-case testing** on a variety of SRs is necessary to catch edge cases and tune prompts.

### Graph building and performance

- **Time**: Building the citation graph involves **fetching and counting** many citing works (hop-1) and then resolving hop-2/hop-3. This can take several minutes (e.g. ~3+ minutes for the graph). Results are **cached** in `citation_graph.json` so subsequent runs reuse the graph.
- **Filtering and mistakes**: The large list of candidate terms from the graph takes time to filter and can still lead to **mistakes** (over-inclusion or under-inclusion), so we combine graph-derived terms with strict splitting rules and cleaning.

### Recall and coverage

- **Vague topics**: When topics are vague, the LLM may not clearly separate relevant from irrelevant terms, and the query can return **very large result sets**. Mitigation: stricter splitting rules (e.g. same-disease-only for population, no generic single words), case-by-case testing, and optional **intervention-only** query when population has 0–1 term.
- **Missing metadata**: Some articles lack MeSH; some are not in PubMed; some DOIs do not resolve. The pipeline still aims for good recall by combining MeSH, freetext, and citation context, and by protecting seed and PROSPERO terms.

---

## Reference

Design notes and strategy sketches: **docs/Strategies for Otto Take Home.pdf**.
