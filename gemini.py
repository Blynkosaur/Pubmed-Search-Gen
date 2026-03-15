from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import pymupdf
from google import genai


_MODEL_NAME = "gemini-2.5-flash"

_PICO_FACETS = ("population", "intervention", "comparator", "outcome")


def _load_dotenv_if_present() -> None:
    """
    Load key=value pairs from a local .env file into os.environ
    if they are not already set.
    """
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # If .env can't be read, silently fall back to existing environment.
        pass


def _load_api_key() -> str:
    _load_dotenv_if_present()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    return api_key


def _extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    try:
        parts = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            parts.append(text)
        return "\n".join(parts)
    finally:
        doc.close()


def pico_extractor(pdf_path: Union[str, Path]) -> Dict[str, object]:
    """
    Extract the systematic review's own PICO using Gemini.

    - Reads the full text of the SR PDF.
    - Sends the text to Gemini and asks for Population, Intervention,
      Comparator, Outcome.
    - Returns a dict with those four keys.
    """
    pdf_path = Path(pdf_path)
    full_text = _extract_pdf_text(pdf_path)

    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    # Truncate very long texts so the prompt stays within model limits.
    max_chars = 40_000
    truncated = full_text[:max_chars]

    prompt = (
        "You are assisting with systematic review reproducibility.\n"
        "Given the following systematic review manuscript text, identify the review's\n"
        "own PICO: Population, Intervention, Comparator, and Outcome. Also provide\n"
        "a one-sentence summary of what the study is about.\n\n"
        "Return a JSON object with exactly these keys:\n"
        '{\n'
        '  "summary": string (one sentence: what the study is about),\n'
        '  "population": string,\n'
        '  "intervention": string,\n'
        '  "comparator": string,\n'
        '  "outcome": string\n'
        "}\n\n"
        "If a facet is not applicable or not clearly stated, use an empty string.\n\n"
        "Manuscript text:\n"
        f"{truncated}\n\n"
        "Return only valid JSON, no markdown backticks"
    )

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text)
    return parsed


def parse_prospero(pdf_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract search terms and criteria from a PROSPERO registration PDF.
    Returns dict with search_terms, population_terms, intervention_terms,
    mesh_terms_population, mesh_terms_intervention, full_query.
    Only extracts what is explicitly written; does not infer.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PROSPERO PDF not found: {pdf_path}")
    full_text = _extract_pdf_text(pdf_path)
    max_chars = 50_000
    truncated = full_text[:max_chars]

    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    prompt = (
        "Extract the following from this PROSPERO registration document.\n\n"
        "1. Search terms or keywords explicitly listed\n"
        "2. Inclusion criteria — population description (terms/phrases used)\n"
        "3. Inclusion criteria — intervention or exposure description (terms/phrases used)\n"
        "4. Any MeSH terms explicitly listed — split into population-related vs intervention/exposure-related if possible\n"
        "5. Full search string if present (exact copy)\n\n"
        "Extraction rules for all term arrays (search_terms, population_terms, intervention_terms, mesh_terms_*):\n"
        "- Output only single words or phrases of maximum 4 words.\n"
        "- Exclude form labels such as Q1, Q2, Q3.\n"
        "- Exclude full sentences.\n"
        "- Exclude punctuation and colons; strip them from extracted terms.\n\n"
        "What to extract:\n"
        "- Only extract disease names, condition synonyms, drug names, treatment names, and therapy types.\n"
        "- Do NOT extract eligibility criteria (e.g. histologically confirmed, cytologically confirmed, patients 18 or above) UNLESS the SR specifically targets a demographic group (e.g. infants, pediatric, elderly).\n"
        "- Do NOT extract staging terms (e.g. Stage I, Stage II, Stage III).\n"
        "- Do NOT extract study design terms (e.g. RCT, clinical trials, randomized controlled trial).\n"
        "- Do NOT extract generic modifiers (e.g. in combination, monotherapy, resectable).\n"
        "- Do NOT extract outcome terms (e.g. overall survival, adverse events, pathological response).\n"
        "- Do NOT extract \"Humans\" as a MeSH term.\n"
        "- Preserve hyphens exactly as written (e.g. anti-PD-1, PD-L1, CTLA-4).\n\n"
        "Return JSON only. Only extract what is explicitly written. Do not infer or generate new terms.\n"
        "{\n"
        '  "search_terms": [],\n'
        '  "population_terms": [],\n'
        '  "intervention_terms": [],\n'
        '  "mesh_terms_population": [],\n'
        '  "mesh_terms_intervention": [],\n'
        '  "full_query": ""\n'
        "}\n\n"
        "Document text:\n"
        f"{truncated}\n\n"
        "Return only valid JSON, no markdown backticks."
    )
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    raw_text = raw_text.strip()
    if not raw_text:
        return {
            "search_terms": [],
            "population_terms": [],
            "intervention_terms": [],
            "mesh_terms_population": [],
            "mesh_terms_intervention": [],
            "full_query": "",
        }
    parsed = json.loads(raw_text)

    def normalize_term(t: str) -> str | None:
        if not t or not isinstance(t, str):
            return None
        # Strip punctuation and colons
        s = re.sub(r"[\s,:;.]+", " ", t).strip()
        s = s.strip(".,;:()[]\"'")
        if not s:
            return None
        # Max 4 words
        words = s.split()[:4]
        s = " ".join(words).strip()
        if not s:
            return None
        # Skip form labels like Q1, Q2, Q3
        if re.match(r"^Q\d+$", s, re.IGNORECASE):
            return None
        return s

    def clean_terms(items: list) -> list:
        seen = set()
        out = []
        for x in items or []:
            v = normalize_term(x) if isinstance(x, str) else normalize_term(str(x))
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    return {
        "search_terms": clean_terms(parsed.get("search_terms")),
        "population_terms": clean_terms(parsed.get("population_terms")),
        "intervention_terms": clean_terms(parsed.get("intervention_terms")),
        "mesh_terms_population": clean_terms(parsed.get("mesh_terms_population")),
        "mesh_terms_intervention": clean_terms(parsed.get("mesh_terms_intervention")),
        "full_query": (parsed.get("full_query") or "").strip(),
    }


def add_wildcards(terms: List[str], pico: Dict[str, Any]) -> List[str]:
    """
    Add PubMed truncation wildcards (*) to freetext terms where it improves recall.
    Uses principle-based rules; temperature=0 for deterministic output.
    Returns the same number of terms, each either unchanged or with * where appropriate.
    """
    if not terms:
        return []
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a systematic review librarian optimizing PubMed freetext search terms.

PICO context:
Population: {pico.get('population', '')}
Intervention: {pico.get('intervention', '')}

For each term below add a wildcard * where truncation improves recall in PubMed.

Rules:
- Add * to the word root when truncation captures meaningful morphological variants
- Never add * to terms under 4 characters
- Never add * to abbreviations or acronyms
- Never add * if truncation makes the term too broad or ambiguous
- For multi-word phrases only consider truncating the final word
- Consider whether the truncated form would retrieve irrelevant results
- Only add * when the benefit to recall clearly outweighs the risk of noise

Terms: {terms}

Return JSON list only.
Return every original term, modified with * where appropriate.
Do not remove terms.
Do not add new terms.
"""
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt.strip(),
        config={"response_mime_type": "application/json", "temperature": 0},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text.strip())
    if not isinstance(parsed, list):
        return list(terms)
    orig_set = {str(t).strip() for t in terms if t}

    def is_valid(s: str) -> bool:
        s = s.strip()
        if s in orig_set:
            return True
        if s.endswith("*") and len(s) > 1:
            stem = s[:-1]
            if stem in orig_set:
                return True
            return any(orig.startswith(stem) or stem == orig for orig in orig_set)
        return False

    result = [str(t).strip() for t in parsed if isinstance(t, str) and is_valid(str(t).strip())]
    return result if result else list(terms)


def clean_search_terms_for_pubmed(
    pico: Dict[str, Any],
    population_mesh: List[str],
    population_freetext: List[str],
    intervention_mesh: List[str],
    intervention_freetext: List[str],
) -> Dict[str, List[str]]:
    """
    Single Gemini call: sanity-check the four term lists for a PubMed query.
    Remove only obvious noise (e.g. form labels, generic terms); keep terms that are at all relevant.
    Only returns terms from the original lists; does not add new terms.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a systematic review librarian doing a sanity check on search terms for a PubMed query.

PICO:
Population: {pico.get('population', '')}
Intervention: {pico.get('intervention', '')}
Outcome: {pico.get('outcome', '')}

Remove only obvious noise:
- Incomplete phrases or partial sentences
- Pure methodology or statistical terms
- Form labels or question numbers (e.g. Q1, Q2)
- Any term under 3 characters

Keep terms that are at all relevant to the PICO (population, intervention, or outcome).
Do not aggressively cut; this is a sanity check. When in doubt, keep the term.

Population MeSH: {population_mesh}
Population freetext: {population_freetext}
Intervention MeSH: {intervention_mesh}
Intervention freetext: {intervention_freetext}

Return JSON only:
{{
    "population_mesh": [],
    "population_freetext": [],
    "intervention_mesh": [],
    "intervention_freetext": []
}}

Only return terms from the original lists.
Do not add new terms.
Do not explain.
"""
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt.strip(),
        config={"response_mime_type": "application/json"},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text.strip())
    # Restrict to terms that appeared in the original lists
    orig_pm = set(str(x).strip() for x in population_mesh if x)
    orig_pf = set(str(x).strip() for x in population_freetext if x)
    orig_im = set(str(x).strip() for x in intervention_mesh if x)
    orig_if = set(str(x).strip() for x in intervention_freetext if x)
    return {
        "population_mesh": [t for t in (parsed.get("population_mesh") or []) if str(t).strip() in orig_pm],
        "population_freetext": [t for t in (parsed.get("population_freetext") or []) if str(t).strip() in orig_pf],
        "intervention_mesh": [t for t in (parsed.get("intervention_mesh") or []) if str(t).strip() in orig_im],
        "intervention_freetext": [t for t in (parsed.get("intervention_freetext") or []) if str(t).strip() in orig_if],
    }


def pick_primary_disease_mesh(pico: Dict[str, Any], population_mesh_list: List[str]) -> str | None:
    """
    Ask Gemini which of the seed population MeSH terms is the single best primary disease heading
    for the SR population. Returns that one term, or None if list is empty or response invalid.
    """
    if not population_mesh_list:
        return None
    population_text = (pico.get("population") or "").strip()
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    mesh_block = ", ".join(population_mesh_list)
    prompt = (
        f"Given this SR population: {population_text}\n\n"
        f"Which of these MeSH terms is the single best primary disease heading for this population? "
        f"Pick one: {mesh_block}\n\n"
        "Return only the term, nothing else."
    )
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt.strip(),
        config={"temperature": 0},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text:
        raw_text = str(response)
    term = raw_text.strip().strip('"')
    valid = {str(t).strip() for t in population_mesh_list if t}
    if term in valid:
        return term
    for t in population_mesh_list:
        if t and term.lower() == str(t).strip().lower():
            return str(t).strip()
    return population_mesh_list[0] if population_mesh_list else None


def build_pubmed_query(
    population_mesh: List[str],
    population_freetext: List[str],
    intervention_mesh: List[str],
    intervention_freetext: List[str],
    pico: Dict[str, Any],
) -> str:
    """
    Build a valid PubMed boolean query from the four term sets using Gemini.
    Exactly 2 blocks (population AND intervention); MeSH [MeSH Terms], freetext [Title/Abstract].
    Returns the query string only; temperature=0 for deterministic output.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a systematic review librarian building a PubMed boolean query.

PICO:
Population: {pico.get('population', '')}
Intervention: {pico.get('intervention', '')}
Outcome: {pico.get('outcome', '')}

Build a valid PubMed boolean query using exactly these terms.

Population MeSH terms: {population_mesh}
Population freetext terms: {population_freetext}
Intervention MeSH terms: {intervention_mesh}
Intervention freetext terms: {intervention_freetext}

Rules:
- Exactly 2 blocks: population AND intervention
- Within each block connect all terms with OR
- Between blocks use AND
- MeSH terms use [MeSH Terms] tag
- Freetext terms use [Title/Abstract] tag
- Wildcards are already in freetext terms do not modify them
- Do not add any new terms not in the lists above
- Do not add NOT operators
- Do not add study design filters
- Return the query string only no explanation no markdown

Format:
(
  "term1"[MeSH Terms]
  OR "term2"[MeSH Terms]
  OR "term3"[Title/Abstract]
  OR "term4*"[Title/Abstract]
)
AND
(
  "term5"[MeSH Terms]
  OR "term6"[Title/Abstract]
  OR "term7*"[Title/Abstract]
)
"""
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt.strip(),
        config={"temperature": 0},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    query = raw_text.strip()
    # Remove markdown code fence if present
    if query.startswith("```"):
        lines = query.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        query = "\n".join(lines)
    return query.strip()


def generate_search_terms_two_blocks(
    pico: Dict[str, Any],
    seed_mesh_terms: List[str],
    seed_abstracts: List[str],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Single Gemini call: generate population + intervention search terms for a
    PubMed boolean query. Uses PICO, seed paper MeSH, seed paper abstracts.
    Returns {"population": {"mesh": [], "freetext": []}, "intervention": {...}}.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    pico_block = "\n".join(
        f"  {k}: {v}" for k, v in pico.items() if isinstance(v, str) and v.strip()
    )
    intervention_text = (pico.get("intervention") or "").strip()

    seed_mesh_block = "\n".join(f"- {t}" for t in seed_mesh_terms[:50])
    seed_abstracts_block = "\n\n---\n\n".join(
        f"Abstract {i+1}:\n{a[:1500]}" for i, a in enumerate(seed_abstracts[:10])
    )

    prompt = (
        "You are a systematic review librarian building a PubMed boolean query.\n\n"
        "PICO:\n"
        f"{pico_block}\n\n"
        "These are MeSH terms from confirmed relevant seed papers:\n"
        f"{seed_mesh_block}\n\n"
        "These are abstracts from the seed papers:\n"
        f"{seed_abstracts_block}\n\n"
        "Generate search terms for exactly 2 blocks:\n\n"
        "Block 1 — Population/Disease:\n"
        "- 3-4 broad MeSH terms covering the disease and age group\n"
        "- 4-5 broad free text synonyms for the disease and age group\n"
        "- No phrases longer than 4 words\n\n"
        "Block 2 — Intervention/Symptoms:\n"
        "- 3-4 specific MeSH terms for symptoms and diagnosis\n"
        "- Use exact clinical symptom names from the PICO intervention section as free text terms\n"
        "- Add: time to diagnosis, diagnostic delay, symptom\n"
        "- No generic descriptors like \"red flag symptoms\" or \"presenting complaints\"\n"
        "- No phrases longer than 4 words\n\n"
        "Return JSON only:\n"
        '{"population": {"mesh": [], "freetext": []}, "intervention": {"mesh": [], "freetext": []}}'
    )
    if intervention_text:
        prompt += f'\n\nPICO intervention (use exact symptom names from here): "{intervention_text}"'

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text:
        raw_text = str(response)
    raw_text = raw_text.strip()
    if not raw_text:
        return {"population": {"mesh": [], "freetext": []}, "intervention": {"mesh": [], "freetext": []}}
    parsed = json.loads(raw_text)
    out = {"population": {"mesh": [], "freetext": []}, "intervention": {"mesh": [], "freetext": []}}
    for block in ("population", "intervention"):
        data = parsed.get(block)
        if isinstance(data, dict):
            out[block]["mesh"] = list(data.get("mesh") or [])[:8]
            out[block]["freetext"] = list(data.get("freetext") or [])[:8]
    return out


def classify_seed_mesh_terms(
    mesh_terms: List[str],
    pico: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Classify MeSH terms from hop-0 seed papers into population, intervention,
    or others (discard). Returns {"population": [], "intervention": [], "others": []}.
    """
    if not mesh_terms:
        return {"population": [], "intervention": [], "others": []}
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    terms_block = "\n".join(f"- {t}" for t in mesh_terms)
    population = (pico.get("population") or "").strip()
    intervention = (pico.get("intervention") or "").strip()
    comparator = (pico.get("comparator") or "").strip()
    outcome = (pico.get("outcome") or "").strip()
    prompt = f"""You are classifying MeSH terms from seed studies into PICO categories for a systematic review PubMed search.

The SR's PICO:
Population: {population}
Intervention: {intervention}
Comparator: {comparator}
Outcome: {outcome}

Classify each MeSH term into one of: population, intervention, others.

Rules:
- "population" = the specific disease or condition being studied and its direct synonyms/subtypes.
- "Humans" is never a defining characteristic of any SR population — always classify as others.
- Demographic terms (Male, Female, Aged, Middle Aged, Adult, Young Adult, Adolescent, Child, Infant, Aged 80 and over) should ONLY be classified as population if the SR specifically targets that demographic as a defining feature of the population. For example, if the population is "elderly patients with dementia" then "Aged" is population. If the population is "patients with NSCLC" then "Aged" is others — age is not what defines this population.
- Study methodology terms (Prospective Studies, Retrospective Studies, Randomized Controlled Trials, Follow-Up Studies, Treatment Outcome, Prognosis, Survival Rate) are always others.
- Generic parent terms (Neoplasms, Humans, Carcinoma) that are much broader than the SR's specific disease are others.
- Staging/grading terms (Neoplasm Staging, Neoplasm Grading) are others unless staging is the intervention or primary focus of the SR.
- "intervention" = treatments, therapies, drugs, procedures being compared.
- "others" = everything else — discard these.

Use the PICO population description to decide what counts as a defining characteristic vs incidental demographic.

MeSH terms from seed papers:
{terms_block}

Return JSON only with exactly three keys: population, intervention, others. Each value is an array of MeSH strings (exact strings from the list above)."""
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text:
        raw_text = str(response)
    raw_text = raw_text.strip()
    if not raw_text:
        return {"population": [], "intervention": [], "others": []}
    parsed = json.loads(raw_text)
    out = {"population": [], "intervention": [], "others": []}
    mesh_set = {str(t).strip() for t in mesh_terms if t}
    for key in ("population", "intervention", "others"):
        arr = parsed.get(key)
        if isinstance(arr, list):
            out[key] = [str(x).strip() for x in arr if str(x).strip() in mesh_set]
    return out


def augment_seed_mesh_with_hop1(
    pico: Dict[str, Any],
    seed_population: List[str],
    seed_intervention: List[str],
    hop1_mesh_list: List[str],
) -> Dict[str, List[str]]:
    """
    Add relevant terms from the hop1 MeSH set onto the two classified seed sets.
    Takes PICO, classified seed population/intervention MeSH, and hop1 MeSH list.
    Returns {"population": [...], "intervention": [...]} with seed terms plus
    relevant hop1 terms added to the appropriate list.
    """
    if not hop1_mesh_list:
        return {"population": list(seed_population), "intervention": list(seed_intervention)}
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    pico_block = "\n".join(
        f"  {k}: {v}" for k, v in pico.items() if isinstance(v, str) and v.strip()
    )
    seed_pop_block = ", ".join(seed_population) if seed_population else "(none)"
    seed_int_block = ", ".join(seed_intervention) if seed_intervention else "(none)"
    hop1_block = "\n".join(f"- {t}" for t in sorted(hop1_mesh_list)[:500])
    prompt = (
        "You are a systematic review librarian. We have:\n\n"
        "1. PICO (the review's question)\n"
        "2. Classified seed MeSH — population and intervention (from confirmed relevant papers)\n"
        "3. A large set of MeSH terms from related papers (hop-1).\n\n"
        "Add relevant terms from the hop-1 set onto the seed population and seed intervention lists. "
        "Only add terms that clearly belong to population/disease/patient group or to intervention/symptoms/diagnosis. "
        "Do not add demographic noise (Male, Female, Humans, Adult, etc.), study design, or methods. "
        "Use the exact MeSH string from the hop-1 list.\n\n"
        "PICO:\n"
        f"{pico_block}\n\n"
        "Seed population MeSH (keep all and add relevant from hop-1):\n"
        f"{seed_pop_block}\n\n"
        "Seed intervention MeSH (keep all and add relevant from hop-1):\n"
        f"{seed_int_block}\n\n"
        "Hop-1 MeSH set (choose relevant terms to add to population or intervention):\n"
        f"{hop1_block}\n\n"
        "Return JSON only with two keys: population, intervention.\n"
        "Each value is an array of MeSH strings: first the original seed terms in order, then any added terms from hop-1.\n"
        'Example: {"population": ["Colorectal Neoplasms", "Young Adult", "Adenocarcinoma"], "intervention": ["Rectal Bleeding", "Delayed Diagnosis", "Time-to-Treatment"]}'
    )
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text:
        raw_text = str(response)
    raw_text = raw_text.strip()
    # Strip markdown code fence if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines)
    hop1_set = {str(t).strip() for t in hop1_mesh_list if t}
    seed_pop_set = set(seed_population)
    seed_int_set = set(seed_intervention)
    if not raw_text:
        return {"population": list(seed_population), "intervention": list(seed_intervention)}
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Model returned malformed JSON (e.g. unterminated string); skip augmentation
        return {"population": list(seed_population), "intervention": list(seed_intervention)}
    out_pop = list(seed_population)
    out_int = list(seed_intervention)
    for key, seed_set, out_list in (
        ("population", seed_pop_set, out_pop),
        ("intervention", seed_int_set, out_int),
    ):
        arr = parsed.get(key)
        if isinstance(arr, list):
            for x in arr:
                s = str(x).strip()
                if not s:
                    continue
                if s in seed_set or s in hop1_set:
                    if s not in out_list:
                        out_list.append(s)
    return {"population": out_pop, "intervention": out_int}


def extract_terms_from_abstract(abstract: str, pico: Dict[str, Any]) -> List[str]:
    """
    One Gemini call per abstract: extract 5-8 relevant search terms from the abstract text.
    PICO is provided as context. Rules: single words or 2-word phrases only; no sentences;
    no generic words (e.g. patients, study); no statistical terms; only what is explicitly in the text.
    """
    abstract = (abstract or "").strip()
    if not abstract:
        return []
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    pico_block = "\n".join(
        f"  {k}: {v}" for k, v in pico.items() if isinstance(v, str) and v.strip()
    )
    # Truncate very long abstracts to stay within model limits
    text = abstract[:6000] if len(abstract) > 6000 else abstract
    prompt = (
        "You are a systematic review librarian. Extract search terms from the following abstract "
        "that would help find similar studies in PubMed.\n\n"
        "Use the PICO only as context for relevance.\n\n"
        "Rules:\n"
        "- Only extract disease names, condition synonyms, drug names, treatment names, and therapy types.\n"
        "- Do NOT extract outcome measures (e.g. overall survival, pathological response, adverse events, mortality).\n"
        "- Do NOT extract study design terms (e.g. RCT, clinical trials, meta-analysis, cohort).\n"
        "- Do NOT extract generic modifiers (e.g. in combination, monotherapy, resectable).\n"
        "- Do NOT extract side effects or safety terms (e.g. rash, irAEs, immunotherapy-related rash).\n"
        "- Do NOT extract eligibility criteria or patient descriptors (e.g. patients 18 or above, histologically confirmed, cytologically confirmed) UNLESS the SR population specifically targets a demographic group (e.g. 'infants', 'pediatric', 'elderly'). If the PICO population defines itself by age or demographic, include that term.\n"
        "- Do NOT extract staging terms (e.g. Stage I, Stage II, Stage III, stage IIIA-N2).\n"
        "- Preserve hyphens exactly as written in the text (e.g. anti-PD-1, PD-L1, CTLA-4).\n"
        "- Single words or short phrases only. No sentences.\n"
        "- Maximum 5-8 terms.\n"
        "- Only extract what is explicitly stated in the text.\n\n"
        "PICO (context):\n"
        f"{pico_block}\n\n"
        "Abstract:\n"
        f"{text}\n\n"
        "Return JSON only: {\"terms\": [\"term1\", \"term2\", ...]}"
    )
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text:
        raw_text = str(response)
    raw_text = raw_text.strip()
    if not raw_text:
        return []
    parsed = json.loads(raw_text)
    terms = parsed.get("terms")
    if not isinstance(terms, list):
        return []
    return [str(t).strip() for t in terms if t and str(t).strip()]


def extract_freetext_terms_from_titles(
    titles: List[str],
    pico: Dict[str, Any],
) -> List[str]:
    """
    Given hop-0 titles and hop-1 (≥2 connection) titles, with PICO as context,
    ask Gemini to extract useful freetext search terms from these titles.
    Returns a list of terms (phrases or single words) for use in a PubMed query.
    """
    if not titles:
        return []
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    pico_block = "\n".join(
        f"  {k}: {v}" for k, v in pico.items() if isinstance(v, str) and v.strip()
    )
    titles_block = "\n".join(f"- {t}" for t in titles[:200])
    prompt = (
        "You are a systematic review librarian building a PubMed search.\n\n"
        "PICO (for context):\n"
        f"{pico_block}\n\n"
        "Below are paper titles from seed papers (hop-0) and from related papers (hop-1 with at least 2 citation links to seeds). "
        "Extract useful freetext search terms from these titles that would help find similar studies. "
        "Include: disease names, symptom phrases, population descriptors, outcome terms, and key concepts. "
        "Use short phrases (2–4 words) or single terms. No MeSH—only free text. "
        "Deduplicate and return a single list of terms.\n\n"
        "Titles:\n"
        f"{titles_block}\n\n"
        "Return JSON only: a single object with one key \"terms\" whose value is an array of strings.\n"
        'Example: {"terms": ["early onset colorectal cancer", "diagnostic delay", "rectal bleeding", "time to diagnosis"]}'
    )
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text:
        raw_text = str(response)
    raw_text = raw_text.strip()
    if not raw_text:
        return []
    parsed = json.loads(raw_text)
    terms = parsed.get("terms")
    if not isinstance(terms, list):
        return []
    return [str(t).strip() for t in terms if t and str(t).strip()]


def split_freetext_terms_by_pico(
    free_terms: List[str],
    pico: Dict[str, Any],
) -> Dict[str, List[str]]:
    """
    Given a set of freetext terms (from titles + abstracts), split them into
    population vs intervention using PICO as context. Returns {"population": [], "intervention": []}.
    Each term is assigned to at most one list; irrelevant terms can be omitted.
    """
    if not free_terms:
        return {"population": [], "intervention": []}
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)
    pico_block = "\n".join(
        f"  {k}: {v}" for k, v in pico.items() if isinstance(v, str) and v.strip()
    )
    terms_block = "\n".join(f"- {t}" for t in sorted(free_terms))
    prompt = (
        "You are a systematic review librarian. We have a set of freetext search terms "
        "(from paper titles and abstracts). Split them into two lists using the PICO as context:\n\n"
        "1. population — terms that describe the disease, condition, patient group, or age/setting.\n"
        "2. intervention — terms that describe symptoms, signs, diagnostic process, time to diagnosis, delays, or exposure.\n\n"
        "Put each term in exactly one list. Use the exact string from the list. "
        "If a term does not clearly fit either, omit it.\n\n"
        "PICO (context):\n"
        f"{pico_block}\n\n"
        "Freetext terms:\n"
        f"{terms_block}\n\n"
        "Return JSON only: {\"population\": [\"term1\", ...], \"intervention\": [\"term1\", ...]}"
    )
    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    raw_text = getattr(response, "text", None) or ""
    if not raw_text:
        raw_text = str(response)
    raw_text = raw_text.strip()
    terms_set = {str(t).strip() for t in free_terms if t}
    out = {"population": [], "intervention": []}
    if not raw_text:
        return out
    parsed = json.loads(raw_text)
    for key in ("population", "intervention"):
        arr = parsed.get(key)
        if isinstance(arr, list):
            out[key] = [str(x).strip() for x in arr if str(x).strip() in terms_set]
    return out


def get_pico_keywords(pico: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Ask Gemini for exactly three keywords per PICO facet.
    Returns e.g. {"population": [...], "intervention": [...], "comparator": [...], "outcome": [...]}.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    pop = (pico.get("population") or "").strip()
    interv = (pico.get("intervention") or "").strip()
    comp = (pico.get("comparator") or "").strip()
    outc = (pico.get("outcome") or "").strip()

    prompt = (
        "You are helping build a concise search strategy for a systematic review.\n\n"
        "PICO:\n"
        f"Population: {pop}\n"
        f"Intervention: {interv}\n"
        f"Comparator: {comp}\n"
        f"Outcome: {outc}\n\n"
        "For each PICO facet, give exactly three keywords or short phrases that best capture it. "
        "Use the most specific, searchable terms. If a facet is empty or not applicable, "
        "return an empty array for it.\n\n"
        "Return a JSON object with exactly four keys: population, intervention, comparator, outcome.\n"
        "Each value must be an array of exactly three strings (or empty if not applicable).\n"
        'Example: {"population":["heart transplant recipients","end-stage heart failure","cardiac transplantation"],'
        '"intervention":["risk prediction model","prognostic score","survival prediction"],'
        '"comparator":[],'
        '"outcome":["post-transplant mortality","graft survival","1-year survival"]}\n'
        "Return only valid JSON, no markdown backticks."
    )

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text)

    result: Dict[str, List[str]] = {}
    for facet in _PICO_FACETS:
        arr = parsed.get(facet)
        if isinstance(arr, list) and len(arr) >= 3:
            result[facet] = [str(arr[0]).strip(), str(arr[1]).strip(), str(arr[2]).strip()]
        elif isinstance(arr, list) and len(arr) == 2:
            result[facet] = [str(arr[0]).strip(), str(arr[1]).strip(), str(arr[1]).strip()]
        elif isinstance(arr, list) and len(arr) == 1:
            v = str(arr[0]).strip()
            result[facet] = [v, v, v]
        else:
            result[facet] = []
    return result


def extract_terms(
    pico: Dict[str, Any],
    references: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract search terms (MeSH + freetext) for each PICO facet from reference
    abstracts and MeSH terms, using the PICO as a guide for relevance.

    Returns a dict with study_design + one entry per PICO facet; each value is
    {"mesh": [...], "freetext": [...]}.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    refs_text = _refs_text_for_prompt(references, max_chars=80_000)
    pop = pico.get("population") or ""
    interv = pico.get("intervention") or ""
    comp = pico.get("comparator") or ""
    outc = pico.get("outcome") or ""

    prompt = (
        "You are helping build a search strategy for a systematic review.\n\n"
        "PICO from the review:\n"
        f"Population: {pop}\nIntervention: {interv}\n"
        f"Comparator: {comp}\nOutcome: {outc}\n\n"
        "Below are reference titles, abstracts, and MeSH terms from included or key papers.\n"
        "Extract search terms relevant to each PICO facet. For each facet provide:\n"
        "- mesh: MeSH terms that appear in the references' MeSH lists for that facet. "
        "Include BOTH specific MeSH headings AND broader commonly-assigned ones (e.g. \"Prognosis\", \"Risk Factors\").\n"
        "- freetext: natural language phrases. IMPORTANT: include both general category terms "
        "(e.g. \"risk score\", \"prognostic model\", \"prediction model\") AND specific named examples "
        "(e.g. \"MELD score\", \"CARRS score\"). General terms are critical for recall.\n\n"
        "Also provide study_design: one of randomized_controlled_trial, observational, systematic_review, or any.\n\n"
        "References (titles, abstracts, MeSH):\n"
        f"{refs_text}\n\n"
        "Return a JSON object with exactly five keys: study_design, population, intervention, comparator, outcome.\n"
        "study_design must be one of: randomized_controlled_trial, observational, systematic_review, any.\n"
        "Each PICO facet must be an object with two keys: \"mesh\" (array of strings) and \"freetext\" (array of strings).\n"
        "If a facet is not applicable, use empty arrays.\n"
        "Return only valid JSON, no markdown backticks."
    )

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text)

    valid_designs = ("randomized_controlled_trial", "observational", "systematic_review", "any")
    study_design = str(parsed.get("study_design") or "any").strip().lower()
    if study_design not in valid_designs:
        study_design = "any"
    result: Dict[str, Any] = {"study_design": study_design}
    for facet in _PICO_FACETS:
        obj = parsed.get(facet)
        if not isinstance(obj, dict):
            result[facet] = {"mesh": [], "freetext": []}
            continue
        mesh = obj.get("mesh")
        freetext = obj.get("freetext")
        result[facet] = {
            "mesh": [str(x) for x in mesh] if isinstance(mesh, list) else [],
            "freetext": [str(x) for x in freetext] if isinstance(freetext, list) else [],
        }
    return result


def filter_terms_by_key_concepts(
    terms: Dict[str, Any],
    key_concepts: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Filter each PICO facet's terms to only those that relate to the
    three key concepts for that facet. Keeps study_design. Produces a more concise list.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    terms_json = json.dumps(terms, indent=2)
    concept_lines = []
    for facet in _PICO_FACETS:
        kw = key_concepts.get(facet) or []
        concepts_str = ", ".join(kw) if kw else "(none / not applicable)"
        concept_lines.append(
            f"Key concepts for {facet} (keep only terms that clearly relate to these): {concepts_str}"
        )

    prompt = (
        "You are filtering search terms to match key concepts only.\n\n"
        + "\n".join(concept_lines) + "\n\n"
        "Current terms:\n"
        f"{terms_json}\n\n"
        "Return a JSON object with the same top-level keys "
        "(study_design, population, intervention, comparator, outcome).\n"
        "Preserve study_design exactly. For each PICO facet, keep only mesh and freetext terms "
        "that are related to the key concepts for that facet. Remove terms that are completely unrelated.\n"
        "IMPORTANT: for each facet, keep BOTH general category terms (e.g. \"risk score\", \"prognosis\") "
        "AND specific named terms. Do NOT over-filter — it is better to keep a somewhat broader term "
        "than to miss relevant papers. If a facet has no key concepts, keep it empty.\n"
        "Each facet must be {\"mesh\": [...], \"freetext\": [...]}.\n"
        "Return only valid JSON, no markdown backticks."
    )

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text)

    result = dict(terms)
    result["study_design"] = terms.get("study_design", "any")
    for facet in _PICO_FACETS:
        obj = parsed.get(facet)
        if isinstance(obj, dict):
            mesh = obj.get("mesh")
            freetext = obj.get("freetext")
            result[facet] = {
                "mesh": [str(x) for x in mesh] if isinstance(mesh, list) else [],
                "freetext": [str(x) for x in freetext] if isinstance(freetext, list) else [],
            }
        else:
            result[facet] = terms.get(facet, {"mesh": [], "freetext": []})
    return result


def extract_titles_from_references(raw_refs: List[str]) -> List[str]:
    """
    Given a list of raw reference strings (no DOI/PMID found), ask Gemini to
    extract a clean, PubMed-searchable article title from each one.

    Returns a list of titles in the same order as raw_refs.
    Empty string for any reference where a title cannot be determined.
    """
    if not raw_refs:
        return []

    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    numbered = "\n".join(f"[{i}] {r.strip()}" for i, r in enumerate(raw_refs))

    prompt = (
        "You are a biomedical reference parser.\n\n"
        "Below is a numbered list of raw reference strings extracted from a PDF.\n"
        "For each reference, extract ONLY the article title — not the authors, "
        "journal name, volume, pages, or year.\n\n"
        "The title should be clean, complete, and suitable for searching PubMed "
        '(e.g. searchable via "title"[Title] in PubMed).\n\n'
        "Return a JSON array of strings, one per reference, in the same order.\n"
        "If you cannot determine the title for a reference, use an empty string.\n\n"
        f"References:\n{numbered}\n\n"
        "Return only valid JSON (an array of strings), no markdown backticks."
    )

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text)

    if not isinstance(parsed, list):
        return [""] * len(raw_refs)

    # Pad or truncate to match input length
    titles = [str(t).strip() if t else "" for t in parsed]
    while len(titles) < len(raw_refs):
        titles.append("")
    return titles[: len(raw_refs)]


def _refs_text_for_prompt(references: List[Dict[str, Any]], max_chars: int = 40_000) -> str:
    """Build concatenated title/abstract/MeSH text for reference list (for prompts)."""
    ref_parts: List[str] = []
    total = 0
    for i, ref in enumerate(references):
        title = (ref.get("title") or "").strip()
        abstract = (ref.get("abstract") or "").strip()
        mesh = ref.get("mesh_terms")
        mesh_str = "; ".join(str(m) for m in mesh if m) if isinstance(mesh, list) else str(mesh or "")
        lines = [f"[Ref {i + 1}]", f"Title: {title}"]
        if abstract:
            lines.append(f"Abstract: {abstract}")
        if mesh_str:
            lines.append(f"MeSH: {mesh_str}")
        block = "\n".join(lines) + "\n"
        if total + len(block) > max_chars:
            ref_parts.append(block[: max_chars - total] + "\n... (truncated)")
            break
        ref_parts.append(block)
        total += len(block)
    return "\n".join(ref_parts) if ref_parts else "(No references provided.)"


def extract_freetext_terms(
    pico: Dict[str, Any],
    references: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Dedicated Gemini call for recall-focused freetext only. Asks for every
    phrase and synonym authors use in titles/abstracts for each PICO facet.
    Returns {facet: [freetext terms]} (no MeSH). Merge with existing terms
    and de-dupe before building the query.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    refs_text = _refs_text_for_prompt(references, max_chars=80_000)
    pop = (pico.get("population") or "").strip()
    interv = (pico.get("intervention") or "").strip()
    comp = (pico.get("comparator") or "").strip()
    outc = (pico.get("outcome") or "").strip()

    prompt = (
        "You are building a high-recall search strategy for a systematic review.\n\n"
        "PICO from the review:\n"
        f"Population: {pop}\nIntervention: {interv}\n"
        f"Comparator: {comp}\nOutcome: {outc}\n\n"
        "Below are reference titles and abstracts from included or key papers.\n"
        "Your task: list every distinct phrase, synonym, and variant that authors use "
        "in these texts to describe each PICO facet. Prioritize RECALL — include:\n"
        "- Multiple ways of saying the same concept (e.g. \"risk score\", \"prognostic score\", \"prediction model\")\n"
        "- Abbreviations and acronyms (prefer as part of a phrase, e.g. \"MELD score\", but include standalone if essential)\n"
        "- Slight wording variants (e.g. \"heart transplant recipients\", \"patients undergoing heart transplantation\")\n"
        "Prefer phrases of 2 or more words where possible. Include both general category terms and specific named tools.\n\n"
        "Return a JSON object with exactly four keys: population, intervention, comparator, outcome.\n"
        "Each value must be an array of strings (freetext terms only; no MeSH).\n"
        "If a facet is not applicable, use an empty array.\n\n"
        "References (titles and abstracts):\n"
        f"{refs_text}\n\n"
        "Return only valid JSON, no markdown backticks."
    )

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text)

    result: Dict[str, List[str]] = {}
    for facet in _PICO_FACETS:
        arr = parsed.get(facet)
        if isinstance(arr, list):
            result[facet] = [str(x).strip() for x in arr if x and str(x).strip()]
        else:
            result[facet] = []
    return result


def expand_terms_variants(
    terms: Dict[str, Any],
    pico: Dict[str, Any],
    key_concepts: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Topic-anchored expansion: given PICO, key concepts, and current terms,
    return additional freetext variants that stay within this review's scope.
    Only adds terms that are direct synonyms/rephrasings of the same concepts.
    Returns {facet: [additional freetext terms]} to merge into terms.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    facets_json: Dict[str, List[str]] = {}
    for facet in _PICO_FACETS:
        data = terms.get(facet) or {}
        freetext = data.get("freetext") if isinstance(data, dict) else []
        facets_json[facet] = list(freetext) if isinstance(freetext, list) else []

    pico_block = "\n".join(
        f"  {k}: {pico.get(k) or ''}" for k in _PICO_FACETS
    )
    concepts_block = "\n".join(
        f"  {k}: {', '.join(key_concepts.get(k) or [])}"
        for k in _PICO_FACETS
    )

    prompt = (
        "You are improving recall for a systematic review search strategy.\n\n"
        "This review's PICO (scope):\n"
        f"{pico_block}\n\n"
        "Key concepts per facet (anchor — stay close to these):\n"
        f"{concepts_block}\n\n"
        "Current freetext terms per facet:\n"
        f"{json.dumps(facets_json, indent=2)}\n\n"
        "Consider these dimensions when adding variants:\n"
        "- Setting (e.g. primary care, emergency department, inpatient, community, in-hospital, out-of-hospital)\n"
        "- Tool/method type (e.g. screening instrument, needs assessment, checklist, algorithm, natural language processing)\n"
        "- Study framing (e.g. diagnostic accuracy, validation, content validity)\n"
        "- Population wording (e.g. young families, families in primary care, age ranges, in-hospital vs out-of-hospital)\n\n"
        "Your task: for each facet, decide which of these dimensions are clearly relevant to this review's topic. "
        "Add ADDITIONAL terms only for dimensions that apply — direct synonyms, alternate phrasings, or abbreviations for the SAME concepts. "
        "If a dimension is not relevant to this review, add nothing for it. Every added term must stay within the review's scope. "
        "Do NOT add broader or generic terms. Do NOT repeat terms already given.\n"
        "Add at most 10–15 additional terms per facet. Prefer phrases of 2+ words. Use empty array if no on-topic additions.\n\n"
        "Return a JSON object with exactly four keys: population, intervention, comparator, outcome.\n"
        "Each value must be an array of strings (additional freetext terms only).\n"
        "Return only valid JSON, no markdown backticks."
    )

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text)

    result: Dict[str, List[str]] = {}
    for facet in _PICO_FACETS:
        arr = parsed.get(facet)
        if isinstance(arr, list):
            result[facet] = [str(x).strip() for x in arr if x and str(x).strip()]
        else:
            result[facet] = []
    return result


def filter_extracted_terms(
    terms: Dict[str, Dict[str, List[str]]],
    references: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Filter extracted terms using rules per PICO facet; prefer specific over broad;
    6-10 terms max per list; prefer terms that appear often in the abstracts.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    refs_text = _refs_text_for_prompt(references, max_chars=40_000)
    terms_json = json.dumps(terms, indent=2)

    rules = (
        "Rules for term selection:\n"
        "- Population: include only terms that define who the patients are "
        "(diagnoses, conditions, procedures that determine eligibility). Exclude demographic terms, "
        "generic terms like \"adult\" or \"patient\", and terms that describe treatments or interventions.\n"
        "- Intervention: include terms that define the core mechanism or modality "
        "of the intervention — what makes it distinct. IMPORTANT: always include BOTH general category terms "
        "(e.g. \"risk score\", \"prognostic model\", \"prediction model\") AND specific named examples "
        "(e.g. \"MELD score\", \"APACHE II\"). General terms are essential for recall — "
        "specific named tools alone will miss many relevant studies that use different tools for the same purpose.\n"
        "- Comparator: include only terms that define what the intervention is compared against "
        "(e.g. placebo, standard care, alternative treatment). If none, leave empty.\n"
        "- Outcome: include only terms that define the measured outcomes or endpoints "
        "(e.g. mortality, survival, recurrence, quality of life). Exclude generic method terms.\n"
        "- Keep each list to 8-12 terms. Include a mix of general and specific terms.\n"
        "- Do not include short abbreviations as standalone freetext — they are too ambiguous. "
        "Only include them if combined with other words.\n"
        "- For MeSH: prefer specific MeSH headings but also keep broader ones that are commonly assigned "
        "to relevant papers (e.g. \"Prognosis\", \"Risk Factors\"). Do NOT drop a MeSH term just because "
        "it is broad — if it appears frequently in the references' MeSH lists, keep it.\n"
    )

    prompt = (
        "You are filtering search terms for a systematic review.\n\n"
        f"{rules}\n"
        "Current extracted terms (to filter):\n"
        f"{terms_json}\n\n"
        "Reference titles, abstracts, and MeSH (use to prefer frequently appearing terms):\n"
        f"{refs_text}\n\n"
        "Return a JSON object with exactly four keys: population, intervention, comparator, outcome.\n"
        "Each value must be an object with two keys: \"mesh\" (array of strings) and \"freetext\" (array of strings).\n"
        "Apply the rules above and keep 6-10 terms per list. Return only valid JSON, no markdown backticks."
    )

    response = client.models.generate_content(
        model=_MODEL_NAME,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    raw_text = getattr(response, "text", None) or ""
    if not raw_text.strip():
        raw_text = str(response)
    parsed = json.loads(raw_text)

    result: Dict[str, Any] = {"study_design": terms.get("study_design", "any")}
    for facet in _PICO_FACETS:
        obj = parsed.get(facet)
        if not isinstance(obj, dict):
            result[facet] = {"mesh": [], "freetext": []}
            continue
        mesh = obj.get("mesh")
        freetext = obj.get("freetext")
        result[facet] = {
            "mesh": [str(x) for x in mesh] if isinstance(mesh, list) else [],
            "freetext": [str(x) for x in freetext] if isinstance(freetext, list) else [],
        }
    return result

