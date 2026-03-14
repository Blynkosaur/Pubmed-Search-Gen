from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import fitz  # PyMuPDF
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

    doc = fitz.open(pdf_path)
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
        "own PICO: Population, Intervention, Comparator, and Outcome.\n\n"
        "Return a JSON object with exactly these keys:\n"
        '{\n'
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

