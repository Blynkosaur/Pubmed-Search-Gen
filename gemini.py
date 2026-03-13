from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import fitz  # PyMuPDF
from google import genai


_MODEL_NAME = "gemini-2.5-flash"


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
        "own PICO: Population and Intervention only.\n\n"
        "Return a JSON object with exactly these keys:\n"
        '{\n'
        '  "population": string,\n'
        '  "intervention": string\n'
        "}\n\n"
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


def extract_terms(
    pico: Dict[str, Any],
    references: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract search terms (MeSH + freetext) for each PICO facet from reference
    abstracts and MeSH terms, using the PICO as a guide for relevance.

    - pico: dict with keys population, intervention.
    - references: list of dicts, each with "abstract" and "mesh_terms" (list of strings).

    Returns a dict with one entry per PICO facet; each value is
    {"mesh": [...], "freetext": [...]}.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    # Build reference text: title + abstract + MeSH for each ref (truncate if huge).
    ref_parts: List[str] = []
    max_ref_chars = 80_000  # leave room for prompt + PICO + response
    total = 0
    for i, ref in enumerate(references):
        title = (ref.get("title") or "").strip()
        abstract = (ref.get("abstract") or "").strip()
        mesh = ref.get("mesh_terms")
        if isinstance(mesh, list):
            mesh_str = "; ".join(str(m) for m in mesh if m)
        else:
            mesh_str = str(mesh or "")
        lines = [f"[Ref {i + 1}]", f"Title: {title}"]
        if abstract:
            lines.append(f"Abstract: {abstract}")
        if mesh_str:
            lines.append(f"MeSH: {mesh_str}")
        block = "\n".join(lines) + "\n"
        if total + len(block) > max_ref_chars:
            block = block[: max_ref_chars - total] + "\n... (truncated)"
            ref_parts.append(block)
            break
        ref_parts.append(block)
        total += len(block)

    refs_text = "\n".join(ref_parts) if ref_parts else "(No references provided.)"
    pop = pico.get("population") or ""
    interv = pico.get("intervention") or ""

    prompt = (
        "You are helping build a search strategy for a systematic review.\n\n"
        "PICO from the review:\n"
        f"Population: {pop}\nIntervention: {interv}\n\n"
        "Below are reference titles, abstracts, and MeSH terms from included or key papers.\n"
        "Extract search terms relevant to each PICO facet. For each facet provide:\n"
        "- mesh: MeSH terms (or similar controlled terms) that match the facet.\n"
        "- freetext: natural language / keyword phrases (synonyms, abbreviations, drug names, etc.).\n\n"
        "Also provide study_design: one of randomized_controlled_trial, observational, systematic_review, or any.\n\n"
        "References (titles, abstracts, MeSH):\n"
        f"{refs_text}\n\n"
        "Return a JSON object with exactly three keys: study_design, population, intervention.\n"
        "study_design must be one of: randomized_controlled_trial, observational, systematic_review, any.\n"
        "population and intervention must each be an object with two keys: \"mesh\" (array of strings) and \"freetext\" (array of strings).\n"
        "Example: {\"study_design\":\"randomized_controlled_trial\",\"population\":{\"mesh\":[\"Renal Insufficiency, Chronic\"],\"freetext\":[\"chronic kidney disease\",\"CKD\"]},"
        "\"intervention\":{\"mesh\":[\"Sodium-Glucose Transporter 2 Inhibitors\"],\"freetext\":[\"SGLT-2 inhibitors\",\"dapagliflozin\"]}}\n"
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

    # Normalize to the expected shape (study_design + mesh + freetext lists per facet).
    valid_designs = ("randomized_controlled_trial", "observational", "systematic_review", "any")
    study_design = str(parsed.get("study_design") or "any").strip().lower()
    if study_design not in valid_designs:
        study_design = "any"
    result: Dict[str, Any] = {"study_design": study_design}
    facets = ("population", "intervention")
    for facet in facets:
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


def filter_extracted_terms(
    terms: Dict[str, Dict[str, List[str]]],
    references: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[str]]]:
    """
    Filter extracted terms using rules: population = who patients are only;
    intervention = core mechanism/modality only; prefer specific over broad;
    6-10 terms max per list; prefer terms that appear often in the abstracts.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    refs_text = _refs_text_for_prompt(references, max_chars=40_000)
    terms_json = json.dumps(terms, indent=2)

    rules = (
        "Rules for term selection:\n"
        "- Population mesh and freetext: include only terms that define who the patients are "
        "(diagnoses, conditions, procedures that determine eligibility). Exclude demographic terms, "
        "generic terms like \"adult\" or \"patient\", and terms that describe treatments or interventions.\n"
        "- Intervention mesh and freetext: include only terms that define the core mechanism or modality "
        "of the intervention — what makes it distinct. Exclude terms describing the content, goals, or outcomes of the intervention.\n"
        "- Prioritize specific terms over broad ones. A term that appears in 10,000 PubMed records is too broad "
        "unless it's the only way to describe the concept.\n"
        "- Keep each list to 6-10 terms maximum. Prefer terms that appear frequently in the provided abstracts.\n"
        "- Do not include abbreviations like CR, MI, ACS as standalone freetext terms — they are too ambiguous. "
        "Only include them if combined with other words.\n"
        "- Exclude terms that are a broader category than necessary. A term is too broad if it would match large numbers "
        "of papers outside the scope of this specific review. For example, a single common word or a parent category "
        "that encompasses many unrelated conditions or interventions. When in doubt, prefer the more specific term over the general one.\n"
    )

    prompt = (
        "You are filtering search terms for a systematic review.\n\n"
        f"{rules}\n"
        "Current extracted terms (to filter):\n"
        f"{terms_json}\n\n"
        "Reference titles, abstracts, and MeSH (use to prefer frequently appearing terms):\n"
        f"{refs_text}\n\n"
        "Return a JSON object with exactly two keys: population, intervention.\n"
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

    # Preserve study_design from input; filter only population and intervention.
    result: Dict[str, Any] = {"study_design": terms.get("study_design", "any")}
    facets = ("population", "intervention")
    for facet in facets:
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

