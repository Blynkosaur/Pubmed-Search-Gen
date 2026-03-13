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
        "own PICO (Population, Intervention, Comparator, Outcome).\n\n"
        "Return a JSON object with exactly these keys:\n"
        '{\n'
        '  "population": string,\n'
        '  "intervention": string,\n'
        '  "comparator": string,\n'
        '  "outcome": string\n'
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

    - pico: dict with keys population, intervention, comparator, outcome.
    - references: list of dicts, each with "abstract" and "mesh_terms" (list of strings).

    Returns a dict with one entry per PICO facet; each value is
    {"mesh": [...], "freetext": [...]}.
    """
    api_key = _load_api_key()
    client = genai.Client(api_key=api_key)

    # Build reference text: abstract + MeSH for each ref (truncate if huge).
    ref_parts: List[str] = []
    max_ref_chars = 80_000  # leave room for prompt + PICO + response
    total = 0
    for i, ref in enumerate(references):
        abstract = (ref.get("abstract") or "").strip()
        mesh = ref.get("mesh_terms")
        if isinstance(mesh, list):
            mesh_str = "; ".join(str(m) for m in mesh if m)
        else:
            mesh_str = str(mesh or "")
        block = f"[Ref {i + 1}]\nAbstract: {abstract}\nMeSH: {mesh_str}\n"
        if total + len(block) > max_ref_chars:
            block = block[: max_ref_chars - total] + "\n... (truncated)"
            ref_parts.append(block)
            break
        ref_parts.append(block)
        total += len(block)

    refs_text = "\n".join(ref_parts) if ref_parts else "(No references provided.)"
    pico_text = json.dumps(pico, indent=2)

    prompt = (
        "You are helping build a search strategy for a systematic review.\n\n"
        "PICO (Population, Intervention, Comparator, Outcome) from the review:\n"
        f"{pico_text}\n\n"
        "Below are reference abstracts and their MeSH terms from included or key papers.\n"
        "Extract search terms that are relevant to each PICO facet. Use the PICO descriptions "
        "to decide what is relevant. For each facet, provide:\n"
        "- mesh: MeSH terms (or similar controlled terms) that match the facet.\n"
        "- freetext: natural language / keyword phrases (synonyms, abbreviations, drug names, etc.).\n\n"
        "Reference abstracts and MeSH:\n"
        f"{refs_text}\n\n"
        "Return a JSON object with exactly four keys: population, intervention, comparator, outcome.\n"
        "Each value must be an object with two keys: \"mesh\" (array of strings) and \"freetext\" (array of strings).\n"
        "Example:\n"
        '{"population":{"mesh":["Renal Insufficiency, Chronic"],"freetext":["chronic kidney disease","CKD"]},'
        '"intervention":{"mesh":["Sodium-Glucose Transporter 2 Inhibitors"],"freetext":["SGLT-2 inhibitors","dapagliflozin"]},'
        '"comparator":{"mesh":[],"freetext":[]},'
        '"outcome":{"mesh":[],"freetext":[]}}\n'
        "Return only valid JSON, no markdown."
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

    # Normalize to the expected shape (mesh + freetext lists per facet).
    facets = ("population", "intervention", "comparator", "outcome")
    result: Dict[str, Dict[str, List[str]]] = {}
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

