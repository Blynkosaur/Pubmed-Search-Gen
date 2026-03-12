from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Union

import fitz  # PyMuPDF
import google.generativeai as genai


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
    genai.configure(api_key=api_key)

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

    model = genai.GenerativeModel(_MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"},
    )

    raw_text = getattr(response, "text", None)
    if raw_text is None:
        # Fallback: try to coerce to string/JSON
        parsed = json.loads(str(response))
    else:
        parsed = json.loads(raw_text)

    return parsed

