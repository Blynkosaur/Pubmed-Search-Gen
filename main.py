from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gemini import pico_extractor


def _ref_cache_path(pdf_path: Path) -> Path:
    return pdf_path.parent / f"{pdf_path.stem}_references.json"


def _doc_for_ref(rec: dict) -> str:
    """Use abstract if present, else mesh_terms as text, else title."""
    abstract = (rec.get("abstract") or "").strip()
    if abstract:
        return abstract
    mesh = rec.get("mesh_terms") or []
    if mesh:
        return " ".join(str(m) for m in mesh if m)
    return (rec.get("title") or "").strip()


def run(pdf_path: Path) -> None:
    pdf_path = Path(pdf_path)
    cache_path = _ref_cache_path(pdf_path)
    if not cache_path.exists():
        print(f"No references cache at {cache_path}. Run the pipeline once to build it.")
        return
    with cache_path.open("r", encoding="utf-8") as f:
        references = json.load(f)
    print(f"Loaded {len(references)} references from {cache_path}\n")

    pico = pico_extractor(pdf_path)
    pico_text = " ".join(str(pico.get(k, "")) for k in ("population", "intervention"))
    ref_docs = [_doc_for_ref(r) for r in references]
    docs = [pico_text] + ref_docs
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    THRESHOLD = 0.05
    kept = [(rec, s) for rec, s in zip(references, sims) if s >= THRESHOLD]

    print("PICO:")
    for key in ("population", "intervention"):
        print(f"  {key}: {pico.get(key, '')}")
    print(f"\nReferences with TF-IDF score >= {THRESHOLD}: {len(kept)} of {len(references)}\n")
    for rec, score in kept:
        title = (rec.get("title") or "").strip()
        print(f"{score:.4f}  {title}")



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
    args = parser.parse_args()
    run(args.pdf)


if __name__ == "__main__":
    main()

