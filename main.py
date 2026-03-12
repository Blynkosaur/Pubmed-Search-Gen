from __future__ import annotations

import argparse
from pathlib import Path

from pubmed import parse, Reference
from gemini import pico_extractor


def run_refs(pdf_path: Path) -> None:
    refs = parse(pdf_path)
    print(f"Found {len(refs)} references in {pdf_path}")
    print()
    for ref in refs:
        line = f"[{ref.index}] {ref.raw}"
        if ref.doi:
            line += f" | DOI: {ref.doi}"
        if ref.year:
            line += f" | Year: {ref.year}"
        print(line)


def run_pico(pdf_path: Path) -> None:
    pico = pico_extractor(pdf_path)
    print("PICO extracted from manuscript:")
    for key in ["population", "intervention", "comparator", "outcome"]:
        value = pico.get(key)
        print(f"- {key.capitalize()}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract references and/or PICO from a systematic review PDF.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to the manuscript PDF to parse for references.",
    )
    parser.add_argument(
        "--pico",
        action="store_true",
        help="Also extract the SR's own PICO using Gemini.",
    )
    args = parser.parse_args()
    if args.pico:
        run_pico(args.pdf)
    else:
        run_refs(args.pdf)


if __name__ == "__main__":
    main()

