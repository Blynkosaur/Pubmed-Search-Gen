from __future__ import annotations

from typing import Any, Dict, List, Set

_PICO_FACETS = ("population", "intervention", "comparator", "outcome")


def _truncate(term: str) -> str:
    """Append PubMed truncation wildcard to the last word if not already present."""
    t = term.strip()
    if not t or t.endswith("*"):
        return t
    return t + "*"


def _hyphen_variants(term: str) -> List[str]:
    """Generate spelling variants for hyphenated terms.

    For a term like "co-transporter", produces both the original and the
    de-hyphenated form ("cotransporter").  Returns a list with at least the
    original term.
    """
    t = term.strip()
    if "-" not in t:
        return [t]
    merged = t.replace("-", "")
    return [t, merged]


def build_query(terms: Dict[str, Any]) -> str:
    """
    Build a PubMed boolean query from PICO terms (population, intervention,
    comparator, outcome).
    - All MeSH uses [MeSH Terms] for broad recall.
    - Freetext terms are truncated (wildcard *) and expanded with hyphen variants.
    - Single-word and multi-word freetext terms are both included.
    - Only Population + Intervention facets are AND'd.
    """
    study_design = (terms.get("study_design") or "any")
    if isinstance(study_design, str):
        study_design = study_design.strip().lower()
    else:
        study_design = "any"

    filtered: Dict[str, Dict[str, List[str]]] = {}
    for facet in _PICO_FACETS:
        data = terms.get(facet)
        if not isinstance(data, dict):
            filtered[facet] = {"mesh": [], "freetext": []}
            continue
        raw_mesh = list(data.get("mesh") or []) if isinstance(data.get("mesh"), list) else []
        mesh = [m for m in raw_mesh if m and str(m).strip()]
        freetext_raw = list(data.get("freetext") or []) if isinstance(data.get("freetext"), list) else []
        freetext = [t for t in freetext_raw if t and str(t).strip()]
        filtered[facet] = {"mesh": mesh, "freetext": freetext}

    def clause(facet: str) -> str:
        mesh = filtered.get(facet, {}).get("mesh") or []
        freetext = filtered.get(facet, {}).get("freetext") or []
        mesh_tag = "[MeSH Terms]"
        parts: List[str] = []
        for m in mesh:
            if m and str(m).strip():
                parts.append(f'"{str(m).strip()}"{mesh_tag}')
        seen_ft: Set[str] = set()
        for f in freetext:
            if not f or not str(f).strip():
                continue
            for variant in _hyphen_variants(str(f).strip()):
                truncated = _truncate(variant)
                if truncated.lower() not in seen_ft:
                    seen_ft.add(truncated.lower())
                    parts.append(f'"{truncated}"[Title/Abstract]')
        if not parts:
            return ""
        return "(" + " OR ".join(parts) + ")"

    _QUERY_FACETS = ("population", "intervention")
    clauses = [clause(f) for f in _QUERY_FACETS]
    non_empty = [c for c in clauses if c]

    if not non_empty:
        return ""

    base = " AND ".join(non_empty)

    if study_design == "randomized_controlled_trial":
        rct_filter = '"Randomized Controlled Trial"[pt]'
        return f"({base}) AND {rct_filter}"
    return base
