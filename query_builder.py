from __future__ import annotations

from typing import Any, Dict, List


def build_query(terms: Dict[str, Any]) -> str:
    """
    Build a PubMed boolean query from PICO terms (population, intervention).
    Freetext: only multi-word terms are included (single-word freetext removed).
    MeSH terms are kept as-is.
    If study_design == randomized_controlled_trial, appends RCT publication type filter.
    """
    study_design = (terms.get("study_design") or "any")
    if isinstance(study_design, str):
        study_design = study_design.strip().lower()
    else:
        study_design = "any"

    # Filter freetext: remove single-word terms (len(term.split()) < 2)
    filtered: Dict[str, Dict[str, List[str]]] = {}
    for facet in ("population", "intervention"):
        data = terms.get(facet)
        if not isinstance(data, dict):
            filtered[facet] = {"mesh": [], "freetext": []}
            continue
        mesh = list(data.get("mesh") or []) if isinstance(data.get("mesh"), list) else []
        freetext_raw = list(data.get("freetext") or []) if isinstance(data.get("freetext"), list) else []
        freetext = [t for t in freetext_raw if len(str(t).split()) >= 2]
        filtered[facet] = {"mesh": mesh, "freetext": freetext}

    # Build query: (population clause) AND (intervention clause)
    def clause(facet: str) -> str:
        mesh = filtered.get(facet, {}).get("mesh") or []
        freetext = filtered.get(facet, {}).get("freetext") or []
        parts: List[str] = []
        for m in mesh:
            if m and str(m).strip():
                parts.append(f'"{str(m).strip()}"[MeSH Terms]')
        for f in freetext:
            if f and str(f).strip():
                parts.append(f'"{str(f).strip()}"[Title/Abstract]')
        if not parts:
            return ""
        return "(" + " OR ".join(parts) + ")"

    pop_clause = clause("population")
    int_clause = clause("intervention")

    base = ""
    if pop_clause and int_clause:
        base = f"{pop_clause} AND {int_clause}"
    elif pop_clause:
        base = pop_clause
    elif int_clause:
        base = int_clause

    if not base:
        return ""

    # Append RCT filter only when study_design is randomized_controlled_trial
    if study_design == "randomized_controlled_trial":
        rct_filter = '"Randomized Controlled Trial"[pt]'
        return f"({base}) AND {rct_filter}"
    return base
