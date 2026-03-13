from __future__ import annotations

from typing import Any, Dict, List, Set

_PICO_FACETS = ("population", "intervention", "comparator", "outcome")

_BROAD_MESH_BLACKLIST: Set[str] = {
    "Humans",
    "Animals",
    "Adult",
    "Aged",
    "Aged, 80 and over",
    "Middle Aged",
    "Young Adult",
    "Adolescent",
    "Child",
    "Child, Preschool",
    "Infant",
    "Infant, Newborn",
    "Male",
    "Female",
    "Age Factors",
    "Sex Factors",
    "Prospective Studies",
    "Retrospective Studies",
    "Follow-Up Studies",
    "Cross-Sectional Studies",
    "Cohort Studies",
    "Case-Control Studies",
    "Longitudinal Studies",
    "Treatment Outcome",
    "Patient Selection",
    "Time Factors",
    "Incidence",
    "Prevalence",
    "Comorbidity",
    "Reproducibility of Results",
    "Predictive Value of Tests",
    "Sensitivity and Specificity",
    "Reference Standards",
}


def _normalize_mesh(term: str) -> str:
    return term.strip().lower()


def _is_blacklisted(term: str) -> bool:
    norm = _normalize_mesh(term)
    return any(_normalize_mesh(b) == norm for b in _BROAD_MESH_BLACKLIST)


def build_query(terms: Dict[str, Any]) -> str:
    """
    Build a PubMed boolean query from PICO terms (population, intervention,
    comparator, outcome).
    - MeSH uses [MeSH Major Topic] for precision.
    - Broad demographic/methodological MeSH are blacklisted.
    - Freetext: only multi-word terms are included.
    - Facets with no terms are skipped (not AND'd).
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
        mesh = [m for m in raw_mesh if m and str(m).strip() and not _is_blacklisted(str(m))]
        freetext_raw = list(data.get("freetext") or []) if isinstance(data.get("freetext"), list) else []
        freetext = [t for t in freetext_raw if len(str(t).split()) >= 2]
        filtered[facet] = {"mesh": mesh, "freetext": freetext}

    def clause(facet: str) -> str:
        mesh = filtered.get(facet, {}).get("mesh") or []
        freetext = filtered.get(facet, {}).get("freetext") or []
        parts: List[str] = []
        for m in mesh:
            if m and str(m).strip():
                parts.append(f'"{str(m).strip()}"[MeSH Major Topic]')
        for f in freetext:
            if f and str(f).strip():
                parts.append(f'"{str(f).strip()}"[Title/Abstract]')
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
