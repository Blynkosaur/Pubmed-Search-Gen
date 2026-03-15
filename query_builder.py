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


_MAX_WORDS_PER_PHRASE = 4
_MAX_TERMS_PER_BLOCK = 8
_MAX_QUERY_WORDS = 150


def _reject_long_phrases(terms: List[str]) -> List[str]:
    """Keep only terms with at most 4 words."""
    return [t for t in terms if t and str(t).strip() and len(str(t).strip().split()) <= _MAX_WORDS_PER_PHRASE]


def build_query_two_blocks(
    terms: Dict[str, Any],
    max_terms_per_block: int = _MAX_TERMS_PER_BLOCK,
    max_query_words: int = _MAX_QUERY_WORDS,
) -> str:
    """
    Build a PubMed boolean query from exactly 2 blocks: population AND intervention.
    - Maximum 8 terms per block (lowest priority dropped first when trimming).
    - Reject any phrase longer than 4 words.
    - If query exceeds max_query_words (150), trim terms until under limit.
    - MeSH: [MeSH Terms]; freetext: [Title/Abstract].
    """
    blocks: Dict[str, Dict[str, List[str]]] = {}
    for facet in ("population", "intervention"):
        data = terms.get(facet)
        if not isinstance(data, dict):
            blocks[facet] = {"mesh": [], "freetext": []}
            continue
        mesh = list(data.get("mesh") or [])[: max_terms_per_block * 2]
        freetext = list(data.get("freetext") or [])[: max_terms_per_block * 2]
        mesh = _reject_long_phrases([str(m).strip() for m in mesh if m])[:max_terms_per_block]
        freetext = _reject_long_phrases([str(t).strip() for t in freetext if t])[:max_terms_per_block]
        blocks[facet] = {"mesh": mesh, "freetext": freetext}

    def clause(facet: str) -> str:
        mesh = blocks.get(facet, {}).get("mesh") or []
        freetext = blocks.get(facet, {}).get("freetext") or []
        parts: List[str] = []
        for m in mesh:
            if m:
                parts.append(f'"{m}"[MeSH Terms]')
        for f in freetext:
            if f:
                parts.append(f'"{f}"[Title/Abstract]')
        if not parts:
            return ""
        return "(" + " OR ".join(parts) + ")"

    def query_word_count(q: str) -> int:
        return len(q.split())

    pop_clause = clause("population")
    int_clause = clause("intervention")
    non_empty = [c for c in (pop_clause, int_clause) if c]
    if not non_empty:
        return ""

    query = " AND ".join(non_empty)
    while query_word_count(query) > max_query_words and (blocks["population"]["mesh"] or blocks["population"]["freetext"] or blocks["intervention"]["mesh"] or blocks["intervention"]["freetext"]):
        # Trim one term from the longest block (pop or intervention)
        pop_len = len(blocks["population"]["mesh"]) + len(blocks["population"]["freetext"])
        int_len = len(blocks["intervention"]["mesh"]) + len(blocks["intervention"]["freetext"])
        if pop_len >= int_len and (blocks["population"]["freetext"] or blocks["population"]["mesh"]):
            if blocks["population"]["freetext"]:
                blocks["population"]["freetext"] = blocks["population"]["freetext"][:-1]
            else:
                blocks["population"]["mesh"] = blocks["population"]["mesh"][:-1]
        elif blocks["intervention"]["freetext"] or blocks["intervention"]["mesh"]:
            if blocks["intervention"]["freetext"]:
                blocks["intervention"]["freetext"] = blocks["intervention"]["freetext"][:-1]
            else:
                blocks["intervention"]["mesh"] = blocks["intervention"]["mesh"][:-1]
        else:
            break
        pop_clause = clause("population")
        int_clause = clause("intervention")
        non_empty = [c for c in (pop_clause, int_clause) if c]
        query = " AND ".join(non_empty) if non_empty else ""
    return query
