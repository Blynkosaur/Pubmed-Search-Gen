from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

_SLEEP_SEC = 0.1
_OPENALEX_BASE = "https://api.openalex.org"
_CITED_BY_PER_PAGE = 200
_BATCH_SIZE = 50


def _openalex_get(url: str, params: dict | None = None) -> dict:
    """GET request to OpenAlex with rate-limit sleep."""
    time.sleep(_SLEEP_SEC)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild plain-text abstract from OpenAlex's inverted-index format."""
    if not inverted_index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return " ".join(w for _, w in words)


def _normalize_doi(doi: str) -> str:
    """Strip https://doi.org/ prefix and lowercase."""
    d = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if d.lower().startswith(prefix):
            d = d[len(prefix):]
    return d.lower().rstrip("/")


def _extract_doi(work: dict) -> str | None:
    raw = work.get("doi") or ""
    return _normalize_doi(raw) if raw.strip() else None


def _extract_mesh(work: dict) -> list[str]:
    """Unique descriptor names, preserving first-seen order."""
    return list(dict.fromkeys(
        m["descriptor_name"]
        for m in (work.get("mesh") or [])
        if m.get("descriptor_name")
    ))


def _make_node(work: dict, hop: int) -> dict:
    """Build a graph node from an OpenAlex work object."""
    return {
        "title": work.get("title") or "",
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "mesh": _extract_mesh(work),
        "cited_by": [],
        "cites": [],
        "hop": hop,
    }


def _add_edge(graph: dict, from_doi: str, to_doi: str) -> None:
    """Record that from_doi cites to_doi (both must already be in the graph)."""
    if from_doi not in graph or to_doi not in graph:
        return
    if to_doi not in graph[from_doi]["cites"]:
        graph[from_doi]["cites"].append(to_doi)
    if from_doi not in graph[to_doi]["cited_by"]:
        graph[to_doi]["cited_by"].append(from_doi)


def _lookup_by_doi(doi: str) -> dict | None:
    try:
        data = _openalex_get(
            f"{_OPENALEX_BASE}/works",
            params={"filter": f"doi:{_normalize_doi(doi)}"},
        )
        results = data.get("results") or []
        return results[0] if results else None
    except Exception:
        return None


def _lookup_by_title(title: str) -> dict | None:
    try:
        data = _openalex_get(
            f"{_OPENALEX_BASE}/works",
            params={"search": title},
        )
        results = data.get("results") or []
        return results[0] if results else None
    except Exception:
        return None


def _batch_fetch_by_openalex_ids(oa_ids: list[str]) -> list[dict]:
    """Fetch works by OpenAlex IDs in batches using pipe-separated filter."""
    all_works: list[dict] = []
    for i in range(0, len(oa_ids), _BATCH_SIZE):
        batch = oa_ids[i : i + _BATCH_SIZE]
        short = [oid.rsplit("/", 1)[-1] if "/" in oid else oid for oid in batch]
        filter_val = "|".join(short)
        try:
            data = _openalex_get(
                f"{_OPENALEX_BASE}/works",
                params={
                    "filter": f"openalex:{filter_val}",
                    "per_page": str(_BATCH_SIZE),
                },
            )
            all_works.extend(data.get("results") or [])
        except Exception:
            pass
    return all_works


def _fetch_cited_by(url: str) -> list[dict]:
    """Fetch first page of works that cite a paper."""
    try:
        data = _openalex_get(url, params={"per_page": str(_CITED_BY_PER_PAGE)})
        return (data.get("results") or [])[:_CITED_BY_PER_PAGE]
    except Exception:
        return []


def build_citation_graph(
    seed_refs: list[dict[str, str | None]],
    sr_doi: str | None = None,
) -> dict[str, dict]:
    """
    Build a two-hop citation graph via OpenAlex.

    Parameters
    ----------
    seed_refs : list of {"doi": str|None, "title": str|None}
        One entry per reference extracted from the SR PDF.
    sr_doi : str or None
        DOI of the systematic review itself.  Papers that cite the SR
        are added as hop-0 seeds so they also get expanded in Loop 1.

    Returns
    -------
    dict keyed by normalized DOI, each value containing
    title, abstract, mesh, cited_by, cites, hop.
    """
    graph: dict[str, dict] = {}
    oa_id_to_doi: dict[str, str] = {}
    hop0_works: list[tuple[str, dict]] = []

    # ── Loop 0a: seed papers (references extracted from the SR) ──────────
    print(f"[OpenAlex] Loop 0a: resolving {len(seed_refs)} references …")
    for i, ref in enumerate(seed_refs):
        doi = (ref.get("doi") or "").strip() or None
        title = (ref.get("title") or "").strip() or None

        work = None
        if doi:
            work = _lookup_by_doi(doi)
        if work is None and title:
            work = _lookup_by_title(title)
        if work is None:
            continue

        paper_doi = _extract_doi(work)
        if not paper_doi:
            continue

        if paper_doi not in graph:
            graph[paper_doi] = _make_node(work, hop=0)
            oa_id = work.get("id") or ""
            if oa_id:
                oa_id_to_doi[oa_id] = paper_doi
            hop0_works.append((paper_doi, work))

        if (i + 1) % 10 == 0:
            print(f"  … {i + 1}/{len(seed_refs)} resolved")

    print(f"[OpenAlex] Loop 0a done — {len(graph)} seed papers from references")

    # ── Loop 0b: papers that cite the SR itself ──────────────────────────
    if sr_doi:
        print(f"[OpenAlex] Loop 0b: looking up SR (doi={sr_doi}) …")
        sr_work = _lookup_by_doi(sr_doi)
        if sr_work:
            cited_by_url = sr_work.get("cited_by_api_url") or ""
            if cited_by_url:
                print("[OpenAlex] Loop 0b: fetching papers that cite the SR …")
                citing_works = _fetch_cited_by(cited_by_url)
                added = 0
                for cw in citing_works:
                    cw_doi = _extract_doi(cw)
                    if not cw_doi:
                        continue
                    if cw_doi not in graph:
                        graph[cw_doi] = _make_node(cw, hop=0)
                        oa_id = cw.get("id") or ""
                        if oa_id:
                            oa_id_to_doi[oa_id] = cw_doi
                        hop0_works.append((cw_doi, cw))
                        added += 1
                print(f"[OpenAlex] Loop 0b done — added {added} papers citing the SR")
            else:
                print(
                    "[OpenAlex] Loop 0b: SR found on OpenAlex but no cited_by URL; "
                    "skipping cited-by-SR seeds."
                )
        else:
            print(f"[OpenAlex] Loop 0b: could not find SR on OpenAlex (doi={sr_doi})")

    print(f"[OpenAlex] Loop 0 complete — {len(graph)} total hop-0 seeds")

    # ── Loop 1: one-hop neighbors ────────────────────────────────────────

    # 1a) Collect referenced_works OpenAlex IDs across all seeds
    ref_id_to_citers: dict[str, set[str]] = {}
    for seed_doi, work in hop0_works:
        for oa_id in work.get("referenced_works") or []:
            ref_id_to_citers.setdefault(oa_id, set()).add(seed_doi)

    unique_ref_ids = list(ref_id_to_citers.keys())
    print(
        f"[OpenAlex] Loop 1a: fetching {len(unique_ref_ids)} referenced works …"
    )
    ref_works = _batch_fetch_by_openalex_ids(unique_ref_ids)

    for work in ref_works:
        paper_doi = _extract_doi(work)
        if not paper_doi:
            continue
        oa_id = work.get("id") or ""
        if oa_id:
            oa_id_to_doi[oa_id] = paper_doi

        if paper_doi not in graph:
            graph[paper_doi] = _make_node(work, hop=1)

        if oa_id in ref_id_to_citers:
            for seed_doi in ref_id_to_citers[oa_id]:
                _add_edge(graph, seed_doi, paper_doi)

    # 1b) Fetch cited_by for each seed
    print(
        f"[OpenAlex] Loop 1b: fetching cited-by for {len(hop0_works)} seeds …"
    )
    for idx, (seed_doi, work) in enumerate(hop0_works):
        cited_by_url = work.get("cited_by_api_url") or ""
        if not cited_by_url:
            continue
        citing_works = _fetch_cited_by(cited_by_url)
        for cw in citing_works:
            cw_doi = _extract_doi(cw)
            if not cw_doi:
                continue
            oa_id = cw.get("id") or ""
            if oa_id:
                oa_id_to_doi[oa_id] = cw_doi

            if cw_doi not in graph:
                graph[cw_doi] = _make_node(cw, hop=1)

            _add_edge(graph, cw_doi, seed_doi)

        if (idx + 1) % 10 == 0:
            print(f"  … {idx + 1}/{len(hop0_works)} seeds processed")

    print(f"[OpenAlex] Loop 1 done — {len(graph)} total nodes")
    return graph


def load_or_build_citation_graph(
    seed_refs: list[dict[str, str | None]],
    pdf_path: Path,
    sr_doi: str | None = None,
) -> dict[str, dict]:
    """Load the citation graph from cache, or build it and save next to the PDF."""
    cache_path = pdf_path.parent / "citation_graph.json"

    if cache_path.exists():
        print(f"[OpenAlex] Loading cached citation graph from {cache_path}")
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    graph = build_citation_graph(seed_refs, sr_doi=sr_doi)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(
        f"[OpenAlex] Saved citation graph ({len(graph)} nodes) → {cache_path}"
    )

    return graph
