from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

_SLEEP_SEC = 0.1
_OPENALEX_BASE = "https://api.openalex.org"
_CITED_BY_PER_PAGE = 200
_BATCH_SIZE = 50
_MAX_CITED_BY_PAGES = 50  # cap cited-by pages per seed to avoid runaway


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


def _make_node(
    work: dict,
    hop: int,
    cited_by_hop1_count: int | None = None,
    connections_to_hop2: int | None = None,
) -> dict:
    """Build a graph node from an OpenAlex work object."""
    node: dict = {
        "title": work.get("title") or "",
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "mesh": _extract_mesh(work),
        "cited_by": [],
        "cites": [],
        "hop": hop,
    }
    if hop == 2 and cited_by_hop1_count is not None:
        node["cited_by_hop1_count"] = cited_by_hop1_count
    if hop == 3 and connections_to_hop2 is not None:
        node["connections_to_hop2"] = connections_to_hop2
    return node


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


def _cited_by_url_for_work(work: dict) -> str:
    """Return the URL to fetch works that cite this one. List endpoint omits cited_by_api_url; build from id."""
    url = (work.get("cited_by_api_url") or "").strip()
    if url:
        return url
    oa_id = work.get("id") or ""
    if not oa_id:
        return ""
    short = oa_id.rsplit("/", 1)[-1] if "/" in oa_id else oa_id
    if short.startswith("W"):
        return f"{_OPENALEX_BASE}/works?filter=cites:{short}"
    return ""


def _fetch_all_cited_by(cited_by_api_url: str) -> list[dict]:
    """Paginate cited_by_api_url and return all citing works (up to _MAX_CITED_BY_PAGES)."""
    if not cited_by_api_url or not cited_by_api_url.strip():
        return []
    all_results: list[dict] = []
    page = 1
    while page <= _MAX_CITED_BY_PAGES:
        try:
            data = _openalex_get(
                cited_by_api_url,
                params={
                    "per_page": str(_CITED_BY_PER_PAGE),
                    "page": str(page),
                    "select": "id,doi,referenced_works",
                },
            )
            results = data.get("results") or []
            if not results:
                break
            all_results.extend(results)
            if len(results) < _CITED_BY_PER_PAGE:
                break
            page += 1
        except Exception:
            break
    return all_results


def _fetch_work_by_openalex_id(oa_id: str) -> dict | None:
    """Fetch a single work by OpenAlex ID (e.g. https://openalex.org/W123 or W123)."""
    if not oa_id or not oa_id.strip():
        return None
    oa_id = oa_id.strip()
    if not oa_id.startswith("http"):
        oa_id = f"https://openalex.org/{oa_id}"
    try:
        return _openalex_get(oa_id)
    except Exception:
        return None


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

    print(f"[OpenAlex] Loop 0 complete — {len(hop0_works)} hop-0 seeds (from Excel/refs only)")

    seed_dois = {doi for doi, _ in hop0_works}

    # ── Loop 1: Hop-1 = papers that CITE the SR (forward citations of the SR only) ───
    hop1_works: list[tuple[str, dict]] = []
    if sr_doi:
        print(f"[OpenAlex] Loop 1: looking up SR (doi={sr_doi}) and fetching papers that cite it …")
        sr_work = _lookup_by_doi(sr_doi)
        if sr_work:
            cited_by_url = _cited_by_url_for_work(sr_work)
            if cited_by_url:
                citing_works = _fetch_all_cited_by(cited_by_url)
                for cw in citing_works:
                    cw_doi = _extract_doi(cw)
                    if not cw_doi or cw_doi in seed_dois:
                        continue
                    oa_id = cw.get("id") or ""
                    if oa_id:
                        oa_id_to_doi[oa_id] = cw_doi
                    if cw_doi not in graph:
                        graph[cw_doi] = _make_node(cw, hop=1)
                        hop1_works.append((cw_doi, cw))
                print(f"[OpenAlex] Loop 1 done — {len(hop1_works)} hop-1 nodes (papers that cite the SR)")
            else:
                print("[OpenAlex] Loop 1: SR has no cited_by URL; 0 hop-1 nodes.")
        else:
            print(f"[OpenAlex] Loop 1: SR not found on OpenAlex (doi={sr_doi}); 0 hop-1 nodes.")
    else:
        print("[OpenAlex] Loop 1: no SR DOI; 0 hop-1 nodes.")

    hop1_dois = {doi for doi, _ in hop1_works}

    # ── Loop 2: Hop-2 = references of hop-1 papers; keep top 30 by citation count ─
    ref_oa_id_to_count: dict[str, int] = {}
    for _hop1_doi, work in hop1_works:
        refs = work.get("referenced_works") or []
        if not refs and work.get("id"):
            full = _fetch_work_by_openalex_id(work["id"])
            if full:
                refs = full.get("referenced_works") or []
        for oa_id in refs:
            if oa_id:
                ref_oa_id_to_count[oa_id] = ref_oa_id_to_count.get(oa_id, 0) + 1

    # Resolve OA IDs to DOIs; exclude hop-0 and hop-1
    to_fetch = [oa_id for oa_id in ref_oa_id_to_count if oa_id not in oa_id_to_doi]
    if to_fetch:
        ref_works = _batch_fetch_by_openalex_ids(to_fetch)
    else:
        ref_works = []

    ref_doi_to_count: dict[str, int] = {}
    for work in ref_works:
        paper_doi = _extract_doi(work)
        if not paper_doi or paper_doi in seed_dois or paper_doi in hop1_dois:
            continue
        oa_id = work.get("id") or ""
        if oa_id:
            oa_id_to_doi[oa_id] = paper_doi
        cnt = ref_oa_id_to_count.get(oa_id, 0)
        ref_doi_to_count[paper_doi] = ref_doi_to_count.get(paper_doi, 0) + cnt

    # Also include refs we already resolved (e.g. from hop1_works) but not in ref_works
    for oa_id, cnt in ref_oa_id_to_count.items():
        ref_doi = oa_id_to_doi.get(oa_id)
        if ref_doi and ref_doi not in seed_dois and ref_doi not in hop1_dois:
            ref_doi_to_count[ref_doi] = ref_doi_to_count.get(ref_doi, 0) + cnt

    # Top 30 by count
    TOP_N_HOP2 = 30
    sorted_hop2 = sorted(
        ref_doi_to_count.items(),
        key=lambda x: -x[1],
    )[:TOP_N_HOP2]
    top30_dois = [doi for doi, _ in sorted_hop2]
    top30_counts = {doi: cnt for doi, cnt in sorted_hop2}

    # Add hop-2 nodes: use work from ref_works when available, else batch fetch
    for work in ref_works:
        paper_doi = _extract_doi(work)
        if paper_doi in top30_dois and paper_doi not in graph:
            cnt = top30_counts.get(paper_doi, 0)
            graph[paper_doi] = _make_node(work, hop=2, cited_by_hop1_count=cnt)
    missing_top30 = [doi for doi in top30_dois if doi not in graph]
    if missing_top30:
        oa_ids_to_fetch = [oa_id for oa_id, d in oa_id_to_doi.items() if d in missing_top30]
        if oa_ids_to_fetch:
            hop2_batch = _batch_fetch_by_openalex_ids(oa_ids_to_fetch)
            for work in hop2_batch:
                paper_doi = _extract_doi(work)
                if paper_doi in missing_top30 and paper_doi not in graph:
                    graph[paper_doi] = _make_node(work, hop=2, cited_by_hop1_count=top30_counts.get(paper_doi, 0))
    for doi in top30_dois:
        if doi in graph:
            continue
        graph[doi] = _make_node(
            {"title": "", "abstract_inverted_index": None, "mesh": []},
            hop=2,
            cited_by_hop1_count=top30_counts.get(doi, 0),
        )

    print(f"[OpenAlex] Loop 2 done — top {TOP_N_HOP2} hop-2 articles (refs of hop-1, by citation count)")
    if sorted_hop2:
        print(f"[OpenAlex] Hop-2 citation counts — min: {sorted_hop2[-1][1]}, max: {sorted_hop2[0][1]}")

    # ── Loop 3: Hop-3 = papers that cite or are cited by the top 30; score by connections; keep top 10 ─
    hop2_dois = set(top30_dois)
    top30_oa_ids = list({oa_id for oa_id, d in oa_id_to_doi.items() if d in hop2_dois})
    hop2_works_for_loop3 = _batch_fetch_by_openalex_ids(top30_oa_ids) if top30_oa_ids else []

    candidate_doi_to_score: dict[str, int] = {}

    # (1) Papers that cite any of the top 30: +1 per top30 they cite
    for work in hop2_works_for_loop3:
        top30_doi = _extract_doi(work)
        if not top30_doi or top30_doi not in hop2_dois:
            continue
        cited_by_url = _cited_by_url_for_work(work)
        if not cited_by_url:
            continue
        citing_list = _fetch_all_cited_by(cited_by_url)
        for cw in citing_list:
            cw_doi = _extract_doi(cw)
            if not cw_doi or cw_doi in seed_dois or cw_doi in hop1_dois or cw_doi in hop2_dois:
                continue
            candidate_doi_to_score[cw_doi] = candidate_doi_to_score.get(cw_doi, 0) + 1
            oa_id = cw.get("id")
            if oa_id:
                oa_id_to_doi[oa_id] = cw_doi

    # (2) Papers that the top 30 cite (refs of top 30): +1 per top30 that cites them
    ref_oa_id_to_count_hop3: dict[str, int] = {}
    for work in hop2_works_for_loop3:
        for ref_oa_id in work.get("referenced_works") or []:
            if ref_oa_id:
                ref_oa_id_to_count_hop3[ref_oa_id] = ref_oa_id_to_count_hop3.get(ref_oa_id, 0) + 1

    need_hop3 = [oa_id for oa_id in ref_oa_id_to_count_hop3 if oa_id not in oa_id_to_doi]
    if need_hop3:
        ref_works_hop3 = _batch_fetch_by_openalex_ids(need_hop3)
        for w in ref_works_hop3:
            oid = w.get("id")
            if oid:
                oa_id_to_doi[oid] = _extract_doi(w)

    for oa_id, cnt in ref_oa_id_to_count_hop3.items():
        ref_doi = oa_id_to_doi.get(oa_id)
        if not ref_doi or ref_doi in seed_dois or ref_doi in hop1_dois or ref_doi in hop2_dois:
            continue
        candidate_doi_to_score[ref_doi] = candidate_doi_to_score.get(ref_doi, 0) + cnt

    TOP_N_HOP3 = 10
    sorted_hop3 = sorted(
        candidate_doi_to_score.items(),
        key=lambda x: -x[1],
    )[:TOP_N_HOP3]
    top10_dois = [doi for doi, _ in sorted_hop3]
    top10_scores = {doi: sc for doi, sc in sorted_hop3}

    # Fetch works for top 10 hop-3; add to graph
    oa_ids_hop3 = [oa_id for oa_id, d in oa_id_to_doi.items() if d in top10_dois]
    if oa_ids_hop3:
        hop3_batch = _batch_fetch_by_openalex_ids(oa_ids_hop3)
    else:
        hop3_batch = []
    for work in hop3_batch:
        paper_doi = _extract_doi(work)
        if paper_doi in top10_dois and paper_doi not in graph:
            graph[paper_doi] = _make_node(work, hop=3, connections_to_hop2=top10_scores.get(paper_doi, 0))
    for doi in top10_dois:
        if doi in graph:
            continue
        graph[doi] = _make_node(
            {"title": "", "abstract_inverted_index": None, "mesh": []},
            hop=3,
            connections_to_hop2=top10_scores.get(doi, 0),
        )

    print(f"[OpenAlex] Loop 3 done — top {TOP_N_HOP3} hop-3 articles (most connections to top 30)")
    if sorted_hop3:
        print(f"[OpenAlex] Hop-3 connection scores — min: {sorted_hop3[-1][1]}, max: {sorted_hop3[0][1]}")

    print(f"[OpenAlex] Graph complete — {len(graph)} total nodes (hop 0: {len(seed_dois)}, hop 1: {len(hop1_dois)}, hop 2: {len(top30_dois)}, hop 3: {len(top10_dois)})")
    return graph


def find_doi_by_title(title: str) -> str | None:
    """
    Search OpenAlex by title and return the DOI of the first matching work, or None.
    Useful when a seed study has title but no DOI (e.g. from Excel).
    """
    if not title or not title.strip():
        return None
    work = _lookup_by_title(title.strip())
    if not work:
        return None
    return _extract_doi(work)


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
            graph = json.load(f)
        hop0_n = sum(1 for n in graph.values() if n.get("hop") == 0)
        hop1_n = sum(1 for n in graph.values() if n.get("hop") == 1)
        if hop0_n > 0 and hop1_n == 0:
            print(
                "[OpenAlex] Cached graph has no hop-1 nodes (stale or from old code). Rebuilding …"
            )
            graph = build_citation_graph(seed_refs, sr_doi=sr_doi)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            print(f"[OpenAlex] Saved citation graph ({len(graph)} nodes) → {cache_path}")
        return graph

    graph = build_citation_graph(seed_refs, sr_doi=sr_doi)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(
        f"[OpenAlex] Saved citation graph ({len(graph)} nodes) → {cache_path}"
    )

    return graph
