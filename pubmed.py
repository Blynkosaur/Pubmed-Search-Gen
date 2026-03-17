from __future__ import annotations

import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Any

import pymupdf
import re
import json
import hashlib
import requests
import xml.etree.ElementTree as ET

# NCBI allows ~3 requests/sec without API key; delay between requests to avoid 429
_EUTILS_DELAY_SEC = 0.34
# Rate limit for batch: 20 req/sec/IP → use 15 per burst then sleep 1 sec to be safe
_PUBMED_BURST_SIZE = 15
_PUBMED_BURST_SLEEP_SEC = 1.0


@dataclass
class Reference:
    """
    Lightweight representation of a reference extracted from a PDF.

    Fields are best-effort and may be None if they cannot be parsed reliably.
    title is authors + title (up to second period); use with doi/pmid for lookup.
    """

    index: int
    raw: str
    doi: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None  # authors + title (up to second period)
    pmid: Optional[str] = None  # if present in raw text (e.g. "PMID: 12345")


_DOI_REGEX = re.compile(
    r"\b10\.\d{4,9}/[^\s\"'>]+\b",
    re.IGNORECASE,
)

_YEAR_REGEX = re.compile(r"\b(19|20)\d{2}\b")

# PMID in reference text: "PMID: 12345", "PubMed PMID 12345", etc.
_PMID_REGEX = re.compile(r"(?:PMID|PubMed(?:\s*ID)?)\s*:?\s*(\d{5,8})\b", re.IGNORECASE)

def extract_doi_or_title(raw: str) -> str:
    """
    Returns only authors and title: DOI if present, otherwise everything up to
    and including the second period (Authors. Title.) — no journal name.
    """
    if not raw or not raw.strip():
        return ""
    s = raw.strip()
    s_no_num = re.sub(r"^\s*\d+[\.\)]\s*", "", s).strip()

    doi_match = _DOI_REGEX.search(s_no_num)
    if doi_match:
        return doi_match.group(0).strip()

    idx = -1
    for _ in range(2):
        idx = s_no_num.find(".", idx + 1)
        if idx == -1:
            break
    if idx != -1:
        return s_no_num[: idx + 1].strip()
    return s_no_num


def _extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    try:
        parts: List[str] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            parts.append(text)
        return "\n".join(parts)
    finally:
        doc.close()


def _find_references_block(full_text: str) -> str:
    """
    Heuristically locate the references section of the manuscript.
    Stops at appendix/search strategy so we only keep actual bibliography.
    """
    lowered = full_text.lower()
    idx = lowered.rfind("\nreferences\n")
    if idx == -1:
        idx = lowered.rfind("\nreference\n")
    if idx == -1:
        start = int(len(full_text) * 0.8)
        block = full_text[start:]
    else:
        block = full_text[idx:]

    # Truncate at appendix or search strategy so we only keep the bibliography.
    stop_markers = [
        "\nappendix",
        "\nsearch strategy",
        "\ndatabase: embase",
        "\ndatabase: ovid",
        "\ndatabase: medline",
        "\ndatabase: pubmed",
    ]
    block_lower = block.lower()
    cut = len(block)
    for marker in stop_markers:
        pos = block_lower.find(marker)
        if pos != -1 and pos < cut:
            cut = pos
    return block[:cut]


def _split_references_block(block: str) -> List[str]:
    """
    Split a references block into individual reference strings.

    Strategy:
    - First, insert synthetic newlines before patterns like '1 Author',
      '12 Smith' when they are not part of a larger number, so that
      compressed reference sections like
      'REFERENCES 1 Smith... 2 Jones...'
      become easier to split.
    - Then split on lines that look like numbered references
      (e.g. "1. ", "12) ").
    """
    # Break up compressed references where numbers and authors all sit
    # on the same long line.
    block = re.sub(r"(?<!\d)(\d{1,3})\s+(?=[A-Z])", r"\n\1 ", block)

    lines = block.splitlines()
    refs: List[str] = []
    current: List[str] = []

    def flush_current():
        if current:
            text = " ".join(part.strip() for part in current if part.strip())
            # Drop trailing appendix/search-strategy text that sometimes sits on same line as last ref
            text = re.sub(r"\s+APPENDIX\s*\d*\s*$", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+Search\s+Strategy\s*$", "", text, flags=re.IGNORECASE)
            if text:
                refs.append(text)

    for line in lines:
        m = re.match(r"^\s*\d+(?:[\.\)])?\s+", line)
        if m:
            rest = line[m.end() :].strip()
            # DOI/URL tail → continuation (e.g. "44. https://doi.org/…")
            if re.match(r"^(https?://|doi\.org|10\.)", rest, re.IGNORECASE):
                current.append(line)
            # Line starts with year then period (e.g. "2016.   " or "2016.   https://...") → continuation
            elif re.match(r"^\s*(19|20)\d{2}\s*[\.\)]\s*", line):
                current.append(line)
            else:
                flush_current()
                current = [line]
        else:
            current.append(line)
    flush_current()
    return refs


def _is_real_reference(raw: str) -> bool:
    """Filter out fragments that are not real citations (section heading, lone numbers, etc.)."""
    s = raw.strip()
    if not s:
        return False
    if re.match(r"^\s*REFERENCES?\s*$", s, re.IGNORECASE):
        return False
    if re.match(r"^\s*\d+\s*$", s):
        return False
    if re.match(r"^\s*(19|20)\d{2}\s*$", s):
        return False
    if _DOI_REGEX.search(s):
        return True
    if _PMID_REGEX.search(s):
        return True
    if len(s) < 15:
        return False
    return True


def _title_from_raw(raw: str) -> str:
    """Extract authors + title: strip leading ref number, then take up to and including the second period."""
    if not raw or not raw.strip():
        return ""
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", raw.strip()).strip()
    if not text:
        return ""
    idx = -1
    for _ in range(2):
        idx = text.find(".", idx + 1)
        if idx == -1:
            break
    if idx != -1:
        return text[: idx + 1].strip()
    return text[:500].strip()


def _title_only_from_raw(raw: str) -> str:
    """Extract just the title (no authors): text between the first and second period."""
    if not raw or not raw.strip():
        return ""
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", raw.strip()).strip()
    if not text:
        return ""
    first = text.find(".")
    if first == -1:
        return text[:500].strip()
    after_first = text[first + 1:].strip()
    if not after_first:
        return text[:500].strip()
    second = after_first.find(".")
    if second != -1:
        return after_first[:second].strip()
    return after_first[:500].strip()


def _parse_single_reference(raw: str, index: int) -> Reference:
    doi_match = _DOI_REGEX.search(raw)
    year_match = _YEAR_REGEX.search(raw)
    pmid_match = _PMID_REGEX.search(raw)

    doi = doi_match.group(0) if doi_match else None
    year = int(year_match.group(0)) if year_match else None
    title = _title_from_raw(raw)
    pmid = pmid_match.group(1) if pmid_match else None

    return Reference(index=index, raw=raw.strip(), doi=doi, year=year, title=title or None, pmid=pmid)


def parse(pdf_path: Union[str, Path]) -> List[Reference]:
    """
    Parse a PDF manuscript, extract the reference list, and return article info.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    List[Reference]
        One entry per reference, with:
        - index: reference order (1-based)
        - raw: full reference string as extracted
        - doi: best-effort DOI (if detected)
        - year: best-effort publication year (if detected)
    """
    path = Path(pdf_path)
    full_text = _extract_pdf_text(path)
    ref_block = _find_references_block(full_text)
    ref_strings = _split_references_block(ref_block)
    ref_strings = [r for r in ref_strings if _is_real_reference(r)]

    references: List[Reference] = []
    for i, ref_text in enumerate(ref_strings, start=1):
        references.append(_parse_single_reference(ref_text, i))

    return references


def fetch_references_metadata(dois: List[str]) -> List[Dict[str, object]]:
    """
    Given a list of DOI strings (no https://doi.org/ prefix), fetch PubMed
    metadata (PMID, DOI, title, abstract, MeSH terms) for all available
    articles in a single batch.

    Results are cached in data/cache/references_metadata_*.json keyed by
    the DOI set so we don't re-fetch on reruns.
    """
    project_root = Path(__file__).resolve().parent
    cache_dir = project_root / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cleaned = sorted({d.strip().lower() for d in dois if d and d.strip()})
    if not cleaned:
        return []

    key_bytes = json.dumps(cleaned, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(key_bytes).hexdigest()[:16]
    cache_path = cache_dir / f"references_metadata_{digest}.json"

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # 1) Batch esearch for all DOIs
    terms = [f'"{d}"[doi]' for d in cleaned]
    query = " OR ".join(terms)

    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": "100000",
        "retmode": "json",
    }
    esearch_resp = requests.get(esearch_url, params=esearch_params, timeout=30)
    esearch_resp.raise_for_status()
    esearch_data = esearch_resp.json()

    id_list = esearch_data.get("esearchresult", {}).get("idlist", []) or []
    if not id_list:
        cache_path.write_text("[]", encoding="utf-8")
        return []

    # 2) Batch efetch for all PMIDs to get abstracts + MeSH
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "rettype": "abstract",
        "retmode": "xml",
    }
    efetch_resp = requests.get(efetch_url, params=efetch_params, timeout=60)
    efetch_resp.raise_for_status()

    root = ET.fromstring(efetch_resp.text)
    records = _parse_pubmed_articles(root)

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return records


def esearch_count(query: str) -> int:
    """
    Run a PubMed ESearch and return only the result count (no IDs).
    Uses retmax=0 to minimize response size.
    """
    if not query or not query.strip():
        return 0
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query.strip(),
        "retmax": "0",
        "retmode": "json",
    }
    resp = requests.get(esearch_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("esearchresult", {})
    count = data.get("count", 0)
    if isinstance(count, str):
        try:
            return int(count)
        except ValueError:
            return 0
    return int(count) if count is not None else 0


def fetch_esearch_counts_batched(
    query_key_pairs: List[Tuple[str, Any]],
    requests_per_sec: int = 10,
) -> Dict[Any, int]:
    """
    Run esearch_count for each (query, key) concurrently in batches.
    Returns dict mapping key -> count. Rate limit: requests_per_sec (default 10 for no API key).
    """
    if not query_key_pairs:
        return {}
    out: Dict[Any, int] = {}
    batch_size = min(requests_per_sec, len(query_key_pairs))
    interval = 1.0 / requests_per_sec if requests_per_sec else 1.0
    for i in range(0, len(query_key_pairs), batch_size):
        batch = query_key_pairs[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(esearch_count, q): k for q, k in batch}
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    out[k] = fut.result()
                except Exception:
                    out[k] = 0
        if i + batch_size < len(query_key_pairs):
            time.sleep(1.0)
    return out


def esearch_query(query: str, retmax: int = 10_000) -> List[str]:
    """
    Run a PubMed ESearch with the given boolean query. Returns list of PMIDs.
    Single request; max 10,000 per request. Use esearch_query_all to fetch every result.
    """
    if not query or not query.strip():
        return []
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query.strip(),
        "retmax": str(min(retmax, 10000)),
        "retmode": "json",
    }
    resp = requests.get(esearch_url, params=params, timeout=60)
    resp.raise_for_status()
    id_list = resp.json().get("esearchresult", {}).get("idlist", []) or []
    return id_list


def _normalize_query_for_esearch(query: str) -> str:
    """Collapse newlines and extra spaces so the API receives a single line (avoids JSON echo with control chars)."""
    return " ".join((query or "").split())


def esearch_query_all_edirect(query: str) -> tuple[Optional[List[str]], Optional[str]]:
    """
    Use EDirect (esearch | efetch -format uid) to retrieve ALL PMIDs for a query.
    EDirect batches automatically and can return more than 10,000 records.
    Returns (list of PMIDs, None) on success, or (None, error_message) if unavailable or failed.
    """
    query = _normalize_query_for_esearch(query)
    if not query:
        return ([], None)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(query)
        query_path = f.name
    try:
        cmd = f'esearch -db pubmed -query "$(cat {shlex.quote(query_path)})" | efetch -format uid'
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "unknown").strip() or f"exit code {result.returncode}"
            return (None, err)
        pmids = [line.strip() for line in result.stdout.splitlines() if line.strip() and line.strip().isdigit()]
        return (pmids, None)
    except FileNotFoundError:
        return (None, "esearch/efetch not found (install EDirect and add to PATH)")
    except subprocess.TimeoutExpired:
        return (None, "EDirect timed out")
    except OSError as e:
        return (None, str(e))
    finally:
        Path(query_path).unlink(missing_ok=True)


def esearch_query_all(
    query: str,
    retmax_per_request: int = 10_000,
) -> List[str]:
    """
    Return ALL result PMIDs for a PubMed query. Tries EDirect first (no 10k limit);
    if EDirect is not available, falls back to E-utilities API (capped at 10,000).
    """
    pmids, _, _ = esearch_query_all_with_source(query, retmax_per_request)
    return pmids


def esearch_query_all_with_source(
    query: str,
    retmax_per_request: int = 10_000,
) -> tuple[List[str], str, Optional[str]]:
    """
    Return (list of PMIDs, source, fallback_reason).
    source is "edirect" or "api". When source is "api", fallback_reason explains why EDirect wasn't used.
    """
    query = _normalize_query_for_esearch(query)
    if not query:
        return ([], "api", None)
    edirect_result, edirect_error = esearch_query_all_edirect(query)
    if edirect_result is not None:
        return (edirect_result, "edirect", None)
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    all_ids: List[str] = []
    retstart = 0
    while True:
        params = {
            "db": "pubmed",
            "term": query,
            "retstart": str(retstart),
            "retmax": str(retmax_per_request),
            "retmode": "json",
        }
        resp = requests.get(esearch_url, params=params, timeout=90)
        resp.raise_for_status()
        try:
            data = resp.json().get("esearchresult", {})
        except Exception as e:
            raw = resp.text
            sanitized = re.sub(r"[\r\n]+", " ", raw)
            try:
                data = json.loads(sanitized).get("esearchresult", {})
            except Exception:
                raise ValueError(f"PubMed response had invalid JSON (control chars?). Query was normalized to one line. Original: {e}") from e
        id_list = data.get("idlist", []) or []
        if not id_list:
            break
        all_ids.extend(id_list)
        retstart += len(id_list)
        if len(id_list) < 9999:
            break
        time.sleep(0.34)
    return (all_ids, "api", edirect_error)


def dois_to_pmids(dois: List[str]) -> Dict[str, str]:
    """
    Resolve DOIs to PMIDs via PubMed. Returns dict mapping normalized DOI -> PMID.
    """
    if not dois:
        return {}
    meta = fetch_references_metadata(dois)
    out: Dict[str, str] = {}
    for m in meta:
        doi = m.get("doi")
        pmid = m.get("pmid")
        if doi and pmid:
            out[doi.strip().lower()] = str(pmid).strip()
    return out


def _parse_pubmed_articles(root: ET.Element) -> List[Dict[str, object]]:
    """Parse PubmedArticle elements from efetch XML into list of dicts (pmid, doi, title, abstract, mesh_terms)."""
    records: List[Dict[str, object]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None

        doi: Optional[str] = None
        aid_list = article.find(".//ArticleIdList")
        if aid_list is not None:
            for aid in aid_list.findall("ArticleId"):
                if aid.get("IdType") == "doi" and aid.text:
                    doi = aid.text.strip().lower()
                    break

        title = ""
        article_el = article.find(".//Article")
        if article_el is not None:
            title_el = article_el.find("ArticleTitle")
            if title_el is not None:
                title = "".join(title_el.itertext()).strip()

        abstract = ""
        if article_el is not None:
            abs_el = article_el.find("Abstract")
            if abs_el is not None:
                parts: List[str] = []
                for at in abs_el.findall("AbstractText"):
                    parts.append("".join(at.itertext()))
                abstract = "\n".join(p.strip() for p in parts if p.strip())

        mesh_terms: List[str] = []
        mesh_list = article.find(".//MeshHeadingList")
        if mesh_list is not None:
            for mh in mesh_list.findall("MeshHeading"):
                desc = mh.find("DescriptorName")
                if desc is not None and desc.text:
                    mesh_terms.append(desc.text.strip())

        if not pmid:
            continue

        records.append(
            {
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "mesh_terms": mesh_terms,
            }
        )
    return records


def fetch_metadata_by_title(title: str) -> Optional[Dict[str, object]]:
    """
    Search PubMed by title and return metadata (abstract, mesh_terms, etc.) for the first hit.
    Returns None if no hit or title is empty. Results are cached by normalized title.
    """
    if not title or not title.strip():
        return None
    title_clean = title.strip()
    project_root = Path(__file__).resolve().parent
    cache_dir = project_root / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_bytes = title_clean.lower().encode("utf-8")
    digest = hashlib.sha256(key_bytes).hexdigest()[:16]
    cache_path = cache_dir / f"title_metadata_{digest}.json"

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if data else None

    # ESearch: title as phrase in [Title]
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": f'"{title_clean}"[Title]',
        "retmax": "1",
        "retmode": "json",
    }
    esearch_resp = requests.get(esearch_url, params=esearch_params, timeout=30)
    esearch_resp.raise_for_status()
    id_list = esearch_resp.json().get("esearchresult", {}).get("idlist", []) or []
    if not id_list:
        cache_path.write_text("null", encoding="utf-8")
        return None

    # EFetch for first PMID
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_resp = requests.get(
        efetch_url,
        params={"db": "pubmed", "id": id_list[0], "rettype": "abstract", "retmode": "xml"},
        timeout=30,
    )
    efetch_resp.raise_for_status()
    root = ET.fromstring(efetch_resp.text)
    records = _parse_pubmed_articles(root)
    first = records[0] if records else None
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(first, f, ensure_ascii=False, indent=2)
    return first


def _batch_efetch_pmids(pmids: List[str], batch_size: int = 200) -> Dict[str, Dict[str, object]]:
    """Fetch PubMed metadata for a list of PMIDs in batches. Returns pmid -> record."""
    meta_by_pmid: Dict[str, Dict[str, object]] = {}
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        time.sleep(_EUTILS_DELAY_SEC)
        resp = requests.get(
            efetch_url,
            params={"db": "pubmed", "id": ",".join(batch), "rettype": "abstract", "retmode": "xml"},
            timeout=60,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        for rec in _parse_pubmed_articles(root):
            pmid = rec.get("pmid")
            if pmid:
                meta_by_pmid[pmid] = rec
    return meta_by_pmid


def fetch_metadata_for_identifiers(
    identifiers: List[str],
    burst_size: int = _PUBMED_BURST_SIZE,
    burst_sleep_sec: float = _PUBMED_BURST_SLEEP_SEC,
) -> List[Dict[str, object]]:
    """
    Send each identifier (exact string, do not transform) to PubMed: ESearch then
    EFetch. Returns metadata including abstract for each, in same order.
    Rate limit: burst_size requests then sleep burst_sleep_sec (default 15 then 1 sec).
    """
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    index_to_pmid: Dict[int, Optional[str]] = {}
    req_count = 0
    for i, ident in enumerate(identifiers):
        if not ident or not str(ident).strip():
            index_to_pmid[i] = None
            continue
        s = str(ident).strip()
        # Numeric 5–8 digits: treat as PMID, skip ESearch
        if s.isdigit() and 5 <= len(s) <= 8:
            index_to_pmid[i] = s
            continue
        if _DOI_REGEX.search(s):
            term = f'"{_DOI_REGEX.search(s).group(0)}"[doi]'
        else:
            term = s
        req_count += 1
        if req_count > 1 and (req_count - 1) % burst_size == 0:
            time.sleep(burst_sleep_sec)
        try:
            resp = requests.get(
                esearch_url,
                params={"db": "pubmed", "term": term, "retmax": "1", "retmode": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            id_list = resp.json().get("esearchresult", {}).get("idlist", []) or []
            index_to_pmid[i] = id_list[0] if id_list else None
        except Exception:
            index_to_pmid[i] = None
    if req_count > 0 and req_count % burst_size != 0:
        time.sleep(burst_sleep_sec)

    unique_pmids = list(dict.fromkeys(p for p in index_to_pmid.values() if p))
    meta_by_pmid: Dict[str, Dict[str, object]] = {}
    req_count = 0
    for i in range(0, len(unique_pmids), 200):
        batch = unique_pmids[i : i + 200]
        req_count += 1
        if req_count > 1 and (req_count - 1) % burst_size == 0:
            time.sleep(burst_sleep_sec)
        try:
            resp = requests.get(
                efetch_url,
                params={"db": "pubmed", "id": ",".join(batch), "rettype": "abstract", "retmode": "xml"},
                timeout=60,
            )
            resp.raise_for_status()
            for rec in _parse_pubmed_articles(ET.fromstring(resp.text)):
                pmid = rec.get("pmid")
                if pmid:
                    meta_by_pmid[pmid] = rec
        except Exception:
            pass
    if req_count > 0 and req_count % burst_size != 0:
        time.sleep(burst_sleep_sec)

    return [
        meta_by_pmid.get(index_to_pmid.get(i, ""), {}) if index_to_pmid.get(i) else {}
        for i in range(len(identifiers))
    ]


def get_reference_texts(refs: List[Reference]) -> List[str]:
    """
    For each reference, return the article text from PubMed: abstract when
    available. If DOI is present, fetch by DOI (batched); if no DOI, search
    by title (rate-limited ESearch) then batch EFetch. If no PubMed hit, use title only.

    Parameters
    ----------
    refs : List[Reference]
        References from parse().

    Returns
    -------
    List[str]
        One string per reference, same order as refs.
    """
    # 1) Batch fetch by DOI
    dois = [r.doi for r in refs if r.doi]
    metadata = fetch_references_metadata(dois)
    meta_by_doi = {m.get("doi"): m for m in metadata if m.get("doi")}

    # 2) No-DOI refs: collect (index, title), then rate-limited ESearch per title, then batch EFetch
    no_doi_titles: List[tuple[int, str]] = []
    for i, ref in enumerate(refs):
        doi_key = (ref.doi or "").strip().lower()
        if doi_key:
            continue
        title = (ref.title or _title_from_raw(ref.raw or "") or "").strip()
        no_doi_titles.append((i, title))

    ref_ix_to_pmid: Dict[int, Optional[str]] = {}
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    for i, title in no_doi_titles:
        if not title:
            ref_ix_to_pmid[i] = None
            continue
        time.sleep(_EUTILS_DELAY_SEC)
        try:
            resp = requests.get(
                esearch_url,
                params={"db": "pubmed", "term": f'"{title}"[Title]', "retmax": "1", "retmode": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            id_list = resp.json().get("esearchresult", {}).get("idlist", []) or []
            ref_ix_to_pmid[i] = id_list[0] if id_list else None
        except Exception:
            ref_ix_to_pmid[i] = None

    # 3) Batch EFetch for all PMIDs from title lookups
    pmids = list(dict.fromkeys(p for p in ref_ix_to_pmid.values() if p))
    meta_by_pmid = _batch_efetch_pmids(pmids) if pmids else {}

    # 4) Build result in ref order
    result: List[str] = []
    for i, ref in enumerate(refs):
        title = (ref.title or _title_from_raw(ref.raw or "") or "").strip()
        doi_key = (ref.doi or "").strip().lower()

        if doi_key and doi_key in meta_by_doi:
            meta = meta_by_doi[doi_key]
            abstract = (meta.get("abstract") or "").strip()
            result.append(abstract if abstract else title)
        else:
            pmid = ref_ix_to_pmid.get(i)
            meta = meta_by_pmid.get(pmid) if pmid else None
            abstract = (meta.get("abstract") or "").strip() if meta else ""
            result.append(abstract if abstract else title)
    return result


def search_and_fetch_nbib(query: str, retmax: int = 10_000) -> str:
    """
    Run a PubMed query (ESearch), fetch full records for the PMIDs,
    and return the result as MEDLINE/NBIB text (rettype=medline, retmode=text).

    query: PubMed boolean query string.
    retmax: Maximum number of PMIDs to fetch (default 10000).
    Returns: Raw MEDLINE text (suitable to write as .nbib).
    """
    if not (query or query.strip()):
        return ""

    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": query.strip(),
        "retmax": str(retmax),
        "retmode": "json",
    }
    esearch_resp = requests.get(esearch_url, params=esearch_params, timeout=60)
    esearch_resp.raise_for_status()
    esearch_data = esearch_resp.json()
    id_list = esearch_data.get("esearchresult", {}).get("idlist", []) or []
    if not id_list:
        return ""

    # EFetch in batches (PubMed allows ~200 IDs per request for stability)
    batch_size = 200
    all_parts: List[str] = []
    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i : i + batch_size]
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(batch_ids),
            "rettype": "medline",
            "retmode": "text",
        }
        efetch_resp = requests.get(efetch_url, params=efetch_params, timeout=60)
        efetch_resp.raise_for_status()
        all_parts.append(efetch_resp.text)

    return "\n".join(all_parts)

