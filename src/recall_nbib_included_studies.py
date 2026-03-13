#!/usr/bin/env python3
"""
Recall of NBIB/RIS search results vs Included Studies spreadsheet.

Reads an Excel "Included Studies" file (standard columns: DOI, Title, Year, etc.,
including PubMed ID) and one or more .nbib or .ris files. Reports:
  - How many included studies appear in the bib file(s)
  - Total number of studies in the bib file(s)
  - Recall % = (included found in bib) / (total included studies)
  - Ratio = (included studies in bib) / (total studies in all bib files)

Matching is done by normalized DOI and/or PubMed ID.
"""

import argparse
import re
import sys
from pathlib import Path


def normalize_doi(doi):
    """Normalize DOI for matching: lowercase, strip protocol and trailing slash.
    Handles full URLs and plain DOI (e.g. '10.1007/...')."""
    if doi is None or (isinstance(doi, float) and (doi != doi)):  # NaN
        return None
    s = str(doi).strip().lower()
    if not s:
        return None
    # Strip common URL prefixes
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # If it looks like a plain DOI (starts with 10.) use as-is after strip
    if s.rstrip("/"):
        return s.rstrip("/")
    return None


def normalize_pmid(pid):
    """Return PubMed ID as string or None."""
    if pid is None or (isinstance(pid, float) and (pid != pid)):
        return None
    s = str(int(pid)) if isinstance(pid, (int, float)) and pid == pid else str(pid).strip()
    return s if s and s.isdigit() else None


def normalize_title_for_match(title):
    """Normalize title for matching: lower, collapse spaces, alphanumeric + spaces."""
    if not title or (isinstance(title, float) and (title != title)):
        return None
    s = re.sub(r"[^\w\s]", " ", str(title).lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def load_included_studies(excel_path):
    """Load included studies from Excel. Returns list of dicts with doi_norm, pmid, title_norm, year."""
    try:
        import openpyxl
    except ImportError:
        sys.exit("Install openpyxl: pip install openpyxl")

    wb = openpyxl.load_workbook(str(excel_path), read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    headers = [str(h or "").strip() for h in next(rows_iter, [])]

    doi_idx = pmid_idx = title_idx = year_idx = None
    for i, h in enumerate(headers):
        hl = h.lower()
        if "doi" in hl and doi_idx is None:
            doi_idx = i
        if ("pubmed" in hl or "pmid" in hl) and pmid_idx is None:
            pmid_idx = i
        if "title" in hl and title_idx is None:
            title_idx = i
        if "year" in hl and year_idx is None:
            year_idx = i

    def _cell(row_tuple, idx):
        if idx is None or idx >= len(row_tuple):
            return None
        return row_tuple[idx]

    studies = []
    seen = set()
    for row in rows_iter:
        doi_norm = normalize_doi(_cell(row, doi_idx))
        pmid = normalize_pmid(_cell(row, pmid_idx))
        title_norm = normalize_title_for_match(_cell(row, title_idx))
        year_raw = _cell(row, year_idx)
        if isinstance(year_raw, (int, float)) and year_raw == year_raw:
            year = int(year_raw)
        else:
            year = None
        key = (doi_norm or "", pmid or "", title_norm or "")
        if key == ("", "", ""):
            continue
        if key in seen:
            continue
        seen.add(key)
        doi_display = str(_cell(row, doi_idx) or "").strip() or doi_norm or ""
        studies.append({
            "doi_norm": doi_norm,
            "pmid": pmid,
            "title_norm": title_norm,
            "year": year,
            "doi_display": doi_display,
        })
    wb.close()
    return studies


def parse_ris_file(path):
    """Parse RIS file; yield dicts with doi_norm, pmid, title_norm per record."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"^\s*ER\s*-\s*$", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        rec = {}
        for line in block.split("\n"):
            m = re.match(r"^\s*([A-Z0-9]{2})\s*-\s*(.*)$", line)
            if not m:
                continue
            tag, value = m.group(1), m.group(2).strip()
            if tag == "DO":
                rec["doi_norm"] = normalize_doi(value)
            elif tag == "ID":
                v = value.strip()
                if v.isdigit():
                    rec["pmid"] = v
            elif tag == "T1":
                rec["title_norm"] = normalize_title_for_match(value)
            elif tag == "Y1":
                # Y1 often like "2025//" or "2009"
                y = re.match(r"(\d{4})", value)
                if y:
                    rec["year"] = int(y.group(1))
        if "doi_norm" not in rec:
            rec["doi_norm"] = None
        if "pmid" not in rec:
            rec["pmid"] = None
        if "title_norm" not in rec:
            rec["title_norm"] = None
        if "year" not in rec:
            rec["year"] = None
        yield rec


def parse_nbib_file(path):
    """Parse NBIB file; yield dicts with doi_norm, pmid, title_norm per record.
    MEDLINE/NBIB puts DOI in LID or AID lines like 'LID - 10.1001/jama.2025.3046 [doi]', not DOI-."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text)
    doi_in_value = re.compile(r"(10\.\d{4,}/[^\s\[\]]+)")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        rec = {}
        for line in block.split("\n"):
            m = re.match(r"^\s*(PMID|DOI|TI|DP|LID|AID)\s*-\s*(.*)$", line, re.IGNORECASE)
            if not m:
                continue
            tag, value = m.group(1).upper(), m.group(2).strip()
            if tag == "DOI":
                rec["doi_norm"] = normalize_doi(value)
            elif tag == "PMID":
                v = value.strip()
                if v.isdigit():
                    rec["pmid"] = v
            elif tag == "TI":
                rec["title_norm"] = normalize_title_for_match(value)
            elif tag == "DP":
                y = re.match(r"(\d{4})", value)
                if y:
                    rec["year"] = int(y.group(1))
            elif tag in ("LID", "AID") and "[doi]" in value.lower():
                # e.g. "10.1001/jama.2025.3046 [doi]" or "AID - 10.1016/j.jcin.2021.09.032 [doi]"
                d = doi_in_value.search(value)
                if d and ("doi_norm" not in rec or not rec["doi_norm"]):
                    rec["doi_norm"] = normalize_doi(d.group(1))
        if "doi_norm" not in rec:
            rec["doi_norm"] = None
        if "pmid" not in rec:
            rec["pmid"] = None
        if "title_norm" not in rec:
            rec["title_norm"] = None
        if "year" not in rec:
            rec["year"] = None
        yield rec


def load_bib_studies(paths):
    """Load all studies from RIS/NBIB files. Returns (list of dicts, total count)."""
    all_records = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"Warning: file not found: {path}", file=sys.stderr)
            continue
        suf = path.suffix.lower()
        if suf == ".ris":
            records = list(parse_ris_file(path))
        elif suf == ".nbib":
            records = list(parse_nbib_file(path))
        else:
            print(f"Warning: unsupported format {suf}, skipping {path}", file=sys.stderr)
            continue
        all_records.extend(records)
    return all_records


def _title_similar(a, b, threshold=0.92):
    """True if normalized titles are similar enough (handles typos like recurrance/recurrence)."""
    if not a or not b:
        return False
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio() >= threshold


def build_bib_lookup(bib_records):
    """Build sets for DOI, PMID, and list of (title_norm, year) for title matching."""
    by_doi = set()
    by_pmid = set()
    by_title = []  # list of (title_norm, year) for fuzzy match
    for r in bib_records:
        if r.get("doi_norm"):
            by_doi.add(r["doi_norm"])
        if r.get("pmid"):
            by_pmid.add(r["pmid"])
        if r.get("title_norm"):
            by_title.append((r["title_norm"], r.get("year")))
    return by_doi, by_pmid, by_title


def _matched_by_title(inc_title_norm, inc_year, by_title):
    """Check if included study matches any bib record by title (and optional year)."""
    if not inc_title_norm:
        return False
    for bib_title, bib_year in by_title:
        if not bib_title:
            continue
        if _title_similar(inc_title_norm, bib_title):
            if inc_year is None or bib_year is None or inc_year == bib_year:
                return True
    return False


def _is_found(s, by_doi, by_pmid, by_title):
    """True if this included study is in the bib (by DOI, PMID, or title)."""
    if s.get("doi_norm") and s["doi_norm"] in by_doi:
        return True
    if s.get("pmid") and s["pmid"] in by_pmid:
        return True
    if _matched_by_title(s.get("title_norm"), s.get("year"), by_title):
        return True
    return False


def count_matches(included_studies, by_doi, by_pmid, by_title):
    """Count how many included studies are found in bib (DOI, PMID, or title match)."""
    return sum(1 for s in included_studies if _is_found(s, by_doi, by_pmid, by_title))


def main():
    parser = argparse.ArgumentParser(
        description="Recall of NBIB/RIS files vs Included Studies Excel"
    )
    parser.add_argument(
        "excel",
        type=Path,
        help="Path to Included Studies Excel file (.xlsx)",
    )
    parser.add_argument(
        "bib_files",
        nargs="+",
        type=Path,
        help="One or more .nbib or .ris files",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only print summary numbers",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List which included studies were found vs not found",
    )
    args = parser.parse_args()

    if not args.excel.exists():
        sys.exit(f"Excel file not found: {args.excel}")

    included = load_included_studies(args.excel)
    n_included = len(included)
    if n_included == 0:
        sys.exit("No included studies found in Excel (no valid DOI or PubMed ID).")

    bib_records = load_bib_studies(args.bib_files)
    n_bib_total = len(bib_records)

    by_doi, by_pmid, by_title = build_bib_lookup(bib_records)
    n_found = count_matches(included, by_doi, by_pmid, by_title)

    recall_pct = 100.0 * n_found / n_included if n_included else 0.0
    ratio = n_found / n_bib_total if n_bib_total else 0.0

    if args.quiet:
        print(f"{n_found}\t{n_included}\t{n_bib_total}\t{recall_pct:.2f}\t{ratio:.4f}")
        return

    print("=== Recall: Included Studies vs NBIB/RIS ===\n")
    print(f"Included studies (Excel):     {n_included}")
    print(f"Included studies found in bib: {n_found}")
    print(f"Total studies in bib file(s):  {n_bib_total}")
    print(f"Recall %:                     {recall_pct:.2f}%")
    print(f"Ratio (included in bib / total in bib): {ratio:.4f}")
    print()
    print("Bib files used:")
    for p in args.bib_files:
        print(f"  - {p}")

    not_found = [s for s in included if not _is_found(s, by_doi, by_pmid, by_title)]
    if not_found:
        print(f"\n--- NOT FOUND ({len(not_found)}) ---\n")
        for i, s in enumerate(not_found, 1):
            doi_display = (s.get("doi_display") or s.get("doi_norm") or "").strip()
            pmid_display = s.get("pmid") or ""
            print(f"  {i:2}. DOI={doi_display or '(none)'}  PMID={pmid_display or '(none)'}")

    if args.list:
        print("\n--- All included studies: FOUND vs NOT FOUND ---\n")
        for i, s in enumerate(included, 1):
            found = _is_found(s, by_doi, by_pmid, by_title)
            label = "FOUND" if found else "NOT FOUND"
            doi_display = (s.get("doi_display") or s.get("doi_norm") or "").strip()
            pmid_display = s.get("pmid") or ""
            print(f"  {i:2}. [{label}] DOI={doi_display or '(none)'}  PMID={pmid_display or '(none)'}")


if __name__ == "__main__":
    main()
