#!/usr/bin/env python3
from __future__ import annotations

"""
Citation graph unified pipeline.

This script merges the previous multi-file implementation into one file, while
still supporting single-step execution.

Stages:
1) extract  -> raw citations (.csv/.json)
2) parse    -> parsed citations (.csv/.json)
3) match    -> citation_matches.csv
4) build    -> citation_edges.csv + citation_graph.graphml + QA sample
5) analyze  -> stats/rankings/communities
6) all      -> run 1~5 in order
"""

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import random
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# =========================
# Shared helpers
# =========================

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def normalize_space(text: str) -> str:
    """Collapse repeated whitespace and strip."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_text(text: str) -> str:
    """Normalize unicode text for stable parsing and matching."""
    text = unicodedata.normalize("NFKC", text or "")
    return normalize_space(text)


def normalize_title(text: str) -> str:
    """Normalize title to lowercase alnum tokens for matching."""
    text = normalize_text(text).lower()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def normalize_author_token(text: str) -> str:
    """Normalize author token for author+year key."""
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z]+", " ", text)
    return normalize_space(text)


def extract_doi(text: str) -> str:
    """Extract first DOI from text."""
    match = DOI_RE.search(text or "")
    if not match:
        return ""
    doi = match.group(1).strip().rstrip(".,);:")
    return doi.lower()


def extract_years(text: str) -> List[str]:
    """Extract all 4-digit years in [1900, 2099]."""
    years = []
    for y in YEAR_RE.findall(text or ""):
        yi = int(y)
        if 1900 <= yi <= 2099:
            years.append(str(yi))
    return years


def first_author_key(authors_text: str) -> str:
    """Build normalized first-author last-name key."""
    authors_text = normalize_text(authors_text)
    if not authors_text:
        return ""
    first = re.split(r";|,| and ", authors_text, maxsplit=1, flags=re.IGNORECASE)[0]
    tokens = [t for t in re.split(r"[\s\-]+", first) if t]
    if not tokens:
        return ""
    return normalize_author_token(tokens[-1])


def title_similarity(a: str, b: str) -> float:
    """Compute title similarity score in [0, 1]."""
    na = normalize_title(a)
    nb = normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV as list of row dicts."""
    # Try common encodings from Excel/PowerShell exports on Windows.
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp1252", "latin-1")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to decode CSV file: {path}. Tried encodings: {', '.join(encodings)}"
    ) from last_err


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    """Write rows into CSV with fixed field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_json(path: Path) -> dict:
    """Read JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: object) -> None:
    """Write JSON file (UTF-8, pretty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_json_path(path: Path) -> bool:
    """Check whether output/input path uses JSON format."""
    return path.suffix.lower() == ".json"


def _stable_id_from_relpath(rel_path: str) -> str:
    """Build stable synthetic paper id from relative path."""
    digest = hashlib.sha1((rel_path or "").encode("utf-8")).hexdigest()[:12]
    return f"paper_auto_{digest}"


def read_raw_citation_rows(path: Path) -> List[Dict[str, str]]:
    """Read raw citation rows from CSV or JSON."""
    if not _is_json_path(path):
        return read_csv(path)
    obj = read_json(path)
    items = obj.get("raw_citations", [])
    if not isinstance(items, list):
        return []
    out: List[Dict[str, str]] = []
    for x in items:
        if not isinstance(x, dict):
            continue
        out.append(
            {
                "source_paper_id": str(x.get("source_paper_id", "") or ""),
                "source_md_path": str(x.get("source_md_path", "") or ""),
                "ref_idx": str(x.get("ref_idx", "") or ""),
                "raw_citation": str(x.get("raw_citation", "") or ""),
            }
        )
    return out


def write_raw_citation_rows(path: Path, rows: Sequence[Dict[str, object]], meta: Optional[Dict[str, object]] = None) -> None:
    """Write raw citation rows to CSV or JSON based on file extension."""
    if not _is_json_path(path):
        write_csv(path, rows, fieldnames=["source_paper_id", "source_md_path", "ref_idx", "raw_citation"])
        return
    payload = {
        "meta": meta or {},
        "raw_citations": [
            {
                "source_paper_id": str(r.get("source_paper_id", "") or ""),
                "source_md_path": str(r.get("source_md_path", "") or ""),
                "ref_idx": int(r.get("ref_idx", 0) or 0),
                "raw_citation": str(r.get("raw_citation", "") or ""),
            }
            for r in rows
        ],
    }
    write_json(path, payload)


class IncrementalRawCitationWriter:
    """Incremental writer that persists raw citations after each paper."""

    def __init__(self, path: Path, meta: Optional[Dict[str, object]] = None):
        self.path = path
        self.meta = meta or {}
        self.count = 0
        self._is_json = _is_json_path(path)
        self._file = None
        self._writer = None
        self._first_item = True
        self._insert_pos = 0
        self._tail = "\n]}\n"

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._is_json:
            self._file = self.path.open("w+", encoding="utf-8", newline="")
            prefix = (
                '{"meta":'
                + json.dumps(self.meta, ensure_ascii=False)
                + ',"raw_citations":[\n'
            )
            self._file.write(prefix)
            self._insert_pos = self._file.tell()
            self._file.write(self._tail)
            self._file.flush()
        else:
            self._file = self.path.open("w", encoding="utf-8", newline="")
            fieldnames = ["source_paper_id", "source_md_path", "ref_idx", "raw_citation"]
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()
            self._file.flush()

    def write_rows(self, rows: Sequence[Dict[str, object]]) -> None:
        if not rows:
            return
        for row in rows:
            self.write_row(row)

    def write_row(self, row: Dict[str, object]) -> None:
        if self._is_json:
            self._write_row_json(row)
        else:
            self._write_row_csv(row)
        self.count += 1

    def _write_row_csv(self, row: Dict[str, object]) -> None:
        assert self._writer is not None and self._file is not None
        self._writer.writerow(
            {
                "source_paper_id": row.get("source_paper_id", ""),
                "source_md_path": row.get("source_md_path", ""),
                "ref_idx": row.get("ref_idx", ""),
                "raw_citation": row.get("raw_citation", ""),
            }
        )
        self._file.flush()

    def _write_row_json(self, row: Dict[str, object]) -> None:
        assert self._file is not None
        payload = {
            "source_paper_id": str(row.get("source_paper_id", "") or ""),
            "source_md_path": str(row.get("source_md_path", "") or ""),
            "ref_idx": int(row.get("ref_idx", 0) or 0),
            "raw_citation": str(row.get("raw_citation", "") or ""),
        }
        chunk = ("" if self._first_item else ",\n") + json.dumps(payload, ensure_ascii=False)
        self._file.seek(self._insert_pos)
        self._file.write(chunk)
        self._insert_pos = self._file.tell()
        self._file.write(self._tail)
        self._file.truncate()
        self._file.flush()
        self._first_item = False

    def close(self, final_meta: Optional[Dict[str, object]] = None) -> None:
        if self._file is None:
            return
        # Optional final metadata patch (one-time rewrite).
        if self._is_json and final_meta:
            self._file.flush()
            try:
                obj = read_json(self.path)
                if isinstance(obj, dict):
                    meta = obj.get("meta", {})
                    if not isinstance(meta, dict):
                        meta = {}
                    meta.update(final_meta)
                    obj["meta"] = meta
                    write_json(self.path, obj)
            except Exception:
                pass
        self._file.close()
        self._file = None


def read_parsed_citation_rows(path: Path) -> List[Dict[str, str]]:
    """Read parsed citation rows from CSV or JSON."""
    if not _is_json_path(path):
        return read_csv(path)
    obj = read_json(path)
    items = obj.get("parsed_citations", [])
    if not isinstance(items, list):
        return []
    out: List[Dict[str, str]] = []
    fields = [
        "source_paper_id",
        "source_md_path",
        "ref_idx",
        "raw_citation",
        "parsed_authors",
        "parsed_title",
        "parsed_year",
        "parsed_doi",
        "first_author_norm",
        "title_norm",
    ]
    for x in items:
        if not isinstance(x, dict):
            continue
        out.append({k: str(x.get(k, "") or "") for k in fields})
    return out


def write_parsed_citation_rows(path: Path, rows: Sequence[Dict[str, object]], meta: Optional[Dict[str, object]] = None) -> None:
    """Write parsed citation rows to CSV or JSON based on file extension."""
    fieldnames = [
        "source_paper_id",
        "source_md_path",
        "ref_idx",
        "raw_citation",
        "parsed_authors",
        "parsed_title",
        "parsed_year",
        "parsed_doi",
        "first_author_norm",
        "title_norm",
    ]
    if not _is_json_path(path):
        write_csv(path, rows, fieldnames=fieldnames)
        return
    payload = {
        "meta": meta or {},
        "parsed_citations": [{k: r.get(k, "") for k in fieldnames} for r in rows],
    }
    write_json(path, payload)


class IncrementalParsedCitationWriter:
    """Incremental writer that persists parsed citations row-by-row."""

    fieldnames = [
        "source_paper_id",
        "source_md_path",
        "ref_idx",
        "raw_citation",
        "parsed_authors",
        "parsed_title",
        "parsed_year",
        "parsed_doi",
        "first_author_norm",
        "title_norm",
    ]

    def __init__(self, path: Path, meta: Optional[Dict[str, object]] = None):
        self.path = path
        self.meta = meta or {}
        self.count = 0
        self._is_json = _is_json_path(path)
        self._file = None
        self._writer = None
        self._first_item = True
        self._insert_pos = 0
        self._tail = "\n]}\n"

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._is_json:
            self._file = self.path.open("w+", encoding="utf-8", newline="")
            prefix = (
                '{"meta":'
                + json.dumps(self.meta, ensure_ascii=False)
                + ',"parsed_citations":[\n'
            )
            self._file.write(prefix)
            self._insert_pos = self._file.tell()
            self._file.write(self._tail)
            self._file.flush()
        else:
            self._file = self.path.open("w", encoding="utf-8", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, row: Dict[str, object]) -> None:
        if self._is_json:
            self._write_row_json(row)
        else:
            self._write_row_csv(row)
        self.count += 1

    def _write_row_csv(self, row: Dict[str, object]) -> None:
        assert self._writer is not None and self._file is not None
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._file.flush()

    def _write_row_json(self, row: Dict[str, object]) -> None:
        assert self._file is not None
        payload = {k: row.get(k, "") for k in self.fieldnames}
        chunk = ("" if self._first_item else ",\n") + json.dumps(payload, ensure_ascii=False)
        self._file.seek(self._insert_pos)
        self._file.write(chunk)
        self._insert_pos = self._file.tell()
        self._file.write(self._tail)
        self._file.truncate()
        self._file.flush()
        self._first_item = False

    def close(self, final_meta: Optional[Dict[str, object]] = None) -> None:
        if self._file is None:
            return
        if self._is_json and final_meta:
            self._file.flush()
            try:
                obj = read_json(self.path)
                if isinstance(obj, dict):
                    meta = obj.get("meta", {})
                    if not isinstance(meta, dict):
                        meta = {}
                    meta.update(final_meta)
                    obj["meta"] = meta
                    write_json(self.path, obj)
            except Exception:
                pass
        self._file.close()
        self._file = None


# =========================
# GROBID helpers
# =========================


def _local_name(tag: object) -> str:
    """Return lowercase XML local tag name without namespace."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].lower()


def _node_text(elem: Optional[ET.Element]) -> str:
    """Collect normalized text from an XML element subtree."""
    if elem is None:
        return ""
    return normalize_space(" ".join(t for t in elem.itertext() if t))


def _all_descendants_by_name(root: ET.Element, name: str) -> List[ET.Element]:
    """Get all descendant nodes by local XML tag name."""
    target = name.lower()
    return [e for e in root.iter() if _local_name(e.tag) == target]


def _encode_multipart_form(
    fields: Optional[Dict[str, str]] = None,
    files: Optional[List[Tuple[str, Path, str]]] = None,
) -> Tuple[bytes, str]:
    """Encode multipart/form-data payload for urllib requests."""
    boundary = f"----CodexBoundary{random.getrandbits(64):x}"
    body = bytearray()
    fields = fields or {}
    files = files or []

    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend((value or "").encode("utf-8"))
        body.extend(b"\r\n")

    for field_name, file_path, content_type in files:
        file_name = file_path.name
        ctype = content_type or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; filename="{file_name}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def grobid_post_multipart(
    endpoint_url: str,
    fields: Optional[Dict[str, str]] = None,
    files: Optional[List[Tuple[str, Path, str]]] = None,
    timeout_sec: int = 120,
    accept: str = "application/xml",
) -> str:
    """POST multipart payload to GROBID endpoint and return text response."""
    payload, content_type = _encode_multipart_form(fields=fields, files=files)
    return grobid_post_bytes(
        endpoint_url=endpoint_url,
        payload=payload,
        content_type=content_type,
        timeout_sec=timeout_sec,
        accept=accept,
    )


def grobid_post_bytes(
    endpoint_url: str,
    payload: bytes,
    content_type: str,
    timeout_sec: int = 120,
    accept: str = "application/xml",
) -> str:
    """POST raw bytes payload with explicit content type."""
    req = Request(
        endpoint_url,
        data=payload,
        headers={"Content-Type": content_type, "Accept": accept},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GROBID HTTP {e.code} at {endpoint_url}: {err_text[:300]}") from e
    except URLError as e:
        raise RuntimeError(f"GROBID request failed at {endpoint_url}: {e}") from e


def grobid_post_form_urlencoded(
    endpoint_url: str,
    fields: Dict[str, str],
    timeout_sec: int = 120,
    accept: str = "application/xml",
) -> str:
    """POST application/x-www-form-urlencoded payload."""
    payload = urlencode(fields or {}).encode("utf-8")
    return grobid_post_bytes(
        endpoint_url=endpoint_url,
        payload=payload,
        content_type="application/x-www-form-urlencoded; charset=utf-8",
        timeout_sec=timeout_sec,
        accept=accept,
    )


def grobid_post_text_plain(
    endpoint_url: str,
    text: str,
    timeout_sec: int = 120,
    accept: str = "application/xml",
) -> str:
    """POST plain text payload."""
    payload = (text or "").encode("utf-8")
    return grobid_post_bytes(
        endpoint_url=endpoint_url,
        payload=payload,
        content_type="text/plain; charset=utf-8",
        timeout_sec=timeout_sec,
        accept=accept,
    )


def extract_tei_bibl_nodes(xml_text: str) -> List[ET.Element]:
    """Parse TEI XML and return citation nodes (biblStruct/bibl)."""
    root = ET.fromstring(xml_text)
    bibl_structs = _all_descendants_by_name(root, "biblstruct")
    if bibl_structs:
        return bibl_structs
    return _all_descendants_by_name(root, "bibl")


def citation_struct_from_bibl_node(bibl_node: ET.Element) -> Dict[str, str]:
    """Parse one TEI citation node into parsed citation fields."""
    full_text = _node_text(bibl_node)

    # DOI
    doi = ""
    for idno in _all_descendants_by_name(bibl_node, "idno"):
        if "doi" in (idno.attrib.get("type", "") or "").lower():
            doi = extract_doi(_node_text(idno))
            if doi:
                break
    if not doi:
        doi = extract_doi(full_text)

    # Year
    year = ""
    for date_node in _all_descendants_by_name(bibl_node, "date"):
        when = date_node.attrib.get("when", "") or ""
        m = re.search(r"(19\d{2}|20\d{2})", when)
        if m:
            year = m.group(1)
            break
        text_years = extract_years(_node_text(date_node))
        if text_years:
            year = text_years[-1]
            break
    if not year:
        y = extract_years(full_text)
        year = y[-1] if y else ""

    # Authors
    authors_list: List[str] = []
    for author in _all_descendants_by_name(bibl_node, "author"):
        pers_names = [p for p in author.iter() if _local_name(p.tag) == "persname"]
        if pers_names:
            pers = pers_names[0]
            surname = ""
            forename_parts: List[str] = []
            for child in pers.iter():
                lname = _local_name(child.tag)
                if lname == "surname":
                    surname = _node_text(child)
                elif lname == "forename":
                    forename_parts.append(_node_text(child))
            person = normalize_space(" ".join(forename_parts + ([surname] if surname else []))).strip(" ,;:.")
            if person:
                authors_list.append(person)
                continue
        fallback_author = _node_text(author).strip(" ,;:.")
        if fallback_author:
            authors_list.append(fallback_author)
    authors = "; ".join(dict.fromkeys([a for a in authors_list if a]))

    # Title and venue
    title_candidates: List[Tuple[int, str]] = []
    venue_candidates: List[Tuple[int, str]] = []
    for title_node in _all_descendants_by_name(bibl_node, "title"):
        txt = _node_text(title_node).strip(" ,;:.")
        if not txt:
            continue
        parent = _local_name(getattr(title_node, "tag", ""))
        level = (title_node.attrib.get("level", "") or "").lower()
        parent_elem = None
        # ElementTree has no parent pointer: infer by scanning ancestors.
        for anc in bibl_node.iter():
            for child in list(anc):
                if child is title_node:
                    parent_elem = anc
                    break
            if parent_elem is not None:
                break
        parent = _local_name(parent_elem.tag) if parent_elem is not None else ""

        score = 0
        if level == "a":
            score += 4
        if parent == "analytic":
            score += 3
        if 3 <= len(txt.split()) <= 40:
            score += 1
        if VENUE_HINT_TOKEN_RE.search(txt):
            venue_candidates.append((score + 3, txt))
        if not VENUE_HINT_TOKEN_RE.search(txt):
            title_candidates.append((score, txt))
        else:
            venue_candidates.append((score, txt))

    title = ""
    venue = ""
    if title_candidates:
        title = sorted(title_candidates, key=lambda x: x[0], reverse=True)[0][1]
    if venue_candidates:
        venue = sorted(venue_candidates, key=lambda x: x[0], reverse=True)[0][1]

    if not title:
        title = clean_title_text(full_text, year)
    if not venue:
        venue = ""

    title = clean_title_text(title, year)
    return {
        "parsed_authors": normalize_space(authors),
        "parsed_title": normalize_space(title),
        "parsed_year": normalize_space(year),
        "parsed_venue": normalize_space(venue),
        "parsed_doi": normalize_space(doi).lower(),
        "first_author_norm": first_author_key(authors),
        "title_norm": normalize_title(title),
    }


def merge_parsed_fields(primary: Dict[str, str], fallback: Dict[str, str]) -> Dict[str, str]:
    """Merge parsed citation fields, preferring primary then fallback values."""
    keys = [
        "parsed_authors",
        "parsed_title",
        "parsed_year",
        "parsed_venue",
        "parsed_doi",
        "first_author_norm",
        "title_norm",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        out[k] = (primary.get(k, "") or fallback.get(k, "")).strip()
    if not out["first_author_norm"] and out["parsed_authors"]:
        out["first_author_norm"] = first_author_key(out["parsed_authors"])
    if not out["title_norm"] and out["parsed_title"]:
        out["title_norm"] = normalize_title(out["parsed_title"])
    return out


# =========================
# LLM helpers (Qwen via llmmelon)
# =========================


def post_json(
    endpoint_url: str,
    payload_obj: dict,
    timeout_sec: int = 120,
    headers: Optional[Dict[str, str]] = None,
) -> dict:
    """POST JSON payload and parse JSON response."""
    payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        req = Request(endpoint_url, data=payload, headers=req_headers, method="POST")
        try:
            with urlopen(req, timeout=timeout_sec) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                text = resp.read().decode(charset, errors="replace")
                if not text.strip():
                    return {}
                return json.loads(text)
        except HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"LLM HTTP {e.code} at {endpoint_url}: {err_text[:300]}")
            if e.code in {429, 500, 502, 503, 504} and attempt < 3:
                time.sleep(1.5 * attempt)
                continue
            raise last_err from e
        except (URLError, RemoteDisconnected) as e:
            last_err = RuntimeError(f"LLM request failed at {endpoint_url}: {e}")
            if attempt < 3:
                time.sleep(1.5 * attempt)
                continue
            raise last_err from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM JSON decode failed at {endpoint_url}: {e}") from e
    raise RuntimeError(f"LLM request failed at {endpoint_url}: {last_err}")


def resolve_chat_completions_url(base_or_full_url: str) -> str:
    """Resolve base URL to OpenAI-compatible chat.completions endpoint."""
    u = (base_or_full_url or "").strip().rstrip("/")
    if not u:
        return "https://llmmelon.cloud/v1/chat/completions"
    low = u.lower()
    if low.endswith("/chat/completions"):
        return u
    if low.endswith("/v1"):
        return u + "/chat/completions"
    return u + "/v1/chat/completions"


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences if present."""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


def _extract_json_from_text(text: str):
    """Parse JSON object/array from text content."""
    t = _strip_code_fence(text)
    if not t:
        return None
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    candidates = []
    lb, rb = t.find("["), t.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        candidates.append(t[lb : rb + 1])
    ob, cb = t.find("{"), t.rfind("}")
    if ob != -1 and cb != -1 and cb > ob:
        candidates.append(t[ob : cb + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def _norm_llm_item(item: Dict[str, object]) -> Dict[str, str]:
    """Normalize one LLM parsed item into internal parsed fields."""
    authors = normalize_space(str(item.get("parsed_authors", "") or ""))
    title = normalize_space(str(item.get("parsed_title", "") or ""))
    year_raw = normalize_space(str(item.get("parsed_year", "") or ""))
    y = extract_years(year_raw)
    year = y[-1] if y else (year_raw if re.fullmatch(r"(19\d{2}|20\d{2})", year_raw) else "")
    doi = extract_doi(str(item.get("parsed_doi", "") or ""))
    return {
        "parsed_authors": authors,
        "parsed_title": title,
        "parsed_year": year,
        "parsed_venue": "",
        "parsed_doi": doi,
        "first_author_norm": first_author_key(authors),
        "title_norm": normalize_title(title),
    }


def parse_citation_list_qwen(
    raw_citations: List[str],
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    llm_timeout_sec: int,
) -> List[Dict[str, str]]:
    """Parse citation list via OpenAI-compatible chat.completions."""
    if not raw_citations:
        return []
    endpoint = resolve_chat_completions_url(llm_base_url)
    headers: Dict[str, str] = {}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"

    items = [{"idx": i + 1, "raw_citation": c} for i, c in enumerate(raw_citations)]
    sys_prompt = (
        "You are an academic citation parser. "
        "Extract fields from each raw citation. "
        "Return ONLY JSON with shape: "
        '{"items":[{"idx":1,"parsed_authors":"","parsed_title":"","parsed_year":"","parsed_doi":""}]}. '
        "Do not add explanations."
    )
    user_prompt = "Input citations JSON:\n" + json.dumps(items, ensure_ascii=False)
    model_candidates: List[str] = []
    if llm_model:
        model_candidates.append(llm_model)
    # Tolerate provider naming differences for Qwen 3.5 Flash.
    if llm_model.lower() == "qwen-3.5-flash":
        model_candidates.append("qwen3.5-flash")
    if llm_model.lower() == "qwen3.5-flash":
        model_candidates.append("qwen-3.5-flash")

    last_err: Optional[Exception] = None
    resp = {}
    for model_name in model_candidates:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        try:
            resp = post_json(endpoint, payload, timeout_sec=llm_timeout_sec, headers=headers)
            break
        except Exception as e:
            last_err = e
            continue
    if not resp and last_err is not None:
        raise RuntimeError(f"Qwen parse call failed for models: {model_candidates}") from last_err
    choices = resp.get("choices", []) if isinstance(resp, dict) else []
    content = ""
    if choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""
    parsed_json = _extract_json_from_text(str(content))
    if parsed_json is None:
        return [{} for _ in raw_citations]

    parsed_items = []
    if isinstance(parsed_json, list):
        parsed_items = parsed_json
    elif isinstance(parsed_json, dict):
        if isinstance(parsed_json.get("items"), list):
            parsed_items = parsed_json.get("items", [])
        else:
            parsed_items = [parsed_json]

    out: List[Dict[str, str]] = [{} for _ in raw_citations]
    for obj in parsed_items:
        if not isinstance(obj, dict):
            continue
        idx_obj = obj.get("idx", None)
        try:
            idx = int(idx_obj)
        except Exception:
            idx = None
        if idx is None or idx < 1 or idx > len(raw_citations):
            continue
        out[idx - 1] = _norm_llm_item(obj)
    return out


def _pick_reference_tail_for_qwen(md_text: str, max_chars: int = 40000) -> str:
    """Pick references-tail text from markdown for LLM extraction."""
    text = md_text or ""
    lines = text.splitlines()
    found = find_reference_range(lines) if lines else None
    if found:
        start, end = found
        ref_text = "\n".join(lines[start:end]).strip()
        if ref_text:
            return ref_text[:max_chars]

    # Fallback: last part of paper usually contains references.
    tail = text[-max_chars:]
    return tail


def extract_raw_citations_qwen(
    markdown_text: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    llm_timeout_sec: int = 120,
) -> List[str]:
    """Extract raw citation strings from OCR markdown via Qwen New API."""
    endpoint = resolve_chat_completions_url(llm_base_url)
    headers: Dict[str, str] = {}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"

    ref_text = _pick_reference_tail_for_qwen(markdown_text, max_chars=40000)
    if not normalize_space(ref_text):
        return []

    sys_prompt = (
        "You are extracting bibliography entries from OCR markdown text.\n"
        "Task: output individual reference strings in original order.\n"
        "Rules:\n"
        "1) Return ONLY JSON: {\"raw_citations\":[\"...\"]}.\n"
        "2) One full citation per item; merge wrapped lines.\n"
        "3) Remove leading index markers like [1], (1), 1., 1).\n"
        "4) Keep original citation content (authors/title/venue/year/doi) without rewriting.\n"
        "5) Exclude non-reference noise lines."
    )
    user_prompt = "References candidate text:\n" + ref_text

    model_candidates: List[str] = []
    if llm_model:
        model_candidates.append(llm_model)
    if llm_model.lower() == "qwen-3.5-flash":
        model_candidates.append("qwen3.5-flash")
    if llm_model.lower() == "qwen3.5-flash":
        model_candidates.append("qwen-3.5-flash")

    resp = {}
    last_err: Optional[Exception] = None
    for model_name in model_candidates:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        try:
            resp = post_json(endpoint, payload, timeout_sec=llm_timeout_sec, headers=headers)
            break
        except Exception as e:
            last_err = e
            continue
    if not resp and last_err is not None:
        raise RuntimeError(f"Qwen raw extraction failed for models: {model_candidates}") from last_err

    choices = resp.get("choices", []) if isinstance(resp, dict) else []
    content = ""
    if choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""
    parsed_json = _extract_json_from_text(str(content))
    if not isinstance(parsed_json, dict):
        return []
    raw_list = parsed_json.get("raw_citations", [])
    if not isinstance(raw_list, list):
        return []

    out: List[str] = []
    seen = set()
    for item in raw_list:
        text = clean_raw_citation_candidate(str(item or ""))
        if not text:
            continue
        key = normalize_citation_key(text)
        if key and key not in seen:
            out.append(text)
            seen.add(key)
    return out


# =========================
# Stage 1: raw citation extraction
# =========================

REF_START_SET = {"references", "bibliography", "works cited"}
REF_END_SET = {
    "appendix",
    "supplementary material",
    "author contributions",
    "acknowledgments",
    "acknowledgements",
    "about the authors",
}
ENTRY_START_PATTERNS = [
    re.compile(r"^\s*\[\d+\]\s*"),
    re.compile(r"^\s*\(\d+\)\s*"),
    re.compile(r"^\s*\d+\s*\.\s+"),
    re.compile(r"^\s*\d+\s*\)\s+"),
    re.compile(r"^\s*\[[A-Za-z][A-Za-z\-]{0,12}\d{2,4}[A-Za-z]?\]\s*"),
    re.compile(r"^\s*\[[A-Za-z][A-Za-z\-]{1,15}\]\s+(?:[A-Z]\.|[A-Z])"),
]
ENTRY_MARKER_ANY_RE = re.compile(
    r"(?:\[\d+\]|\(\d+\)|\d+\s*[\.\)]|\[[A-Za-z][A-Za-z\-]{0,12}\d{2,4}[A-Za-z]?\])"
)
RAW_CITATION_NOISE_PATTERNS = [
    re.compile(r"\bdownloaded\b", re.IGNORECASE),
    re.compile(r"\bredistribution subject to\b", re.IGNORECASE),
    re.compile(r"\blicense or copyright\b", re.IGNORECASE),
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    re.compile(r"\btable\s+\d+\b", re.IGNORECASE),
]


def normalize_citation_key(text: str) -> str:
    """Build normalized key for within-paper deduplication."""
    t = normalize_space(text).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return normalize_space(t)


def digit_char_ratio(text: str) -> float:
    """Compute ratio of digit chars among non-space chars."""
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    digit_n = sum(1 for c in non_space if c.isdigit())
    return float(digit_n) / float(len(non_space))


def has_reference_signal(text: str) -> bool:
    """Check if candidate has reference-like signal (year/doi/author pattern)."""
    if extract_doi(text):
        return True
    if extract_years(text):
        return True
    if re.search(r"\b[A-Z][A-Za-z'\-]{2,}\s+[A-Z][A-Za-z'\-]{2,}\b", text):
        return True
    if re.search(r"\b[A-Z]\.\s*[A-Z][A-Za-z'\-]{2,}\b", text):
        return True
    return False


def is_noisy_reference_text(text: str) -> bool:
    """Detect obvious non-reference noise text."""
    return any(p.search(text) for p in RAW_CITATION_NOISE_PATTERNS)


def is_valid_raw_citation_text(text: str) -> bool:
    """Quality gate for raw citations extracted from GROBID/OCR."""
    t = normalize_space(text)
    if not t:
        return False
    if len(t) < 25:
        return False
    if len(t) > 650:
        return False
    if is_noisy_reference_text(t):
        return False
    if digit_char_ratio(t) > 0.45 and not extract_doi(t):
        return False
    if not has_reference_signal(t):
        return False
    return True


def render_structured_raw_from_bibl_node(bibl_node: ET.Element) -> str:
    """Render a stable citation string from TEI bibl node fields."""
    parsed = citation_struct_from_bibl_node(bibl_node)
    authors = normalize_space(parsed.get("parsed_authors", "")).strip(" ,;:.")
    title = normalize_space(parsed.get("parsed_title", "")).strip(" ,;:.")
    venue = normalize_space(parsed.get("parsed_venue", "")).strip(" ,;:.")
    year = normalize_space(parsed.get("parsed_year", "")).strip(" ,;:.")
    doi = normalize_space(parsed.get("parsed_doi", "")).strip(" ,;:.")

    parts: List[str] = []
    if authors:
        parts.append(authors.replace("; ", ", "))
    if title:
        parts.append(title)
    if venue:
        parts.append(venue)
    if year and year not in venue:
        parts.append(year)
    if doi:
        parts.append(doi.lower())

    return normalize_space(", ".join([p for p in parts if p]))


def clean_raw_citation_candidate(text: str) -> str:
    """Normalize one citation candidate before filtering and dedup."""
    t = strip_entry_prefix(text or "")
    t = t.strip(" ,;:.")
    t = normalize_space(t)
    return t


def normalize_heading_line(line: str) -> str:
    """Normalize heading line for reference section boundary detection."""
    line = line.strip()
    line = re.sub(r"^#+\s*", "", line)
    line = re.sub(r"[\s:.\-_=]+$", "", line)
    line = normalize_space(line).lower()
    return line


def find_reference_range(lines: List[str]) -> Optional[Tuple[int, int]]:
    """Find [start, end) line range of references section."""
    start_idx = None
    for i, line in enumerate(lines):
        if normalize_heading_line(line) in REF_START_SET:
            start_idx = i + 1
            break
    if start_idx is None:
        return None

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        if normalize_heading_line(lines[j]) in REF_END_SET:
            end_idx = j
            break
    return start_idx, end_idx


def is_new_entry(line: str) -> bool:
    """Check if line starts a new numbered citation entry."""
    return any(p.search(line) for p in ENTRY_START_PATTERNS)


def strip_entry_prefix(line: str) -> str:
    """Remove citation index marker such as [1], (1), 1., 1)."""
    line = re.sub(
        r"^\s*(\[\d+\]|\(\d+\)|\d+\s*[\.\)]|\[[A-Za-z][A-Za-z\-]{0,12}\d{2,4}[A-Za-z]?\]|\[[A-Za-z][A-Za-z\-]{1,15}\])\s*",
        "",
        line,
    )
    return normalize_space(line)


def _likely_split_boundary(prefix: str) -> bool:
    """Check whether the text before a marker looks like an entry boundary."""
    if re.search(r"[.;]\s*$", prefix):
        return True
    if re.search(r"(?:pp?\.\s*)?\d+\s*[-–]\s*\d+\s*$", prefix, re.IGNORECASE):
        return True
    tail = prefix[-36:]
    if YEAR_RE.search(tail) and re.search(r"[\s)\]]$", prefix):
        return True
    return False


def _chunk_looks_like_citation(text: str) -> bool:
    """Lightweight validity check to avoid over-splitting inline markers."""
    t = normalize_space(text)
    if len(t) < 24:
        return False
    if extract_doi(t):
        return True
    if extract_years(t):
        return True
    # Authors + title style entries usually have at least one comma.
    return "," in t and len(t.split()) >= 5


def split_inline_entries(text: str) -> List[str]:
    """
    Secondary split for merged entries containing multiple refs in one line.
    Example:
    "... 1993. [BGL09] ... 2009."
    """
    s = normalize_space(text)
    if not s:
        return []

    starts = []
    for m in ENTRY_MARKER_ANY_RE.finditer(s):
        st = m.start()
        if st == 0:
            starts.append(0)
            continue
        if _likely_split_boundary(s[:st]):
            starts.append(st)

    starts = sorted(set(starts))
    if len(starts) <= 1:
        return [s]

    if starts[0] != 0:
        starts = [0] + starts
    starts.append(len(s))

    parts: List[str] = []
    for i in range(len(starts) - 1):
        part = normalize_space(s[starts[i] : starts[i + 1]]).strip(" ;.")
        if part:
            parts.append(part)

    # Merge back tiny/invalid chunks to prevent false splits.
    merged: List[str] = []
    for part in parts:
        if not _chunk_looks_like_citation(part) and merged:
            merged[-1] = normalize_space(merged[-1] + " " + part)
        else:
            merged.append(part)
    return merged


def split_and_merge_reference_entries(ref_lines: List[str]) -> List[str]:
    """Split entries by numbering and merge multi-line wrapped citation text."""
    lines = [normalize_space(x) for x in ref_lines]
    lines = [x for x in lines if x]
    if not lines:
        return []

    entries: List[str] = []
    current: List[str] = []
    has_numbered_entry = any(is_new_entry(x) for x in lines)

    if has_numbered_entry:
        for line in lines:
            if is_new_entry(line):
                if current:
                    entries.append(normalize_space(" ".join(current)))
                    current = []
                current.append(strip_entry_prefix(line))
            else:
                if current:
                    current.append(line)
        if current:
            entries.append(normalize_space(" ".join(current)))
        refined = []
        for entry in entries:
            refined.extend(split_inline_entries(entry))
        return [strip_entry_prefix(e) for e in refined if e]

    block: List[str] = []
    for raw in ref_lines:
        if not raw.strip():
            if block:
                entries.append(normalize_space(" ".join(block)))
                block = []
            continue
        block.append(normalize_space(raw))
    if block:
        entries.append(normalize_space(" ".join(block)))
    refined = []
    for entry in entries:
        refined.extend(split_inline_entries(entry))
    return [strip_entry_prefix(e) for e in refined if e]


def extract_one_paper_raw_citations(md_path: Path) -> List[str]:
    """Extract all raw references from one OCR markdown paper."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    found = find_reference_range(lines)
    if not found:
        return []
    start, end = found
    return split_and_merge_reference_entries(lines[start:end])


def extract_raw_references_from_grobid_xml(xml_text: str) -> List[str]:
    """Extract raw citation strings from GROBID TEI XML output."""
    out: List[str] = []
    try:
        nodes = extract_tei_bibl_nodes(xml_text)
    except ET.ParseError:
        nodes = []

    if nodes:
        for node in nodes:
            raw = ""
            # Prefer explicit raw-reference note when provided by GROBID.
            for note in _all_descendants_by_name(node, "note"):
                if (note.attrib.get("type", "") or "").lower() == "raw_reference":
                    raw = clean_raw_citation_candidate(_node_text(note))
                    break
            if not raw:
                # Build stable citation string from TEI fields instead of whole-node text.
                raw = clean_raw_citation_candidate(render_structured_raw_from_bibl_node(node))
            if not raw:
                raw = clean_raw_citation_candidate(_node_text(node))
            if is_valid_raw_citation_text(raw):
                out.append(raw)
    else:
        # Fallback: parse plain-text line output.
        lines = [normalize_space(x) for x in xml_text.splitlines() if normalize_space(x)]
        for item in split_and_merge_reference_entries(lines):
            candidate = clean_raw_citation_candidate(item)
            if is_valid_raw_citation_text(candidate):
                out.append(candidate)

    deduped: List[str] = []
    seen = set()
    for item in out:
        key = normalize_citation_key(item)
        if key and key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def infer_source_rel_path(paper: Dict[str, str]) -> str:
    """Pick source relative path from paper record."""
    for key in ("source_pdf_path", "source_md_path", "source_path", "pdf_path", "file_path"):
        v = normalize_space(paper.get(key, ""))
        if v:
            return v.replace("\\", "/")
    return ""


def build_pdf_name_index(search_roots: Sequence[Optional[Path]]) -> Dict[str, List[Path]]:
    """Build PDF basename index for fallback path resolution."""
    idx: Dict[str, List[Path]] = defaultdict(list)
    for root in search_roots:
        if root is None:
            continue
        if not root.exists():
            continue
        for p in root.rglob("*.pdf"):
            idx[p.name.lower()].append(p.resolve())
    return idx


def resolve_pdf_path(
    paper: Dict[str, str],
    corpus_root: Optional[Path],
    pdf_root: Optional[Path],
    pdf_name_index: Optional[Dict[str, List[Path]]] = None,
) -> Tuple[Optional[Path], str]:
    """Resolve paper PDF absolute path from papers.csv row and roots."""
    source_rel = infer_source_rel_path(paper)
    if source_rel:
        rel_path = Path(source_rel)
        candidates: List[Path] = []
        if rel_path.is_absolute():
            candidates.append(rel_path)
        if pdf_root is not None:
            candidates.append(pdf_root / rel_path)
        if corpus_root is not None:
            candidates.append(corpus_root / rel_path)
        for c in candidates:
            c_resolved = c.expanduser().resolve()
            if c_resolved.exists() and c_resolved.is_file() and c_resolved.suffix.lower() == ".pdf":
                return c_resolved, source_rel

    if pdf_name_index and source_rel:
        file_name = Path(source_rel).name.lower()
        matches = pdf_name_index.get(file_name, [])
        if len(matches) == 1:
            return matches[0], source_rel

    return None, source_rel


def extract_one_paper_raw_citations_grobid(pdf_path: Path, grobid_url: str, timeout_sec: int) -> List[str]:
    """Extract references from one PDF via GROBID /api/processReferences."""
    endpoint = grobid_url.rstrip("/") + "/api/processReferences"
    mime = mimetypes.guess_type(pdf_path.name)[0] or "application/pdf"
    xml_text = grobid_post_multipart(
        endpoint_url=endpoint,
        files=[("input", pdf_path, mime)],
        timeout_sec=timeout_sec,
        accept="application/xml",
    )
    return extract_raw_references_from_grobid_xml(xml_text)


def stage_extract_raw_citations(
    papers_csv: Optional[Path],
    corpus_root: Path,
    raw_citations_csv: Path,
    extract_source: str = "qwen_api",
    llm_base_url: str = "https://llmmelon.cloud",
    llm_api_key: str = "",
    llm_model: str = "qwen-3.5-flash",
    llm_timeout_sec: int = 120,
) -> int:
    """Run stage 1 from OCR markdown and output raw citations (CSV/JSON)."""
    papers: List[Dict[str, str]] = []
    if papers_csv is not None and papers_csv.exists():
        papers = read_csv(papers_csv)
    corpus_root = corpus_root.resolve()
    writer = IncrementalRawCitationWriter(
        raw_citations_csv,
        meta={
            "stage": "extract",
            "extract_source": extract_source,
            "corpus_root": str(corpus_root),
        },
    )

    # Preferred: use papers.csv to preserve paper_id mapping.
    if papers:
        iter_items: List[Tuple[str, str, Path]] = []
        for paper in papers:
            paper_id = paper.get("paper_id", "")
            source_rel = infer_source_rel_path(paper)
            if not paper_id or not source_rel:
                continue

            rel_path = Path(source_rel)
            candidates = [(corpus_root / rel_path).resolve()]
            if rel_path.suffix.lower() == ".pdf":
                candidates.append((corpus_root / rel_path.with_suffix(".md")).resolve())
                candidates.append((corpus_root / rel_path.with_suffix(".txt")).resolve())

            text_path = None
            for c in candidates:
                if c.exists() and c.is_file():
                    text_path = c
                    break
            if text_path is None:
                continue
            iter_items.append((paper_id, source_rel.replace("\\", "/"), text_path))
    else:
        # Fallback: directly scan markdown files under corpus_root.
        iter_items = []
        for p in sorted(corpus_root.rglob("*.md"), key=lambda x: str(x).lower()):
            if not p.is_file():
                continue
            rel = str(p.relative_to(corpus_root)).replace("\\", "/")
            pid = _stable_id_from_relpath(rel)
            iter_items.append((pid, rel, p))

    qwen_fail_papers = 0
    qwen_total_papers = 0
    first_qwen_err: Optional[str] = None
    try:
        for paper_id, source_rel, text_path in iter_items:
            if extract_source == "qwen_api":
                qwen_total_papers += 1
                md_text = text_path.read_text(encoding="utf-8", errors="replace")
                try:
                    raw_entries = extract_raw_citations_qwen(
                        markdown_text=md_text,
                        llm_base_url=llm_base_url,
                        llm_api_key=llm_api_key,
                        llm_model=llm_model,
                        llm_timeout_sec=llm_timeout_sec,
                    )
                except Exception as e:
                    qwen_fail_papers += 1
                    if first_qwen_err is None:
                        first_qwen_err = str(e)
                    raw_entries = []
            else:
                raw_entries = extract_one_paper_raw_citations(text_path)

            paper_rows: List[Dict[str, object]] = []
            for idx, raw in enumerate(raw_entries, start=1):
                paper_rows.append(
                    {
                        "source_paper_id": paper_id,
                        "source_md_path": source_rel.replace("\\", "/"),
                        "ref_idx": idx,
                        "raw_citation": raw,
                    }
                )
            # Persist per paper (not all-at-once).
            writer.write_rows(paper_rows)
    finally:
        writer.close(
            final_meta={
                "num_rows": writer.count,
                "num_papers": len(iter_items),
            }
        )

    if extract_source == "qwen_api" and qwen_total_papers > 0 and qwen_fail_papers == qwen_total_papers:
        msg = first_qwen_err or "unknown qwen_api error"
        raise RuntimeError(
            "All Qwen extract requests failed; no API calls succeeded. "
            f"First error: {msg}"
        )

    return writer.count


# =========================
# Stage 2: citation parsing
# =========================

LEADING_INDEX_RE = re.compile(
    r"^\s*(\[\d+\]|\(\d+\)|\d+\s*[\.\)]|\[[A-Za-z][A-Za-z\-]{0,12}\d{2,4}[A-Za-z]?\]|\[[A-Za-z][A-Za-z\-]{1,15}\])\s*"
)
TAIL_VENUE_RE = re.compile(
    r"\b(vol\.?|volume|no\.?|issue|pp\.?|pages?|journal|transactions|proceedings|conference)\b.*$",
    re.IGNORECASE,
)
QUOTE_TITLE_RE = re.compile(r"[\"'\u201c\u201d]([^\"'\u201c\u201d]{8,300})[\"'\u201c\u201d]")
SEG_SPLIT_RE = re.compile(r"\s*,\s*")
INITIAL_RE = re.compile(r"^[A-Z]\.?$")
NAME_WORD_RE = re.compile(r"^[A-Z][A-Za-z'\-\.]{1,}$")

TITLE_HINT_WORDS = {
    "optimization",
    "programming",
    "stability",
    "analysis",
    "theory",
    "algorithm",
    "method",
    "methods",
    "model",
    "models",
    "variational",
    "robust",
    "convex",
    "linear",
    "nonlinear",
    "stochastic",
    "equilibrium",
    "distance",
    "continuity",
}
TITLE_LINK_WORDS = {"of", "in", "for", "with", "to", "on", "under", "via"}
VENUE_HINT_WORDS = {
    "journal",
    "proc",
    "proceedings",
    "conference",
    "symposium",
    "workshop",
    "transactions",
    "springer",
    "wiley",
    "elsevier",
    "press",
    "university",
    "cambridge",
    "oxford",
    "berlin",
    "london",
    "new york",
    "amsterdam",
    "siam",
    "ieee",
    "acm",
    "lecture notes",
    "math program",
    "oper res",
    "eur j",
    "ann oper",
    "vol",
    "volume",
    "pp",
    "pages",
    "ed",
    "edition",
}
URL_RE = re.compile(r"(https?://|www\.|\.com\b|\.org\b|\.edu\b)", re.IGNORECASE)
VENUE_HINT_TOKEN_RE = re.compile(
    r"\b(journal|proc|proceedings|conference|symposium|workshop|transactions|springer|wiley|elsevier|press|university|cambridge|oxford|berlin|london|amsterdam|siam|ieee|acm|vol|volume|pp|pages|no|issue|ed|edition)\b",
    re.IGNORECASE,
)
JOURNAL_SINGLE_WORDS = {
    "optimization",
    "programming",
    "research",
    "operations",
    "science",
    "review",
    "letters",
    "computing",
    "economics",
}


def remove_leading_index(text: str) -> str:
    """Remove leading citation index markers."""
    return normalize_space(LEADING_INDEX_RE.sub("", text or ""))


def pick_year(raw: str) -> str:
    """Pick likely publication year (prefer the last year token)."""
    years = extract_years(raw)
    if not years:
        return ""
    return years[-1]


def split_by_last_year(text: str) -> Tuple[str, str]:
    """
    Split citation text by the last year token.
    Returns:
    - pre_year: text before the last detected year
    - post_year: text after that year
    """
    matches = list(YEAR_RE.finditer(text or ""))
    if not matches:
        return text, ""
    last = matches[-1]
    return normalize_space(text[: last.start()]), normalize_space(text[last.end() :])


def _word_tokens(seg: str) -> List[str]:
    """Extract word-like tokens while preserving unicode letters."""
    out: List[str] = []
    for raw in re.split(r"\s+", seg or ""):
        t = raw.strip(" ,;:()[]{}\"")
        if not t:
            continue
        if any(ch.isalpha() for ch in t):
            out.append(t)
    return out


def _token_core(token: str) -> str:
    """Normalized token core for lexical checks."""
    return token.strip(" .,:;()[]{}\"'")


def _is_initial_token(token: str) -> bool:
    """
    Detect initial-like token:
    - A.
    - J
    - J.B.
    - J.-B.
    """
    t = _token_core(token)
    if not t:
        return False
    if len(t) == 1 and t.isalpha() and t.isupper():
        return True
    return bool(re.fullmatch(r"[A-Z](?:\.?[A-Z]|\.?-[A-Z]){0,4}\.?", t))


def _ends_with_initial_chain(text: str) -> bool:
    """Check whether text ends with one or more initial-like tokens."""
    toks = [_token_core(t) for t in normalize_space(text).split() if _token_core(t)]
    if not toks:
        return False
    n = 0
    for tok in reversed(toks):
        if _is_initial_token(tok):
            n += 1
            continue
        break
    return n >= 1


def _starts_with_surname_then_comma(text: str) -> bool:
    """
    Detect leading surname-like phrase followed by comma, e.g.:
    'WETS, Quantitative ...' or 'PESQUET, Proximal ...'
    """
    t = normalize_space(text)
    return bool(re.match(r"^[A-Z][A-Za-z'\-]{1,30}(?:\s+[A-Z][A-Za-z'\-]{1,30}){0,2}\s*,\s+", t))


def fix_author_tail_leak(authors: str, title: str, year: str) -> Tuple[str, str]:
    """
    Repair boundary leak where title starts with a co-author surname, e.g.:
    authors='R. J.-B' + title='WETS, Variational Analysis' ->
    authors='R. J.-B. WETS' + title='Variational Analysis'
    """
    a = normalize_space(authors).strip(" ,;:.")
    t = normalize_space(title).strip(" ,;:.")
    if not a or not t:
        return a, t
    if not _ends_with_initial_chain(a):
        return a, t
    m = re.match(r"^([A-Z][A-Za-z'\-]{1,30}(?:\s+[A-Z][A-Za-z'\-]{1,30}){0,2})\s*,\s*(.+)$", t)
    if not m:
        return a, t
    surname = normalize_space(m.group(1))
    rest = normalize_space(m.group(2))
    if len(_word_tokens(rest)) < 2:
        return a, t
    if re.search(rf"\b{re.escape(surname)}\b$", a, flags=re.IGNORECASE):
        return a, clean_title_text(rest, year)
    if not a.endswith("."):
        a = a + "."
    fixed_authors = normalize_space(f"{a} {surname}")
    fixed_title = clean_title_text(rest, year)
    return fixed_authors, fixed_title


def is_author_like_segment(seg: str) -> bool:
    """
    Heuristic to detect author segment.
    Works for patterns like:
    - A. Ben-Tal and A. Nemirovski
    - J. Cánovas
    - R. J.-B. Wets
    """
    s = normalize_space(seg).strip(" ,;:.")
    if not s:
        return False
    if URL_RE.search(s):
        return False
    if is_venue_like_segment(s):
        return False

    words = _word_tokens(s)
    if len(words) < 2 or len(words) > 14:
        return False

    lower_words = [_token_core(w).lower() for w in words]
    if any(w in TITLE_LINK_WORDS for w in lower_words if w != "and"):
        return False
    if any(w in TITLE_HINT_WORDS for w in lower_words):
        return False
    if any(w in {"proc", "proceedings", "journal", "conference", "symposium"} for w in lower_words):
        return False

    initial_count = 0
    name_like = 0
    lexical_count = 0
    for w in words:
        core = _token_core(w)
        if not core:
            continue
        lw = core.lower()
        if lw in {"and", "&", "et", "al", "de", "van", "von", "da", "del", "der", "la", "le"}:
            continue
        lexical_count += 1
        if _is_initial_token(core):
            initial_count += 1
            name_like += 1
            continue
        if NAME_WORD_RE.match(core):
            name_like += 1
            continue
        if core.isupper() and len(core) > 1:
            name_like += 1
            continue
        if core[:1].isupper() and len(core) <= 18 and any(ch.isalpha() for ch in core):
            name_like += 1
            continue

    if lexical_count == 0:
        return False
    # Long phrase with no initials is usually title/venue, not author list.
    if initial_count == 0 and lexical_count >= 4:
        return False

    ratio = name_like / lexical_count
    return ratio >= 0.75


def is_title_like_segment(seg: str) -> bool:
    """Heuristic to detect title segment from comma-separated parts."""
    s = normalize_space(seg).strip(" ,;:.")
    if not s:
        return False
    if URL_RE.search(s):
        return False
    if is_venue_like_segment(s):
        return False
    words = _word_tokens(s)
    if len(words) < 2:
        if len(words) == 1 and words[0].lower().strip(".") in TITLE_HINT_WORDS:
            return True
        return False
    low = [w.lower().strip(".") for w in words]

    if any(w in TITLE_HINT_WORDS for w in low):
        return True
    if any(w in TITLE_LINK_WORDS for w in low):
        return True

    initials = sum(1 for w in words if INITIAL_RE.match(w))
    if initials / max(len(words), 1) > 0.35:
        return False
    return True


def is_venue_like_segment(seg: str) -> bool:
    """Heuristic to detect venue/publisher/location segment."""
    s = normalize_space(seg).strip(" ,;:.")
    if not s:
        return False
    low = s.lower()
    if URL_RE.search(s):
        return True
    if low.startswith("in "):
        return True
    if re.search(r"\b(vol|volume|pp|pages|no|issue|ed|edition)\b", low):
        return True
    if VENUE_HINT_TOKEN_RE.search(low):
        return True
    return any(
        phrase in low
        for phrase in ("new york", "lecture notes", "math program", "oper res", "eur j", "ann oper")
    )


def split_author_dot_prefix(text: str) -> Tuple[str, str]:
    """
    Try to split 'Author. Title ...' safely, avoiding splits on initials.
    Returns (authors, rest). Empty authors means no confident split.
    """
    t = normalize_space(text)
    if not t:
        return "", t
    candidates = []
    for m in re.finditer(r"\.\s+", t[:220]):
        pre = normalize_space(t[: m.start()]).strip(" ,;:.")
        post = normalize_space(t[m.end() :]).strip(" ,;:.")
        if not pre or not post:
            continue
        # Avoid splitting after initials such as "A. W." / "R. J. B."
        if re.search(r"(?:^|\s)[A-Z]$", pre):
            continue
        # Avoid splitting inside author name tail: "... R. J.-B. WETS, ..."
        if _ends_with_initial_chain(pre) and _starts_with_surname_then_comma(post):
            continue
        if is_author_like_segment(pre):
            # post should not look like another pure author list
            post_head = normalize_space(post.split(",")[0])
            if not is_author_like_segment(post_head):
                candidates.append((pre, post))
    if not candidates:
        return "", t
    # choose the last valid boundary to include initials fully.
    return candidates[-1]


def split_author_no_comma(seg: str) -> Tuple[str, str]:
    """
    Split patterns like:
    'B. MORDUKHOVICH Variational Analysis ...'
    into (author, title+venue).
    """
    s = normalize_space(seg).strip(" ,;:.")
    if not s or "," in s:
        return "", s
    m = re.match(
        r"^((?:[A-Z]\.\s*){1,4}[A-Z][A-Za-z'\-]+(?:\s+AND\s+(?:[A-Z]\.\s*){0,4}[A-Z][A-Za-z'\-]+)?)\s+(.+)$",
        s,
    )
    if not m:
        return "", s
    a = normalize_space(m.group(1))
    rest = normalize_space(m.group(2))
    # False split guard: author list continuation, not title start.
    if re.match(r"^(and|&)\b", rest, flags=re.IGNORECASE):
        return "", s
    if is_author_like_segment(a) and is_title_like_segment(rest):
        return a, rest
    return "", s


def is_surname_segment(seg: str) -> bool:
    """
    Detect surname-like segment used by styles such as:
    'Ben-Tal, A., El Ghaoui, L., Nemirovsky, A.: Title ...'
    """
    s = normalize_space(seg).strip(" ,;:.")
    if not s:
        return False
    if URL_RE.search(s):
        return False
    # Surname chunk should not already include initials.
    if re.search(r"(?:^|\s)[A-Z]\.", s):
        return False
    words = _word_tokens(s)
    if not (1 <= len(words) <= 3):
        return False
    low = [w.lower().strip(".") for w in words]
    if any(w in TITLE_HINT_WORDS or w in TITLE_LINK_WORDS for w in low):
        return False
    # At least one token should start uppercase.
    if not any(w[:1].isupper() for w in words):
        return False
    return True


def extract_initial_and_tail(seg: str) -> Tuple[str, str]:
    """Extract initial block and tail text from a segment like 'A.: Title'."""
    s = normalize_space(seg).strip(" ,;:")
    if not s:
        return "", ""
    i = 0
    letters: List[str] = []
    while i < len(s):
        ch = s[i]
        if ch.isspace() or ch == "-":
            i += 1
            continue
        if ch.isalpha() and ch.isupper():
            letters.append(ch)
            i += 1
            if i < len(s) and s[i] == ".":
                i += 1
            continue
        break
    if not (1 <= len(letters) <= 6):
        return "", ""

    rest = s[i:].lstrip()
    # Reject false split like "Moments" -> "M" + "oments"
    if rest and rest[0].islower():
        return "", ""
    if rest.startswith(":"):
        rest = rest[1:].lstrip()

    init = ". ".join(letters) + "."
    tail = normalize_space(rest).strip(" ,;:.")
    return init, tail


def consume_surname_initial_segments(segments: List[str]) -> Tuple[List[str], List[str]]:
    """
    Consume leading surname/initial pairs and return:
    - normalized author pieces: ['Ben-Tal, A.', 'El Ghaoui, L.']
    - remaining segments where title starts
    """
    if len(segments) < 2:
        return [], segments

    i = 0
    authors: List[str] = []
    title_seed = ""
    while i + 1 < len(segments):
        sname = segments[i]
        sinit = segments[i + 1]
        if not is_surname_segment(sname):
            break
        init, tail = extract_initial_and_tail(sinit)
        if not init:
            break
        authors.append(f"{sname}, {init}")
        i += 2
        if tail:
            if is_title_like_segment(tail) or len(_word_tokens(tail)) >= 2:
                title_seed = tail
            break

    if not authors:
        return [], segments

    remaining = []
    if title_seed:
        remaining.append(title_seed)
    remaining.extend(segments[i:])
    return authors, remaining


def clean_title_text(title: str, year: str) -> str:
    """Normalize and cleanup title text."""
    title = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", "", title, flags=re.IGNORECASE)
    title = TAIL_VENUE_RE.sub("", title)
    if year:
        title = re.sub(rf"\b{re.escape(year)}\b.*$", "", title)
    title = normalize_space(title).strip(" ,;:.")
    return title


def parse_author_title_venue(citation_no_idx: str, year: str) -> Tuple[str, str, str]:
    """
    Parse author/title/venue using:
    1) quoted title extraction (if any)
    2) year-based split
    3) comma-segment heuristics
    """
    text = normalize_space(citation_no_idx)
    if not text:
        return "", "", ""

    authors_dot, text_after_dot = split_author_dot_prefix(text)
    if authors_dot:
        text = text_after_dot

    q = QUOTE_TITLE_RE.search(text)
    if q:
        title = clean_title_text(normalize_space(q.group(1)), year)
        left = normalize_space(text[: q.start()]).strip(" ,;:.")
        right = normalize_space(text[q.end() :]).strip(" ,;:.")
        authors = authors_dot or left.rstrip(",")
        venue = right
        return authors, title, venue

    pre_year, post_year = split_by_last_year(text) if year else (text, "")
    segments = [normalize_space(s).strip(" ,;:.") for s in SEG_SPLIT_RE.split(pre_year) if normalize_space(s).strip(" ,;:.")]
    if not segments:
        return authors_dot, "", normalize_space(post_year)

    # Handle no-comma author-title pattern in the first segment.
    if segments and not authors_dot:
        a_nc, rest_nc = split_author_no_comma(segments[0])
        if a_nc:
            authors_dot = a_nc
            segments = [rest_nc] + segments[1:]

    # Handle surname-initial citation style.
    authors_pair, segments = consume_surname_initial_segments(segments)
    if authors_pair:
        if authors_dot:
            authors_dot = normalize_space(authors_dot + ", " + ", ".join(authors_pair))
        else:
            authors_dot = ", ".join(authors_pair)
        # Sometimes title seed also contains first venue sentence after a dot.
        if segments and ". " in segments[0]:
            first, rest = segments[0].split(". ", 1)
            first = normalize_space(first).strip(" ,;:.")
            rest = normalize_space(rest).strip(" ,;:.")
            if first and rest and is_title_like_segment(first):
                segments = [first, rest] + segments[1:]

    title_idx = None
    author_parts: List[str] = []
    author_end = 0
    if authors_dot:
        author_parts = [authors_dot]
    else:
        k = 0
        while k < len(segments) and is_author_like_segment(segments[k]):
            author_parts.append(segments[k])
            k += 1
        author_end = k

    for i, seg in enumerate(segments):
        if i < author_end and not authors_dot:
            continue
        if i == 0 and len(segments) >= 2 and not authors_dot:
            # First part can still be author list; avoid picking too early.
            if is_author_like_segment(seg):
                continue
        # One-word title prefix case: "Moments, Positive Polynomials ..."
        if (
            len(_word_tokens(seg)) == 1
            and i + 1 < len(segments)
            and is_title_like_segment(segments[i + 1])
            and not is_venue_like_segment(segments[i + 1])
        ):
            title_idx = i
            break
        if is_title_like_segment(seg) and not is_author_like_segment(seg) and not is_venue_like_segment(seg):
            title_idx = i
            break

    if title_idx is None:
        guess = len(author_parts) if len(author_parts) < len(segments) else 0
        if guess + 1 < len(segments):
            # If first candidate is one-word and next segment looks like title, include both.
            if len(_word_tokens(segments[guess])) == 1 and is_title_like_segment(segments[guess + 1]):
                title_idx = guess
            else:
                title_idx = guess if not is_venue_like_segment(segments[guess]) else min(guess + 1, len(segments) - 1)
        else:
            title_idx = min(guess, len(segments) - 1)

    # Special fallback: if title landed on first segment but second one looks like title.
    if title_idx == 0 and len(segments) >= 2 and (not authors_dot) and is_title_like_segment(segments[1]):
        title_idx = 1

    if not authors_dot and not author_parts:
        author_parts = segments[:title_idx] if title_idx > 0 else author_parts

    # Build title from contiguous title-like segments, stop when venue-like starts.
    tparts: List[str] = []
    j = title_idx
    while 0 <= j < len(segments):
        seg = segments[j]
        if not tparts:
            if URL_RE.search(seg) and j + 1 < len(segments):
                j += 1
                continue
            tparts.append(seg)
            j += 1
            continue
        if is_venue_like_segment(seg):
            break
        if (
            tparts
            and len(_word_tokens(seg)) <= 2
            and _token_core(seg).lower() in JOURNAL_SINGLE_WORDS
            and j + 1 < len(segments)
            and re.search(r"\b\d", segments[j + 1])
        ):
            # Prefer assigning journal name to venue when volume/pages follow.
            break
        if is_title_like_segment(seg) and not is_author_like_segment(seg):
            tparts.append(seg)
            j += 1
            continue
        break
    if not tparts and 0 <= title_idx < len(segments):
        tparts = [segments[title_idx]]
        j = title_idx + 1
    title_part = normalize_space(", ".join(tparts))
    venue_parts = segments[j:] if j < len(segments) else []

    authors = normalize_space(", ".join(author_parts)).strip(" ,;:.")
    title = clean_title_text(title_part, year)
    venue = normalize_space(", ".join(venue_parts + ([post_year] if post_year else []))).strip(" ,;:.")
    authors, title = fix_author_tail_leak(authors, title, year)

    # Fallback when author extraction is empty but first segment likely author.
    if not authors and len(segments) >= 2 and is_author_like_segment(segments[0]):
        authors = segments[0]
        if title_idx == 0:
            title = clean_title_text(segments[1], year)
            venue = normalize_space(", ".join(segments[2:] + ([post_year] if post_year else []))).strip(" ,;:.")
    authors, title = fix_author_tail_leak(authors, title, year)

    return authors, title, venue


def parse_one_citation(raw_citation: str) -> Dict[str, str]:
    """Parse one raw citation into structured fields."""
    raw = normalize_space(raw_citation)
    raw_no_idx = remove_leading_index(raw)
    raw_no_idx = re.sub(r"^\s*\$[^$]{0,120}\$\s*", "", raw_no_idx).strip()
    raw_no_idx = re.sub(r"^\s*[\.,;:]+\s*", "", raw_no_idx)
    doi = extract_doi(raw_no_idx)
    year = pick_year(raw_no_idx)
    text_wo_doi = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", "", raw_no_idx, flags=re.IGNORECASE)
    authors, title, venue = parse_author_title_venue(text_wo_doi, year)

    if not title:
        fallback = normalize_space(text_wo_doi)
        if year:
            fallback = re.sub(rf"\b{re.escape(year)}\b", "", fallback)
        parts = [normalize_space(p).strip(" ,;:.") for p in SEG_SPLIT_RE.split(fallback) if normalize_space(p).strip(" ,;:.")]
        if len(parts) >= 2:
            if not authors:
                authors = parts[0]
            title = clean_title_text(parts[1], year)
            if not venue:
                venue = normalize_space(", ".join(parts[2:])).strip(" ,;:.")
        elif parts:
            title = clean_title_text(parts[0], year)

    return {
        "parsed_authors": authors,
        "parsed_title": title,
        "parsed_year": year,
        "parsed_venue": venue,
        "parsed_doi": doi,
        "first_author_norm": first_author_key(authors),
        "title_norm": normalize_title(title),
    }


def parse_one_citation_grobid(raw_citation: str, grobid_url: str, timeout_sec: int) -> Dict[str, str]:
    """Parse one citation string using GROBID /api/processCitation."""
    # Many GROBID setups (e.g. 0.8.0) are more reliable on processCitationList
    # than processCitation; prefer list endpoint even for one citation.
    try:
        one = parse_citation_list_grobid([raw_citation], grobid_url, timeout_sec)
        if one:
            return one[0]
    except Exception:
        pass

    endpoint = grobid_url.rstrip("/") + "/api/processCitation"
    attempts = []
    xml_candidates: List[str] = []
    success_count = 0

    for mode in ("form", "plain", "multipart"):
        try:
            if mode == "form":
                xml_text = grobid_post_form_urlencoded(
                    endpoint_url=endpoint,
                    fields={"citation": raw_citation},
                    timeout_sec=timeout_sec,
                    accept="application/xml",
                )
            elif mode == "plain":
                xml_text = grobid_post_text_plain(
                    endpoint_url=endpoint,
                    text=raw_citation,
                    timeout_sec=timeout_sec,
                    accept="application/xml",
                )
            else:
                xml_text = grobid_post_multipart(
                    endpoint_url=endpoint,
                    fields={"citation": raw_citation},
                    timeout_sec=timeout_sec,
                    accept="application/xml",
                )
            success_count += 1
            xml_candidates.append(xml_text)
        except Exception as e:
            attempts.append(f"{mode}:{e}")

    for xml_text in xml_candidates:
        try:
            nodes = extract_tei_bibl_nodes(xml_text)
        except ET.ParseError:
            nodes = []
        if nodes:
            return citation_struct_from_bibl_node(nodes[0])

    # Endpoint reachable but this citation produced no parseable TEI.
    if success_count > 0:
        return {}
    if attempts:
        raise RuntimeError(f"GROBID processCitation failed across content types: {' | '.join(attempts[:3])}")
    return {}


def parse_citation_list_grobid(raw_citations: List[str], grobid_url: str, timeout_sec: int) -> List[Dict[str, str]]:
    """Parse batch citation strings using GROBID /api/processCitationList."""
    if not raw_citations:
        return []
    endpoint = grobid_url.rstrip("/") + "/api/processCitationList"
    payload = "\n".join(raw_citations)
    attempts = []
    xml_candidates: List[str] = []
    success_count = 0

    for mode in ("form", "plain", "multipart"):
        try:
            if mode == "form":
                xml_text = grobid_post_form_urlencoded(
                    endpoint_url=endpoint,
                    fields={"citations": payload},
                    timeout_sec=timeout_sec,
                    accept="application/xml",
                )
            elif mode == "plain":
                xml_text = grobid_post_text_plain(
                    endpoint_url=endpoint,
                    text=payload,
                    timeout_sec=timeout_sec,
                    accept="application/xml",
                )
            else:
                xml_text = grobid_post_multipart(
                    endpoint_url=endpoint,
                    fields={"citations": payload},
                    timeout_sec=timeout_sec,
                    accept="application/xml",
                )
            success_count += 1
            xml_candidates.append(xml_text)
        except Exception as e:
            attempts.append(f"{mode}:{e}")

    for xml_text in xml_candidates:
        try:
            nodes = extract_tei_bibl_nodes(xml_text)
        except ET.ParseError:
            nodes = []
        if nodes:
            return [citation_struct_from_bibl_node(n) for n in nodes]

    # Endpoint reachable but returned no parsable bibl nodes for this payload.
    if success_count > 0:
        return []
    if attempts:
        raise RuntimeError(f"GROBID processCitationList failed across content types: {' | '.join(attempts[:3])}")
    return []


def batched(items: Sequence[Dict[str, str]], batch_size: int) -> Iterable[List[Dict[str, str]]]:
    """Yield chunks with fixed max size."""
    n = max(1, int(batch_size))
    for i in range(0, len(items), n):
        yield list(items[i : i + n])


def stage_parse_citations(
    raw_citations_csv: Path,
    parsed_citations_csv: Path,
    parse_source: str = "qwen_api",
    grobid_url: str = "http://localhost:8070",
    grobid_timeout_sec: int = 120,
    grobid_batch_size: int = 32,
    llm_base_url: str = "https://llmmelon.cloud",
    llm_api_key: str = "",
    llm_model: str = "qwen-3.5-flash",
    llm_timeout_sec: int = 120,
    llm_batch_size: int = 20,
) -> int:
    """Run stage 2 and output parsed citations (CSV/JSON)."""
    rows = read_raw_citation_rows(raw_citations_csv)
    writer = IncrementalParsedCitationWriter(
        parsed_citations_csv,
        meta={
            "stage": "parse",
            "parse_source": parse_source,
            "raw_input": str(raw_citations_csv),
        },
    )

    qwen_fail_chunks = 0
    qwen_total_chunks = 0
    first_qwen_err: Optional[str] = None

    def _emit(row: Dict[str, str], parsed: Dict[str, str]) -> None:
        out_row = {
            "source_paper_id": row.get("source_paper_id", ""),
            "source_md_path": row.get("source_md_path", ""),
            "ref_idx": row.get("ref_idx", ""),
            "raw_citation": row.get("raw_citation", ""),
            **parsed,
        }
        writer.write_row(out_row)

    try:
        if parse_source == "heuristic":
            for row in rows:
                parsed = parse_one_citation(row.get("raw_citation", ""))
                _emit(row, parsed)
        elif parse_source in {"grobid_api", "auto"}:
            grobid_available = True
            for chunk in batched(rows, grobid_batch_size):
                raw_list = [r.get("raw_citation", "") for r in chunk]
                parsed_chunk: List[Dict[str, str]] = []

                if grobid_available:
                    try:
                        parsed_chunk = parse_citation_list_grobid(raw_list, grobid_url, grobid_timeout_sec)
                    except Exception as e:
                        parsed_chunk = []
                        if parse_source == "grobid_api":
                            # In pure-GROBID mode, endpoint failures should be explicit.
                            raise RuntimeError("GROBID list parsing failed in pure grobid_api mode.") from e

                # GROBID list parsing can occasionally return fewer items; fallback per entry.
                if len(parsed_chunk) != len(chunk):
                    parsed_chunk = []
                    first_probe_done = False
                    for raw in raw_list:
                        p: Dict[str, str] = {}
                        try:
                            if grobid_available:
                                p = parse_one_citation_grobid(raw, grobid_url, grobid_timeout_sec)
                                first_probe_done = True
                        except Exception:
                            p = {}
                            # If first single-citation request already fails, stop calling GROBID repeatedly.
                            if not first_probe_done:
                                grobid_available = False
                                if parse_source == "grobid_api":
                                    raise RuntimeError("GROBID single-citation parsing failed in pure grobid_api mode.")
                        parsed_chunk.append(p)
                    if not grobid_available and parsed_chunk:
                        parsed_chunk = [{} for _ in chunk]

                for row, p_primary in zip(chunk, parsed_chunk):
                    if parse_source == "auto":
                        p_heuristic = parse_one_citation(row.get("raw_citation", ""))
                        parsed = merge_parsed_fields(p_primary, p_heuristic)
                    else:  # grobid_api
                        parsed = merge_parsed_fields(p_primary, {})
                    _emit(row, parsed)
        else:  # qwen_api
            for chunk in batched(rows, llm_batch_size):
                qwen_total_chunks += 1
                raw_list = [r.get("raw_citation", "") for r in chunk]
                try:
                    parsed_chunk = parse_citation_list_qwen(
                        raw_citations=raw_list,
                        llm_base_url=llm_base_url,
                        llm_api_key=llm_api_key,
                        llm_model=llm_model,
                        llm_timeout_sec=llm_timeout_sec,
                    )
                except Exception as e:
                    qwen_fail_chunks += 1
                    if first_qwen_err is None:
                        first_qwen_err = str(e)
                    parsed_chunk = [{} for _ in chunk]

                for row, p_primary in zip(chunk, parsed_chunk):
                    parsed = merge_parsed_fields(p_primary, {})
                    _emit(row, parsed)
    finally:
        writer.close(
            final_meta={
                "num_rows": writer.count,
            }
        )

    if parse_source == "qwen_api" and qwen_total_chunks > 0 and qwen_fail_chunks == qwen_total_chunks:
        msg = first_qwen_err or "unknown qwen_api error"
        raise RuntimeError(
            "All Qwen parse requests failed; no API calls succeeded. "
            f"First error: {msg}"
        )

    return writer.count


# =========================
# Stage: OCR markdown -> one summary JSON (Qwen New API)
# =========================


def _iter_markdown_files(corpus_root: Path, include_dirs: Optional[List[str]] = None) -> List[Path]:
    """Collect markdown files from OCR root (optionally limited to subdirs)."""
    files: List[Path] = []
    roots: List[Path] = []
    if include_dirs:
        for d in include_dirs:
            p = (corpus_root / d).resolve()
            if p.exists() and p.is_dir():
                roots.append(p)
    else:
        roots = [corpus_root.resolve()]

    for root in roots:
        files.extend([p for p in root.rglob("*.md") if p.is_file()])
    files = sorted(set(files), key=lambda x: str(x).lower())
    return files


def _relative_posix(path: Path, root: Path) -> str:
    """Convert path to root-relative POSIX style string."""
    try:
        rel = path.resolve().relative_to(root.resolve())
        return str(rel).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def stage_ocr_markdown_to_json(
    corpus_root: Path,
    output_json: Path,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    llm_timeout_sec: int,
    include_dirs: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Read OCR markdown files, extract raw citations, parse with Qwen New API,
    and write one merged JSON output.
    """
    corpus_root = corpus_root.resolve()
    md_files = _iter_markdown_files(corpus_root, include_dirs=include_dirs)

    papers_out: List[Dict[str, object]] = []
    total_citations = 0
    paper_seq = 0

    for md_path in md_files:
        md_text = md_path.read_text(encoding="utf-8", errors="replace")
        try:
            raw_entries = extract_raw_citations_qwen(
                markdown_text=md_text,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                llm_timeout_sec=llm_timeout_sec,
            )
        except Exception:
            raw_entries = []
        if not raw_entries:
            continue

        paper_seq += 1
        source_rel = _relative_posix(md_path, corpus_root)
        paper_id = f"paper_ocr_{paper_seq:06d}"
        citations_out: List[Dict[str, object]] = []

        for idx, raw in enumerate(raw_entries, start=1):
            parsed_primary: Dict[str, str] = {}
            try:
                one = parse_citation_list_qwen(
                    raw_citations=[raw],
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    llm_timeout_sec=llm_timeout_sec,
                )
                parsed_primary = one[0] if one else {}
            except Exception:
                parsed_primary = {}

            parsed = merge_parsed_fields(parsed_primary, {})
            citations_out.append(
                {
                    "ref_idx": idx,
                    "raw_citation": raw,
                    "parsed_authors": parsed.get("parsed_authors", ""),
                    "parsed_title": parsed.get("parsed_title", ""),
                    "parsed_year": parsed.get("parsed_year", ""),
                    "parsed_doi": parsed.get("parsed_doi", ""),
                    "first_author_norm": parsed.get("first_author_norm", ""),
                    "title_norm": parsed.get("title_norm", ""),
                }
            )
            total_citations += 1

        papers_out.append(
            {
                "paper_id": paper_id,
                "source_md_path": source_rel,
                "num_citations": len(citations_out),
                "citations": citations_out,
            }
        )

    out_obj = {
        "meta": {
            "stage": "ocr_json",
            "corpus_root": str(corpus_root),
            "llm_base_url": llm_base_url,
            "llm_model": llm_model,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_papers": len(papers_out),
            "num_citations": total_citations,
        },
        "papers": papers_out,
    }
    write_json(output_json, out_obj)
    return {"num_papers": len(papers_out), "num_citations": total_citations}


# =========================
# Stage 3: library matching
# =========================


def build_indexes_from_papers(papers: List[Dict[str, str]]) -> Tuple[dict, dict, dict]:
    """Build DOI/title/author+year index maps from papers.csv."""
    doi_index: Dict[str, List[str]] = {}
    title_index: Dict[str, List[str]] = {}
    author_year_index: Dict[str, List[str]] = {}

    for p in papers:
        pid = p.get("paper_id", "")
        if not pid:
            continue
        doi = (p.get("doi") or "").strip().lower()
        if doi:
            doi_index.setdefault(doi, []).append(pid)
        title_norm = p.get("title_norm", "") or normalize_title(p.get("title", ""))
        if title_norm:
            title_index.setdefault(title_norm, []).append(pid)
        author_key = p.get("first_author_norm", "").strip()
        year = (p.get("year") or "").strip()
        if author_key and year:
            author_year_index.setdefault(f"{author_key}|{year}", []).append(pid)

    return doi_index, title_index, author_year_index


def load_or_build_indexes(paper_index_dir: Path, papers: List[Dict[str, str]]) -> Tuple[dict, dict, dict]:
    """Load prebuilt paper indexes or fallback to in-memory build."""
    doi_path = paper_index_dir / "paper_index_doi.json"
    title_path = paper_index_dir / "paper_index_title.json"
    author_year_path = paper_index_dir / "paper_index_author_year.json"
    if doi_path.exists() and title_path.exists() and author_year_path.exists():
        return read_json(doi_path), read_json(title_path), read_json(author_year_path)
    return build_indexes_from_papers(papers)


def build_title_lookup(papers: List[Dict[str, str]]) -> Dict[str, str]:
    """Build paper_id -> normalized title map."""
    out = {}
    for p in papers:
        pid = p.get("paper_id", "")
        if not pid:
            continue
        out[pid] = p.get("title_norm", "") or normalize_title(p.get("title", ""))
    return out


def match_by_doi(parsed_doi: str, doi_index: dict) -> Tuple[str, str, float, str, int]:
    """Match by DOI (highest priority)."""
    doi = (parsed_doi or "").strip().lower()
    if not doi:
        return "", "", 0.0, "", 0
    candidates = doi_index.get(doi, [])
    if not candidates:
        return "", "", 0.0, "", 0
    if len(candidates) == 1:
        return candidates[0], "doi_exact", 1.0, "matched", 1
    return candidates[0], "doi_multi", 0.96, "ambiguous", len(candidates)


def match_by_title(parsed_title: str, title_index: dict, paper_titles: Dict[str, str]) -> Tuple[str, str, float, str, int, float]:
    """Match by exact/fuzzy title."""
    title_norm = normalize_title(parsed_title)
    if not title_norm:
        return "", "", 0.0, "", 0, 0.0

    exact = title_index.get(title_norm, [])
    if len(exact) == 1:
        return exact[0], "title_exact", 0.95, "matched", 1, 0.0
    if len(exact) > 1:
        return exact[0], "title_exact_multi", 0.86, "ambiguous", len(exact), 0.0

    scored = []
    for pid, lib_title in paper_titles.items():
        if not lib_title:
            continue
        s = title_similarity(title_norm, lib_title)
        scored.append((pid, s))
    if not scored:
        return "", "", 0.0, "", 0, 0.0
    scored.sort(key=lambda x: x[1], reverse=True)
    best_pid, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    margin = best_score - second_score

    if best_score >= 0.93 and margin >= 0.03:
        return best_pid, "title_fuzzy", round(best_score, 4), "matched", 1, round(second_score, 4)
    if best_score >= 0.83:
        return best_pid, "title_fuzzy", round(best_score, 4), "ambiguous", 1, round(second_score, 4)
    return "", "", round(best_score, 4), "", 0, round(second_score, 4)


def match_by_author_year(first_author_norm: str, year: str, author_year_index: dict) -> Tuple[str, str, float, str, int]:
    """Match by (first author + year) fallback."""
    a = (first_author_norm or "").strip().lower()
    y = (year or "").strip()
    if not a or not y:
        return "", "", 0.0, "", 0
    key = f"{a}|{y}"
    candidates = author_year_index.get(key, [])
    if not candidates:
        return "", "", 0.0, "", 0
    if len(candidates) == 1:
        return candidates[0], "author_year", 0.78, "matched", 1
    return candidates[0], "author_year_multi", 0.66, "ambiguous", len(candidates)


def stage_match_citations(
    parsed_citations_csv: Path,
    papers_csv: Path,
    paper_index_dir: Path,
    citation_matches_csv: Path,
) -> int:
    """Run stage 3 and output citation_matches.csv."""
    parsed_rows = read_parsed_citation_rows(parsed_citations_csv)
    papers = read_csv(papers_csv)
    doi_index, title_index, author_year_index = load_or_build_indexes(paper_index_dir, papers)
    paper_titles = build_title_lookup(papers)

    out_rows = []
    for row in parsed_rows:
        source_paper_id = row.get("source_paper_id", "")
        ref_idx = row.get("ref_idx", "")
        parsed_title = row.get("parsed_title", "")
        parsed_year = row.get("parsed_year", "")
        parsed_doi = (row.get("parsed_doi", "") or "").lower()
        first_author_norm = row.get("first_author_norm", "")

        target_id = ""
        method = ""
        score = 0.0
        status = ""
        candidate_count = 0
        top2_score = 0.0

        target_id, method, score, status, candidate_count = match_by_doi(parsed_doi, doi_index)
        if not status:
            target_id, method, score, status, candidate_count, top2_score = match_by_title(
                parsed_title, title_index, paper_titles
            )
        if not status:
            target_id, method, score, status, candidate_count = match_by_author_year(
                first_author_norm, parsed_year, author_year_index
            )
        if not status:
            status = "unmatched"
            method = "none"
            score = 0.0
            target_id = ""
            candidate_count = 0

        out_rows.append(
            {
                "source_paper_id": source_paper_id,
                "ref_idx": ref_idx,
                "parsed_title": parsed_title,
                "parsed_year": parsed_year,
                "parsed_doi": parsed_doi,
                "matched_target_paper_id": target_id,
                "match_method": method,
                "match_score": round(float(score), 4),
                "match_status": status,
                "candidate_count": candidate_count,
                "top2_score": round(float(top2_score), 4),
            }
        )

    write_csv(
        citation_matches_csv,
        out_rows,
        fieldnames=[
            "source_paper_id",
            "ref_idx",
            "parsed_title",
            "parsed_year",
            "parsed_doi",
            "matched_target_paper_id",
            "match_method",
            "match_score",
            "match_status",
            "candidate_count",
            "top2_score",
        ],
    )
    return len(out_rows)


# =========================
# Stage 4: build edges + graph
# =========================


def load_paper_maps(papers_csv: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Build paper maps for graph attributes and QA diagnostics."""
    papers = read_csv(papers_csv)
    by_id = {p.get("paper_id", ""): p for p in papers if p.get("paper_id", "")}

    doi_index: Dict[str, List[str]] = {}
    title_index: Dict[str, List[str]] = {}
    for p in papers:
        pid = p.get("paper_id", "")
        if not pid:
            continue
        doi = (p.get("doi") or "").strip().lower()
        if doi:
            doi_index.setdefault(doi, []).append(pid)
        title_norm = p.get("title_norm", "") or normalize_title(p.get("title", ""))
        if title_norm:
            title_index.setdefault(title_norm, []).append(pid)
    return by_id, doi_index, title_index


def build_clean_edges(matches: List[Dict[str, str]], min_match_score: float) -> List[Dict[str, object]]:
    """Create edge table from matched rows and clean loops/duplicates/low score."""
    kept = []
    for m in matches:
        if (m.get("match_status") or "").strip().lower() != "matched":
            continue
        src = m.get("source_paper_id", "")
        tgt = m.get("matched_target_paper_id", "")
        if not src or not tgt:
            continue
        score = float(m.get("match_score") or 0.0)
        if score < min_match_score:
            continue
        if src == tgt:
            continue
        kept.append(
            {
                "source_paper_id": src,
                "target_paper_id": tgt,
                "ref_idx": m.get("ref_idx", ""),
                "match_method": m.get("match_method", ""),
                "match_score": round(score, 4),
            }
        )

    best_by_edge: Dict[Tuple[str, str], Dict[str, object]] = {}
    for e in kept:
        key = (str(e["source_paper_id"]), str(e["target_paper_id"]))
        if key not in best_by_edge or float(e["match_score"]) > float(best_by_edge[key]["match_score"]):
            best_by_edge[key] = e
    return list(best_by_edge.values())


def filter_edges_for_graph(
    edges: List[Dict[str, object]],
    paper_by_id: Dict[str, Dict[str, str]],
    exclude_auto_nodes: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Keep only edges whose endpoints exist in papers.csv; optionally drop paper_auto_* ids."""
    out: List[Dict[str, object]] = []
    dropped_auto = 0
    dropped_not_in_papers = 0

    for e in edges:
        src = str(e.get("source_paper_id", "") or "")
        tgt = str(e.get("target_paper_id", "") or "")
        if not src or not tgt:
            dropped_not_in_papers += 1
            continue

        if exclude_auto_nodes and (src.startswith("paper_auto_") or tgt.startswith("paper_auto_")):
            dropped_auto += 1
            continue

        if src not in paper_by_id or tgt not in paper_by_id:
            dropped_not_in_papers += 1
            continue

        out.append(e)

    return out, {
        "kept_edges": len(out),
        "dropped_auto_edges": dropped_auto,
        "dropped_nonpaper_edges": dropped_not_in_papers,
    }


def export_graphml(edges: List[Dict[str, object]], paper_by_id: Dict[str, Dict[str, str]], out_graphml: Path) -> None:
    """Build NetworkX directed graph and export graphml."""
    import networkx as nx

    graph = nx.DiGraph()
    for pid, p in paper_by_id.items():
        graph.add_node(
            pid,
            title=p.get("title", ""),
            year=p.get("year", ""),
            journal=p.get("journal", ""),
            doi=p.get("doi", ""),
        )
    for e in edges:
        src = str(e["source_paper_id"])
        tgt = str(e["target_paper_id"])
        graph.add_edge(src, tgt, match_method=e.get("match_method", ""), match_score=float(e.get("match_score", 0.0)))

    out_graphml.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, out_graphml)


def infer_unmatched_reason(row: Dict[str, str], doi_index: dict, title_index: dict) -> str:
    """Infer likely reason for unmatched/ambiguous records in QA sample."""
    status = (row.get("match_status") or "").lower()
    if status == "matched":
        return "check_target_correctness"
    if status == "ambiguous":
        return "multiple_candidates_or_low_margin"
    doi = (row.get("parsed_doi") or "").lower()
    title_norm = normalize_title(row.get("parsed_title", ""))
    if doi and doi not in doi_index:
        return "doi_not_in_library"
    if not title_norm:
        return "parse_issue_missing_title"
    if title_norm not in title_index:
        return "title_not_in_library_or_normalization_issue"
    return "insufficient_parsed_fields"


def sample_match_quality_rows(
    matches: List[Dict[str, str]],
    doi_index: dict,
    title_index: dict,
    per_status_n: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Generate stratified QA sample over matched/ambiguous/unmatched."""
    rng = random.Random(seed)
    buckets = {"matched": [], "ambiguous": [], "unmatched": []}
    for m in matches:
        status = (m.get("match_status") or "").strip().lower()
        if status in buckets:
            buckets[status].append(m)

    sampled = []
    for status, rows in buckets.items():
        chosen = rows if len(rows) <= per_status_n else rng.sample(rows, per_status_n)
        for row in chosen:
            sampled.append(
                {
                    "match_status": status,
                    "source_paper_id": row.get("source_paper_id", ""),
                    "ref_idx": row.get("ref_idx", ""),
                    "parsed_title": row.get("parsed_title", ""),
                    "parsed_year": row.get("parsed_year", ""),
                    "parsed_doi": row.get("parsed_doi", ""),
                    "matched_target_paper_id": row.get("matched_target_paper_id", ""),
                    "match_method": row.get("match_method", ""),
                    "match_score": row.get("match_score", ""),
                    "qa_reason_hint": infer_unmatched_reason(row, doi_index, title_index),
                }
            )
    return sampled


def stage_build_graph(
    citation_matches_csv: Path,
    papers_csv: Path,
    citation_edges_csv: Path,
    graphml_path: Path,
    audit_csv: Path,
    min_match_score: float,
    audit_per_status_n: int,
    seed: int,
) -> Tuple[int, int]:
    """Run stage 4 and output cleaned edges, graphml, and QA samples."""
    matches = read_csv(citation_matches_csv)
    paper_by_id, doi_index, title_index = load_paper_maps(papers_csv)

    edges = build_clean_edges(matches, min_match_score=min_match_score)
    edges, edge_filter_stats = filter_edges_for_graph(edges, paper_by_id, exclude_auto_nodes=True)
    write_csv(
        citation_edges_csv,
        edges,
        fieldnames=["source_paper_id", "target_paper_id", "ref_idx", "match_method", "match_score"],
    )
    export_graphml(edges, paper_by_id, graphml_path)

    audit_rows = sample_match_quality_rows(
        matches=matches,
        doi_index=doi_index,
        title_index=title_index,
        per_status_n=audit_per_status_n,
        seed=seed,
    )
    write_csv(
        audit_csv,
        audit_rows,
        fieldnames=[
            "match_status",
            "source_paper_id",
            "ref_idx",
            "parsed_title",
            "parsed_year",
            "parsed_doi",
            "matched_target_paper_id",
            "match_method",
            "match_score",
            "qa_reason_hint",
        ],
    )
    print(
        "[INFO] build edge filtering: "
        f"kept={edge_filter_stats['kept_edges']}, "
        f"dropped_auto={edge_filter_stats['dropped_auto_edges']}, "
        f"dropped_nonpaper={edge_filter_stats['dropped_nonpaper_edges']}"
    )
    return len(edges), len(audit_rows)


# =========================
# Stage 5: graph analysis + Leiden
# =========================

STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "for",
    "of",
    "in",
    "on",
    "to",
    "with",
    "from",
    "by",
    "using",
    "via",
    "over",
    "under",
    "at",
    "is",
    "are",
    "as",
    "new",
    "study",
    "analysis",
    "method",
    "methods",
    "approach",
    "based",
}


def build_nx_graph(edges_csv: Path, papers_csv: Path):
    """Load edge and paper tables to construct directed graph for analytics."""
    import networkx as nx

    papers = read_csv(papers_csv)
    edges = read_csv(edges_csv)
    paper_by_id = {p.get("paper_id", ""): p for p in papers if p.get("paper_id", "")}

    g = nx.DiGraph()
    for pid, p in paper_by_id.items():
        g.add_node(
            pid,
            title=p.get("title", ""),
            year=p.get("year", ""),
            journal=p.get("journal", ""),
            doi=p.get("doi", ""),
        )
    for e in edges:
        src = e.get("source_paper_id", "")
        tgt = e.get("target_paper_id", "")
        if not src or not tgt:
            continue
        if src not in g:
            g.add_node(src)
        if tgt not in g:
            g.add_node(tgt)
        g.add_edge(src, tgt, match_method=e.get("match_method", ""), match_score=float(e.get("match_score") or 0.0))
    return g, paper_by_id


def graph_basic_stats(g) -> dict:
    """Compute basic graph metrics required by spec."""
    import networkx as nx

    n = g.number_of_nodes()
    m = g.number_of_edges()
    density = nx.density(g) if n > 1 else 0.0
    components = list(nx.weakly_connected_components(g)) if n else []
    largest_cc_size = max((len(c) for c in components), default=0)
    in_degrees = [d for _, d in g.in_degree()]
    out_degrees = [d for _, d in g.out_degree()]
    in_degree_hist = Counter(in_degrees)
    return {
        "num_nodes": n,
        "num_edges": m,
        "density": round(float(density), 8),
        "num_weakly_connected_components": len(components),
        "largest_weakly_component_size": int(largest_cc_size),
        "avg_in_degree": round(float(sum(in_degrees) / n), 6) if n else 0.0,
        "avg_out_degree": round(float(sum(out_degrees) / n), 6) if n else 0.0,
        "in_degree_distribution": {str(k): int(v) for k, v in sorted(in_degree_hist.items(), key=lambda x: x[0])},
    }


def compute_rankings(g, paper_by_id: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    """Compute in_degree, out_degree, PageRank, betweenness for each paper."""
    import networkx as nx

    in_degree = dict(g.in_degree())
    out_degree = dict(g.out_degree())
    pagerank = nx.pagerank(g, alpha=0.85) if g.number_of_nodes() else {}
    betweenness = nx.betweenness_centrality(g, normalized=True) if g.number_of_nodes() else {}

    rows = []
    for pid in g.nodes():
        p = paper_by_id.get(pid, {})
        rows.append(
            {
                "paper_id": pid,
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "journal": p.get("journal", ""),
                "in_degree": int(in_degree.get(pid, 0)),
                "out_degree": int(out_degree.get(pid, 0)),
                "pagerank": round(float(pagerank.get(pid, 0.0)), 10),
                "betweenness": round(float(betweenness.get(pid, 0.0)), 10),
            }
        )
    return rows


def write_top_lists(rank_rows: List[Dict[str, object]], out_dir: Path, top_n: int) -> None:
    """Write top paper lists for classic/center/bridge/outgoing roles."""
    fields = ["paper_id", "title", "year", "journal", "in_degree", "out_degree", "pagerank", "betweenness"]
    by_in = sorted(rank_rows, key=lambda x: (x["in_degree"], x["pagerank"]), reverse=True)[:top_n]
    by_pr = sorted(rank_rows, key=lambda x: (x["pagerank"], x["in_degree"]), reverse=True)[:top_n]
    by_bt = sorted(rank_rows, key=lambda x: (x["betweenness"], x["pagerank"]), reverse=True)[:top_n]
    by_out = sorted(rank_rows, key=lambda x: (x["out_degree"], x["pagerank"]), reverse=True)[:top_n]
    write_csv(out_dir / "top_classic_papers.csv", by_in, fields)
    write_csv(out_dir / "top_center_papers.csv", by_pr, fields)
    write_csv(out_dir / "top_bridge_papers.csv", by_bt, fields)
    write_csv(out_dir / "top_outgoing_papers.csv", by_out, fields)


def run_leiden(g, seed: int) -> Dict[str, int]:
    """Run modularity-based Leiden community detection."""
    import networkx as nx

    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Leiden dependencies missing. Install: pip install python-igraph leidenalg") from exc

    nodes = list(g.nodes())
    if not nodes:
        return {}

    h = nx.Graph()
    h.add_nodes_from(nodes)
    for u, v in g.edges():
        if u == v:
            continue
        if h.has_edge(u, v):
            h[u][v]["weight"] += 1.0
        else:
            h.add_edge(u, v, weight=1.0)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in h.edges()]
    weights = [float(h[u][v].get("weight", 1.0)) for u, v in h.edges()]
    if not edges:
        return {n: i for i, n in enumerate(nodes)}

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=weights,
        seed=seed,
    )
    membership = partition.membership
    return {n: int(membership[node_to_idx[n]]) for n in nodes}


def title_keywords(titles: List[str], top_k: int = 6) -> List[str]:
    """Extract high-frequency keywords from community titles."""
    tokens = []
    for t in titles:
        for w in re.findall(r"[A-Za-z]{3,}", (t or "").lower()):
            if w in STOPWORDS:
                continue
            tokens.append(w)
    counter = Counter(tokens)
    return [w for w, _ in counter.most_common(top_k)]


def build_community_summary(rank_rows: List[Dict[str, object]], communities: Dict[str, int], top_rep_n: int = 3):
    """Summarize each community with keywords and representative/high-impact papers."""
    rows_by_pid = {r["paper_id"]: r for r in rank_rows}
    bucket = defaultdict(list)
    for pid, cid in communities.items():
        if pid in rows_by_pid:
            bucket[cid].append(rows_by_pid[pid])

    out = []
    for cid, rows in sorted(bucket.items(), key=lambda x: x[0]):
        rows_sorted_pr = sorted(rows, key=lambda x: (x["pagerank"], x["in_degree"]), reverse=True)
        rows_sorted_in = sorted(rows, key=lambda x: (x["in_degree"], x["pagerank"]), reverse=True)
        keywords = title_keywords([str(r.get("title", "")) for r in rows], top_k=6)
        out.append(
            {
                "community_id": cid,
                "community_size": len(rows),
                "top_keywords": "; ".join(keywords),
                "representative_papers": "; ".join([str(r["paper_id"]) for r in rows_sorted_pr[:top_rep_n]]),
                "high_impact_papers": "; ".join([str(r["paper_id"]) for r in rows_sorted_in[:top_rep_n]]),
            }
        )
    return out


def stage_analyze_graph(
    citation_edges_csv: Path,
    papers_csv: Path,
    analysis_out_dir: Path,
    top_n: int,
    seed: int,
) -> Dict[str, int]:
    """Run stage 5 and output graph statistics/ranking/community files."""
    analysis_out_dir = analysis_out_dir.resolve()
    analysis_out_dir.mkdir(parents=True, exist_ok=True)

    g, paper_by_id = build_nx_graph(citation_edges_csv, papers_csv)
    stats = graph_basic_stats(g)
    write_json(analysis_out_dir / "graph_basic_stats.json", stats)

    rank_rows = compute_rankings(g, paper_by_id)
    write_csv(
        analysis_out_dir / "paper_rankings.csv",
        rank_rows,
        fieldnames=["paper_id", "title", "year", "journal", "in_degree", "out_degree", "pagerank", "betweenness"],
    )
    write_top_lists(rank_rows, analysis_out_dir, top_n=top_n)

    communities = run_leiden(g, seed=seed)
    comm_rows = [{"paper_id": pid, "community_id": cid} for pid, cid in sorted(communities.items(), key=lambda x: x[0])]
    write_csv(analysis_out_dir / "paper_communities.csv", comm_rows, fieldnames=["paper_id", "community_id"])

    community_summary = build_community_summary(rank_rows, communities)
    write_csv(
        analysis_out_dir / "community_summary.csv",
        community_summary,
        fieldnames=["community_id", "community_size", "top_keywords", "representative_papers", "high_impact_papers"],
    )
    return {"num_nodes": stats["num_nodes"], "num_edges": stats["num_edges"]}


# =========================
# CLI
# =========================


def parse_args() -> argparse.Namespace:
    """Parse CLI args for single-stage or full pipeline runs."""
    parser = argparse.ArgumentParser(description="Unified citation graph pipeline.")
    parser.add_argument(
        "--stage",
        choices=["extract", "parse", "match", "build", "analyze", "ocr_json", "all"],
        default="all",
        help="Run one stage or full pipeline.",
    )

    # Input/output paths are explicitly provided by user.
    parser.add_argument("--papers-csv", type=Path, default=None)
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument("--pdf-root", type=Path, default=None, help="Deprecated for extract stage (kept for compatibility).")
    parser.add_argument("--paper-index-dir", type=Path, default=None)
    parser.add_argument(
        "--extract-source",
        choices=["ocr", "qwen_api"],
        default="qwen_api",
        help="Raw citation extraction source.",
    )
    parser.add_argument(
        "--parse-source",
        choices=["heuristic", "grobid_api", "auto", "qwen_api"],
        default="qwen_api",
        help="Citation parsing source.",
    )
    parser.add_argument("--grobid-url", type=str, default="http://localhost:8070")
    parser.add_argument("--grobid-timeout-sec", type=int, default=120)
    parser.add_argument("--grobid-batch-size", type=int, default=32)
    parser.add_argument("--llm-base-url", type=str, default="https://llmmelon.cloud")
    parser.add_argument("--llm-api-key", type=str, default=os.getenv("LLMMELON_API_KEY", ""))
    parser.add_argument("--llm-model", type=str, default="qwen-3.5-flash")
    parser.add_argument("--llm-timeout-sec", type=int, default=120)
    parser.add_argument("--llm-batch-size", type=int, default=20)
    parser.add_argument("--ocr-output-json", type=Path, default=None)
    parser.add_argument(
        "--ocr-include-dirs",
        nargs="*",
        default=None,
        help="Optional subdirectories under corpus root to include (e.g. three journal folders).",
    )

    parser.add_argument(
        "--raw-citations-csv",
        type=Path,
        default=None,
        help="Raw citations output/input path (.csv or .json).",
    )
    parser.add_argument(
        "--parsed-citations-csv",
        type=Path,
        default=None,
        help="Parsed citations output/input path (.csv or .json).",
    )
    parser.add_argument(
        "--citation-matches-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--citation-edges-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--graphml-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--analysis-out-dir",
        type=Path,
        default=None,
    )

    parser.add_argument("--min-match-score", type=float, default=0.75)
    parser.add_argument("--audit-per-status-n", type=int, default=150)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _require_args(args: argparse.Namespace, names: List[str], stage: str) -> None:
    """Validate required path args for a stage."""
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        labels = ", ".join(f"--{m.replace('_', '-')}" for m in missing)
        raise ValueError(f"Stage '{stage}' missing required arguments: {labels}")


def _resolve_paths(args: argparse.Namespace) -> None:
    """Normalize provided path arguments to absolute paths."""
    path_fields = [
        "papers_csv",
        "corpus_root",
        "pdf_root",
        "paper_index_dir",
        "raw_citations_csv",
        "parsed_citations_csv",
        "citation_matches_csv",
        "citation_edges_csv",
        "graphml_path",
        "audit_csv",
        "analysis_out_dir",
        "ocr_output_json",
    ]
    for name in path_fields:
        v = getattr(args, name)
        if v is not None:
            setattr(args, name, Path(v).expanduser().resolve())

    # Optional fallback: if paper index dir is omitted, use papers.csv parent.
    if args.paper_index_dir is None and args.papers_csv is not None:
        args.paper_index_dir = args.papers_csv.parent


def main() -> None:
    """Dispatch one stage or all stages and print output summaries."""
    args = parse_args()
    stage = args.stage
    _resolve_paths(args)

    if stage == "extract":
        _require_args(args, ["corpus_root", "raw_citations_csv"], stage)
    elif stage == "parse":
        _require_args(args, ["raw_citations_csv", "parsed_citations_csv"], stage)
    elif stage == "ocr_json":
        _require_args(args, ["corpus_root", "ocr_output_json"], stage)
    elif stage == "match":
        _require_args(args, ["parsed_citations_csv", "papers_csv", "citation_matches_csv"], stage)
    elif stage == "build":
        _require_args(
            args,
            ["citation_matches_csv", "papers_csv", "citation_edges_csv", "graphml_path", "audit_csv"],
            stage,
        )
    elif stage == "analyze":
        _require_args(args, ["citation_edges_csv", "papers_csv", "analysis_out_dir"], stage)
    else:  # all
        _require_args(
            args,
            [
                "papers_csv",
                "corpus_root",
                "raw_citations_csv",
                "parsed_citations_csv",
                "citation_matches_csv",
                "citation_edges_csv",
                "graphml_path",
                "audit_csv",
                "analysis_out_dir",
            ],
            stage,
        )

    if stage in {"extract", "all"}:
        n = stage_extract_raw_citations(
            papers_csv=args.papers_csv,
            corpus_root=args.corpus_root,
            raw_citations_csv=args.raw_citations_csv,
            extract_source=args.extract_source,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            llm_timeout_sec=args.llm_timeout_sec,
        )
        print(f"[OK] extract -> {args.raw_citations_csv.resolve()} (rows={n})")
        if stage == "extract":
            return

    if stage == "ocr_json":
        summary = stage_ocr_markdown_to_json(
            corpus_root=args.corpus_root,
            output_json=args.ocr_output_json,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            llm_timeout_sec=args.llm_timeout_sec,
            include_dirs=args.ocr_include_dirs,
        )
        print(
            f"[OK] ocr_json -> {args.ocr_output_json.resolve()} "
            f"(papers={summary['num_papers']}, citations={summary['num_citations']})"
        )
        return

    if stage in {"parse", "all"}:
        n = stage_parse_citations(
            raw_citations_csv=args.raw_citations_csv,
            parsed_citations_csv=args.parsed_citations_csv,
            parse_source=args.parse_source,
            grobid_url=args.grobid_url,
            grobid_timeout_sec=args.grobid_timeout_sec,
            grobid_batch_size=args.grobid_batch_size,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            llm_timeout_sec=args.llm_timeout_sec,
            llm_batch_size=args.llm_batch_size,
        )
        print(f"[OK] parse -> {args.parsed_citations_csv.resolve()} (rows={n})")
        if stage == "parse":
            return

    if stage in {"match", "all"}:
        n = stage_match_citations(
            args.parsed_citations_csv,
            args.papers_csv,
            args.paper_index_dir,
            args.citation_matches_csv,
        )
        print(f"[OK] match -> {args.citation_matches_csv.resolve()} (rows={n})")
        if stage == "match":
            return

    if stage in {"build", "all"}:
        n_edges, n_audit = stage_build_graph(
            args.citation_matches_csv,
            args.papers_csv,
            args.citation_edges_csv,
            args.graphml_path,
            args.audit_csv,
            args.min_match_score,
            args.audit_per_status_n,
            args.seed,
        )
        print(f"[OK] build -> {args.citation_edges_csv.resolve()} (edges={n_edges})")
        print(f"[OK] graphml -> {args.graphml_path.resolve()}")
        print(f"[OK] audit -> {args.audit_csv.resolve()} (rows={n_audit})")
        if stage == "build":
            return

    if stage in {"analyze", "all"}:
        summary = stage_analyze_graph(
            args.citation_edges_csv,
            args.papers_csv,
            args.analysis_out_dir,
            args.top_n,
            args.seed,
        )
        print(f"[OK] analyze -> {args.analysis_out_dir.resolve()} (nodes={summary['num_nodes']}, edges={summary['num_edges']})")


if __name__ == "__main__":
    main()

