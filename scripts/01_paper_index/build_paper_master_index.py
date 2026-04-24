#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from urllib.request import Request, urlopen

DOI_RE = re.compile(r"(10\.\d{4,9}/\S+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
MARKDOWN_TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*$")
SUP_RE = re.compile(r"<sup>.*?</sup>", re.IGNORECASE)
LATEX_FOOTNOTE_RE = re.compile(r"\$\^\{[^}]+\}\$")
DOI_CLEANUP_RE = re.compile(r"[\s<>]+")

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
GROBID_ENDPOINTS = {
    "header": "/api/processHeaderDocument",
}

GENERIC_TITLE_KEYWORDS = {
    "abstract",
    "introduction",
    "references",
    "bibliography",
    "mathematics of operations research",
    "siam journal on optimization",
    "supplementary material",
    "acknowledgments",
}

AUTHOR_LINE_BLOCKLIST = (
    "received",
    "accepted",
    "published",
    "doi",
    "https://",
    "http://",
    "abstract",
    "keywords",
    "copyright",
    "msc",
    "subject classification",
)

PAPER_CSV_FIELDS = [
    "paper_id",
    "collection",
    "journal",
    "source_md_path",
    "doi",
    "title",
    "title_norm",
    "authors",
    "first_author_norm",
    "year",
]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFKC", (text or "")).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def normalize_author(text: str) -> str:
    text = unicodedata.normalize("NFKC", (text or "")).lower()
    text = re.sub(r"[^a-z]+", " ", text)
    return normalize_space(text)


def cleanup_line(text: str) -> str:
    text = SUP_RE.sub(" ", text)
    text = LATEX_FOOTNOTE_RE.sub(" ", text)
    text = text.replace("路", ",").replace("，", ",")
    return normalize_space(text)


def decode_filename_doi(md_path: Path) -> str:
    stem = unquote(md_path.stem)
    match = DOI_RE.search(stem)
    if not match:
        return ""
    doi = match.group(1).strip().rstrip(".,);:")
    doi = re.sub(r"\.fixed$", "", doi, flags=re.IGNORECASE)
    return doi.lower()


def clean_doi(text: str) -> str:
    """Normalize DOI token text."""
    if not text:
        return ""
    t = DOI_CLEANUP_RE.sub("", text.strip().rstrip(".,);:"))
    m = DOI_RE.search(t)
    if not m:
        return ""
    return m.group(1).strip().rstrip(".,);:").lower()


def _year_from_text(text: str) -> str:
    m = YEAR_RE.search(text or "")
    if not m:
        return ""
    y = int(m.group(1))
    if 1900 <= y <= 2099:
        return str(y)
    return ""


def _first_nonempty(candidates: Sequence[str]) -> str:
    for c in candidates:
        if c:
            return c
    return ""


def candidate_rel_paths(source_rel_path: str, target_kind: str) -> List[str]:
    """
    Build likely relative path candidates for sidecar files.
    target_kind: 'pdf' or 'tei'
    """
    s = source_rel_path.replace("\\", "/")
    out: List[str] = []
    if target_kind == "pdf":
        if s.endswith(".pdf"):
            out.append(s)
        if s.endswith(".fixed.md"):
            out.append(s[: -len(".fixed.md")] + ".pdf")
        if s.endswith(".md"):
            out.append(s[: -len(".md")] + ".pdf")
    elif target_kind == "tei":
        if s.endswith(".pdf"):
            base = s[: -len(".pdf")]
            out.extend([base + ".tei.xml", base + ".grobid.tei.xml", base + ".xml"])
        if s.endswith(".fixed.md"):
            base = s[: -len(".fixed.md")]
            out.extend([base + ".tei.xml", base + ".grobid.tei.xml", base + ".xml"])
        if s.endswith(".md"):
            out.append(s[: -len(".md")] + ".tei.xml")
            out.append(s[: -len(".md")] + ".grobid.tei.xml")

    # Deduplicate, keep order.
    dedup: List[str] = []
    seen = set()
    for p in out:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def build_file_lookup(root: Optional[Path], suffixes: Sequence[str]) -> Dict[str, List[Path]]:
    """Build filename->path list map for quick fallback resolution."""
    lookup: Dict[str, List[Path]] = defaultdict(list)
    if root is None or not root.exists():
        return {}
    suffix_set = {s.lower() for s in suffixes}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in suffix_set:
            lookup[p.name.lower()].append(p)
    return dict(lookup)


def resolve_sidecar_path(
    md_rel_path: str,
    root: Optional[Path],
    target_kind: str,
    lookup: Optional[Dict[str, List[Path]]] = None,
) -> Optional[Path]:
    """Resolve a PDF/TEI sidecar path by relative candidates then basename lookup."""
    if root is None or not root.exists():
        return None
    md_rel = md_rel_path.replace("\\", "/")
    candidates = candidate_rel_paths(md_rel, target_kind=target_kind)
    for rel in candidates:
        p = (root / rel).resolve()
        if p.exists():
            return p

    if not lookup:
        return None
    md_collection = md_rel.split("/", 1)[0].lower() if "/" in md_rel else ""
    for rel in candidates:
        name = Path(rel).name.lower()
        hits = lookup.get(name) or []
        if not hits:
            continue
        if len(hits) == 1:
            return hits[0]
        for h in hits:
            if md_collection and md_collection in str(h).lower():
                return h
        return hits[0]
    return None


def _tei_text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return normalize_space("".join(el.itertext()))


def _tei_find_first_text(root: ET.Element, xpaths: Sequence[str]) -> str:
    for xp in xpaths:
        t = _tei_text(root.find(xp, TEI_NS))
        if t:
            return t
    return ""


def _pick_best_title(candidates: Sequence[str]) -> str:
    """Pick best title candidate from TEI title nodes."""
    for t in candidates:
        s = normalize_space(t)
        if not s:
            continue
        if len(s) < 8:
            continue
        if not re.search(r"[A-Za-z]{3,}", s):
            continue
        low = s.lower()
        if any(k in low for k in ("references", "bibliography", "supplementary", "acknowledg")):
            continue
        return s
    return ""


def _tei_parse_authors(root: ET.Element) -> List[str]:
    author_nodes = []
    author_nodes.extend(root.findall(".//tei:teiHeader//tei:fileDesc//tei:sourceDesc//tei:biblStruct//tei:analytic//tei:author", TEI_NS))
    author_nodes.extend(root.findall(".//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:author", TEI_NS))

    names: List[str] = []
    for node in author_nodes:
        pers = node.find(".//tei:persName", TEI_NS)
        if pers is None:
            candidate = _tei_text(node)
            if candidate:
                names.extend(split_authors(candidate))
            continue
        parts = []
        for fn in pers.findall(".//tei:forename", TEI_NS):
            t = _tei_text(fn)
            if t:
                parts.append(t)
        for sn in pers.findall(".//tei:surname", TEI_NS):
            t = _tei_text(sn)
            if t:
                parts.append(t)
        full = normalize_space(" ".join(parts))
        if not full:
            full = _tei_text(pers)
        if full:
            names.append(full)

    dedup: List[str] = []
    seen = set()
    for n in names:
        key = normalize_space(n).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(n)
    return dedup


def parse_grobid_tei_metadata(tei_xml_text: str) -> Dict[str, object]:
    """Parse title/authors/year/doi from GROBID TEI XML string."""
    out: Dict[str, object] = {"title": "", "authors": [], "year": "", "doi": ""}
    if not tei_xml_text.strip():
        return out
    try:
        root = ET.fromstring(tei_xml_text)
    except ET.ParseError:
        return out

    title_candidates: List[str] = []
    for t in root.findall(".//tei:teiHeader//tei:titleStmt//tei:title", TEI_NS):
        if (t.attrib.get("level") or "").lower() in {"a", ""}:
            title_candidates.append(_tei_text(t))
    for t in root.findall(".//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:analytic//tei:title", TEI_NS):
        title_candidates.append(_tei_text(t))
    for t in root.findall(".//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:monogr//tei:title", TEI_NS):
        # Keep as late fallback (can be journal/book title)
        title_candidates.append(_tei_text(t))
    title = _pick_best_title(title_candidates)
    authors = _tei_parse_authors(root)

    doi = ""
    for idno in root.findall(".//tei:teiHeader//tei:idno", TEI_NS):
        typ = (idno.attrib.get("type") or "").lower()
        txt = clean_doi(_tei_text(idno))
        if not txt:
            continue
        if typ == "doi":
            doi = txt
            break
        if not doi:
            doi = txt
    if not doi:
        doi = clean_doi(_tei_text(root))

    year_candidates: List[str] = []
    for d in root.findall(".//tei:teiHeader//tei:date", TEI_NS):
        year_candidates.append(_year_from_text(d.attrib.get("when", "")))
        year_candidates.append(_year_from_text(_tei_text(d)))
    for d in root.findall(".//tei:teiHeader//tei:biblStruct//tei:imprint//tei:date", TEI_NS):
        year_candidates.append(_year_from_text(d.attrib.get("when", "")))
        year_candidates.append(_year_from_text(_tei_text(d)))
    year = ""
    for y in year_candidates:
        if y:
            year = y
            break
    if not year:
        year = _year_from_text(doi)

    out["title"] = title
    out["authors"] = authors
    out["year"] = year
    out["doi"] = doi
    return out


def load_grobid_tei_metadata(tei_path: Path) -> Dict[str, object]:
    """Load and parse local TEI XML metadata."""
    try:
        text = tei_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"title": "", "authors": [], "year": "", "doi": ""}
    return parse_grobid_tei_metadata(text)


def _encode_multipart_pdf(
    pdf_path: Path,
    fields: Dict[str, str],
    boundary: str,
) -> bytes:
    """Build multipart/form-data body for GROBID endpoints."""
    crlf = b"\r\n"
    body = bytearray()
    for k, v in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.extend((v or "").encode("utf-8"))
        body.extend(crlf)

    content_type = mimetypes.guess_type(pdf_path.name)[0] or "application/pdf"
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="input"; filename="{pdf_path.name}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(pdf_path.read_bytes())
    body.extend(crlf)
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body)


def request_grobid_tei(
    pdf_path: Path,
    grobid_url: str,
    mode: str,
    timeout_sec: int = 180,
) -> Tuple[str, str]:
    """
    Request TEI XML from GROBID API.
    Returns (tei_xml_text, endpoint_used). Empty text means failure.
    """
    # Header extraction is enough for master index fields (title/authors/year/doi).
    if mode in {"auto", "header"}:
        endpoint_order = ["header"]
    else:
        endpoint_order = ["header"]

    for ep in endpoint_order:
        endpoint = GROBID_ENDPOINTS[ep]
        boundary = "----CodexFormBoundary7MA4YWxkTrZu0gW"
        fields = {"consolidateHeader": "1"}
        body = _encode_multipart_pdf(pdf_path, fields, boundary)
        url = grobid_url.rstrip("/") + endpoint
        req = Request(url=url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Accept", "application/xml,text/xml")
        try:
            with urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read()
            tei = raw.decode("utf-8", errors="replace")
            if tei.strip():
                return tei, ep
        except (HTTPError, URLError, TimeoutError, OSError):
            continue
    return "", ""


def post_json(
    endpoint_url: str,
    payload_obj: dict,
    timeout_sec: int = 120,
    headers: Optional[Dict[str, str]] = None,
) -> dict:
    """POST JSON and parse JSON response with retry for transient failures."""
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
        except (URLError, RemoteDisconnected, TimeoutError, OSError) as e:
            last_err = RuntimeError(f"LLM request failed at {endpoint_url}: {e}")
            if attempt < 3:
                time.sleep(1.5 * attempt)
                continue
            raise last_err from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM JSON decode failed at {endpoint_url}: {e}") from e
    raise RuntimeError(f"LLM request failed at {endpoint_url}: {last_err}")


def resolve_chat_completions_url(base_or_full_url: str) -> str:
    """Resolve base URL into OpenAI-compatible /v1/chat/completions endpoint."""
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
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


def _extract_json_from_text(text: str):
    """Parse first valid JSON object/array from model text."""
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


def _normalize_llm_authors(authors_obj: object) -> List[str]:
    """Normalize LLM author output that may be list/string/mixed format."""
    if isinstance(authors_obj, list):
        raw = [normalize_space(str(x)) for x in authors_obj]
        out = [x for x in raw if x]
    elif isinstance(authors_obj, str):
        # Reuse existing splitter for robust punctuation handling.
        out = split_authors(authors_obj)
    else:
        out = []

    deduped: List[str] = []
    seen = set()
    for a in out:
        k = normalize_space(a).lower()
        if not k or k in seen:
            continue
        seen.add(k)
        deduped.append(a)
    return deduped


def request_qwen_metadata(
    md_text: str,
    md_path: Path,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    llm_timeout_sec: int = 120,
) -> Dict[str, object]:
    """
    Extract title/authors/year/doi from OCR markdown via New API.
    Returns normalized dict with keys: doi/title/authors/year.
    """
    endpoint = resolve_chat_completions_url(llm_base_url)
    headers: Dict[str, str] = {}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"

    # Long OCR can be noisy and expensive; use head section for header metadata.
    snippet = (md_text or "")[:22000]
    filename_hint_doi = decode_filename_doi(md_path)

    sys_prompt = (
        "You extract paper metadata from OCR markdown.\n"
        "Return ONLY JSON with keys: title, authors, year, doi.\n"
        "Rules:\n"
        "1) title: paper title string, no journal name.\n"
        "2) authors: array of author full names in order.\n"
        "3) year: 4-digit year in 1900-2099, else empty string.\n"
        "4) doi: canonical DOI like 10.xxxx/..., no URL prefix.\n"
        "No explanation."
    )
    user_prompt = (
        f"filename: {md_path.name}\n"
        f"doi_hint_from_filename: {filename_hint_doi}\n"
        "OCR markdown:\n"
        f"{snippet}"
    )

    model_candidates: List[str] = []
    if llm_model:
        model_candidates.append(llm_model)
    low_name = llm_model.lower() if llm_model else ""
    if low_name == "qwen-3.5-flash":
        model_candidates.append("qwen3.5-flash")
    elif low_name == "qwen3.5-flash":
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
        raise RuntimeError(f"Qwen metadata call failed for models: {model_candidates}") from last_err

    choices = resp.get("choices", []) if isinstance(resp, dict) else []
    content = ""
    if choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""

    obj = _extract_json_from_text(content)
    if not isinstance(obj, dict):
        return {"doi": filename_hint_doi, "title": "", "authors": [], "year": _year_from_text(filename_hint_doi)}

    title = normalize_space(str(obj.get("title", "") or ""))
    authors = _normalize_llm_authors(obj.get("authors"))
    doi = clean_doi(str(obj.get("doi", "") or "")) or filename_hint_doi
    year = _year_from_text(str(obj.get("year", "") or "")) or _year_from_text(doi)

    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
    }


def extract_title(lines: Sequence[str]) -> str:
    candidates: List[str] = []
    for line in lines[:200]:
        m = MARKDOWN_TITLE_RE.match(line)
        if not m:
            continue
        title = normalize_space(m.group(1))
        if not title:
            continue
        title_norm = normalize_title(title)
        if len(title_norm.split()) < 3:
            continue
        if any(k in title_norm for k in GENERIC_TITLE_KEYWORDS):
            continue
        candidates.append(title)

    if candidates:
        return candidates[0]

    for line in lines[:80]:
        clean = cleanup_line(line)
        norm = normalize_title(clean)
        if len(norm.split()) < 4:
            continue
        if any(k in norm for k in GENERIC_TITLE_KEYWORDS):
            continue
        if len(clean) > 180:
            continue
        if clean.isupper():
            continue
        return clean

    return ""


def looks_like_author_line(line: str) -> bool:
    low = line.lower()
    if any(x in low for x in AUTHOR_LINE_BLOCKLIST):
        return False
    if "@" in low:
        return False
    if len(line) < 6 or len(line) > 180:
        return False
    if sum(ch.isdigit() for ch in line) > 8:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z\.\-']*", line)
    if len(tokens) < 2:
        return False
    long_tokens = [t for t in tokens if len(re.sub(r"[^A-Za-z]", "", t)) >= 2]
    return len(long_tokens) >= 2


def split_authors(author_line: str) -> List[str]:
    line = cleanup_line(author_line)
    line = re.sub(r"\band\b", ",", line, flags=re.IGNORECASE)
    line = re.sub(r"\s*&\s*", ",", line)
    line = line.replace(";", ",")
    parts = [normalize_space(p) for p in line.split(",")]
    names: List[str] = []
    for p in parts:
        p = re.sub(r"[\*\u2020\u2021]+", " ", p)
        p = re.sub(r"\s{2,}", " ", p).strip()
        tokens = re.findall(r"[A-Za-z][A-Za-z\.\-']*", p)
        if len(tokens) < 2:
            continue
        name = " ".join(tokens)
        if not re.search(r"[A-Za-z]", name):
            continue
        names.append(name)
    deduped: List[str] = []
    seen = set()
    for n in names:
        key = normalize_space(n).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(n)
    return deduped


def extract_authors(lines: Sequence[str], title: str) -> List[str]:
    start = 0
    if title:
        title_norm = normalize_space(title).lower()
        for idx, line in enumerate(lines[:220]):
            if title_norm and title_norm in normalize_space(line).lower():
                start = idx + 1
                break

    for line in lines[start : min(start + 24, len(lines))]:
        clean = cleanup_line(line)
        if not looks_like_author_line(clean):
            continue
        names = split_authors(clean)
        if len(names) >= 1:
            return names
    return []


def extract_year(lines: Sequence[str], doi: str) -> str:
    years: List[int] = []
    for line in lines[:280]:
        line_low = line.lower()
        if "reference" in line_low:
            break
        for y in YEAR_RE.findall(line):
            yi = int(y)
            if 1900 <= yi <= 2099:
                years.append(yi)

    if years:
        counts = Counter(years)
        best = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))[0][0]
        return str(best)

    for y in YEAR_RE.findall(doi or ""):
        yi = int(y)
        if 1900 <= yi <= 2099:
            return str(yi)
    return ""


def extract_ocr_metadata(md_path: Path) -> Dict[str, object]:
    """Extract metadata from OCR markdown using heuristic rules."""
    content = md_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    doi = decode_filename_doi(md_path)
    title = extract_title(lines)
    authors = extract_authors(lines, title)
    year = extract_year(lines, doi)
    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
    }


def pick_first_author_key(authors: Sequence[str]) -> str:
    if not authors:
        return ""
    first = normalize_space(authors[0])
    tokens = [t for t in re.split(r"[\s\-]+", first) if t]
    if not tokens:
        return ""
    return normalize_author(tokens[-1])


def stable_paper_id(rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    return f"paper_{digest}"


def infer_title_from_filename(path: Path) -> str:
    """Best-effort title fallback from filename when only PDF is available."""
    stem = unquote(path.stem)
    # Drop leading numeric chunks like "820_8113123_..."
    stem = re.sub(r"^(?:\d+[_\-]+)+", "", stem)
    # Remove DOI token from name if present.
    stem = DOI_RE.sub(" ", stem)
    stem = stem.replace("%2F", "/").replace("%2f", "/")
    stem = stem.replace("_", " ").replace("-", " ")
    title = normalize_space(stem)
    # Do not keep pure id-like filename as title.
    if not re.search(r"[A-Za-z]{2,}", title):
        return ""
    return title


def extract_pdf_fallback_metadata(pdf_path: Path) -> Dict[str, object]:
    """Fallback metadata when only PDF exists and GROBID isn't available."""
    doi = decode_filename_doi(pdf_path)
    title = infer_title_from_filename(pdf_path)
    year = _year_from_text(doi)
    return {
        "doi": doi,
        "title": title,
        "authors": [],
        "year": year,
    }


def iter_md_files(corpus_root: Path, collections: Optional[Sequence[str]]) -> Iterable[Tuple[str, Path]]:
    if collections:
        target_dirs = [corpus_root / c for c in collections]
    else:
        target_dirs = sorted([p for p in corpus_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    for directory in target_dirs:
        if not directory.exists():
            continue
        for md_file in sorted(directory.glob("*.fixed.md"), key=lambda p: p.name.lower()):
            yield directory.name, md_file


def iter_pdf_files(pdf_root: Path, collections: Optional[Sequence[str]]) -> Iterable[Tuple[str, Path]]:
    """Iterate PDF files by optional collections under pdf_root."""
    if collections:
        target_dirs = [pdf_root / c for c in collections]
    else:
        target_dirs = sorted([p for p in pdf_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    for directory in target_dirs:
        if not directory.exists():
            continue
        for pdf_file in sorted(directory.rglob("*.pdf"), key=lambda p: str(p).lower()):
            yield directory.name, pdf_file


def merge_metadata(
    ocr_meta: Dict[str, object],
    grobid_meta: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Merge OCR and GROBID metadata with GROBID-first preference."""
    g = grobid_meta or {"doi": "", "title": "", "authors": [], "year": ""}
    o = ocr_meta

    g_doi = clean_doi(str(g.get("doi", "")))
    o_doi = clean_doi(str(o.get("doi", "")))
    doi = _first_nonempty([g_doi, o_doi])

    g_title = normalize_space(str(g.get("title", "")))
    o_title = normalize_space(str(o.get("title", "")))
    title = _first_nonempty([g_title, o_title])

    g_authors = [normalize_space(str(x)) for x in (g.get("authors") or []) if normalize_space(str(x))]
    o_authors = [normalize_space(str(x)) for x in (o.get("authors") or []) if normalize_space(str(x))]
    authors = g_authors or o_authors

    g_year = _year_from_text(str(g.get("year", "")))
    o_year = _year_from_text(str(o.get("year", "")))
    year = _first_nonempty([g_year, o_year, _year_from_text(doi)])

    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
    }


def build_records(
    corpus_root: Path,
    collections: Optional[Sequence[str]],
    input_source: str = "md",
    metadata_source: str = "auto",
    tei_root: Optional[Path] = None,
    pdf_root: Optional[Path] = None,
    grobid_url: str = "http://localhost:8070",
    grobid_mode: str = "auto",
    grobid_timeout_sec: int = 180,
    grobid_cache_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Build paper records from markdown and/or PDF sources.

    input_source:
    - md: iterate *.fixed.md under corpus_root
    - pdf: iterate *.pdf under pdf_root (or corpus_root when pdf_root is None)
    - auto: md first, then add remaining pdf files not covered by md mapping

    metadata_source:
    - ocr: only OCR markdown heuristics
    - grobid_tei: prefer local TEI files, fallback to OCR
    - grobid_api: prefer GROBID API from PDF, fallback to OCR
    - auto: try local TEI, then API, then OCR
    """
    src_mode = (input_source or "md").strip().lower()
    if src_mode not in {"md", "pdf", "auto"}:
        raise ValueError("input_source must be one of: md, pdf, auto")

    source = (metadata_source or "auto").strip().lower()
    if source not in {"ocr", "grobid_tei", "grobid_api", "auto"}:
        raise ValueError("metadata_source must be one of: ocr, grobid_tei, grobid_api, auto")

    stats = Counter()
    records: List[Dict[str, str]] = []
    pdf_base = pdf_root if pdf_root is not None else corpus_root
    pdf_lookup = build_file_lookup(pdf_base, suffixes=[".pdf"]) if pdf_base else {}
    tei_lookup = build_file_lookup(tei_root, suffixes=[".xml"]) if tei_root else {}
    seen_md_rels = set()

    sources: List[Tuple[str, str, Path]] = []
    if src_mode in {"md", "auto"} and corpus_root.exists():
        for collection, md_path in iter_md_files(corpus_root, collections):
            rel = str(md_path.relative_to(corpus_root)).replace("\\", "/")
            sources.append(("md", collection, md_path))
            seen_md_rels.add(rel.lower())

    if src_mode in {"pdf", "auto"} and pdf_base and pdf_base.exists():
        for collection, pdf_path in iter_pdf_files(pdf_base, collections):
            rel = str(pdf_path.relative_to(pdf_base)).replace("\\", "/")
            if src_mode == "auto":
                md_equiv = (rel[: -len(".pdf")] + ".fixed.md") if rel.lower().endswith(".pdf") else rel
                if md_equiv.lower() in seen_md_rels:
                    continue
            sources.append(("pdf", collection, pdf_path))

    for source_kind, collection, source_path in sources:
        if source_kind == "md":
            rel_path = str(source_path.relative_to(corpus_root)).replace("\\", "/")
            ocr_meta = extract_ocr_metadata(source_path)
            direct_pdf_path: Optional[Path] = None
        else:
            rel_path = str(source_path.relative_to(pdf_base)).replace("\\", "/")
            ocr_meta = extract_pdf_fallback_metadata(source_path)
            direct_pdf_path = source_path

        grobid_meta: Optional[Dict[str, object]] = None
        grobid_from = ""

        if source in {"grobid_tei", "auto"} and tei_root:
            tei_path = resolve_sidecar_path(rel_path, tei_root, target_kind="tei", lookup=tei_lookup)
            if tei_path and tei_path.exists():
                parsed = load_grobid_tei_metadata(tei_path)
                if parsed.get("title") or parsed.get("authors") or parsed.get("doi"):
                    grobid_meta = parsed
                    grobid_from = "tei"
                    stats["grobid_tei_hits"] += 1
                else:
                    stats["grobid_tei_empty"] += 1
            else:
                stats["grobid_tei_miss"] += 1

        need_api = source == "grobid_api" or (source == "auto" and not grobid_meta)
        if need_api and pdf_base:
            pdf_path = direct_pdf_path
            if pdf_path is None:
                pdf_path = resolve_sidecar_path(rel_path, pdf_base, target_kind="pdf", lookup=pdf_lookup)
            if pdf_path and pdf_path.exists():
                tei_text = ""
                endpoint_used = ""
                cache_path = None
                if grobid_cache_dir:
                    cache_path = (grobid_cache_dir / rel_path).with_suffix(".tei.xml")
                    if cache_path.exists():
                        tei_text = cache_path.read_text(encoding="utf-8", errors="replace")
                        endpoint_used = "cache"
                        stats["grobid_cache_hits"] += 1

                if not tei_text:
                    tei_text, endpoint_used = request_grobid_tei(
                        pdf_path,
                        grobid_url=grobid_url,
                        mode=grobid_mode,
                        timeout_sec=grobid_timeout_sec,
                    )

                if tei_text:
                    parsed = parse_grobid_tei_metadata(tei_text)
                    if parsed.get("title") or parsed.get("authors") or parsed.get("doi"):
                        grobid_meta = parsed
                        grobid_from = endpoint_used or "api"
                        stats["grobid_api_hits"] += 1
                        if cache_path and not cache_path.exists():
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            cache_path.write_text(tei_text, encoding="utf-8")
                    else:
                        stats["grobid_api_empty"] += 1
                else:
                    stats["grobid_api_fail"] += 1
            else:
                stats["grobid_pdf_miss"] += 1
        elif need_api and source == "grobid_api":
            stats["grobid_pdf_root_missing"] += 1

        merged = merge_metadata(ocr_meta, grobid_meta)
        doi = str(merged.get("doi") or "")
        title = str(merged.get("title") or "")
        authors_list = list(merged.get("authors") or [])
        year = str(merged.get("year") or "")

        title_norm = normalize_title(title)
        authors_text = "; ".join(authors_list)
        first_author = pick_first_author_key(authors_list)
        journal = collection.replace("_", " ").replace("-副本", "").strip()

        stats["record_total"] += 1
        stats[f"record_input_{source_kind}"] += 1
        if grobid_from:
            stats[f"record_from_{grobid_from}"] += 1
        else:
            stats["record_from_ocr"] += 1

        records.append(
            {
                "paper_id": stable_paper_id(rel_path),
                "collection": collection,
                "journal": journal,
                "source_md_path": rel_path,
                "doi": doi,
                "title": title,
                "title_norm": title_norm,
                "authors": authors_text,
                "first_author_norm": first_author,
                "year": year,
            }
        )
    return records, dict(stats)


def build_records_qwen_from_ocr_markdown(
    corpus_root: Path,
    collections: Optional[Sequence[str]],
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str = "qwen-3.5-flash",
    llm_timeout_sec: int = 120,
    request_interval_sec: float = 0.0,
    out_csv: Optional[Path] = None,
    append_existing_csv: bool = True,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Build paper records from OCR markdown, extracting metadata via Qwen New API.
    Falls back to OCR heuristics when API fails or returns empty fields.
    If out_csv is provided, write one row immediately after each paper is processed.
    """
    stats = Counter()
    records: List[Dict[str, str]] = []
    existing_source_paths = set()
    existing_dois = set()
    existing_title_norms = set()
    existing_title_year = set()

    if out_csv and append_existing_csv and out_csv.exists():
        with out_csv.open("r", encoding="utf-8", newline="") as rf:
            reader = csv.DictReader(rf)
            for raw in reader:
                row = {k: (raw.get(k, "") or "") for k in PAPER_CSV_FIELDS}
                row["source_md_path"] = row["source_md_path"].replace("\\", "/")
                if row["source_md_path"] and not row["paper_id"]:
                    row["paper_id"] = stable_paper_id(row["source_md_path"])
                row["doi"] = clean_doi(row.get("doi", ""))
                row["title_norm"] = row.get("title_norm", "") or normalize_title(row.get("title", ""))
                row["year"] = _year_from_text(row.get("year", ""))
                if row["source_md_path"]:
                    existing_source_paths.add(row["source_md_path"])
                if row["doi"]:
                    existing_dois.add(row["doi"])
                if row["title_norm"]:
                    existing_title_norms.add(row["title_norm"])
                    if row["year"]:
                        existing_title_year.add(f"{row['title_norm']}|{row['year']}")
                records.append(row)
        stats["csv_loaded_existing"] = len(records)

    csv_file = None
    csv_writer = None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_exists = out_csv.exists()
        csv_size = out_csv.stat().st_size if csv_exists else 0
        mode = "a" if append_existing_csv and csv_exists else "w"
        need_header = (mode == "w") or (csv_size == 0)
        csv_file = out_csv.open(mode, encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=PAPER_CSV_FIELDS)
        if need_header:
            csv_writer.writeheader()
            csv_file.flush()

    stats["record_total"] = len(records)

    try:
        for collection, md_path in iter_md_files(corpus_root, collections):
            rel_path = str(md_path.relative_to(corpus_root)).replace("\\", "/")
            if rel_path in existing_source_paths:
                stats["csv_skip_existing"] += 1
                continue
            ocr_meta = extract_ocr_metadata(md_path)
            candidate_doi = clean_doi(str(ocr_meta.get("doi", ""))) or decode_filename_doi(md_path)
            candidate_title_norm = normalize_title(str(ocr_meta.get("title", "") or ""))
            candidate_year = _year_from_text(str(ocr_meta.get("year", "") or ""))
            candidate_title_year = f"{candidate_title_norm}|{candidate_year}" if candidate_title_norm and candidate_year else ""

            # Compare with existing papers.csv before any API request.
            if candidate_doi and candidate_doi in existing_dois:
                stats["csv_skip_duplicate_doi"] += 1
                continue
            if candidate_title_year and candidate_title_year in existing_title_year:
                stats["csv_skip_duplicate_title_year"] += 1
                continue
            if candidate_title_norm and candidate_title_norm in existing_title_norms:
                stats["csv_skip_duplicate_title"] += 1
                continue

            md_text = md_path.read_text(encoding="utf-8", errors="replace")

            llm_meta: Optional[Dict[str, object]] = None
            try:
                llm_meta = request_qwen_metadata(
                    md_text=md_text,
                    md_path=md_path,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    llm_model=llm_model,
                    llm_timeout_sec=llm_timeout_sec,
                )
                if any(
                    [
                        llm_meta.get("title"),
                        llm_meta.get("doi"),
                        llm_meta.get("year"),
                        bool(llm_meta.get("authors")),
                    ]
                ):
                    stats["llm_success"] += 1
                else:
                    stats["llm_empty"] += 1
            except Exception:
                llm_meta = None
                stats["llm_fail"] += 1

            merged = merge_metadata(ocr_meta, llm_meta)
            doi = str(merged.get("doi") or "")
            title = str(merged.get("title") or "")
            authors_list = list(merged.get("authors") or [])
            year = str(merged.get("year") or "")
            title_norm = normalize_title(title)
            authors_text = "; ".join(authors_list)
            first_author = pick_first_author_key(authors_list)
            journal = collection.replace("_", " ").replace("-副本", "").strip()

            stats["record_total"] += 1
            if llm_meta and (llm_meta.get("title") or llm_meta.get("doi") or llm_meta.get("year") or llm_meta.get("authors")):
                stats["record_from_llm"] += 1
            else:
                stats["record_from_ocr_fallback"] += 1

            row = {
                "paper_id": stable_paper_id(rel_path),
                "collection": collection,
                "journal": journal,
                "source_md_path": rel_path,
                "doi": doi,
                "title": title,
                "title_norm": title_norm,
                "authors": authors_text,
                "first_author_norm": first_author,
                "year": year,
            }
            records.append(row)

            if csv_writer and csv_file:
                csv_writer.writerow(row)
                csv_file.flush()
                existing_source_paths.add(rel_path)
                if row["doi"]:
                    existing_dois.add(row["doi"])
                if row["title_norm"]:
                    existing_title_norms.add(row["title_norm"])
                    if row["year"]:
                        existing_title_year.add(f"{row['title_norm']}|{row['year']}")
                stats["csv_written"] += 1

            if request_interval_sec > 0:
                time.sleep(request_interval_sec)
    finally:
        if csv_file:
            csv_file.close()

    return records, dict(stats)


def write_csv(records: Sequence[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_CSV_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_indexes(records: Sequence[Dict[str, str]]) -> Tuple[dict, dict, dict, dict]:
    doi_index: Dict[str, List[str]] = defaultdict(list)
    title_index: Dict[str, List[str]] = defaultdict(list)
    author_year_index: Dict[str, List[str]] = defaultdict(list)

    for r in records:
        paper_id = r["paper_id"]
        doi = (r.get("doi") or "").lower()
        title_norm = r.get("title_norm") or ""
        first_author = r.get("first_author_norm") or ""
        year = r.get("year") or ""

        if doi:
            doi_index[doi].append(paper_id)
        if title_norm:
            title_index[title_norm].append(paper_id)
        if first_author and year:
            key = f"{first_author}|{year}"
            author_year_index[key].append(paper_id)

    summary = {
        "paper_count": len(records),
        "doi_index_size": len(doi_index),
        "title_index_size": len(title_index),
        "author_year_index_size": len(author_year_index),
        "duplicate_doi_keys": sum(1 for ids in doi_index.values() if len(ids) > 1),
        "duplicate_title_keys": sum(1 for ids in title_index.values() if len(ids) > 1),
    }
    return dict(doi_index), dict(title_index), dict(author_year_index), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper master index from OCR markdown.")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("fixed_cropus_md"),
        help="Root directory containing OCR markdown collection directories.",
    )
    parser.add_argument(
        "--input-source",
        choices=["md", "pdf", "auto"],
        default="md",
        help="Input corpus source. For qwen_api mode, use md.",
    )
    parser.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help="Optional explicit collection directory names under corpus root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("source_discovery") / "paper_index",
        help="Output directory for merged JSON file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Merged output JSON path. Default: <out-dir>/paper_master_index.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV path. Default: <out-dir>/papers.csv",
    )
    parser.add_argument(
        "--csv-overwrite",
        action="store_true",
        help="Overwrite output CSV before run. Default is append/resume for qwen_api mode.",
    )
    parser.add_argument(
        "--metadata-source",
        choices=["qwen_api", "ocr", "grobid_tei", "grobid_api", "auto"],
        default="qwen_api",
        help="Metadata source mode. qwen_api uses New API (qwen-3.5-flash).",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="https://llmmelon.cloud",
        help="New API base URL (OpenAI-compatible).",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=os.getenv("LLMMELON_API_KEY", ""),
        help="New API token. Prefer env var LLMMELON_API_KEY.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen-3.5-flash",
        help="LLM model name for metadata extraction.",
    )
    parser.add_argument(
        "--llm-timeout-sec",
        type=int,
        default=120,
        help="Timeout seconds for one LLM request.",
    )
    parser.add_argument(
        "--llm-request-interval-sec",
        type=float,
        default=0.0,
        help="Optional sleep interval between LLM requests.",
    )

    # Legacy args kept for backward compatibility.
    parser.add_argument(
        "--tei-root",
        type=Path,
        default=None,
        help="Root directory of local GROBID TEI XML files (optional).",
    )
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=None,
        help="Root directory of source PDFs for GROBID API mode (optional).",
    )
    parser.add_argument(
        "--grobid-url",
        type=str,
        default="http://localhost:8070",
        help="GROBID service base URL.",
    )
    parser.add_argument(
        "--grobid-mode",
        choices=["auto", "header"],
        default="auto",
        help="GROBID endpoint mode. both auto and header use processHeaderDocument.",
    )
    parser.add_argument(
        "--grobid-timeout-sec",
        type=int,
        default=180,
        help="Timeout in seconds for one GROBID request.",
    )
    parser.add_argument(
        "--grobid-cache-dir",
        type=Path,
        default=None,
        help="Optional cache dir for TEI XML returned by GROBID API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_root = args.corpus_root.resolve()
    out_dir = args.out_dir.resolve()
    collections = args.collections
    input_source = args.input_source
    tei_root = args.tei_root.resolve() if args.tei_root else None
    pdf_root = args.pdf_root.resolve() if args.pdf_root else None
    grobid_cache_dir = args.grobid_cache_dir.resolve() if args.grobid_cache_dir else None
    output_json = args.output_json.resolve() if args.output_json else (out_dir / "paper_master_index.json")
    output_csv = args.output_csv.resolve() if args.output_csv else (out_dir / "papers.csv")

    if input_source in {"md", "auto"} and not corpus_root.exists():
        raise FileNotFoundError(f"Corpus root not found: {corpus_root}")
    if args.metadata_source == "qwen_api" and input_source != "md":
        raise ValueError("--metadata-source qwen_api currently supports only --input-source md.")
    if input_source == "pdf":
        pdf_base = pdf_root if pdf_root is not None else corpus_root
        if not pdf_base.exists():
            raise FileNotFoundError(f"PDF root not found: {pdf_base}")
        if args.metadata_source == "ocr":
            raise ValueError("For --input-source pdf, --metadata-source ocr will produce poor metadata. Use grobid_api or auto.")

    if args.metadata_source == "qwen_api":
        if not args.llm_api_key:
            raise ValueError(
                "Missing API key. Set --llm-api-key or environment variable LLMMELON_API_KEY."
            )
        records, metadata_stats = build_records_qwen_from_ocr_markdown(
            corpus_root=corpus_root,
            collections=collections,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            llm_timeout_sec=args.llm_timeout_sec,
            request_interval_sec=args.llm_request_interval_sec,
            out_csv=output_csv,
            append_existing_csv=not args.csv_overwrite,
        )
    else:
        records, metadata_stats = build_records(
            corpus_root=corpus_root,
            collections=collections,
            input_source=input_source,
            metadata_source=args.metadata_source,
            tei_root=tei_root,
            pdf_root=pdf_root,
            grobid_url=args.grobid_url,
            grobid_mode=args.grobid_mode,
            grobid_timeout_sec=args.grobid_timeout_sec,
            grobid_cache_dir=grobid_cache_dir,
        )
        if input_source == "pdf" and args.metadata_source == "grobid_api":
            grobid_hits = (
                metadata_stats.get("record_from_header", 0)
                + metadata_stats.get("record_from_cache", 0)
            )
            if grobid_hits == 0:
                raise RuntimeError(
                    "No metadata was parsed from GROBID API. Please check --grobid-url, service status, and PDF accessibility."
                )
        write_csv(records, output_csv)
    if not records:
        if input_source == "pdf":
            pdf_base = pdf_root if pdf_root is not None else corpus_root
            raise RuntimeError(f"No .pdf files found under: {pdf_base}")
        raise RuntimeError(f"No .fixed.md files found under: {corpus_root}")

    records = sorted(records, key=lambda x: x["source_md_path"])
    doi_index, title_index, author_year_index, summary = build_indexes(records)
    summary["input_source_mode"] = input_source
    summary["metadata_source_mode"] = args.metadata_source
    summary["metadata_stats"] = metadata_stats

    output_obj = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corpus_root": str(corpus_root),
            "collections": list(collections) if collections else [],
            "input_source_mode": input_source,
            "metadata_source_mode": args.metadata_source,
            "llm_base_url": args.llm_base_url if args.metadata_source == "qwen_api" else "",
            "llm_model": args.llm_model if args.metadata_source == "qwen_api" else "",
        },
        "records": records,
        "indexes": {
            "doi_index": doi_index,
            "title_index": title_index,
            "author_year_index": author_year_index,
            "summary": summary,
        },
    }
    write_json(output_obj, output_json)

    print(f"[OK] merged_json: {output_json}")
    print(f"[OK] papers_csv: {output_csv}")
    print(f"[INFO] paper_count={summary['paper_count']}, doi_keys={summary['doi_index_size']}")
    if args.metadata_source == "qwen_api":
        print(
            f"[INFO] metadata_mode=qwen_api, llm_success={metadata_stats.get('llm_success', 0)}, "
            f"llm_fail={metadata_stats.get('llm_fail', 0)}, fallback_ocr={metadata_stats.get('record_from_ocr_fallback', 0)}, "
            f"csv_written={metadata_stats.get('csv_written', 0)}, csv_skip_existing={metadata_stats.get('csv_skip_existing', 0)}, "
            f"skip_dup_doi={metadata_stats.get('csv_skip_duplicate_doi', 0)}, "
            f"skip_dup_title_year={metadata_stats.get('csv_skip_duplicate_title_year', 0)}, "
            f"skip_dup_title={metadata_stats.get('csv_skip_duplicate_title', 0)}"
        )
    else:
        print(
            f"[INFO] input_mode={input_source}, metadata_mode={args.metadata_source}, "
            f"from_grobid={metadata_stats.get('record_from_tei', 0) + metadata_stats.get('record_from_header', 0) + metadata_stats.get('record_from_cache', 0)}"
        )


if __name__ == "__main__":
    main()

