import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_jsonl(path: Path, rows: Iterable[Dict[str, Any]], key_fields: List[str]) -> None:
    key_fields = key_fields or []
    existing = load_jsonl(path)

    def key_of(x: Dict[str, Any]) -> str:
        return "||".join(str(x.get(k, "")) for k in key_fields)

    by_key = {key_of(r): r for r in existing}
    for r in rows:
        by_key[key_of(r)] = r
    merged = list(by_key.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def resolve_chat_url(base_or_full: str) -> str:
    s = (base_or_full or "").rstrip("/")
    if not s:
        return "https://llmmelon.cloud/v1/chat/completions"
    low = s.lower()
    if low.endswith("/chat/completions"):
        return s
    if low.endswith("/v1"):
        return s + "/chat/completions"
    return s + "/v1/chat/completions"


def extract_json_from_text(text: str) -> Optional[Any]:
    t = (text or "").strip()
    if not t:
        return None
    if "```" in t:
        import re

        m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.I | re.S)
        if m:
            t = m.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    import re

    m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    timeout_sec: int = 180,
    max_retries: int = 3,
) -> Dict[str, Any]:
    endpoint = resolve_chat_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
            if r.status_code >= 400:
                txt = r.text[:500]
                if r.status_code in {429, 500, 502, 503, 504} and i + 1 < max_retries:
                    time.sleep(1.5 * (i + 1))
                    continue
                raise RuntimeError(f"HTTP {r.status_code}: {txt}")
            data = r.json()
            msg = data["choices"][0]["message"]["content"]
            return {"ok": True, "content": msg, "raw": data, "error": ""}
        except Exception as e:
            last_err = str(e)
            if i + 1 < max_retries:
                time.sleep(1.5 * (i + 1))
                continue
    return {"ok": False, "content": "", "raw": {}, "error": last_err or "unknown_error"}


def get_env_or_value(env_or_value: str) -> str:
    if not env_or_value:
        return ""
    if env_or_value.startswith("env:"):
        return os.getenv(env_or_value[4:], "")
    return env_or_value
