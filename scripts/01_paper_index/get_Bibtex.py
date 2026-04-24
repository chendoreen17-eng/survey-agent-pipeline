#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "paper_index" / "papers.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "papers_with_bibtex.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch BibTeX entries by DOI from a papers CSV.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input papers CSV path (must contain at least: paper_id, doi, title).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path with appended bibtex column.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay seconds between DOI requests.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=15,
        help="HTTP timeout for one DOI request.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist progress every N rows.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding for input/output.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output file and start fresh.",
    )
    return parser.parse_args()


def fetch_bibtex_by_doi(doi: object, timeout_sec: int = 15) -> Optional[str]:
    """Fetch standard BibTeX by DOI."""
    if pd.isna(doi) or str(doi).strip() == "":
        return None

    url = f"https://doi.org/{str(doi).strip()}"
    headers = {"Accept": "application/x-bibtex"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout_sec)
        if response.status_code == 200:
            return response.text.strip()
        return None
    except Exception as e:  # pragma: no cover
        print(f"  [Error] DOI {doi}: {e}")
        return None


def ensure_required_columns(df: pd.DataFrame, input_csv: Path) -> None:
    required = {"paper_id", "doi", "title"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns {missing}: {input_csv}")


def prepare_working_df(
    input_csv: Path,
    output_csv: Path,
    encoding: str,
    resume: bool = True,
) -> pd.DataFrame:
    """Load working dataframe with bibtex column and resume support."""
    df_input = pd.read_csv(input_csv, encoding=encoding)
    ensure_required_columns(df_input, input_csv)

    if not resume or not output_csv.exists():
        df_work = df_input.copy()
        if "bibtex" not in df_work.columns:
            df_work["bibtex"] = ""
        return df_work

    # Resume from existing output file.
    df_out = pd.read_csv(output_csv, encoding=encoding)
    if "bibtex" not in df_out.columns:
        df_out["bibtex"] = ""

    # Ensure same row count/order for safe resume; otherwise restart from input.
    if len(df_out) != len(df_input):
        print("[WARN] Existing output row count differs from input. Restarting from input CSV.")
        df_work = df_input.copy()
        df_work["bibtex"] = ""
        return df_work

    if "paper_id" in df_out.columns and "paper_id" in df_input.columns:
        if not (df_out["paper_id"].astype(str).fillna("") == df_input["paper_id"].astype(str).fillna("")).all():
            print("[WARN] Existing output paper_id order differs from input. Restarting from input CSV.")
            df_work = df_input.copy()
            df_work["bibtex"] = ""
            return df_work

    # Merge latest input columns with resumed bibtex values.
    df_work = df_input.copy()
    df_work["bibtex"] = df_out["bibtex"]
    return df_work


def process_papers(
    input_csv: Path,
    output_csv: Path,
    delay: float = 0.5,
    timeout_sec: int = 15,
    save_every: int = 10,
    encoding: str = "utf-8-sig",
    resume: bool = True,
) -> None:
    input_csv = input_csv.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = prepare_working_df(input_csv, output_csv, encoding=encoding, resume=resume)
    # Save initial structure so interrupted run can resume.
    df.to_csv(output_csv, index=False, encoding=encoding)

    total = len(df)
    print(f"[INFO] Input: {input_csv}")
    print(f"[INFO] Output: {output_csv}")
    print(f"[INFO] Total papers: {total}")

    for index, row in df.iterrows():
        current_bib = row.get("bibtex", "")
        if pd.notna(current_bib) and str(current_bib).strip() != "":
            continue

        paper_id = row.get("paper_id", "")
        doi = row.get("doi", "")
        title = str(row.get("title", "") or "")
        print(f"[{index + 1}/{total}] {paper_id} | {title[:60]}")

        bib_entry = fetch_bibtex_by_doi(doi, timeout_sec=timeout_sec)
        if bib_entry:
            df.at[index, "bibtex"] = bib_entry
            print("  [Success] BibTeX saved")
        else:
            print("  [Failed] No BibTeX found")

        if save_every > 0 and (index + 1) % save_every == 0:
            df.to_csv(output_csv, index=False, encoding=encoding)

        if delay > 0:
            time.sleep(delay)

    df.to_csv(output_csv, index=False, encoding=encoding)
    print(f"[OK] Done. Saved: {output_csv}")


def main() -> None:
    args = parse_args()
    process_papers(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        delay=args.delay,
        timeout_sec=args.timeout_sec,
        save_every=args.save_every,
        encoding=args.encoding,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
