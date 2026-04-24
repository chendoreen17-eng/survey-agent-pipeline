import argparse
import os
from pathlib import Path

import pandas as pd
import requests


def get_llm_response(api_key: str, api_url: str, model: str, prompt: str, temperature: float = 0.3) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_args():
    parser = argparse.ArgumentParser(description="Name L1 domains from L2 labels via LLM.")
    parser.add_argument("--input-csv", required=True, help="CSV containing final_domain_id and parent_domain_id")
    parser.add_argument("--l2-csv", required=True, help="CSV containing final_domain_id and l2_label")
    parser.add_argument("--output-csv", required=True, help="Output l1_domain_names.csv path")
    parser.add_argument("--api-url", default="https://llmmelon.cloud/v1/chat/completions")
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--api-key", default=os.getenv("LLMMELON_API_KEY", "").strip())
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set --api-key or LLMMELON_API_KEY.")

    df = pd.read_csv(args.input_csv)
    l2_names = pd.read_csv(args.l2_csv)
    df = df.merge(l2_names, on="final_domain_id", how="left")

    if "parent_domain_id" not in df.columns:
        raise ValueError("input csv must contain parent_domain_id")
    if "l2_label" not in df.columns:
        raise ValueError("l2 csv must contain l2_label")

    out_rows = []
    for parent_id in sorted(df["parent_domain_id"].dropna().unique()):
        sub_domains = (
            df[df["parent_domain_id"] == parent_id]["l2_label"].fillna("").astype(str).unique().tolist()
        )
        sub_domains = [x for x in sub_domains if x.strip()]
        sub_domains_str = "\n".join(sub_domains)

        prompt = f"""以下是同一个研究大领域下的多个子方向：
{sub_domains_str}

请为这个一级领域命名，输出格式：
一级领域名称（中英文）：xxx
一级领域核心问题：xxx"""

        print(f"Naming parent_domain_id={parent_id} ...")
        label = get_llm_response(args.api_key, args.api_url, args.model, prompt)
        out_rows.append({"parent_domain_id": int(parent_id), "l1_label": label})

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] saved {out_path} (rows={len(out_rows)})")


if __name__ == "__main__":
    main()
