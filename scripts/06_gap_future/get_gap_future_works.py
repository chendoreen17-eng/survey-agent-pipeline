import argparse
import os
from pathlib import Path

import pandas as pd
import requests


def call_llm(api_key: str, api_url: str, model: str, prompt: str, temperature: float) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate gap/future-works report by L1/L2 domains.")
    parser.add_argument("--input-csv", required=True, help="CSV with l1_label, l2_label, title, year, abstract")
    parser.add_argument("--output-md", required=True, help="Output markdown report")
    parser.add_argument("--api-url", default="https://llmmelon.cloud/v1/chat/completions")
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--api-key", default=os.getenv("LLMMELON_API_KEY", "").strip())
    return parser.parse_args()


def safe_col(df: pd.DataFrame, col: str, default: str = ""):
    if col not in df.columns:
        df[col] = default
    return df[col].fillna(default)


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set --api-key or LLMMELON_API_KEY.")

    df = pd.read_csv(args.input_csv)
    safe_col(df, "l1_label", "")
    safe_col(df, "l2_label", "")
    safe_col(df, "title", "")
    safe_col(df, "abstract", "")
    df["year"] = pd.to_numeric(safe_col(df, "year", 0), errors="coerce").fillna(0)

    report = ["# Research Gap & Future Works Report"]

    # L2 bottlenecks
    report.append("\n## 1. L2 Technical Gaps\n")
    for l2 in sorted([x for x in df["l2_label"].astype(str).unique().tolist() if x.strip()]):
        group = df[df["l2_label"].astype(str) == l2].sort_values("year", ascending=False).head(5)
        context = "\n".join(
            [f"- {row.title}: {str(row.abstract)[:250]}" for row in group.itertuples(index=False)]
        )
        prompt = f"""请作为学术专家，基于以下子领域文献样本，提炼3条技术gap，并给出对应future works建议。\n\n子领域：{l2}\n样本：\n{context}\n\n输出格式：\n1) gap\n   future work\n2) ..."""
        ans = call_llm(args.api_key, args.api_url, args.model, prompt, temperature=0.4)
        report.append(f"### L2: {l2}\n{ans}\n")

    # L1 industrial gaps
    report.append("\n## 2. L1 Industrial Gaps\n")
    for l1 in sorted([x for x in df["l1_label"].astype(str).unique().tolist() if x.strip()]):
        sub_l2 = sorted(
            [x for x in df[df["l1_label"].astype(str) == l1]["l2_label"].astype(str).unique().tolist() if x.strip()]
        )
        prompt = f"""你是工业落地专家。一级领域：{l1}。子方向包括：{', '.join(sub_l2)}。\n请总结该一级领域的工业落地gap（数据、部署、鲁棒性），并给future works建议。"""
        ans = call_llm(args.api_key, args.api_url, args.model, prompt, temperature=0.4)
        report.append(f"### L1: {l1}\n{ans}\n")

    # Global trends
    report.append("\n## 3. Global Future Trends (2 years)\n")
    l2_seq = (
        df.groupby("l2_label", as_index=False)["year"].mean().sort_values("year")["l2_label"].astype(str).tolist()
    )
    evo = "\n".join([f"- stage {i+1}: {name}" for i, name in enumerate(l2_seq) if name.strip()])
    prompt = f"""基于以下研究重心迁移序列，预测未来2年4个高价值方向，并说明理由：\n{evo}"""
    ans = call_llm(args.api_key, args.api_url, args.model, prompt, temperature=0.7)
    report.append(ans)

    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(report), encoding="utf-8")
    print(f"[OK] report saved: {out_path}")


if __name__ == "__main__":
    main()
