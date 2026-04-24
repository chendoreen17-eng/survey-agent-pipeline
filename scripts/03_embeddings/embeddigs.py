import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper embeddings via llmmelon API.")
    parser.add_argument("--input-csv", required=True, help="Input papers csv, e.g. papers.csv")
    parser.add_argument("--output-npy", required=True, help="Output embedding matrix .npy path")
    parser.add_argument("--output-metadata", required=True, help="Output metadata csv path")
    parser.add_argument("--api-key", required=True, help="LLMMELON API key")
    parser.add_argument("--base-url", default="https://llmmelon.cloud/v1", help="Embedding API base URL")
    parser.add_argument("--model", default="text-embedding-3-small", help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size per API request")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry times per failed batch")
    parser.add_argument("--retry-wait-sec", type=float, default=3.0, help="Seconds to wait before retry")
    return parser.parse_args()


def _safe_text(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).replace("\n", " ").strip()


def build_texts(df: pd.DataFrame) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        title = _safe_text(row["title"]) if "title" in df.columns else ""
        title_norm = _safe_text(row["title_norm"]) if "title_norm" in df.columns else ""
        journal = _safe_text(row["journal"]) if "journal" in df.columns else ""
        combined = f"Title: {title} | Keywords: {title_norm} | Venue: {journal}"
        texts.append(combined)
    return texts


def generate_embeddings():
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    input_csv = Path(args.input_csv)
    output_npy = Path(args.output_npy)
    output_metadata = Path(args.output_metadata)

    print("正在加载数据...")
    df = pd.read_csv(input_csv)
    texts = build_texts(df)

    print(f"开始生成向量（共 {len(texts)} 篇）...")
    all_embeddings = []

    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i : i + args.batch_size]

        ok = False
        last_err = None
        for attempt in range(1, args.max_retries + 1):
            try:
                response = client.embeddings.create(input=batch, model=args.model)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                ok = True
                break
            except Exception as e:
                last_err = e
                if attempt < args.max_retries:
                    time.sleep(args.retry_wait_sec)

        if not ok:
            raise RuntimeError(f"批次失败，start={i}, size={len(batch)}, error={last_err}") from last_err

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    if embeddings_array.shape[0] != len(df):
        raise RuntimeError(
            f"向量数量与输入行数不一致: embeddings={embeddings_array.shape[0]}, rows={len(df)}"
        )

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_npy, embeddings_array)
    df.to_csv(output_metadata, index=False, encoding="utf-8-sig")

    print(f"向量生成完毕，形状: {embeddings_array.shape}")
    print(f"保存成功：\n- 向量文件: {output_npy}\n- 元数据文件: {output_metadata}")


if __name__ == "__main__":
    generate_embeddings()
