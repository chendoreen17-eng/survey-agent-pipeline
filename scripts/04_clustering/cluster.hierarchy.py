import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage


def parse_args():
    parser = argparse.ArgumentParser(description="Build L1 hierarchy from L2 clusters.")
    parser.add_argument("--input-csv", required=True, help="CSV with final_domain_id column")
    parser.add_argument("--embeddings-npy", required=True, help="Numpy file aligned with input rows")
    parser.add_argument("--num-l1", type=int, default=6, help="Number of parent domains")
    parser.add_argument("--output-csv", required=True, help="Output CSV with parent_domain_id")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    x = np.load(args.embeddings_npy)

    if "final_domain_id" not in df.columns:
        raise ValueError("input csv must contain final_domain_id")
    if len(df) != x.shape[0]:
        raise ValueError(f"row mismatch: csv={len(df)} vs embeddings={x.shape[0]}")

    l2_ids = sorted(df["final_domain_id"].dropna().unique().tolist())
    centroids = []
    for cid in l2_ids:
        mask = df["final_domain_id"] == cid
        centroids.append(x[mask.to_numpy()].mean(axis=0))
    centroids = np.array(centroids)

    z = linkage(centroids, method="ward")
    parents = fcluster(z, t=args.num_l1, criterion="maxclust")
    l2_to_parent = {int(cid): int(pid) for cid, pid in zip(l2_ids, parents)}

    df["parent_domain_id"] = df["final_domain_id"].map(lambda v: l2_to_parent.get(int(v), -1))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] hierarchy saved: {out_path} rows={len(df)} l1={args.num_l1}")


if __name__ == "__main__":
    main()
