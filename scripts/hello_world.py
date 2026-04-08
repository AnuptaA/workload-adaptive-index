"""End-to-end smoke test: load SIFT-1M, build HNSW, query, print stats."""

import argparse
from pathlib import Path

from src.config import DATA_DIR, HNSW_PARAMS
from src.data_loader import load_dataset
from src.index_builder import build_index
from src.utils import compute_recall_at_k, time_fn

def main(data_dir: Path) -> None:
    print("Loading sift-1M...")
    train, queries, gt = load_dataset("sift-1M", data_dir)
    print(f"  train: {train.shape}, queries: {queries.shape}, gt: {gt.shape}")

    print("Building HNSW index...")
    index, build_time = time_fn(build_index, "HNSW", train)
    print(f"  build time: {build_time:.2f}s")

    k = 10
    print(f"Querying {len(queries)} vectors (k={k})...")
    from src.benchmark import query_index
    retrieved, mean_lat, p99_lat = query_index(index, queries, k)

    recall = compute_recall_at_k(retrieved, gt, k)
    print(f"  mean latency: {mean_lat:.3f}ms  p99: {p99_lat:.3f}ms  recall@{k}: {recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()
    main(Path(args.data_dir))
