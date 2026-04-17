import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    DATASETS,
    INDEX_TYPES,
    IVF_FLAT_PARAMS,
    IVF_PQ_PARAMS,
    HNSW_PARAMS,
    K_VALUES,
    MEMORY_BUDGETS_MB,
    N_FRACTIONS,
    RECALL_TARGETS,
)
from src.data_loader import load_dataset
from src.index_builder import build_index
from src.utils import compute_recall_at_k, measure_peak_memory_mb, subsample_vectors


def query_index(
    index: faiss.Index,
    query_vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float, float]:
    """Returns (retrieved_indices, mean_latency_ms, p99_latency_ms).

    Queries are run one at a time for consistent per-query timing.
    """
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    latencies = []
    all_indices = []

    for q in query_vectors:
        q_row = q[np.newaxis, :]
        t0 = time.perf_counter()
        _, I = index.search(q_row, k)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_indices.append(I[0])

    retrieved = np.array(all_indices, dtype=np.int64)
    mean_lat = float(np.mean(latencies))
    p99_lat = float(np.percentile(latencies, 99))
    return retrieved, mean_lat, p99_lat


def benchmark_single(
    index: faiss.Index,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> dict:
    """Query a pre-built index and return performance metrics for one k value."""
    retrieved, mean_lat, p99_lat = query_index(index, query_vectors, k)
    recall = compute_recall_at_k(retrieved, ground_truth, k)
    return {
        "k": k,
        "mean_latency_ms": mean_lat,
        "p99_latency_ms": p99_lat,
        "recall_at_k": recall,
    }


def _adapt_params(index_type: str, n: int) -> dict:
    """Cap nlist to n // 39 for small N.

    FAISS k-means needs >= 39 * nlist training points. nprobe is also capped
    to nlist so it never exceeds the number of clusters.
    """
    if index_type == "HNSW":
        return dict(HNSW_PARAMS)

    base = dict(IVF_FLAT_PARAMS) if index_type == "IVF_FLAT" else dict(IVF_PQ_PARAMS)
    max_nlist = max(1, n // 39)
    base["nlist"] = min(base["nlist"], max_nlist)
    base["nprobe"] = min(base["nprobe"], base["nlist"])
    return base


def _timed_build(
    index_type: str, vectors: np.ndarray, params: dict
) -> tuple[faiss.Index, float]:
    """Build index and return (index, build_time_s).

    Returned as a tuple so measure_peak_memory_mb captures both the index
    object and elapsed time in a single build pass.
    """
    t0 = time.perf_counter()
    index = build_index(index_type, vectors, params)
    return index, time.perf_counter() - t0


def run_benchmark(data_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Build each (dataset, N_fraction, index_type) once, query per k, cross-join constraints.

    memory_budget_mb and recall_target are added as columns — they don't affect
    index building or querying, only the downstream labeling objective.
    Saves results_dir/benchmarks.csv and returns the dataframe.
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_builds = len(DATASETS) * len(N_FRACTIONS) * len(INDEX_TYPES)

    with tqdm(total=n_builds, desc="benchmarking") as bar:
        for dataset_name in DATASETS:
            train, queries, gt = load_dataset(dataset_name, data_dir)
            d = train.shape[1]

            for fraction in N_FRACTIONS:
                sub_train = subsample_vectors(train, fraction)
                n = len(sub_train)

                for index_type in INDEX_TYPES:
                    params = _adapt_params(index_type, n)
                    (index, build_time_s), peak_mb = measure_peak_memory_mb(
                        _timed_build, index_type, sub_train, params
                    )

                    for k in K_VALUES:
                        metrics = benchmark_single(index, queries, gt, k)

                        for mem_budget in MEMORY_BUDGETS_MB:
                            for recall_target in RECALL_TARGETS:
                                rows.append({
                                    "dataset": dataset_name,
                                    "n_fraction": fraction,
                                    "N": n,
                                    "d": d,
                                    "k": k,
                                    "memory_budget_mb": mem_budget,
                                    "recall_target": recall_target,
                                    "index_type": index_type,
                                    "mean_latency_ms": metrics["mean_latency_ms"],
                                    "p99_latency_ms": metrics["p99_latency_ms"],
                                    "recall_at_k": metrics["recall_at_k"],
                                    "peak_memory_mb": peak_mb,
                                    "build_time_s": build_time_s,
                                })

                    bar.update(1)

    df = pd.DataFrame(rows)
    out = results_dir / "benchmarks.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    return df
