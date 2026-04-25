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
from src.utils import compute_recall_at_k, subsample_vectors


def _format_params(params: dict) -> str:
    """Render index params in a stable, compact format for logs."""
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _log(verbose: bool, message: str) -> None:
    """Print benchmark progress messages only when verbose logging is enabled."""
    if verbose:
        print(message, flush=True)


def query_index(
    index: faiss.Index,
    query_vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float, float]:
    """Returns (retrieved_indices, mean_latency_ms, p99_latency_ms).

    Queries are run one at a time for consistent per-query timing.
    """
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    n_queries = len(query_vectors)
    latencies = np.empty(n_queries, dtype=np.float64)
    retrieved = np.empty((n_queries, k), dtype=np.int64)

    for i, q in enumerate(query_vectors):
        q_row = q[np.newaxis, :]
        t0 = time.perf_counter()
        _, I = index.search(q_row, k)
        latencies[i] = (time.perf_counter() - t0) * 1000
        retrieved[i] = I[0]

    mean_lat = float(np.mean(latencies))
    p99_lat = float(np.percentile(latencies, 99))
    return retrieved, mean_lat, p99_lat


def compute_exact_ground_truth(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute exact top-k neighbors for the current indexed base vectors."""
    base_vectors = np.ascontiguousarray(base_vectors, dtype=np.float32)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    exact_index = faiss.IndexFlatL2(base_vectors.shape[1])
    exact_index.add(base_vectors)
    _, neighbors = exact_index.search(query_vectors, k)
    return neighbors


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
    """Build index and return (index, build_time_s)."""
    t0 = time.perf_counter()
    index = build_index(index_type, vectors, params)
    return index, time.perf_counter() - t0


def _serialized_index_size_mb(index: faiss.Index) -> float:
    """Return deployed FAISS index footprint in MB via serialization."""
    serialized = faiss.serialize_index(index)
    return float(serialized.nbytes) / (1024 ** 2)


def run_benchmark(
    data_dir: Path,
    results_dir: Path,
    verbose: bool = False,
) -> pd.DataFrame:
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
            train, queries, _ = load_dataset(dataset_name, data_dir)
            queries = np.ascontiguousarray(queries, dtype=np.float32)
            d = train.shape[1]

            for fraction in N_FRACTIONS:
                sub_train = subsample_vectors(train, fraction)
                n = len(sub_train)
                exact_gt = compute_exact_ground_truth(sub_train, queries, max(K_VALUES))
                _log(
                    verbose,
                    f"[benchmark] dataset={dataset_name} n_fraction={fraction:.2f} "
                    f"N={n} d={d} exact_gt_k={max(K_VALUES)}",
                )

                for index_type in INDEX_TYPES:
                    params = _adapt_params(index_type, n)
                    index, build_time_s = _timed_build(index_type, sub_train, params)
                    index_size_mb = _serialized_index_size_mb(index)
                    _log(
                        verbose,
                        f"[build] dataset={dataset_name} index={index_type} "
                        f"n_fraction={fraction:.2f} params=({_format_params(params)}) "
                        f"index_size_mb={index_size_mb:.2f} build_time_s={build_time_s:.3f}",
                    )

                    for k in K_VALUES:
                        metrics = benchmark_single(index, queries, exact_gt, k)
                        _log(
                            verbose,
                            f"[query] dataset={dataset_name} index={index_type} "
                            f"n_fraction={fraction:.2f} k={k} "
                            f"recall_at_k={metrics['recall_at_k']:.4f} "
                            f"mean_latency_ms={metrics['mean_latency_ms']:.4f} "
                            f"p99_latency_ms={metrics['p99_latency_ms']:.4f} "
                            f"index_size_mb={index_size_mb:.2f}",
                        )

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
                                    "index_size_mb": index_size_mb,
                                    "build_time_s": build_time_s,
                                })

                    bar.update(1)

    df = pd.DataFrame(rows)
    out = results_dir / "benchmarks.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    return df
