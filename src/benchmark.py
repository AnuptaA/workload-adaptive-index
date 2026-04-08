from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.utils import compute_recall_at_k

def query_index(
    index: faiss.Index,
    query_vectors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float, float]:
    """Returns (retrieved_indices, mean_latency_ms, p99_latency_ms).

    Queries are run one at a time for consistent per-query timing.
    """
    import time

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
    index_type: str,
    train_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    params: dict | None = None,
) -> dict:
    """# TODO: design build/query methodology before implementing.

    Open question: build once per (dataset, N, index_type) and reuse across
    k/memory_budget/recall_target sweeps, or rebuild each time?
    Signature is fixed; body to be filled in.
    """
    raise NotImplementedError

def run_benchmark(data_dir: Path, results_dir: Path) -> pd.DataFrame:
    """# TODO: implement once benchmark_single methodology is settled.

    Should iterate (dataset, N_fraction, k, memory_budget, recall_target),
    build all three indices per config, record all metrics.
    memory_budget_mb and recall_target are config columns added here —
    not used to filter builds, they are features for the model.
    Saves results_dir/benchmarks.csv.
    """
    raise NotImplementedError
