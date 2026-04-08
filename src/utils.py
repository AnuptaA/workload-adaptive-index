import time
from collections.abc import Callable
from typing import Any

import numpy as np
from memory_profiler import memory_usage

from src.config import RANDOM_SEED

def time_fn(fn: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Returns (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def measure_peak_memory_mb(fn: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Returns (result, peak_mb).

    Runs fn in the current process and tracks peak RSS via memory_profiler.
    interval=0.01 trades resolution for overhead.
    """
    result_holder: list[Any] = []

    def _wrapper() -> None:
        result_holder.append(fn(*args, **kwargs))

    mem = memory_usage(_wrapper, interval=0.01, max_usage=True)
    return result_holder[0], float(mem)

def subsample_vectors(
    vectors: np.ndarray, fraction: float, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Random row subsample without replacement."""
    n = len(vectors)
    k = max(1, int(n * fraction))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return vectors[idx]

def compute_recall_at_k(
    retrieved: np.ndarray, ground_truth: np.ndarray, k: int
) -> float:
    """Fraction of true top-k neighbors found across all queries.

    retrieved: (n_queries, k) int array of returned indices
    ground_truth: (n_queries, >=k) int array of true nearest neighbors
    """
    n_queries = retrieved.shape[0]
    hits = 0
    for i in range(n_queries):
        true_set = set(ground_truth[i, :k].tolist())
        hits += len(set(retrieved[i].tolist()) & true_set)
    return hits / (n_queries * k)
