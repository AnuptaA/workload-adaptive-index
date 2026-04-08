import numpy as np
import pandas as pd

from src.config import INDEX_TYPES, RANDOM_SEED

_CONFIG_COLS = ["dataset", "n_fraction", "N", "d", "k", "memory_budget_mb", "recall_target"]
_PERF_COLS = ["mean_latency_ms", "p99_latency_ms", "recall_at_k", "peak_memory_mb"]

def _lookup_performance(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    """For each row in test_df, look up measured performance of the given index label."""
    result = test_df[_CONFIG_COLS].copy()
    result["predicted_index"] = labels

    rows = []
    for (_, cfg_row), label in zip(test_df.iterrows(), labels):
        mask = benchmarks["index_type"] == label
        for col in _CONFIG_COLS:
            mask &= benchmarks[col] == cfg_row[col]
        match = benchmarks[mask]
        if match.empty:
            rows.append({col: float("nan") for col in _PERF_COLS})
        else:
            rows.append(match.iloc[0][_PERF_COLS].to_dict())

    perf_df = pd.DataFrame(rows)
    return pd.concat([result.reset_index(drop=True), perf_df], axis=1)

def always_hnsw(
    test_df: pd.DataFrame, benchmarks: pd.DataFrame
) -> pd.DataFrame:
    """Look up measured HNSW performance for each test config."""
    labels = ["HNSW"] * len(test_df)
    return _lookup_performance(test_df, benchmarks, labels)

def random_baseline(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Randomly assign an index type per config, look up measured performance."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(INDEX_TYPES, size=len(test_df)).tolist()
    return _lookup_performance(test_df, benchmarks, labels)

def faiss_rule_based(
    test_df: pd.DataFrame, benchmarks: pd.DataFrame
) -> pd.DataFrame:
    """Apply FAISS-style heuristic to each config, look up measured performance.

    # TODO: define heuristic rules before implementing.
    """
    raise NotImplementedError
