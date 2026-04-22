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


def mean_latency_for_labels(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
) -> float:
    """Mean measured ``mean_latency_ms`` across configs for the given index choice per row."""
    perf = _lookup_performance(test_df, benchmarks, labels)
    return float(perf["mean_latency_ms"].mean())


def expected_mean_latency_uniform_random(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> float:
    """Expected mean latency if the index is chosen uniformly over ``INDEX_TYPES``.

    For each config, this is the average of measured ``mean_latency_ms`` across
    the three index types (one value per type; benchmarks repeat the same metric
    across ``memory_budget_mb`` / ``recall_target``).
    """
    means: list[float] = []
    for _, cfg in test_df.iterrows():
        mask = pd.Series(True, index=benchmarks.index)
        for col in _CONFIG_COLS:
            mask &= benchmarks[col] == cfg[col]
        sub = benchmarks[mask]
        if sub.empty:
            continue
        per_type = sub.groupby("index_type", sort=False)["mean_latency_ms"].first()
        if set(per_type.index) != set(INDEX_TYPES):
            continue
        means.append(float(np.mean([float(per_type[t]) for t in INDEX_TYPES])))
    if not means:
        return float("nan")
    return float(np.mean(means))


def random_mean_latency_monte_carlo(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    n_trials: int,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Mean and standard error of the per-trial average latency (one random draw per config)."""
    rng = np.random.default_rng(seed)
    trial_means: list[float] = []
    for _ in range(n_trials):
        labels = rng.choice(INDEX_TYPES, size=len(test_df)).tolist()
        trial_means.append(mean_latency_for_labels(test_df, benchmarks, labels))
    arr = np.asarray(trial_means, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))


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
