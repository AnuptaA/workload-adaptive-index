import numpy as np
import pandas as pd

from src.config import INDEX_TYPES, RANDOM_SEED
from src.labeling import CONFIG_COLS

_PERF_COLS = ["mean_latency_ms", "p99_latency_ms", "recall_at_k", "index_size_mb"]

def _lookup_performance(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    """For each row in test_df, look up measured performance of the given index label."""
    result = test_df[CONFIG_COLS].copy()
    result["predicted_index"] = labels

    rows = []
    missing = 0
    for (_, cfg_row), label in zip(test_df.iterrows(), labels):
        mask = benchmarks["index_type"] == label
        for col in CONFIG_COLS:
            mask &= benchmarks[col] == cfg_row[col]
        match = benchmarks[mask]
        if match.empty:
            rows.append({col: float("nan") for col in _PERF_COLS})
            missing += 1
        else:
            rows.append(match.iloc[0][_PERF_COLS].to_dict())

    if missing:
        print(f"Warning: {missing}/{len(labels)} benchmark lookups returned no match.")

    perf_df = pd.DataFrame(rows)
    return pd.concat([result.reset_index(drop=True), perf_df], axis=1)


def mean_metric_for_labels(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
    metric: str,
) -> float:
    """Mean measured ``metric`` across configs for the given index choice per row."""
    perf = _lookup_performance(test_df, benchmarks, labels)
    return float(perf[metric].mean())


def mean_latency_for_labels(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
) -> float:
    """Mean measured ``mean_latency_ms`` across configs for the given index choice per row."""
    return mean_metric_for_labels(test_df, benchmarks, labels, "mean_latency_ms")


def expected_mean_metric_uniform_random(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    metric: str,
) -> float:
    """Expected mean ``metric`` if the index is chosen uniformly over ``INDEX_TYPES``."""
    means: list[float] = []
    for _, cfg in test_df.iterrows():
        mask = pd.Series(True, index=benchmarks.index)
        for col in CONFIG_COLS:
            mask &= benchmarks[col] == cfg[col]
        sub = benchmarks[mask]
        if sub.empty:
            continue
        per_type = sub.groupby("index_type", sort=False)[metric].first()
        if set(per_type.index) != set(INDEX_TYPES):
            continue
        means.append(float(np.mean([float(per_type[t]) for t in INDEX_TYPES])))
    if not means:
        return float("nan")
    return float(np.mean(means))


def expected_mean_latency_uniform_random(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> float:
    """Expected mean latency if the index is chosen uniformly over ``INDEX_TYPES``."""
    return expected_mean_metric_uniform_random(test_df, benchmarks, "mean_latency_ms")


def random_mean_metric_monte_carlo(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    metric: str,
    n_trials: int,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Mean and standard error of the per-trial average ``metric`` (one random draw per config)."""
    rng = np.random.default_rng(seed)
    trial_means: list[float] = []
    for _ in range(n_trials):
        labels = rng.choice(INDEX_TYPES, size=len(test_df)).tolist()
        trial_means.append(mean_metric_for_labels(test_df, benchmarks, labels, metric))
    arr = np.asarray(trial_means, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))


def random_mean_latency_monte_carlo(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    n_trials: int,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Mean and standard error of the per-trial average latency (one random draw per config)."""
    return random_mean_metric_monte_carlo(test_df, benchmarks, "mean_latency_ms", n_trials, seed)


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


def faiss_rule_based_labels(test_df: pd.DataFrame) -> list[str]:
    """Return labels from a simple FAISS-style heuristic."""
    labels: list[str] = []
    for _, row in test_df.iterrows():
        memory_budget_mb = row.get("memory_budget_mb")
        recall_target = row.get("recall_target")
        raw_mb = float(row["N"]) * float(row["d"]) * 4.0 / (1024 ** 2)

        if pd.notna(memory_budget_mb) and raw_mb > float(memory_budget_mb):
            labels.append("IVF_PQ")
        elif pd.notna(recall_target) and float(recall_target) >= 0.95:
            labels.append("HNSW")
        else:
            labels.append("IVF_FLAT")

    return labels


def faiss_rule_based(
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a simple FAISS-style heuristic and look up measured performance.

    If memory_budget_mb is present and raw vectors exceed it, choose IVF_PQ.
    If recall_target is present and high, choose HNSW. Otherwise choose IVF_FLAT.
    """
    labels = faiss_rule_based_labels(test_df)
    return _lookup_performance(test_df, benchmarks, labels)

