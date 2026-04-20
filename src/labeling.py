import numpy as np
import pandas as pd

from src.config import MEMORY_VIOLATION_WEIGHT, RANDOM_SEED, RECALL_VIOLATION_WEIGHT

CONFIG_COLS = ["dataset", "n_fraction", "N", "d", "k", "memory_budget_mb", "recall_target"]

def compute_violation_score(
    row: pd.Series,
    memory_weight: float = MEMORY_VIOLATION_WEIGHT,
    recall_weight: float = RECALL_VIOLATION_WEIGHT,
) -> float:
    """Weighted sum of constraint violations.

    memory_violation = max(0, peak_memory_mb - memory_budget_mb) / memory_budget_mb
    recall_violation = max(0, recall_target - recall_at_k)
    score = memory_weight * memory_violation + recall_weight * recall_violation
    """
    mem_violation = max(0.0, row["peak_memory_mb"] - row["memory_budget_mb"]) / row["memory_budget_mb"]
    rec_violation = max(0.0, row["recall_target"] - row["recall_at_k"])
    return memory_weight * mem_violation + recall_weight * rec_violation


def _restore_group_config_columns(group: pd.DataFrame) -> pd.DataFrame:
    """Reattach group-by keys when pandas excludes grouping columns in apply()."""
    missing = [col for col in CONFIG_COLS if col not in group.columns]
    if not missing:
        return group

    if not hasattr(group, "name"):
        missing_str = ", ".join(missing)
        raise KeyError(f"Grouped dataframe is missing required config columns: {missing_str}")

    key = group.name
    if not isinstance(key, tuple):
        key = (key,)
    if len(key) != len(CONFIG_COLS):
        raise KeyError("Grouped dataframe is missing config columns and group key shape is unexpected")

    restored = group.copy()
    for col, value in zip(CONFIG_COLS, key):
        restored[col] = value
    return restored

def select_winner(group: pd.DataFrame) -> str:
    """Given rows for one configuration (one row per index_type), return the winning index.

    Among feasible indices (violation_score == 0): argmin(mean_latency_ms).
    If none feasible: argmin(violation_score).
    # TODO: define exact tiebreak rule when violation scores are equal.
    """
    group = _restore_group_config_columns(group)
    scores = group.apply(compute_violation_score, axis=1)
    feasible = group[scores == 0.0]

    if not feasible.empty:
        winner_idx = feasible["mean_latency_ms"].idxmin()
    else:
        winner_idx = scores.idxmin()

    return group.loc[winner_idx, "index_type"]

def label_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Apply select_winner per configuration group.

    Groups by all config columns except index_type. Returns df with added
    'label' column (the winning index type string for each row's config).
    """
    labels = (
        df.groupby(CONFIG_COLS, group_keys=False)
        .apply(_assign_winner_label)
    )
    return labels

def _assign_winner_label(group: pd.DataFrame) -> pd.DataFrame:
    group = _restore_group_config_columns(group)
    winner = select_winner(group)
    group = group.copy()
    group["label"] = winner
    return group

def check_class_distribution(df: pd.DataFrame) -> dict[str, float]:
    """Returns label fraction per index type."""
    counts = df["label"].value_counts(normalize=True)
    return counts.to_dict()

def balance_labels(
    df: pd.DataFrame,
    threshold: float = 0.60,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Subsample majority-class configs when any label fraction exceeds threshold.

    Operates on config-level groups (not individual rows) to avoid splitting
    rows that belong to the same configuration.
    """
    dist = check_class_distribution(df)
    dominant = [label for label, frac in dist.items() if frac > threshold]

    if not dominant:
        return df

    rng = np.random.default_rng(seed)

    minority_size = min(
        len(df[df["label"] == label]) for label in dist if label not in dominant
    )

    parts = []
    for label in dist:
        subset = df[df["label"] == label]
        if label in dominant:
            # subsample to minority_size rows, keeping whole config groups intact
            configs = subset[CONFIG_COLS].drop_duplicates()
            n_keep = max(1, int(minority_size / len(subset) * len(configs)))
            chosen = configs.sample(n=min(n_keep, len(configs)), random_state=int(rng.integers(0, 2**31)))
            subset = subset.merge(chosen, on=CONFIG_COLS)
        parts.append(subset)

    return pd.concat(parts).reset_index(drop=True)
