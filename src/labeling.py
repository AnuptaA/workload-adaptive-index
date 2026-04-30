"""Benchmark labeling by metric objectives (oracle labels per workload group)."""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from src.config import INDEX_ORDER, MEMORY_BUDGET_RATIOS, RANDOM_SEED, RECALL_TARGETS

Objective = Literal["memory", "recall", "latency", "composite", "constrained"]
PureObjective = Literal["memory", "recall", "latency"]

CONFIG_COLS = ["dataset", "n_fraction", "N", "d", "k"]
WEIGHT_COLS = ["w_recall", "w_latency", "w_memory"]
CONSTRAINT_COLS = ["memory_budget_ratio", "memory_budget_mb", "recall_target"]
CONSTRAINT_MODEL_COLS = ["memory_budget_ratio", "recall_target"]

ORACLE_MEMORY_LABEL = "oracle_memory_label"
ORACLE_RECALL_LABEL = "oracle_recall_label"
ORACLE_LATENCY_LABEL = "oracle_latency_label"
COMPOSITE_ORACLE_LABEL = "composite_oracle_label"
CONSTRAINED_ORACLE_LABEL = "constrained_oracle_label"

ORACLE_LABEL_COLS = [ORACLE_MEMORY_LABEL, ORACLE_RECALL_LABEL, ORACLE_LATENCY_LABEL]
DEFAULT_WEIGHT_GRID_STEP = 0.25
DEFAULT_MEMORY_PENALTY_WEIGHT = 100.0
DEFAULT_RECALL_PENALTY_WEIGHT = 100.0


class CompositeWeights(NamedTuple):
    w_recall: float
    w_latency: float
    w_memory: float


def _stable_rank(index_type: str) -> int:
    try:
        return INDEX_ORDER.index(index_type)
    except ValueError:
        raise ValueError(f"Unexpected index_type {index_type!r}; expected one of {INDEX_ORDER}") from None


def validate_composite_weights(
    weights: CompositeWeights | tuple[float, float, float],
    *,
    tolerance: float = 1e-9,
) -> CompositeWeights:
    """Return composite weights after validating simplex constraints."""
    w = CompositeWeights(*[float(v) for v in weights])
    if any(v < -tolerance for v in w):
        raise ValueError(f"composite weights must be nonnegative, got {w}")
    total = sum(w)
    if not np.isclose(total, 1.0, atol=tolerance):
        raise ValueError(f"composite weights must sum to 1, got {total:.12f}")
    return CompositeWeights(
        max(0.0, w.w_recall),
        max(0.0, w.w_latency),
        max(0.0, w.w_memory),
    )


def generate_weight_grid(step: float = DEFAULT_WEIGHT_GRID_STEP) -> pd.DataFrame:
    """Generate recall/latency/memory weight triples on the 2-simplex."""
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    n_steps = round(1.0 / step)
    if not np.isclose(n_steps * step, 1.0):
        raise ValueError("step must evenly divide 1.0")

    rows = []
    for i in range(n_steps + 1):
        for j in range(n_steps - i + 1):
            k = n_steps - i - j
            weights = validate_composite_weights((i * step, j * step, k * step))
            rows.append(dict(zip(WEIGHT_COLS, weights)))
    return pd.DataFrame(rows, columns=WEIGHT_COLS)


def _minmax_normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(hi, lo):
        return pd.Series(0.0, index=values.index)
    return (values - lo) / (hi - lo)


def raw_vector_mb(n: float, d: float) -> float:
    """Raw float32 vector footprint in MB for a workload."""
    return float(n) * float(d) * 4.0 / (1024**2)


def expand_constraint_grid(
    configs: pd.DataFrame,
    *,
    memory_budget_ratios: list[float] | tuple[float, ...] = MEMORY_BUDGET_RATIOS,
    recall_targets: list[float] | tuple[float, ...] = RECALL_TARGETS,
) -> pd.DataFrame:
    """Expand workload configs over memory-budget ratios and recall targets."""
    base = configs[CONFIG_COLS].drop_duplicates(subset=CONFIG_COLS).copy()
    rows: list[pd.DataFrame] = []
    for ratio in memory_budget_ratios:
        for target in recall_targets:
            part = base.copy()
            part["memory_budget_ratio"] = float(ratio)
            part["memory_budget_mb"] = [
                raw_vector_mb(n, d) * float(ratio)
                for n, d in zip(part["N"], part["d"])
            ]
            part["recall_target"] = float(target)
            rows.append(part)
    if not rows:
        return pd.DataFrame(columns=CONFIG_COLS + CONSTRAINT_COLS)
    return pd.concat(rows, ignore_index=True)[CONFIG_COLS + CONSTRAINT_COLS]


def composite_scores(
    group: pd.DataFrame,
    weights: CompositeWeights | tuple[float, float, float],
) -> pd.DataFrame:
    """Return per-index normalized composite scores for one workload config."""
    required = {"index_type", "mean_latency_ms", "index_size_mb", "recall_at_k"}
    missing = required - set(group.columns)
    if missing:
        raise KeyError(f"group dataframe missing columns: {sorted(missing)}")

    w = validate_composite_weights(weights)
    cand = group[list(required)].copy()
    recall_loss = float(cand["recall_at_k"].max()) - cand["recall_at_k"].astype(float)
    cand["recall_loss_norm"] = _minmax_normalize(recall_loss)
    cand["latency_norm"] = _minmax_normalize(cand["mean_latency_ms"])
    cand["memory_norm"] = _minmax_normalize(cand["index_size_mb"])
    cand["composite_score"] = (
        w.w_recall * cand["recall_loss_norm"]
        + w.w_latency * cand["latency_norm"]
        + w.w_memory * cand["memory_norm"]
    )
    return cand


def constraint_scores(
    group: pd.DataFrame,
    memory_budget_mb: float,
    recall_target: float,
    *,
    memory_penalty_weight: float = DEFAULT_MEMORY_PENALTY_WEIGHT,
    recall_penalty_weight: float = DEFAULT_RECALL_PENALTY_WEIGHT,
) -> pd.DataFrame:
    """Return per-index penalty scores for one workload and deployment constraint."""
    required = {"index_type", "mean_latency_ms", "index_size_mb", "recall_at_k"}
    missing = required - set(group.columns)
    if missing:
        raise KeyError(f"group dataframe missing columns: {sorted(missing)}")
    if memory_budget_mb <= 0:
        raise ValueError("memory_budget_mb must be positive")
    if recall_target <= 0:
        raise ValueError("recall_target must be positive")
    if memory_penalty_weight < 0 or recall_penalty_weight < 0:
        raise ValueError("constraint penalty weights must be nonnegative")

    cand = group[list(required)].copy()
    cand["latency_norm"] = _minmax_normalize(cand["mean_latency_ms"])
    cand["memory_overrun"] = np.maximum(
        0.0,
        cand["index_size_mb"].astype(float) / float(memory_budget_mb) - 1.0,
    )
    cand["recall_shortfall"] = np.maximum(
        0.0,
        float(recall_target) - cand["recall_at_k"].astype(float),
    )
    cand["constraint_score"] = (
        cand["latency_norm"]
        + float(memory_penalty_weight) * cand["memory_overrun"]
        + float(recall_penalty_weight) * cand["recall_shortfall"]
    )
    return cand


def score_predicted_constraints(
    predictions: pd.DataFrame,
    memory_budget_mb: float,
    recall_target: float,
    *,
    memory_penalty_weight: float = DEFAULT_MEMORY_PENALTY_WEIGHT,
    recall_penalty_weight: float = DEFAULT_RECALL_PENALTY_WEIGHT,
) -> pd.DataFrame:
    """Score predicted latency/memory/recall rows with the constrained objective."""
    rename_map = {
        "predicted_latency_ms": "mean_latency_ms",
        "predicted_memory_mb": "index_size_mb",
        "predicted_recall": "recall_at_k",
    }
    required = {"index_type", *rename_map}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions dataframe missing columns: {sorted(missing)}")
    scored_input = predictions[["index_type", *rename_map]].rename(columns=rename_map)
    scored = constraint_scores(
        scored_input,
        memory_budget_mb,
        recall_target,
        memory_penalty_weight=memory_penalty_weight,
        recall_penalty_weight=recall_penalty_weight,
    )
    for predicted_col in rename_map:
        scored[predicted_col] = predictions[predicted_col].to_numpy()
    return scored


def select_winner_for_constraints(
    group: pd.DataFrame,
    memory_budget_mb: float,
    recall_target: float,
    *,
    memory_penalty_weight: float = DEFAULT_MEMORY_PENALTY_WEIGHT,
    recall_penalty_weight: float = DEFAULT_RECALL_PENALTY_WEIGHT,
) -> str:
    """Pick the index type minimizing the constrained penalty objective."""
    cand = constraint_scores(
        group,
        memory_budget_mb,
        recall_target,
        memory_penalty_weight=memory_penalty_weight,
        recall_penalty_weight=recall_penalty_weight,
    )
    cand["_rk"] = cand["index_type"].astype(str).map(_stable_rank)
    cand = cand.sort_values(
        by=[
            "constraint_score",
            "memory_overrun",
            "recall_shortfall",
            "mean_latency_ms",
            "recall_at_k",
            "index_size_mb",
            "_rk",
        ],
        ascending=[True, True, True, True, False, True, True],
        kind="mergesort",
    )
    return str(cand.iloc[0]["index_type"])


def select_winner_for_weights(
    group: pd.DataFrame,
    weights: CompositeWeights | tuple[float, float, float],
) -> str:
    """Pick index type minimizing the weighted normalized composite score."""
    cand = composite_scores(group, weights)
    cand["_rk"] = cand["index_type"].astype(str).map(_stable_rank)
    cand = cand.sort_values(
        by=["composite_score", "recall_at_k", "mean_latency_ms", "index_size_mb", "_rk"],
        ascending=[True, False, True, True, True],
        kind="mergesort",
    )
    return str(cand.iloc[0]["index_type"])


def select_winner_for_objective(group: pd.DataFrame, objective: Objective) -> str:
    """Pick index type optimizing the given measured metric among candidates.

    Tie-break order (deterministic):

    - ``memory``: minimize ``index_size_mb``, then ``mean_latency_ms``, then stable INDEX_ORDER.
    - ``recall``: maximize ``recall_at_k``, then minimize ``mean_latency_ms``, minimize
      ``index_size_mb``, then stable INDEX_ORDER.
    - ``latency``: minimize ``mean_latency_ms``, then ``index_size_mb``, then stable INDEX_ORDER.
    """
    required = {"index_type", "mean_latency_ms", "index_size_mb", "recall_at_k"}
    missing = required - set(group.columns)
    if missing:
        raise KeyError(f"group dataframe missing columns: {sorted(missing)}")

    cand = group[list(required)].copy()
    cand["_rk"] = cand["index_type"].astype(str).map(_stable_rank)

    if objective == "memory":
        cand = cand.sort_values(
            by=["index_size_mb", "mean_latency_ms", "_rk"],
            ascending=[True, True, True],
            kind="mergesort",
        )
    elif objective == "recall":
        cand = cand.sort_values(
            by=["recall_at_k", "mean_latency_ms", "index_size_mb", "_rk"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
    elif objective == "latency":
        cand = cand.sort_values(
            by=["mean_latency_ms", "index_size_mb", "_rk"],
            ascending=[True, True, True],
            kind="mergesort",
        )
    elif objective == "composite":
        raise ValueError("composite objective requires select_winner_for_weights(group, weights)")
    else:
        raise ValueError(f"unknown objective {objective!r}")

    return str(cand.iloc[0]["index_type"])


def select_winner(group: pd.DataFrame, objective: Objective | None = None) -> str:
    """Backward-compatible alias defaulting to latency oracle."""
    obj: Objective = objective if objective is not None else "latency"
    group = _restore_group_config_columns(group)
    return select_winner_for_objective(group, obj)


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


def _assign_oracle_labels(group: pd.DataFrame) -> pd.DataFrame:
    group = _restore_group_config_columns(group)
    group = group.copy()
    group[ORACLE_MEMORY_LABEL] = select_winner_for_objective(group, "memory")
    group[ORACLE_RECALL_LABEL] = select_winner_for_objective(group, "recall")
    group[ORACLE_LATENCY_LABEL] = select_winner_for_objective(group, "latency")
    return group


def label_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Assign pure objective oracle labels.

    Raw benchmark measurements are deduped by ``CONFIG_COLS + ["index_type"]`` and
    labels are computed per raw workload.
    """
    raw = df.drop_duplicates(subset=CONFIG_COLS + ["index_type"]).copy()
    return raw.groupby(CONFIG_COLS, group_keys=False).apply(_assign_oracle_labels)


def check_class_distribution(df: pd.DataFrame, label_col: str = ORACLE_LATENCY_LABEL) -> dict[str, float]:
    """Returns label fraction per index type for ``label_col``."""
    counts = df[label_col].value_counts(normalize=True)
    return counts.to_dict()


def balance_labels(
    df: pd.DataFrame,
    label_col: str = ORACLE_LATENCY_LABEL,
    threshold: float = 0.60,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Subsample majority-class configs when any label fraction exceeds threshold."""
    dist = check_class_distribution(df, label_col)
    dominant = [label for label, frac in dist.items() if frac > threshold]

    if not dominant:
        return df

    rng = np.random.default_rng(seed)

    minority_size = min(len(df[df[label_col] == label]) for label in dist if label not in dominant)

    parts = []
    for label in dist:
        subset = df[df[label_col] == label]
        if label in dominant:
            configs = subset[CONFIG_COLS].drop_duplicates()
            n_keep = max(1, int(minority_size / len(subset) * len(configs)))
            chosen = configs.sample(n=min(n_keep, len(configs)), random_state=int(rng.integers(0, 2**31)))
            subset = subset.merge(chosen, on=CONFIG_COLS)
        parts.append(subset)

    return pd.concat(parts).reset_index(drop=True)
