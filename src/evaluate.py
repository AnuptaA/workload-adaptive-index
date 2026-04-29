from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import (
    expected_mean_metric_uniform_random,
    faiss_rule_based_labels,
    mean_metric_for_labels,
    random_mean_metric_monte_carlo,
)
from src.config import RANDOM_SEED
from src.labeling import (
    CONFIG_COLS,
    CONSTRAINED_ORACLE_LABEL,
    CompositeWeights,
    Objective,
    composite_scores,
    constraint_scores,
    select_winner_for_weights,
    select_winner_for_constraints,
    validate_composite_weights,
)


OBJECTIVE_METRIC: dict[Objective, str] = {
    "memory": "index_size_mb",
    "recall": "recall_at_k",
    "latency": "mean_latency_ms",
    "composite": "composite_score",
    "constrained": "constraint_score",
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Scatter plot of predicted vs actual; saves to ``save_path``."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.35, s=12)
    lims = [
        min(float(np.min(y_true)), float(np.min(y_pred))),
        max(float(np.max(y_true)), float(np.max(y_pred))),
    ]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_index_selection(
    predicted_labels: list[str],
    test_df: pd.DataFrame,
    oracle_col: str,
) -> dict[str, float]:
    """Accuracy of predicted index vs ``oracle_col`` (one prediction per row of ``test_df``)."""
    if len(predicted_labels) != len(test_df):
        raise ValueError("predicted_labels length must match test_df rows")
    labels = test_df[oracle_col].tolist()
    correct = sum(1 for p, g in zip(predicted_labels, labels) if p == g)
    return {"accuracy": correct / len(labels)}


def index_selection_metric_comparison(
    objective: Objective,
    test_configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    predicted_labels: list[str],
    *,
    oracle_col: str,
    random_mc_trials: int = 400,
    random_mc_seed: int = RANDOM_SEED,
) -> dict[str, float | str]:
    """Oracle / model / baselines: mean measured metric for the objective on held-out configs.

    Oracle uses the tabular column ``oracle_col``. Model uses ``predicted_labels``.
    """
    if len(predicted_labels) != len(test_configs):
        raise ValueError("predicted_labels length must match test_configs rows")

    metric = OBJECTIVE_METRIC[objective]
    oracle_labels = test_configs[oracle_col].tolist()
    hnsw_labels = ["HNSW"] * len(test_configs)

    mc_mean, mc_se = random_mean_metric_monte_carlo(
        test_configs,
        benchmarks,
        metric,
        random_mc_trials,
        random_mc_seed,
    )

    return {
        "metric": metric,
        "oracle_mean": mean_metric_for_labels(test_configs, benchmarks, oracle_labels, metric),
        "model_mean": mean_metric_for_labels(test_configs, benchmarks, predicted_labels, metric),
        "always_hnsw_mean": mean_metric_for_labels(test_configs, benchmarks, hnsw_labels, metric),
        "uniform_random_expected_mean": expected_mean_metric_uniform_random(
            test_configs,
            benchmarks,
            metric,
        ),
        "random_policy_mc_mean": mc_mean,
        "random_policy_mc_se": mc_se,
    }


def composite_score_for_labels(
    test_configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
    weights: CompositeWeights | tuple[float, float, float],
) -> list[float]:
    """Look up normalized composite score for each selected index label."""
    if len(labels) != len(test_configs):
        raise ValueError("labels length must match test_configs rows")
    w = validate_composite_weights(weights)
    scores: list[float] = []
    missing = 0
    for (_, cfg_row), label in zip(test_configs.iterrows(), labels):
        mask = pd.Series(True, index=benchmarks.index)
        for col in CONFIG_COLS:
            mask &= benchmarks[col] == cfg_row[col]
        group = benchmarks[mask]
        if group.empty:
            scores.append(float("nan"))
            missing += 1
            continue
        scored = composite_scores(group, w)
        selected = scored[scored["index_type"] == label]
        if selected.empty:
            scores.append(float("nan"))
            missing += 1
        else:
            scores.append(float(selected.iloc[0]["composite_score"]))
    if missing:
        print(f"Warning: {missing}/{len(labels)} composite score lookups returned no match.")
    return scores


def mean_composite_score_for_labels(
    test_configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
    weights: CompositeWeights | tuple[float, float, float],
) -> float:
    """Mean normalized composite score across configs for selected labels."""
    return float(np.nanmean(composite_score_for_labels(test_configs, benchmarks, labels, weights)))


def expected_mean_composite_score_uniform_random(
    test_configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    weights: CompositeWeights | tuple[float, float, float],
) -> float:
    """Expected composite score if each index is chosen uniformly per config."""
    w = validate_composite_weights(weights)
    means: list[float] = []
    for _, cfg_row in test_configs.iterrows():
        mask = pd.Series(True, index=benchmarks.index)
        for col in CONFIG_COLS:
            mask &= benchmarks[col] == cfg_row[col]
        group = benchmarks[mask]
        if group.empty:
            continue
        means.append(float(composite_scores(group, w)["composite_score"].mean()))
    if not means:
        return float("nan")
    return float(np.mean(means))


def composite_index_selection_comparison(
    test_configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    predicted_labels: list[str],
    weights: CompositeWeights | tuple[float, float, float],
) -> dict[str, float | str]:
    """Oracle / model / baselines by mean normalized composite score."""
    if len(predicted_labels) != len(test_configs):
        raise ValueError("predicted_labels length must match test_configs rows")
    w = validate_composite_weights(weights)
    oracle_labels = [
        select_winner_for_weights(
            benchmarks[
                (benchmarks[CONFIG_COLS] == row[CONFIG_COLS]).all(axis=1)
            ],
            w,
        )
        for _, row in test_configs.iterrows()
    ]
    hnsw_labels = ["HNSW"] * len(test_configs)
    return {
        "metric": "composite_score",
        "oracle_mean": mean_composite_score_for_labels(test_configs, benchmarks, oracle_labels, w),
        "model_mean": mean_composite_score_for_labels(test_configs, benchmarks, predicted_labels, w),
        "always_hnsw_mean": mean_composite_score_for_labels(test_configs, benchmarks, hnsw_labels, w),
        "uniform_random_expected_mean": expected_mean_composite_score_uniform_random(
            test_configs,
            benchmarks,
            w,
        ),
    }


def constraint_outcomes_for_labels(
    test_configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    """Look up penalty objective outcomes for each selected index label."""
    required = set(CONFIG_COLS + ["memory_budget_mb", "recall_target"])
    missing = required - set(test_configs.columns)
    if missing:
        raise KeyError(f"test_configs missing columns: {sorted(missing)}")
    if len(labels) != len(test_configs):
        raise ValueError("labels length must match test_configs rows")

    rows = []
    missing_count = 0
    for (_, cfg_row), label in zip(test_configs.iterrows(), labels):
        mask = pd.Series(True, index=benchmarks.index)
        for col in CONFIG_COLS:
            mask &= benchmarks[col] == cfg_row[col]
        group = benchmarks[mask]
        if group.empty:
            rows.append({
                "constraint_score": np.nan,
                "memory_overrun": np.nan,
                "recall_shortfall": np.nan,
                "mean_latency_ms": np.nan,
                "recall_at_k": np.nan,
                "index_size_mb": np.nan,
            })
            missing_count += 1
            continue
        scored = constraint_scores(
            group,
            float(cfg_row["memory_budget_mb"]),
            float(cfg_row["recall_target"]),
        )
        selected = scored[scored["index_type"] == label]
        if selected.empty:
            rows.append({
                "constraint_score": np.nan,
                "memory_overrun": np.nan,
                "recall_shortfall": np.nan,
                "mean_latency_ms": np.nan,
                "recall_at_k": np.nan,
                "index_size_mb": np.nan,
            })
            missing_count += 1
        else:
            rows.append(
                selected.iloc[0][
                    [
                        "constraint_score",
                        "memory_overrun",
                        "recall_shortfall",
                        "mean_latency_ms",
                        "recall_at_k",
                        "index_size_mb",
                    ]
                ].to_dict()
            )
    if missing_count:
        print(f"Warning: {missing_count}/{len(labels)} constraint lookups returned no match.")

    out = pd.DataFrame(rows)
    out["memory_budget_satisfied"] = out["memory_overrun"] <= 0
    out["recall_target_satisfied"] = out["recall_shortfall"] <= 0
    out["constraints_satisfied"] = out["memory_budget_satisfied"] & out["recall_target_satisfied"]
    return out


def summarize_constraint_outcomes(outcomes: pd.DataFrame) -> dict[str, float]:
    """Aggregate penalty objective outcomes for a strategy."""
    return {
        "mean_objective_score": float(outcomes["constraint_score"].mean()),
        "memory_budget_satisfaction_rate": float(outcomes["memory_budget_satisfied"].mean()),
        "recall_target_satisfaction_rate": float(outcomes["recall_target_satisfied"].mean()),
        "constraint_satisfaction_rate": float(outcomes["constraints_satisfied"].mean()),
        "mean_latency_ms": float(outcomes["mean_latency_ms"].mean()),
        "mean_memory_overrun": float(outcomes["memory_overrun"].mean()),
        "mean_recall_shortfall": float(outcomes["recall_shortfall"].mean()),
    }


def constrained_index_selection_comparison(
    test_configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    predicted_labels: list[str],
) -> dict[str, float | str]:
    """Oracle / model / baselines by mean constrained penalty score."""
    if len(predicted_labels) != len(test_configs):
        raise ValueError("predicted_labels length must match test_configs rows")

    if CONSTRAINED_ORACLE_LABEL in test_configs.columns:
        oracle_labels = test_configs[CONSTRAINED_ORACLE_LABEL].astype(str).tolist()
    else:
        oracle_labels = [
            select_winner_for_constraints(
                benchmarks[
                    (benchmarks[CONFIG_COLS] == row[CONFIG_COLS]).all(axis=1)
                ],
                float(row["memory_budget_mb"]),
                float(row["recall_target"]),
            )
            for _, row in test_configs.iterrows()
        ]

    strategies = {
        "oracle": oracle_labels,
        "model": predicted_labels,
        "rule_based": faiss_rule_based_labels(test_configs),
        "always_hnsw": ["HNSW"] * len(test_configs),
    }
    report: dict[str, float | str] = {"metric": "constraint_score"}
    for name, labels in strategies.items():
        outcomes = constraint_outcomes_for_labels(test_configs, benchmarks, labels)
        summary = summarize_constraint_outcomes(outcomes)
        for key, value in summary.items():
            report[f"{name}_{key}"] = value
    return report


def index_selection_latency_comparison(
    test_configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    predicted_labels: list[str],
    *,
    oracle_col: str,
    random_mc_trials: int = 400,
    random_mc_seed: int = RANDOM_SEED,
) -> dict[str, float | str]:
    """Latency objective convenience wrapper (keys match legacy names where useful)."""
    r = index_selection_metric_comparison(
        "latency",
        test_configs,
        benchmarks,
        predicted_labels,
        oracle_col=oracle_col,
        random_mc_trials=random_mc_trials,
        random_mc_seed=random_mc_seed,
    )
    return {
        "oracle_mean_latency_ms": float(r["oracle_mean"]),
        "model_mean_latency_ms": float(r["model_mean"]),
        "always_hnsw_mean_latency_ms": float(r["always_hnsw_mean"]),
        "uniform_random_expected_mean_latency_ms": float(r["uniform_random_expected_mean"]),
        "random_policy_mc_mean_latency_ms": float(r["random_policy_mc_mean"]),
        "random_policy_mc_se_latency_ms": float(r["random_policy_mc_se"]),
    }
