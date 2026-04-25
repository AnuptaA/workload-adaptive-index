from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.baselines import (
    expected_mean_latency_uniform_random,
    mean_latency_for_labels,
    random_mean_latency_monte_carlo,
)
from src.config import MEMORY_VIOLATION_WEIGHT, RANDOM_SEED, RECALL_VIOLATION_WEIGHT
from src.features import apply_scaler
from src.labeling import CONFIG_COLS, compute_violation_score


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


def evaluate_regressors(
    models: dict,
    scaler: StandardScaler,
    X: np.ndarray,
    y_latency: np.ndarray,
    y_memory: np.ndarray,
    y_recall: np.ndarray,
    split_name: str,
) -> dict[str, float]:
    """RMSE for each regressor on a split (raw ``X``, same scaler as training)."""
    X_scaled = apply_scaler(X, scaler)
    pred_lat = models["latency_model"].predict(X_scaled)
    pred_mem = models["memory_model"].predict(X_scaled)
    pred_rec = models["recall_model"].predict(X_scaled)
    return {
        f"{split_name}_latency_rmse": rmse(y_latency, pred_lat),
        f"{split_name}_memory_rmse": rmse(y_memory, pred_mem),
        f"{split_name}_recall_rmse": rmse(y_recall, pred_rec),
    }


def evaluate_index_selection(
    predicted_labels: list[str],
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> dict[str, float]:
    """Accuracy of predicted index vs ``label`` (one prediction per row of ``test_df``)."""
    if len(predicted_labels) != len(test_df):
        raise ValueError("predicted_labels length must match test_df rows")
    labels = test_df["label"].tolist()
    correct = sum(1 for p, g in zip(predicted_labels, labels) if p == g)
    return {"accuracy": correct / len(labels)}


def index_selection_latency_comparison(
    test_configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    predicted_labels: list[str],
    *,
    random_mc_trials: int = 400,
    random_mc_seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Oracle / model / baselines: mean measured ``mean_latency_ms`` on held-out configs.

    Oracle uses the tabular ``label`` (winner from benchmarks). Model uses
    ``predicted_labels``. Always-HNSW and random baselines use the same
    benchmark lookup table (not regressor predictions).
    """
    if len(predicted_labels) != len(test_configs):
        raise ValueError("predicted_labels length must match test_configs rows")

    oracle_labels = test_configs["label"].tolist()
    hnsw_labels = ["HNSW"] * len(test_configs)

    mc_mean, mc_se = random_mean_latency_monte_carlo(
        test_configs,
        benchmarks,
        random_mc_trials,
        random_mc_seed,
    )

    return {
        "oracle_mean_latency_ms": mean_latency_for_labels(
            test_configs, benchmarks, oracle_labels
        ),
        "model_mean_latency_ms": mean_latency_for_labels(
            test_configs, benchmarks, predicted_labels
        ),
        "always_hnsw_mean_latency_ms": mean_latency_for_labels(
            test_configs, benchmarks, hnsw_labels
        ),
        "uniform_random_expected_mean_latency_ms": expected_mean_latency_uniform_random(
            test_configs, benchmarks
        ),
        "random_policy_mc_mean_latency_ms": mc_mean,
        "random_policy_mc_se_latency_ms": mc_se,
    }


def constraint_violation_rate(
    predicted_labels: list[str],
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
    memory_weight: float = MEMORY_VIOLATION_WEIGHT,
    recall_weight: float = RECALL_VIOLATION_WEIGHT,
) -> float:
    """Fraction of predictions whose *measured* row violates deployment constraints."""
    n_viol = 0
    for pred_label, (_, cfg_row) in zip(predicted_labels, test_df.iterrows()):
        mask = benchmarks["index_type"] == pred_label
        for col in CONFIG_COLS:
            mask &= benchmarks[col] == cfg_row[col]
        match = benchmarks[mask]
        if match.empty:
            n_viol += 1
            continue
        row = match.iloc[0]
        synthetic = pd.Series({
            "index_size_mb": row["index_size_mb"],
            "memory_budget_mb": cfg_row["memory_budget_mb"],
            "recall_at_k": row["recall_at_k"],
            "recall_target": cfg_row["recall_target"],
        })
        if compute_violation_score(synthetic, memory_weight, recall_weight) > 0.0:
            n_viol += 1
    return n_viol / len(predicted_labels) if predicted_labels else 0.0
