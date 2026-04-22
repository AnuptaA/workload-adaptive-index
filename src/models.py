from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.config import INDEX_TYPES
from src.features import FEATURE_COLS, _INDEX_ONE_HOT_COLS, apply_scaler
from src.labeling import choose_index_from_metrics

# --- Model ---

def _fit_linear_regression_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    name: str,
    cv: int = 5,
) -> LinearRegression:
    """Fit ``LinearRegression`` on all of ``X_train``; print CV RMSE for diagnostics."""
    n_samples = len(X_train)
    folds = min(cv, n_samples) if n_samples >= 2 else 1
    model = LinearRegression()
    if folds >= 2:
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=folds,
            scoring="neg_root_mean_squared_error",
        )
        rmse_mean = float(-np.mean(scores))
        rmse_std = float(np.std(scores))
        print(f"  {name} {folds}-fold CV RMSE: {rmse_mean:.6f} (std {rmse_std:.6f})")
    else:
        print(f"  {name}: skipping CV (n_samples={n_samples})")
    model.fit(X_train, y_train)
    return model


def train_latency_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """``LinearRegression`` with K-fold CV RMSE reported on the training set."""
    return _fit_linear_regression_with_cv(X_train, y_train, "latency")

def train_memory_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """``LinearRegression`` with K-fold CV RMSE reported on the training set."""
    return _fit_linear_regression_with_cv(X_train, y_train, "memory")

def train_recall_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """``LinearRegression`` with K-fold CV RMSE reported on the training set."""
    return _fit_linear_regression_with_cv(X_train, y_train, "recall")

# --- Artifact I/O ---

def save_artifacts(
    models: dict, scaler: StandardScaler, artifacts_dir: Path
) -> None:
    """Save models and scaler to artifacts_dir using joblib."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, artifacts_dir / f"{name}.joblib")
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")

def load_artifacts(artifacts_dir: Path) -> tuple[dict, StandardScaler]:
    """Load and return (models_dict, scaler) from artifacts_dir."""
    artifacts_dir = Path(artifacts_dir)
    model_names = ["latency_model", "memory_model", "recall_model"]
    models = {name: joblib.load(artifacts_dir / f"{name}.joblib") for name in model_names}
    scaler = joblib.load(artifacts_dir / "scaler.joblib")
    return models, scaler

# --- Inference ---

def predict_for_index(
    workload: dict,
    index_type: str,
    models: dict,
    scaler: StandardScaler,
) -> dict[str, float]:
    """Build and scale feature vector for one index type, run all three regressors.

    workload keys: N, d, k, memory_budget_mb, recall_target
    Returns {"latency": float, "memory": float, "recall": float}.
    """
    feature_values = [float(workload[col]) for col in FEATURE_COLS]
    one_hot = [1.0 if t == index_type else 0.0 for t in INDEX_TYPES]
    x = np.array(feature_values + one_hot, dtype=np.float32).reshape(1, -1)
    x_scaled = apply_scaler(x, scaler)

    return {
        "latency": float(models["latency_model"].predict(x_scaled)[0]),
        "memory": float(models["memory_model"].predict(x_scaled)[0]),
        "recall": float(models["recall_model"].predict(x_scaled)[0]),
    }

def select_index(
    workload: dict,
    models: dict,
    scaler: StandardScaler,
) -> str:
    """Predict lat/mem/rec per index type, then apply the labeling objective.

    Matches ``labeling.select_winner``: weighted constraint violations
    (``compute_violation_score``), feasible-first with minimum predicted
    ``mean_latency_ms``, otherwise minimum violation score.
    """
    mem_budget = float(workload["memory_budget_mb"])
    recall_target = float(workload["recall_target"])
    rows = []
    for t in INDEX_TYPES:
        pred = predict_for_index(workload, t, models, scaler)
        rows.append({
            "index_type": t,
            "mean_latency_ms": pred["latency"],
            "peak_memory_mb": pred["memory"],
            "recall_at_k": pred["recall"],
            "memory_budget_mb": mem_budget,
            "recall_target": recall_target,
        })
    return choose_index_from_metrics(pd.DataFrame(rows))
