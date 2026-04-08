from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import INDEX_TYPES
from src.features import FEATURE_COLS, _INDEX_ONE_HOT_COLS, apply_scaler

# --- Model ---

def train_latency_model(X_train: np.ndarray, y_train: np.ndarray):
    """# TODO: implement with LinearRegression + 5-fold CV."""
    raise NotImplementedError

def train_memory_model(X_train: np.ndarray, y_train: np.ndarray):
    """# TODO: implement with LinearRegression + 5-fold CV."""
    raise NotImplementedError

def train_recall_model(X_train: np.ndarray, y_train: np.ndarray):
    """# TODO: implement with LinearRegression + 5-fold CV."""
    raise NotImplementedError

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
    """Predict lat/mem/rec for all three index types, apply objective function.

    # TODO: fill in objective function call once labeling.py is finalized.
    Currently returns the index with lowest predicted latency among all candidates
    as a placeholder.
    """
    predictions = {t: predict_for_index(workload, t, models, scaler) for t in INDEX_TYPES}
    # TODO: apply compute_violation_score logic from labeling.py here
    return min(predictions, key=lambda t: predictions[t]["latency"])
