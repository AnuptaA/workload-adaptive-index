"""Selector and performance models for workload-adaptive index choice."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    INDEX_TYPES,
    MEMORY_BUDGET_RATIOS,
    RANDOM_SEED,
    RECALL_TARGETS,
    SELECTOR_NUM_COLS,
    WORKLOAD_COLS,
)
from src.labeling import (
    CONSTRAINT_MODEL_COLS,
    Objective,
    ORACLE_LATENCY_LABEL,
    ORACLE_MEMORY_LABEL,
    ORACLE_RECALL_LABEL,
    WEIGHT_COLS,
    raw_vector_mb,
    score_predicted_constraints,
)

_SELECTOR_KEYS = (
    "memory_selector_model",
    "recall_selector_model",
    "latency_selector_model",
    "latency_regressor_model",
    "memory_regressor_model",
    "recall_regressor_model",
)

_ARTIFACT_FILENAMES = {
    "memory_selector_model": "memory_selector_model.joblib",
    "recall_selector_model": "recall_selector_model.joblib",
    "latency_selector_model": "latency_selector_model.joblib",
    "latency_regressor_model": "latency_regressor_model.joblib",
    "memory_regressor_model": "memory_regressor_model.joblib",
    "recall_regressor_model": "recall_regressor_model.joblib",
}

_OBJECTIVE_TO_KEY: dict[Objective, str] = {
    "memory": "memory_selector_model",
    "recall": "recall_selector_model",
    "latency": "latency_selector_model",
}

_ORACLE_COL = {
    "memory": ORACLE_MEMORY_LABEL,
    "recall": ORACLE_RECALL_LABEL,
    "latency": ORACLE_LATENCY_LABEL,
}


def make_selector_pipeline(
    *,
    numeric_cols: list[str] = SELECTOR_NUM_COLS,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Preprocess selector features and train a multiclass RF for index choice."""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["dataset"],
            ),
        ],
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


def make_performance_regressor_pipeline(
    *,
    numeric_cols: list[str] = SELECTOR_NUM_COLS,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Preprocess workload/index features and train an RF regressor."""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["dataset", "index_type"],
            ),
        ],
    )
    reg = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("prep", pre), ("reg", reg)])


def train_selector_model(
    X_df: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Fit one selector pipeline; ``X_df`` must contain ``WORKLOAD_COLS`` columns."""
    pipe = make_selector_pipeline(seed=seed)
    pipe.fit(X_df[WORKLOAD_COLS], y)
    return pipe


def train_metric_regressor(
    X_df: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Fit one performance regressor over workload features plus index type."""
    pipe = make_performance_regressor_pipeline(seed=seed)
    pipe.fit(X_df[WORKLOAD_COLS + ["index_type"]], y.astype(float))
    return pipe


def train_metric_regressors(
    benchmark_df: pd.DataFrame,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Pipeline]:
    """Train latency, memory, and recall regressors from measured benchmark rows."""
    required = set(WORKLOAD_COLS + ["index_type", "mean_latency_ms", "index_size_mb", "recall_at_k"])
    missing = required - set(benchmark_df.columns)
    if missing:
        raise KeyError(f"benchmark_df missing columns: {sorted(missing)}")
    return {
        "latency_regressor_model": train_metric_regressor(
            benchmark_df,
            benchmark_df["mean_latency_ms"],
            seed=seed,
        ),
        "memory_regressor_model": train_metric_regressor(
            benchmark_df,
            benchmark_df["index_size_mb"],
            seed=seed,
        ),
        "recall_regressor_model": train_metric_regressor(
            benchmark_df,
            benchmark_df["recall_at_k"],
            seed=seed,
        ),
    }


def train_constrained_selector_model(
    X_df: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Deprecated compatibility wrapper; constrained selection now uses metric regressors."""
    numeric_cols = SELECTOR_NUM_COLS + CONSTRAINT_MODEL_COLS
    pipe = make_selector_pipeline(numeric_cols=numeric_cols, seed=seed)
    pipe.fit(X_df[WORKLOAD_COLS + CONSTRAINT_MODEL_COLS], y)
    return pipe


def train_composite_selector_model(
    X_df: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Deprecated compatibility wrapper for old composite selector tests/callers."""
    if set(CONSTRAINT_MODEL_COLS).issubset(X_df.columns):
        return train_constrained_selector_model(X_df, y, seed=seed)
    if set(WEIGHT_COLS).issubset(X_df.columns):
        converted = X_df.copy()
        recall_min, recall_max = min(RECALL_TARGETS), max(RECALL_TARGETS)
        memory_min, memory_max = min(MEMORY_BUDGET_RATIOS), max(MEMORY_BUDGET_RATIOS)
        converted["recall_target"] = recall_min + converted["w_recall"].astype(float) * (
            recall_max - recall_min
        )
        converted["memory_budget_ratio"] = memory_max - converted["w_memory"].astype(float) * (
            memory_max - memory_min
        )
        return train_constrained_selector_model(converted, y, seed=seed)
    return train_constrained_selector_model(X_df, y, seed=seed)


def save_selector_artifacts(models: dict[str, Pipeline], artifacts_dir: Path) -> None:
    """Persist selector pipelines under artifacts_dir."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for key in _SELECTOR_KEYS:
        if key not in models:
            raise KeyError(f"missing model key {key!r}")
        joblib.dump(models[key], artifacts_dir / _ARTIFACT_FILENAMES[key])


def _resolve_artifact_base_dir(artifacts_dir: Path) -> Path:
    """Resolve artifact directory, preferring latest timestamped run when present."""
    artifacts_dir = Path(artifacts_dir)
    required_files = [_ARTIFACT_FILENAMES[k] for k in _SELECTOR_KEYS]
    if all((artifacts_dir / name).exists() for name in required_files):
        return artifacts_dir

    runs_root = artifacts_dir / "runs"
    latest_file = artifacts_dir / "latest_run_id.txt"
    if latest_file.exists():
        run_id = latest_file.read_text(encoding="utf-8").strip()
        candidate = runs_root / run_id
        if run_id and all((candidate / name).exists() for name in required_files):
            return candidate

    if runs_root.exists():
        run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
        for candidate in reversed(run_dirs):
            if all((candidate / name).exists() for name in required_files):
                return candidate

    return artifacts_dir


def load_artifacts(artifacts_dir: Path) -> dict[str, Pipeline]:
    """Load selector classifiers and performance regressors from ``artifacts_dir``."""
    base = _resolve_artifact_base_dir(Path(artifacts_dir))
    return {k: joblib.load(base / _ARTIFACT_FILENAMES[k]) for k in _SELECTOR_KEYS}


def predict_index(
    models: dict[str, Pipeline],
    workload_row: pd.Series,
    objective: Objective,
) -> str:
    """Return predicted index type for one workload row and objective."""
    if objective in {"composite", "constrained"}:
        raise ValueError("parameterized predictions require a dedicated prediction helper")
    return str(
        models[_OBJECTIVE_TO_KEY[objective]].predict(
            pd.DataFrame([workload_row[WORKLOAD_COLS]])
        )[0]
    )


def predict_index_for_constraints(
    models: dict[str, Pipeline],
    workload_row: pd.Series,
    memory_budget_ratio: float,
    recall_target: float,
) -> str:
    """Return index type chosen by predicted metrics plus constrained objective."""
    row = workload_row.copy()
    row["memory_budget_ratio"] = float(memory_budget_ratio)
    row["recall_target"] = float(recall_target)
    if "memory_budget_mb" not in row.index or pd.isna(row["memory_budget_mb"]):
        row["memory_budget_mb"] = raw_vector_mb(float(row["N"]), float(row["d"])) * float(memory_budget_ratio)
    return predict_indices_for_constraints(models, pd.DataFrame([row]))[0]


def predict_indices_for_constraints(
    models: dict[str, Pipeline],
    configs: pd.DataFrame,
) -> list[str]:
    """Return constrained-policy choices for many workload/constraint rows."""
    required = set(WORKLOAD_COLS + CONSTRAINT_MODEL_COLS)
    missing = required - set(configs.columns)
    if missing:
        raise KeyError(f"configs missing columns: {sorted(missing)}")

    candidate_rows = []
    constraint_rows = []
    for row_id, (_, row) in enumerate(configs.iterrows()):
        budget_mb = (
            float(row["memory_budget_mb"])
            if "memory_budget_mb" in configs.columns and pd.notna(row.get("memory_budget_mb"))
            else raw_vector_mb(float(row["N"]), float(row["d"])) * float(row["memory_budget_ratio"])
        )
        constraint_rows.append({
            "row_id": row_id,
            "memory_budget_mb": budget_mb,
            "recall_target": float(row["recall_target"]),
        })
        base = row[WORKLOAD_COLS].to_dict()
        for index_type in INDEX_TYPES:
            candidate_rows.append({"row_id": row_id, **base, "index_type": index_type})

    candidates = pd.DataFrame(candidate_rows)
    predictions = pd.DataFrame({
        "row_id": candidates["row_id"],
        "index_type": candidates["index_type"],
        "predicted_latency_ms": models["latency_regressor_model"].predict(candidates),
        "predicted_memory_mb": models["memory_regressor_model"].predict(candidates),
        "predicted_recall": models["recall_regressor_model"].predict(candidates),
    })
    predictions["predicted_latency_ms"] = np.maximum(
        0.0,
        predictions["predicted_latency_ms"].astype(float),
    )
    predictions["predicted_memory_mb"] = np.maximum(
        0.0,
        predictions["predicted_memory_mb"].astype(float),
    )
    predictions["predicted_recall"] = np.clip(
        predictions["predicted_recall"].astype(float),
        0.0,
        1.0,
    )

    constraints = pd.DataFrame(constraint_rows).set_index("row_id")
    labels: list[str] = []
    for row_id, group in predictions.groupby("row_id", sort=False):
        constraint = constraints.loc[row_id]
        scored = score_predicted_constraints(
            group,
            float(constraint["memory_budget_mb"]),
            float(constraint["recall_target"]),
        )
        scored["_rk"] = scored["index_type"].astype(str).map(lambda t: INDEX_TYPES.index(t))
        scored = scored.sort_values(
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
        labels.append(str(scored.iloc[0]["index_type"]))
    return labels


def _candidate_rows(workload_row: pd.Series) -> pd.DataFrame:
    base = workload_row[WORKLOAD_COLS].to_dict()
    return pd.DataFrame([{**base, "index_type": index_type} for index_type in INDEX_TYPES])


def predict_performance_for_candidates(
    models: dict[str, Pipeline],
    workload_row: pd.Series,
) -> pd.DataFrame:
    """Predict latency, memory, and recall for every candidate index type."""
    candidates = _candidate_rows(workload_row)
    predictions = pd.DataFrame({
        "index_type": candidates["index_type"],
        "predicted_latency_ms": models["latency_regressor_model"].predict(candidates),
        "predicted_memory_mb": models["memory_regressor_model"].predict(candidates),
        "predicted_recall": models["recall_regressor_model"].predict(candidates),
    })
    predictions["predicted_latency_ms"] = np.maximum(
        0.0,
        predictions["predicted_latency_ms"].astype(float),
    )
    predictions["predicted_memory_mb"] = np.maximum(
        0.0,
        predictions["predicted_memory_mb"].astype(float),
    )
    predictions["predicted_recall"] = np.clip(
        predictions["predicted_recall"].astype(float),
        0.0,
        1.0,
    )
    return predictions


def predict_index_for_weights(
    models: dict[str, Pipeline],
    workload_row: pd.Series,
    weights: tuple[float, float, float],
) -> str:
    """Deprecated compatibility wrapper mapping composite weights to constraints."""
    w_recall, _, w_memory = weights
    recall_min, recall_max = min(RECALL_TARGETS), max(RECALL_TARGETS)
    memory_min, memory_max = min(MEMORY_BUDGET_RATIOS), max(MEMORY_BUDGET_RATIOS)
    recall_target = recall_min + float(w_recall) * (recall_max - recall_min)
    memory_budget_ratio = memory_max - float(w_memory) * (memory_max - memory_min)
    return predict_index_for_constraints(
        models,
        workload_row,
        memory_budget_ratio,
        recall_target,
    )


def objective_oracle_column(objective: Objective) -> str:
    """Column name in labeled data for the oracle labels."""
    return _ORACLE_COL[objective]
