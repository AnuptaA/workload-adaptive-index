from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """# TODO"""
    raise NotImplementedError

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """# TODO"""
    raise NotImplementedError

def evaluate_regressors(
    models: dict,
    scaler: StandardScaler,
    X: np.ndarray,
    y_latency: np.ndarray,
    y_memory: np.ndarray,
    y_recall: np.ndarray,
    split_name: str,
) -> dict[str, float]:
    """# TODO"""
    raise NotImplementedError

def evaluate_index_selection(
    predicted_labels: list[str],
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> dict[str, float]:
    """# TODO"""
    raise NotImplementedError

def constraint_violation_rate(
    predicted_labels: list[str],
    test_df: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> float:
    """# TODO"""
    raise NotImplementedError
