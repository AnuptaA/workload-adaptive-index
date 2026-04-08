import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import INDEX_TYPES

FEATURE_COLS = ["N", "d", "k", "memory_budget_mb", "recall_target"]
# latency is the quantity minimized by the objective, not a feature

_INDEX_ONE_HOT_COLS = [f"index_{t}" for t in INDEX_TYPES]

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract FEATURE_COLS and one-hot encode index_type.

    Added columns: index_IVF_FLAT, index_IVF_PQ, index_HNSW.
    """
    out = df[FEATURE_COLS].copy()
    for t in INDEX_TYPES:
        out[f"index_{t}"] = (df["index_type"] == t).astype(np.float32)
    return out

def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Returns (X, feature_names). Does not scale — caller handles scaling."""
    feature_df = extract_features(df)
    feature_names = FEATURE_COLS + _INDEX_ONE_HOT_COLS
    X = feature_df[feature_names].to_numpy(dtype=np.float32)
    return X, feature_names

def make_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit and return a StandardScaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Transform X using a pre-fit scaler."""
    return scaler.transform(X)
