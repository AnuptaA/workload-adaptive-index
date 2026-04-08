"""Train latency, memory, and recall regressors.

TODO: implement this script after labeling is complete.
Intended call sequence:
  1. Load labeled.csv
  2. 70/15/15 train/val/test split (stratified by label)
  3. build_feature_matrix -> X, feature_names
  4. make_scaler(X_train) -> fit on train only
  5. apply_scaler on train/val/test
  6. train_latency_model, train_memory_model, train_recall_model
  7. save_artifacts(models, scaler, artifacts_dir)
  8. evaluate_regressors on val and test
  9. evaluate_index_selection on test
  10. report baselines
"""

from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, RESULTS_DIR
from src.evaluate import evaluate_index_selection, evaluate_regressors
from src.features import build_feature_matrix, make_scaler, apply_scaler
from src.models import (
    load_artifacts,
    save_artifacts,
    select_index,
    train_latency_model,
    train_memory_model,
    train_recall_model,
)
from src.baselines import always_hnsw, faiss_rule_based, random_baseline

def main() -> None:
    # TODO: implement
    raise NotImplementedError

if __name__ == "__main__":
    main()
