"""Train latency, memory, and recall regressors."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, RANDOM_SEED, RESULTS_DIR
from src.evaluate import (
    constraint_violation_rate,
    evaluate_index_selection,
    evaluate_regressors,
    index_selection_latency_comparison,
)
from src.features import FEATURE_COLS, apply_scaler, build_feature_matrix, make_scaler
from src.labeling import CONFIG_COLS
from src.models import (
    load_artifacts,
    save_artifacts,
    select_index,
    train_latency_model,
    train_memory_model,
    train_recall_model,
)
def _config_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test by unique ``CONFIG_COLS`` (keeps all index_type rows)."""
    configs = df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)
    n = len(configs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(train_frac * n))
    n_val = max(1, int(val_frac * n))
    if n_train + n_val >= n:
        n_val = max(1, (n - n_train) // 2)
    train_keys = configs.iloc[perm[:n_train]][CONFIG_COLS]
    val_keys = configs.iloc[perm[n_train : n_train + n_val]][CONFIG_COLS]
    test_keys = configs.iloc[perm[n_train + n_val :]][CONFIG_COLS]

    train_df = df.merge(train_keys, on=CONFIG_COLS, how="inner")
    val_df = df.merge(val_keys, on=CONFIG_COLS, how="inner")
    test_df = df.merge(test_keys, on=CONFIG_COLS, how="inner")
    return train_df, val_df, test_df


def _predict_indices_for_configs(
    test_configs: pd.DataFrame,
    models: dict,
    scaler,
) -> list[str]:
    """One ``select_index`` per unique configuration row."""
    out: list[str] = []
    for _, row in test_configs.iterrows():
        workload = {k: float(row[k]) for k in FEATURE_COLS}
        out.append(select_index(workload, models, scaler))
    return out


def _print_latency_comparison(
    title: str, report: dict[str, float], random_mc_trials: int
) -> None:
    print(f"\n{title} (mean measured query latency ms, benchmark lookup)")
    print(f"  Oracle winner (tabular label):     {report['oracle_mean_latency_ms']:.6f}")
    print(f"  Trained selector (predictions):   {report['model_mean_latency_ms']:.6f}")
    print(f"  Always HNSW:                      {report['always_hnsw_mean_latency_ms']:.6f}")
    print(
        f"  Uniform random (exact E[.]):      {report['uniform_random_expected_mean_latency_ms']:.6f}"
    )
    print(
        f"  Uniform random (MC mean ± SE):    {report['random_policy_mc_mean_latency_ms']:.6f} "
        f"± {report['random_policy_mc_se_latency_ms']:.6f} ({random_mc_trials} trials)"
    )


def main(
    results_dir: Path,
    artifacts_dir: Path,
    seed: int = RANDOM_SEED,
    random_mc_trials: int = 400,
) -> None:
    results_dir = Path(results_dir)
    artifacts_dir = Path(artifacts_dir)
    labeled_path = results_dir / "labeled.csv"
    benchmarks_path = results_dir / "benchmarks.csv"

    if not labeled_path.exists():
        raise FileNotFoundError(
            f"{labeled_path} not found. Run scripts/label_data.py after benchmarking."
        )
    if not benchmarks_path.exists():
        raise FileNotFoundError(f"{benchmarks_path} not found.")

    df = pd.read_csv(labeled_path)
    benchmarks = pd.read_csv(benchmarks_path)

    train_df, val_df, test_df = _config_split(df, 0.70, 0.15, seed)
    print(
        f"Split unique configs: train={train_df[CONFIG_COLS].drop_duplicates().shape[0]}, "
        f"val={val_df[CONFIG_COLS].drop_duplicates().shape[0]}, "
        f"test={test_df[CONFIG_COLS].drop_duplicates().shape[0]}",
    )

    X_train, _ = build_feature_matrix(train_df)
    X_val, _ = build_feature_matrix(val_df)
    X_test, _ = build_feature_matrix(test_df)

    scaler = make_scaler(X_train)
    X_train_s = apply_scaler(X_train, scaler)

    y_lat_train = train_df["mean_latency_ms"].to_numpy()
    y_mem_train = train_df["peak_memory_mb"].to_numpy()
    y_rec_train = train_df["recall_at_k"].to_numpy()

    print("Training regressors (CV on train rows):")
    latency_model = train_latency_model(X_train_s, y_lat_train)
    memory_model = train_memory_model(X_train_s, y_mem_train)
    recall_model = train_recall_model(X_train_s, y_rec_train)

    models = {
        "latency_model": latency_model,
        "memory_model": memory_model,
        "recall_model": recall_model,
    }
    save_artifacts(models, scaler, artifacts_dir)
    print(f"Saved models and scaler to {artifacts_dir}")

    train_metrics = evaluate_regressors(
        models,
        scaler,
        X_train,
        train_df["mean_latency_ms"].to_numpy(),
        train_df["peak_memory_mb"].to_numpy(),
        train_df["recall_at_k"].to_numpy(),
        "train",
    )
    print("train RMSE (in-sample):", train_metrics)

    for split_name, X_split, part in (
        ("val", X_val, val_df),
        ("test", X_test, test_df),
    ):
        metrics = evaluate_regressors(
            models,
            scaler,
            X_split,
            part["mean_latency_ms"].to_numpy(),
            part["peak_memory_mb"].to_numpy(),
            part["recall_at_k"].to_numpy(),
            split_name,
        )
        print(f"{split_name} RMSE:", metrics)

    val_configs = val_df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)
    predicted_val = _predict_indices_for_configs(val_configs, models, scaler)
    val_report = index_selection_latency_comparison(
        val_configs,
        benchmarks,
        predicted_val,
        random_mc_trials=random_mc_trials,
        random_mc_seed=seed + 1,
    )
    val_sel = evaluate_index_selection(predicted_val, val_configs, benchmarks)
    print(
        f"\nValidation selection accuracy: {val_sel['accuracy']:.4f}, "
        f"constraint_violation_rate={constraint_violation_rate(predicted_val, val_configs, benchmarks):.4f}",
    )
    _print_latency_comparison("Validation split", val_report, random_mc_trials)

    test_configs = test_df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)
    predicted = _predict_indices_for_configs(test_configs, models, scaler)
    test_report = index_selection_latency_comparison(
        test_configs,
        benchmarks,
        predicted,
        random_mc_trials=random_mc_trials,
        random_mc_seed=seed + 2,
    )
    sel_metrics = evaluate_index_selection(predicted, test_configs, benchmarks)
    viol = constraint_violation_rate(predicted, test_configs, benchmarks)
    print(
        f"\nTest selection accuracy: {sel_metrics['accuracy']:.4f}, "
        f"constraint_violation_rate={viol:.4f}",
    )
    _print_latency_comparison("Test split", test_report, random_mc_trials)

    # Sanity check: reload artifacts
    models2, scaler2 = load_artifacts(artifacts_dir)
    assert len(_predict_indices_for_configs(test_configs.head(3), models2, scaler2)) == 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(RESULTS_DIR))
    parser.add_argument("--artifacts-dir", type=Path, default=Path(ARTIFACTS_DIR))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--random-mc-trials",
        type=int,
        default=400,
        help="Monte Carlo trials for uniform-random policy mean latency stderr",
    )
    args = parser.parse_args()
    main(
        args.results_dir,
        args.artifacts_dir,
        args.seed,
        random_mc_trials=args.random_mc_trials,
    )
