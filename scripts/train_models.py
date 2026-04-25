"""Train latency, memory, and recall regressors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import faiss_rule_based, mean_latency_for_labels
from src.config import ARTIFACTS_DIR, MEMORY_VIOLATION_WEIGHT, RANDOM_SEED, RECALL_VIOLATION_WEIGHT, RESULTS_DIR
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
from src.run_store import resolve_run_dir


def _config_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by label: sample train_frac/val_frac within each label group.

    Keeps all index_type rows for each config together. If a label group has
    fewer than 3 configs, all its configs go to train with a warning.
    """
    configs = df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)
    rng = np.random.default_rng(seed)

    train_parts, val_parts, test_parts = [], [], []

    for label, group in configs.groupby("label"):
        n = len(group)
        perm = rng.permutation(n)
        if n < 3:
            print(f"Warning: label '{label}' has only {n} config(s); assigning all to train.")
            train_parts.append(group.iloc[perm][CONFIG_COLS])
            continue
        n_train = max(1, round(train_frac * n))
        n_val = max(1, round(val_frac * n))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        train_parts.append(group.iloc[perm[:n_train]][CONFIG_COLS])
        val_parts.append(group.iloc[perm[n_train:n_train + n_val]][CONFIG_COLS])
        test_parts.append(group.iloc[perm[n_train + n_val:]][CONFIG_COLS])

    def _merge(parts: list) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=CONFIG_COLS)
        return df.merge(pd.concat(parts), on=CONFIG_COLS, how="inner")

    return _merge(train_parts), _merge(val_parts), _merge(test_parts)


def _predict_indices_for_configs(
    test_configs: pd.DataFrame,
    models: dict,
    scaler,
    memory_weight: float,
    recall_weight: float,
) -> list[str]:
    """One ``select_index`` per unique configuration row."""
    out: list[str] = []
    for _, row in test_configs.iterrows():
        workload = {k: float(row[k]) for k in FEATURE_COLS + ["memory_budget_mb", "recall_target"]}
        out.append(select_index(workload, models, scaler, memory_weight, recall_weight))
    return out


def _balance_train_configs_equal_labels(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample each label to the minority label count at config granularity.

    This keeps all index-type rows for a selected config and avoids splitting
    per-config groups across labels.
    """
    # Map each unique CONFIG_COLS tuple to an integer to avoid float-column merge instability.
    config_tuples = [tuple(row) for row in train_df[CONFIG_COLS].to_numpy()]
    unique_tuples = list(dict.fromkeys(config_tuples))
    tuple_to_id = {t: i for i, t in enumerate(unique_tuples)}

    work = train_df.copy()
    work["_cfg_id"] = [tuple_to_id[t] for t in config_tuples]

    id_label = work[["_cfg_id", "label"]].drop_duplicates(subset=["_cfg_id"])
    counts = id_label["label"].value_counts()
    if counts.empty:
        return train_df

    min_count = int(counts.min())
    rng = np.random.default_rng(seed)

    chosen_ids: list[int] = []
    for label, group in id_label.groupby("label"):
        if len(group) <= min_count:
            chosen_ids.extend(group["_cfg_id"].tolist())
        else:
            random_state = int(rng.integers(0, 2**31))
            chosen_ids.extend(
                group.sample(n=min_count, random_state=random_state)["_cfg_id"].tolist()
            )

    balanced = work[work["_cfg_id"].isin(chosen_ids)].drop(columns=["_cfg_id"])

    # Verify using the same ID mapping (no float-column merge).
    post_tuples = [tuple(row) for row in balanced[CONFIG_COLS].to_numpy()]
    post_id_label = (
        balanced.assign(_cfg_id=[tuple_to_id[t] for t in post_tuples])
        [["_cfg_id", "label"]].drop_duplicates(subset=["_cfg_id"])
    )
    post_counts = post_id_label["label"].value_counts()
    if post_counts.empty:
        raise ValueError("Balanced train split is empty after downsampling.")
    if int(post_counts.min()) != int(post_counts.max()):
        raise ValueError(
            "Train downsampling failed to equalize labels. "
            f"Counts={post_counts.to_dict()}"
        )

    return balanced


def _config_label_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return label counts at config granularity for a split."""
    config_labels = df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)
    counts = config_labels["label"].value_counts()
    return {str(label): int(counts[label]) for label in counts.index}


def _save_split_artifacts(
    run_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    balance_train_labels: bool,
    seed: int,
) -> None:
    """Persist split tables and balancing metadata for reproducibility."""
    splits_dir = Path(run_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    metadata = {
        "seed": int(seed),
        "balance_train_labels": bool(balance_train_labels),
        "train_config_label_counts": _config_label_counts(train_df),
        "val_config_label_counts": _config_label_counts(val_df),
        "test_config_label_counts": _config_label_counts(test_df),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }
    (splits_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _print_coefficients(models: dict, feature_names: list[str]) -> None:
    """Print LinearRegression coefficients for each regressor, sorted by |coef|."""
    targets = {
        "latency_model": "mean_latency_ms",
        "memory_model": "index_size_mb",
        "recall_model": "recall_at_k",
    }
    print("\nRegressor coefficients (sorted by |coef|):")
    for model_key, target_name in targets.items():
        model = models[model_key]
        pairs = sorted(
            zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True
        )
        print(f"  {target_name}:")
        for feat, coef in pairs:
            print(f"    {feat:30s} {coef:+.6f}")


def _print_latency_comparison(
    title: str, report: dict[str, float], random_mc_trials: int, rule_based_ms: float
) -> None:
    print(f"\n{title} (mean measured query latency ms, benchmark lookup)")
    print(f"  Oracle winner (tabular label):     {report['oracle_mean_latency_ms']:.6f}")
    print(f"  Trained selector (predictions):   {report['model_mean_latency_ms']:.6f}")
    print(f"  Rule-based (FAISS heuristic):     {rule_based_ms:.6f}")
    print(f"  Always HNSW:                      {report['always_hnsw_mean_latency_ms']:.6f}")
    print(
        f"  Uniform random (exact E[.]):      {report['uniform_random_expected_mean_latency_ms']:.6f}"
    )
    print(
        f"  Uniform random (MC mean +/- SE):  {report['random_policy_mc_mean_latency_ms']:.6f} "
        f"+/- {report['random_policy_mc_se_latency_ms']:.6f} ({random_mc_trials} trials)"
    )


def main(
    results_dir: Path,
    artifacts_dir: Path,
    memory_weight: float = MEMORY_VIOLATION_WEIGHT,
    recall_weight: float = RECALL_VIOLATION_WEIGHT,
    run_id: str = "",
    seed: int = RANDOM_SEED,
    random_mc_trials: int = 400,
    balance_train_labels: bool = True,
) -> None:
    results_dir = Path(results_dir)
    artifacts_dir = Path(artifacts_dir)

    resolved_run_id, run_dir = resolve_run_dir(results_dir, run_id)
    labeled_path = run_dir / "labeled.csv"
    benchmarks_path = run_dir / "benchmarks.csv"
    run_artifacts_dir = artifacts_dir / "runs" / resolved_run_id

    if not labeled_path.exists():
        raise FileNotFoundError(
            f"{labeled_path} not found. Run scripts/label_data.py first."
        )
    if not benchmarks_path.exists():
        raise FileNotFoundError(f"{benchmarks_path} not found.")

    df = pd.read_csv(labeled_path)
    benchmarks = pd.read_csv(benchmarks_path)

    train_df, val_df, test_df = _config_split(df, 0.70, 0.15, seed)

    if balance_train_labels:
        pre = train_df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)
        pre_dist = pre["label"].value_counts().to_dict()
        train_df = _balance_train_configs_equal_labels(train_df, seed)
        post = train_df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)
        post_dist = post["label"].value_counts().to_dict()
        print("Balanced train configs to minority label count:")
        print(f"  before: {pre_dist}")
        print(f"  after:  {post_dist}")

        # Additional explicit safety check.
        if post_dist and (min(post_dist.values()) != max(post_dist.values())):
            raise ValueError(f"Train labels not strictly balanced after downsampling: {post_dist}")

    _save_split_artifacts(
        run_dir,
        train_df,
        val_df,
        test_df,
        balance_train_labels,
        seed,
    )

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        n_configs = split_df[CONFIG_COLS].drop_duplicates().shape[0]
        dist = split_df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)["label"].value_counts().to_dict()
        print(f"  {split_name}: {n_configs} configs | {dist}")

    X_train, feature_names = build_feature_matrix(train_df)
    X_val, _ = build_feature_matrix(val_df)
    X_test, _ = build_feature_matrix(test_df)

    scaler = make_scaler(X_train)
    X_train_s = apply_scaler(X_train, scaler)

    y_lat_train = train_df["mean_latency_ms"].to_numpy()
    y_mem_train = train_df["index_size_mb"].to_numpy()
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
    save_artifacts(models, scaler, run_artifacts_dir)
    (artifacts_dir / "latest_run_id.txt").write_text(f"{resolved_run_id}\n", encoding="utf-8")
    print(f"Run id: {resolved_run_id}")
    print(f"Saved models and scaler to {run_artifacts_dir}")

    _print_coefficients(models, feature_names)

    train_metrics = evaluate_regressors(
        models, scaler, X_train,
        train_df["mean_latency_ms"].to_numpy(),
        train_df["index_size_mb"].to_numpy(),
        train_df["recall_at_k"].to_numpy(),
        "train",
    )
    print("train RMSE (in-sample):", train_metrics)

    for split_name, X_split, part in (
        ("val", X_val, val_df),
        ("test", X_test, test_df),
    ):
        metrics = evaluate_regressors(
            models, scaler, X_split,
            part["mean_latency_ms"].to_numpy(),
            part["index_size_mb"].to_numpy(),
            part["recall_at_k"].to_numpy(),
            split_name,
        )
        print(f"{split_name} RMSE:", metrics)

    for split_name, split_df in (("Validation", val_df), ("Test", test_df)):
        configs = split_df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)
        predicted = _predict_indices_for_configs(configs, models, scaler, memory_weight, recall_weight)
        report = index_selection_latency_comparison(
            configs, benchmarks, predicted,
            random_mc_trials=random_mc_trials,
            random_mc_seed=seed + (1 if split_name == "Validation" else 2),
        )
        rule_perf = faiss_rule_based(configs, benchmarks)
        rule_ms = float(rule_perf["mean_latency_ms"].mean())
        sel = evaluate_index_selection(predicted, configs, benchmarks)
        viol = constraint_violation_rate(predicted, configs, benchmarks, memory_weight, recall_weight)
        print(
            f"\n{split_name} selection accuracy: {sel['accuracy']:.4f}, "
            f"constraint_violation_rate={viol:.4f}"
        )
        _print_latency_comparison(f"{split_name} split", report, random_mc_trials, rule_ms)

    # Sanity check: reload artifacts and confirm inference works.
    models2, scaler2 = load_artifacts(run_artifacts_dir)
    test_configs = test_df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)
    assert len(_predict_indices_for_configs(test_configs.head(3), models2, scaler2, memory_weight, recall_weight)) == 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(RESULTS_DIR))
    parser.add_argument("--artifacts-dir", type=Path, default=Path(ARTIFACTS_DIR))
    parser.add_argument("--memory-weight", type=float, default=MEMORY_VIOLATION_WEIGHT)
    parser.add_argument("--recall-weight", type=float, default=RECALL_VIOLATION_WEIGHT)
    parser.add_argument("--run-id", default="", help="subdirectory for per-run outputs")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--random-mc-trials", type=int, default=400)
    parser.add_argument(
        "--no-balance-train-labels",
        action="store_true",
        help="Disable equal-frequency downsampling of train configs by label.",
    )
    args = parser.parse_args()
    main(
        args.results_dir,
        args.artifacts_dir,
        args.memory_weight,
        args.recall_weight,
        args.run_id,
        args.seed,
        args.random_mc_trials,
        balance_train_labels=not args.no_balance_train_labels,
    )
