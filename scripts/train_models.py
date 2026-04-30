"""Train objective selectors plus performance regressors for constrained selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, RANDOM_SEED, RESULTS_DIR
from src.evaluate import (
    constrained_index_selection_comparison,
    evaluate_index_selection,
    index_selection_metric_comparison,
)
from src.labeling import (
    CONSTRAINED_ORACLE_LABEL,
    CONSTRAINT_COLS,
    CONFIG_COLS,
    ORACLE_LATENCY_LABEL,
    ORACLE_LABEL_COLS,
    ORACLE_MEMORY_LABEL,
    ORACLE_RECALL_LABEL,
    Objective,
    expand_constraint_grid,
    select_winner_for_constraints,
)
from src.models import (
    load_artifacts,
    predict_index,
    predict_indices_for_constraints,
    save_selector_artifacts,
    train_metric_regressors,
    train_selector_model,
)
from src.run_store import resolve_run_dir

_OBJECTIVE_SPECS: list[tuple[Objective, str]] = [
    ("memory", ORACLE_MEMORY_LABEL),
    ("recall", ORACLE_RECALL_LABEL),
    ("latency", ORACLE_LATENCY_LABEL),
]


def _stratum_key(row: pd.Series) -> str:
    return "|".join(str(row[c]) for c in ORACLE_LABEL_COLS)


def _config_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by (memory, recall, latency) oracle triplet.

    Keeps all index_type rows for each config together.
    """
    configs = df[CONFIG_COLS + ORACLE_LABEL_COLS].drop_duplicates(subset=CONFIG_COLS)
    configs = configs.copy()
    configs["_stratum"] = configs.apply(_stratum_key, axis=1)
    rng = np.random.default_rng(seed)

    train_parts, val_parts, test_parts = [], [], []

    for strat, group in configs.groupby("_stratum", sort=False):
        n = len(group)
        perm = rng.permutation(n)
        if n < 3:
            print(f"Warning: stratum '{strat}' has only {n} config(s); assigning all to train.")
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


def _predict_for_objective(
    test_configs: pd.DataFrame,
    models: dict,
    objective: Objective,
) -> list[str]:
    out: list[str] = []
    for _, row in test_configs.iterrows():
        out.append(predict_index(models, row, objective))
    return out


def _predict_for_constraints(
    test_configs: pd.DataFrame,
    models: dict,
) -> list[str]:
    return predict_indices_for_constraints(models, test_configs)


def _constrained_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cross configs with deployment constraints and compute constrained oracle labels."""
    constraint_grid = expand_constraint_grid(df)
    rows: list[dict] = []
    for _, group in df.groupby(CONFIG_COLS, sort=False):
        config = group.iloc[0][CONFIG_COLS].to_dict()
        mask = pd.Series(True, index=constraint_grid.index)
        for col, value in config.items():
            mask &= constraint_grid[col] == value
        matching_constraints = constraint_grid[mask]
        for _, constraint_row in matching_constraints.iterrows():
            rows.append({
                **config,
                **{col: float(constraint_row[col]) for col in CONSTRAINT_COLS},
                CONSTRAINED_ORACLE_LABEL: select_winner_for_constraints(
                    group,
                    float(constraint_row["memory_budget_mb"]),
                    float(constraint_row["recall_target"]),
                ),
            })
    return pd.DataFrame(rows, columns=CONFIG_COLS + CONSTRAINT_COLS + [CONSTRAINED_ORACLE_LABEL])


def _config_label_counts(df: pd.DataFrame, label_col: str) -> dict[str, int]:
    """Return label counts at config granularity for a split."""
    config_labels = df[CONFIG_COLS + [label_col]].drop_duplicates(subset=CONFIG_COLS)
    counts = config_labels[label_col].value_counts()
    return {str(label): int(counts[label]) for label in counts.index}


def _save_split_artifacts(
    run_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> None:
    """Persist split tables and balancing metadata for reproducibility."""
    splits_dir = Path(run_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    meta_counts = {}
    for obj_name, col in _OBJECTIVE_SPECS:
        meta_counts[f"train_{obj_name}_oracle_label_counts"] = _config_label_counts(train_df, col)
        meta_counts[f"val_{obj_name}_oracle_label_counts"] = _config_label_counts(val_df, col)
        meta_counts[f"test_{obj_name}_oracle_label_counts"] = _config_label_counts(test_df, col)

    metadata = {
        "seed": int(seed),
        **meta_counts,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }
    (splits_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _print_metric_comparison(
    title: str,
    objective: Objective,
    report: dict[str, float | str],
    random_mc_trials: int,
) -> None:
    metric = str(report["metric"])
    label = {"memory": "index size (MB)", "recall": "recall@k", "latency": "latency (ms)"}[objective]
    print(f"\n{title} — objective={objective} ({label}, benchmark lookup)")
    print(f"  Oracle (tabular):                  {report['oracle_mean']:.6f}")
    print(f"  Trained selector:                  {report['model_mean']:.6f}")
    print(f"  Always HNSW:                      {report['always_hnsw_mean']:.6f}")
    print(f"  Uniform random (exact E[.]):      {report['uniform_random_expected_mean']:.6f}")
    print(
        f"  Uniform random (MC mean +/- SE): {report['random_policy_mc_mean']:.6f} "
        f"+/- {report['random_policy_mc_se']:.6f} ({random_mc_trials} trials)"
    )


def _print_constrained_summary(
    title: str,
    report: dict[str, float | str],
    accuracy: float,
) -> None:
    print(f"\n{title} — objective=constrained (penalty score over constraint grid)")
    print(f"  Accuracy vs oracle:                       {accuracy:.4f}")
    print(f"  Oracle objective score:                   {report['oracle_mean_objective_score']:.6f}")
    print(f"  Regressor policy objective score:        {report['model_mean_objective_score']:.6f}")
    print(f"  Rule-based objective score:               {report['rule_based_mean_objective_score']:.6f}")
    print(f"  Always HNSW objective score:              {report['always_hnsw_mean_objective_score']:.6f}")
    print(f"  Regressor policy constraint satisfaction: {report['model_constraint_satisfaction_rate']:.4f}")
    print(f"  Regressor policy mean latency (ms):       {report['model_mean_latency_ms']:.6f}")
    print(f"  Regressor policy mean memory overrun:     {report['model_mean_memory_overrun']:.6f}")
    print(f"  Regressor policy mean recall shortfall:   {report['model_mean_recall_shortfall']:.6f}")


def main(
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str = "",
    seed: int = RANDOM_SEED,
    random_mc_trials: int = 400,
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

    _save_split_artifacts(
        run_dir,
        train_df,
        val_df,
        test_df,
        seed,
    )

    winners_train = train_df[CONFIG_COLS + ORACLE_LABEL_COLS].drop_duplicates(subset=CONFIG_COLS)
    constrained_train = _constrained_training_table(train_df)

    models_dict = {}
    for objective, oracle_col in _OBJECTIVE_SPECS:
        y = winners_train[oracle_col].astype(str)
        pipe = train_selector_model(winners_train, y, seed=seed)
        models_dict[f"{objective}_selector_model"] = pipe

    models_dict.update(train_metric_regressors(train_df, seed=seed))

    save_selector_artifacts(models_dict, run_artifacts_dir)
    (artifacts_dir / "latest_run_id.txt").write_text(f"{resolved_run_id}\n", encoding="utf-8")
    print(f"Run id: {resolved_run_id}")
    print(f"Saved selector and performance models to {run_artifacts_dir}")

    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        n_configs = split_df[CONFIG_COLS].drop_duplicates().shape[0]
        parts = []
        for objective, col in _OBJECTIVE_SPECS:
            dist = (
                split_df[CONFIG_COLS + [col]].drop_duplicates(subset=CONFIG_COLS)[col]
                .value_counts().to_dict()
            )
            parts.append(f"{objective}={dist}")
        print(f"  {split_name}: {n_configs} configs | " + " | ".join(parts))
    print(
        "  constrained grid: "
        f"{len(expand_constraint_grid(train_df)) // max(1, train_df[CONFIG_COLS].drop_duplicates().shape[0])} "
        "constraints per config"
    )
    print("  constrained policy: metric regressors -> predicted metrics -> objective score")

    models_loaded = load_artifacts(run_artifacts_dir)

    for split_title, split_df in (("Validation", val_df), ("Test", test_df)):
        configs = split_df[CONFIG_COLS + ORACLE_LABEL_COLS].drop_duplicates(subset=CONFIG_COLS)

        for objective, oracle_col in _OBJECTIVE_SPECS:
            predicted = _predict_for_objective(configs, models_loaded, objective)

            report = index_selection_metric_comparison(
                objective,
                configs,
                benchmarks,
                predicted,
                oracle_col=oracle_col,
                random_mc_trials=random_mc_trials,
                random_mc_seed=seed + (1 if split_title == "Validation" else 2),
            )
            sel = evaluate_index_selection(predicted, configs, oracle_col)
            print(
                f"\n{split_title} — objective={objective}: "
                f"accuracy vs oracle={sel['accuracy']:.4f}"
            )
            _print_metric_comparison(
                f"{split_title} split",
                objective,
                report,
                random_mc_trials,
            )

        constrained_configs = _constrained_training_table(split_df)
        predicted = _predict_for_constraints(constrained_configs, models_loaded)
        report = constrained_index_selection_comparison(
            constrained_configs,
            benchmarks,
            predicted,
        )
        sel = evaluate_index_selection(
            predicted,
            constrained_configs,
            CONSTRAINED_ORACLE_LABEL,
        )
        _print_constrained_summary(f"{split_title} split", report, float(sel["accuracy"]))

    models_check = load_artifacts(run_artifacts_dir)
    test_configs = test_df[CONFIG_COLS + ORACLE_LABEL_COLS].drop_duplicates(subset=CONFIG_COLS)
    assert len(_predict_for_objective(test_configs.head(3), models_check, "latency")) == 3
    assert len(_predict_for_constraints(_constrained_training_table(test_df).head(3), models_check)) == 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(RESULTS_DIR))
    parser.add_argument("--artifacts-dir", type=Path, default=Path(ARTIFACTS_DIR))
    parser.add_argument("--run-id", default="", help="subdirectory for per-run outputs")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--random-mc-trials", type=int, default=400)
    args = parser.parse_args()
    main(
        args.results_dir,
        args.artifacts_dir,
        args.run_id,
        args.seed,
        args.random_mc_trials,
    )
