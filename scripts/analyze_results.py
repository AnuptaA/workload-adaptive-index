"""Analyze oracle and prediction distributions for a benchmark run (per objective)."""

import argparse
from pathlib import Path

import pandas as pd
from src.config import ARTIFACTS_DIR, RESULTS_DIR
from src.labeling import (
    CONFIG_COLS,
    ORACLE_LATENCY_LABEL,
    ORACLE_MEMORY_LABEL,
    ORACLE_RECALL_LABEL,
    ORACLE_LABEL_COLS,
    Objective,
)
from src.models import load_artifacts, predict_index
from src.run_store import resolve_run_dir

_OBJECTIVES: list[tuple[Objective, str]] = [
    ("memory", ORACLE_MEMORY_LABEL),
    ("recall", ORACLE_RECALL_LABEL),
    ("latency", ORACLE_LATENCY_LABEL),
]


def _run_artifact_dir(artifacts_dir: Path, run_id: str) -> Path:
    return Path(artifacts_dir) / "runs" / run_id


def _has_run_artifacts(artifacts_dir: Path, run_id: str) -> bool:
    run_artifacts_dir = _run_artifact_dir(artifacts_dir, run_id)
    required = [
        "memory_selector_model.joblib",
        "recall_selector_model.joblib",
        "latency_selector_model.joblib",
        "latency_regressor_model.joblib",
        "memory_regressor_model.joblib",
        "recall_regressor_model.joblib",
    ]
    return all((run_artifacts_dir / name).exists() for name in required)


def main(results_dir: Path, artifacts_dir: Path, run_id: str = "") -> None:
    resolved_run_id, run_dir = resolve_run_dir(results_dir, run_id)
    labeled_path = run_dir / "labeled.csv"
    benchmarks_path = run_dir / "benchmarks.csv"

    if not labeled_path.exists():
        raise FileNotFoundError(f"{labeled_path} not found. Run label_data first.")
    if not benchmarks_path.exists():
        raise FileNotFoundError(f"{benchmarks_path} not found. Run benchmark first.")

    labeled = pd.read_csv(labeled_path)
    benchmarks = pd.read_csv(benchmarks_path)
    configs = labeled[CONFIG_COLS + ORACLE_LABEL_COLS].drop_duplicates(subset=CONFIG_COLS).copy()

    print(f"Run id: {resolved_run_id}")
    print(f"Unique configs: {len(configs)}")
    print(f"Total rows: {len(labeled)}")

    for obj, col in _OBJECTIVES:
        print(f"\nOracle label distribution ({obj} / {col}) overall:")
        vc = configs[col].value_counts().sort_index()
        for idx in vc.index:
            c = int(vc[idx])
            print(f"  {idx}: {c} ({100*c/len(configs):.2f}%)")

        print(f"\nOracle label distribution ({obj}) by dataset:")
        for ds in sorted(configs["dataset"].unique()):
            group = configs[configs["dataset"] == ds]
            print(f"  {ds} (n={len(group)}):")
            gvc = group[col].value_counts().sort_index()
            for idx in gvc.index:
                c = int(gvc[idx])
                print(f"    {idx}: {c} ({100*c/len(group):.2f}%)")

    has_artifacts = _has_run_artifacts(artifacts_dir, resolved_run_id)

    if has_artifacts:
        models = load_artifacts(_run_artifact_dir(artifacts_dir, resolved_run_id))

        def pred_cfg(row: pd.Series, objective: Objective) -> str:
            return predict_index(models, row, objective)

        for obj, oracle_col in _OBJECTIVES:
            col = f"model_pred_{obj}"
            configs[col] = configs.apply(lambda r, o=obj: pred_cfg(r, o), axis=1)
            configs[f"model_correct_{obj}"] = configs[col] == configs[oracle_col]

        print("\nModel prediction accuracy vs oracle (by dataset):")
        for obj, oracle_col in _OBJECTIVES:
            print(f"  objective={obj}:")
            for ds in sorted(configs["dataset"].unique()):
                group = configs[configs["dataset"] == ds]
                acc = 100 * float(group[f"model_correct_{obj}"].mean())
                print(f"    {ds}: {acc:.2f}%")
                vc = group[f"model_pred_{obj}"].value_counts().sort_index()
                for idx in vc.index:
                    c = int(vc[idx])
                    print(f"      pred {idx}: {c} ({100*c/len(group):.2f}%)")
        fm_pred = configs[configs["dataset"] == "fashion-mnist"]
        if not fm_pred.empty:
            for obj in ("memory", "recall", "latency"):
                always_hnsw = bool((fm_pred[f"model_pred_{obj}"] == "HNSW").all())
                print(f"\nFashion-mnist always predicted HNSW by {obj} model: {always_hnsw}")
    else:
        print(
            "\nPrediction analysis skipped: missing selector artifacts for "
            f"run {resolved_run_id}. Run `make train` first."
        )

    configs["always_hnsw_pred"] = "HNSW"

    print("\nAlways HNSW accuracy vs each oracle (overall):")
    for obj, oracle_col in _OBJECTIVES:
        acc = 100 * float((configs["always_hnsw_pred"] == configs[oracle_col]).mean())
        print(f"  {obj}: {acc:.2f}%")

    splits_dir = run_dir / "splits"
    for split_name in ("val", "test"):
        split_path = splits_dir / f"{split_name}.csv"
        if not split_path.exists():
            continue
        split_df = pd.read_csv(split_path)
        split_configs = split_df[CONFIG_COLS].drop_duplicates(subset=CONFIG_COLS)
        merged = split_configs.merge(configs, on=CONFIG_COLS, how="left")
        print(f"\nSplit {split_name}: {len(merged)} configs (merged with summary)")
        if has_artifacts:
            for obj in ("memory", "recall", "latency"):
                acc = merged[f"model_correct_{obj}"].mean()
                print(f"  model accuracy ({obj}): {100*float(acc):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(RESULTS_DIR))
    parser.add_argument("--artifacts-dir", type=Path, default=Path(ARTIFACTS_DIR))
    parser.add_argument(
        "--run-id",
        default="",
        help="Run id to analyze; defaults to latest run under results/runs/.",
    )
    args = parser.parse_args()
    main(args.results_dir, args.artifacts_dir, args.run_id)
