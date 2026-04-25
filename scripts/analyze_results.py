"""Analyze winner/prediction distributions for a benchmark run."""

import argparse
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, RESULTS_DIR
from src.labeling import CONFIG_COLS
from src.models import load_artifacts, select_index
from src.run_store import resolve_run_dir


def main(results_dir: Path, artifacts_dir: Path, run_id: str = "") -> None:
    resolved_run_id, run_dir = resolve_run_dir(results_dir, run_id)
    labeled_path = run_dir / "labeled.csv"

    if not labeled_path.exists():
        raise FileNotFoundError(f"{labeled_path} not found. Run label_data first.")

    labeled = pd.read_csv(labeled_path)
    configs = labeled[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS).copy()

    print(f"Run id: {resolved_run_id}")
    print(f"Unique configs: {len(configs)}")
    print(f"Total rows: {len(labeled)}")

    print("\nOracle label distribution (overall):")
    vc = configs["label"].value_counts().sort_index()
    for idx in vc.index:
        c = int(vc[idx])
        print(f"  {idx}: {c} ({100*c/len(configs):.2f}%)")

    print("\nOracle label distribution by dataset:")
    for ds in sorted(configs["dataset"].unique()):
        group = configs[configs["dataset"] == ds]
        print(f"  {ds} (n={len(group)}):")
        vc = group["label"].value_counts().sort_index()
        for idx in vc.index:
            c = int(vc[idx])
            print(f"    {idx}: {c} ({100*c/len(group):.2f}%)")

    fm = configs[configs["dataset"] == "fashion-mnist"]
    if not fm.empty:
        print("\nFashion-mnist winner table by budget/recall:")
        print(pd.crosstab([fm["memory_budget_mb"], fm["recall_target"]], fm["label"]).to_string())

    models, scaler = load_artifacts(Path(artifacts_dir) / "runs" / resolved_run_id)

    def pred_cfg(row: pd.Series) -> str:
        workload = {
            "N": float(row["N"]),
            "d": float(row["d"]),
            "k": float(row["k"]),
            "memory_budget_mb": float(row["memory_budget_mb"]),
            "recall_target": float(row["recall_target"]),
        }
        return select_index(workload, models, scaler)

    configs["pred"] = configs.apply(pred_cfg, axis=1)
    configs["correct"] = configs["pred"] == configs["label"]

    print("\nPredicted label distribution by dataset:")
    for ds in sorted(configs["dataset"].unique()):
        group = configs[configs["dataset"] == ds]
        print(f"  {ds}: accuracy={100*group['correct'].mean():.2f}%")
        vc = group["pred"].value_counts().sort_index()
        for idx in vc.index:
            c = int(vc[idx])
            print(f"    {idx}: {c} ({100*c/len(group):.2f}%)")

    if not fm.empty:
        fm_pred = configs[configs["dataset"] == "fashion-mnist"]
        always_hnsw = bool((fm_pred["pred"] == "HNSW").all())
        print(f"\nFashion-mnist always predicted HNSW: {always_hnsw}")


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
