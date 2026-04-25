"""Label benchmark results and save labeled training pairs."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import MEMORY_VIOLATION_WEIGHT, RECALL_VIOLATION_WEIGHT, RESULTS_DIR
from src.labeling import check_class_distribution, label_benchmarks
from src.run_store import resolve_run_dir


def main(
    results_dir: Path,
    memory_weight: float = MEMORY_VIOLATION_WEIGHT,
    recall_weight: float = RECALL_VIOLATION_WEIGHT,
    run_id: str = "",
) -> None:
    results_dir = Path(results_dir)
    resolved_run_id, run_dir = resolve_run_dir(results_dir, run_id)
    benchmarks_path = run_dir / "benchmarks.csv"

    if not benchmarks_path.exists():
        raise FileNotFoundError(f"{benchmarks_path} not found. Run run_benchmark.py first.")

    df = pd.read_csv(benchmarks_path)
    labeled = label_benchmarks(df, memory_weight=memory_weight, recall_weight=recall_weight)

    dist = check_class_distribution(labeled)
    print(f"Class distribution (memory_weight={memory_weight}, recall_weight={recall_weight}):")
    for label, frac in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {label}: {frac:.3f}")

    out = run_dir / "labeled.csv"
    labeled.to_csv(out, index=False)
    meta = {
        "run_id": resolved_run_id,
        "memory_weight": float(memory_weight),
        "recall_weight": float(recall_weight),
        "cons_ratio": float(memory_weight) / float(recall_weight),
    }
    (run_dir / "labeling_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Run id: {resolved_run_id}")
    print(f"Saved {len(labeled)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(RESULTS_DIR))
    parser.add_argument("--memory-weight", type=float, default=MEMORY_VIOLATION_WEIGHT)
    parser.add_argument("--recall-weight", type=float, default=RECALL_VIOLATION_WEIGHT)
    parser.add_argument(
        "--run-id",
        default="",
        help="Run id to label; defaults to latest run under results/runs/.",
    )
    args = parser.parse_args()
    main(args.results_dir, args.memory_weight, args.recall_weight, args.run_id)
