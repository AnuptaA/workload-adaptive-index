"""Run label + train for each constraint-ratio weight pair."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import ARTIFACTS_DIR, RESULTS_DIR

WEIGHT_PAIRS = [
    ("A", 1.0, 4.0),  # cons_ratio = 0.25 (recall penalized more)
    ("B", 1.0, 2.0),  # cons_ratio = 0.50
    ("C", 1.0, 1.0),  # cons_ratio = 1.00
    ("D", 2.0, 1.0),  # cons_ratio = 2.00
    ("E", 4.0, 1.0),  # cons_ratio = 4.00
    ("F", 8.0, 1.0),  # cons_ratio = 8.00 (memory penalized more)
]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main(results_dir: Path, artifacts_dir: Path) -> None:
    python = sys.executable
    for run_id, mem_w, rec_w in WEIGHT_PAIRS:
        ratio = mem_w / rec_w
        print(f"\n{'=' * 60}")
        print(f"Run {run_id}: memory_weight={mem_w}, recall_weight={rec_w}, cons_ratio={ratio:.2f}")
        print(f"{'=' * 60}")

        _run([
            python, "scripts/label_data.py",
            "--results-dir", str(results_dir),
            "--memory-weight", str(mem_w),
            "--recall-weight", str(rec_w),
            "--run-id", run_id,
        ])
        _run([
            python, "scripts/train_models.py",
            "--results-dir", str(results_dir),
            "--artifacts-dir", str(artifacts_dir),
            "--memory-weight", str(mem_w),
            "--recall-weight", str(rec_w),
            "--run-id", run_id,
        ])

    print(f"\nSweep complete. Results in {results_dir}/{{A-F}}/, artifacts in {artifacts_dir}/{{A-F}}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(RESULTS_DIR))
    parser.add_argument("--artifacts-dir", type=Path, default=Path(ARTIFACTS_DIR))
    args = parser.parse_args()
    main(args.results_dir, args.artifacts_dir)
