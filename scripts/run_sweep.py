"""Label benchmarks and train metric-objective selector models."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import ARTIFACTS_DIR, RESULTS_DIR


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main(results_dir: Path, artifacts_dir: Path) -> None:
    python = sys.executable
    print(f"\n{'=' * 60}")
    print("Metric objective pipeline: label -> train (memory, recall, latency selectors)")
    print(f"{'=' * 60}")

    _run([
        python, "scripts/label_data.py",
        "--results-dir", str(results_dir),
    ])
    _run([
        python, "scripts/train_models.py",
        "--results-dir", str(results_dir),
        "--artifacts-dir", str(artifacts_dir),
    ])

    print(f"\nPipeline complete. Results in {results_dir}/runs/<run_id>/, artifacts in {artifacts_dir}/runs/<run_id>/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(RESULTS_DIR))
    parser.add_argument("--artifacts-dir", type=Path, default=Path(ARTIFACTS_DIR))
    args = parser.parse_args()
    main(args.results_dir, args.artifacts_dir)
