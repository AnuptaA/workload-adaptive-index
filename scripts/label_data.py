"""Label benchmark results and save labeled training pairs."""

import argparse
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR
from src.labeling import balance_labels, check_class_distribution, label_benchmarks

def main(results_dir: Path, balance: bool = False) -> None:
    results_dir = Path(results_dir)
    benchmarks_path = results_dir / "benchmarks.csv"

    if not benchmarks_path.exists():
        raise FileNotFoundError(f"{benchmarks_path} not found. Run run_benchmark.py first.")

    df = pd.read_csv(benchmarks_path)
    labeled = label_benchmarks(df)

    if balance:
        labeled = balance_labels(labeled)

    dist = check_class_distribution(labeled)
    print("Class distribution:")
    for label, frac in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {label}: {frac:.3f}")

    out = results_dir / "labeled.csv"
    labeled.to_csv(out, index=False)
    print(f"Saved {len(labeled)} rows to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--balance", action="store_true")
    args = parser.parse_args()
    main(Path(args.results_dir), args.balance)
