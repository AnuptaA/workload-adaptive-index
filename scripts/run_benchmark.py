"""Run full benchmark sweep and save results."""

import argparse
from pathlib import Path

from src.config import DATA_DIR, RESULTS_DIR
from src.benchmark import run_benchmark
from src.run_store import create_run_dir

def main(
    data_dir: Path,
    results_dir: Path,
    verbose: bool = False,
    run_id: str = "",
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = create_run_dir(results_dir, run_id)

    df = run_benchmark(Path(data_dir), run_dir, verbose=verbose)
    out = run_dir / "benchmarks.csv"
    df.to_csv(out, index=False)
    print(f"Run id: {run_id}")
    print(f"Saved {len(df)} rows to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--run-id",
        default="",
        help="Run id to use; if omitted a timestamped run id is created.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed build and query metrics during benchmarking.",
    )
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.results_dir), verbose=args.verbose, run_id=args.run_id)
