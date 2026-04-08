"""Download all benchmark datasets."""

import argparse
from pathlib import Path

from src.config import DATA_DIR, DATASETS
from src.data_loader import download_dataset

def main(data_dir: Path) -> None:
    data_dir = Path(data_dir)
    for name in DATASETS:
        path = download_dataset(name, data_dir)
        print(f"  {name}: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()
    main(Path(args.data_dir))
