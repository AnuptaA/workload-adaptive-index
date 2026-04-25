from pathlib import Path

import h5py
import numpy as np
import requests
from tqdm import tqdm

from src.config import DATASET_URLS

def download_dataset(name: str, data_dir: Path) -> Path:
    """Download dataset HDF5 if not present. Returns local path."""
    if name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {name}. Known: {list(DATASET_URLS)}")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    url = DATASET_URLS[name]
    dest = data_dir / f"{name}.hdf5"

    print(f"Checking {name}...")
    response = requests.head(url, timeout=30, allow_redirects=True)
    response.raise_for_status()
    expected_size = int(response.headers.get("content-length", 0))

    if dest.exists() and (expected_size == 0 or dest.stat().st_size == expected_size):
        return dest

    if dest.exists():
        print(f"Partial download detected for {name} ({dest.stat().st_size} / {expected_size} bytes). Re-downloading.")
        dest.unlink()

    tmp = dest.with_suffix(".tmp")
    print(f"Downloading {name} from {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))

    tmp.rename(dest)
    return dest

def load_dataset(
    name: str, data_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (train_vectors float32, query_vectors float32, ground_truth_neighbors int32).

    Expects HDF5 with keys 'train', 'test', 'neighbors' (ANN benchmarks standard).
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{name}.hdf5"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_dataset('{name}', ...) first."
        )

    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        queries = np.array(f["test"], dtype=np.float32)
        gt = np.array(f["neighbors"], dtype=np.int32)

    return train, queries, gt
