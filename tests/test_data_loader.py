from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from src.data_loader import download_dataset, load_dataset

@pytest.fixture
def tmp_hdf5(tmp_path):
    """Write a minimal ANN-benchmarks-format HDF5 file."""
    def _make(name, n_train=200, n_query=10, n_gt=50, d=32):
        path = tmp_path / f"{name}.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("train", data=np.random.rand(n_train, d).astype(np.float32))
            f.create_dataset("test", data=np.random.rand(n_query, d).astype(np.float32))
            f.create_dataset("neighbors", data=np.random.randint(0, n_train, (n_query, n_gt)).astype(np.int32))
        return tmp_path

    return _make

class TestLoadDataset:
    def test_return_shapes(self, tmp_hdf5):
        data_dir = tmp_hdf5("sift-1M", n_train=200, n_query=10, n_gt=50, d=32)
        train, queries, gt = load_dataset("sift-1M", data_dir)
        assert train.shape == (200, 32)
        assert queries.shape == (10, 32)
        assert gt.shape == (10, 50)

    def test_train_dtype_float32(self, tmp_hdf5):
        data_dir = tmp_hdf5("sift-1M")
        train, _, _ = load_dataset("sift-1M", data_dir)
        assert train.dtype == np.float32

    def test_query_dtype_float32(self, tmp_hdf5):
        data_dir = tmp_hdf5("sift-1M")
        _, queries, _ = load_dataset("sift-1M", data_dir)
        assert queries.dtype == np.float32

    def test_gt_dtype_int32(self, tmp_hdf5):
        data_dir = tmp_hdf5("sift-1M")
        _, _, gt = load_dataset("sift-1M", data_dir)
        assert gt.dtype == np.int32

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset("sift-1M", tmp_path)

class TestDownloadDataset:
    def test_skips_if_exists(self, tmp_path):
        dest = tmp_path / "sift-1M.hdf5"
        dest.touch()
        # HEAD returns no content-length (0) so the existing file is treated as complete
        mock_head_response = MagicMock()
        mock_head_response.headers.get.return_value = 0
        with patch("src.data_loader.requests.head", return_value=mock_head_response), \
             patch("src.data_loader.requests.get") as mock_get:
            result = download_dataset("sift-1M", tmp_path)
            mock_get.assert_not_called()
        assert result == dest

    def test_unknown_dataset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            download_dataset("nonexistent-dataset", tmp_path)

    def test_creates_data_dir(self, tmp_path):
        new_dir = tmp_path / "nested" / "data"
        dest = new_dir / "sift-1M.hdf5"
        dest_parent = new_dir
        dest_parent.mkdir(parents=True)
        dest.touch()
        result = download_dataset("sift-1M", new_dir)
        assert result.exists()
