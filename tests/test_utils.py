import time

import numpy as np
import pytest

from src.utils import compute_recall_at_k, measure_peak_memory_mb, subsample_vectors, time_fn

class TestSubsampleVectors:
    def test_output_shape(self):
        vecs = np.zeros((1000, 128), dtype=np.float32)
        out = subsample_vectors(vecs, fraction=0.1)
        assert out.shape == (100, 128)

    def test_fraction_respected(self):
        vecs = np.zeros((500, 64), dtype=np.float32)
        out = subsample_vectors(vecs, fraction=0.2)
        assert out.shape[0] == 100

    def test_seed_reproducibility(self):
        vecs = np.arange(1000 * 4, dtype=np.float32).reshape(1000, 4)
        a = subsample_vectors(vecs, fraction=0.1, seed=0)
        b = subsample_vectors(vecs, fraction=0.1, seed=0)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        vecs = np.arange(1000 * 4, dtype=np.float32).reshape(1000, 4)
        a = subsample_vectors(vecs, fraction=0.1, seed=0)
        b = subsample_vectors(vecs, fraction=0.1, seed=1)
        assert not np.array_equal(a, b)

    def test_minimum_one_row(self):
        vecs = np.zeros((10, 4), dtype=np.float32)
        out = subsample_vectors(vecs, fraction=0.001)
        assert out.shape[0] >= 1

    def test_no_replacement(self):
        vecs = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
        out = subsample_vectors(vecs, fraction=0.5)
        # check no duplicate rows by looking at first column values
        first_col = out[:, 0].tolist()
        assert len(first_col) == len(set(first_col))

class TestTimeFn:
    def test_elapsed_positive(self):
        _, elapsed = time_fn(time.sleep, 0.01)
        assert elapsed > 0

    def test_result_matches_direct_call(self):
        result, _ = time_fn(lambda x: x * 2, 21)
        assert result == 42

    def test_passes_kwargs(self):
        def add(a, b=0):
            return a + b

        result, _ = time_fn(add, 1, b=2)
        assert result == 3

class TestComputeRecallAtK:
    def test_perfect_retrieval(self):
        gt = np.array([[0, 1, 2], [3, 4, 5]])
        retrieved = np.array([[0, 1, 2], [3, 4, 5]])
        assert compute_recall_at_k(retrieved, gt, k=3) == pytest.approx(1.0)

    def test_zero_overlap(self):
        gt = np.array([[0, 1, 2], [3, 4, 5]])
        retrieved = np.array([[6, 7, 8], [9, 10, 11]])
        assert compute_recall_at_k(retrieved, gt, k=3) == pytest.approx(0.0)

    def test_partial_overlap(self):
        gt = np.array([[0, 1, 2, 3]])
        retrieved = np.array([[0, 1, 9, 10]])
        # 2 of 2 true neighbors found (k=2: gt[:2] = {0,1}, retrieved hits both)
        assert compute_recall_at_k(retrieved, gt, k=2) == pytest.approx(1.0)

    def test_k_equals_one(self):
        gt = np.array([[5, 1, 2]])
        retrieved = np.array([[5]])
        assert compute_recall_at_k(retrieved, gt, k=1) == pytest.approx(1.0)

    def test_k_equals_one_miss(self):
        gt = np.array([[5, 1, 2]])
        retrieved = np.array([[99]])
        assert compute_recall_at_k(retrieved, gt, k=1) == pytest.approx(0.0)

class TestMeasurePeakMemoryMb:
    def test_peak_positive(self):
        def allocate():
            return list(range(100_000))

        _, peak_mb = measure_peak_memory_mb(allocate)
        assert peak_mb > 0

    def test_result_returned(self):
        result, _ = measure_peak_memory_mb(lambda: 42)
        assert result == 42
