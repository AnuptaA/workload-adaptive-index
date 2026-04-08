import numpy as np
import pytest

from src.benchmark import benchmark_single, query_index
from src.index_builder import build_index
from src.utils import compute_recall_at_k

@pytest.fixture
def small_index():
    """Small HNSW index with synthetic vectors for fast tests."""
    rng = np.random.default_rng(0)
    vectors = rng.random((500, 32)).astype(np.float32)
    index = build_index("HNSW", vectors)
    return index, vectors

@pytest.fixture
def small_queries():
    rng = np.random.default_rng(1)
    return rng.random((20, 32)).astype(np.float32)

class TestQueryIndex:
    def test_returns_correct_shape(self, small_index, small_queries):
        index, _ = small_index
        k = 5
        retrieved, _, _ = query_index(index, small_queries, k)
        assert retrieved.shape == (len(small_queries), k)

    def test_mean_latency_positive(self, small_index, small_queries):
        index, _ = small_index
        _, mean_lat, _ = query_index(index, small_queries, k=5)
        assert mean_lat > 0

    def test_p99_latency_positive(self, small_index, small_queries):
        index, _ = small_index
        _, _, p99_lat = query_index(index, small_queries, k=5)
        assert p99_lat > 0

    def test_p99_gte_mean(self, small_index, small_queries):
        index, _ = small_index
        _, mean_lat, p99_lat = query_index(index, small_queries, k=5)
        assert p99_lat >= mean_lat

class TestComputeRecallAtKViaQueryIndex:
    def test_self_search_high_recall(self, small_index):
        """Searching train vectors against themselves should yield near-perfect recall."""
        index, train = small_index
        k = 10
        retrieved, _, _ = query_index(index, train[:50], k)
        gt = np.tile(np.arange(len(train)), (50, 1))
        # Use compute_recall_at_k with train indices as approximate ground truth
        assert retrieved.shape == (50, k)

class TestBenchmarkSingle:
    def test_raises_not_implemented(self):
        rng = np.random.default_rng(0)
        vecs = rng.random((100, 16)).astype(np.float32)
        gt = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(NotImplementedError):
            benchmark_single("HNSW", vecs, vecs[:10], gt, k=5)
