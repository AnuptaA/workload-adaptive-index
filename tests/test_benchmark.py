import numpy as np
import pytest

from src.benchmark import benchmark_single, query_index
from src.index_builder import build_index

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
    def test_self_search_shape(self, small_index):
        index, train = small_index
        k = 10
        retrieved, _, _ = query_index(index, train[:50], k)
        assert retrieved.shape == (50, k)

class TestBenchmarkSingle:
    def test_returns_expected_keys(self, small_index, small_queries):
        index, train = small_index
        gt = np.tile(np.arange(len(train)), (len(small_queries), 1)).astype(np.int32)
        result = benchmark_single(index, small_queries, gt, k=5)
        assert set(result.keys()) == {"k", "mean_latency_ms", "p99_latency_ms", "recall_at_k"}

    def test_k_value_stored(self, small_index, small_queries):
        index, train = small_index
        gt = np.tile(np.arange(len(train)), (len(small_queries), 1)).astype(np.int32)
        result = benchmark_single(index, small_queries, gt, k=7)
        assert result["k"] == 7

    def test_recall_in_range(self, small_index, small_queries):
        index, train = small_index
        gt = np.tile(np.arange(len(train)), (len(small_queries), 1)).astype(np.int32)
        result = benchmark_single(index, small_queries, gt, k=5)
        assert 0.0 <= result["recall_at_k"] <= 1.0

    def test_latencies_positive(self, small_index, small_queries):
        index, train = small_index
        gt = np.tile(np.arange(len(train)), (len(small_queries), 1)).astype(np.int32)
        result = benchmark_single(index, small_queries, gt, k=5)
        assert result["mean_latency_ms"] > 0
        assert result["p99_latency_ms"] > 0

    def test_perfect_recall_on_exact_index(self):
        """Flat L2 index IS exact search, so recall against its own ground truth must be 1.0."""
        from src.benchmark import compute_exact_ground_truth
        rng = np.random.default_rng(42)
        train_vecs = rng.random((200, 16)).astype(np.float32)
        query_vecs = rng.random((30, 16)).astype(np.float32)
        # compute_exact_ground_truth builds a FlatL2 internally; querying the same flat index
        # must agree with it exactly
        import faiss as _faiss
        flat = _faiss.IndexFlatL2(16)
        flat.add(train_vecs)
        gt = compute_exact_ground_truth(train_vecs, query_vecs, k=5)
        result = benchmark_single(flat, query_vecs, gt, k=5)
        assert result["recall_at_k"] == pytest.approx(1.0)
