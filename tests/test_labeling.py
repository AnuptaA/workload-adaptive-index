import numpy as np
import pandas as pd
import pytest

from src.config import MEMORY_VIOLATION_WEIGHT, RECALL_VIOLATION_WEIGHT
from src.labeling import (
    balance_labels,
    check_class_distribution,
    compute_violation_score,
    select_winner,
)

def _make_row(
    index_type: str,
    index_size_mb: float,
    memory_budget_mb: float,
    recall_at_k: float,
    recall_target: float,
    mean_latency_ms: float = 10.0,
) -> dict:
    return {
        "index_type": index_type,
        "index_size_mb": index_size_mb,
        "memory_budget_mb": memory_budget_mb,
        "recall_at_k": recall_at_k,
        "recall_target": recall_target,
        "mean_latency_ms": mean_latency_ms,
        "dataset": "sift-1M",
        "n_fraction": 0.05,
        "N": 50000,
        "d": 128,
        "k": 10,
    }

def _make_group(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)

class TestComputeViolationScore:
    def test_no_violation(self):
        row = pd.Series(_make_row("HNSW", index_size_mb=100, memory_budget_mb=256,
                                  recall_at_k=0.95, recall_target=0.90))
        assert compute_violation_score(row) == pytest.approx(0.0)

    def test_memory_violation_only(self):
        row = pd.Series(_make_row("HNSW", index_size_mb=300, memory_budget_mb=256,
                                  recall_at_k=0.95, recall_target=0.90))
        expected = MEMORY_VIOLATION_WEIGHT * ((300 - 256) / 256)
        assert compute_violation_score(row) == pytest.approx(expected)

    def test_recall_violation_only(self):
        row = pd.Series(_make_row("HNSW", index_size_mb=100, memory_budget_mb=256,
                                  recall_at_k=0.80, recall_target=0.95))
        expected = RECALL_VIOLATION_WEIGHT * (0.95 - 0.80)
        assert compute_violation_score(row) == pytest.approx(expected)

    def test_both_violations(self):
        row = pd.Series(_make_row("HNSW", index_size_mb=300, memory_budget_mb=256,
                                  recall_at_k=0.80, recall_target=0.95))
        expected = (
            MEMORY_VIOLATION_WEIGHT * ((300 - 256) / 256)
            + RECALL_VIOLATION_WEIGHT * (0.95 - 0.80)
        )
        assert compute_violation_score(row) == pytest.approx(expected)

class TestSelectWinner:
    def test_tight_memory_large_n_ivf_pq_wins(self):
        """IVF_PQ should win under tight memory budget with large N."""
        group = _make_group([
            _make_row("IVF_PQ", index_size_mb=60, memory_budget_mb=64,
                      recall_at_k=0.85, recall_target=0.80, mean_latency_ms=20.0),
            _make_row("IVF_FLAT", index_size_mb=512, memory_budget_mb=64,
                      recall_at_k=0.92, recall_target=0.80, mean_latency_ms=15.0),
            _make_row("HNSW", index_size_mb=800, memory_budget_mb=64,
                      recall_at_k=0.98, recall_target=0.80, mean_latency_ms=5.0),
        ])
        assert select_winner(group) == "IVF_PQ"

    def test_small_n_high_recall_not_ivf_pq(self):
        """With small N and high recall, IVF_FLAT or HNSW should win (not IVF_PQ)."""
        group = _make_group([
            _make_row("IVF_PQ", index_size_mb=10, memory_budget_mb=512,
                      recall_at_k=0.75, recall_target=0.95, mean_latency_ms=5.0),
            _make_row("IVF_FLAT", index_size_mb=30, memory_budget_mb=512,
                      recall_at_k=0.96, recall_target=0.95, mean_latency_ms=8.0),
            _make_row("HNSW", index_size_mb=50, memory_budget_mb=512,
                      recall_at_k=0.99, recall_target=0.95, mean_latency_ms=3.0),
        ])
        winner = select_winner(group)
        assert winner in ("IVF_FLAT", "HNSW")

    def test_unconstrained_memory_high_recall_hnsw_wins(self):
        """HNSW wins with unconstrained memory and high recall target."""
        group = _make_group([
            _make_row("HNSW", index_size_mb=500, memory_budget_mb=512,
                      recall_at_k=0.99, recall_target=0.99, mean_latency_ms=3.0),
            _make_row("IVF_FLAT", index_size_mb=300, memory_budget_mb=512,
                      recall_at_k=0.99, recall_target=0.99, mean_latency_ms=12.0),
            _make_row("IVF_PQ", index_size_mb=50, memory_budget_mb=512,
                      recall_at_k=0.78, recall_target=0.99, mean_latency_ms=6.0),
        ])
        assert select_winner(group) == "HNSW"

    def test_all_infeasible_picks_minimum_violation(self):
        """When no index satisfies constraints, pick the one with lowest violation score."""
        group = _make_group([
            _make_row("IVF_PQ", index_size_mb=100, memory_budget_mb=64,
                      recall_at_k=0.70, recall_target=0.95, mean_latency_ms=5.0),
            _make_row("IVF_FLAT", index_size_mb=200, memory_budget_mb=64,
                      recall_at_k=0.70, recall_target=0.95, mean_latency_ms=8.0),
            _make_row("HNSW", index_size_mb=500, memory_budget_mb=64,
                      recall_at_k=0.70, recall_target=0.95, mean_latency_ms=3.0),
        ])
        # IVF_PQ has lowest memory violation (100 vs 64 budget) + same recall violation
        winner = select_winner(group)
        assert winner == "IVF_PQ"

class TestCheckClassDistribution:
    def test_sums_to_one(self):
        df = pd.DataFrame({"label": ["HNSW", "IVF_FLAT", "IVF_PQ", "HNSW", "HNSW"]})
        dist = check_class_distribution(df)
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_correct_fractions(self):
        df = pd.DataFrame({"label": ["HNSW"] * 6 + ["IVF_FLAT"] * 4})
        dist = check_class_distribution(df)
        assert dist["HNSW"] == pytest.approx(0.6)
        assert dist["IVF_FLAT"] == pytest.approx(0.4)

class TestBalanceLabels:
    def _make_labeled_df(self, n_hnsw: int, n_ivf: int) -> pd.DataFrame:
        rows = []
        for i in range(n_hnsw):
            rows.append({"label": "HNSW", "dataset": "sift-1M", "n_fraction": 0.05,
                         "N": i, "d": 128, "k": 10, "memory_budget_mb": 512,
                         "recall_target": 0.90, "index_type": "HNSW"})
        for i in range(n_ivf):
            rows.append({"label": "IVF_FLAT", "dataset": "sift-1M", "n_fraction": 0.10,
                         "N": i + 1000, "d": 128, "k": 10, "memory_budget_mb": 512,
                         "recall_target": 0.90, "index_type": "IVF_FLAT"})
        return pd.DataFrame(rows)

    def test_balancing_reduces_dominant(self):
        df = self._make_labeled_df(n_hnsw=100, n_ivf=20)
        balanced = balance_labels(df, threshold=0.60, seed=42)
        dist = check_class_distribution(balanced)
        assert dist.get("HNSW", 0) <= 0.75  # reduced from ~0.83

    def test_no_balancing_needed(self):
        df = self._make_labeled_df(n_hnsw=50, n_ivf=50)
        balanced = balance_labels(df, threshold=0.60, seed=42)
        assert len(balanced) == len(df)

    def test_seed_reproducibility(self):
        df = self._make_labeled_df(n_hnsw=100, n_ivf=20)
        a = balance_labels(df, threshold=0.60, seed=7)
        b = balance_labels(df, threshold=0.60, seed=7)
        assert len(a) == len(b)
