import pandas as pd
import pytest

from src.baselines import faiss_rule_based_labels, mean_latency_for_labels
from src.labeling import CONFIG_COLS


def _make_benchmarks() -> pd.DataFrame:
    rows = []
    for index_type, lat, mem, rec in [
        ("HNSW", 3.0, 400.0, 0.99),
        ("IVF_FLAT", 12.0, 200.0, 0.95),
        ("IVF_PQ", 8.0, 50.0, 0.82),
    ]:
        rows.append({
            "dataset": "sift-1M",
            "n_fraction": 0.05,
            "N": 50000,
            "d": 128,
            "k": 10,
            "index_type": index_type,
            "mean_latency_ms": lat,
            "p99_latency_ms": lat * 2,
            "recall_at_k": rec,
            "index_size_mb": mem,
            "build_time_s": 1.0,
        })
    return pd.DataFrame(rows)


class TestMeanLatencyForLabels:
    def test_correct_lookup(self):
        benchmarks = _make_benchmarks()
        test_df = benchmarks[CONFIG_COLS].drop_duplicates().copy()
        test_df["label"] = "HNSW"
        test_df = test_df[CONFIG_COLS]
        result = mean_latency_for_labels(test_df, benchmarks, ["HNSW"])
        assert result == pytest.approx(3.0)

    def test_different_labels(self):
        benchmarks = _make_benchmarks()
        test_df = benchmarks[CONFIG_COLS].drop_duplicates().copy()
        result = mean_latency_for_labels(test_df, benchmarks, ["IVF_FLAT"])
        assert result == pytest.approx(12.0)

    def test_missing_lookup_warns(self, capsys):
        benchmarks = _make_benchmarks()
        bad_config = pd.DataFrame([{
            "dataset": "nonexistent",
            "n_fraction": 0.05,
            "N": 99,
            "d": 128,
            "k": 10,
        }])
        mean_latency_for_labels(bad_config, benchmarks, ["HNSW"])
        captured = capsys.readouterr()
        assert "Warning" in captured.out


class TestFaissRuleBasedLabels:
    def test_defaults_to_ivf_flat_without_constraints(self):
        config = pd.DataFrame([{
            "dataset": "sift-1M",
            "n_fraction": 0.05,
            "N": 50000,
            "d": 128,
            "k": 10,
        }])
        assert faiss_rule_based_labels(config) == ["IVF_FLAT"]

    def test_tight_memory_uses_ivf_pq(self):
        config = pd.DataFrame([{
            "dataset": "sift-1M",
            "n_fraction": 0.05,
            "N": 50000,
            "d": 128,
            "k": 10,
            "memory_budget_mb": 1.0,
        }])
        assert faiss_rule_based_labels(config) == ["IVF_PQ"]

    def test_high_recall_target_uses_hnsw(self):
        config = pd.DataFrame([{
            "dataset": "sift-1M",
            "n_fraction": 0.05,
            "N": 50000,
            "d": 128,
            "k": 10,
            "memory_budget_mb": 1000.0,
            "recall_target": 0.99,
        }])
        assert faiss_rule_based_labels(config) == ["HNSW"]

    def test_memory_budget_takes_precedence_over_recall_target(self):
        config = pd.DataFrame([{
            "dataset": "sift-1M",
            "n_fraction": 0.05,
            "N": 50000,
            "d": 128,
            "k": 10,
            "memory_budget_mb": 1.0,
            "recall_target": 0.99,
        }])
        assert faiss_rule_based_labels(config) == ["IVF_PQ"]
