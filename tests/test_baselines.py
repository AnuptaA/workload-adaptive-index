import numpy as np
import pandas as pd
import pytest

from src.baselines import faiss_rule_based, mean_latency_for_labels
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
            "memory_budget_mb": 256,
            "recall_target": 0.90,
            "index_type": index_type,
            "mean_latency_ms": lat,
            "p99_latency_ms": lat * 2,
            "recall_at_k": rec,
            "index_size_mb": mem,
            "build_time_s": 1.0,
        })
    return pd.DataFrame(rows)


def _make_test_config(memory_budget_mb: float, recall_target: float, N: int = 50000) -> pd.DataFrame:
    return pd.DataFrame([{
        "dataset": "sift-1M",
        "n_fraction": 0.05,
        "N": N,
        "d": 128,
        "k": 10,
        "memory_budget_mb": memory_budget_mb,
        "recall_target": recall_target,
        "label": "HNSW",
    }])


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
            "memory_budget_mb": 256,
            "recall_target": 0.90,
        }])
        mean_latency_for_labels(bad_config, benchmarks, ["HNSW"])
        captured = capsys.readouterr()
        assert "Warning" in captured.out


class TestFaissRuleBased:
    def test_tight_memory_gives_ivf_pq(self):
        benchmarks = _make_benchmarks()
        # N=50000, d=128 → raw_mb = 50000*128*4/1e6 = 25.6 MB; budget=20 → raw > budget
        test_df = _make_test_config(memory_budget_mb=20, recall_target=0.90)
        result = faiss_rule_based(test_df, benchmarks)
        assert result["predicted_index"].iloc[0] == "IVF_PQ"

    def test_high_recall_gives_hnsw(self):
        benchmarks = _make_benchmarks()
        # raw_mb = 25.6 < budget=512, recall_target=0.99 >= 0.95
        test_df = _make_test_config(memory_budget_mb=512, recall_target=0.99)
        result = faiss_rule_based(test_df, benchmarks)
        assert result["predicted_index"].iloc[0] == "HNSW"

    def test_default_gives_ivf_flat(self):
        benchmarks = _make_benchmarks()
        # raw_mb = 25.6 < budget=512, recall_target=0.90 < 0.95
        test_df = _make_test_config(memory_budget_mb=512, recall_target=0.90)
        result = faiss_rule_based(test_df, benchmarks)
        assert result["predicted_index"].iloc[0] == "IVF_FLAT"
