import numpy as np
import pandas as pd
import pytest

from src.evaluate import constraint_violation_rate, evaluate_index_selection, rmse
from src.labeling import CONFIG_COLS


def _make_benchmarks() -> pd.DataFrame:
    rows = []
    for dataset in ["sift-1M"]:
        for n_frac, N in [(0.05, 50000)]:
            for k in [10]:
                for mem_budget in [256]:
                    for recall_target in [0.90]:
                        for index_type, lat, mem, rec in [
                            ("HNSW", 3.0, 400.0, 0.99),
                            ("IVF_FLAT", 12.0, 200.0, 0.95),
                            ("IVF_PQ", 8.0, 50.0, 0.82),
                        ]:
                            rows.append({
                                "dataset": dataset,
                                "n_fraction": n_frac,
                                "N": N,
                                "d": 128,
                                "k": k,
                                "memory_budget_mb": mem_budget,
                                "recall_target": recall_target,
                                "index_type": index_type,
                                "mean_latency_ms": lat,
                                "p99_latency_ms": lat * 2,
                                "recall_at_k": rec,
                                "index_size_mb": mem,
                                "build_time_s": 1.0,
                            })
    return pd.DataFrame(rows)


def _make_test_configs(benchmarks: pd.DataFrame) -> pd.DataFrame:
    configs = benchmarks[CONFIG_COLS + ["index_type"]].copy()
    configs["label"] = "HNSW"
    return configs[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)


class TestRmse:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_error(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((9 + 16) / 2))


class TestEvaluateIndexSelection:
    def test_perfect_accuracy(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs(benchmarks)
        predicted = ["HNSW"] * len(test_df)
        result = evaluate_index_selection(predicted, test_df, benchmarks)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs(benchmarks)
        predicted = ["IVF_PQ"] * len(test_df)
        result = evaluate_index_selection(predicted, test_df, benchmarks)
        assert result["accuracy"] == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs(benchmarks)
        with pytest.raises(ValueError):
            evaluate_index_selection(["HNSW", "IVF_FLAT"], test_df, benchmarks)


class TestConstraintViolationRate:
    def test_no_violations(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs(benchmarks)
        # HNSW has index_size_mb=400, budget=256 -> memory violation
        # use IVF_FLAT which has index_size_mb=200 and recall=0.95 > target=0.90
        predicted = ["IVF_FLAT"] * len(test_df)
        rate = constraint_violation_rate(predicted, test_df, benchmarks)
        assert rate == pytest.approx(0.0)

    def test_all_violations(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs(benchmarks)
        # HNSW: index_size_mb=400 > budget=256 -> memory violation
        predicted = ["HNSW"] * len(test_df)
        rate = constraint_violation_rate(predicted, test_df, benchmarks)
        assert rate == pytest.approx(1.0)

    def test_empty_predictions(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs(benchmarks)
        rate = constraint_violation_rate([], test_df.iloc[:0], benchmarks)
        assert rate == pytest.approx(0.0)
