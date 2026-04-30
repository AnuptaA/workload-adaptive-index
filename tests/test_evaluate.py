import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    composite_index_selection_comparison,
    constrained_index_selection_comparison,
    constraint_outcomes_for_labels,
    evaluate_index_selection,
    index_selection_metric_comparison,
    mean_composite_score_for_labels,
    rmse,
)
from src.labeling import CONFIG_COLS, CONSTRAINED_ORACLE_LABEL, ORACLE_LATENCY_LABEL, ORACLE_MEMORY_LABEL


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


def _make_test_configs() -> pd.DataFrame:
    return pd.DataFrame([{
        "dataset": "sift-1M",
        "n_fraction": 0.05,
        "N": 50000,
        "d": 128,
        "k": 10,
        ORACLE_LATENCY_LABEL: "HNSW",
        ORACLE_MEMORY_LABEL: "IVF_PQ",
        "memory_budget_ratio": 20.0,
        "memory_budget_mb": 500.0,
        "recall_target": 0.95,
        CONSTRAINED_ORACLE_LABEL: "HNSW",
    }])


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
        test_df = _make_test_configs()
        predicted = ["HNSW"] * len(test_df)
        result = evaluate_index_selection(predicted, test_df, ORACLE_LATENCY_LABEL)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        test_df = _make_test_configs()
        predicted = ["IVF_PQ"] * len(test_df)
        result = evaluate_index_selection(predicted, test_df, ORACLE_LATENCY_LABEL)
        assert result["accuracy"] == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        test_df = _make_test_configs()
        with pytest.raises(ValueError):
            evaluate_index_selection(["HNSW", "IVF_FLAT"], test_df, ORACLE_LATENCY_LABEL)


class TestIndexSelectionMetricComparison:
    def test_latency_objective_uses_latency_metric(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs()
        result = index_selection_metric_comparison(
            "latency",
            test_df,
            benchmarks,
            ["IVF_PQ"],
            oracle_col=ORACLE_LATENCY_LABEL,
            random_mc_trials=4,
        )
        assert result["metric"] == "mean_latency_ms"
        assert result["oracle_mean"] == pytest.approx(3.0)
        assert result["model_mean"] == pytest.approx(8.0)

    def test_memory_objective_uses_memory_metric(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs()
        result = index_selection_metric_comparison(
            "memory",
            test_df,
            benchmarks,
            ["HNSW"],
            oracle_col=ORACLE_MEMORY_LABEL,
            random_mc_trials=4,
        )
        assert result["metric"] == "index_size_mb"
        assert result["oracle_mean"] == pytest.approx(50.0)
        assert result["model_mean"] == pytest.approx(400.0)
        assert test_df[CONFIG_COLS].shape[1] == len(CONFIG_COLS)

    def test_composite_objective_oracle_beats_wrong_prediction(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs()
        result = composite_index_selection_comparison(
            test_df,
            benchmarks,
            ["IVF_PQ"],
            (0.6, 0.2, 0.2),
        )
        assert result["metric"] == "composite_score"
        assert result["oracle_mean"] < result["model_mean"]
        assert result["oracle_mean"] == pytest.approx(
            mean_composite_score_for_labels(test_df, benchmarks, ["HNSW"], (0.6, 0.2, 0.2))
        )

    def test_constraint_outcomes_report_penalties(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs()
        outcomes = constraint_outcomes_for_labels(test_df, benchmarks, ["IVF_PQ"])
        assert outcomes.iloc[0]["memory_budget_satisfied"]
        assert not outcomes.iloc[0]["recall_target_satisfied"]
        assert outcomes.iloc[0]["recall_shortfall"] == pytest.approx(0.13)

    def test_constrained_comparison_uses_penalty_metric(self):
        benchmarks = _make_benchmarks()
        test_df = _make_test_configs()
        result = constrained_index_selection_comparison(
            test_df,
            benchmarks,
            ["IVF_PQ"],
        )
        assert result["metric"] == "constraint_score"
        assert result["oracle_mean_objective_score"] < result["model_mean_objective_score"]
        assert result["model_recall_target_satisfaction_rate"] == pytest.approx(0.0)
