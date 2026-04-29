import pandas as pd
import pytest

from src.labeling import (
    ORACLE_LATENCY_LABEL,
    ORACLE_MEMORY_LABEL,
    ORACLE_RECALL_LABEL,
    WEIGHT_COLS,
    balance_labels,
    check_class_distribution,
    constraint_scores,
    expand_constraint_grid,
    generate_weight_grid,
    label_benchmarks,
    score_predicted_constraints,
    select_winner_for_constraints,
    select_winner_for_weights,
    select_winner_for_objective,
    validate_composite_weights,
)


def _make_row(
    index_type: str,
    index_size_mb: float,
    recall_at_k: float,
    mean_latency_ms: float,
) -> dict:
    return {
        "index_type": index_type,
        "index_size_mb": index_size_mb,
        "recall_at_k": recall_at_k,
        "mean_latency_ms": mean_latency_ms,
        "dataset": "sift-1M",
        "n_fraction": 0.05,
        "N": 50000,
        "d": 128,
        "k": 10,
    }


class TestSelectWinnerForObjective:
    def test_memory_oracle_minimizes_index_size(self):
        group = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=8.0),
            _make_row("IVF_FLAT", index_size_mb=200, recall_at_k=0.95, mean_latency_ms=12.0),
            _make_row("HNSW", index_size_mb=400, recall_at_k=0.99, mean_latency_ms=3.0),
        ])
        assert select_winner_for_objective(group, "memory") == "IVF_PQ"

    def test_recall_oracle_maximizes_recall(self):
        group = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=8.0),
            _make_row("IVF_FLAT", index_size_mb=200, recall_at_k=0.95, mean_latency_ms=12.0),
            _make_row("HNSW", index_size_mb=400, recall_at_k=0.99, mean_latency_ms=3.0),
        ])
        assert select_winner_for_objective(group, "recall") == "HNSW"

    def test_latency_oracle_minimizes_mean_latency(self):
        group = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=8.0),
            _make_row("IVF_FLAT", index_size_mb=200, recall_at_k=0.95, mean_latency_ms=12.0),
            _make_row("HNSW", index_size_mb=400, recall_at_k=0.99, mean_latency_ms=3.0),
        ])
        assert select_winner_for_objective(group, "latency") == "HNSW"

    def test_tie_breaks_are_deterministic(self):
        group = pd.DataFrame([
            _make_row("HNSW", index_size_mb=10, recall_at_k=0.90, mean_latency_ms=5.0),
            _make_row("IVF_FLAT", index_size_mb=10, recall_at_k=0.90, mean_latency_ms=5.0),
            _make_row("IVF_PQ", index_size_mb=10, recall_at_k=0.90, mean_latency_ms=5.0),
        ])
        assert select_winner_for_objective(group, "memory") == "IVF_FLAT"
        assert select_winner_for_objective(group, "latency") == "IVF_FLAT"


class TestSelectWinnerForWeights:
    def test_weights_change_composite_oracle(self):
        group = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=8.0),
            _make_row("IVF_FLAT", index_size_mb=200, recall_at_k=0.95, mean_latency_ms=12.0),
            _make_row("HNSW", index_size_mb=400, recall_at_k=0.99, mean_latency_ms=3.0),
        ])
        assert select_winner_for_weights(group, (1.0, 0.0, 0.0)) == "HNSW"
        assert select_winner_for_weights(group, (0.0, 0.0, 1.0)) == "IVF_PQ"

    def test_invalid_weight_triples_raise(self):
        group = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=8.0),
            _make_row("HNSW", index_size_mb=400, recall_at_k=0.99, mean_latency_ms=3.0),
        ])
        with pytest.raises(ValueError):
            select_winner_for_weights(group, (0.5, 0.5, 0.5))
        with pytest.raises(ValueError):
            select_winner_for_weights(group, (-0.1, 0.6, 0.5))

    def test_equal_metric_ranges_tie_break_deterministically(self):
        group = pd.DataFrame([
            _make_row("HNSW", index_size_mb=10, recall_at_k=0.90, mean_latency_ms=5.0),
            _make_row("IVF_FLAT", index_size_mb=10, recall_at_k=0.90, mean_latency_ms=5.0),
            _make_row("IVF_PQ", index_size_mb=10, recall_at_k=0.90, mean_latency_ms=5.0),
        ])
        assert select_winner_for_weights(group, (1 / 3, 1 / 3, 1 / 3)) == "IVF_FLAT"

    def test_weight_grid_sums_to_one(self):
        grid = generate_weight_grid(0.5)
        assert list(grid.columns) == WEIGHT_COLS
        assert len(grid) == 6
        for total in grid.sum(axis=1):
            assert total == pytest.approx(1.0)
        assert validate_composite_weights((0.25, 0.25, 0.5)).w_memory == pytest.approx(0.5)


class TestConstrainedObjective:
    def test_penalty_score_prefers_fast_index_when_constraints_satisfied(self):
        group = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=8.0),
            _make_row("IVF_FLAT", index_size_mb=200, recall_at_k=0.95, mean_latency_ms=12.0),
            _make_row("HNSW", index_size_mb=400, recall_at_k=0.99, mean_latency_ms=3.0),
        ])
        assert select_winner_for_constraints(group, memory_budget_mb=500.0, recall_target=0.90) == "HNSW"

    def test_memory_overrun_is_heavily_penalized(self):
        group = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=8.0),
            _make_row("HNSW", index_size_mb=400, recall_at_k=0.99, mean_latency_ms=3.0),
        ])
        assert select_winner_for_constraints(group, memory_budget_mb=100.0, recall_target=0.80) == "IVF_PQ"

    def test_recall_shortfall_is_heavily_penalized(self):
        group = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=1.0),
            _make_row("IVF_FLAT", index_size_mb=200, recall_at_k=0.95, mean_latency_ms=12.0),
        ])
        assert select_winner_for_constraints(group, memory_budget_mb=500.0, recall_target=0.95) == "IVF_FLAT"

    def test_constraint_scores_expose_penalty_components(self):
        group = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=8.0),
            _make_row("HNSW", index_size_mb=400, recall_at_k=0.99, mean_latency_ms=3.0),
        ])
        scored = constraint_scores(group, memory_budget_mb=100.0, recall_target=0.90)
        hnsw = scored[scored["index_type"] == "HNSW"].iloc[0]
        assert hnsw["memory_overrun"] == pytest.approx(3.0)
        assert hnsw["recall_shortfall"] == pytest.approx(0.0)

    def test_score_predicted_constraints_uses_predicted_metric_columns(self):
        predictions = pd.DataFrame([
            {
                "index_type": "IVF_PQ",
                "predicted_latency_ms": 1.0,
                "predicted_memory_mb": 50.0,
                "predicted_recall": 0.82,
            },
            {
                "index_type": "HNSW",
                "predicted_latency_ms": 3.0,
                "predicted_memory_mb": 400.0,
                "predicted_recall": 0.99,
            },
        ])
        scored = score_predicted_constraints(predictions, memory_budget_mb=100.0, recall_target=0.90)
        hnsw = scored[scored["index_type"] == "HNSW"].iloc[0]
        assert hnsw["memory_overrun"] == pytest.approx(3.0)
        assert hnsw["predicted_recall"] == pytest.approx(0.99)

    def test_expand_constraint_grid_adds_budget_and_target_columns(self):
        configs = pd.DataFrame([_make_row("HNSW", 400, 0.99, 3.0)])
        expanded = expand_constraint_grid(configs, memory_budget_ratios=[0.5, 1.0], recall_targets=[0.9])
        assert len(expanded) == 2
        assert set(expanded["recall_target"]) == {0.9}
        assert set(expanded["memory_budget_ratio"]) == {0.5, 1.0}
        assert (expanded["memory_budget_mb"] > 0).all()


class TestLabelBenchmarks:
    def test_assigns_oracle_columns_per_config(self):
        df = pd.DataFrame([
            _make_row("IVF_PQ", index_size_mb=50, recall_at_k=0.82, mean_latency_ms=8.0),
            _make_row("IVF_FLAT", index_size_mb=200, recall_at_k=0.95, mean_latency_ms=12.0),
            _make_row("HNSW", index_size_mb=400, recall_at_k=0.99, mean_latency_ms=3.0),
        ])
        labeled = label_benchmarks(df)
        assert set(labeled[ORACLE_MEMORY_LABEL]) == {"IVF_PQ"}
        assert set(labeled[ORACLE_RECALL_LABEL]) == {"HNSW"}
        assert set(labeled[ORACLE_LATENCY_LABEL]) == {"HNSW"}


class TestCheckClassDistribution:
    def test_sums_to_one(self):
        df = pd.DataFrame({ORACLE_LATENCY_LABEL: ["HNSW", "IVF_FLAT", "IVF_PQ", "HNSW", "HNSW"]})
        dist = check_class_distribution(df, ORACLE_LATENCY_LABEL)
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_correct_fractions(self):
        df = pd.DataFrame({ORACLE_LATENCY_LABEL: ["HNSW"] * 6 + ["IVF_FLAT"] * 4})
        dist = check_class_distribution(df, ORACLE_LATENCY_LABEL)
        assert dist["HNSW"] == pytest.approx(0.6)
        assert dist["IVF_FLAT"] == pytest.approx(0.4)


class TestBalanceLabels:
    def _make_labeled_df(self, n_hnsw: int, n_ivf: int) -> pd.DataFrame:
        rows = []
        for i in range(n_hnsw):
            rows.append({
                ORACLE_LATENCY_LABEL: "HNSW",
                "dataset": "sift-1M",
                "n_fraction": 0.05,
                "N": i,
                "d": 128,
                "k": 10,
                "index_type": "HNSW",
            })
        for i in range(n_ivf):
            rows.append({
                ORACLE_LATENCY_LABEL: "IVF_FLAT",
                "dataset": "sift-1M",
                "n_fraction": 0.10,
                "N": i + 1000,
                "d": 128,
                "k": 10,
                "index_type": "IVF_FLAT",
            })
        return pd.DataFrame(rows)

    def test_balancing_reduces_dominant(self):
        df = self._make_labeled_df(n_hnsw=100, n_ivf=20)
        balanced = balance_labels(df, label_col=ORACLE_LATENCY_LABEL, threshold=0.60, seed=42)
        dist = check_class_distribution(balanced, ORACLE_LATENCY_LABEL)
        assert dist.get("HNSW", 0) <= 0.75

    def test_no_balancing_needed(self):
        df = self._make_labeled_df(n_hnsw=50, n_ivf=50)
        balanced = balance_labels(df, label_col=ORACLE_LATENCY_LABEL, threshold=0.60, seed=42)
        assert len(balanced) == len(df)

    def test_seed_reproducibility(self):
        df = self._make_labeled_df(n_hnsw=100, n_ivf=20)
        a = balance_labels(df, label_col=ORACLE_LATENCY_LABEL, threshold=0.60, seed=7)
        b = balance_labels(df, label_col=ORACLE_LATENCY_LABEL, threshold=0.60, seed=7)
        assert len(a) == len(b)
