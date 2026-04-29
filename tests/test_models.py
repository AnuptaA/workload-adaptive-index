import pandas as pd

from src.models import (
    load_artifacts,
    predict_index,
    predict_index_for_constraints,
    predict_performance_for_candidates,
    save_selector_artifacts,
    train_metric_regressors,
    train_selector_model,
)


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset": "sift-1M",
            "n_fraction": 0.05,
            "N": 50000,
            "d": 128,
            "k": 10,
        },
        {
            "dataset": "sift-1M",
            "n_fraction": 0.10,
            "N": 100000,
            "d": 128,
            "k": 10,
        },
        {
            "dataset": "gist-1M",
            "n_fraction": 0.05,
            "N": 50000,
            "d": 960,
            "k": 50,
        },
        {
            "dataset": "fashion-mnist",
            "n_fraction": 0.20,
            "N": 200000,
            "d": 784,
            "k": 100,
        },
    ])


def test_save_load_and_predict_selector_models(tmp_path):
    train = _training_frame()
    labels = pd.Series(["IVF_PQ", "IVF_FLAT", "HNSW", "IVF_PQ"])
    benchmark_rows = []
    for _, row in train.iterrows():
        for index_type, latency, memory, recall in [
            ("IVF_FLAT", 4.0, 100.0, 0.94),
            ("IVF_PQ", 2.0, 20.0, 0.82),
            ("HNSW", 1.0, 180.0, 0.99),
        ]:
            benchmark_rows.append({
                **row.to_dict(),
                "index_type": index_type,
                "mean_latency_ms": latency,
                "index_size_mb": memory,
                "recall_at_k": recall,
            })
    benchmark = pd.DataFrame(benchmark_rows)
    models = {
        "memory_selector_model": train_selector_model(train, labels, seed=1),
        "recall_selector_model": train_selector_model(train, labels, seed=1),
        "latency_selector_model": train_selector_model(train, labels, seed=1),
        **train_metric_regressors(benchmark, seed=1),
    }

    save_selector_artifacts(models, tmp_path)
    loaded = load_artifacts(tmp_path)

    assert set(loaded) == set(models)
    row = train.iloc[0]
    assert predict_index(loaded, row, "memory") in {"IVF_PQ", "IVF_FLAT", "HNSW"}
    assert predict_index(loaded, row, "recall") in {"IVF_PQ", "IVF_FLAT", "HNSW"}
    assert predict_index(loaded, row, "latency") in {"IVF_PQ", "IVF_FLAT", "HNSW"}
    predicted_metrics = predict_performance_for_candidates(loaded, row)
    assert set(predicted_metrics["index_type"]) == {"IVF_PQ", "IVF_FLAT", "HNSW"}
    assert {
        "predicted_latency_ms",
        "predicted_memory_mb",
        "predicted_recall",
    }.issubset(predicted_metrics.columns)
    assert predict_index_for_constraints(loaded, row, 1.0, 0.95) in {"IVF_PQ", "IVF_FLAT", "HNSW"}
