"""Generate objective-aware plots and summaries for benchmark runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.baselines import faiss_rule_based_labels
from src.config import (
    ARTIFACTS_DIR,
    MEMORY_BUDGET_RATIOS,
    RANDOM_SEED,
    RECALL_TARGETS,
    RESULTS_DIR,
)
from src.evaluate import OBJECTIVE_METRIC
from src.labeling import (
    CONFIG_COLS,
    CONSTRAINT_COLS,
    ORACLE_LATENCY_LABEL,
    ORACLE_MEMORY_LABEL,
    ORACLE_RECALL_LABEL,
    ORACLE_LABEL_COLS,
    WEIGHT_COLS,
    Objective,
    composite_scores,
    constraint_scores,
    expand_constraint_grid,
    generate_weight_grid,
    label_benchmarks,
    select_winner_for_constraints,
    select_winner_for_weights,
)
from src.models import load_artifacts, predict_index, predict_indices_for_constraints, predict_index_for_weights
from src.run_store import resolve_run_dir

INDEX_ORDER = ["IVF_FLAT", "IVF_PQ", "HNSW"]
INDEX_COLORS = {
    "IVF_FLAT": "#1f77b4",
    "IVF_PQ": "#ff7f0e",
    "HNSW": "#2ca02c",
}
STRATEGY_ORDER = ["Oracle winner", "Trained selector", "Rule-based", "Always HNSW", "Uniform random"]
STRATEGY_COLORS = {
    "Oracle winner": "#4c78a8",
    "Trained selector": "#f58518",
    "Rule-based": "#54a24b",
    "Always HNSW": "#b279a2",
    "Uniform random": "#9d755d",
}


def _constrained_grid_figsize(
    n_cols: int,
    n_rows: int,
    *,
    col_inch: float,
    row_inch: float = 3.2,
    max_width_inch: float = 24.0,
) -> tuple[float, float]:
    """Widen grids only up to ``max_width_inch`` so many memory targets stay printable."""
    col = col_inch
    if n_cols * col > max_width_inch:
        col = max(2.35, max_width_inch / max(n_cols, 1))
    return (n_cols * col, n_rows * row_inch)


RAW_ID_COLS = CONFIG_COLS + ["index_type"]
METRIC_COLS = ["mean_latency_ms", "recall_at_k", "index_size_mb"]
OBJECTIVES: list[tuple[Objective, str]] = [
    ("memory", ORACLE_MEMORY_LABEL),
    ("recall", ORACLE_RECALL_LABEL),
    ("latency", ORACLE_LATENCY_LABEL),
]


def _dedupe_raw_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(RAW_ID_COLS).drop_duplicates(subset=RAW_ID_COLS).copy()


def _load_labeled(run_dir: Path, benchmarks: pd.DataFrame) -> pd.DataFrame:
    labeled_path = run_dir / "labeled.csv"
    if labeled_path.exists():
        return pd.read_csv(labeled_path)
    return label_benchmarks(benchmarks.copy())


def _config_table(labeled_df: pd.DataFrame) -> pd.DataFrame:
    return labeled_df[CONFIG_COLS + ORACLE_LABEL_COLS].drop_duplicates(subset=CONFIG_COLS)


def _load_plot_eval_tables(
    run_dir: Path,
    configs: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Use held-out test split for model match plots when available."""
    test_path = run_dir / "splits" / "test.csv"
    if not test_path.exists():
        return configs, "all configs"

    test_df = pd.read_csv(test_path)
    eval_configs = test_df[CONFIG_COLS + ORACLE_LABEL_COLS].drop_duplicates(
        subset=CONFIG_COLS
    )
    return eval_configs, "test split"


def _run_artifact_dir(artifacts_dir: Path, run_id: str) -> Path:
    return Path(artifacts_dir) / "runs" / run_id


def _has_run_artifacts(artifacts_dir: Path, run_id: str) -> bool:
    run_artifacts_dir = _run_artifact_dir(artifacts_dir, run_id)
    required = [
        "memory_selector_model.joblib",
        "recall_selector_model.joblib",
        "latency_selector_model.joblib",
        "latency_regressor_model.joblib",
        "memory_regressor_model.joblib",
        "recall_regressor_model.joblib",
    ]
    return all((run_artifacts_dir / name).exists() for name in required)


def _model_predictions(configs: pd.DataFrame, artifacts_dir: Path, run_id: str, objective: Objective) -> list[str]:
    models = load_artifacts(_run_artifact_dir(artifacts_dir, run_id))
    return [predict_index(models, row, objective) for _, row in configs.iterrows()]


def _composite_model_predictions(
    configs: pd.DataFrame,
    models: dict,
    weights: tuple[float, float, float],
) -> list[str]:
    return [predict_index_for_weights(models, row, weights) for _, row in configs.iterrows()]


def _constrained_model_predictions(configs: pd.DataFrame, models: dict) -> list[str]:
    return predict_indices_for_constraints(models, configs)


def _lookup_strategy_metrics(
    configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
    strategy: str,
) -> pd.DataFrame:
    choices = configs[CONFIG_COLS].copy()
    choices["index_type"] = labels
    choices["strategy"] = strategy
    lookup = benchmarks[CONFIG_COLS + ["index_type"] + METRIC_COLS].drop_duplicates(
        subset=CONFIG_COLS + ["index_type"]
    )
    merged = choices.merge(lookup, on=CONFIG_COLS + ["index_type"], how="left")
    missing = merged[METRIC_COLS].isna().any(axis=1).sum()
    if missing:
        print(f"Warning: {missing}/{len(merged)} {strategy} benchmark lookups returned no match.")
    return merged


def _uniform_random_expected_metrics(configs: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    lookup = benchmarks[CONFIG_COLS + ["index_type"] + METRIC_COLS].drop_duplicates(
        subset=CONFIG_COLS + ["index_type"]
    )
    expected = (
        lookup.groupby(CONFIG_COLS, as_index=False)[METRIC_COLS]
        .mean()
        .merge(configs[CONFIG_COLS], on=CONFIG_COLS, how="inner")
    )
    expected["strategy"] = "Uniform random"
    expected["index_type"] = "Uniform random"
    return expected[CONFIG_COLS + ["strategy", "index_type"] + METRIC_COLS]


def _raw_vector_mb(n: pd.Series, d: pd.Series) -> pd.Series:
    return n.astype(float) * d.astype(float) * 4.0 / (1024 ** 2)


def _rule_based_eval_configs(configs: pd.DataFrame) -> pd.DataFrame:
    """Expand configs so FAISS rule sees the same memory/recall inputs as the original baseline."""
    expanded = configs.copy()

    if "memory_budget_mb" not in expanded.columns:
        parts = []
        for ratio in MEMORY_BUDGET_RATIOS:
            part = expanded.copy()
            part["memory_budget_ratio"] = float(ratio)
            part["memory_budget_mb"] = _raw_vector_mb(part["N"], part["d"]) * float(ratio)
            parts.append(part)
        expanded = pd.concat(parts, ignore_index=True)

    if "recall_target" not in expanded.columns:
        parts = []
        for target in RECALL_TARGETS:
            part = expanded.copy()
            part["recall_target"] = float(target)
            parts.append(part)
        expanded = pd.concat(parts, ignore_index=True)

    return expanded


def _rule_based_eval_configs_for_weights(
    configs: pd.DataFrame,
    weights: tuple[float, float, float],
) -> pd.DataFrame:
    """Attach FAISS rule constraints derived from composite weights."""
    w_recall, _, w_memory = weights
    recall_min, recall_max = min(RECALL_TARGETS), max(RECALL_TARGETS)
    memory_min, memory_max = min(MEMORY_BUDGET_RATIOS), max(MEMORY_BUDGET_RATIOS)

    expanded = configs.copy()
    expanded["recall_target"] = recall_min + float(w_recall) * (recall_max - recall_min)
    # Higher memory weight means stronger pressure to fit a tighter memory budget.
    expanded["memory_budget_ratio"] = memory_max - float(w_memory) * (memory_max - memory_min)
    expanded["memory_budget_mb"] = (
        _raw_vector_mb(expanded["N"], expanded["d"]) * expanded["memory_budget_ratio"]
    )
    return expanded


def _rule_based_strategy_metrics(configs: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    rule_configs = _rule_based_eval_configs(configs)
    return _lookup_strategy_metrics(
        rule_configs,
        benchmarks,
        faiss_rule_based_labels(rule_configs),
        "Rule-based",
    )


def _build_objective_strategy_metrics(
    benchmarks: pd.DataFrame,
    configs: pd.DataFrame,
    artifacts_dir: Path,
    run_id: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    has_model = _has_run_artifacts(artifacts_dir, run_id)
    if not has_model:
        print(f"Skipping trained selector plots: missing artifacts for run {run_id}.")

    for objective, oracle_col in OBJECTIVES:
        oracle_labels = configs[oracle_col].astype(str).tolist()
        objective_frames = [
            _lookup_strategy_metrics(configs, benchmarks, oracle_labels, "Oracle winner"),
            _rule_based_strategy_metrics(configs, benchmarks),
            _lookup_strategy_metrics(configs, benchmarks, ["HNSW"] * len(configs), "Always HNSW"),
            _uniform_random_expected_metrics(configs, benchmarks),
        ]
        if has_model:
            objective_frames.insert(
                1,
                _lookup_strategy_metrics(
                    configs,
                    benchmarks,
                    _model_predictions(configs, artifacts_dir, run_id, objective),
                    "Trained selector",
                ),
            )
        obj_df = pd.concat(objective_frames, ignore_index=True)
        obj_df["objective"] = objective
        frames.append(obj_df)

    return pd.concat(frames, ignore_index=True)


def _config_benchmark_group(benchmarks: pd.DataFrame, cfg_row: pd.Series) -> pd.DataFrame:
    mask = pd.Series(True, index=benchmarks.index)
    for col in CONFIG_COLS:
        mask &= benchmarks[col] == cfg_row[col]
    return benchmarks[mask]


def _build_composite_weight_sweep(
    benchmarks: pd.DataFrame,
    configs: pd.DataFrame,
    artifacts_dir: Path,
    run_id: str,
    *,
    weight_grid_step: float = 0.25,
) -> pd.DataFrame:
    """Evaluate oracle/model choices across weight triples that sum to 1."""
    weight_grid = generate_weight_grid(weight_grid_step)
    has_model = _has_run_artifacts(artifacts_dir, run_id)
    models = load_artifacts(_run_artifact_dir(artifacts_dir, run_id)) if has_model else None
    rows = []

    for _, weight_row in weight_grid.iterrows():
        weights = tuple(float(weight_row[col]) for col in WEIGHT_COLS)
        for dataset, dataset_configs in configs.groupby("dataset", sort=True):
            oracle_labels = []
            oracle_scores = []
            for _, cfg_row in dataset_configs.iterrows():
                group = _config_benchmark_group(benchmarks, cfg_row)
                label = select_winner_for_weights(group, weights)
                oracle_labels.append(label)
                score_row = composite_scores(group, weights)
                oracle_scores.append(float(score_row[score_row["index_type"] == label]["composite_score"].iloc[0]))

            model_labels = (
                _composite_model_predictions(dataset_configs, models, weights)
                if models is not None
                else [None] * len(dataset_configs)
            )
            match_rate = (
                float(np.mean([p == o for p, o in zip(model_labels, oracle_labels)]))
                if models is not None
                else np.nan
            )
            counts = pd.Series(oracle_labels).value_counts(normalize=True)
            model_counts = (
                pd.Series(model_labels).value_counts(normalize=True)
                if models is not None
                else pd.Series(dtype=float)
            )
            for index_type in INDEX_ORDER:
                rows.append({
                    "dataset": str(dataset),
                    **{col: float(weight_row[col]) for col in WEIGHT_COLS},
                    "index_type": index_type,
                    "oracle_share": float(counts.get(index_type, 0.0)),
                    "model_share": float(model_counts.get(index_type, np.nan)),
                    "model_match_rate": match_rate,
                    "oracle_mean_composite_score": float(np.mean(oracle_scores)),
                })

    return pd.DataFrame(rows)


def _weight_label(weight_row: pd.Series) -> str:
    return (
        f"R {float(weight_row['w_recall']):.2f}\n"
        f"L {float(weight_row['w_latency']):.2f}\n"
        f"M {float(weight_row['w_memory']):.2f}"
    )


def _mean_strategy_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {metric: float(frame[metric].mean()) for metric in METRIC_COLS}


def _build_composite_weight_metric_sweep(
    benchmarks: pd.DataFrame,
    configs: pd.DataFrame,
    artifacts_dir: Path,
    run_id: str,
    *,
    weight_grid_step: float = 0.25,
) -> pd.DataFrame:
    """Mean recall, memory, and latency outcomes for each composite weight split."""
    weight_grid = generate_weight_grid(weight_grid_step)
    has_model = _has_run_artifacts(artifacts_dir, run_id)
    models = load_artifacts(_run_artifact_dir(artifacts_dir, run_id)) if has_model else None
    rows = []

    for _, weight_row in weight_grid.iterrows():
        weights = tuple(float(weight_row[col]) for col in WEIGHT_COLS)
        for dataset, dataset_configs in configs.groupby("dataset", sort=True):
            oracle_labels = [
                select_winner_for_weights(_config_benchmark_group(benchmarks, cfg_row), weights)
                for _, cfg_row in dataset_configs.iterrows()
            ]
            strategy_specs = [
                ("Oracle", oracle_labels),
            ]
            if models is not None:
                strategy_specs.append(
                    ("Model", _composite_model_predictions(dataset_configs, models, weights))
                )
            rule_configs = _rule_based_eval_configs_for_weights(dataset_configs, weights)
            strategy_specs.append(("Rule-based", faiss_rule_based_labels(rule_configs)))

            for strategy, labels in strategy_specs:
                lookup_configs = rule_configs if strategy == "Rule-based" else dataset_configs
                means = _mean_strategy_metrics(
                    _lookup_strategy_metrics(lookup_configs, benchmarks, labels, strategy)
                )
                rows.append({
                    "dataset": str(dataset),
                    "strategy": strategy,
                    "weight_label": _weight_label(weight_row),
                    **{col: float(weight_row[col]) for col in WEIGHT_COLS},
                    **means,
                })

    return pd.DataFrame(rows)


def _constraint_grid_for_configs(configs: pd.DataFrame) -> pd.DataFrame:
    return expand_constraint_grid(
        configs,
        memory_budget_ratios=MEMORY_BUDGET_RATIOS,
        recall_targets=RECALL_TARGETS,
    )


def _build_constrained_objective_sweep(
    benchmarks: pd.DataFrame,
    configs: pd.DataFrame,
    artifacts_dir: Path,
    run_id: str,
) -> pd.DataFrame:
    """Evaluate oracle/model/rule choices across memory budgets and recall targets."""
    constraint_configs = _constraint_grid_for_configs(configs)
    has_model = _has_run_artifacts(artifacts_dir, run_id)
    models = load_artifacts(_run_artifact_dir(artifacts_dir, run_id)) if has_model else None
    rows = []

    for (dataset, ratio, target), dataset_configs in constraint_configs.groupby(
        ["dataset", "memory_budget_ratio", "recall_target"],
        sort=True,
    ):
        oracle_labels = []
        oracle_scores = []
        for _, cfg_row in dataset_configs.iterrows():
            group = _config_benchmark_group(benchmarks, cfg_row)
            label = select_winner_for_constraints(
                group,
                float(cfg_row["memory_budget_mb"]),
                float(cfg_row["recall_target"]),
            )
            oracle_labels.append(label)
            scored = constraint_scores(
                group,
                float(cfg_row["memory_budget_mb"]),
                float(cfg_row["recall_target"]),
            )
            oracle_scores.append(float(scored[scored["index_type"] == label]["constraint_score"].iloc[0]))

        model_labels = (
            _constrained_model_predictions(dataset_configs, models)
            if models is not None
            else [None] * len(dataset_configs)
        )
        rule_labels = faiss_rule_based_labels(dataset_configs)
        match_rate = (
            float(np.mean([p == o for p, o in zip(model_labels, oracle_labels)]))
            if models is not None
            else np.nan
        )
        oracle_counts = pd.Series(oracle_labels).value_counts(normalize=True)
        model_counts = (
            pd.Series(model_labels).value_counts(normalize=True)
            if models is not None
            else pd.Series(dtype=float)
        )
        rule_counts = pd.Series(rule_labels).value_counts(normalize=True)
        for index_type in INDEX_ORDER:
            rows.append({
                "dataset": str(dataset),
                "memory_budget_ratio": float(ratio),
                "recall_target": float(target),
                "index_type": index_type,
                "oracle_share": float(oracle_counts.get(index_type, 0.0)),
                "model_share": float(model_counts.get(index_type, np.nan)),
                "rule_share": float(rule_counts.get(index_type, 0.0)),
                "model_match_rate": match_rate,
                "oracle_mean_constraint_score": float(np.mean(oracle_scores)),
            })

    return pd.DataFrame(rows)


def _constraint_metrics_for_labels(
    configs: pd.DataFrame,
    benchmarks: pd.DataFrame,
    labels: list[str],
    strategy: str,
) -> pd.DataFrame:
    """Mean measured and constrained-objective outcomes by dataset/constraint."""
    lookup = _lookup_strategy_metrics(configs, benchmarks, labels, strategy)
    rows = []
    for (_, cfg_row), label in zip(configs.iterrows(), labels):
        group = _config_benchmark_group(benchmarks, cfg_row)
        scored = constraint_scores(
            group,
            float(cfg_row["memory_budget_mb"]),
            float(cfg_row["recall_target"]),
        )
        selected = scored[scored["index_type"] == label]
        if selected.empty:
            rows.append({
                "constraint_score": np.nan,
                "memory_budget_satisfied": np.nan,
                "recall_target_satisfied": np.nan,
                "constraints_satisfied": np.nan,
            })
            continue
        row = selected.iloc[0]
        memory_ok = float(row["memory_overrun"]) <= 0.0
        recall_ok = float(row["recall_shortfall"]) <= 0.0
        rows.append({
            "constraint_score": float(row["constraint_score"]),
            "memory_budget_satisfied": float(memory_ok),
            "recall_target_satisfied": float(recall_ok),
            "constraints_satisfied": float(memory_ok and recall_ok),
        })

    detail = pd.concat(
        [
            lookup.reset_index(drop=True),
            configs[CONSTRAINT_COLS].reset_index(drop=True),
            pd.DataFrame(rows),
        ],
        axis=1,
    )
    return (
        detail.groupby(["dataset", "memory_budget_ratio", "recall_target", "strategy"], as_index=False)[
            METRIC_COLS
            + [
                "memory_budget_mb",
                "constraint_score",
                "memory_budget_satisfied",
                "recall_target_satisfied",
                "constraints_satisfied",
            ]
        ]
        .mean()
    )


def _build_constrained_objective_metric_sweep(
    benchmarks: pd.DataFrame,
    configs: pd.DataFrame,
    artifacts_dir: Path,
    run_id: str,
) -> pd.DataFrame:
    """Mean outcomes for each memory-budget/recall-target constraint pair."""
    constraint_configs = _constraint_grid_for_configs(configs)
    has_model = _has_run_artifacts(artifacts_dir, run_id)
    models = load_artifacts(_run_artifact_dir(artifacts_dir, run_id)) if has_model else None

    oracle_labels = [
        select_winner_for_constraints(
            _config_benchmark_group(benchmarks, cfg_row),
            float(cfg_row["memory_budget_mb"]),
            float(cfg_row["recall_target"]),
        )
        for _, cfg_row in constraint_configs.iterrows()
    ]
    frames = [
        _constraint_metrics_for_labels(constraint_configs, benchmarks, oracle_labels, "Oracle"),
        _constraint_metrics_for_labels(
            constraint_configs,
            benchmarks,
            faiss_rule_based_labels(constraint_configs),
            "Rule-based",
        ),
        _constraint_metrics_for_labels(
            constraint_configs,
            benchmarks,
            ["HNSW"] * len(constraint_configs),
            "Always HNSW",
        ),
    ]
    if models is not None:
        frames.insert(
            1,
            _constraint_metrics_for_labels(
                constraint_configs,
                benchmarks,
                _constrained_model_predictions(constraint_configs, models),
                "Model",
            ),
        )
    return pd.concat(frames, ignore_index=True)


def _build_composite_baseline_metric_table(
    benchmarks: pd.DataFrame,
    configs: pd.DataFrame,
) -> pd.DataFrame:
    """Mean measured outcomes for non-weighted baseline strategies by dataset."""
    frames = [
        _rule_based_strategy_metrics(configs, benchmarks),
        _lookup_strategy_metrics(configs, benchmarks, ["HNSW"] * len(configs), "Always HNSW"),
        _uniform_random_expected_metrics(configs, benchmarks),
    ]
    rows = []
    for frame in frames:
        for (dataset, strategy), group in frame.groupby(["dataset", "strategy"], sort=True):
            rows.append({
                "dataset": str(dataset),
                "strategy": str(strategy),
                **_mean_strategy_metrics(group),
            })
    return pd.DataFrame(rows)


def _build_baseline_choice_distribution(configs: pd.DataFrame) -> pd.DataFrame:
    """Index choice shares for non-weighted baseline strategies by dataset."""
    rows = []
    for dataset, dataset_configs in configs.groupby("dataset", sort=True):
        rule_configs = _rule_based_eval_configs(dataset_configs)
        strategy_labels = {
            "Rule-based": faiss_rule_based_labels(rule_configs),
            "Always HNSW": ["HNSW"] * len(dataset_configs),
        }
        for strategy, labels in strategy_labels.items():
            counts = pd.Series(labels).value_counts(normalize=True)
            for index_type in INDEX_ORDER:
                rows.append({
                    "dataset": str(dataset),
                    "strategy": strategy,
                    "index_type": index_type,
                    "share": float(counts.get(index_type, 0.0)),
                })

        for index_type in INDEX_ORDER:
            rows.append({
                "dataset": str(dataset),
                "strategy": "Uniform random",
                "index_type": index_type,
                "share": 1.0 / len(INDEX_ORDER),
            })
    return pd.DataFrame(rows)


def _plot_metric_medians(raw_df: pd.DataFrame, out_path: Path) -> None:
    grouped = (
        raw_df.groupby(["dataset", "index_type"])[METRIC_COLS]
        .median()
        .reset_index()
    )
    datasets = sorted(grouped["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.24
    specs = [
        ("mean_latency_ms", "Median latency (ms)"),
        ("recall_at_k", "Median recall@k"),
        ("index_size_mb", "Median index size (MB)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), squeeze=False)
    for ax, (metric, title) in zip(axes[0], specs):
        for i, index_type in enumerate(INDEX_ORDER):
            vals = []
            for ds in datasets:
                row = grouped[(grouped["dataset"] == ds) & (grouped["index_type"] == index_type)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(
                x + (i - 1) * width,
                vals,
                width=width,
                label=index_type,
                color=INDEX_COLORS[index_type],
                edgecolor="black",
                linewidth=0.3,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
    axes[0][0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_oracle_distribution(configs: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(configs["dataset"].unique())
    fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(5 * len(OBJECTIVES), 4.5), squeeze=False)
    for ax, (objective, oracle_col) in zip(axes[0], OBJECTIVES):
        share = (
            configs.groupby(["dataset", oracle_col]).size().unstack(fill_value=0)
            .reindex(index=datasets, columns=INDEX_ORDER, fill_value=0)
        )
        share = share.div(share.sum(axis=1), axis=0)
        share.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[INDEX_COLORS[idx] for idx in INDEX_ORDER],
            edgecolor="black",
            linewidth=0.3,
            legend=False,
        )
        ax.set_title(f"{objective} oracle")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Oracle share")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    handles = [Patch(facecolor=INDEX_COLORS[k], edgecolor="black", label=k) for k in INDEX_ORDER]
    fig.legend(handles=handles, title="Index type", loc="outside upper right")
    fig.suptitle("Oracle index distribution by dataset and objective")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_model_distribution(configs: pd.DataFrame, artifacts_dir: Path, run_id: str, out_path: Path) -> bool:
    if not _has_run_artifacts(artifacts_dir, run_id):
        return False
    work = configs[CONFIG_COLS].copy()
    for objective, _ in OBJECTIVES:
        work[f"model_{objective}"] = _model_predictions(configs, artifacts_dir, run_id, objective)

    datasets = sorted(work["dataset"].unique())
    fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(5 * len(OBJECTIVES), 4.5), squeeze=False)
    for ax, (objective, _) in zip(axes[0], OBJECTIVES):
        col = f"model_{objective}"
        share = (
            work.groupby(["dataset", col]).size().unstack(fill_value=0)
            .reindex(index=datasets, columns=INDEX_ORDER, fill_value=0)
        )
        share = share.div(share.sum(axis=1), axis=0)
        share.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[INDEX_COLORS[idx] for idx in INDEX_ORDER],
            edgecolor="black",
            linewidth=0.3,
            legend=False,
        )
        ax.set_title(f"{objective} model")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Prediction share")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    handles = [Patch(facecolor=INDEX_COLORS[k], edgecolor="black", label=k) for k in INDEX_ORDER]
    fig.legend(handles=handles, title="Index type", loc="outside upper right")
    fig.suptitle("Model index distribution by dataset and objective")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_model_match_rate(
    configs: pd.DataFrame,
    artifacts_dir: Path,
    run_id: str,
    out_path: Path,
    eval_label: str,
) -> bool:
    if not _has_run_artifacts(artifacts_dir, run_id):
        return False
    rows = []
    for objective, oracle_col in OBJECTIVES:
        preds = _model_predictions(configs, artifacts_dir, run_id, objective)
        work = configs[["dataset", oracle_col]].copy()
        work["match"] = (pd.Series(preds, dtype=object).values == work[oracle_col].values).astype(float)
        for dataset, group in work.groupby("dataset"):
            rows.append({"dataset": dataset, "objective": objective, "match_rate": float(group["match"].mean())})
    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index="dataset", columns="objective", values="match_rate")
    objectives = [obj for obj, _ in OBJECTIVES]
    pivot = pivot.reindex(columns=objectives)

    ax = pivot.plot(kind="bar", figsize=(9, 5), edgecolor="black", linewidth=0.3)
    ax.set_title(f"Model match rate vs oracle by dataset and objective ({eval_label})")
    ax.set_ylabel("Match rate")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Objective")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _plot_choice_distribution_comparison(
    configs: pd.DataFrame,
    artifacts_dir: Path,
    run_id: str,
    out_path: Path,
) -> Path:
    """Single figure comparing oracle, model, and random choices by objective."""
    has_model = _has_run_artifacts(artifacts_dir, run_id)
    rng = np.random.default_rng(RANDOM_SEED)
    work = configs[CONFIG_COLS].copy()

    choice_specs: list[tuple[str, Objective, str]] = []
    for objective, oracle_col in OBJECTIVES:
        oracle_choice_col = f"oracle_{objective}_choice"
        random_choice_col = f"random_{objective}_choice"
        work[oracle_choice_col] = configs[oracle_col].astype(str).to_numpy()
        work[random_choice_col] = rng.choice(INDEX_ORDER, size=len(configs)).tolist()
        choice_specs.append(("Oracle", objective, oracle_choice_col))
        if has_model:
            model_choice_col = f"model_{objective}_choice"
            work[model_choice_col] = _model_predictions(configs, artifacts_dir, run_id, objective)
            choice_specs.append(("Model", objective, model_choice_col))
        choice_specs.append(("Random", objective, random_choice_col))

    datasets = sorted(work["dataset"].unique())
    n_cols = 3 if has_model else 2
    fig, axes = plt.subplots(
        len(OBJECTIVES),
        n_cols,
        figsize=(4.8 * n_cols, 3.6 * len(OBJECTIVES)),
        squeeze=False,
    )

    for row_i, (objective, _) in enumerate(OBJECTIVES):
        row_specs = [spec for spec in choice_specs if spec[1] == objective]
        for col_i, (strategy, _, choice_col) in enumerate(row_specs):
            ax = axes[row_i][col_i]
            share = (
                work.groupby(["dataset", choice_col]).size().unstack(fill_value=0)
                .reindex(index=datasets, columns=INDEX_ORDER, fill_value=0)
            )
            share = share.div(share.sum(axis=1), axis=0)
            share.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=[INDEX_COLORS[idx] for idx in INDEX_ORDER],
                edgecolor="black",
                linewidth=0.3,
                legend=False,
            )
            if row_i == 0:
                ax.set_title(strategy)
            ax.set_ylabel(f"{objective}\nchoice share")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.2)

    handles = [Patch(facecolor=INDEX_COLORS[k], edgecolor="black", label=k) for k in INDEX_ORDER]
    fig.legend(handles=handles, title="Index type", loc="outside upper right")
    fig.suptitle("Oracle vs model vs random index choices by objective and dataset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_strategy_metric_by_objective(strategy_metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    labels = {"memory": "Mean index size (MB)", "recall": "Mean recall@k", "latency": "Mean latency (ms)"}
    for objective, _ in OBJECTIVES:
        metric = OBJECTIVE_METRIC[objective]
        sub = strategy_metrics[strategy_metrics["objective"] == objective]
        summary = (
            sub.groupby(["dataset", "strategy"], as_index=False)[metric]
            .mean()
            .pivot(index="dataset", columns="strategy", values=metric)
        )
        strategies = [strategy for strategy in STRATEGY_ORDER if strategy in summary.columns]
        summary = summary.reindex(columns=strategies)
        datasets = sorted(summary.index.tolist())
        x = np.arange(len(datasets))
        width = min(0.18, 0.82 / max(len(strategies), 1))
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, strategy in enumerate(strategies):
            vals = [float(summary.loc[dataset, strategy]) for dataset in datasets]
            ax.bar(
                x + (i - (len(strategies) - 1) / 2) * width,
                vals,
                width=width,
                label=strategy,
                color=STRATEGY_COLORS[strategy],
                edgecolor="black",
                linewidth=0.3,
            )
        ax.set_title(f"Strategy comparison for {objective} objective")
        ax.set_ylabel(labels[objective])
        ax.set_xlabel("Dataset")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        path = output_dir / f"strategy_{objective}_objective_by_dataset.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_strategy_metric_grid(strategy_metrics: pd.DataFrame, out_path: Path) -> Path:
    """3x3 grid: objective used for selection vs measured metric outcome."""
    metric_specs = [
        ("index_size_mb", "Index size (MB)"),
        ("recall_at_k", "Recall@k"),
        ("mean_latency_ms", "Latency (ms)"),
    ]
    objective_names = [objective for objective, _ in OBJECTIVES]
    strategies = [strategy for strategy in STRATEGY_ORDER if strategy in set(strategy_metrics["strategy"])]
    x = np.arange(len(strategies))

    fig, axes = plt.subplots(
        len(objective_names),
        len(metric_specs),
        figsize=(4.8 * len(metric_specs), 3.6 * len(objective_names)),
        squeeze=False,
    )

    for row_i, objective in enumerate(objective_names):
        sub = strategy_metrics[strategy_metrics["objective"] == objective]
        for col_i, (metric, metric_label) in enumerate(metric_specs):
            ax = axes[row_i][col_i]
            vals = [
                float(sub[sub["strategy"] == strategy][metric].mean())
                if not sub[sub["strategy"] == strategy].empty
                else np.nan
                for strategy in strategies
            ]
            ax.bar(
                x,
                vals,
                color=[STRATEGY_COLORS[strategy] for strategy in strategies],
                edgecolor="black",
                linewidth=0.3,
            )
            if row_i == 0:
                ax.set_title(metric_label)
            if col_i == 0:
                ax.set_ylabel(f"{objective} objective\nmean value")
            else:
                ax.set_ylabel("Mean value")
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=35, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Measured metric outcomes by objective and strategy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_composite_weight_sweep_distribution(
    sweep: pd.DataFrame,
    baseline_choices: pd.DataFrame,
    out_path: Path,
    title_suffix: str = "",
) -> Path:
    """Grid of per-weight-split bars with model and oracle overlaid."""
    sweep = (
        sweep.groupby(WEIGHT_COLS + ["index_type"], as_index=False)[
            ["oracle_share", "model_share", "model_match_rate", "oracle_mean_composite_score"]
        ]
        .mean()
    )
    splits = (
        sweep[WEIGHT_COLS]
        .drop_duplicates()
        .sort_values(["w_recall", "w_latency", "w_memory"])
        .reset_index(drop=True)
    )
    recall_values = sorted(splits["w_recall"].unique())
    latency_values = sorted(splits["w_latency"].unique())
    n_cols = len(latency_values)
    n_rows = len(recall_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.4 * n_rows), squeeze=False)
    has_model = sweep["model_share"].notna().any()
    bar_share_col = "model_share" if has_model else "oracle_share"
    bar_label = "model" if has_model else "oracle"

    for ax in axes.ravel():
        ax.axis("off")

    x = np.arange(len(INDEX_ORDER))
    for row_i, w_recall in enumerate(recall_values):
        row_splits = splits[np.isclose(splits["w_recall"], w_recall)].reset_index(drop=True)
        for split in row_splits.itertuples(index=False):
            col_i = next(
                i for i, w_latency in enumerate(latency_values)
                if np.isclose(w_latency, split.w_latency)
            )
            ax = axes[row_i][col_i]
            ax.axis("on")
            sub = sweep[
                np.isclose(sweep["w_recall"], split.w_recall)
                & np.isclose(sweep["w_latency"], split.w_latency)
                & np.isclose(sweep["w_memory"], split.w_memory)
            ].copy()
            sub = sub.set_index("index_type").reindex(INDEX_ORDER)
            actual_vals = sub[bar_share_col].astype(float).to_numpy()
            oracle_vals = sub["oracle_share"].astype(float).to_numpy()

            ax.bar(
                x,
                actual_vals,
                width=0.72,
                color=[INDEX_COLORS[index_type] for index_type in INDEX_ORDER],
                alpha=0.55 if has_model else 0.85,
                edgecolor="black",
                linewidth=0.4,
                label=bar_label.title(),
            )
            if has_model:
                ax.bar(
                    x,
                    oracle_vals,
                    width=0.72,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1.2,
                    hatch="//",
                    label="Oracle",
                )

            if sub["model_match_rate"].notna().any():
                match_rate = float(sub["model_match_rate"].iloc[0])
                ax.text(
                    0.98,
                    0.96,
                    f"match={match_rate:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=8,
                )

            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Choice share")
            ax.set_xlabel(
                f"Recall {split.w_recall:.2f} | Latency {split.w_latency:.2f} | Memory {split.w_memory:.2f}"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(INDEX_ORDER, rotation=20, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.2)

    baseline_ax = axes[-1][-1]
    baseline_ax.axis("on")
    baseline_strategies = ["Rule-based", "Always HNSW", "Uniform random"]
    width = min(0.22, 0.82 / len(baseline_strategies))
    for i, strategy in enumerate(baseline_strategies):
        vals = []
        for index_type in INDEX_ORDER:
            row = baseline_choices[
                (baseline_choices["strategy"] == strategy)
                & (baseline_choices["index_type"] == index_type)
            ]
            vals.append(float(row["share"].mean()) if not row.empty else np.nan)
        baseline_ax.bar(
            x + (i - (len(baseline_strategies) - 1) / 2) * width,
            vals,
            width=width,
            color=STRATEGY_COLORS[strategy],
            edgecolor="black",
            linewidth=0.4,
            label=strategy,
        )
    baseline_ax.set_title("Baselines")
    baseline_ax.set_ylabel("Choice share")
    baseline_ax.set_xlabel("Non-weighted strategies")
    baseline_ax.set_xticks(x)
    baseline_ax.set_xticklabels(INDEX_ORDER, rotation=20, ha="right", fontsize=8)
    baseline_ax.set_ylim(0, 1.05)
    baseline_ax.grid(axis="y", alpha=0.2)

    handles = [
        Patch(facecolor=INDEX_COLORS[k], alpha=0.55 if has_model else 0.85, edgecolor="black", label=f"{k} {bar_label}")
        for k in INDEX_ORDER
    ]
    if has_model:
        handles.append(Patch(facecolor="none", edgecolor="black", hatch="//", label="Oracle overlay"))
    handles.extend(
        Patch(facecolor=STRATEGY_COLORS[strategy], edgecolor="black", label=strategy)
        for strategy in baseline_strategies
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4)
    title = "Composite weight sweep by split: model bars with oracle overlay"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    fig.suptitle(title, y=0.975, fontsize=15)
    fig.tight_layout(rect=(0, 0.07, 1, 0.965), h_pad=2.0, w_pad=1.2)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_composite_weight_sweep_oracle_share(sweep: pd.DataFrame, out_path: Path) -> Path:
    """Overlay oracle and model index shares as each individual weight increases."""
    fig, axes = plt.subplots(1, len(WEIGHT_COLS), figsize=(15, 4.5), squeeze=False)
    has_model = sweep["model_share"].notna().any()
    labels = {
        "w_recall": "Recall weight",
        "w_latency": "Latency weight",
        "w_memory": "Memory weight",
    }
    for ax, weight_col in zip(axes[0], WEIGHT_COLS):
        summary = (
            sweep.groupby([weight_col, "index_type"], as_index=False)[["oracle_share", "model_share"]]
            .mean()
            .sort_values(weight_col)
        )
        for index_type in INDEX_ORDER:
            sub = summary[summary["index_type"] == index_type]
            ax.plot(
                sub[weight_col],
                sub["oracle_share"],
                marker="o",
                linestyle="--",
                label=f"{index_type} oracle",
                color=INDEX_COLORS[index_type],
            )
            if has_model:
                ax.plot(
                    sub[weight_col],
                    sub["model_share"],
                    marker="x",
                    linestyle="-",
                    label=f"{index_type} regressor",
                    color=INDEX_COLORS[index_type],
                )
        ax.set_title(labels[weight_col])
        ax.set_xlabel(labels[weight_col])
        ax.set_ylabel("Mean choice share")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
    axes[0][0].legend(loc="best")
    fig.suptitle("Composite choice share by individual weight, oracle overlaid")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_composite_weight_metric_by_dataset(
    metric_sweep: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Grid plots of measured outcomes by composite weight split."""
    metric_specs = [
        ("recall_at_k", "Recall performance", "Mean recall@k", "composite_weight_sweep_recall_by_dataset.png"),
        ("index_size_mb", "Memory footprint", "Mean index size (MB)", "composite_weight_sweep_memory_by_dataset.png"),
        ("mean_latency_ms", "Latency", "Mean latency (ms)", "composite_weight_sweep_latency_by_dataset.png"),
    ]
    datasets = sorted(metric_sweep["dataset"].unique())
    splits = (
        metric_sweep[WEIGHT_COLS + ["weight_label"]]
        .drop_duplicates()
        .sort_values(WEIGHT_COLS)
        .reset_index(drop=True)
    )
    recall_values = sorted(splits["w_recall"].unique())
    latency_values = sorted(splits["w_latency"].unique())
    has_model = "Model" in set(metric_sweep["strategy"])
    bar_strategy = "Model" if has_model else "Oracle"
    cell_strategies = [bar_strategy]
    if "Rule-based" in set(metric_sweep["strategy"]):
        cell_strategies.append("Rule-based")
    n_rows = len(recall_values)
    n_cols = len(latency_values)
    paths = []

    for metric, title, ylabel, filename in metric_specs:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.2 * n_rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")

        x = np.arange(len(datasets))
        for row_i, w_recall in enumerate(recall_values):
            row_splits = splits[np.isclose(splits["w_recall"], w_recall)].reset_index(drop=True)
            for split in row_splits.itertuples(index=False):
                col_i = next(
                    i for i, w_latency in enumerate(latency_values)
                    if np.isclose(w_latency, split.w_latency)
                )
                ax = axes[row_i][col_i]
                ax.axis("on")
                oracle_vals = []
                strategy_vals: dict[str, list[float]] = {strategy: [] for strategy in cell_strategies}
                for dataset in datasets:
                    base_mask = (
                        (metric_sweep["dataset"] == dataset)
                        & np.isclose(metric_sweep["w_recall"], split.w_recall)
                        & np.isclose(metric_sweep["w_latency"], split.w_latency)
                        & np.isclose(metric_sweep["w_memory"], split.w_memory)
                    )
                    for strategy in cell_strategies:
                        actual = metric_sweep[base_mask & (metric_sweep["strategy"] == strategy)]
                        strategy_vals[strategy].append(
                            float(actual[metric].iloc[0]) if not actual.empty else np.nan
                        )
                    oracle = metric_sweep[
                        base_mask & (metric_sweep["strategy"] == "Oracle")
                    ]
                    oracle_vals.append(float(oracle[metric].iloc[0]) if not oracle.empty else np.nan)

                width = min(0.34, 0.72 / max(len(cell_strategies), 1))
                for i, strategy in enumerate(cell_strategies):
                    offset = (i - (len(cell_strategies) - 1) / 2) * width
                    color = (
                        STRATEGY_COLORS["Trained selector"]
                        if strategy == "Model"
                        else STRATEGY_COLORS.get(strategy, STRATEGY_COLORS["Oracle winner"])
                    )
                    ax.bar(
                        x + offset,
                        strategy_vals[strategy],
                        width=width,
                        color=color,
                        alpha=0.65,
                        edgecolor="black",
                        linewidth=0.4,
                        label=strategy,
                    )
                if has_model:
                    model_i = cell_strategies.index("Model")
                    model_offset = (model_i - (len(cell_strategies) - 1) / 2) * width
                    ax.bar(
                        x + model_offset,
                        oracle_vals,
                        width=width,
                        facecolor="none",
                        edgecolor="black",
                        linewidth=1.0,
                        hatch="//",
                        label="Oracle",
                    )
                ax.set_ylabel(ylabel)
                ax.set_xlabel(
                    f"R {split.w_recall:.2f} | L {split.w_latency:.2f} | M {split.w_memory:.2f}"
                )
                ax.set_xticks(x)
                ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=8)
                ax.grid(axis="y", alpha=0.2)

        baseline_ax = axes[-1][-1]
        baseline_ax.axis("on")
        baseline_strategies = ["Always HNSW", "Uniform random"]
        width = min(0.22, 0.82 / len(baseline_strategies))
        for i, strategy in enumerate(baseline_strategies):
            vals = []
            for dataset in datasets:
                row = baseline_metrics[
                    (baseline_metrics["dataset"] == dataset)
                    & (baseline_metrics["strategy"] == strategy)
                ]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            baseline_ax.bar(
                x + (i - (len(baseline_strategies) - 1) / 2) * width,
                vals,
                width=width,
                color=STRATEGY_COLORS[strategy],
                edgecolor="black",
                linewidth=0.4,
                label=strategy,
            )
        baseline_ax.set_title("Baselines")
        baseline_ax.set_ylabel(ylabel)
        baseline_ax.set_xlabel("Non-weighted strategies")
        baseline_ax.set_xticks(x)
        baseline_ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=8)
        baseline_ax.grid(axis="y", alpha=0.2)

        handles = [
            Patch(
                facecolor=STRATEGY_COLORS["Trained selector"] if has_model else STRATEGY_COLORS["Oracle winner"],
                alpha=0.65,
                edgecolor="black",
                label=bar_strategy,
            )
        ]
        if "Rule-based" in cell_strategies:
            handles.append(
                Patch(
                    facecolor=STRATEGY_COLORS["Rule-based"],
                    alpha=0.65,
                    edgecolor="black",
                    label="Rule-based aligned to weights",
                )
            )
        if has_model:
            handles.append(Patch(facecolor="none", edgecolor="black", hatch="//", label="Oracle overlay"))
        handles.extend(
            Patch(facecolor=STRATEGY_COLORS[strategy], edgecolor="black", label=strategy)
            for strategy in baseline_strategies
            if strategy != "Rule-based"
        )
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=3)
        fig.suptitle(f"Composite weight sweep {title.lower()} by dataset", y=0.975, fontsize=15)
        fig.tight_layout(rect=(0, 0.07, 1, 0.965), h_pad=2.0, w_pad=1.2)
        path = output_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def _plot_constrained_objective_distribution(
    sweep: pd.DataFrame,
    out_path: Path,
    title_suffix: str = "",
) -> Path:
    """Grid of choice shares by recall target and memory-budget ratio."""
    sweep = (
        sweep.groupby(["recall_target", "memory_budget_ratio", "index_type"], as_index=False)[
            ["oracle_share", "model_share", "rule_share", "model_match_rate"]
        ]
        .mean()
    )
    recall_values = sorted(sweep["recall_target"].unique())
    budget_values = sorted(sweep["memory_budget_ratio"].unique())
    n_rows = len(recall_values)
    n_cols = len(budget_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=_constrained_grid_figsize(n_cols, n_rows, col_inch=4.6),
        squeeze=False,
    )
    has_model = sweep["model_share"].notna().any()
    strategies = [("Regressor policy", "model_share", STRATEGY_COLORS["Trained selector"])] if has_model else []
    strategies.append(("FAISS rule-based", "rule_share", STRATEGY_COLORS["Rule-based"]))

    x = np.arange(len(INDEX_ORDER))
    width = min(0.32, 0.76 / max(len(strategies), 1))
    for row_i, target in enumerate(recall_values):
        for col_i, ratio in enumerate(budget_values):
            ax = axes[row_i][col_i]
            sub = sweep[
                np.isclose(sweep["recall_target"], target)
                & np.isclose(sweep["memory_budget_ratio"], ratio)
            ].copy()
            sub = sub.set_index("index_type").reindex(INDEX_ORDER)

            for i, (label, col, color) in enumerate(strategies):
                offset = (i - (len(strategies) - 1) / 2) * width
                ax.bar(
                    x + offset,
                    sub[col].astype(float).to_numpy(),
                    width=width,
                    color=color,
                    alpha=0.72,
                    edgecolor="black",
                    linewidth=0.4,
                    label=label,
                )
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Choice share")
            ax.set_xlabel(f"budget x{ratio:g}")
            ax.set_title(f"recall target {target:.2f}")
            ax.set_xticks(x)
            ax.set_xticklabels(
                INDEX_ORDER,
                rotation=20,
                ha="right",
                fontsize=7 if n_cols >= 6 else 8,
            )
            ax.grid(axis="y", alpha=0.2)

    handles = [
        Patch(facecolor=color, alpha=0.72, edgecolor="black", label=label)
        for label, _, color in strategies
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(len(handles), 3),
    )
    title = "Constrained objective choice share by recall target and memory budget"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    fig.suptitle(title, y=0.975, fontsize=15)
    fig.tight_layout(rect=(0, 0.07, 1, 0.965), h_pad=2.0, w_pad=1.2)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_constrained_objective_oracle_share(sweep: pd.DataFrame, out_path: Path) -> Path:
    """Model and rule-based index shares across each constrained input dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    has_model = sweep["model_share"].notna().any()
    specs = [
        ("recall_target", "Recall target"),
        ("memory_budget_ratio", "Memory budget ratio"),
    ]
    for ax, (col, label) in zip(axes[0], specs):
        summary = (
            sweep.groupby([col, "index_type"], as_index=False)[["model_share", "rule_share"]]
            .mean()
            .sort_values(col)
        )
        for index_type in INDEX_ORDER:
            sub = summary[summary["index_type"] == index_type]
            if has_model:
                ax.plot(
                    sub[col],
                    sub["model_share"],
                    marker="x",
                    linestyle={"IVF_FLAT": "-", "IVF_PQ": "--", "HNSW": ":"}[index_type],
                    label=f"{index_type} model",
                    color=STRATEGY_COLORS["Trained selector"],
                )
            ax.plot(
                sub[col],
                sub["rule_share"],
                marker="D",
                linestyle={"IVF_FLAT": "-", "IVF_PQ": "--", "HNSW": ":"}[index_type],
                label=f"{index_type} rule",
                color=STRATEGY_COLORS["Rule-based"],
                alpha=0.85,
            )
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Mean choice share")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
    axes[0][0].legend(loc="best", fontsize=7)
    fig.suptitle("Constrained choice share by objective input")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_constrained_objective_metric_by_dataset(
    metric_sweep: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Grid plots of measured and penalty outcomes by constrained objective inputs."""
    metric_specs = [
        ("recall_at_k", "Recall performance", "Mean recall@k", "constrained_objective_recall_by_dataset.png"),
        ("index_size_mb", "Memory footprint", "Mean index size (MB)", "constrained_objective_memory_by_dataset.png"),
        ("mean_latency_ms", "Latency", "Mean latency (ms)", "constrained_objective_latency_by_dataset.png"),
        ("constraint_score", "Penalty objective", "Mean constraint score", "constrained_objective_score_by_dataset.png"),
        (
            "constraints_satisfied",
            "Constraint satisfaction",
            "Fraction satisfying budget and recall",
            "constrained_objective_satisfaction_by_dataset.png",
        ),
    ]
    datasets = sorted(metric_sweep["dataset"].unique())
    recall_values = sorted(metric_sweep["recall_target"].unique())
    budget_values = sorted(metric_sweep["memory_budget_ratio"].unique())
    strategies = [s for s in ["Model", "Rule-based"] if s in set(metric_sweep["strategy"])]
    strategy_labels = {"Model": "Regressor policy", "Rule-based": "FAISS rule-based"}
    paths = []

    for metric, title, ylabel, filename in metric_specs:
        fig, axes = plt.subplots(
            len(recall_values),
            len(budget_values),
            figsize=_constrained_grid_figsize(len(budget_values), len(recall_values), col_inch=4.7),
            squeeze=False,
        )
        x = np.arange(len(datasets))
        width = min(0.18, 0.72 / max(len(strategies), 1))
        for row_i, target in enumerate(recall_values):
            for col_i, ratio in enumerate(budget_values):
                ax = axes[row_i][col_i]
                base_mask = (
                    np.isclose(metric_sweep["recall_target"], target)
                    & np.isclose(metric_sweep["memory_budget_ratio"], ratio)
                )
                for i, strategy in enumerate(strategies):
                    vals = []
                    for dataset in datasets:
                        row = metric_sweep[
                            base_mask
                            & (metric_sweep["dataset"] == dataset)
                            & (metric_sweep["strategy"] == strategy)
                        ]
                        vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
                    color = (
                        STRATEGY_COLORS["Trained selector"]
                        if strategy == "Model"
                        else STRATEGY_COLORS.get(strategy, STRATEGY_COLORS["Oracle winner"])
                    )
                    ax.bar(
                        x + (i - (len(strategies) - 1) / 2) * width,
                        vals,
                        width=width,
                        color=color,
                        alpha=0.68,
                        edgecolor="black",
                        linewidth=0.4,
                        label=strategy_labels.get(strategy, strategy),
                    )
                if metric == "recall_at_k":
                    ax.axhline(
                        float(target),
                        color="black",
                        linestyle=":",
                        linewidth=1.2,
                    )
                elif metric == "index_size_mb":
                    budget_targets = []
                    for dataset in datasets:
                        row = metric_sweep[
                            base_mask
                            & (metric_sweep["dataset"] == dataset)
                        ]
                        budget_targets.append(
                            float(row["memory_budget_mb"].mean()) if not row.empty else np.nan
                        )
                    for x_pos, budget in zip(x, budget_targets):
                        if np.isfinite(budget):
                            ax.hlines(
                                budget,
                                x_pos - 0.34,
                                x_pos + 0.34,
                                color="black",
                                linestyle=":",
                                linewidth=1.2,
                            )
                ax.set_title(f"target {target:.2f}, budget x{ratio:g}")
                ax.set_ylabel(ylabel)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    datasets,
                    rotation=20,
                    ha="right",
                    fontsize=7 if len(budget_values) >= 6 else 8,
                )
                if metric == "constraints_satisfied":
                    ax.set_ylim(0, 1.05)
                ax.grid(axis="y", alpha=0.2)

        handles = []
        for strategy in strategies:
            color = (
                STRATEGY_COLORS["Trained selector"]
                if strategy == "Model"
                else STRATEGY_COLORS.get(strategy, STRATEGY_COLORS["Oracle winner"])
            )
            handles.append(
                Patch(
                    facecolor=color,
                    alpha=0.68,
                    edgecolor="black",
                    label=strategy_labels.get(strategy, strategy),
                )
            )
        if metric in {"recall_at_k", "index_size_mb"}:
            target_label = "Recall target" if metric == "recall_at_k" else "Memory budget"
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle=":",
                    linewidth=1.2,
                    label=target_label,
                )
            )
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=len(handles))
        fig.suptitle(f"Constrained objective {title.lower()} by dataset", y=0.975, fontsize=15)
        fig.tight_layout(rect=(0, 0.07, 1, 0.965), h_pad=2.0, w_pad=1.2)
        path = output_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def _safe_plot_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _write_summary(
    run_id: str,
    raw_df: pd.DataFrame,
    configs: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    constrained_sweep: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        f"Run id: {run_id}",
        "",
        f"Unique raw rows: {len(raw_df)}",
        f"Unique config rows: {len(configs)}",
        "",
    ]
    for objective, oracle_col in OBJECTIVES:
        lines.append(f"Oracle distribution ({objective}):")
        overall = configs[oracle_col].value_counts(normalize=True)
        for label in INDEX_ORDER:
            lines.append(f"  {label}: {float(overall.get(label, 0.0)):.3f}")
        metric = OBJECTIVE_METRIC[objective]
        sub = strategy_metrics[strategy_metrics["objective"] == objective]
        lines.append(f"Mean optimized metric ({metric}):")
        for strategy in STRATEGY_ORDER:
            ssub = sub[sub["strategy"] == strategy]
            if not ssub.empty:
                lines.append(f"  {strategy}: {float(ssub[metric].mean()):.6f}")
        lines.append("")

    lines.append("Constrained objective oracle distribution:")
    for index_type in INDEX_ORDER:
        sub = constrained_sweep[constrained_sweep["index_type"] == index_type]
        lines.append(f"  {index_type}: {float(sub['oracle_share'].mean()):.3f}")
    if constrained_sweep["model_match_rate"].notna().any():
        matches = constrained_sweep[
            ["memory_budget_ratio", "recall_target", "model_match_rate"]
        ].drop_duplicates()
        lines.append(f"  Model match rate: {float(matches['model_match_rate'].mean()):.3f}")
    lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(
    results_dir: Path,
    run_id: str = "",
    output_dir: Path | None = None,
    artifacts_dir: Path = Path(ARTIFACTS_DIR),
) -> None:
    results_dir = Path(results_dir)
    artifacts_dir = Path(artifacts_dir)
    resolved_run_id, run_dir = resolve_run_dir(results_dir, run_id)
    results_csv = run_dir / "benchmarks.csv"
    if output_dir is None:
        output_dir = run_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = pd.read_csv(results_csv)
    raw_df = _dedupe_raw_rows(benchmarks)
    labeled = _load_labeled(run_dir, benchmarks)
    configs = _config_table(labeled)
    eval_configs, eval_label = _load_plot_eval_tables(
        run_dir,
        configs,
    )
    strategy_metrics = _build_objective_strategy_metrics(
        benchmarks,
        configs,
        artifacts_dir,
        resolved_run_id,
    )
    constrained_sweep = _build_constrained_objective_sweep(
        benchmarks,
        configs,
        artifacts_dir,
        resolved_run_id,
    )
    constrained_metric_sweep = _build_constrained_objective_metric_sweep(
        benchmarks,
        configs,
        artifacts_dir,
        resolved_run_id,
    )
    for legacy_path in output_dir.glob("composite_weight_sweep*.png"):
        legacy_path.unlink()

    outputs: list[Path] = []
    metric_medians_path = output_dir / "metric_medians_by_index.png"
    _plot_metric_medians(raw_df, metric_medians_path)
    outputs.append(metric_medians_path)

    oracle_distribution_path = output_dir / "oracle_index_distribution_by_objective.png"
    _plot_oracle_distribution(configs, oracle_distribution_path)
    outputs.append(oracle_distribution_path)

    model_distribution_path = output_dir / "model_index_distribution_by_objective.png"
    if _plot_model_distribution(configs, artifacts_dir, resolved_run_id, model_distribution_path):
        outputs.append(model_distribution_path)

    match_rate_path = output_dir / "model_oracle_match_rate_by_objective.png"
    if _plot_model_match_rate(
        eval_configs,
        artifacts_dir,
        resolved_run_id,
        match_rate_path,
        eval_label,
    ):
        outputs.append(match_rate_path)

    choice_comparison_path = output_dir / "oracle_model_random_choice_distribution.png"
    outputs.append(
        _plot_choice_distribution_comparison(
            configs,
            artifacts_dir,
            resolved_run_id,
            choice_comparison_path,
        )
    )

    outputs.extend(_plot_strategy_metric_by_objective(strategy_metrics, output_dir))

    strategy_metric_grid_path = output_dir / "strategy_metric_grid_by_objective.png"
    outputs.append(_plot_strategy_metric_grid(strategy_metrics, strategy_metric_grid_path))

    constrained_distribution_path = output_dir / "constrained_objective_distribution_grid.png"
    outputs.append(
        _plot_constrained_objective_distribution(
            constrained_sweep,
            constrained_distribution_path,
            "all benchmarks",
        )
    )
    for dataset in sorted(constrained_sweep["dataset"].unique()):
        dataset_sweep = constrained_sweep[constrained_sweep["dataset"] == dataset].copy()
        dataset_path = output_dir / f"constrained_objective_distribution_grid_{_safe_plot_name(str(dataset))}.png"
        outputs.append(
            _plot_constrained_objective_distribution(
                dataset_sweep,
                dataset_path,
                str(dataset),
            )
        )

    constrained_share_path = output_dir / "constrained_objective_choice_share.png"
    outputs.append(_plot_constrained_objective_oracle_share(constrained_sweep, constrained_share_path))

    outputs.extend(
        _plot_constrained_objective_metric_by_dataset(
            constrained_metric_sweep,
            output_dir,
        )
    )

    summary_path = output_dir / "summary.txt"
    _write_summary(resolved_run_id, raw_df, configs, strategy_metrics, constrained_sweep, summary_path)
    outputs.append(summary_path)

    print(f"Run id: {resolved_run_id}")
    print(f"Wrote plots to {output_dir}")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--artifacts-dir", default=ARTIFACTS_DIR)
    parser.add_argument(
        "--run-id",
        default="",
        help="Run id to plot; defaults to latest run under results/runs/.",
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else None
    main(
        Path(args.results_dir),
        run_id=args.run_id,
        output_dir=out_dir,
        artifacts_dir=Path(args.artifacts_dir),
    )
