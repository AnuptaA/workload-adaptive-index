"""Generate plots and a short text summary for benchmark results."""

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from src.labeling import label_benchmarks

INDEX_ORDER = ["IVF_FLAT", "IVF_PQ", "HNSW"]
INDEX_COLORS = {
    "IVF_FLAT": "#1f77b4",
    "IVF_PQ": "#ff7f0e",
    "HNSW": "#2ca02c",
}
K_MARKERS = {10: "o", 50: "s", 100: "^"}
FRACTION_SIZES = {0.05: 60, 0.10: 110, 0.20: 170}
RAW_ID_COLS = ["dataset", "n_fraction", "N", "d", "k", "index_type"]
CONFIG_COLS = ["dataset", "n_fraction", "N", "d", "k", "memory_budget_mb", "recall_target"]


def _dedupe_raw_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(RAW_ID_COLS).drop_duplicates(subset=RAW_ID_COLS).copy()


def _plot_latency_recall(raw_df: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(raw_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        subset = raw_df[raw_df["dataset"] == dataset]
        for index_type in INDEX_ORDER:
            type_df = subset[subset["index_type"] == index_type]
            if type_df.empty:
                continue

            for k, marker in K_MARKERS.items():
                points = type_df[type_df["k"] == k]
                if points.empty:
                    continue

                sizes = points["n_fraction"].map(FRACTION_SIZES).fillna(80)
                ax.scatter(
                    points["mean_latency_ms"],
                    points["recall_at_k"],
                    s=sizes,
                    marker=marker,
                    c=INDEX_COLORS[index_type],
                    alpha=0.85,
                    edgecolors="black",
                    linewidths=0.4,
                )

        ax.set_title(dataset)
        ax.set_xlabel("Mean latency (ms)")
        ax.set_ylabel("Recall@k")
        ax.grid(alpha=0.2)

    color_handles = [
        Line2D([0], [0], marker="o", color="w", label=index_type,
               markerfacecolor=INDEX_COLORS[index_type], markeredgecolor="black", markersize=8)
        for index_type in INDEX_ORDER
    ]
    marker_handles = [
        Line2D([0], [0], marker=marker, color="black", linestyle="", label=f"k={k}", markersize=7)
        for k, marker in K_MARKERS.items()
    ]
    size_handles = [
        plt.scatter([], [], s=size, c="#999999", alpha=0.6, edgecolors="black", linewidths=0.4,
                    label=f"n_fraction={fraction:g}")
        for fraction, size in FRACTION_SIZES.items()
    ]

    fig.legend(
        handles=color_handles + marker_handles + size_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
    )
    fig.suptitle("Latency vs Recall Tradeoff", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_memory_build(raw_df: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(raw_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        subset = raw_df[raw_df["dataset"] == dataset]
        for index_type in INDEX_ORDER:
            points = subset[subset["index_type"] == index_type]
            if points.empty:
                continue

            ax.scatter(
                points["peak_memory_mb"],
                points["build_time_s"],
                s=90,
                c=INDEX_COLORS[index_type],
                alpha=0.85,
                edgecolors="black",
                linewidths=0.4,
                label=index_type,
            )

        ax.set_title(dataset)
        ax.set_xlabel("Peak memory during build (MB)")
        ax.set_ylabel("Build time (s)")
        ax.grid(alpha=0.2)

    handles = [
        Line2D([0], [0], marker="o", color="w", label=index_type,
               markerfacecolor=INDEX_COLORS[index_type], markeredgecolor="black", markersize=8)
        for index_type in INDEX_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    fig.suptitle("Build Cost Tradeoff", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_constraint_winners(df: pd.DataFrame, out_path: Path) -> None:
    labeled = label_benchmarks(df.copy())
    winners = labeled[CONFIG_COLS + ["label"]].drop_duplicates()
    counts = (
        winners.groupby(["dataset", "memory_budget_mb", "recall_target", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=INDEX_ORDER, fill_value=0)
    )

    datasets = sorted(winners["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.8), squeeze=False, sharey=True)

    for ax, dataset in zip(axes[0], datasets):
        dataset_counts = counts.loc[dataset]
        x_labels = [
            f"{int(mem)}MB\nR={recall:.2f}"
            for mem, recall in dataset_counts.index.tolist()
        ]
        bottom = pd.Series(0, index=dataset_counts.index)

        for index_type in INDEX_ORDER:
            values = dataset_counts[index_type]
            ax.bar(
                range(len(values)),
                values,
                bottom=bottom,
                color=INDEX_COLORS[index_type],
                edgecolor="black",
                linewidth=0.3,
                label=index_type,
            )
            bottom = bottom + values

        ax.set_title(dataset)
        ax.set_xlabel("Constraint pair")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.2)

    axes[0][0].set_ylabel("Winning configurations\n(across n_fraction and k)")
    handles = [
        Line2D([0], [0], marker="s", color="w", label=index_type,
               markerfacecolor=INDEX_COLORS[index_type], markeredgecolor="black", markersize=8)
        for index_type in INDEX_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    fig.suptitle("Which Index Wins Under Each Constraint Pair?", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_summary(df: pd.DataFrame, raw_df: pd.DataFrame, out_path: Path) -> None:
    labeled = label_benchmarks(df.copy())
    winners = labeled[CONFIG_COLS + ["label"]].drop_duplicates()
    winner_counts = (
        winners.groupby(["dataset", "label"]).size().unstack(fill_value=0).reindex(columns=INDEX_ORDER, fill_value=0)
    )

    lines = [
        "How to read results/benchmarks.csv",
        "",
        "1. Each raw ANN measurement is identified by: dataset, n_fraction, k, index_type.",
        "   Those rows are then repeated across memory_budget_mb and recall_target so the labeling step can decide",
        "   which index would win under different deployment constraints.",
        "",
        "2. Raw performance columns:",
        "   - mean_latency_ms / p99_latency_ms: lower is better",
        "   - recall_at_k: higher is better",
        "   - peak_memory_mb / build_time_s: lower is better",
        "",
        "3. Constraint columns:",
        "   - memory_budget_mb: maximum allowed build-time memory footprint",
        "   - recall_target: minimum acceptable recall@k",
        "",
        "4. The key caveat: changing memory_budget_mb or recall_target does not rerun the benchmark.",
        "   It only changes how the same measured rows are judged downstream.",
        "",
        f"Total CSV rows: {len(df)}",
        f"Unique raw measurement rows: {len(raw_df)}",
        "",
        "Winner counts by dataset:",
    ]

    for dataset in winner_counts.index:
        counts = winner_counts.loc[dataset]
        parts = [f"{index_type}={int(counts[index_type])}" for index_type in INDEX_ORDER]
        lines.append(f"  - {dataset}: " + ", ".join(parts))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(results_csv: Path, output_dir: Path) -> None:
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    raw_df = _dedupe_raw_rows(df)

    _plot_latency_recall(raw_df, output_dir / "latency_vs_recall.png")
    _plot_memory_build(raw_df, output_dir / "build_cost_tradeoff.png")
    _plot_constraint_winners(df, output_dir / "constraint_winners.png")
    _write_summary(df, raw_df, output_dir / "summary.txt")

    print(f"Wrote plots to {output_dir}")
    print(f"  - {output_dir / 'latency_vs_recall.png'}")
    print(f"  - {output_dir / 'build_cost_tradeoff.png'}")
    print(f"  - {output_dir / 'constraint_winners.png'}")
    print(f"  - {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", default="results/benchmarks.csv")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()
    main(Path(args.results_csv), Path(args.output_dir))
