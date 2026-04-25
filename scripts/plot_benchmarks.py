"""Generate clean, aggregated plots and summaries for benchmark runs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.labeling import CONFIG_COLS, label_benchmarks
from src.run_store import resolve_run_dir

INDEX_ORDER = ["IVF_FLAT", "IVF_PQ", "HNSW"]
INDEX_COLORS = {
    "IVF_FLAT": "#1f77b4",
    "IVF_PQ": "#ff7f0e",
    "HNSW": "#2ca02c",
}
RAW_ID_COLS = ["dataset", "n_fraction", "N", "d", "k", "index_type"]


def _dedupe_raw_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(RAW_ID_COLS).drop_duplicates(subset=RAW_ID_COLS).copy()


def _load_labeled(run_dir: Path, benchmarks: pd.DataFrame) -> pd.DataFrame:
    labeled_path = run_dir / "labeled.csv"
    if labeled_path.exists():
        return pd.read_csv(labeled_path)
    return label_benchmarks(benchmarks.copy())


def _winner_table(labeled_df: pd.DataFrame) -> pd.DataFrame:
    return labeled_df[CONFIG_COLS + ["label"]].drop_duplicates(subset=CONFIG_COLS)


def _plot_winner_share_by_dataset(winners: pd.DataFrame, out_path: Path) -> None:
    share = (
        winners.groupby(["dataset", "label"]).size().unstack(fill_value=0)
        .reindex(columns=INDEX_ORDER, fill_value=0)
    )
    share = share.div(share.sum(axis=1), axis=0)

    ax = share.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=[INDEX_COLORS[idx] for idx in INDEX_ORDER],
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_ylabel("Winner share")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 1)
    ax.set_title("Winner distribution by dataset")
    ax.legend(title="Winner", loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_constraint_matrix(winners: pd.DataFrame, out_path: Path) -> None:
    datasets = sorted(winners["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    label_to_code = {label: idx for idx, label in enumerate(INDEX_ORDER)}

    for ax, dataset in zip(axes[0], datasets):
        subset = winners[winners["dataset"] == dataset]
        pivot = (
            subset.groupby(["memory_budget_mb", "recall_target", "label"]).size()
            .reset_index(name="count")
            .sort_values(["memory_budget_mb", "recall_target", "count"], ascending=[True, True, False])
            .drop_duplicates(["memory_budget_mb", "recall_target"])
            .pivot(index="memory_budget_mb", columns="recall_target", values="label")
        )

        mem_vals = sorted(pivot.index.tolist())
        rec_vals = sorted(pivot.columns.tolist())
        matrix = np.full((len(mem_vals), len(rec_vals)), np.nan)

        for i, mem in enumerate(mem_vals):
            for j, rec in enumerate(rec_vals):
                label = pivot.loc[mem, rec]
                matrix[i, j] = label_to_code.get(label, np.nan)

        cmap = plt.get_cmap("tab10", len(INDEX_ORDER))
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=len(INDEX_ORDER) - 1)
        ax.set_title(dataset)
        ax.set_xlabel("recall_target")
        ax.set_ylabel("memory_budget_mb")
        ax.set_xticks(range(len(rec_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in rec_vals])
        ax.set_yticks(range(len(mem_vals)))
        ax.set_yticklabels([str(int(v)) for v in mem_vals])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=range(len(INDEX_ORDER)))
    cbar.ax.set_yticklabels(INDEX_ORDER)
    fig.suptitle("Dominant winner by constraint pair")
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.92, wspace=0.25)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_metric_medians(raw_df: pd.DataFrame, out_path: Path) -> None:
    grouped = (
        raw_df.groupby(["dataset", "index_type"])[["mean_latency_ms", "recall_at_k", "index_size_mb"]]
        .median()
        .reset_index()
    )
    datasets = sorted(grouped["dataset"].unique())
    x = np.arange(len(datasets))
    width = 0.24

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), squeeze=False)
    metrics = [
        ("mean_latency_ms", "Median latency (ms)"),
        ("recall_at_k", "Median recall@k"),
        ("index_size_mb", "Median index size (MB)"),
    ]

    for ax, (metric, title) in zip(axes[0], metrics):
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


def _plot_cross_run_winner_share(results_dir: Path, out_path: Path) -> bool:
    runs_root = Path(results_dir) / "runs"
    if not runs_root.exists():
        return False

    rows = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        labeled_path = run_dir / "labeled.csv"
        if not labeled_path.exists():
            continue
        labeled = pd.read_csv(labeled_path)
        winners = _winner_table(labeled)

        meta_path = run_dir / "labeling_meta.json"
        objective = run_dir.name
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mw = meta.get("memory_weight", "?")
            rw = meta.get("recall_weight", "?")
            objective = f"{run_dir.name} (m={mw}, r={rw})"

        total = len(winners)
        for label, cnt in winners["label"].value_counts().items():
            rows.append({
                "run": objective,
                "label": label,
                "share": float(cnt) / float(total),
            })

    if len(rows) <= 0:
        return False

    frame = pd.DataFrame(rows)
    share = frame.pivot(index="run", columns="label", values="share").fillna(0)
    share = share.reindex(columns=INDEX_ORDER, fill_value=0)
    if len(share) <= 1:
        return False

    ax = share.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5),
        color=[INDEX_COLORS[idx] for idx in INDEX_ORDER],
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_ylabel("Winner share")
    ax.set_xlabel("Run / objective")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-run winner distribution")
    ax.legend(title="Winner", loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def _write_summary(run_id: str, raw_df: pd.DataFrame, winners: pd.DataFrame, out_path: Path) -> None:
    lines = [
        f"Run id: {run_id}",
        "",
        f"Unique raw rows: {len(raw_df)}",
        f"Unique config rows: {len(winners)}",
        "",
        "Winner distribution overall:",
    ]

    overall = winners["label"].value_counts(normalize=True)
    for label in INDEX_ORDER:
        frac = float(overall.get(label, 0.0))
        lines.append(f"  {label}: {frac:.3f}")

    lines.append("")
    lines.append("Winner distribution by dataset:")
    for ds, group in winners.groupby("dataset"):
        dist = group["label"].value_counts(normalize=True)
        parts = [f"{label}={float(dist.get(label, 0.0)):.3f}" for label in INDEX_ORDER]
        lines.append(f"  {ds}: " + ", ".join(parts))

    lines.append("")
    lines.append("Note:")
    lines.append("  Plots are aggregated for expanded grids and objective-sweep comparisons.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(results_dir: Path, run_id: str = "", output_dir: Path | None = None) -> None:
    results_dir = Path(results_dir)
    resolved_run_id, run_dir = resolve_run_dir(results_dir, run_id)
    results_csv = run_dir / "benchmarks.csv"
    if output_dir is None:
        output_dir = run_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = pd.read_csv(results_csv)
    raw_df = _dedupe_raw_rows(benchmarks)
    labeled = _load_labeled(run_dir, benchmarks)
    winners = _winner_table(labeled)

    _plot_winner_share_by_dataset(winners, output_dir / "winner_share_by_dataset.png")
    _plot_constraint_matrix(winners, output_dir / "winner_by_constraint_matrix.png")
    _plot_metric_medians(raw_df, output_dir / "metric_medians_by_index.png")
    has_cross_run = _plot_cross_run_winner_share(results_dir, output_dir / "cross_run_winner_share.png")
    _write_summary(resolved_run_id, raw_df, winners, output_dir / "summary.txt")

    print(f"Run id: {resolved_run_id}")
    print(f"Wrote plots to {output_dir}")
    print(f"  - {output_dir / 'winner_share_by_dataset.png'}")
    print(f"  - {output_dir / 'winner_by_constraint_matrix.png'}")
    print(f"  - {output_dir / 'metric_medians_by_index.png'}")
    if has_cross_run:
        print(f"  - {output_dir / 'cross_run_winner_share.png'}")
    print(f"  - {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--run-id",
        default="",
        help="Run id to plot; defaults to latest run under results/runs/.",
    )
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else None
    main(Path(args.results_dir), run_id=args.run_id, output_dir=out_dir)
