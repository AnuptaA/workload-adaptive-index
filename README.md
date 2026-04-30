# workload-adaptive-index

Experiments for workload-adaptive approximate nearest neighbor indexing: benchmarks, labeling, and models over ANN benchmark datasets.

## Prerequisites

- **Python 3.11+** (3.11 is what the project is tested with locally)
- **pip** and a shell (bash or zsh)

Optional: enough disk space and time for dataset downloads (HDF5 files from [ann-benchmarks](http://ann-benchmarks.com/); see `src/config.py` for URLs).

## Setup

### 1. Clone and enter the repository

```bash
git clone <repository-url>
cd workload-adaptive-index
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

Either use the Makefile:

```bash
make install
```

Or pip directly:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the install

```bash
make test
# or: pytest tests/ -v
```

### 5. (Optional) Download data and run the pipeline

```bash
make download    # writes under data/ (gitignored)
make benchmark   # needs data/
make plot        # turns results/benchmarks.csv into charts under results/plots/
make label       # needs results/ from benchmark
```

Smaller sanity check without full datasets:

```bash
make hello
```

## Makefile targets

| Target      | Description                                      |
|------------|---------------------------------------------------|
| `install`  | `pip install -r requirements.txt`                 |
| `download` | `python scripts/download_datasets.py --data-dir data/` |
| `benchmark`| Run benchmark script with default dirs             |
| `plot`     | Generate benchmark plots and a short summary under `results/plots/` |
| `label`    | Label benchmark outputs                          |
| `hello`    | Quick script smoke test                          |
| `test`     | Run pytest on `tests/`                           |
| `clean`    | Remove `__pycache__` and `.pytest_cache`        |
| `all`      | `download` then `benchmark` then `label`         |

## Interpreting benchmark outputs

`results/benchmarks.csv` stores raw ANN measurements for each `(dataset, n_fraction, k, index_type)`:
`mean_latency_ms`, `p99_latency_ms`, `recall_at_k`, `index_size_mb`, and `build_time_s`.

That means:

- Compare `mean_latency_ms` vs `recall_at_k` to understand the raw search tradeoff
- Compare `index_size_mb` and `build_time_s` to understand deployment footprint vs build cost
- Use `make label` / `make train` to derive objective-specific selectors:
  memory minimizes `index_size_mb`, recall maximizes `recall_at_k`, latency minimizes `mean_latency_ms`, and the constrained policy predicts per-index latency, memory, and recall before applying the deployment objective

`make plot` generates objective-aware views such as:

- `metric_medians_by_index.png`: raw metric medians for each index type
- `oracle_index_distribution_by_objective.png`: oracle choices by dataset and objective
- `strategy_<objective>_objective_by_dataset.png`: oracle/model/baseline metric comparison for each objective
- `strategy_metric_grid_by_objective.png`: all measured metrics compared across selection objectives
- `constrained_objective_distribution_grid.png`: model vs FAISS rule choices over recall targets and memory budgets
- `constrained_objective_<metric>_by_dataset.png`: measured outcomes for the constrained policy and FAISS rule baseline

## Constrained Objective

The constrained policy trains three performance regressors over workload features plus candidate index type:

```text
dataset, n_fraction, N, d, k, index_type -> mean_latency_ms
dataset, n_fraction, N, d, k, index_type -> index_size_mb
dataset, n_fraction, N, d, k, index_type -> recall_at_k
```

The exact memory budget used for oracle labeling is:

```text
memory_budget_mb = N * d * sizeof(float32) * memory_budget_ratio
```

For each workload config and constraint pair, each candidate index is scored with predicted metrics:

```text
score =
  predicted_latency_norm
+ memory_penalty_weight * max(0, predicted_memory_mb / memory_budget_mb - 1)
+ recall_penalty_weight * max(0, recall_target - predicted_recall)
```

Lower score is better. The default penalty weights are both `100.0`, so going over the memory budget or below the recall target is heavily penalized relative to latency differences.

The policy chooses the index with the lowest predicted penalty score. Ties prefer lower memory overrun, lower recall shortfall, lower predicted latency, higher predicted recall, lower predicted memory, then the stable index order. The measured constrained oracle is still used for evaluation.

The default constraint grid comes from `src/config.py`:

```text
MEMORY_BUDGET_RATIOS = [0.5, 1.0, 1.5, 2.0]
RECALL_TARGETS = [0.85, 0.90, 0.95, 0.99]
```

## Layout

- `src/` — core library (data loading, indexes, labeling, models, evaluation)
- `scripts/` — CLI entry points (download, benchmark, train, etc.)
- `tests/` — pytest suite
- `data/`, `artifacts/`, `results/` — runtime outputs (ignored by git)
