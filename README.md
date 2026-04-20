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

**Note:** As of this revision, three unit tests in `tests/test_labeling.py` (`TestComputeViolationScore::test_memory_violation_only`, `test_recall_violation_only`, and `test_both_violations`) expect violation weights (1.0 for memory, 2.0 for recall) that do not match `MEMORY_VIOLATION_WEIGHT` / `RECALL_VIOLATION_WEIGHT` in `src/config.py`. Those tests are intentionally left failing until weights and expectations are aligned.

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

`results/benchmarks.csv` mixes two kinds of information:

- Raw ANN measurements: `mean_latency_ms`, `p99_latency_ms`, `recall_at_k`, `peak_memory_mb`, and `build_time_s`
- Constraint columns for downstream decision-making: `memory_budget_mb` and `recall_target`

One important caveat: the benchmark is only run once per `(dataset, n_fraction, k, index_type)`. The rows are then repeated across `memory_budget_mb` and `recall_target` so the labeling step can ask, "which index would be best if these were the deployment constraints?"

That means:

- Compare `mean_latency_ms` vs `recall_at_k` to understand the raw search tradeoff
- Compare `peak_memory_mb` and `build_time_s` to understand build cost
- Use `memory_budget_mb` and `recall_target` only to decide which raw result is feasible for a deployment setting

`make plot` generates three views that separate those ideas:

- `latency_vs_recall.png`: raw search-quality tradeoff
- `build_cost_tradeoff.png`: memory/build-time tradeoff
- `constraint_winners.png`: which index wins after applying memory and recall constraints

## Layout

- `src/` — core library (data loading, indexes, labeling, models, evaluation)
- `scripts/` — CLI entry points (download, benchmark, train, etc.)
- `tests/` — pytest suite
- `data/`, `artifacts/`, `results/` — runtime outputs (ignored by git)
