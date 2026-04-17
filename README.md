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
| `label`    | Label benchmark outputs                          |
| `hello`    | Quick script smoke test                          |
| `test`     | Run pytest on `tests/`                           |
| `clean`    | Remove `__pycache__` and `.pytest_cache`        |
| `all`      | `download` then `benchmark` then `label`         |

## Layout

- `src/` — core library (data loading, indexes, labeling, models, evaluation)
- `scripts/` — CLI entry points (download, benchmark, train, etc.)
- `tests/` — pytest suite
- `data/`, `artifacts/`, `results/` — runtime outputs (ignored by git)
