.PHONY: install download benchmark plot label train hello test clean all

PYTHON_BIN ?= python3
PYTHON := PYTHONPATH=$(CURDIR) $(PYTHON_BIN)
BENCHMARK_FLAGS ?=

install:
	pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_datasets.py --data-dir data/

benchmark:
	$(PYTHON) scripts/run_benchmark.py --data-dir data/ --results-dir results/ $(BENCHMARK_FLAGS)

plot:
	MPLCONFIGDIR=$(CURDIR)/.mplconfig XDG_CACHE_HOME=$(CURDIR)/.cache $(PYTHON) scripts/plot_benchmarks.py --results-csv results/benchmarks.csv --output-dir results/plots

label:
	$(PYTHON) scripts/label_data.py --results-dir results/

train:
	$(PYTHON) scripts/train_models.py --results-dir results/ --artifacts-dir artifacts/

hello:
	$(PYTHON) scripts/hello_world.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; \
	true

all: download benchmark label
