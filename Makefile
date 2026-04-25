.PHONY: install download benchmark plot label train analyze hello test clean clean-runs all

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
BENCHMARK_FLAGS ?=

install:
	$(PIP) install -r requirements.txt

download:
	$(PYTHON) -m scripts.download_datasets --data-dir data/

benchmark:
	$(PYTHON) -m scripts.run_benchmark --data-dir data/ --results-dir results/ $(BENCHMARK_FLAGS)

plot:
	$(PYTHON) -m scripts.plot_benchmarks --results-dir results/

label:
	$(PYTHON) -m scripts.label_data --results-dir results/

train: label
	$(PYTHON) -m scripts.train_models --results-dir results/ --artifacts-dir artifacts/

analyze:
	$(PYTHON) -m scripts.analyze_results --results-dir results/ --artifacts-dir artifacts/

hello:
	$(PYTHON) -m scripts.hello_world

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; \
	true

clean-runs:
	rm -rf results artifacts
	mkdir -p results artifacts

all: download benchmark label train
