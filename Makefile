.PHONY: install download benchmark label hello test clean all

install:
	pip install -r requirements.txt

download:
	python scripts/download_datasets.py --data-dir data/

benchmark:
	python scripts/run_benchmark.py --data-dir data/ --results-dir results/

label:
	python scripts/label_data.py --results-dir results/

hello:
	python scripts/hello_world.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; \
	true

all: download benchmark label
