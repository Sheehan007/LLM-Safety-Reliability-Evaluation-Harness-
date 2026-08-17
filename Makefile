.PHONY: install install-ml data validate test lint clean-results

install:
	python -m pip install -e ".[dev]"

install-ml:
	python -m pip install -e ".[dev,ml]"

data:
	PYTHONPATH=src python -m llm_safety_harness build-data --force

validate: data
	PYTHONPATH=src python -m llm_safety_harness run --config configs/validation.json

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

lint:
	ruff check src tests

clean-results:
	PYTHONPATH=src python -m llm_safety_harness clean --config configs/validation.json

