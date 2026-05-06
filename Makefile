.PHONY: install test lint format run-experiment clean

# ── installation ──────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

# ── quality ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ --cov=src --cov-report=term-missing

lint:
	ruff check src/ experiments/ tests/
	black --check src/ experiments/ tests/

format:
	ruff check --fix src/ experiments/ tests/
	black src/ experiments/ tests/

# ── experiments ───────────────────────────────────────────────────────────────

CONFIG ?= configs/default.yaml

run-experiment:
	python -m experiments.run --config $(CONFIG)

# ── housekeeping ──────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info
