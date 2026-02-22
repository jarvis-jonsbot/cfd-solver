.PHONY: setup test run clean lint lint-fix typecheck md-lint check

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check . && $(PYTHON) -m ruff format --check .

lint-fix:
	$(PYTHON) -m ruff check --fix . && $(PYTHON) -m ruff format .

typecheck:
	$(PYTHON) -m mypy src/

md-lint:
	npx markdownlint-cli2 "**/*.md" "#node_modules"

check: lint typecheck md-lint test

run:
	$(PYTHON) scripts/run_cylinder.py --mach 0.3 --cfl 0.5 --steps 5000

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__ output/
