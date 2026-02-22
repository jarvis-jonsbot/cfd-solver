.PHONY: setup test run clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

run:
	$(PYTHON) scripts/run_cylinder.py --mach 0.3 --cfl 0.5 --steps 5000

clean:
	rm -rf $(VENV) .pytest_cache __pycache__ src/__pycache__ tests/__pycache__ output/
