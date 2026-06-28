# Makefile — task runner for xgboost-lab.
#
# `make` is not installed by default on Windows. Two options:
#   1. Run targets through Git Bash / WSL2 where `make` exists, OR
#   2. Run the underlying command directly (every target is a one-liner you
#      can copy). The commands assume your .venv is ACTIVATED so that `python`
#      resolves to .venv\Scripts\python.exe (Windows) or .venv/bin/python (POSIX).
#
# Override the interpreter or port if needed:  make train PYTHON=.venv/bin/python

PYTHON ?= python
PORT   ?= 8080

.DEFAULT_GOAL := help

.PHONY: help setup lint format typecheck test check-gpu train tune evaluate mlflow-ui clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create .venv and install all dependencies (run with base python)
	python -m venv .venv
	@echo "Activate it, then: pip install --upgrade pip && pip install -r requirements.txt -r requirements-dev.txt && pip install -e ."

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check src scripts tests

format:  ## Auto-format with ruff
	$(PYTHON) -m ruff format src scripts tests
	$(PYTHON) -m ruff check --fix src scripts tests

typecheck:  ## Static type check with mypy
	$(PYTHON) -m mypy

test:  ## Run pytest
	$(PYTHON) -m pytest

check-gpu:  ## Verify the XGBoost GPU path (Blackwell sm_120)
	$(PYTHON) scripts/check_gpu.py

train:  ## Train the baseline model
	$(PYTHON) scripts/train.py

tune:  ## Run Optuna hyperparameter tuning
	$(PYTHON) scripts/tune.py

evaluate:  ## Evaluate a trained model
	$(PYTHON) scripts/evaluate.py

mlflow-ui:  ## Launch the MLflow tracking UI (reads the sqlite store)
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(PORT)

clean:  ## Remove caches and build artifacts
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+['.ruff_cache','.mypy_cache','.pytest_cache','build','dist']]"
