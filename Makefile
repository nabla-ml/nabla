# Nabla ML Infrastructure Makefile

# Absolute path for the project root
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Virtual environment detection
VENV := $(ROOT_DIR)/venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
    BIN := $(VENV)/Scripts
endif

# Check if venv exists, otherwise use global python
PYTHON := python3
ifneq ("$(wildcard $(BIN)/python*)","")
    PYTHON := $(BIN)/python
endif

.PHONY: help test test-unit test-mojo lint format typecheck docs docs-serve clean install dev-install

help:
	@echo "Nabla ML Development Commands:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev-install    Install development dependencies and package in editable mode"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-mojo      Run Mojo-specific tests"
	@echo "  make lint           Run ruff check"
	@echo "  make format         Run ruff format"
	@echo "  make typecheck      Run mypy type checking"
	@echo "  make docs           Build Sphinx documentation"
	@echo "  make docs-serve     Serve documentation locally on port 8000"
	@echo "  make clean          Clean build artifacts and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

dev-install:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/

test-unit:
	$(PYTHON) -m pytest tests/unit/

test-mojo:
	$(PYTHON) -m pytest tests/mojo/

lint:
	$(PYTHON) -m ruff check nabla/ tests/

format:
	$(PYTHON) -m ruff format nabla/ tests/

typecheck:
	$(PYTHON) -m mypy nabla/

docs:
	export PATH="$(BIN):$$PATH" && bash docs/build.sh

docs-serve:
	$(PYTHON) -m http.server -d docs/_build/html 8000

clean:
	rm -rf docs/_build/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
