# venv.mk - Reusable virtual environment setup for projects using uv and pyproject.toml
# Include this file in your project Makefile to get automatic venv management

VENV_BIN := .venv/bin/
PYTHON := $(VENV_BIN)python
PYTEST := $(VENV_BIN)pytest
MATURIN := $(VENV_BIN)maturin

# Use local caches for sandboxed environments
export CARGO_HOME := $(CURDIR)/.cargo-cache
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

.venv:
	uv venv .venv --python 3.12

$(VENV_BIN)activate: .venv pyproject.toml
	uv pip install -e .[dev]
	touch $(VENV_BIN)activate
