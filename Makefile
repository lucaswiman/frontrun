.PHONY: test clean docs docs-clean docs-html docs-clean-build lint type-check check

# Python versions to test
PYTHON_VERSIONS := 3.14t 3.10 3.14

PYTEST_ARGS?=--tb=short -v

# Virtual environment setup
VENV_BIN := .venv-3.10/bin/
PYTHON := $(VENV_BIN)python
PYTEST := $(VENV_BIN)pytest
MATURIN := $(VENV_BIN)maturin

# Rust source files in dpor directory
DPOR_RUST_SOURCES := $(wildcard frontrun-dpor/src/*.rs) frontrun-dpor/Cargo.toml

# Use local caches for sandboxed environments
export CARGO_HOME := $(CURDIR)/.cargo-cache
export UV_CACHE_DIR := $(CURDIR)/.uv-cache

# Pattern rule for creating version-specific virtualenvs
.venv-%:
	uv venv .venv-$* --python=$*

# Pattern rule for installing dependencies in version-specific virtualenvs
.venv-%/activate: .venv-% pyproject.toml
	uv pip install -e .[dev] --python=$(CURDIR)/.venv-$*/bin/python
	touch $(CURDIR)/.venv-$*/bin/activate

# Build the frontrun-dpor Rust extension for a specific Python version.
# Uses maturin to compile the PyO3 extension module and install it into
# the version-specific virtualenv. Rebuilds when Rust files in dpor change.
build-dpor-%: .venv-%/activate $(DPOR_RUST_SOURCES)
	uv pip install maturin --python=$(CURDIR)/.venv-$*/bin/python
	cd frontrun-dpor && VIRTUAL_ENV=$(CURDIR)/.venv-$* $(CURDIR)/.venv-$*/bin/maturin develop --release

# Build example venv with SQLAlchemy + psycopg2 for examples/orm_race.py
build-examples-%: build-dpor-%
	uv pip install sqlalchemy psycopg2-binary --python=$(CURDIR)/.venv-$*/bin/python

.PHONY: default-venv
default-venv: .venv-3.10/activate


# Pattern rule for running tests with specific Python versions.
# Builds the Rust DPOR extension first.
test-%: build-dpor-%
	$(CURDIR)/.venv-$*/bin/pytest $(PYTEST_ARGS)

# Main test target - runs tests for all Python versions
test: $(addprefix test-,$(PYTHON_VERSIONS))

lint: default-venv
	$(VENV_BIN)ruff check frontrun tests
	$(VENV_BIN)ruff format --check frontrun tests

type-check: default-venv
	$(VENV_BIN)pyright

check: lint type-check

# Read The Docs documentation targets
docs: docs-html
	@echo "Documentation built in docs/_build/html"

docs-html:
	cd docs && $(MAKE) clean html

docs-clean:
	cd docs && rm -rf _build

docs-clean-build: docs-clean docs-html

clean: docs-clean
	rm -rf __pycache__ .pytest_cache .eggs *.egg-info dist build .uv-cache .venv $(addprefix .venv-,$(PYTHON_VERSIONS))
	rm -rf frontrun-dpor/target .cargo-cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
