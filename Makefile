.PHONY: test test-interlace test-tokens-regex clean docs docs-clean docs-html docs-clean-build lint type-check check

# Virtual environment setup
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

# Main test target
test: $(VENV_BIN)activate
	$(PYTEST)

test-interlace: $(VENV_BIN)activate
	$(PYTEST) tests/

test-tokens-regex:
	@echo "tokens_regex package not found in this repository"
	@exit 0

lint: $(VENV_BIN)activate
	$(VENV_BIN)ruff check interlace tests
	$(VENV_BIN)ruff format --check interlace tests

type-check: $(VENV_BIN)activate
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
	rm -rf __pycache__ .pytest_cache .eggs *.egg-info dist build .uv-cache .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
