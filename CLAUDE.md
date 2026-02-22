# CLAUDE.md

Deterministic concurrency testing library. Three approaches: trace markers (`# frontrun:` comments, line-level `sys.settrace`), random bytecode exploration (opcode-level fuzzing with Hypothesis), and systematic DPOR (Rust engine, shared-memory conflict analysis). First two have async variants.

## Environment setup

Use the Makefile to build virtualenvs. Prefer working in the **3.14t** (free-threaded) virtualenv:

- `make build-dpor-3.14t` — build the 3.14t venv with Rust DPOR extension (preferred)
- `make build-dpor-3.10` / `make build-dpor-3.14` — other versions
- Virtualenvs live at `.venv-3.14t/`, `.venv-3.10/`, `.venv-3.14/`
- Run tools via e.g. `.venv-3.14t/bin/pytest`, `.venv-3.14t/bin/python`

## Commands

- `make check` — lint + type-check
- `make lint` / `make type-check` — run separately
- `make test` — all Python versions (3.10, 3.14, 3.14t); needs Rust DPOR build
- `make test-3.14t` / `make test-3.10` — single version (builds DPOR automatically)
- `make clean` — remove artifacts
- Auto-fix: `ruff check --fix frontrun tests && ruff format frontrun tests`

## Conventions

- Python >=3.10, line length 120
- ruff (E, F, W, I, N, UP, A, C4, ISC, PIE, Q, LOG, PERF), pyright strict
- Tests: pytest + hypothesis
- Rust extension: maturin (PyO3)
