# CLAUDE.md

Deterministic concurrency testing library. Three approaches: trace markers (`# frontrun:` comments, line-level `sys.settrace`), random bytecode exploration (opcode-level fuzzing with Hypothesis), and systematic DPOR (Rust engine, shared-memory conflict analysis). First two have async variants. C-level I/O interception via `LD_PRELOAD` library.

## Project layout

- `frontrun/` — Python package (pure Python + compiled `_dpor` extension)
- `crates/dpor/` — Rust PyO3 DPOR extension (built by maturin → `frontrun/_dpor.so`)
- `crates/io/` — Rust LD_PRELOAD I/O interception library (built by cargo → `frontrun/libfrontrun_io.so`)
- `Cargo.toml` — Cargo workspace root (members: `crates/dpor`, `crates/io`)
- `pyproject.toml` — maturin build backend (builds DPOR extension as `frontrun._dpor`)

## Environment setup

Use the Makefile to build virtualenvs. Prefer working in the **3.14t** (free-threaded) virtualenv:

- `make build-dpor-3.14t` — build the 3.14t venv with Rust DPOR extension (preferred)
- `make build-dpor-3.10` / `make build-dpor-3.14` — other versions
- `make build-io` — build the LD_PRELOAD I/O interception library (libfrontrun_io.so)
- Virtualenvs live at `.venv-3.14t/`, `.venv-3.10/`, `.venv-3.14/`
- Run tools via e.g. `.venv-3.14t/bin/pytest`, `.venv-3.14t/bin/python`
- Use `frontrun` CLI to run commands with I/O interception: `frontrun pytest -v tests/`

## Commands

- `make check` — lint + type-check
- `make lint` / `make type-check` — run separately
- `make test` — all Python versions (3.10, 3.14, 3.14t); needs Rust DPOR + I/O build
- `make test-3.14t` / `make test-3.10` — single version (builds DPOR + I/O automatically)
- `make clean` — remove artifacts
- Auto-fix: `ruff check --fix frontrun tests && ruff format frontrun tests`

## Conventions

- Python >=3.10, line length 120
- ruff (E, F, W, I, N, UP, A, C4, ISC, PIE, Q, LOG, PERF), pyright strict
- Tests: pytest + hypothesis
- Rust extensions: maturin (PyO3) for DPOR (`crates/dpor/`), cargo for LD_PRELOAD library (`crates/io/`)
