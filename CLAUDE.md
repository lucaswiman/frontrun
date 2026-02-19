# CLAUDE.md

Deterministic concurrency testing library. Three approaches: trace markers (`# frontrun:` comments, line-level `sys.settrace`), random bytecode exploration (opcode-level fuzzing with Hypothesis), and systematic DPOR (Rust engine, shared-memory conflict analysis). First two have async variants.

## Commands

- `make check` — lint + type-check
- `make lint` / `make type-check` — run separately
- `make test` — all Python versions (3.10, 3.14, 3.14t); needs Rust DPOR build
- `make test-3.10` / `make build-dpor-3.10` — single version
- `make clean` — remove artifacts
- Auto-fix: `ruff check --fix frontrun tests && ruff format frontrun tests`

## Conventions

- Python >=3.10, line length 120
- ruff (E, F, W, I, N, UP, A, C4, ISC, PIE, Q, LOG, PERF), pyright strict
- Tests: pytest + hypothesis
- Rust extension: maturin (PyO3)
