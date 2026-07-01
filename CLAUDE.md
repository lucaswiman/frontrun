# CLAUDE.md

> `AGENTS.md` and `GEMINI.md` are symlinks to this file — edits here apply to all three.

Deterministic concurrency testing library. Four exploration approaches:

- **Trace markers** — `# frontrun: name` comments + line-level `sys.settrace` (`frontrun/trace_markers.py`, async: `async_trace_markers.py`)
- **Marker schedule exploration** — exhaustive interleaving at the marker level (also in `trace_markers.py`)
- **Random bytecode exploration** — opcode-level fuzzing with Hypothesis (`frontrun/bytecode.py`, async: `async_shuffler.py`)
- **Systematic DPOR** — Rust engine, vector clocks, wakeup tree (sync: `frontrun/_dpor_runtime/` subpackage; async: `frontrun/async_dpor.py`, still a single ~1.3 kLOC module). Pure helpers shared by both live in `frontrun/_dpor_core/` (`engine.py`, `invariants.py`, `concurrency.py`, `failures.py`, `row_locks.py`, `utils.py`) — in-progress unification layer.

Unified entry point is `frontrun.explore(strategy=...)`, which dispatches through the `Strategy` / `AsyncStrategy` Protocol registries in `frontrun/_strategy.py`. C-level I/O interception is handled by the `LD_PRELOAD` library in `crates/io/`.

## Project layout

- `frontrun/` — Python package (pure Python + compiled `_dpor` extension)
  - `_dpor_core/` — pure helpers shared by sync + async DPOR (`engine.py`, `invariants.py`, `concurrency.py`, `failures.py`, `row_locks.py`, `scheduling.py`, `utils.py`, `worker.py` — the `WorkerSet` port)
  - `_dpor_runtime/` — sync DPOR internals (`explore.py`, `scheduler.py`, `runner.py`, `worker_set.py`, `replay.py`, `preload_bridge.py`)
    - `xproc/` — cross-process exploration (`frontrun.explore(execution="process")`): schedules separate Python OS processes (spawned via `multiprocessing`/`subprocess`) contending on shared SQL/Redis state. Each worker runs frontrun's SQL/Redis patching and coordinates over a socket. `protocol.py` (wire framing), `proxy.py` (worker-side `SchedulerProxy`), `coordinator.py` (exhaustive), `dpor_coordinator.py` (engine-driven via per-worker relay threads that reuse `DporScheduler`), `launch.py`/`worker_main.py` (spawn + bootstrap). Public API in `frontrun/cross_process.py`.
  - `_strategy.py` — `Strategy` / `AsyncStrategy` Protocols + adapter registries used by `frontrun.explore()`
  - `contrib/django/`, `contrib/sqlalchemy/` — framework-specific helpers (`_sync.py`, `_async.py`, `_shared.py`)
  - Large modules worth knowing about before editing:
    - `_opcode_observer.py` — sole owner of `sys.settrace` / `sys.monitoring`. Exposes `start_opcode_trace`, `stop_opcode_trace`, `install_thread_opcode_trace`, `install_thread_line_trace`. **No other module should touch `sys.settrace` / `sys.monitoring` / `f_trace_opcodes` directly** — go through this API.
    - `_cooperative.py` — lock/queue/event patching (sync only)
    - SQL access tracking is split: `_sql_cursor.py` (driver patching surface), `_sql_parsing.py`, `_sql_transactions.py` (TX state machinery), `_sql_row_locks.py` (DPOR row-lock glue), `_sql_endpoint_suppression.py` (LD_PRELOAD bridge), `_sql_cursor_async.py` (delegates into `_sql_cursor`)
    - `_redis_client.py` / `_redis_client_async.py` — Redis access tracking; async delegates the parse-and-report path into sync
    - `async_dpor.py` — monolithic async DPOR (no subpackage yet, unlike sync; uses `_dpor_core/` for shared pure helpers)
- `crates/dpor/` — Rust PyO3 DPOR extension (built by maturin → `frontrun/_dpor.so`)
- `crates/io/` — Rust LD_PRELOAD I/O interception library (built by cargo → `frontrun/libfrontrun_io.so`)
- `tests/` — organized by feature: `test_dpor.py`, `test_bytecode.py`, `test_async_*.py`, `test_sql_*.py`, plus numbered `test_defect*.py` regressions
- `specs/` — TLA+ specifications for DPOR correctness
- `examples/`, `docs/`, `benchmarks/` — as named
- `Cargo.toml` — workspace root (`crates/dpor`, `crates/io`)
- `pyproject.toml` — maturin build backend (builds DPOR extension as `frontrun._dpor`)

## Environment setup

Use the Makefile to build virtualenvs. Prefer working in the **3.14** virtualenv:

- `make build-dpor-3.14` — build the 3.14 venv with Rust DPOR extension (preferred)
- `make build-dpor-3.10` / `make build-dpor-3.14t` — other versions
- `make build-io` — build the LD_PRELOAD I/O interception library (libfrontrun_io.so)
- Virtualenvs live at `.venv-3.14t/`, `.venv-3.10/`, `.venv-3.14/`
- Run tools via e.g. `.venv-3.14/bin/pytest`, `.venv-3.14/bin/python`
- Use `frontrun` CLI to run commands with I/O interception: `frontrun pytest -v tests/`

## Running tests

Always use `make test-<version>` to run tests. This builds the DPOR extension and I/O library, then runs pytest through the `frontrun` CLI wrapper (which sets up `LD_PRELOAD` for C-level I/O interception). Do **not** run `.venv-3.14/bin/pytest` directly — tests that use `frontrun.explore()` will be skipped or misconfigured without the `frontrun` wrapper.

- `make test-3.14` / `make test-3.10` / `make test-3.14t` — single version
- `make test` — all Python versions (3.10, 3.11, 3.12, 3.13, 3.14, 3.14t)
- Override pytest args: `make test-3.14 PYTEST_ARGS="-v --timeout=120 -k test_name"`
- Fast inner loop: scope with `-k` (e.g. `PYTEST_ARGS="-k test_dpor"`); the full suite is slow.

### Integration tests (Redis, HTTP, ORM)

Integration tests require additional packages and services:

- `make build-integration-3.14` / `make build-integration-3.10` — install redis, requests, sqlalchemy, psycopg2-binary
- `make test-integration-3.14` — run integration tests only
- Redis and Postgres are available but not running by default. Start them with:
  - `redis-server --daemonize yes`
  - `sudo pg_ctlcluster 16 main start`

## Commands

- `make check` — lint + type-check
- `make lint` / `make type-check` — run separately
- `make clean` — remove artifacts
- Auto-fix: `ruff check --fix frontrun tests && ruff format frontrun tests`

## Conventions

- Python >=3.10, line length 120
- ruff (E, F, W, I, N, UP, A, C4, ISC, PIE, Q, LOG, PERF), pyright strict
- Tests: pytest + hypothesis
- Rust extensions: maturin (PyO3) for DPOR (`crates/dpor/`), cargo for LD_PRELOAD library (`crates/io/`)

## Gotchas

- 3.14t (free-threaded) routes opcode tracing through `sys.monitoring` rather than `sys.settrace` (CPython #118415). The choice is encapsulated in `_opcode_observer.py::start_opcode_trace`; callers don't pick. If you need a new tracer hook, extend the API there rather than reaching for `sys.*` directly.
- `_cooperative.py` is sync-only by design: asyncio is already cooperative, so async paths use stock primitives. Don't try to add an async mirror.
- The async DPOR pipeline still lives in a single `async_dpor.py` instead of mirroring `_dpor_runtime/`. The shared `_dpor_core/` package is the staging area for pulling more out of both — currently holds engine construction, serializability-baseline computation, and race-failure formatting. The next slice (worker/scheduler abstraction) is not yet done.
- When adding a new exploration approach, register it as a `Strategy` / `AsyncStrategy` adapter in `frontrun/_strategy.py` so `frontrun.explore(strategy=...)` picks it up — don't add another branch in `explore.py`.
- The package exposes `explore_random` / `explore_async_random` lazily via `frontrun.__getattr__` (PEP 562) to avoid importing the bytecode shuffler at startup. Add new lazy attributes by extending `_LAZY_IMPORTS` in `frontrun/__init__.py`.
- Benchmarks live under `benchmarks/` only — there is no top-level bench script.

## Development workflow

* Start out a session with `make rebuild` — cleans and rebuilds everything, only outputs on failure or success. This is important since sometimes development is done on MacOS and sometimes in a docker container so incompatible binaries may be present.
* Always use red/green TDD: create a failing test case (or failing manual test procedure), then verify that it fails for the right reason. Commit the failing test. Then fix the failing test and commit that.
* Make sure that `make check` passes at the very end before declaring a task done. If possible, use a cheap subagent with a fresh context.
