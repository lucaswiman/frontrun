# Frontrun

Deterministic concurrency testing for Python.

```bash
pip install frontrun
```

Frontrun runs your concurrent code under a scheduler it controls, explores the ways threads (or asyncio tasks, or OS processes) can interleave, and — when an interleaving breaks your invariant — hands you a deterministic, replayable counterexample plus a causal explanation of exactly which operations conflicted. Races either reproduce every time or are proven absent at the explored granularity: no `sleep()` tuning, no stress loops, no flaky retries. In the spirit of Rust's [loom](https://github.com/tokio-rs/loom), for Python.

## Sixty seconds: find a race and prove it

This counter is not atomic. Frontrun finds the interleaving that proves it — no markers, no annotations, no code changes:

```python
import frontrun

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        temp = self.value
        self.value = temp + 1

def test_counter_is_atomic():
    result = frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    result.assert_holds()
```

The test fails, and the failure message is the deliverable — a causal trace of the conflicting accesses:

```
Race condition found after 2 interleavings.

  Write-write conflict: threads 0 and 1 both wrote to value.

  Thread 0 | counter.py:7             temp = self.value
           | [read Counter.value]
  Thread 0 | counter.py:8             self.value = temp + 1
           | [write Counter.value]
  Thread 1 | counter.py:7             temp = self.value
           | [read Counter.value]
  Thread 1 | counter.py:8             self.value = temp + 1
           | [write Counter.value]

  Reproduced 10/10 times (100%)
```

Behind that output: frontrun detected the shared-memory accesses at the bytecode level, used DPOR (dynamic partial order reduction, powered by a Rust engine with vector clocks) to prune the 6 possible interleavings down to the 2 meaningfully different ones, found the lost update, and replayed the counterexample schedule 10 more times to confirm it reproduces deterministically. For a detailed walkthrough of how this works, see the [DPOR algorithm documentation](docs/dpor.rst).

## Not just toy counters: real, unmodified libraries

The counter above is deliberately small so the mechanics are clear, but the same `frontrun.explore()` workflow runs unchanged against code you already depend on — no forks, no injected `sleep()`, no rewritten internals. Two worked examples, each reproduced 10/10 against the released package, both in libraries whose *documented* usage is to share one object across threads (see [case studies](docs/case_studies.rst) for the full traces and reproduction commands):

- **[sendgrid](docs/case_studies.rst)** (6.12.5, via python-http-client 3.3.7; ~10M downloads/month) — the docs' own pattern is a module-level `SendGridAPIClient` reused across requests. That client's `request_headers` dict is shared and mutated without a lock, so two concurrent requests setting per-call headers race, and one request's outgoing headers (`Authorization` override, subuser, custom `X-`) are built from the other's. DPOR points at the shared `request_headers.update()` in `client.py:145`.
- **[python-socketio](docs/case_studies.rst)** (5.16.3, ~5.6M downloads/month) — under `async_mode='threading'`, two clients entering rooms in the same fresh namespace both pass the `namespace not in self.rooms` check before either writes, so the second `self.rooms[namespace] = {}` overwrites the first and a client's room registration is silently lost. DPOR reports the write-write conflict on the shared dict at `base_manager.py:116`.

These are ordinary check-then-act and read-modify-write patterns — easy to write, hard to see in review, and narrow enough under the GIL that they usually pass. frontrun makes the losing interleaving deterministic and hands you a replayable counterexample.

## Why deterministic scheduling?

Race conditions are hard to test because they depend on timing. A test that fails 5% of the time is worse than a test that always fails — it breeds false confidence, gets retried until green, and ships the bug. Frontrun replaces timing-dependent thread interleaving with deterministic scheduling, so a race either always happens or never happens, and a found counterexample is a constructive proof: run these operations in this exact order and the invariant fails.

**Free-threaded Python raises the stakes.** The GIL never made `+=` atomic — it just kept race windows narrow enough that tests usually pass. On free-threaded builds (3.13t, 3.14t), threads run truly concurrently and the races the GIL used to hide become real. Frontrun supports free-threaded Python and exists for exactly this transition: enumerate the interleavings your code can experience, and prove which one is buggy before it ships.

*About the name:* front-running is the insider-trading crime of using advance knowledge of order flow to time trades for maximum profit. The principle is the same here, except you use insider information about event ordering for maximum concurrency bugs.

## Choosing an approach

| Approach | What you get | Reach for it when |
|---|---|---|
| **DPOR** (`strategy="dpor"`, the default) | Systematic coverage of every meaningfully different interleaving, causal conflict explanations, deadlock detection | You want thoroughness and an explanation of *why* the race happens |
| **Random bytecode exploration** (`strategy="random"`) | Fast probabilistic search over opcode-level schedules, no annotations needed | You want quick fuzzing, or the shared state lives where DPOR can't see it (e.g. inside C extensions) |
| **Marker schedule exploration** (`explore_marker_interleavings`) | Exhaustive search over all interleavings of your `# frontrun:` markers | You can annotate the critical sections and want completeness at that granularity |
| **Trace markers** (`TraceExecutor`) | One exact, hand-scripted interleaving | You already know the race window and want a deterministic regression test |

All approaches have async variants, and `frontrun.explore()` detects async workers automatically. The DPOR engine also drives [cross-process exploration](#cross-process-exploration) — the same interface applied to separate Python processes contending on shared SQL/Redis state — and a C-level `LD_PRELOAD` library extends conflict detection into opaque C extensions like database drivers. For timeout, retry, and TTL races, the [virtual clock](#virtual-clock-timeout-retry-and-ttl-races) (`clock="virtual"` / `clock="explored"`) makes time itself a scheduled quantity.

## Usage Approaches

### 1. DPOR (Systematic Exploration)

DPOR (Dynamic Partial Order Reduction) *systematically* explores every meaningfully different thread interleaving. It automatically detects shared-memory accesses at the bytecode level — attribute reads/writes, subscript accesses, lock operations — and uses vector clocks to determine which orderings are equivalent. Two interleavings that differ only in the order of independent operations (two reads of different objects, say) produce the same outcome, so DPOR runs only one representative from each equivalence class. The quick-start example above uses DPOR: it explored exactly 2 interleavings out of the 6 possible, because the other 4 are equivalent to one of the first two.

DPOR also detects deadlocks, via wait-for-graph cycle analysis. Here it finds the circular wait in the classic 3-philosopher dining problem:

![Deadlock diagram showing DPOR exploration of the dining philosophers problem. Three threads each acquire one fork (lock) then block waiting for the next, forming a cycle.](docs/_static/deadlock-diagram.png)

The timeline shows each thread's lock acquisitions (green), context switches (pink arrows), and the point where the deadlock is detected. Run `make screenshot` to regenerate this image from `examples/dpor_dining_philosophers.py`.

**Search strategies:** The default DFS strategy is optimal for **exhaustive exploration** (`stop_on_first=False`) — it produces the minimum number of executions. When the trace space is very large and you have a limited execution budget (`stop_on_first=True` or a low `max_executions`), use a non-DFS strategy like `search="bit-reversal"` to spread exploration across diverse conflict points early, finding bugs faster on average. See [search strategy documentation](docs/search.rst) for details.

**Scope and limitations:** DPOR tracks Python bytecode-level conflicts (attribute and subscript reads/writes, lock operations) plus I/O. Redis key-level conflicts are detected by intercepting redis-py's `execute_command()`; activate with `detect_io=True` (works in both sync and async from 0.5). SQL conflicts are detected by intercepting DBAPI `cursor.execute()`. These key/table-level detectors are important: raw socket detection uses `host:port` as the resource ID, so every send and recv to the same server appears to conflict — without key-level or SQL-level refinement this causes a combinatorial explosion of spurious interleavings. C-extension shared state (NumPy arrays, etc.) is not tracked at all. The `frontrun` CLI adds C-level socket interception via `LD_PRELOAD` for opaque drivers, also at the coarse `host:port` level.

### 2. Bytecode Exploration (Random Strategy)

Bytecode exploration generates random opcode-level schedules and checks an invariant under each one, in the style of [Hypothesis](https://hypothesis.readthedocs.io/). Each thread fires a [`sys.settrace`](https://docs.python.org/3/library/sys.html#sys.settrace) callback at every bytecode instruction, pausing to wait for its scheduler turn. No markers or annotations needed.

The random strategy often finds races very quickly — sometimes on the first attempt. It can also find races that are invisible to DPOR, because it doesn't need to understand *why* a schedule is bad; it just checks whether the invariant holds after the threads finish. If a C extension mutates shared state in a way that breaks your invariant, random exploration will stumble into it. DPOR won't, because it can't see the C-level mutation.

The trade-off: error traces are less interpretable. You get the specific opcode schedule that broke the invariant and a best-effort interleaved source trace, but not the causal conflict analysis that DPOR provides.

```python
import frontrun

class Counter:
    def __init__(self, value=0):
        self.value = value

    def increment(self):
        temp = self.value
        self.value = temp + 1

def test_counter_is_atomic():
    result = frontrun.explore(
        setup=lambda: Counter(value=0),
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
        strategy="random",
        max_attempts=200,
        max_ops=200,
        seed=42,
    )
    result.assert_holds()
```

This fails with output like:

```
Race condition found after 1 interleavings.

  Lost update: threads 0 and 1 both read value before either wrote it back.

  Thread 1 | counter.py:7             temp = self.value
           | [read value]
  Thread 0 | counter.py:7             temp = self.value
           | [read value]
  Thread 1 | counter.py:8             self.value = temp + 1
           | [write value]
  Thread 0 | counter.py:8             self.value = temp + 1
           | [write value]

  Reproduced 10/10 times (100%)
```

The `reproduce_on_failure` parameter (default 10) controls how many times the counterexample schedule is replayed to measure reproducibility. Set to 0 to skip.

> **Note:** Opcode-level schedules are not stable across Python versions. CPython does not guarantee bytecode compatibility between releases, so a counterexample from Python 3.12 may not reproduce on 3.13. Treat counterexample schedules as ephemeral debugging artifacts.

### 3. Trace Markers

Trace markers are special comments (`# frontrun: <marker-name>`) that mark synchronization points in multithreaded or async code. A [`sys.settrace`](https://docs.python.org/3/library/sys.html#sys.settrace) callback pauses each thread at its markers and waits for a schedule to grant the next execution turn. This gives deterministic control over execution order without modifying code semantics — markers are just comments.

A marker **gates** the code that follows it: the thread pauses at the marker and only executes the gated code after the scheduler says so. Name markers after the operation they gate (e.g. `read_balance`, `write_balance`) rather than with temporal prefixes like `before_` or `after_`.

Use trace markers when you already know the race window and want to reproduce it deterministically in a regression test — here, a classic lost-update on a bank account:

```python
from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def transfer(self, amount):
        current = self.balance  # frontrun: read_balance
        new_balance = current + amount
        self.balance = new_balance  # frontrun: write_balance

def test_transfer_lost_update():
    account = BankAccount(balance=100)

    # Both threads read before either writes
    schedule = Schedule([
        Step("thread1", "read_balance"),    # T1 reads 100
        Step("thread2", "read_balance"),    # T2 reads 100 (both see same value!)
        Step("thread1", "write_balance"),   # T1 writes 150
        Step("thread2", "write_balance"),   # T2 writes 150 (overwrites T1's update!)
    ])

    executor = TraceExecutor(schedule)
    executor.run({
        "thread1": lambda: account.transfer(50),
        "thread2": lambda: account.transfer(50),
    }, timeout=5.0)

    # One update was lost: balance is 150, not 200
    assert account.balance == 150
```

### 4. Marker Schedule Exploration

When you've annotated the critical sections with markers but don't want to hand-write the losing schedule, `explore_marker_interleavings` generates *every* valid interleaving of the declared markers (preserving per-thread order) and checks the invariant under each one. The search space at marker granularity is small enough to explore exhaustively — completeness guarantees without opcode-level cost:

```python
from frontrun.trace_markers import explore_marker_interleavings

def test_transfer_has_no_lost_update():
    result = explore_marker_interleavings(
        setup=lambda: BankAccount(balance=100),
        threads={
            "thread1": (lambda account: account.transfer(50), ["read_balance", "write_balance"]),
            "thread2": (lambda account: account.transfer(50), ["read_balance", "write_balance"]),
        },
        invariant=lambda account: account.balance == 200,
    )
    result.assert_holds()  # fails, reporting the schedule that loses an update
```

### Automatic I/O Detection

Both the bytecode explorer and DPOR automatically detect socket and file I/O operations (enabled by default via `detect_io=True`). When two threads access the same network endpoint or file path, the operation is reported as a conflict so the scheduler explores their reorderings.

**Python-level detection** (monkey-patching):
- **Sockets:** `connect`, `send`, `sendall`, `sendto`, `recv`, `recv_into`, `recvfrom`
- **Files:** `open()` (read vs write determined by mode)

Resource identity is derived from the socket's peer address (`host:port`) or the file's resolved path — two threads hitting the same endpoint or file conflict; different endpoints are independent.

### Redis Key-Level Conflict Detection

DPOR goes beyond coarse socket-level detection for Redis: it intercepts `execute_command()` on redis-py clients, classifies each command as a read or write on specific keys, and reports per-key resource IDs to the engine. Two threads operating on different Redis keys are independent; only operations on the same key (with at least one write) trigger interleaving exploration.

**Sync DPOR** — Redis patching is active automatically when `detect_io=True` (the default):

```python
import frontrun
import redis

def test_redis_counter_race(redis_port):
    class State:
        def __init__(self):
            r = redis.Redis(port=redis_port, decode_responses=True)
            r.set("counter", "0")
            r.close()

    def increment(state):
        r = redis.Redis(port=redis_port, decode_responses=True)
        val = int(r.get("counter"))
        r.set("counter", str(val + 1))
        r.close()

    result = frontrun.explore(
        setup=State,
        workers=[increment, increment],
        invariant=lambda s: int(redis.Redis(port=redis_port).get("counter")) == 2,
        detect_io=True,   # default — activates Redis key-level patching
    )
    assert not result.property_holds  # DPOR finds the lost-update race
```

**Async DPOR** — `detect_io=True` covers Redis in async too:

```python
import frontrun
import redis.asyncio as aioredis

async def test_async_redis_race(redis_port):
    async def increment(state):
        r = aioredis.Redis(port=redis_port, decode_responses=True)
        val = int(await r.get("counter"))
        await r.set("counter", str(val + 1))
        await r.aclose()

    result = await frontrun.explore(
        setup=lambda: None,
        workers=increment,
        count=2,
        invariant=lambda s: True,  # check Redis directly in a real test
        detect_io=True,
    )
```

The same key-level precision applies to hashes (`HGET`/`HSET`), lists, sets, sorted sets, and all other Redis data structures — 160+ commands are classified. See the [Redis technical details](docs/redis.rst) for a full walkthrough.

### Cross-Process Exploration

The approaches above interleave concurrency *within one Python process*. Cross-process exploration extends DPOR to **separate Python processes** that contend on shared *external* state (SQL and Redis). Each worker runs frontrun's SQL/Redis interception and coordinates with a parent over a socket; the Rust DPOR engine drives the search at external-access granularity, pruning equivalent interleavings and detecting cross-worker `SELECT FOR UPDATE` deadlocks.

`frontrun.explore(..., execution="process")` mirrors the threads/async interface (install the `process` extra: `pip install frontrun[process]`). The only differences are inherent to processes: workers are serialised with **dill** (so closures and lambdas work, not just module-level functions), and `setup()` returns a **handle** to the external state (e.g. a SQLite path or DB URL) that is passed to each `worker(state)` and to `invariant(state)`. `setup` and `invariant` run in the coordinator process; workers run in their own spawned processes. It returns the same `InterleavingResult` as threads/async.

This multiprocessing path must be launched from a file-backed `.py` module, not stdin, `python -c`, or a REPL/notebook cell. For those environments, use `explore_processes()` with importable `"module:callable"` targets.

```python
import sqlite3
import frontrun

# Module-level (picklable) worker: racy read-modify-write over two statements.
def increment(db_path):
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        val = conn.execute("SELECT val FROM counter WHERE id = 1").fetchone()[0]
        conn.execute("UPDATE counter SET val = ? WHERE id = 1", (val + 1,))
    finally:
        conn.close()

def read(db_path):
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        return conn.execute("SELECT val FROM counter WHERE id = 1").fetchone()[0]
    finally:
        conn.close()

def test_counter_race(tmp_path):
    db = str(tmp_path / "counter.db")

    def setup():
        conn = sqlite3.connect(db, isolation_level=None)
        conn.execute("CREATE TABLE IF NOT EXISTS counter (id INTEGER PRIMARY KEY, val INTEGER)")
        conn.execute("DELETE FROM counter")
        conn.execute("INSERT INTO counter (id, val) VALUES (1, 0)")
        conn.close()
        return db  # picklable handle passed to each worker(state) and invariant(state)

    result = frontrun.explore(
        setup=setup,
        workers=increment,
        count=2,
        invariant=lambda state: read(state) == 2,
        execution="process",
    )
    assert not result.property_holds  # lost-update race found across processes
```

`execution="process"` accepts sync `strategy="dpor"` only; async workers and other strategies raise `ValueError`. SQLite needs nothing extra; Redis workers need the `redis` package and a running server.

The lower-level `frontrun.explore_processes(...)` spawns `frontrun.Subprocess("module:callable", args)` targets as real OS processes (the target must be importable in a fresh interpreter) and returns a `CrossProcessResult` (`.ok`, `.failure`, `.failure_kind`, `.failing_schedule`, `.iterations`). `setup` returns a handle to the shared state that is passed to `invariant(state)` (matching `execution="process"`); both run in the coordinator and may reach the shared store directly:

```python
import frontrun

# Illustrative: myapp.checkout:reserve is your own importable target, and
# reset_inventory / stock_never_negative are your own coordinator-side helpers.
# Args are passed to the child as JSON (tuples arrive as lists); use
# execution="process" above when you need richer, pickled arguments.
result = frontrun.explore_processes(
    frontrun.Subprocess("myapp.checkout:reserve", ("order-1",)),
    count=2,                                  # replicate the spec (or pass a dict/list of specs)
    setup=reset_inventory,                    # runs in the coordinator; resets the DB, returns a handle
    invariant=lambda state: stock_never_negative(state),  # receives setup()'s handle; returns True/False
    max_iterations=50,
)
if not result.ok:
    raise AssertionError(result.failure)
```

`strategy="dpor"` (default) prunes equivalent interleavings and detects deadlocks; `strategy="exhaustive"` brute-forces every interleaving as a reduction-free cross-check. `reuse_workers=True` keeps workers alive across iterations (available on both entry points). Cross-process tests spawn real processes and are marked with the pytest `e2e` marker — run them via `make test-e2e-3.14` or `pytest -m e2e`. (Cross-process mode installs its interception in Python, so it does not need the `frontrun` CLI wrapper.)

### Virtual Clock: Timeout, Retry, and TTL Races

Races involving timeouts, retries with backoff, TTL caches, and rate limiters depend on *when a timer fires*, which a wall-clock scheduler cannot control. Pass `clock="virtual"` and frontrun gives each execution a scheduler-owned virtual clock: `time.time()` / `time.monotonic()` / `time.perf_counter()` and module-qualified `datetime` current-time reads return virtual time in explored code, sleeps and async timeout wrappers become zero-wall-time virtual deadlines, timed `Lock.acquire(timeout=...)` calls resolve deterministically, and the clock autojumps to the earliest pending deadline when nothing is runnable (the same model as Trio's autojump `MockClock`). Under DPOR (sync and async), deadlocks with no pending timer are reported exactly instead of via a wall-clock fallback.

`clock="explored"` goes further: the clock advance itself becomes a schedulable DPOR step (a synthetic "clock actor"), so "the retry fired exactly between your read and your write" is explored — and replayed — like any other interleaving:

```python
import time
import frontrun

class State:
    def __init__(self):
        self.x = 0

def rmw_worker(s):        # read-modify-write over two statements
    tmp = s.x
    s.x = tmp + 1

def delayed_writer(s):    # a retry/timer firing one virtual second later
    time.sleep(1.0)
    s.x = 100

result = frontrun.explore(
    setup=State,
    workers=[rmw_worker, delayed_writer],
    invariant=lambda s: s.x == 100,
    clock="explored",
)
assert not result.property_holds  # found: the timer fired inside the RMW window
```

Works for sync and async workers with both `strategy="dpor"` and `strategy="random"`. Raw event-loop timers stay on the wall clock, but `asyncio.wait_for` and `asyncio.timeout` inside explored tasks use virtual deadlines. See [Virtual clock](docs/virtual_clock.rst) for semantics and limitations.

### C-Level I/O Interception

When run under the `frontrun` CLI, a native `LD_PRELOAD` library (`libfrontrun_io.so`) intercepts libc I/O functions directly — `connect`, `send`, `sendto`, `sendmsg`, `write`, `writev`, `recv`, `recvfrom`, `recvmsg`, `read`, `readv`, `close`. This covers opaque C extensions that Python-level patching can't see: database drivers (libpq, mysqlclient), Redis clients, HTTP libraries, and anything else that does its own I/O in C.

The library maintains a process-global map from file descriptor to resource ID (e.g. `connect(fd, 127.0.0.1:5432)` records `fd=7 → "socket:127.0.0.1:5432"`), so every subsequent `send`/`recv` on that descriptor is reported as a read or write on a stable resource the scheduler can reason about. Events stream to the Python side over a pipe in arrival order. See [How It Works Under the Hood](docs/internals.rst) for the transport details and the `IOEventDispatcher` API.

### Trace Filtering (`trace_packages`)

By default, frontrun only traces user code — files outside the stdlib, `site-packages`, and frontrun's own internals. When the code under test lives inside an installed package (Django apps, plugin architectures, etc.), pass `trace_packages` to widen the filter:

```python
import frontrun

result = frontrun.explore(
    setup=make_state,
    workers=[thread_a, thread_b],
    invariant=check_invariant,
    trace_packages=["mylib.*", "django_filters.*"],
)
```

Patterns use [`fnmatch`](https://docs.python.org/3/library/fnmatch.html) syntax and are matched against dotted module names (e.g. `django_filters.views`). All exploration entry points (`explore`, `explore_random`, and their async variants) accept this parameter. See [trace filtering docs](docs/trace_filtering.rst) for details.

## Async Support

Trace markers, random interleaving exploration, and DPOR all have async support.

### Async Trace Markers

```python
from frontrun import TraceExecutor
from frontrun.common import Schedule, Step

class AsyncCounter:
    def __init__(self):
        self.value = 0

    async def get_value(self):
        return self.value

    async def set_value(self, new_value):
        self.value = new_value

    async def increment(self):
        # frontrun: read_value
        temp = await self.get_value()
        # frontrun: write_value
        await self.set_value(temp + 1)

def test_async_counter_lost_update():
    counter = AsyncCounter()

    schedule = Schedule([
        Step("task1", "read_value"),
        Step("task2", "read_value"),
        Step("task1", "write_value"),
        Step("task2", "write_value"),
    ])

    executor = TraceExecutor(schedule)
    executor.run({
        "task1": counter.increment,
        "task2": counter.increment,
    })

    assert counter.value == 1  # One increment lost
```

### Async Exploration

Async exploration works at natural ``await`` boundaries instead of opcodes, making schedules stable across Python versions. ``frontrun.explore()`` detects async workers automatically:

```python
import asyncio
import frontrun

class Counter:
    def __init__(self):
        self.value = 0

    async def increment(self):
        temp = self.value
        await asyncio.sleep(0)  # any natural await is a scheduling point
        self.value = temp + 1

# DPOR (default) — systematic
async def test_async_counter_dpor():
    result = await frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    result.assert_holds()

# Random strategy — fast, probabilistic
async def test_async_counter_random():
    result = await frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
        strategy="random",
        max_attempts=200,
    )
    result.assert_holds()
```

## CLI

The `frontrun` CLI wraps any command with the I/O interception environment:

```bash
# Run pytest with frontrun I/O interception
frontrun pytest -vv tests/

# Run any Python program
frontrun python examples/orm_race.py

# Run a web server
frontrun uvicorn myapp:app
```

The CLI:
1. Sets `FRONTRUN_ACTIVE=1` so frontrun knows it's running under the CLI
2. Sets `LD_PRELOAD` (Linux) or `DYLD_INSERT_LIBRARIES` (macOS) to load `libfrontrun_io.so`/`.dylib`
3. Runs the command as a subprocess

## Pytest Plugin

Frontrun ships a pytest plugin (registered via the `pytest11` entry point) that
patches `threading.Lock`, `threading.RLock`, `queue.Queue`, and related
primitives with cooperative versions **before test collection**.

Patching is **on by default when running under the `frontrun` CLI**. When
running plain `pytest` without the CLI, patching is off unless explicitly
requested:

```bash
frontrun pytest                    # cooperative lock patching is active (auto)
pytest --frontrun-patch-locks      # explicitly enable without CLI
pytest --no-frontrun-patch-locks   # explicitly disable even under CLI
```

Tests that use `frontrun.explore()` or `frontrun.explore_random()` will be
**automatically skipped** when run without the frontrun CLI, preventing
confusing failures when the environment isn't properly set up.

## Platform Compatibility

| Feature | Linux | macOS | Windows |
|---|---|---|---|
| Trace markers (sync + async) | Yes | Yes | Yes |
| Bytecode exploration (sync + async) | Yes | Yes | Yes |
| DPOR (Rust engine) | Yes | Yes | Yes |
| `frontrun` CLI + C-level I/O interception | Yes | Yes | No |

**Linux** is the primary development platform and has full support for all features including the `LD_PRELOAD` I/O interception library.

**macOS** supports all features.  The `frontrun` CLI uses `DYLD_INSERT_LIBRARIES` to load `libfrontrun_io.dylib`.  Note that macOS System Integrity Protection (SIP) strips `DYLD_INSERT_LIBRARIES` from Apple-signed system binaries (`/usr/bin/python3`, etc.).  Use a Homebrew, pyenv, or venv Python to avoid this limitation.

**Windows** support is limited to trace markers, bytecode exploration, and DPOR — the pure-Python and Rust PyO3 components that don't rely on `LD_PRELOAD`.  The `frontrun` CLI and C-level I/O interception library are not available on Windows because they depend on the Unix dynamic linker's symbol interposition mechanism, which has no direct Windows equivalent.

## Development

### Prefer `assert_holds()` over manual asserts

`InterleavingResult` exposes a convenience helper that raises `AssertionError`
with the race explanation on failure and returns `None` silently on success:

```python
result = frontrun.explore(setup=setup, workers=[thread1, thread2], invariant=invariant)
result.assert_holds()  # preferred over: assert result.property_holds, result.explanation
```

An optional `msg_prefix` is prepended to the explanation:

```python
result.assert_holds(msg_prefix="transfer race: ")
```

### Running Tests

```bash
# Build everything and run tests
make test-3.10

# Or via the frontrun CLI
make build-dpor-3.10 build-io
frontrun .venv-3.10/bin/pytest -v
```
