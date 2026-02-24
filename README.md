# Frontrun

A library for deterministic concurrency testing.

```bash
pip install frontrun
```

## Overview

Frontrun is named after the insider trading crime where someone uses insider information to make a timed trade for maximum profit. The principle is the same here, except you use insider information about event ordering for maximum concurrency bugs.

The core problem: race conditions are hard to test because they depend on timing. A test that passes 95% of the time is worse than a test that always fails, because it breeds false confidence. Frontrun replaces timing-dependent thread interleaving with deterministic scheduling, so race conditions either always happen or never happen.

Three approaches, in order of decreasing interpretability:

1. **Trace markers** — Comment-based synchronization points (`# frontrun: marker_name`) that let you force a specific execution order. Useful when you already know the race window and want to reproduce it deterministically in a test.

2. **DPOR** — Systematically explores every meaningfully different interleaving. When it finds a race, it tells you exactly which shared-memory accesses conflicted and in what order. Powered by a Rust engine using vector clocks to prune redundant orderings.

3. **Bytecode exploration** — Generates random opcode-level schedules and checks an invariant under each one. Often finds races very efficiently (sometimes on the first attempt), and can catch races that are invisible to DPOR (e.g. shared state inside C extensions). The trade-off: error traces show *what happened* but not *why* — you get the interleaving that broke the invariant, not a causal explanation.

All three have async variants. A C-level `LD_PRELOAD` library intercepts libc I/O for database drivers and other opaque extensions.

## Quick Start: Bank Account Race Condition

A pytest test that uses trace markers to trigger a lost-update race:

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
    executor.run("thread1", lambda: account.transfer(50))
    executor.run("thread2", lambda: account.transfer(50))
    executor.wait(timeout=5.0)

    # One update was lost: balance is 150, not 200
    assert account.balance == 150
```

## Case Studies

46 concurrency bugs found across 12 libraries by running bytecode exploration directly against unmodified library code: TPool, threadpoolctl, cachetools, PyDispatcher, pydis, pybreaker, urllib3, SQLAlchemy, amqtt, pykka, and tenacity. See [detailed case studies](docs/CASE_STUDIES.rst).

## Usage Approaches

### 1. Trace Markers

Trace markers are special comments (`# frontrun: <marker-name>`) that mark synchronization points in multithreaded or async code. A [`sys.settrace`](https://docs.python.org/3/library/sys.html#sys.settrace) callback pauses each thread at its markers and waits for a schedule to grant the next execution turn. This gives deterministic control over execution order without modifying code semantics — markers are just comments.

A marker **gates** the code that follows it: the thread pauses at the marker and only executes the gated code after the scheduler says so. Name markers after the operation they gate (e.g. `read_value`, `write_balance`) rather than with temporal prefixes like `before_` or `after_`.

```python
from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        temp = self.value  # frontrun: read_value
        temp += 1
        self.value = temp  # frontrun: write_value

def test_counter_lost_update():
    counter = Counter()

    schedule = Schedule([
        Step("thread1", "read_value"),
        Step("thread2", "read_value"),
        Step("thread1", "write_value"),
        Step("thread2", "write_value"),
    ])

    executor = TraceExecutor(schedule)
    executor.run("thread1", counter.increment)
    executor.run("thread2", counter.increment)
    executor.wait(timeout=5.0)

    assert counter.value == 1  # One increment lost
```

### 2. DPOR (Systematic Exploration)

DPOR (Dynamic Partial Order Reduction) *systematically* explores every meaningfully different thread interleaving. It automatically detects shared-memory accesses at the bytecode level — attribute reads/writes, subscript accesses, lock operations — and uses vector clocks to determine which orderings are equivalent. Two interleavings that differ only in the order of independent operations (two reads of different objects, say) produce the same outcome, so DPOR runs only one representative from each equivalence class.

When a race is found, the error trace shows the exact sequence of conflicting accesses and which threads were involved:

```python
from frontrun.dpor import explore_dpor

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        temp = self.value
        self.value = temp + 1

def test_counter_is_atomic():
    result = explore_dpor(
        setup=Counter,
        threads=[lambda c: c.increment(), lambda c: c.increment()],
        invariant=lambda c: c.value == 2,
    )

    assert result.property_holds, result.explanation
```

This test fails because `Counter.increment` is not atomic. The `result.explanation` shows the conflict:

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

DPOR explored exactly 2 interleavings out of the 6 possible (the other 4 are equivalent to one of the first two). For a detailed walkthrough of how this works, see the [DPOR algorithm documentation](docs/dpor.rst).

**Scope and limitations:** DPOR tracks conflicts at the Python bytecode level. It sees attribute reads/writes, subscript operations, and lock acquire/release. It does *not* see shared state managed entirely in C extensions (database rows, NumPy arrays, Redis keys). For those, the code runs fine but DPOR concludes — incorrectly — that the threads are independent and explores only one interleaving. Direct socket I/O is detected via monkey-patching when `detect_io=True` (the default), and the `frontrun` CLI adds C-level interception via `LD_PRELOAD` for opaque drivers.

### 3. Bytecode Exploration

Bytecode exploration generates random opcode-level schedules and checks an invariant under each one, in the style of [Hypothesis](https://hypothesis.readthedocs.io/). Each thread fires a [`sys.settrace`](https://docs.python.org/3/library/sys.html#sys.settrace) callback at every bytecode instruction, pausing to wait for its scheduler turn. No markers or annotations needed.

`explore_interleavings()` often finds races very quickly — sometimes on the first attempt. It can also find races that are invisible to DPOR, because it doesn't need to understand *why* a schedule is bad; it just checks whether the invariant holds after the threads finish. If a C extension mutates shared state in a way that breaks your invariant, bytecode exploration will stumble into it. DPOR won't, because it can't see the C-level mutation.

The trade-off: error traces are less interpretable. You get the specific opcode schedule that broke the invariant and a best-effort interleaved source trace, but not the causal conflict analysis that DPOR provides.

```python
from frontrun.bytecode import explore_interleavings

class Counter:
    def __init__(self, value=0):
        self.value = value

    def increment(self):
        temp = self.value
        self.value = temp + 1

def test_counter_is_atomic():
    result = explore_interleavings(
        setup=lambda: Counter(value=0),
        threads=[
            lambda c: c.increment(),
            lambda c: c.increment(),
        ],
        invariant=lambda c: c.value == 2,
        max_attempts=200,
        max_ops=200,
        seed=42,
    )

    assert result.property_holds, result.explanation
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

### Automatic I/O Detection

Both the bytecode explorer and DPOR automatically detect socket and file I/O operations (enabled by default via `detect_io=True`). When two threads access the same network endpoint or file path, the operation is reported as a conflict so the scheduler explores their reorderings.

**Python-level detection** (monkey-patching):
- **Sockets:** `connect`, `send`, `sendall`, `sendto`, `recv`, `recv_into`, `recvfrom`
- **Files:** `open()` (read vs write determined by mode)

Resource identity is derived from the socket's peer address (`host:port`) or the file's resolved path — two threads hitting the same endpoint or file conflict; different endpoints are independent.

### C-Level I/O Interception

When run under the `frontrun` CLI, a native `LD_PRELOAD` library (`libfrontrun_io.so`) intercepts libc I/O functions directly. This covers opaque C extensions — database drivers (libpq, mysqlclient), Redis clients, HTTP libraries, and anything else that calls libc's `send()`, `recv()`, `read()`, `write()`, etc.

**Intercepted functions:** `connect`, `send`, `sendto`, `sendmsg`, `write`, `writev`, `recv`, `recvfrom`, `recvmsg`, `read`, `readv`, `close`

The library maintains a process-global file-descriptor → resource map:

```
connect(fd, sockaddr{127.0.0.1:5432}, ...)  →  record fd=7 → "socket:127.0.0.1:5432"
send(fd=7, ...)                              →  report write to "socket:127.0.0.1:5432"
recv(fd=7, ...)                              →  report read from "socket:127.0.0.1:5432"
close(fd=7)                                  →  remove fd=7 from map
```

Events are transmitted to the Python side via one of two channels:

- **Pipe (preferred):** `IOEventDispatcher` creates an `os.pipe()` and sets `FRONTRUN_IO_FD` to the write-end fd.  The Rust library writes directly to the pipe (no open/close overhead per event), and a Python reader thread dispatches events to registered listener callbacks in arrival order.  The pipe's FIFO ordering provides a natural total order without timestamps.
- **Log file (legacy):** `FRONTRUN_IO_LOG` points to a temp file.  Events are appended per-call (open + write + close each time) and read back in batch after execution.

```python
from frontrun._preload_io import IOEventDispatcher

with IOEventDispatcher() as dispatcher:
    dispatcher.add_listener(lambda ev: print(f"{ev.kind} {ev.resource_id}"))
    # ... run code under LD_PRELOAD / DYLD_INSERT_LIBRARIES ...
# all events are also available as dispatcher.events
```

## Async Support

Trace markers and bytecode exploration have async variants (DPOR does not yet have an async version).

### Async Trace Markers

```python
from frontrun.async_trace_markers import AsyncTraceExecutor
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

    executor = AsyncTraceExecutor(schedule)
    executor.run({
        "task1": counter.increment,
        "task2": counter.increment,
    })

    assert counter.value == 1  # One increment lost
```

### Async Bytecode Exploration

Async bytecode exploration works at await points instead of opcodes, making schedules stable across Python versions:

```python
import asyncio
from frontrun.async_bytecode import explore_interleavings, await_point

class Counter:
    def __init__(self):
        self.value = 0

    async def increment(self):
        temp = self.value
        await await_point()  # Yield control; race can happen here
        self.value = temp + 1

async def test_async_counter_race():
    result = await explore_interleavings(
        setup=lambda: Counter(),
        tasks=[lambda c: c.increment(), lambda c: c.increment()],
        invariant=lambda c: c.value == 2,
        max_attempts=200,
    )

    assert result.property_holds, result.explanation
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

Tests that use `explore_interleavings()` or `explore_dpor()` will be
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

### Running Tests

```bash
# Build everything and run tests
make test-3.10

# Or via the frontrun CLI
make build-dpor-3.10 build-io
frontrun .venv-3.10/bin/pytest -v
```
