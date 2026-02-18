# Interlace

A library for deterministic concurrency testing that helps you reliably reproduce and test race conditions.

```bash
pip install interlace
```

## Overview

Interlace provides tools for controlling thread interleaving at a fine-grained level, allowing you to:

- **Deterministically reproduce race conditions** - Force specific execution ordering to make race conditions happen reliably in tests
- **Test concurrent code exhaustively** - Explore different execution orders to find bugs
- **Verify synchronization correctness** - Ensure that proper locking prevents race conditions

Instead of relying on timing-based race detection (which is unreliable), Interlace lets you control exactly when threads execute, making concurrency testing deterministic and reproducible.

## Quick Start: Bank Account Race Condition

Here's a pytest test that uses Interlace to trigger a race condition:

```python
from interlace.trace_markers import Schedule, Step, TraceExecutor

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def transfer(self, amount):
        current = self.balance  # interlace: read_balance
        new_balance = current + amount
        self.balance = new_balance  # interlace: write_balance

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

See [detailed case studies](docs/CASE_STUDIES.rst) of searching for concurrency bugs in ten libraries: TPool, threadpoolctl, cachetools, PyDispatcher, pydis, pybreaker, urllib3, SQLAlchemy, amqtt, and pykka. Run the test suites with: `PYTHONPATH=interlace python interlace/docs/tests/run_external_tests.py`

## Usage Approaches

Interlace provides two different ways to control thread interleaving:

### 1. Trace Markers

Trace markers are special comments (`# interlace: <marker-name>`) which mark particular synchronization points in multithreaded or async code. These are intended to make it easier to reproduce race conditions in test cases and inspect whether some race conditions are possible.

The execution ordering is controlled with a "schedule" object that says what order the threads / markers should run in.

Each thread runs with a [`sys.settrace`](https://docs.python.org/3/library/sys.html#sys.settrace) callback that pauses at markers and waits for a schedule to grant the next execution turn. This gives deterministic control over execution order without modifying code semantics — markers are just comments. A marker **gates** the code that follows it: the thread pauses at the marker and only executes the gated code after the scheduler grants it a turn. Name markers after the operation they gate (e.g. `read_value`, `write_balance`) rather than with temporal prefixes like `before_` or `after_`.

Markers can be placed inline or on a separate line before the operation:

```python
from interlace.trace_markers import Schedule, Step, TraceExecutor

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        temp = self.value  # interlace: read_value
        temp += 1
        self.value = temp  # interlace: write_value

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

### 2. Bytecode Manipulation (Experimental)

> ⚠️ **Experimental:** Bytecode instrumentation is experimental and may change. It requires monkey-patching concurrency primitives and relies on `f_trace_opcodes` (Python 3.7+). Use with caution.

Automatically instrument functions using bytecode rewriting — no markers needed. Each thread fires a [`sys.settrace`](https://docs.python.org/3/library/sys.html#sys.settrace) callback at every bytecode instruction, pausing at each one to wait for its scheduler turn. This gives fine-grained control but requires monkey-patching standard threading primitives (`Lock`, `Semaphore`, `Event`, `Queue`, etc.) to prevent deadlocks.

`explore_interleavings()` does property-based exploration in the style of [Hypothesis](https://hypothesis.readthedocs.io/): it generates random opcode-level schedules and checks that an invariant holds under each one, returning any counterexample schedule.

```python
from interlace.bytecode import explore_interleavings

class Counter:
    def __init__(self, value=0):
        self.value = value

    def increment(self):
        temp = self.value
        self.value = temp + 1

def test_counter_no_race():
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

    assert not result.property_holds, "Expected a race condition"
    assert result.counterexample.value == 1
```

## Async Support

Both approaches have async variants.

### Async Trace Markers

```python
from interlace.async_trace_markers import AsyncTraceExecutor
from interlace.common import Schedule, Step

class AsyncCounter:
    def __init__(self):
        self.value = 0

    async def get_value(self):
        return self.value

    async def set_value(self, new_value):
        self.value = new_value

    async def increment(self):
        # interlace: read_value
        temp = await self.get_value()
        # interlace: write_value
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

## Development

### Running Tests

```bash
make test
```
