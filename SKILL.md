# Skill: Finding Concurrency Bugs with Frontrun

You are an expert at using the **frontrun** library to find, reproduce, and
document race conditions in Python code.  When asked to investigate thread
safety or find concurrency bugs, follow the workflow below.

---

## What is frontrun?

Frontrun provides **deterministic concurrency testing** for Python.  Instead
of relying on timing (which is unreliable), it controls the interleaving of
threads, asyncio tasks, or worker processes from the inside, and hands back a
deterministic, replayable counterexample schedule when an invariant breaks.

The front door is `frontrun.explore()`: hand it a way to build shared state,
some workers, and an invariant.  By default it runs **DPOR** (Dynamic Partial
Order Reduction), which *systematically* explores every meaningfully different
interleaving — every distinct schedule is tried exactly once, redundant
orderings are never re-run, and a clean pass is a proof over the whole space.

| Selector | Values | Use when |
|----------|--------|----------|
| `strategy=` | `"dpor"` (default), `"random"` | `"random"` samples schedules Hypothesis-style; use it when DPOR can't see the conflict (state mutated inside a C extension with no Python-visible accesses) |
| `execution=` | `"thread"` (default), `"process"` | `"process"` runs each worker in its own spawned Python process contending on external SQL/Redis state |
| `clock=` | `"real"` (default), `"virtual"`, `"explored"` | `"virtual"` makes sleeps/timeouts/TTLs zero-wall-time and deterministic; `"explored"` additionally makes *when the timer fires* a schedulable choice |

Sync workers run as threads; async workers (coroutine functions) are detected
automatically and run as asyncio tasks — `await` the result in that case.

Two lower-level helpers remain for special cases (covered at the end):
**trace markers** (`# frontrun: name` comments) to pin a *known* interleaving
as a regression test, and `frontrun.explore_random` / marker schedule
exploration as direct entry points to the non-DPOR engines.

---

## Workflow: Finding a Bug with `explore()`

### Step 1 — Identify a Target

Look for code that:
- Modifies shared mutable state (`self.x += 1`, `dict[k] = v`, `list.append`)
- Has a **check-then-act** pattern (`if k not in d: d[k] = ...`)
- Has **no lock** protecting the shared state, or a gap between releasing one
  lock and acquiring another

**Common vulnerable patterns:**

```python
# Lost update — the classic
self.counter += 1          # read / add / write: not atomic

# TOCTOU — check and act are separate
if key not in mapping:     # CHECK
    mapping[key] = value   # ACT  ← another thread can insert between these

# TOCTOU in lifecycle methods
if not self.is_alive():    # CHECK
    raise Dead()
self.inbox.put(msg)        # ACT  ← actor can die between check and put
```

### Step 2 — Check Whether frontrun Can Trace the Code

Frontrun's tracer **skips** files in `site-packages` and `lib/python` by
default.  To explore an installed third-party package, either opt it in with
`trace_packages` (fnmatch patterns on module names):

```python
result = frontrun.explore(..., trace_packages=["cachetools", "cachetools.*"])
```

or import it from a local source checkout instead of the installed package:

```python
import sys
sys.path.insert(0, "/path/to/cloned/repo/src")   # must come before any import
from mylib import TheClassUnderTest
```

### Step 3 — Write a State Class

Encapsulate setup, worker actions, and any extra tracking in a single class:

```python
class MyState:
    def __init__(self):
        self.obj = TheClassUnderTest()      # the object under test
        # add tracking fields if needed
        self.action_count = 0

    def worker1(self):
        self.obj.some_method()

    def worker2(self):
        self.obj.some_method()
```

For bugs only visible through side effects (e.g. a ghost message that is sent
but never received), add tracking fields in `__init__` and record outcomes in
the workers.

### Step 4 — Define a Clear Invariant

The invariant must be a **callable that takes the state object** and returns
`True` when everything is correct, `False` when a bug occurred.  Raising
`AssertionError` inside the invariant also counts as a failure and its
message is included in the explanation.

| Bug type | Invariant example |
|----------|-------------------|
| Lost update on counter | `lambda s: s.obj.counter == 2` |
| Duplicate IDs | `lambda s: s.id1 != s.id2` |
| Cache size mismatch | `lambda s: s.cache.currsize == len(s.cache)` |
| Ghost message | `lambda s: s.successes == len(s.received)` |
| TOCTOU key insert | `lambda s: len(d.get(k, [])) == 2` |

### Step 5 — Run `explore()`

```python
import frontrun

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        temp = self.value
        self.value = temp + 1

result = frontrun.explore(
    setup=Counter,                       # fresh state per execution
    workers=Counter.increment,           # or a list: [w1, w2, ...]
    count=2,                             # replicate a single worker N times
    invariant=lambda c: c.value == 2,
)
print(f"Property holds: {result.property_holds}")
print(f"Explored {result.num_explored} interleavings")
if not result.property_holds:
    print(result.explanation)            # interleaved source-line trace
```

In a test, prefer `result.assert_holds()` over manual asserts — it raises
`AssertionError` with the full race explanation on failure:

```python
def test_increment_is_atomic():
    result = frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    result.assert_holds()
```

**Run through the `frontrun` CLI wrapper** — it sets up lock patching and
C-level I/O interception; plain `pytest` *skips* `frontrun.explore()` tests:

```bash
frontrun pytest test_counter.py
frontrun python find_bug.py
```

Async workers use the same call shape; `explore()` returns a coroutine:

```python
class AsyncCounter:
    def __init__(self):
        self.value = 0

    async def increment(self):
        temp = self.value
        await asyncio.sleep(0)
        self.value = temp + 1

result = asyncio.run(frontrun.explore(
    setup=AsyncCounter,
    workers=AsyncCounter.increment,
    count=2,
    invariant=lambda c: c.value == 2,
))
```

### Step 6 — Read the Failure

When a race is found, `result.explanation` names the conflicting accesses and
shows the interleaved source-line trace; DPOR *knows why* the interleaving
matters because it detected the specific conflicting accesses.  The
counterexample is replayed automatically (`reproduce_on_failure`, default 10)
to confirm it reproduces deterministically before being reported.

A DPOR pass is a completeness result: every distinct interleaving (up to
`preemption_bound`, default 2 preemptions) was proven safe.

### Step 7 — Escalate When the Default Doesn't Fit

**Timeout / retry / TTL races** — real sleeps make these unreachable.
`clock="virtual"` gives each execution a scheduler-owned virtual clock:
`time.time()` / `time.monotonic()` read virtual time, sleeps cost zero wall
time, and timed lock acquires time out deterministically.  Use
`clock="explored"` when the *timing* of a timer firing is itself the race:

```python
class Timeout:
    def __init__(self):
        self.elapsed = 0.0

    def worker(self):
        start = time.monotonic()     # reads virtual time
        time.sleep(60)               # zero wall time under clock="virtual"
        self.elapsed = time.monotonic() - start

result = frontrun.explore(
    setup=Timeout,
    workers=Timeout.worker,
    count=1,
    invariant=lambda s: s.elapsed >= 60,   # the timeout path was really taken
    clock="virtual",
)
```

**Cross-process contention on SQL/Redis** — `execution="process"` spawns each
worker as its own Python process, coordinating over a socket.  `setup()` must
return a picklable *handle* to external state (a DB path/URL, not a live
connection); workers are serialised with dill, so closures work.  Requires
the `process` extra (`pip install frontrun[process]`) and
`strategy="dpor"` with sync workers.

**DPOR can't see the state** — shared state mutated entirely inside a C
extension (no Python-visible attribute/subscript access, no I/O) is invisible
to DPOR's conflict detection.  `strategy="random"` samples random opcode-level
schedules and can stumble into the bad interleaving anyway:

```python
result = frontrun.explore(
    setup=Counter,
    workers=Counter.increment,
    count=2,
    invariant=lambda c: c.value == 2,
    strategy="random",
    max_attempts=500,   # schedules to sample
    seed=42,            # reproducible starting point
)
```

Options are strategy-specific and `explore()` **raises `ValueError`** if you
pass one the selected strategy does not support (e.g. `seed=` with DPOR, or
`preemption_bound=` with `strategy="random"`) — fix the call rather than
working around the error:

| Options | Apply to |
|---------|----------|
| `max_executions`, `preemption_bound`, `max_branches`, `stop_on_first`, `lock_timeout` | DPOR only |
| `search`, `track_dunder_dict_accesses` | sync DPOR only |
| `max_attempts`, `max_ops`, `seed` | random only |
| `debug` | sync random only |
| `detect_sql` | async workers only |
| `clock`, `clock_diagnostics`, `timeout_per_run`, `total_timeout`, `detect_io`, `patch_sleep`, `trace_packages`, `serializable_invariant` | both strategies |

### Step 8 — Pin a Regression Test

The simplest regression test is the `explore()` call itself with
`result.assert_holds()` — after the fix, DPOR proves the whole space clean.

To pin one *specific* interleaving (e.g. to document the exact bug window),
use **trace markers**: `# frontrun: name` comments that gate the line that
follows them, plus an explicit schedule:

```python
from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        temp = self.value  # frontrun: read_value
        self.value = temp + 1  # frontrun: write_value

counter = Counter()
schedule = Schedule([
    Step("thread1", "read_value"),   # T1 reads 0
    Step("thread2", "read_value"),   # T2 reads 0 (both see the same value!)
    Step("thread1", "write_value"),  # T1 writes 1
    Step("thread2", "write_value"),  # T2 writes 1 (overwrites T1's update!)
])
TraceExecutor(schedule).run(
    {"thread1": counter.increment, "thread2": counter.increment},
    timeout=5.0,
)
assert counter.value == 1  # one increment lost — deterministically
```

To verify a fix eliminates **all** marker-level interleavings, not just the
one counterexample, use `explore_marker_interleavings()` from
`frontrun.trace_markers` — it exhaustively runs every valid ordering of the
markers.

---

## Common Pitfalls

### Pitfall 1 — Running without the `frontrun` CLI wrapper

Under plain `pytest`, tests that call `frontrun.explore()` are skipped (lock
patching and C-level I/O interception are not set up).  Always run
`frontrun pytest ...` / `frontrun python ...`.

### Pitfall 2 — Tracing installed packages

```python
# WRONG — frontrun skips site-packages by default
from cachetools import Cache

# RIGHT — opt the package in
frontrun.explore(..., trace_packages=["cachetools", "cachetools.*"])

# ALSO RIGHT — import from a local source checkout
sys.path.insert(0, "/path/to/cachetools/src")
from cachetools import Cache
```

### Pitfall 3 — Letting worker exceptions crash the run

If the code under test can raise (e.g. `ActorDeadError`, `KeyError`), catch
it in the worker so it becomes an observable outcome instead of aborting the
exploration:

```python
def worker1(self):
    try:
        self.ref.tell("ping")
        self.successes += 1
    except ActorDeadError:
        self.errors += 1
```

### Pitfall 4 — Non-daemon threads blocking program exit

If your state class starts background threads (actor frameworks, thread
pools), always use daemon threads so the process can exit cleanly:

```python
t = threading.Thread(target=worker, daemon=True)
```

### Pitfall 5 — Invariant not observable from final state

Some TOCTOU bugs are invisible in the final state alone.  Introduce
**tracking fields** that record the outcome of each action (a `successes`
counter incremented only on success, a `received` list filled by the
consumer), then write the invariant over the tracking fields:

```python
invariant = lambda s: s.successes == len(s.received)
```

### Pitfall 6 — Confusing "true race / no impact" with a bug

Whether a detected race *matters* requires reading the call sites.  Ask:

* **Is the racy value used for correctness decisions** (admission control,
  protocol IDs, state transitions) or only for diagnostics (logging,
  monitoring counters)?
* **Can the value be changed at runtime?**  A race on a flag fixed at
  construction time may be unexploitable.
* **Is there an explicit comment** acknowledging the intentional omission of
  a lock ("fast path, no lock needed")?

File true-race/no-impact findings as informational notes rather than bugs.
(`error_on_any_race=True` makes DPOR treat every unsynchronized race as a
failure — useful for auditing, noisy for triage.)

### Pitfall 7 — Blocking C-level primitives

Frontrun replaces `threading.Lock`, `threading.Event`, `queue.Queue` etc.
with cooperative versions that yield turns instead of blocking.  Code that
blocks in **unpatched** C-level primitives (e.g. `multiprocessing`
primitives, some C extensions) can deadlock the scheduler.  Workaround: add
explicit `time.sleep(0)` checkpoints in the code under test so the scheduler
can interleave around the C-level block.

---

## Quick Reference: Choosing Invariants

| Scenario | Recommended Invariant |
|----------|----------------------|
| Counter incremented N times | `obj.counter == N` |
| Two inserts into a set/list | `len(collection) == 2` |
| Cache size consistent | `cache.currsize == len(cache)` |
| Two unique IDs allocated | `id1 != id2` |
| Both receivers registered | `len(signal_receivers) == 2` |
| Message delivered to actor | `tell_successes == len(received)` |
| Overflow counter correct | `pool._overflow == initial + N` |
| Item found after insert | `key in mapping` |

---

## Template: Complete Test File

Run with `frontrun pytest test_mylib_races.py`.

```python
"""
Real-code exploration: <Library> <ClassName>.<method>() <bug type>.

<One-paragraph description of the bug and why it matters.>

Repository: <GitHub URL>
"""

import frontrun

from mylib import TheClass  # opt in below via trace_packages if installed


class State:
    def __init__(self):
        self.obj = TheClass()

    def worker1(self):
        self.obj.some_method()

    def worker2(self):
        self.obj.some_method()


def _invariant(s):
    return s.obj.counter == 2  # condition that should always hold


def test_method_is_atomic():
    """DPOR proves the property over all interleavings, or hands back a
    deterministic counterexample in the AssertionError message."""
    result = frontrun.explore(
        setup=State,
        workers=[State.worker1, State.worker2],
        invariant=_invariant,
        trace_packages=["mylib", "mylib.*"],
    )
    result.assert_holds()


def test_method_is_atomic_random_fallback():
    """Random sampling — reaches interleavings DPOR cannot see when the
    conflict lives inside a C extension."""
    result = frontrun.explore(
        setup=State,
        workers=[State.worker1, State.worker2],
        invariant=_invariant,
        strategy="random",
        max_attempts=500,
        seed=42,
        trace_packages=["mylib", "mylib.*"],
    )
    result.assert_holds()
```
