# Skill: Finding Concurrency Bugs with Frontrun

You are an expert at using the **frontrun** library to find, reproduce, and
document threading race conditions in Python code.  When asked to investigate
thread safety or find concurrency bugs, follow the workflow below.

---

## What is frontrun?

Frontrun provides **deterministic concurrency testing** for Python.  Instead
of relying on timing (which is unreliable), it controls thread interleaving at
the bytecode instruction level via `sys.settrace` + `f_trace_opcodes`.

Two approaches are available:

| Approach | Use when | Stability |
|----------|----------|-----------|
| **Bytecode exploration** (`frontrun.bytecode`) | Testing unmodified third-party code; property-based search | Experimental but effective |
| **Trace markers** (`frontrun.trace_markers`) | You can add `# frontrun: name` comments to source | Stable |

For finding bugs in external libraries, **bytecode exploration** is the right
choice.

---

## Workflow: Finding a Bug with `explore_interleavings`

### Step 1 — Identify a Target

Look for code that:
- Modifies shared mutable state (`self.x += 1`, `dict[k] = v`, `list.append`)
- Has a **check-then-act** pattern (`if k not in d: d[k] = ...`)
- Has **no lock** protecting the shared state, or a gap between releasing one
  lock and acquiring another

**Common vulnerable patterns:**

```python
# Lost update — the classic
self.counter += 1          # LOAD_ATTR / BINARY_OP / STORE_ATTR: not atomic

# TOCTOU — check and act are separate
if key not in mapping:     # CHECK
    mapping[key] = value   # ACT  ← another thread can insert between these

# TOCTOU in lifecycle methods
if not self.is_alive():    # CHECK
    raise Dead()
self.inbox.put(msg)        # ACT  ← actor can die between check and put
```

### Step 2 — Check Whether frontrun Can Trace the Code

Frontrun.s opcode tracer **skips** files in `site-packages` and `lib/python`.
To trace third-party code, import from a **local source checkout**, not from
the installed package:

```python
import sys
sys.path.insert(0, "/path/to/cloned/repo/src")   # must come before any import
from mylib import TheClassUnderTest
```

### Step 3 — Write a State Class

Encapsulate setup, thread actions, and any extra tracking in a single class:

```python
class MyState:
    def __init__(self):
        self.obj = TheClassUnderTest()      # the object under test
        # add tracking fields if needed
        self.action_count = 0

    def thread1(self):
        self.obj.some_method()

    def thread2(self):
        self.obj.some_method()
```

For bugs where the violation is only visible through side effects (e.g. a
ghost message that is sent but never received), add tracking in `__init__`:

```python
class ActorState:
    def __init__(self):
        self.received = []                  # filled by the actor's on_receive
        received = self.received
        class Tracker(SomeActor):
            use_daemon_thread = True        # always use daemon threads!
            def on_receive(self, msg):
                received.append(msg)
        self.ref = Tracker.start()
        self.successes = 0
```

### Step 4 — Define a Clear Invariant

The invariant must be a **callable that takes the state object** and returns
`True` when everything is correct, `False` when a bug occurred.

| Bug type | Invariant example |
|----------|-------------------|
| Lost update on counter | `lambda s: s.obj.counter == 2` |
| Duplicate IDs | `lambda s: s.id1 != s.id2` |
| Cache size mismatch | `lambda s: s.cache.currsize == len(s.cache)` |
| Ghost message | `lambda s: s.successes == len(s.received)` |
| TOCTOU key insert | `lambda s: len(d.get(k, [])) == 2` |

### Step 5 — Run `explore_interleavings`

```python
import signal
from contextlib import contextmanager
from frontrun.bytecode import explore_interleavings, run_with_schedule

@contextmanager
def timeout_minutes(n=10):
    """Hard timeout using SIGALRM (Unix only, main thread only)."""
    def _h(sig, frame): raise TimeoutError(f"Timed out after {n}m")
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(n * 60)
    try: yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

with timeout_minutes(10):
    result = explore_interleavings(
        setup=lambda: MyState(),
        threads=[
            lambda s: s.thread1(),
            lambda s: s.thread2(),
        ],
        invariant=lambda s: s.obj.counter == 2,
        max_attempts=500,   # upper bound on schedules tried
        max_ops=300,        # max opcodes per thread before free-running
        seed=42,            # reproducible starting point
    )

print(f"Property holds: {result.property_holds}")
print(f"Explored {result.num_explored} interleavings")
if result.counterexample:
    print(f"Bug found! Schedule length: {len(result.counterexample)}")
```

**Tuning tips:**

- If no bug is found with 500 attempts, try increasing `max_ops` (longer
  schedules reach deeper into the method body) or try a different seed.
- If the bug is very rare, increase `max_attempts` to 1000–5000.
- Short methods (5–15 opcodes) are found on the **first attempt** almost
  every time.  Long methods or bugs behind locks need more attempts.

### Step 6 — Sweep Seeds

Run 20 seeds to measure how reliably the bug is found:

```python
found_seeds = []
for seed in range(20):
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: MyState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=lambda s: s.obj.counter == 2,
            max_attempts=200,
            max_ops=300,
            seed=seed,
        )
    if not result.property_holds:
        found_seeds.append((seed, result.num_explored))

print(f"Seeds found: {len(found_seeds)}/20")
```

### Step 7 — Reproduce Deterministically

Once you have a counterexample schedule, it can be replayed exactly:

```python
state = run_with_schedule(
    result.counterexample,
    setup=lambda: MyState(),
    threads=[lambda s: s.thread1(), lambda s: s.thread2()],
)
assert state.obj.counter != 2, "Bug reproduced!"
```

Replay is **100% deterministic** — the same schedule always produces the
same outcome, making it ideal as a regression test.

---

## Common Pitfalls

### Pitfall 1 — Importing from site-packages

```python
# WRONG — frontrun cannot trace site-packages
import requests
from cachetools import Cache

# RIGHT — use a local source checkout
sys.path.insert(0, "/path/to/cachetools/src")
from cachetools import Cache
```

### Pitfall 2 — Letting thread exceptions crash the run

If the code under test can raise (e.g. `ActorDeadError`, `KeyError`), catch
it in the thread function so it does not abort the entire exploration:

```python
def thread1(self):
    try:
        self.ref.tell("ping")
        self.successes += 1
    except ActorDeadError:
        self.errors += 1
```

### Pitfall 3 — Non-daemon threads blocking program exit

If your state class starts background threads (actor frameworks, thread pools),
always use daemon threads so the process can exit cleanly:

```python
class MyActor(pykka.ThreadingActor):
    use_daemon_thread = True   # ← critical
```

Or patch before starting threads:

```python
import threading
t = threading.Thread(target=worker, daemon=True)
```

### Pitfall 4 — Invariant not observable from final state

Some TOCTOU bugs are invisible in the final state alone (e.g. "actor checked
alive, actor died, message was put — both happen before invariant is checked,
so state looks the same either way").

Fix: introduce **tracking fields** that record the outcome of each action:

```python
# Instead of just calling tell(), track the result:
self.successes = 0
def thread1(self):
    try:
        self.ref.tell("ping")
        self.successes += 1   # only incremented on success
    except ActorDeadError:
        pass

# Use a shared list filled by the actor itself:
self.received = []
def on_receive(self, msg):
    self.received.append(msg)   # actor-side confirmation

# Now the invariant is observable:
invariant = lambda s: s.successes == len(s.received)
```

### Pitfall 5 — Confusing "true race / no impact" with a bug

Frontrun finds races at the bytecode level.  Whether the race *matters*
requires reading the call sites.  Key questions to ask:

* **Is the racy value used for correctness decisions** (admission control,
  protocol IDs, state transitions) or only for diagnostics (logging,
  monitoring counters)?
* **Can the value be changed at runtime?**  If a "mode flag" is fixed at
  construction time, a race on it may be unexploitable.
* **Is there an explicit comment** in the code acknowledging the intentional
  omission of a lock (e.g. "fast path, no lock needed")?

A counter that is only ever read by `logging.debug(...)` and cannot affect
any control-flow decision is a *true race / no-impact* finding — frontrun is
correct that the memory model is unsound, but the library authors may have
deliberately accepted the imprecision for performance.  File such findings as
informational notes rather than bugs.

### Pitfall 6 — Deadlocking under the cooperative scheduler

Frontrun replaces `threading.Lock`, `threading.Event`, `queue.Queue` etc. with
cooperative versions that yield turns instead of blocking.  Code that uses
**unpatched** C-level locks (e.g. `multiprocessing` primitives, some C
extensions) can deadlock.  Workaround: set `cooperative_locks=False` in
`BytecodeShuffler` and add explicit `time.sleep(0)` checkpoints instead.

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

```python
"""
Real-code exploration: <Library> <ClassName>.<method>() <bug type>.

<One-paragraph description of the bug and why it matters.>

Repository: <GitHub URL>
"""

import os, sys, signal
from contextlib import contextmanager

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, "external_repos", "<library>", "src"))

from <library> import <TheClass>
from frontrun.bytecode import explore_interleavings, run_with_schedule


@contextmanager
def timeout_minutes(n=10):
    def _h(sig, frame): raise TimeoutError(f"Timed out after {n}m")
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(n * 60)
    try: yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class State:
    def __init__(self):
        self.obj = <TheClass>(...)

    def thread1(self): self.obj.<method>()
    def thread2(self): self.obj.<method>()


def _invariant(s): return <condition that should always hold>


def test_single():
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: State(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_invariant,
            max_attempts=500, max_ops=300, seed=42,
        )
    print(f"Property holds: {result.property_holds}")
    print(f"Explored: {result.num_explored}")
    if result.counterexample:
        print(f"Schedule length: {len(result.counterexample)}")
    return result


def test_sweep():
    found = []
    for seed in range(20):
        with timeout_minutes(10):
            r = explore_interleavings(
                setup=lambda: State(),
                threads=[lambda s: s.thread1(), lambda s: s.thread2()],
                invariant=_invariant,
                max_attempts=200, max_ops=300, seed=seed,
            )
        if not r.property_holds:
            found.append((seed, r.num_explored))
    print(f"Seeds found: {len(found)}/20")
    return found


def test_reproduce():
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: State(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_invariant,
            max_attempts=500, max_ops=300, seed=42,
        )
    if not result.counterexample:
        print("No counterexample found")
        return 0
    bugs = 0
    for i in range(10):
        s = run_with_schedule(
            result.counterexample,
            setup=lambda: State(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        )
        ok = _invariant(s)
        bugs += not ok
        print(f"  Run {i+1}: [{'BUG' if not ok else 'ok'}]")
    print(f"Reproduced: {bugs}/10")
    return bugs


if __name__ == "__main__":
    print("=== Single run ===")
    test_single()
    print("\n=== Seed sweep ===")
    test_sweep()
    print("\n=== Reproduction ===")
    test_reproduce()
```

---

## Case Studies

See **docs/CASE_STUDIES.rst** for ten worked examples:
TPool, threadpoolctl, cachetools, PyDispatcher, pydis, pybreaker, urllib3,
SQLAlchemy pool, amqtt, and pykka. All demonstrate **20/20 seed detection**
with deterministic 10/10 reproduction.
