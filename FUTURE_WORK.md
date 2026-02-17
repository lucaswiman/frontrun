# Future Work

## Known Issues and Limitations

### Monkey-Patching Fragility

The bytecode approach patches `threading.Lock`, `threading.Semaphore`, etc. at the
module level, which is global mutable state. This creates several problems:

1. **Internal resolution leaks**: When stdlib code resolves names from
   `threading`'s module globals, it picks up the patched cooperative versions
   instead of the real ones. For example, `BoundedSemaphore.__init__` resolves
   `Semaphore` from `threading`'s module globals, getting our patched version.
   Every new primitive risks similar interactions.

2. **Parallel test runners**: If tests are run in parallel (e.g., `pytest-xdist`
   with `--forked` or in-process parallelism), the global patches will collide
   across test sessions. The patching is scoped per-run via `_patch_locks()` /
   `_unpatch_locks()`, but there is no protection against concurrent test
   processes sharing the same `threading` module.

3. **Import-time lock creation**: Libraries that create locks at import time
   (before patching) will hold real locks. This is generally fine — cooperative
   wrappers only affect locks created *during* the controlled run — but it means
   we can't test lock interactions inside third-party code that eagerly creates
   synchronization primitives.

### Cooperative Condition Semantics

The `_CooperativeCondition` implementation has subtle semantic issues:

1. **Disconnected notification channel**: The implementation releases the
   user-visible lock, then spin-yields checking a separate internal real
   condition (`_real_cond`). But `notify()` acquires the internal real lock, not
   the user's cooperative lock. This means `notify()` doesn't actually require
   holding the user lock (violating the `threading.Condition` contract).

2. **Lost notifications and spurious wakeups**: Because the notification channel
   is disconnected from the user-visible lock, the semantics around spurious
   wakeups and lost notifications under interleaving are unclear. There are
   likely edge cases where this silently does the wrong thing.

### Schedule Exhausted Fallback

Every cooperative wrapper contains a fallback branch:

```python
if scheduler._finished or scheduler._error:
    return self._lock.acquire(blocking=blocking, timeout=1.0)
```

When the random schedule runs out before the program finishes, threads fall back
to real concurrency with a 1-second timeout. This means the scheduler only
controls a *prefix* of the interleaving and hopes the suffix works out. In
practice this is usually fine, but it undermines any claim of full deterministic
control over thread scheduling.

### Random Exploration Lacks Coverage Guarantees

`explore_interleavings()` generates random schedules, which provides no feedback
about how much of the interleaving space has been covered. For simple programs
(a few opcodes, 2 threads), random works well. For anything with loops or
complex synchronization, you might need thousands of attempts to hit the one bad
interleaving, with no way to know if you've missed it. See the DPOR section
below for the principled solution.

---

## Async Trace Markers: Comment-Based Syntax

**Priority: High** — Implemented! Async trace markers now use `# interlace: marker_name` comments matching the sync API.

**Status**:
- [x] Design coroutine-driving approach with `sys.settrace`
- [x] Implement `_drive_with_markers()` driver that wraps user coroutines
- [x] Reuse `MarkerRegistry` from sync trace markers
- [x] Update tests to remove explicit `await mark()` calls
- [ ] Add optional marker validation (warn if two markers with no `await` between)
- [ ] Add comprehensive documentation and examples

### Implementation: Coroutine-Driving with settrace

**Key insight**: `sys.settrace` fires line events during coroutine execution but you can't `await` from a trace function. Solution: wrap each user coroutine in a "driver" that manually steps it with `coro.send()`, checking for markers between each yield.

**Driver flow for each task**:
1. Install `sys.settrace` (marker-detecting trace function)
2. `coro.send(value)` — runs coroutine until next await/return
3. Uninstall `sys.settrace`
4. For each marker detected during step 2:
   - `await coordinator.pause(task_id, marker_name)`
5. Await the yielded value (the thing the coroutine awaited)
6. Feed result back to step 2

**Why this works**:
- Async is single-threaded, so `coro.send()` is synchronous — no trace function conflicts between tasks
- Reuses `MarkerRegistry` from sync for comment detection
- Reuses `InterleavedLoop`/`AsyncTaskCoordinator` for scheduling
- Works with any awaitable (sleep, I/O, sub-coroutines)

**Marker semantics**: A `# interlace: name` comment marks the beginning of an atomic block that extends to the next `await`. The scheduler gates when the task resumes after that await. This correctly models async race conditions, which only occur at await points.

---

## Explicit Cooperative Primitives (Loom-Style)

**Priority: Low** — Alternative to monkey-patching for production use.

**Status**:
- [ ] Design `interlace.sync` module API (mirror `threading` and `queue`)
- [ ] Implement thin dispatch wrappers over existing cooperative primitives
- [ ] Add mode detection (controlled vs. production)
- [ ] Create migration guide for users

An alternative to monkey-patching is providing explicit cooperative replacements
that users opt into, inspired by the Rust
[loom](https://github.com/tokio-rs/loom) library.

In loom, instead of `std::sync::Mutex`, you use `loom::sync::Mutex`. Under test,
loom's Mutex cooperates with loom's scheduler. In production, it compiles down to
the real Mutex. The user changes imports, not runtime behavior.

The equivalent for interlace:

```python
# Instead of:
import threading
lock = threading.Lock()

# User writes:
from interlace.sync import Lock
lock = Lock()
```

Under `controlled_interleaving`, `interlace.sync.Lock` behaves like
`_CooperativeLock`. Outside of it, it delegates to `threading.Lock` with zero
overhead.

### Advantages over monkey-patching

- **No global state mutation**: Patching `threading.Lock` affects the entire
  process, including the scheduler, unrelated libraries, and other test threads.
  Explicit imports are scoped to the code that uses them.
- **No recursion risk**: The scheduler uses real `threading.Condition` by
  construction — there's nothing to accidentally patch.
- **Clearer intent**: Reading the code makes it obvious which primitives are
  under scheduler control.
- **Composable**: Multiple test harnesses can coexist without conflicting patches.

### Disadvantages

- **Requires import changes**: Can't test unmodified code. This is the main
  drawback — the whole point of the bytecode approach is testing code as-is.
- **Dual-import maintenance**: `interlace.sync` must mirror the `threading` and
  `queue` APIs exactly, and stay in sync as CPython evolves.

### Why this matters

The monkey-patching approach works for demos and simple cases, but the fragility
issues described above (global state, internal resolution leaks, parallel test
collisions) make it a poor fit for production test suites. The loom-style
approach would be more robust for users willing to change imports. Since all the
cooperative wrapper implementations already exist, `interlace.sync` would just be
thin dispatch wrappers over them.

---

## Dynamic Partial Order Reduction (DPOR)

**Priority: Medium** — Future work for exhaustive async/thread interleaving exploration.

**Status**:
- [ ] Design conflict detection for memory locations (object attributes, dict keys, list indices)
- [ ] Implement happens-before tracking based on synchronization
- [ ] Implement backtrack set computation for classic DPOR
- [ ] Add execution replay support
- [ ] For async DPOR: Create custom `InterlaceEventLoop(asyncio.SelectorEventLoop)` subclass
- [ ] Integrate coroutine-driving from async trace markers into custom event loop
- [ ] Research and potentially implement Optimal DPOR (Abdulla et al., 2014)

The current exploration strategy in `explore_interleavings()` generates random
schedules — essentially a random walk through the interleaving space. This works
well for finding bugs quickly (a random schedule has a reasonable probability of
hitting most concurrency bugs), but it provides no coverage guarantees and may
redundantly explore equivalent interleavings.

**Dynamic Partial Order Reduction (DPOR)** is a family of algorithms that
systematically explore only the *distinct* interleavings — those that differ in
the ordering of conflicting (dependent) operations. Two interleavings that differ
only in the ordering of independent operations (e.g., two threads writing to
different variables) are equivalent and only one needs to be tested.

### How DPOR works

1. Execute the program under one interleaving, recording which operations each
   thread performs and which operations *conflict* (access the same shared state
   with at least one write).
2. Identify **backtrack points**: positions in the execution where a different
   thread ordering would produce a genuinely different result (because the
   reordered operations conflict).
3. Explore those alternative orderings, repeating the process.
4. The algorithm terminates when all non-equivalent interleavings have been
   explored.

### Variants

- **Classic DPOR** (Flanagan & Godefroid, 2005): The original algorithm. Uses
  persistent sets to identify necessary backtrack points.
- **Optimal DPOR** (Abdulla et al., 2014): Guarantees that each distinct
  Mazurkiewicz trace is explored exactly once. Eliminates redundant explorations
  that classic DPOR can still produce.
- **Source-DPOR** and **context-sensitive DPOR**: Further refinements that reduce
  the explored state space.

### What interlace would need

1. **Conflict detection**: Track which memory locations (Python object
   attributes, dictionary keys, list indices) each thread reads and writes at
   each opcode. The opcode trace infrastructure already intercepts every
   instruction — it would need to additionally record the accessed
   addresses.

2. **Happens-before tracking**: Maintain a partial order of events based on
   synchronization (lock acquire/release pairs establish ordering). This lets the
   algorithm determine which operations are truly concurrent vs. already ordered.

3. **Backtrack set computation**: After each full execution, compute which
   alternative thread orderings at conflict points would produce distinct
   behaviors, and add those to the exploration queue.

4. **Execution replay**: The ability to replay an execution up to a backtrack
   point and then diverge. The current schedule-based infrastructure supports
   this naturally — a schedule prefix defines the execution up to a point, and
   the suffix can be varied.

### Why this is important

DPOR is the principled solution to the random exploration coverage gap. Random
exploration with a few hundred attempts works for small, focused tests, but
provides no way to know if you've missed a rare interleaving. DPOR gives
*exhaustive* verification — proving the absence of bugs, not just finding them.
It becomes especially valuable as program complexity and interleaving space size
grow.

### Custom Event Loop for Async DPOR

For async code, the custom `InterlaceEventLoop` approach offers lower-level control than the current `asyncio.Condition`-based scheduler:

- Subclass `asyncio.SelectorEventLoop` to intercept all task scheduling decisions
- Observe all task interactions (not just marked points), enabling conflict detection
- The coroutine-driving approach from async trace markers can be integrated into the event loop for unified control
- This becomes the foundation for async DPOR implementation

### References

- Flanagan, C. and Godefroid, P. "Dynamic Partial-Order Reduction for Model
  Checking Software." POPL 2005.
- Abdulla, P. et al. "Optimal Dynamic Partial Order Reduction." POPL 2014.
- The [loom](https://github.com/tokio-rs/loom) Rust library implements a variant
  of DPOR for exhaustive async interleaving exploration.

---

## Documentation Bugs

**Priority: High** — These bugs were discovered during the async marker implementation and should be fixed.

**Status**:
- [ ] **README Quick Start** (line 22): `from interlace.mock_events import Interlace` — `mock_events` module doesn't exist. Should use trace markers example instead.
- [ ] **README Async Example** (line 244): Uses nonexistent `AsyncSchedule`, `AsyncStep` (should be `Schedule`, `Step` from `interlace.common`)
- [ ] **README Async Example** (lines 268-278): Shows wrong API — `executor.run("task1", worker1)` + `await executor.wait()` don't exist. Actual API is `await executor.run({"task1": ...})`
- [ ] **async_scheduler.py docstring** (line 18): References nonexistent `async_mock_events` module
- [ ] **README test count** (lines 299-306): Lists 40 tests, missing newer test files
- [ ] **README project structure** (lines 309-328): Missing `common.py`, `conftest.py`, `test_concurrency_bug_classes.py`, `buggy_programs.py`, `test_threading_primitives.py`
