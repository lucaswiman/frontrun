# Known Issues and Limitations

## Monkey-Patching Fragility

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
   (before patching) will hold real locks. This is generally fine -- cooperative
   wrappers only affect locks created *during* the controlled run -- but it means
   we can't test lock interactions inside third-party code that eagerly creates
   synchronization primitives.

## Cooperative Condition Semantics

The `CooperativeCondition` implementation uses a notification counter instead of
the real `threading.Condition` notification channel.  This fixes the
lost-notification bug (see git log), but has a remaining semantic gap:

1. **`notify()` doesn't require holding the user lock**: The standard
   `threading.Condition` contract requires the caller to hold the associated
   lock when calling `notify()`.  `CooperativeCondition.notify()` just
   increments an integer counter and works regardless.  Well-behaved user
   code holds the lock anyway, but the invariant is not enforced.

## Schedule Exhausted Fallback

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

## Random Exploration Lacks Coverage Guarantees

`explore_interleavings()` generates random schedules, which provides no feedback
about how much of the interleaving space has been covered. For simple programs
(a few opcodes, 2 threads), random works well. For anything with loops or
complex synchronization, you might need thousands of attempts to hit the one bad
interleaving, with no way to know if you've missed it. See
[dpor_spec.md](dpor_spec.md) for the principled solution.

## DPOR `ObjectState` Tracks Only Last Access (3+ Thread Blind Spot)

The Rust DPOR engine's `ObjectState` (`frontrun-dpor/src/object.rs`) stores only
a single `last_access` and `last_write_access`.  Standard DPOR tracks access
sets per-thread so that all concurrent accesses can be checked for conflicts.

With the current design, when 3+ threads access the same object, earlier accesses
are overwritten and their conflicts are never explored:

1. Thread A reads object X → recorded as `last_access`
2. Thread B reads object X → overwrites `last_access` (Thread A's read is lost)
3. Thread C writes object X → DPOR checks Thread B's read but not Thread A's

This means DPOR may miss valid interleavings with 3+ threads on the same object.
For 2-thread scenarios (the common case), this works correctly.

**Fix sketch**: Track a `HashMap<usize, Access>` mapping thread_id to last access,
or maintain a vector of all reads since the last write.

## `_INSTR_CACHE` Keyed by `id(code)` Is Fragile

`dpor.py` caches `dis.get_instructions()` results keyed by `id(code)`.  If a code
object is garbage collected and a new one is allocated at the same address within
a single DPOR execution, the cache returns stale data.  The cache is cleared
between executions but not within one.

This is unlikely to matter in practice (code objects are typically long-lived),
but could cause issues with `exec()`, `eval()`, or dynamically generated code.

## Hardcoded 5-Second Deadlock Timeout

All sync schedulers use `condition.wait(timeout=5.0)` as a fallback deadlock
detector.  This is not configurable.  Code that does legitimate long-running
work inside C extensions (NumPy, database queries, network I/O) will hit this
timeout and get a spurious `TimeoutError("Deadlock: ...")`.

See `ideas/whatever.md` for the proposed configurable timeout feature.
