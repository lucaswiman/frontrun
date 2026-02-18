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
