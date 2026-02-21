# Known Issues and Limitations

## Monkey-Patching Fragility (Partially Fixed)

The bytecode approach patches `threading.Lock`, `threading.Semaphore`, etc. at the
module level, which is global mutable state. This creates several problems:

1. **Internal resolution leaks**: When stdlib code resolves names from
   `threading`'s module globals, it picks up the patched cooperative versions
   instead of the real ones. For example, `BoundedSemaphore.__init__` resolves
   `Semaphore` from `threading`'s module globals, getting our patched version.
   Every new primitive risks similar interactions. **Mitigated**: cooperative
   primitives check TLS scheduler context and fall back to real behaviour when
   no scheduler is active, so stdlib code on unmanaged threads works correctly.

2. **Parallel test runners**: **Fixed.** `patch_locks()` / `unpatch_locks()`
   now use reference counting.  Concurrent callers (e.g. pytest-xdist
   in-process parallelism) safely share a single patch — the first call
   patches, subsequent calls increment the count, and the originals are only
   restored when the count drops to zero.

3. **Import-time lock creation**: Libraries that create locks at import time
   (before patching) will hold real locks. This is inherent to the patching
   approach — cooperative wrappers only affect locks created *during* the
   controlled run.  It is not fixable without a fundamentally different
   mechanism (e.g. import hooks or bytecode rewriting).

## ~~Cooperative Condition Semantics~~ (Fixed)

~~The `CooperativeCondition` implementation uses a notification counter instead of
the real `threading.Condition` notification channel.~~

**Fixed.** `CooperativeCondition.notify()` and `notify_all()` now enforce the
standard `threading.Condition` contract: the caller must hold the associated
lock.  A `RuntimeError("cannot notify on un-acquired lock")` is raised
otherwise, matching CPython's behaviour.

## ~~Schedule Exhausted Fallback~~ (Fixed)

~~When the random schedule runs out before the program finishes, threads fall back
to real concurrency.~~

**Fixed.** `OpcodeScheduler` now dynamically extends the schedule with
round-robin entries when the explicit schedule is exhausted.  Threads remain
under deterministic scheduler control for the full duration of the run.  A
configurable `max_ops` cap (default: `10 * len(schedule) + 10000`) prevents
infinite runs.

## Random Exploration Lacks Coverage Guarantees (Improved)

`explore_interleavings()` generates random schedules, which provides no feedback
about how much of the interleaving space has been covered. For simple programs
(a few opcodes, 2 threads), random works well. For anything with loops or
complex synchronization, you might need thousands of attempts to hit the one bad
interleaving, with no way to know if you've missed it. See
[dpor_spec.md](dpor_spec.md) for the principled solution.

**Improved.** `InterleavingResult` now includes a `unique_interleavings` field
that reports how many distinct schedule orderings were actually observed during
exploration.  This provides a lower bound on coverage — if `unique_interleavings`
is much less than `num_explored`, the exploration is converging and additional
attempts are unlikely to find new behaviour.  Both the sync and async
`explore_interleavings()` functions populate this field.

## ~~DPOR `ObjectState` Tracks Only Last Access (3+ Thread Blind Spot)~~ (Fixed)

~~The Rust DPOR engine's `ObjectState` stored only a single `last_access` and
`last_write_access`, losing earlier accesses when 3+ threads touched the same
object.~~

**Fixed.** `ObjectState` now maintains `per_thread_access: HashMap<usize, Access>`
and `per_thread_write: HashMap<usize, Access>`, storing the most recent access
per thread.  `dependent_accesses(kind, current_thread)` iterates all accesses
from *other* threads, ensuring no conflicts are missed regardless of thread
count.  New Rust unit tests verify correct exploration counts for 3-thread
write-write and read-write scenarios.

## ~~`_INSTR_CACHE` Keyed by `id(code)` Is Fragile~~ (Fixed)

~~`dpor.py` caches `dis.get_instructions()` results keyed by `id(code)`, which
could return stale data if a code object was garbage collected and a new one
allocated at the same address.~~

**Fixed.** The cache is now keyed by the code object itself (not `id(code)`).
Since the dict holds a strong reference to the key, the code object cannot be
garbage-collected while cached, eliminating the address-reuse problem.  The
cache is still cleared between executions to bound memory usage.

## ~~Hardcoded 5-Second Deadlock Timeout~~ (Fixed)

~~All sync schedulers used `condition.wait(timeout=5.0)` as a non-configurable
fallback deadlock detector.~~

**Fixed.** All schedulers now accept a `deadlock_timeout` parameter (default
5.0 seconds for backward compatibility).  The parameter is exposed through
every public API:

- `OpcodeScheduler(deadlock_timeout=...)` and `DporScheduler(deadlock_timeout=...)`
- `run_with_schedule(deadlock_timeout=...)` and `explore_interleavings(deadlock_timeout=...)`
- `explore_dpor(deadlock_timeout=...)`
- `ThreadCoordinator(deadlock_timeout=...)`, `TraceExecutor(deadlock_timeout=...)`
- `frontrun(deadlock_timeout=...)` and `async_frontrun(deadlock_timeout=...)`
- `InterleavedLoop(deadlock_timeout=...)` and `AwaitScheduler(deadlock_timeout=...)`
- `async_bytecode.run_with_schedule(deadlock_timeout=...)` and `async_bytecode.explore_interleavings(deadlock_timeout=...)`

Increase the timeout for code that legitimately blocks in C extensions (NumPy,
database queries, network I/O).
