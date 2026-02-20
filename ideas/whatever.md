# Missing Features / Ideas

## 1. Configurable deadlock timeout

All three schedulers (`OpcodeScheduler`, `DporScheduler`, `InterleavedLoop`) hardcode
`condition.wait(timeout=5.0)` as the fallback deadlock detection timeout.  Users with
code that does legitimate long-running C-extension work (NumPy, database queries,
network I/O) will hit this timeout spuriously.

Expose a `deadlock_timeout` parameter through the user-facing APIs:

- `OpcodeScheduler.__init__(schedule, num_threads, deadlock_timeout=5.0)`
- `explore_interleavings(..., deadlock_timeout=5.0)`
- `explore_dpor(..., deadlock_timeout=5.0)`

## 2. All-threads-waiting instant deadlock detection for sync schedulers

`InterleavedLoop` (async) already has instant all-tasks-waiting deadlock detection:
when `_waiting_count >= alive`, it fires immediately instead of waiting for the
5-second timeout.

`OpcodeScheduler` and `DporScheduler` lack this.  Adding it would turn most
5-second timeout waits into instant detections:

```python
# In OpcodeScheduler.wait_for_turn:
self._waiting_count += 1
try:
    alive = self.num_threads - len(self._threads_done)
    if self._waiting_count >= alive:
        self._error = TimeoutError("All threads waiting, none can proceed")
        self._condition.notify_all()
        return False
    ...
finally:
    self._waiting_count -= 1
```

## 3. Async DPOR

DPOR is the most powerful exploration approach but only supports threaded code.
An async variant would let users of `asyncio` benefit from systematic exploration
instead of random sampling.

## 4. Schedule shrinking / minimization

When a bug is found, counterexample schedules can be hundreds of entries long.
Most entries are irrelevant — only a few key ordering decisions trigger the bug.
A schedule minimizer (binary search for minimal reproducing schedule) would make
debugging much easier.

The bytecode `schedule_strategy` already integrates with Hypothesis, which has
built-in shrinking, but the interaction with threading makes shrinking unreliable
(the docstring itself recommends `phases=[Phase.generate]` to skip shrinking).

## 5. Progress reporting / callbacks

DPOR can explore thousands of interleavings over minutes.  There's currently no
callback or progress reporting mechanism.  A simple callback would help:

```python
result = explore_dpor(
    ...,
    on_progress=lambda explored, total_estimate: print(f"{explored} explored"),
)
```

## 6. `threading.Barrier` in cooperative primitives

`_cooperative.py` covers Lock, RLock, Semaphore, BoundedSemaphore, Event,
Condition, and all Queue variants.  `threading.Barrier` is absent.  User code
that uses `Barrier` will deadlock under the scheduler because the real `Barrier`
blocks in C.

## 7. Dynamic thread creation in DPOR

`PyDporEngine` takes a fixed `num_threads` at creation.  If user code spawns
threads dynamically (a common pattern), those threads aren't tracked by the
DPOR engine.  Supporting dynamic thread creation would require the engine to
grow its vector clocks on the fly.

## 8. File include/exclude patterns for tracing

`_tracing.py` uses an automatic heuristic (skip stdlib, site-packages, frontrun
itself).  Users can't control which files to trace.  This matters when:

- Testing code inside third-party libraries the user owns
- The automatic detection misclassifies files (e.g., editable installs)
- Users want to exclude specific modules from tracing for performance

An include/exclude pattern list (glob-based) would help.

## 9. Stop-on-first for DPOR

`explore_dpor` currently explores all remaining interleavings after finding a
violation, collecting all failures.  An early-exit option (`stop_on_first=True`)
would give users who only want one counterexample a faster experience.
(See corresponding bug fix.)
