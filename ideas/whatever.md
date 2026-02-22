# Missing Features / Ideas

## 1. All-threads-waiting instant deadlock detection for sync schedulers

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

## 2. Async DPOR

DPOR is the most powerful exploration approach but only supports threaded code.
An async variant would let users of `asyncio` benefit from systematic exploration
instead of random sampling.

## 3. Schedule shrinking / minimization

When a bug is found, counterexample schedules can be hundreds of entries long.
Most entries are irrelevant — only a few key ordering decisions trigger the bug.
A schedule minimizer (binary search for minimal reproducing schedule) would make
debugging much easier.

The bytecode `schedule_strategy` already integrates with Hypothesis, which has
built-in shrinking, but the interaction with threading makes shrinking unreliable
(the docstring itself recommends `phases=[Phase.generate]` to skip shrinking).

## 4. Progress reporting / callbacks

DPOR can explore thousands of interleavings over minutes.  There's currently no
callback or progress reporting mechanism.  A simple callback would help:

```python
result = explore_dpor(
    ...,
    on_progress=lambda explored, total_estimate: print(f"{explored} explored"),
)
```

## 5. `threading.Barrier` in cooperative primitives

`_cooperative.py` covers Lock, RLock, Semaphore, BoundedSemaphore, Event,
Condition, and all Queue variants.  `threading.Barrier` is absent.  User code
that uses `Barrier` will deadlock under the scheduler because the real `Barrier`
blocks in C.

## 6. Dynamic thread creation in DPOR

`PyDporEngine` takes a fixed `num_threads` at creation.  If user code spawns
threads dynamically (a common pattern), those threads aren't tracked by the
DPOR engine.  Supporting dynamic thread creation would require the engine to
grow its vector clocks on the fly.

## 7. File include/exclude patterns for tracing

`_tracing.py` uses an automatic heuristic (skip stdlib, site-packages, frontrun
itself).  Users can't control which files to trace.  This matters when:

- Testing code inside third-party libraries the user owns
- The automatic detection misclassifies files (e.g., editable installs)
- Users want to exclude specific modules from tracing for performance

An include/exclude pattern list (glob-based) would help.
