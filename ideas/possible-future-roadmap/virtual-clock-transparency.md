# Virtual Clock Transparency Follow-ups

Last reviewed: 2026-07-07.

Status: phases 1-5 implemented. The remaining open item is Python
cross-process virtual clocks, which stays exploratory until a real workload
needs it.

## Summary

The transparency follow-ups that fit frontrun's in-process white-box model are
implemented:

- a shared deadline coordinator;
- virtual `asyncio.wait_for` / `asyncio.timeout` wrappers;
- engine-visible async `Queue` and `Condition` waiters;
- module-qualified `datetime` support;
- opt-in captured `time.*` diagnostics via `clock_diagnostics=True`.

The release-grade contract remains intentionally scoped: virtual time covers
scheduler-visible Python code, not every possible source of time in the process.

## Goals

- Make common timeout code work without users rewriting tests around
  `time.sleep()` / `asyncio.sleep()`.
- Preserve replayability: every timer firing in a counterexample must be
  represented by a deterministic schedule event or by documented autojump
  semantics.
- Avoid false deadlocks: framework-internal or user wall-clock timers must not
  be mislabeled as exact virtual-clock deadlocks.
- Keep unsupported boundaries explicit. Silent partial virtualization is worse
  than a clear limitation.

## Non-goals

- No transparent control of unmodified non-Python workers. That remains outside
  frontrun's white-box scope.
- No best-effort mutation of arbitrary already-captured function objects. A
  reference captured before frontrun patches `time` cannot generally be replaced
  safely.
- No broad C runtime time virtualization in the core scheduler. Intercepting
  `nanosleep`, `select`, `poll`, database-driver sleeps, etc. belongs behind a
  separate integration boundary if ever pursued.

## Phase 1: Consolidate Deadline State

Status: implemented. Complexity: medium. Priority: high for maintainability.

Today the virtual-clock state is duplicated across sync DPOR, sync random,
async DPOR, async random, and replay schedulers: sleepers, timed waits, spin
waiters, clock-actor enabledness, and replay drift all have similar but not
identical code paths.

Introduce a small internal deadline coordinator, owned by each scheduler:

```python
class DeadlineCoordinator:
    def add_sleep(actor_id: int, deadline: float, wake_id: int | None) -> None: ...
    def add_timeout(actor_id: int, deadline: float, token: object) -> None: ...
    def cancel(actor_id: int, token: object | None = None) -> None: ...
    def next_deadline(self) -> float | None: ...
    def advance_to_next(self) -> list[WakeEvent]: ...
    def has_pending(self) -> bool: ...
```

The scheduler still owns engine blocking/unblocking and happens-before reports.
The coordinator only owns deadline ordering and due-event calculation.

Acceptance tests:

- Existing virtual-clock suites remain green.
- Equal-deadline sleepers wake deterministically but their post-wake work stays
  raceable.
- Replay drift tests still reproduce without burning `deadlock_timeout`.
- Sync and async schedulers use the same due-deadline ordering rules.

## Phase 2: Virtual `asyncio.wait_for` and `asyncio.timeout`

Status: implemented. Complexity: medium-high. Priority: high user value.

Current behavior is deliberately conservative: event-loop timers stay on the
wall clock so frontrun's scheduler watchdogs do not fire when virtual time
jumps. That avoids false deadlocks, but it is a user papercut because
`asyncio.wait_for` is the standard timeout surface.

Patch `asyncio.wait_for` and `asyncio.timeout` during exploration instead of
virtualizing `loop.time()`.

Design sketch:

- Keep `loop.time()`, `loop.call_later`, and raw loop timers real.
- For `asyncio.wait_for(awaitable, timeout=t)` inside an explored task:
  - wrap the awaitable in a task if needed;
  - register a virtual timeout deadline with the active scheduler;
  - race two scheduler-visible events: inner completion and timeout deadline;
  - when the virtual deadline wins, cancel the inner task and raise
    `asyncio.TimeoutError` / `TimeoutError` with normal Python semantics;
  - when inner completion wins, cancel the virtual deadline.
- For `asyncio.timeout(t)`, implement the same virtual deadline around the
  current task's cancellation scope.
- Preserve wall-clock behavior outside explored task contexts.

DPOR semantics:

- In `clock="virtual"`, the timeout fires only when no real actor can proceed
  before the deadline.
- In `clock="explored"`, timeout firing is a clock-actor step, so DPOR explores
  "operation completed before timeout" and "timeout fired first" when both are
  schedulable.
- Timeout wake/cancel should have a stable sync id so replay performs the same
  timeout at the same schedule point.

Acceptance tests:

- `await asyncio.wait_for(asyncio.sleep(10), timeout=1)` consumes one virtual
  second and no meaningful wall time.
- A task waiting on a patched `asyncio.Event` can either be set before the
  virtual timeout or time out first under `clock="explored"`.
- A timed-out inner task is cancelled exactly once and does not leak into later
  replay attempts.
- Existing user wall-clock loop callbacks still prevent exact deadlock
  classification until they fire or are cancelled.

## Phase 3: Engine-visible Async Queue and Condition Waiters

Status: implemented. Complexity: medium. Priority: medium.

Async DPOR currently patches `asyncio.Lock` and `asyncio.Event`, but raw
`asyncio.Queue`, `asyncio.Condition`, and bare futures are not engine-visible.
They behave correctly in many tests, but deadlocks through them fall back to
wall-clock detection, and virtual-clock autojump has to be conservative.

Add cooperative wrappers for:

- `asyncio.Queue.get` / `put` waiters,
- `asyncio.Condition.wait` / `notify` / `notify_all`,
- optionally `asyncio.Semaphore` if it shows up in real workloads.

Semantics:

- Waiters block in the engine when the primitive cannot proceed.
- Producers/notifiers unblock waiters and report stable wake happens-before
  edges.
- Timeout support should use Phase 2's virtual async timeout mechanism rather
  than loop timers.

Acceptance tests:

- Queue get/get and put/put deadlocks are reported exactly under
  `clock="virtual"` with no pending deadlines.
- Queue producer delayed by `asyncio.sleep()` autojumps and wakes a blocked
  consumer without burning `deadlock_timeout`.
- Condition `notify(1)` wakes exactly one async waiter and does not create a
  broadcast-style false schedule.

## Phase 4: `datetime` Support

Status: implemented. Complexity: medium. Priority: medium.

Many TTL/rate-limit implementations use `datetime.datetime.now()` or
`datetime.date.today()` instead of `time.monotonic()`. Supporting these is
achievable but should be explicit because the standard `datetime` C types are
not as simple to monkey-patch as module-level `time` functions.

Design sketch:

- During virtual-time patch scopes, replace `datetime.datetime` with a subclass
  whose current-time reads (`now()`, `utcnow()`) use the active virtual clock.
  Leave `fromtimestamp(ts, tz=...)` as the normal deterministic conversion from
  its supplied timestamp.
- Replace `datetime.date` with a subclass for `today()` if needed.
- Preserve timezone semantics for `datetime.now(tz=...)`.
- Return instances that behave like normal `datetime` objects and pass common
  arithmetic/comparison operations.
- Document that `from datetime import datetime` captured before the patch still
  bypasses the virtual clock.

Acceptance tests:

- `datetime.datetime.now()` advances after virtual `sleep`.
- `datetime.datetime.now(tz=timezone.utc)` returns an aware value derived from
  virtual time.
- `datetime.date.today()` follows virtual time when supported.
- Existing `time.*` patch restoration tests also cover `datetime` restoration.

## Phase 5: Captured-reference Diagnostics

Status: implemented. Complexity: low-medium. Priority: medium.

Fully fixing pre-existing captured references is not generally possible, but
frontrun can make the failure mode less surprising.

Possible mitigations:

- Add a docs section with a clear bad/good example:
  `from time import monotonic` before exploration bypasses the virtual clock;
  `import time; time.monotonic()` works.
- Add an optional diagnostic mode that scans traced frames for globals/defaults
  equal to the saved real `time` functions and emits a warning when
  `clock != "real"`.
- Consider a CLI/pytest prepatch mode for tests that need imports to happen
  under virtual-time wrappers. This should be opt-in because global time
  prepatching can affect test harness code.

Acceptance tests:

- Diagnostic mode warns when a worker uses a default argument captured from
  `time.monotonic`.
- No warning is emitted for module-qualified `time.monotonic()`.
- Diagnostics do not change scheduling behavior.

## Phase 6: Python Cross-process Virtual Clock

Status: exploratory. Complexity: high. Priority: low until a real workload
needs it.

Cross-process exploration currently rejects virtual clocks because worker
processes read their own real clocks. A Python-only version is achievable:
worker processes already run frontrun code and coordinate with the parent, so
sleep/time operations could relay deadline requests to the coordinator.

Design sketch:

- Worker-side time patches send clock-read and sleep-deadline operations to the
  coordinator.
- The coordinator owns the virtual clock and includes deadline unblock events
  in the process DPOR schedule.
- Replay sends the same clock advances to the same workers.

Risks:

- Higher protocol complexity and more round trips.
- External stores may have their own real-time behavior, such as database lock
  timeouts, that cannot be virtualized by Python worker patches.

Acceptance tests:

- Two Python worker processes with `time.sleep()` use virtual time.
- A cross-process TTL race over SQLite/Redis can be reproduced with a recorded
  schedule.
- Process-mode tests still reject virtual clocks for unsupported unmodified or
  non-Python worker paths.

## Release Guidance

The release notes should keep the scoped contract:

- `clock="virtual"` and `clock="explored"` control scheduler-visible Python
  timing.
- Raw async loop timers remain wall-clock; virtual async timeouts are provided
  by patched `asyncio.wait_for` / `asyncio.timeout`.
- Captured references, process workers, bare futures, and C-level sleeps remain
  documented limitations.
