# Fault-Point Exploration: Injected Exceptions and Cancellation as Schedulable Events

**Status:** Proposal (2026-06-12). Not implemented.

## Problem statement

Frontrun explores *orderings*; the other half of concurrency bugs is *partial failure*.
Code that is race-free under every interleaving can still corrupt state when a thread dies
between two writes, a coroutine is cancelled at an await point inside a critical section,
or a DB call raises mid-transaction. Today none of frontrun's strategies can ask: "what if
this step *fails* instead of succeeding?"

`asyncio` cancellation-safety is the highest-value target: cancellation can occur at any
await point, almost no library tests it systematically, and frontrun already interposes on
every await point.

## Key idea

Treat a fault as an extra transition available at a scheduling point. At any point where
the scheduler grants a thread/task a turn, the explorer may instead deliver a fault:

- **async:** raise `asyncio.CancelledError` (or a user-supplied exception) inside the task
  at that await point
- **sync:** raise an injected exception in the thread at that scheduling point (lock
  acquire, SQL execute, marker)

Bound the search with a **fault budget** *k* (default 1, like crash-consistency testers):
explore schedules with at most *k* injected faults. With budget 1 the added search space is
linear in the number of scheduling points per schedule.

## Why the existing machinery makes this cheap

All grant paths already funnel through a small number of choke points:

- **Async:** every await point goes through `_AutoPauseIterator` →
  `InterleavedLoop.pause(task_id, marker)` (`frontrun/async_scheduler.py`). The pause
  executes *inside the paused coroutine*, so raising `CancelledError` from `pause()` when
  the schedule says "inject here" propagates exactly like a real cancellation arriving at
  that await — `finally` blocks, `except CancelledError`, and async context managers all
  behave authentically. No task handle surgery needed.
- **Sync DPOR:** every step goes through `DporScheduler.report_and_wait(frame, thread_id)`
  (`frontrun/_dpor_runtime/scheduler.py`). On grant, check an injection plan and raise in
  the target thread before the opcode executes. Lock acquisition points additionally pass
  through `before_sync_retry()` and `CooperativeLock.acquire` (`frontrun/_cooperative.py`).
- **SQL:** `_dpor_schedule_and_suppress_sync()` (`frontrun/_sql_cursor.py`) already forces
  a scheduling point before each `execute()` — the natural site for simulated driver errors
  (`OperationalError` before the statement, or after it but before the result is consumed).
  Its existing exception path already releases DPOR row locks.

## Built-in invariants (checked even without a user invariant)

A fault run "passes" only if, after the fault propagates and surviving threads finish:

1. **No leaked locks** — no `CooperativeLock` still owned by the faulted thread
   (owner tracking already exists), and `RowLockRegistry` has no locks held by it.
2. **No dangling transaction** — `_sql_transactions.py` TX state for the faulted
   connection is committed or rolled back, not open.
3. **No scheduler hang** — surviving threads can still run to completion (a faulted thread
   holding a patched lock forever is itself the bug being hunted).
4. User invariant, if provided, evaluated as usual.

## API sketch

```python
from frontrun import explore

result = explore(
    strategy="bytecode",          # v1: bytecode + marker strategies
    threads={...},
    invariant=...,
    faults=FaultPlan(
        budget=1,
        kinds=("cancel",),         # async; ("exception", SomeError) for sync/SQL
        at=("await", "lock_acquire", "sql"),   # injection point classes
    ),
)
```

`FaultPlan` lives in `frontrun/common.py`; strategies that support it consume it, others
raise. Register any new strategy variant as a `Strategy`/`AsyncStrategy` adapter in
`frontrun/_strategy.py` per the existing convention.

## Phasing

1. **Async cancellation sweep (v1, highest value/effort ratio).** No new exploration
   machinery: enumerate await points by running once fault-free and counting pauses per
   task, then re-run once per (task, pause-index) injecting `CancelledError` at that pause.
   Works for both `async_shuffler` and `async_dpor` paths since both share
   `_async_autopause.py`. Deliverable: `explore(strategy="async-cancellation", ...)`.
2. **Sync exception injection at marker/lock/SQL points** with the same enumerate-then-
   inject loop, plus the built-in lock/transaction invariants above.
3. **Interleaving × fault product.** Combine with random schedule exploration (Hypothesis
   draws the injection point alongside the schedule) so faults can land in *adversarial*
   orderings, not just the baseline one.
4. **DPOR integration (deferred).** Making the fault a first-class transition in the Rust
   engine (so wakeup trees reason about fault/no-fault branches) is sound but invasive —
   an injected fault changes which transitions are enabled afterward. Only worth it if
   phases 1–3 show DPOR-only races that fault sampling misses.

## Test plan (red/green)

- Red: an async resource pool whose `acquire()` leaks a slot when cancelled between the
  semaphore acquire and the bookkeeping write — assert phase-1 sweep finds it.
- Red: a sync function that acquires a `Lock`, then calls a function that raises mid-way,
  with a missing `finally: release` — assert built-in leaked-lock invariant fires.
- Red: SQL transfer that opens a transaction, UPDATEs one row, then gets an injected
  driver error before the second UPDATE and never rolls back — assert dangling-TX
  invariant fires.
- Green for all three after adding the obvious `try/finally`.

## Open questions

- Exception identity for sync injection: a dedicated `FrontrunInjectedError` is easy to
  filter from "real" failures, but realistic bugs often depend on the *actual* exception
  type (e.g. `except OperationalError` handlers). Probably: default to
  `FrontrunInjectedError`, allow override per injection-point class.
- Should an injected fault in one thread count as test failure if uncaught? Proposal: no —
  uncaught injected faults terminate that thread "normally" for exploration purposes; only
  invariant violations fail.
- Reporting: the failure report should name the injection point (source line of the await
  / lock / SQL statement) alongside the schedule, reusing the existing counterexample
  formatting in `frontrun/_dpor_core/failures.py` / `_report.py`.
