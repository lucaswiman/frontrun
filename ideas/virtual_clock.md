# Virtual Clock: Making Timeout, Retry, and TTL Races Explorable

**Status:** Implemented (2026-07-05) — `frontrun.explore(clock="virtual"|"explored")`,
sync + async, DPOR + random. See `docs/virtual_clock.rst`. Implementation deltas
from this proposal:

- One mechanism serves both modes: the clock actor exists in v1 too, but is
  *enabled* only when nothing else is runnable (autojump) vs. whenever a
  deadline is pending (explored). Wake edges are reported as
  `lock_release` (actor) / `lock_acquire` (woken worker) sync events.
- The async `loop.time()` spike concluded **against** virtualising loop time:
  the scheduler's own deadlock-timeout timers share the loop's timer heap, so
  a clock jump would fire them spuriously. `asyncio.wait_for` / `asyncio.timeout`
  are patched directly inside explored task contexts instead.
- Sync cooperative timed waits now use virtual deadlines: lock/RLock/semaphore
  acquires, `Event.wait(timeout=...)`, `Condition.wait(timeout=...)`,
  `Condition.wait_for(..., timeout=...)`, and `Queue.get`/`put` timeouts.
  Async `asyncio.wait_for` and `asyncio.timeout` use virtual deadlines.
- `serializable_invariant` is rejected with a virtual clock (the sequential
  baseline runs execute outside the scheduler).
- Post-review hardening: cooperative `Event.wait()` engine-blocks under DPOR
  (a spinner is indistinguishable from useful work, so branches scheduling
  the waiter before the setter ran unboundedly); the random scheduler tracks
  untimed lock/event spinners so autojump still fires; `clock_scope` owns the
  `time.*` patch so invariants see virtual time; replay defers recorded
  clock-actor steps that arrive before their sleeper registers (drift);
  async random uses a quiescence heuristic for tasks parked on unpatched
  asyncio primitives. Async `Queue` and `Condition` waiters are now patched;
  bare futures remain outside exact detection.
- Exact deadlock detection covers async DPOR too: `asyncio.Event` is patched
  (waiters engine-block with wake happens-before edges, like `asyncio.Lock`),
  and a "nobody runnable, no deadline pending" observation is confirmed after
  draining in-flight loop wakes and checking that no *user* loop timer is
  pending (frontrun's own watchdog timers are tagged via a `loop.call_at`
  wrapper). `asyncio.Queue` and `asyncio.Condition` waiters are now managed;
  bare futures still look runnable to the engine, so the check safely declines
  and the wall fallback applies.

## Problem statement

Races involving timeouts, retries with backoff, TTL caches, debouncing, and rate limiters
are invisible to frontrun today because `time.time()` / `time.monotonic()` / `sleep()` are
real. "The retry fired exactly between the read and the write" is not an interleaving the
scheduler can choose — it is wall-clock luck. The tenacity and cachetools case studies both
sit squarely in this space, so validation targets already exist in-repo.

Concretely, wall-clock time leaks into exploration in three places today:

1. `CooperativeLock.acquire(timeout=...)` busy-waits against a `time.monotonic()` deadline
   (`frontrun/_cooperative.py`) — whether a timed acquire succeeds depends on host speed.
2. User code calling `time.sleep()` / `asyncio.sleep()` actually blocks, slowing
   exploration and introducing nondeterminism.
3. User code *reading* clocks (TTL expiry checks) sees real time, so expiry never happens
   inside a test, or happens flakily.

(The scheduler's own `deadlock_timeout` waits are *not* in scope — they guard against
threads stuck in unmanaged C code, which a virtual clock cannot see. They stay wall-clock.)

## Design

### Core: clock owned by the scheduler, advanced only when nothing is runnable

A `VirtualClock` (start at an arbitrary epoch, e.g. 1_000_000.0) lives on the scheduler.
Patched `time.time` / `time.monotonic` / `time.perf_counter` return it; patching is gated
on scheduler TLS context (same `get_context()` gating `_cooperative.py` already uses), so
non-explored threads see real time. Use the `frontrun/_patching.py` toolkit.

`sleep(d)` becomes a **timed block**: the thread registers a deadline `now + d`, reports a
scheduling point, and is marked blocked (the engine already supports
`execution.block_thread()` / `unblock_thread()` via `frontrun/_dpor_core/engine.py`).

The clock advances only when it must: when every live thread is either finished or
deadline-blocked, jump the clock to the **earliest pending deadline** and unblock the
threads it wakes. This is exactly Trio/AnyIO's "autojump clock" (prior art:
`trio.testing.MockClock(autojump_threshold=0)`), and it composes with deadlock detection:
all threads blocked *and no pending deadlines* = genuine deadlock, reported exactly instead
of via the 5-second wall fallback.

This autojump-only model is deterministic and is **v1**. It already fixes (1)–(3): timed
lock acquires become "blocked until deadline-or-lock", sleeps cost zero wall time, and TTL
expiry is reachable by sleeping past it.

### v2: time advancement as an explored choice (the interesting part)

Autojump always advances time as *late* as possible, so it explores only one timing.
The race we actually want — "timer fires between your read and your write" — requires the
clock advance itself to be schedulable. The cheap, sound way to do that with zero new
engine machinery:

**Model the clock as one extra DPOR actor.** Register a synthetic thread (id N) whose only
enabled transition, whenever at least one deadline is pending, is "advance clock to the
next deadline and unblock its sleepers". The Rust engine then explores orderings of this
clock-step against other threads' steps like any other interleaving choice, and vector
clocks handle the rest: waking a sleeper is a scheduling event the engine already
understands as block/unblock. No changes to `crates/dpor` — the clock actor is plumbing in
`DporScheduler._schedule_next()` and the runner's thread bookkeeping.

Search-space note: each pending deadline adds at most one clock-step per wake, so the
blowup is modest (comparable to adding one short thread per timer). For bytecode/random
strategies, the equivalent is: at each scheduling point where a deadline is pending, the
random scheduler may pick "advance clock" with some probability.

### Clock *reads* are not scheduling points

Reading the clock is side-effect-free; making every `time.time()` call a DPOR access would
explode the search space for nothing. Only **advancement** (the clock actor's step) and
**deadline wakes** are events. This is the key trade-off that keeps v2 tractable.

### Async

- `asyncio.sleep(d)` → wrapper that registers a deadline with the async scheduler,
  rather than yielding to the real loop timer.
- `loop.time()` stays wall-clock. While `time.monotonic()` is patched for explored
  tasks, the runners pin the event loop's own clock to the saved real monotonic
  function so scheduler watchdogs and user loop timers remain real.
- `asyncio.wait_for` / `asyncio.timeout` are patched directly during
  exploration. Raw loop timers remain wall-clock.

### Timed lock acquires

`CooperativeLock.acquire(timeout=t)` replaces its `time.monotonic()` busy-wait deadline
with a virtual deadline: the thread is blocked-with-deadline, and either the lock is
released (normal wake, acquire retries) or the clock actor advances past the deadline
(acquire returns `False`). Today's "timeout skips the wait-for graph" special case
(`_timed_acquire_state`) can then be removed: a timed acquire genuinely cannot deadlock,
and the virtual clock proves it by construction instead of by wall-clock escape.

## API sketch

```python
import frontrun

result = frontrun.explore(
    strategy="dpor",
    threads={...},
    invariant=...,
    clock="virtual",        # default "real"; "virtual" implies autojump (v1)
    # v2: clock="explored" adds the clock actor
)
```

## Phasing

1. **v1 sync autojump:** patch `time.{time,monotonic,perf_counter,sleep}` under scheduler
   context; deadline-blocked thread state; advance-on-idle; exact deadlock detection for
   time-blocked threads. Validate on a TTL-cache lost-expiry bug (cachetools-style).
2. **v1 async autojump:** wrapped `asyncio.sleep`; keep loop timers on wall-clock time.
3. **v2 clock actor** for DPOR + a "maybe advance" branch for the random strategies.
   Validate on a tenacity-style retry race ("retry fires between read and write").
4. **Timed-acquire migration:** route `CooperativeLock` timeouts through virtual
   deadlines; delete the wall-clock busy-wait.

## Open questions

- Granularity of "earliest deadline": if two deadlines are equal, wake order is itself a
  race — with the clock-actor model this falls out naturally (two unblocks = two explored
  events), but v1 autojump needs a deterministic tiebreak (thread id).
- Should `time.sleep(0)` remain a pure yield (current Python semantics) rather than a
  deadline? Yes — special-case it.
- Third-party clock reads via module-qualified `datetime.datetime.now()` /
  `utcnow()` and `datetime.date.today()` are patched. Captured datetime class
  references still bypass the patch.
- C-level sleeps (e.g. drivers) are invisible; document as out of scope (same boundary as
  the existing `deadlock_timeout` C-code caveat).
