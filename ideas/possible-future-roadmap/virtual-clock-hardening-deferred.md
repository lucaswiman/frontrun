# Virtual Clock Hardening Follow-Ups

Last reviewed: 2026-07-09.

Status: open. These are scoped, non-blocking virtual-clock limitations and
cleanup items. None is a missed-bug hazard; the costs are extra branches,
possible false race reports, best-effort determinism, replay-accounting debt, or
maintainability debt. The shipped limitations are documented in
`docs/virtual_clock.rst`.

## 1. Spurious clock-actor steps vs replay accounting

**Symptom.** A recorded schedule can contain a clock-actor step that reproduction
mis-handles: an owed advance mis-fires at the next deadline registration and
instantly expires it, diverging the reproduction from the explored run.

**Mechanism.** During exploration the engine can commit a clock-actor step to the
trace when no deadline is actually due — the advance is a no-op and the actor
re-blocks (`_advance_virtual_clock_locked` in `_dpor_runtime/scheduler.py`, the
"spurious actor step" branch). Replay (`_ReplayDporScheduler._schedule_next` in
the same file) treats *every* recorded actor entry as either a real advance (a
deadline is pending) or an owed advance (positional drift — sleeper not yet
registered), and cannot tell the no-op apart. For a no-op step exploration
advanced nothing, but replay's owed advance fires at the next deadline
registration.

**Fix directions.** Record an effective-advance flag or target per actor step in
the trace so replay can distinguish real, owed, and no-op advances; *or* prevent
exploration from committing no-op actor steps at all (needs engine support to
un-schedule a step already offered). The trace-format change is invasive and
belongs with a later trace/refactor pass.

## 2. Post-wake reads can observe a later deadline's clock value (explored clock)

**Fixed half (2026-07-09).** The speculative "maybe advance" no longer jumps the
clock *past* an earlier pending deadline: every explored-mode hop is clamped to
`min(target, next_deadline())`, so each deadline fires at its own clock value on
both the sync and async random strategies (`OpcodeScheduler.wait_for_turn` in
`bytecode.py` — where a clamped hop also consumes the sleeper's contiguous
schedule entries so the woken waiter can be granted a turn — and
`AwaitScheduler.should_proceed` in `async_shuffler.py`). Regressions:
`test_virtual_clock.py::test_explored_maybe_advance_clamps_to_earliest_pending_deadline`
(+ explore-level seed-pinned companion) and their async twins in
`test_async_virtual_clock.py`.

**Remaining open half.** A *woken* waiter can still observe a later clock value:
after its deadline fires correctly at its own value, another advance (a later
sleeper's maybe-advance hop, or an autojump) may run before the waiter reads
`monotonic()`, so the post-wake read sees the later value. The same effect
exists under DPOR explored clocks via repeated clock-actor scheduling — the
engine may legitimately schedule further clock-actor steps between the wake and
the read, which is arguably exactly the "time passes between statements"
semantics explored mode is for. Whether post-wake advance is intended explored
behavior is the open design question.

**Fix directions (if post-wake reads must be pinned).** Make the explored
maybe-advance (and the DPOR clock actor) yield to woken-but-not-yet-run waiters
before any further advance — e.g. defer speculative hops until every actor woken
by a previous advance has taken a turn. Needs a decision on semantics first,
since it removes real interleavings from the explored space.

## 3. Raw loop-timer diagnostics

**Symptom.** A `time.monotonic()`-derived absolute deadline passed to
`loop.call_at` inside an explored task is in the wrong clock domain; the callback
can run at the wrong wall-clock time or the run can die by the wall-clock
watchdog with no diagnostic pointing at the cause.

**Mechanism.** `loop.time` is pinned to real monotonic, but virtual-derived
`when` values are compared against the loop's real-time clock. The
frontrun timer-tagging wrapper (`_install_frontrun_timer_tagging` in
`async_scheduler.py`) already sees every `call_at` / `call_later` and tags
frontrun's own watchdog timers, so it is positioned to notice untagged timers.

**Fix directions.** Add a `clock_diagnostics`-gated warning when an *untagged*
loop timer is scheduled with a `when` implausibly far from `loop.time()` (beyond
some threshold). Cheap, since the wrapper already intercepts the call. Needs a
false-positive think-through for legitimate long-lived timers.

## 4. `sleep_until` watchdog abort coverage

**Symptom.** The sync and async `sleep_until` watchdog-timeout branches wake
parked waiters via the shared abort hook, but have no deterministic end-to-end
regression tests — only the `_handle_timeout` path does.

**Mechanism.** Forcing a phase-1 / phase-2 `sleep_until` watchdog timeout
deterministically is hard, so the wake-on-abort behavior there is covered only
by inspection and by the shared `_on_error_set` hook (`async_dpor.py`) the other
abort paths exercise.

**Fix directions.** If a deterministic harness for forcing `sleep_until`
watchdog timeouts becomes available, add direct regressions for both phases.

## 5. Cooperative spin-loop consolidation (`_spin_until`)

**Symptom.** Roughly seven structurally-divergent spin loops in `_cooperative.py`
resist a single-driver rewrite, and a redundant double
`_timed_acquire_cleanup(gave_up=False)` call remains.

**Mechanism.** The loops differ enough (turn-holding vs ticket system vs
try/except probes vs `is_set`) that a common `_spin_until` driver did not fall
out. The cleanup redundancy is entangled with deadline lifetime versus the 1-second
real-wait fallback, so removing it is not purely mechanical.

**Fix directions.** Revisit alongside any future sync/async primitive
unification, when a shared spin driver and deadline-lifetime model can be
designed together rather than retrofitted.

## 6. Remaining sync/async clock divergences

**Symptom.** Sync and async clock paths still diverge in several places — a
deliberate 80/20 stop for the unification wave.

**Mechanism / fix direction, per site.**

- `sleep_until` two-phase loops use different wait primitives and different abort
  bookkeeping (replay owed-advance vs `_on_error_set`); unify the phase model.
- Dispatch entry differs (`VirtualClockPort.advance_clock_to` with spin-flag pops
  vs `advance_and_dispatch`); collapse to one entry point.
- Async keeps a redundant `_sleepers` dict alongside coordinator membership;
  collapsing it changes the phase-1 hot-loop condition, so do it with care.
- Replay owed-advance extraction is duplicated; hoist into the shared port.
- `mark_done` scrub logic differs; share it.

## 7. `_PathPinnedEngine` adapter

**Symptom.** Async `_on_opcode` still swaps a public engine attribute
(`_PathPinnedEngine` in `async_dpor.py`) as a way of passing the path id, rather
than threading it as a parameter.

**Mechanism.** The clean form threads `path_id` through the access-report call,
but that requires `*_at` report variants (`report_at`-style) in the shared
`_opcode_observer` API, which sync also uses.

**Fix direction.** Add the `*_at` report variants to `_opcode_observer` and pass
`path_id` explicitly on both sync and async paths, retiring the attribute swap.

## 8. Row-lock modeling as an explicit capability

**Symptom.** "Row locks are modeled" is inferred rather than declared: a `None`
return means acquired-all, `[]` means none. `SELECT ... FOR UPDATE` downgrade
decisions (`weak_read`) key off `dpor_ctx is not None` as a proxy.

**Mechanism.** There is no capability flag on the scheduler protocol saying "this
scheduler models row locks", so callers overload return-value shapes and context
presence.

**Fix directions.** Add a capability flag to the scheduler protocol so downgrade
decisions stop inferring it. Adjacent cleanups live in
`_sql_endpoint_suppression` (text-keyed suppression, execution-lifetime scope,
and mogrified-parameter mismatch produce under/over-suppression asymmetries) and
could be addressed in the same pass.

## 9. Happens-before edges for sync `Condition` / `Queue` wakes

**Symptom.** DPOR may explore, or flag as racy, some `Condition` / `Queue`
orderings that are actually ordered (companion to the documented limitation in
`docs/virtual_clock.rst`). Extra branches and possible false race reports, never
missed bugs.

**Mechanism.** A `notify` / `put` wake goes through the spin-release path
(`_note_spin_release` in `_cooperative.py`), which only clears the waiter's
blocking-spin flag — it reports no happens-before edge. `Event` reports
`event_set` / `event_wait` edges; sleeper and timed-wait wakes report
release/acquire edges; `Condition` / `Queue` wakes do not.

**Fix direction.** Report per-waiter release/acquire edges on notify / put wakes
the way `Event` does (the `event_wake_sync_id` pattern), eliminating the false
race branches. Needs stable wake ids and replay compatibility.

## 10. `clock="explored"` never fires a timeout-kind deadline against a runnable holder

**Symptom.** Under `clock="explored"`, "timer fires before the holder's next
step" is explored for *sleep* deadlines but not for *timeout*-kind deadlines
(timed `lock.acquire(timeout=)` in sync, `asyncio.wait_for` in async). If the
lock holder / event setter is runnable and completes in zero virtual time, the
timeout branch is never explored and `property_holds=True` is a false negative
for a reachable outcome. Holders that sleep or block across virtual time are
unaffected (the timeout fires correctly). Encoded as a strict xfail:
`tests/test_virtual_clock.py::test_explored_clock_finds_timed_acquire_timeout_against_runnable_holder`.

**Mechanism.** The clock actor's sleep wake reports a release/acquire
happens-before pair (`report_clock_sleep_wake`), and the woken sleeper's
subsequent memory ops race with other workers' ops, seeding the
"timer fires first" branch in the wakeup tree. The timeout wake deliberately
carries no engine-visible event (`_on_clock_wake`'s `timeout` branch just
calls `execution.unblock_thread`; see the comment there): the waiter
re-reports `lock_wait` before it can observe expiry. That makes the actor's
timeout step commute with every worker step, so no reversal is ever seeded —
verified independent of `preemption_bound` (2, 10, None all explore the same
3 interleavings for the two-worker repro).

**Fix direction.** The firing must be made *dependent* on the operations that
can beat it — most naturally the holder's `lock_release` on the same resource
(the timeout and the release genuinely race: distinct final states). Options:
report a sync event from the actor on the timed-wait's `wake_sync_id` at fire
time paired with an acquire-half on the waiter's give-up path (mirroring the
sleep-wake pattern, but requiring the timed-acquire retry loop to
distinguish "woke to retry" from "woke to expire"), or report a synthetic
access by the actor to the contended lock's object id. Either changes
engine-visible semantics of every timed wait, so it needs wakeup-tree /
replay validation (and the TLA spec) — deferred rather than patched
pre-release.
