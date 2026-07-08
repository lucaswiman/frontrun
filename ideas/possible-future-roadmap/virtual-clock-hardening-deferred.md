# Virtual Clock Hardening: Deferred Fixes

Last reviewed: 2026-07-08.

Status: open. These are the items the `virtual-clock-hardening` wave deliberately
deferred after fixing the reproduction and false-positive bugs it set out to
close. Each is scoped, understood, and non-blocking; several have DEFERRED
comments at the exact code sites. None is a missed-bug hazard — the costs are
extra branches, possible false race reports, best-effort determinism, or
maintainability debt. The shipped limitations are documented in
`docs/virtual_clock.rst`; this file tracks the fixes.

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
registration. Reachability is defensive today (no test hits the no-op branch),
so the two live DEFERRED comments mark it rather than a fix.

**Fix directions.** Record an effective-advance flag or target per actor step in
the trace so replay can distinguish real, owed, and no-op advances; *or* prevent
exploration from committing no-op actor steps at all (needs engine support to
un-schedule a step already offered). The trace-format change is invasive and
belongs to a later trace/refactor wave. See the DEFERRED comments at both sites.

## 2. Explored-clock advance past intermediate deadlines (random strategy)

**Symptom.** With `strategy="random"`, `clock="explored"`, a timed wait can
observe a *later* deadline's clock value. A deadline event can therefore fire at
the wrong clock value.

**Mechanism.** After a wait times out correctly at its own deadline, the next
scheduling point's speculative "maybe advance" (`_advance_clock_to(target)` in
`async_shuffler.py`) can jump to a *later* sleeper's deadline before the just-woken
waiter reads `monotonic()`. `_advance_clock_to` firing an earlier pending
deadline while the clock already reads the later target means the earlier
deadline event fires at the later value. Whether post-wait "time may pass between
statements" is legitimate explored semantics is itself the open question.

**Fix directions.** First decide whether post-wait advance is intended explored
behavior. If deadlines must each fire at their own value, clamp the advance to
`min(target, next_deadline())` (prototyped, drops in cleanly, needs an isolating
test). Complementarily, make the explored maybe-advance yield to the woken waiter
before any further advance.

## 3. Raw loop-timer diagnostics

**Symptom.** A `time.monotonic()`-derived absolute deadline passed to
`loop.call_at` inside an explored task silently never fires; the run dies by the
wall-clock watchdog with no diagnostic pointing at the cause (the companion
limitation is documented in `docs/virtual_clock.rst`, "Hazard: virtual-derived
deadlines in raw loop-timer APIs").

**Mechanism.** `loop.time` is pinned to real monotonic, but virtual-derived
`when` values land ≈ `VIRTUAL_EPOCH` seconds in the loop's real-time future. The
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

## 9. `notify_all` no-context redundancy (minor)

**Symptom.** `real_condition.notify_all` followed by a delegate to `notify(len)`
re-touches already-done futures.

**Mechanism.** With no scheduler context the notify_all path double-handles
served waiters.

**Fix direction.** Simplify to direct cooperative wakes instead of routing
through `notify(len)`.

## 10. Happens-before edges for `Condition` / `Queue` wakes

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

## 11. Cross-test SQL connection-thread teardown leak

**Symptom.** `tests/test_async_shuffler_timeout.py::test_detect_sql_reports_table_accesses`
intermittently errors at teardown (a non-daemon `_connection_worker_thread` is
still alive) when run after other tests; it passes in isolation.

**Mechanism.** Pre-existing and unrelated to the clock — a SQL connection worker
thread outlives the test under some run orders.

**Fix direction.** Track separately so it does not get pinned on future clock
work; ensure the connection worker thread is joined/torn down deterministically
at test end.
