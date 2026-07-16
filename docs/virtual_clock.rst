Virtual clock: timeout, retry, and TTL races
============================================

Races involving timeouts, retries with backoff, TTL caches, debouncing, and
rate limiters are invisible to a scheduler that runs on wall-clock time:
"the retry fired exactly between the read and the write" is not an
interleaving the scheduler can choose — it is wall-clock luck.  The virtual
clock makes time a *scheduled* quantity, so those races become explorable,
deterministic, and replayable like any other interleaving.

Enable it with the ``clock=`` parameter of :func:`frontrun.explore`:

.. code-block:: python

    import time
    import frontrun

    class Cache:
        def __init__(self):
            self.expires_at = time.monotonic() + 60.0

        def expired(self):
            return time.monotonic() >= self.expires_at

    def worker(cache):
        time.sleep(120.0)              # zero wall time under clock="virtual"
        assert cache.expired()         # TTL expiry is actually reachable

    result = frontrun.explore(
        setup=Cache,
        workers=[worker],
        invariant=lambda c: c.expired(),
        clock="virtual",               # default "real"
    )
    assert result.property_holds

Both sync (threads) and async (asyncio tasks) workers are supported, with
``strategy="dpor"`` and ``strategy="random"``.

Rule of thumb: use ``"virtual"`` to make timeout/TTL logic reachable
deterministically at zero wall cost; use ``"explored"`` when the *timing*
of a timer firing is itself the race you are hunting — it makes the clock
advance a scheduling choice the engine explores.

What a virtual clock changes
----------------------------

With ``clock="virtual"`` each execution gets a fresh clock starting at an
arbitrary epoch (1,000,000.0 seconds), owned by the scheduler:

* **Clock reads are virtual.** ``time.time()``, ``time.monotonic()``,
  ``time.perf_counter()`` (and their ``_ns`` variants) return virtual time
  inside explored code.  Module-qualified ``datetime.datetime.now()`` /
  ``utcnow()`` and ``datetime.date.today()`` also read virtual time. Patching
  is gated per thread/context: worker threads and tasks, ``setup()``, and
  the ``invariant`` see virtual time; unrelated threads (pytest machinery,
  background daemons) see real time.
  ``datetime.datetime.fromtimestamp(ts, tz=...)`` is left alone because it is
  a deterministic conversion from the supplied timestamp rather than a clock
  read.
* **Sleeps are timed blocks.** ``time.sleep(d)`` / ``asyncio.sleep(d)``
  with ``d > 0`` register a deadline at ``now + d`` and block until the
  clock reaches it — in zero wall time.  ``sleep(0)`` remains a pure yield,
  matching stock Python semantics.
* **Timed lock acquires are deterministic.** A contended
  ``threading.Lock.acquire(timeout=t)`` registers a virtual deadline
  instead of busy-waiting against the host clock; whether it succeeds no
  longer depends on machine speed.
* **The clock advances only when it must.** When no real worker is runnable
  and at least one virtual deadline is pending, the clock jumps to the
  *earliest* pending deadline and wakes due sleepers.  This is the
  "autojump" model (prior art:
  ``trio.testing.MockClock(autojump_threshold=0)``).
* **Async timeout wrappers are virtual.** Inside explored tasks,
  ``asyncio.wait_for(awaitable, timeout=t)``, ``asyncio.timeout(t)``, and
  ``asyncio.timeout_at(when)`` register virtual deadlines. The event loop's own
  timer heap still stays on wall time; raw ``loop.call_later`` / ``call_at``
  callbacks are not virtualized.
* **Deadlocks are detected exactly (sync and async DPOR).** All workers
  blocked with *no* pending deadline is a genuine deadlock; the DPOR
  schedulers report it immediately instead of via the wall-clock fallback
  timeout.  Async DPOR manages ``Lock``, ``Event``, ``Queue`` and
  ``Condition`` waiters. Bare futures and user-created loop timers remain
  outside the exact model and keep the conservative wall-clock fallback.

``clock="explored"``: timer firings as interleaving choices
-----------------------------------------------------------

Autojump always advances time as *late* as possible, so it explores only
one timing.  The race you usually want — "the timer fired between your
read and your write" — requires the clock advance itself to be a
scheduling choice.

With ``clock="explored"``, the clock is modelled as one extra DPOR actor:
a synthetic thread whose only enabled transition, whenever at least one
deadline is pending, is "advance the clock to the next deadline and wake
its sleepers".  The engine then explores orderings of this clock step
against the workers' steps like any other interleaving, and waking a
sleeper carries a happens-before edge (the actor releases a virtual wake
object; the woken worker acquires it), so DPOR's race reversal knows how
to move timer firings around:

.. code-block:: python

    import time
    import frontrun

    class State:
        def __init__(self):
            self.x = 0

    def rmw_worker(s):          # read-modify-write over two statements
        tmp = s.x
        s.x = tmp + 1

    def delayed_writer(s):      # a retry/timer firing one virtual second later
        time.sleep(1.0)
        s.x = 100

    result = frontrun.explore(
        setup=State,
        workers=[rmw_worker, delayed_writer],
        invariant=lambda s: s.x == 100,
        clock="explored",
    )
    assert not result.property_holds   # found: timer fired inside the RMW window

Under ``clock="virtual"`` the same test passes (the delayed write always
lands last); under ``clock="explored"`` DPOR finds the interleaving where
the timer fires between the read and the write, and the counterexample
replays like any other frontrun schedule — the recorded schedule includes
the clock steps.

For ``strategy="random"``, ``clock="explored"`` gives the sampler a
"maybe advance time" branch: whenever a random schedule entry lands on a
sleeping worker, the clock advances to that worker's deadline and it wakes
early.

Search-space note: each pending deadline adds at most one clock step per
wake, so the blowup is modest — comparable to adding one short worker per
timer.  Clock *reads* are deliberately not scheduling points; only
advancement and deadline wakes are events.

Semantics and limitations
-------------------------

* ``clock=`` requires ``patch_sleep=True`` (the default) and thread/async
  execution; ``execution="process"`` rejects it (worker processes read
  real time).
* ``serializable_invariant`` cannot be combined with a virtual clock: the
  sequential baseline runs execute outside the scheduler, so their sleeps
  and clock reads would use real wall-clock time.
* ``time.time`` and ``time.monotonic`` return the *same* virtual value
  (there is one clock).
* **Raw async loop timers stay on the wall clock.** ``loop.time()``,
  ``loop.call_later`` and ``loop.call_at`` are not virtualized: the
  scheduler's own watchdog timers share the event loop's timer heap, and
  virtualizing it would let a clock jump fire them spuriously. Use
  ``asyncio.wait_for`` / ``asyncio.timeout`` / ``asyncio.timeout_at`` inside
  explored tasks when you want a virtual async timeout. The handle returned by
  ``asyncio.timeout`` keeps asyncio's normal loop-time API for ``when()`` /
  ``reschedule(when)``; pass values derived from ``loop.time()``. frontrun
  converts the remaining delay to a virtual scheduler deadline internally.
* Sync timed waits on frontrun's cooperative primitives use virtual
  deadlines: ``Lock`` / ``RLock`` / ``Semaphore`` acquire timeouts,
  ``Event.wait(timeout=...)``, ``Condition.wait(timeout=...)`` /
  ``wait_for(..., timeout=...)``, and ``Queue.get`` / ``put`` timeouts.
  Async ``wait_for`` / ``timeout`` / ``timeout_at`` deadlines are virtual, and
  async DPOR makes ``Event``, ``Queue`` and ``Condition`` waiters engine-visible
  with wake happens-before edges.
* An async ``Event.set()`` issued from outside the explored tasks (a loop
  callback or a foreign thread) is not an engine-visible wake. Exact-deadlock
  detection therefore declines while a user loop timer is pending or an
  external thread is alive, and the run falls back to the wall-clock watchdog
  if the wake never arrives. Setters inside explored tasks — the normal case —
  are fully tracked and remain eligible for exact deadlock detection.
* **Captured references bypass the patch.** ``from time import monotonic``,
  ``from time import sleep``, ``from datetime import datetime``, or captured
  ``asyncio.sleep`` / ``wait_for`` / ``timeout`` objects taken before
  exploration keep pointing at the real object. Call through the module
  (``time.monotonic()``, ``datetime.datetime.now()``, ``asyncio.wait_for(...)``)
  in code under test. Set ``clock_diagnostics=True`` to warn when traced worker
  frames hold captured real ``time.*`` clock-read functions. Diagnostics do not
  detect captured sleeps, async helpers, or datetime/date classes. They require
  frame tracing, so they apply to DPOR and sync random exploration; async random
  accepts the unified option but cannot inspect frames.
* ``strategy="random"`` with *async* workers cannot see tasks blocked on
  raw ``asyncio`` primitives (e.g. ``asyncio.Lock``); a quiescence
  heuristic advances the clock when nothing has progressed for a short
  interval.  Prefer ``strategy="dpor"`` for lock-heavy async code — it
  patches ``asyncio.Lock`` and needs no heuristic.
* **Dynamically created child tasks are not separate explored actors.** An
  explored async worker should not use ``asyncio.create_task()`` to introduce
  additional concurrency: descendants inherit the worker's scheduler identity,
  so overlapping virtual sleeps or waits can serialize unexpectedly. Pass each
  concurrent coroutine as its own item in ``workers=`` instead.
* **Broken-down and formatted wall-clock time is not patched.**
  ``time.localtime`` / ``time.gmtime`` / ``time.strftime`` (no-arg) /
  ``time.ctime`` / ``time.asctime`` / ``time.clock_gettime`` /
  ``time.clock_gettime_ns`` read the *real* clock even under a virtual clock —
  only ``time.time`` / ``monotonic`` / ``perf_counter`` (and their ``_ns``
  variants) plus module-qualified ``datetime`` current-time reads are
  virtualized (see the ``_PATCHES`` table in ``_virtual_clock.py``). TTL code
  that takes its deadline from ``time.time()`` but formats "now" via
  ``time.localtime()`` / ``strftime`` silently mixes a virtual epoch with a real
  one.
* **The virtual epoch is a fixed, low value.** Each execution starts a fresh
  clock at ``VIRTUAL_EPOCH`` (1,000,000.0 s ≈ 11.6 days), which is below typical
  server uptimes.  A ``time.monotonic()`` delta that straddles the patch
  boundary is therefore meaningless — for example a baseline captured with the
  real clock before exploration and re-read as virtual time inside it, or state
  that persists across executions (every execution gets a *new* clock at the
  same epoch)::

      t0 = time.monotonic()             # real (e.g. 53_000.0), captured pre-explore
      # ... later, inside an explored worker ...
      elapsed = time.monotonic() - t0   # virtual 1_000_000 - real 53_000 -> garbage

  This is inherent to the deterministic fixed-epoch design; keep all timing
  arithmetic within a single execution's clock.
* **Async random + virtual clock is best-effort deterministic.** The async
  random strategy advances the clock with a *wall-clock* quiescence heuristic
  (``_QUIESCENCE_SLICE`` = 0.01 s of no progress in ``async_shuffler.py`` ->
  autojump), so under load (slow I/O, CI GC pauses) the *same* seed can explore
  different interleavings.  Async DPOR's autojump is engine-state-driven and
  stays deterministic; prefer ``strategy="dpor"`` when you need a reproducible
  schedule.
* **Sync cooperative ``Condition`` / ``Queue`` wakes report no happens-before edge.**
  A sync ``notify`` / ``put`` wake goes through the spin-release path
  (``_note_spin_release`` in ``_cooperative.py``), which only clears the
  waiter's blocking-spin flag so it re-probes.  Unlike ``Event`` (whose
  ``set()`` -> ``wait()``-return carries an ``event_set`` / ``event_wait`` edge)
  and sleeper / timed-wait wakes (release / acquire edges), vector clocks
  under-order notify->wake, so DPOR may explore, or flag as racy, some orderings
  that are actually ordered.  Async DPOR Queue/Condition wakeups do report wake
  edges.  The sync limitation costs extra branches and possible false race
  reports, not missed bugs; a fix is tracked in
  ``ideas/possible-future-roadmap/virtual-clock-hardening-deferred.md``.
* **Under ``clock="explored"``, a timeout may beat a runnable lock holder.**
  A timed ``lock.acquire(timeout=...)`` is a real schedulable race even when
  the holder would release in zero virtual time. frontrun explores both the
  release-first and timeout-first outcomes; code that appears "obviously
  fast enough" on a wall clock can therefore produce an honest timeout
  counterexample. This is deliberate explored-clock semantics, not a claim
  about likely production latency.
* **Under ``clock="explored"``, a woken timed wait may read a later clock
  value than its own deadline.** Every deadline *fires* at its own clock
  value, but "timer fires" and "waiter reads the clock" are separate steps
  under an explored clock: another advance can be scheduled between them,
  so the woken waiter's ``time.monotonic()`` may already see a later
  deadline's value.  This mirrors real time passing between statements;
  assert against deadline ordering rather than requiring the first clock read
  after a wake to equal that waiter's exact deadline.
* C-level clock reads and sleeps (including inside extension modules) are
  outside the virtual clock unless a frontrun integration explicitly models
  them.

Hazard: virtual-derived deadlines in raw loop-timer APIs
--------------------------------------------------------

frontrun virtualizes ``asyncio.wait_for`` / ``asyncio.timeout`` /
``asyncio.timeout_at`` inside explored tasks, but it does **not** virtualize the
event loop's own timer heap.  Under a virtual clock the runners pin
``loop.time`` to the real monotonic clock (``_pin_loop_time`` in
``async_scheduler.py``) so that frontrun's own watchdog timers — and any raw
``loop.call_later`` / ``loop.call_at`` callbacks — compare against the same
wall-clock reference the loop uses to fire them.  That pin has two consequences.

**Relative loop timers still fire, but on the wall clock.**
``loop.call_later(delay, cb)`` computes ``when = loop.time() + delay``; with
``loop.time`` pinned to real monotonic, ``when`` is a real timestamp and the
callback fires after ``delay`` *real* seconds.  It is correct but not
virtualized: it costs real wall time (defeating the zero-cost model), runs
outside the schedule (so it is neither explored nor deterministically
replayable), and is invisible to exact deadlock detection.  A third-party
library that builds timeouts on ``loop.call_later`` (an aiohttp timeout handle,
say) keeps working but reintroduces wall-clock timing.

**Absolute loop timers derived from ``time.monotonic()`` use the wrong time
domain.** Inside an explored task ``time.monotonic()`` returns virtual time
(starting at ``VIRTUAL_EPOCH`` = 1,000,000 s), so:

.. code-block:: python

    loop.call_at(time.monotonic() + 1.0, callback)   # HAZARD

schedules ``callback`` in the loop's real-time timer heap using a virtual-clock
timestamp.  Depending on the host's real monotonic value, that can be far in the
future, already due, or otherwise unrelated to the intended virtual deadline.  A
task awaiting that timer — a future the callback was meant to resolve, a
cancellation it was meant to deliver — can therefore hang until the wall-clock
watchdog, complete at an unintended wall-clock time, or produce an inconclusive
"all executions timed out before completion" result, with **no diagnostic
pointing at the clock-domain mismatch**.

**Workaround.** Inside explored tasks, express timeouts with
``asyncio.wait_for`` / ``asyncio.timeout`` / ``asyncio.timeout_at`` — these are
virtualized.  Keep code that schedules raw ``loop.call_later`` / ``loop.call_at``
timers, or that derives absolute deadlines from ``time.monotonic()``, *outside*
explored task contexts. ``clock_diagnostics`` does not currently detect this
clock-domain mismatch.

How it works
------------

One design serves both modes.  The DPOR engine is constructed with one
extra thread — the *clock actor* (id ``len(workers)``); its steps advance
the clock to the earliest pending deadline and unblock the workers whose
deadlines are due (equal deadlines wake in deterministic
``(deadline, worker id)`` order).  The two modes differ only in when the
actor is *enabled*:

* ``"virtual"`` — the actor is enabled only when no real worker is
  runnable, which is exactly when time must pass for anything to happen.
* ``"explored"`` — the actor is enabled whenever a deadline is pending,
  so the engine explores its position in the schedule.

Because the actor's steps are ordinary engine steps, recorded schedules
contain them, vector clocks order them (via the wake edges), and the
replay schedulers perform the same advances at the same positions — the
counterexample stays a deterministic, replayable proof.

Stale wakeup-tree choices can still select the clock actor after every deadline
was canceled. Such a choice performs no physical transition and is omitted
from the constructive schedule, so every recorded clock-actor entry denotes a
real advance. If positional drift makes replay reach that entry before its
deadline registration, replay owes exactly that real advance and performs it
when registration arrives. A reproduction count
below 100% is the visible symptom; inspect the original counterexample schedule
when it occurs.
