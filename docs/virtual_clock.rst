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

    import frontrun

    result = frontrun.explore(
        setup=...,
        workers=[...],
        invariant=...,
        clock="virtual",      # default "real"; "explored" adds the clock actor
    )

Both sync (threads) and async (asyncio tasks) workers are supported, with
``strategy="dpor"`` and ``strategy="random"``.

What a virtual clock changes
----------------------------

With ``clock="virtual"`` each execution gets a fresh clock starting at an
arbitrary epoch (1,000,000.0 seconds), owned by the scheduler:

* **Clock reads are virtual.** ``time.time()``, ``time.monotonic()``,
  ``time.perf_counter()`` (and their ``_ns`` variants) return virtual time
  inside explored code.  Patching is gated per thread/context: worker
  threads and tasks, ``setup()``, and the ``invariant`` see virtual time;
  unrelated threads (pytest machinery, background daemons) see real time.
* **Sleeps are timed blocks.** ``time.sleep(d)`` / ``asyncio.sleep(d)``
  with ``d > 0`` register a deadline at ``now + d`` and block until the
  clock reaches it — in zero wall time.  ``sleep(0)`` remains a pure yield,
  matching stock Python semantics.
* **Timed lock acquires are deterministic.** A contended
  ``threading.Lock.acquire(timeout=t)`` registers a virtual deadline
  instead of busy-waiting against the host clock; whether it succeeds no
  longer depends on machine speed.
* **The clock advances only when it must.** When every live worker is
  finished or deadline-blocked, the clock jumps to the *earliest* pending
  deadline and wakes its sleepers.  This is the "autojump" model
  (prior art: ``trio.testing.MockClock(autojump_threshold=0)``).
* **Deadlocks are detected exactly.** All workers blocked with *no*
  pending deadline is a genuine deadlock, reported immediately instead of
  via the wall-clock fallback timeout.

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
* **Async loop timers stay on the wall clock.** ``loop.time()``,
  ``loop.call_later``, and therefore ``asyncio.wait_for`` /
  ``asyncio.timeout`` deadlines are not virtualised: the scheduler's own
  deadlock-timeout timers share the event loop's timer heap, and
  virtualising it would let a clock jump fire them spuriously.  A
  ``wait_for`` with a short real timeout still works — it just measures
  wall time, not virtual time.  (This is the "spike" outcome from the
  proposal; wrapping ``wait_for`` over virtual deadlines is future work.)
* Timed waits on ``Event`` / ``Condition`` / ``Queue`` keep wall-clock
  timeouts (they spin-yield exactly as with ``clock="real"``).  Fully
  virtualised timeouts currently cover ``sleep`` and lock acquires.
* C-level sleeps (e.g. inside database drivers) are invisible to the
  clock — the same boundary as the existing ``deadlock_timeout``
  caveat for unmanaged C code.
* ``datetime.datetime.now()`` is not patched; the supported surface is
  ``time.time`` / ``time.monotonic`` / ``time.perf_counter`` (+ ``_ns``).

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
