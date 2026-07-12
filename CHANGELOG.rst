Changelog
=========

All releases: https://github.com/lucaswiman/frontrun/releases

Unreleased
----------

* **Virtual clock for timeout, retry, and TTL races.** ``frontrun.explore(...)``
  now accepts ``clock="virtual"`` and ``clock="explored"`` for sync and async
  workers with DPOR and random strategies. Explored code reads scheduler time,
  sleeps and timeout wrappers use zero-wall-time virtual deadlines, and
  ``clock="explored"`` makes timer firings schedulable. This includes
  cooperative timed waits, async ``wait_for`` / ``timeout`` / ``timeout_at``,
  async Event/Queue/Condition wakeups, concrete ``datetime`` / ``date`` values,
  captured ``time.*`` diagnostics where tracing is available, and preservation
  of pre-existing third-party time patches. The invariant is evaluated under
  the same clock and sleep patches as setup and the workers (a TTL-style
  invariant that sleeps ages the virtual clock instead of blocking for real
  wall time), and the random strategy's autojump yields to the event loop
  after a virtual timeout fires so ``asyncio.wait_for`` / ``asyncio.timeout``
  cancel the timed-out task instead of letting it run to completion. See
  :doc:`virtual_clock` (including its known-limitations section).

* **Cross-process exploration.** ``frontrun.explore(...)`` gains an
  ``execution="process"`` mode that runs each worker in its own Python process,
  using the same ``setup`` / ``workers`` / ``invariant`` shape as thread mode
  for shared SQL/Redis state. Workers are serialised with dill (install the
  ``process`` extra), and process runs support ``count=`` and
  ``reuse_workers=True``; the lower-level ``explore_processes`` API still
  supports explicit ``Subprocess`` targets and exhaustive search. Process-mode
  errors now fail fast with clearer messages, report truncation honestly via
  ``CrossProcessResult.exhausted`` (``exhausted=True`` requires a genuinely
  unbounded search: the default ``preemption_bound=2`` truncates the DPOR
  tree, so bounded runs report ``False`` — pass ``preemption_bound=None`` to
  claim full coverage), honor ``total_timeout`` even while a single execution
  is in flight, and reject in-process-only options instead
  of silently ignoring them. A scheduler stall (``deadlock_timeout`` expiry —
  e.g. unmodeled database-level blocking) is reported as
  ``failure_kind="timeout"`` rather than counting as a clean pass, and an
  execution truncated by the DPOR ``max_branches`` cap is reported as its own
  ``failure_kind="branch_limit"`` (previously it burned ``deadlock_timeout``
  and masqueraded as a ``"timeout"`` whose message pointed at the wrong knob).
  Both strategies now surface the launcher's captured child stderr on a
  connect failure (previously ``strategy="exhaustive"`` reported a bare
  connect timeout, hiding e.g. the child's ``ModuleNotFoundError``).
  See :doc:`cross_process`.

  Invalid process bounds now fail before workers launch, and lower-level
  ``Subprocess`` targets that return awaitables report an unsupported-async
  worker error instead of silently succeeding with an unawaited coroutine.
  Mapping-input labels are preserved in ``CrossProcessResult.worker_labels``;
  poisoned reused processes are killed, reaped, and freshly launched before
  exploration continues.

* **explore() rejects options its strategy would ignore.** Thread-mode
  ``frontrun.explore(...)`` now raises ``ValueError`` for any explicitly-passed
  option the selected strategy does not support (e.g. ``seed=`` with
  ``strategy="dpor"``, ``preemption_bound=`` with ``strategy="random"``,
  ``reproduce_on_failure=`` with async random), extending the process branch's
  no-silent-no-op principle to every entry path. Code that passed such options
  before relied on them silently doing nothing — drop the option or switch
  strategy. Newly covered by the same principle: ``reuse_workers=True`` with
  thread execution (previously a silent no-op) now raises, ``explore_processes``
  rejects ``max_iterations`` under ``strategy="dpor"`` (it only bounds the
  exhaustive coordinator — use ``max_executions``) and the DPOR knobs under
  ``strategy="exhaustive"``, and every process-branch rejection message now
  follows the same sentence shape as the thread-branch ones (what was passed,
  why it is invalid, what to do instead).

  *Caveat:* explicit-option detection is value-based — an option passed at its
  signature default is indistinguishable from an omitted one, so e.g.
  ``explore(..., seed=None, strategy="dpor")`` is accepted (a no-op either
  way).

* **Mixed sync/async worker lists get an actionable diagnosis.** Passing a mix
  of ``async def`` and plain ``def`` workers to ``frontrun.explore(...)``
  previously routed the whole list to the async engine, where the sync workers
  failed with an opaque ``can't be used in 'await' expression``. A plain
  callable that *returns* an awaitable is a valid async worker and statically
  indistinguishable from a sync one, so the mix cannot be rejected up front;
  instead the first execution now fails with a ``TypeError`` naming the sync
  worker and stating the fix (make every worker ``async def`` or a callable
  returning an awaitable, or every worker a plain ``def``).

* **Structured process-mode results.** ``CrossProcessResult`` is now exported
  at the top level (``frontrun.CrossProcessResult``) and carries a
  ``failures`` list of every failing ``(execution_number, schedule)`` pair —
  the DPOR strategy accumulates all of them with the new
  ``stop_on_first=False``. ``explore_processes`` also gains the thread-mode
  DPOR knobs ``stop_on_first``, ``total_timeout``, ``search``, and
  ``max_branches``. The ``InterleavingResult`` returned by
  ``explore(execution="process")`` no longer flattens everything into the
  explanation string: it gains ``exhausted`` and ``failure_kind`` fields and
  populates ``failures``, alongside the human-readable ``explanation``
  (thread/async results leave ``exhausted`` as ``None`` — they do not report
  it yet).

* **Cross-process robustness.** Iteration liveness is now judged on relay
  *progress* rather than wall time, so a long-but-healthy run is no longer
  aborted mid-flight; a worker that stays connected but silent past
  ``deadlock_timeout`` is diagnosed as ``failure_kind="timeout"`` (with
  raise-``deadlock_timeout`` advice) instead of a misleading "worker
  disconnected"; and the exhaustive coordinator bounds each run with
  ``max_steps_per_run`` (``failure_kind="step_limit"``) so a nonterminating
  worker cannot hang exploration forever.

* **Release-blocking proof-integrity fixes.** The ``process`` extra now installs
  the SQL parser required by cross-process workers, and process SQL
  interception fails closed if that parser is unavailable instead of silently
  certifying a run with no semantic accesses. Failed physical COMMIT/ROLLBACK
  operations retain modeled transaction state and row locks. Marker exploration
  no longer counts an unconsumed or timed-out schedule as an exhaustive pass;
  async marker workers with arguments are awaited correctly. Random exploration
  keeps sampled prefixes within ``max_ops``, returns sync worker crashes with
  every deterministically extended turn in the structured counterexample, and
  keeps async work controlled when its sampled prefix ends. Async SQL row-lock
  contenders now park inside the scheduler (including replay) rather than
  entering a blocking database call, preserving cross-resource deadlock cycles
  and replayability. A second audit now also rejects timed-out sync/async runs,
  self-cancelled async workers, and ``SystemExit`` from sync workers instead of
  treating partial state as a proof; makes ``Queue.join()`` cooperative; restores
  nested async trace filters; and replays task-crash counterexamples. SQL
  transaction state is tied to its owning connection, statement failures keep
  earlier row locks, and failed transaction-control statements take effect only
  after physical success. Cross-process opaque SQL uses a conservative
  database-wide conflict instead of relying on the unavailable preload fallback,
  while Redis replay no longer invents boundaries for empty/keyless pipelines.

* **Release artifacts fail closed.** The release workflow no longer publishes
  a source distribution that omits ``libfrontrun_io`` and silently weakens
  C-extension I/O detection. Windows x86-64 wheels are built for the supported
  DPOR/marker/bytecode subset; Linux and macOS wheels continue to include the
  native preload library.

* **Virtual-clock fixes.** User subclasses of ``datetime.datetime`` /
  ``datetime.date`` keep stdlib semantics under a virtual clock (the shims now
  dispatch on the subclass instead of always returning the patched base
  class). Async ``asyncio.timeout`` deadlines fire at exact virtual times,
  ``asyncio.wait_for`` on a bare future is a schedulable wait, and
  ``Condition.wait_for`` timeouts behave consistently across supported Python
  versions.

* **Replay fixes.** Replaying a counterexample now reproduces exact deadlocks
  (a replayed schedule that ends in the discovered deadlock no longer aborts
  the replay machinery) and timed-wait expiries under IO-anchored replay, so
  ``reproduce_on_failure`` statistics stay meaningful for deadlock-, timeout-,
  and task-crash-shaped counterexamples. Timed-out replays never return partial
  state for invariant evaluation.

* **DPOR correctness.** Accesses after ``await`` are now attributed before
  scheduling successors, ``asyncio.Lock`` / event state races replay
  consistently, async Redis commands create post-command scheduling boundaries
  for TOCTOU races, SQL row-lock schedules stay exact without hiding row data
  races, and pure-lock deadlocks are found reliably across supported Python
  versions.

0.6.0 (2026-06-30)
------------------

**Removals.** The deprecation shims announced in 0.5 have been deleted.
There is now exactly one public entry point for interleaving exploration —
``frontrun.explore()`` — plus the renamed bytecode helpers
``frontrun.explore_random`` / ``frontrun.explore_async_random``.

Removed (previously deprecated in 0.5):

* ``frontrun.explore_dpor`` and ``frontrun.dpor.explore_dpor`` — use
  ``frontrun.explore(...)`` (strategy ``"dpor"`` is the default).
* ``frontrun.explore_async_dpor`` and
  ``frontrun.async_dpor.explore_async_dpor`` — use ``frontrun.explore(...)``
  with async ``workers=`` (the dispatcher detects coroutine functions
  automatically).
* ``frontrun.explore_interleavings`` (sync, from package or
  ``frontrun.bytecode``) — use ``frontrun.explore_random(...)`` or
  ``frontrun.explore(..., strategy="random")``.
* ``frontrun.explore_async_interleavings`` (and
  ``frontrun.async_shuffler.explore_interleavings``) — use
  ``frontrun.explore_async_random(...)`` or ``frontrun.explore(...,
  strategy="random")`` with async workers.
* ``TraceExecutor.run(name, fn)`` positional form — use the dict form
  ``executor.run({"name": fn, ...}, timeout=...)``. The dict form starts
  every thread and waits for completion in one call, so the matching
  ``executor.wait(timeout=...)`` is no longer needed.
* ``detect_redis=True`` on the async DPOR public surface — superseded by
  ``detect_io=True``, which now activates Redis key-level patching in
  both sync and async DPOR.

Also removed: the ``frontrun.common.DEPRECATION_MESSAGES`` registry and
the ``frontrun.common.deprecate()`` helper that backed the 0.5 shims.

Migration guide
~~~~~~~~~~~~~~~

Every replacement is a one-line edit. ``threads=`` becomes ``workers=``
when moving to ``frontrun.explore``; the helper aliases keep their
original ``threads=`` / ``tasks=`` parameter names.

Sync DPOR::

    # Before
    from frontrun.dpor import explore_dpor
    result = explore_dpor(setup=Counter, threads=[Counter.increment, Counter.increment], invariant=inv)

    # After
    import frontrun
    result = frontrun.explore(setup=Counter, workers=[Counter.increment, Counter.increment], invariant=inv)
    # ...or, with the count shorthand:
    result = frontrun.explore(setup=Counter, workers=Counter.increment, count=2, invariant=inv)

Async DPOR (with Redis I/O detection)::

    # Before
    from frontrun.async_dpor import explore_async_dpor
    result = await explore_async_dpor(setup=make_state, tasks=[worker, worker], invariant=inv, detect_redis=True)

    # After
    import frontrun
    result = await frontrun.explore(setup=make_state, workers=[worker, worker], invariant=inv, detect_io=True)

Sync random/bytecode exploration::

    # Before
    from frontrun.bytecode import explore_interleavings
    result = explore_interleavings(setup=Counter, threads=[Counter.increment, Counter.increment], invariant=inv)

    # After (option A — preferred, single import)
    import frontrun
    result = frontrun.explore_random(setup=Counter, threads=[Counter.increment, Counter.increment], invariant=inv)
    # (option B — uniform entry point)
    result = frontrun.explore(setup=Counter, workers=[Counter.increment, Counter.increment], invariant=inv, strategy="random")

Async random/shuffler exploration::

    # Before
    from frontrun.async_shuffler import explore_interleavings
    result = await explore_interleavings(setup=make_state, tasks=[worker, worker], invariant=inv)

    # After
    import frontrun
    result = await frontrun.explore_async_random(setup=make_state, tasks=[worker, worker], invariant=inv)

``TraceExecutor`` (sync trace-marker schedule)::

    # Before
    executor = TraceExecutor(schedule)
    executor.run("thread1", worker1)
    executor.run("thread2", worker2)
    executor.wait(timeout=5.0)

    # After
    executor = TraceExecutor(schedule)
    executor.run({"thread1": worker1, "thread2": worker2}, timeout=5.0)

Agent upgrade prompt
~~~~~~~~~~~~~~~~~~~~

Copy-paste the block below to a coding agent (Claude Code, Cursor,
Copilot, etc.) and point it at your repo to perform the migration
mechanically. It assumes Python and is written so the agent does not
need any other context.

.. code-block:: text

    You are migrating a Python project that uses `frontrun` from 0.5 to 0.6.
    In 0.6 the deprecated APIs from 0.5 were removed. Update every call site
    so the project keeps working on 0.6. Make the smallest possible change at
    each site.

    Apply these textual rewrites in order (each rewrite covers imports + the
    call sites that follow). Pay attention to keyword names — `threads=` only
    becomes `workers=` when migrating to `frontrun.explore`; the
    `explore_random` / `explore_async_random` aliases keep `threads=` / `tasks=`.

    1. Sync DPOR:
         - Replace `from frontrun.dpor import explore_dpor` (and `from frontrun
           import explore_dpor`) with `import frontrun`.
         - Rewrite `explore_dpor(setup=..., threads=[a, b], invariant=..., **kw)`
           to `frontrun.explore(setup=..., workers=[a, b], invariant=..., **kw)`.
           Strategy defaults to "dpor"; do NOT add `strategy="dpor"` unless the
           call site also passes some other strategy in a sibling branch.
         - If the threads list is `[fn, fn, ..., fn]` of length N, prefer
           `workers=fn, count=N` for clarity.

    2. Async DPOR:
         - Replace `from frontrun.async_dpor import explore_async_dpor` with
           `import frontrun`.
         - Rewrite `await explore_async_dpor(setup=..., tasks=[a, b],
           invariant=..., detect_sql=True, detect_redis=True, **kw)` to
           `await frontrun.explore(setup=..., workers=[a, b], invariant=...,
           detect_io=True, **kw)`. The new `detect_io=True` covers both SQL
           and Redis patching in async DPOR.
         - If the original call only set `detect_sql=True` (no Redis), keep
           `detect_sql=True` and do NOT add `detect_io=True`.
         - If the original call only set `detect_redis=True`, replace it with
           `detect_io=True`.

    3. Sync random / bytecode:
         - Replace `from frontrun.bytecode import explore_interleavings` (and
           `from frontrun import explore_interleavings` when used with
           `threads=`) with `import frontrun`.
         - Rename `explore_interleavings(...)` calls to
           `frontrun.explore_random(...)`. Keep the `threads=` keyword as-is.

    4. Async random / shuffler:
         - Replace `from frontrun.async_shuffler import explore_interleavings`
           and `from frontrun import explore_async_interleavings` with
           `import frontrun`.
         - Rename `explore_interleavings(...)` / `explore_async_interleavings(...)`
           calls to `frontrun.explore_async_random(...)`. Keep `tasks=` as-is.

    5. `frontrun import explore_interleavings` ambiguous dispatcher:
         - Inspect the call. If the kwarg is `threads=`, use rule 3 (sync).
           If the kwarg is `tasks=`, use rule 4 (async).

    6. `TraceExecutor` legacy form:
         - Replace consecutive `executor.run("name1", fn1)` /
           `executor.run("name2", fn2)` / ... / `executor.wait(timeout=T)`
           with a single `executor.run({"name1": fn1, "name2": fn2, ...},
           timeout=T)`.
         - If a single `executor.run("name", fn)` is followed later by
           `executor.wait()`, collapse to `executor.run({"name": fn})` (no
           timeout) or `executor.run({"name": fn}, timeout=T)` if a timeout
           was passed.
         - The new `run()` no longer accepts a positional `name` + `fn`;
           there is no separate `wait()` step.

    7. Documentation/comments referring to the removed names should be
       updated to mention the canonical replacement (e.g. `frontrun.explore`).
       Do NOT leave parenthetical "(formerly explore_dpor)" notes — write
       the docs as if the new API has always been the only one.

    Finally, run the project's test suite and fix anything that imports
    the removed names indirectly (e.g. modules that re-exported them).
    Confirm `python -c "import frontrun; frontrun.explore; frontrun.explore_random;
    frontrun.explore_async_random"` succeeds before considering the migration done.

0.5.0 (2026-06-29)
-----------------

**Public API refresh.** Several of the library's public entry points have
been unified or renamed for ergonomics. The old names keep working in
this release but emit ``DeprecationWarning``; they are scheduled for
removal in **0.6** (deprecation lands in 0.5, removal in the next minor).

New API:

* ``frontrun.explore(setup, workers, invariant, ..., strategy="dpor"|"random")``
  — single entry point that dispatches to sync or async, DPOR or
  bytecode exploration, based on the workers and ``strategy`` kwarg.
  Replaces the four-function matrix (``explore_dpor`` /
  ``explore_async_dpor`` / ``explore_interleavings`` /
  ``explore_async_interleavings``).
* **Worker-count shorthand** — pass a single callable plus
  ``count=N`` instead of repeating it: ``workers=Counter.increment,
  count=2``.
* ``frontrun.explore_random`` / ``frontrun.explore_async_random`` —
  canonical names for random bytecode exploration. Replace
  ``explore_interleavings`` / ``explore_async_interleavings``.
* ``InterleavingResult.assert_holds(msg_prefix="")`` — convenience
  method that raises ``AssertionError`` with the full race
  explanation if the invariant failed. Prefer this over
  ``assert result.property_holds, result.explanation``.
* ``TraceExecutor.run({"name": fn, ...}, timeout=...)`` — sync API
  now accepts the dict form that async has always supported; starts
  all threads and waits for them in one call. Replaces the
  ``run(name, fn)`` + ``wait(timeout=...)`` pair.
* **Invariants may now raise** ``AssertionError`` — all four
  exploration entry points catch it and fold the assertion message
  into ``result.explanation``, so invariants can be written as natural
  ``assert`` statements with pytest-style messages.
* **Async ``detect_io=True`` now covers Redis** — the old async-only
  ``detect_redis=True`` kwarg is deprecated; ``detect_io=True`` in
  async DPOR now enables Redis key-level patching the same way sync
  DPOR already did.

Deprecated (planned removal in 0.6):

* ``explore_dpor`` and ``explore_async_dpor`` — use
  ``explore(..., strategy="dpor")`` (the default).
* ``explore_interleavings`` and ``explore_async_interleavings`` — use
  ``explore_random`` / ``explore_async_random`` or
  ``explore(..., strategy="random")``.
* ``TraceExecutor.run(name, fn)`` individual-call form — use the dict
  form.
* ``detect_redis=True`` in async DPOR — use ``detect_io=True``.

Internal cleanup:

* Fixed three pre-existing ``reportDeprecated`` pyright errors so
  ``make check`` passes cleanly: drop the deprecated
  ``show_caches=False`` argument to ``dis.get_instructions`` in
  ``_opcode_observer`` and ``_trace_format``; swap ``Iterator`` for
  ``Generator`` on a ``@contextmanager``-decorated function in
  ``_real_threading``.

0.4.1 (2026-04-01)
------------------

* Misc bug fixes

0.4.0 (2026-03-28)
------------------

* **Search strategies for DPOR** — ``explore_dpor()`` accepts a new ``search``
  parameter (``SearchStrategy`` enum) to control the order in which wakeup tree
  branches are explored.  All strategies visit the same set of Mazurkiewicz
  trace equivalence classes; only the exploration order differs.  Non-DFS
  strategies can find bugs 30–35% faster with ``stop_on_first=True``.

  - **DFS** — classic min-thread-ID depth-first search (default, optimal for
    exhaustive runs)
  - **Bit-reversal** — van der Corput low-discrepancy sequence for maximal
    early spread across conflict points
  - **Round-robin** — cycles through available threads in rotating order
  - **Stride** — coprime-stride permutation for orderly exploration
  - **Conflict-first** — reverse DFS (max thread ID first), preferring threads
    added by race reversals
* **Marker-level exhaustive exploration** — new ``explore_marker_interleavings()``, ``all_marker_schedules()``, and ``marker_schedule_strategy()`` provide completeness guarantees at trace-marker granularity.
* Fixed multiple DPOR correctness bugs: ``BoundedSemaphore.release()`` missing ``_report()``, ``Condition.notify(1)`` waking all waiters instead of one, false deadlock detection in async DPOR, lock over-exploration, and ``record_access()`` unconditionally upgrading ``AccessKind`` to ``Write``.
* Fixed SQL parsing of quoted schema-qualified table names and Redis blocking-pop classification and reproduction issues.

0.3.0 (2026-03-23)
-------------------

* **Interactive HTML report** — ``--frontrun-report=path.html`` pytest flag
  generates a self-contained HTML file visualising the full DPOR exploration:
  SVG timelines, clickable switch points, side-by-side race views with source
  context and values, keyboard/swipe navigation.

* **``track_dunder_dict_accesses`` parameter** — ``explore_dpor()`` no longer
  reports ``obj.__dict__`` accesses by default.  The duplicate conflict points
  doubled wakeup tree insertions for every attribute race with negligible
  benefit (only catches the rare ``self.x`` vs ``self.__dict__['x']`` cross-path).
  Pass ``track_dunder_dict_accesses=True`` to restore the old behaviour.

* **Improved Free-threading support** — There were some bugs on freethreaded python
  (3.14t) which lead to an explosion of spurious conflict points, where the scheduler
  was depending on GIL-synchronization to avoid writing an incorrect conflict index.

* **Missed traces** — Fixed bug where all traces had to start with the first thread as the first operation.
  All distinct Mazurkiewicz traces should now be explored.


0.2.0 (2026-03-20)
-------------------

**Redis and SQL conflict detection**

DPOR now understands Redis and SQL.  Instead of treating all traffic to the same
``host:port`` as a single conflict point, frontrun intercepts execute methods
on the db drivers in common sql and redis clients. This means that DPOR can
analyze whether two SQL queries can conflict (e.g. read or update the same row)
and not explore all interleavings of independent SQL queries.

This means that Frontrun can detect complex race conditions involving interactions
between threading primitives, sql databases and redis while keeping the exploration
space manageably small.

**Async DPOR**

DPOR now automatically treats await points in coroutines as possible conficts,
using tracing/io interception/sql+redis parsing to identify which interleavings
of awaits might could lead to resource conflicts and race conditions.

**Many bugfixes**

A large number of bugfixes from having Claude run frontrun against dozens of open
source libraries. DPOR should now be more accurate about identifying conflict
points and succeed in identifying more complex races.

**Optimal DPOR**

Switched DPOR backend to use wakeup trees and source sets, meaning that each equivalence class of trace should be explored exactly once.

0.1.0 (2026-02-27)
-------------------

**DPOR (Dynamic Partial Order Reduction)**

Systematic interleaving exploration via a Rust engine (``frontrun._dpor``,
built with PyO3/maturin).  Unlike the bytecode explorer which samples random
schedules, DPOR guarantees that every meaningfully distinct interleaving is
tried exactly once.  Shared-memory conflicts are detected automatically at the
bytecode level using a shadow stack; vector clocks prune redundant orderings.

**Automatic I/O detection**

Both the bytecode explorer and DPOR now detect socket and file I/O
automatically (``detect_io=True`` by default).  When two threads touch the same
network endpoint or file path the scheduler treats the operations as
conflicting and explores their reorderings.

**C-level I/O interception**

A new ``frontrun`` CLI wraps any command with an ``LD_PRELOAD`` library
(``libfrontrun_io.so``, built from ``crates/io/``) that intercepts libc I/O
functions (``connect``, ``send``, ``recv``, ``read``, ``write``, etc.).  This
covers opaque C extensions such as database drivers, Redis clients, and HTTP
libraries.

**Interpretable error messages**

When a race condition is found, ``result.explanation`` now contains a
human-readable trace showing interleaved source lines, the conflict pattern
(lost update, write–write, etc.), and reproduction statistics.

**LD_PRELOAD events wired into DPOR**

When run under the ``frontrun`` CLI with ``detect_io=True`` (the default),
``explore_dpor()`` now consumes C-level I/O events from the ``LD_PRELOAD``
library via ``IOEventDispatcher`` → ``_PreloadBridge``.  This means DPOR
detects races involving opaque C extensions (e.g. psycopg2/libpq calling
libc ``send()``/``recv()`` directly) that previously went unnoticed.

**Improved DPOR race detection**

- Global variable and module-level attribute accesses are now tracked as
  shared-memory conflicts.
- C-level container mutations (``list.append``, ``dict.__setitem__``, etc.)
  detected via ``sys.setprofile``.
- Closure variable (``LOAD_DEREF`` / ``STORE_DEREF``) accesses tracked.
- Builtin function calls that mutate containers (e.g. ``sorted()``,
  ``list()``) treated as reads on their arguments.
- Container iteration (``GET_ITER`` / ``FOR_ITER``) tracked as reads.
- Known limitation: C-level iteration interleaving (e.g.
  ``list(od.keys())`` vs ``OrderedDict.move_to_end()``, or ``itertools``
  combinators racing with mutations) is undetectable at the opcode level.
  See ``PEP-703-REPORT.md``.

**Other changes**

- Cooperative threading primitives (``Lock``, ``RLock``, ``Semaphore``,
  ``Event``, ``Condition``, ``Queue``, etc.) extracted to a shared module so
  both the bytecode explorer and DPOR use the same wrappers.
- Deadlock detection via wait-for graph cycle detection.
- ``--frontrun-patch-locks`` pytest plugin for early cooperative patching;
  tests that need the frontrun environment are auto-skipped when it is absent.
- Free-threaded Python (3.13t, 3.14t) support, including a fix for a PyO3
  "Already borrowed" panic and intermittent hangs in cooperative lock
  patching on 3.14t.
- Multi-version test matrix: Python 3.10, 3.14, 3.14t.
- ``DporResult`` merged into ``InterleavingResult``; all three ``explore_*``
  functions return the same type.
- Improved ``_dpor`` import error with build instructions.

0.0.2 (2026-02-17)
-------------------

Rename library from interlace to frontrun.

0.0.1 (2026-02-17)
-------------------

Initial release (as "interlace").  Includes trace markers, bytecode
exploration, and async variants.
