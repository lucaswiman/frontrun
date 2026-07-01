Cross-Process Exploration
=========================

Frontrun's DPOR, bytecode, and marker approaches all interleave concurrency
*within a single Python process* --- across threads or async tasks. Cross-process
exploration extends the same idea to **separate Python processes** that contend
on shared *external* state (SQL and Redis).

Each worker process runs frontrun's SQL/Redis interception and coordinates with
a parent coordinator over a socket. The Rust DPOR engine drives the search, so
equivalent interleavings are pruned (partial-order reduction) and cross-worker
``SELECT FOR UPDATE`` deadlocks are detected --- the same guarantees as the
in-process DPOR path, applied at external-access (SQL statement / Redis command)
granularity.

.. note::

   The supported model is: **if the workers are Python, run frontrun inside
   them.** There is no scheduling of unmodified non-Python processes.


``execution="process"`` --- the ergonomic mirror
-------------------------------------------------

The simplest way in is :func:`frontrun.explore` with ``execution="process"``. It
has the same ``setup`` / ``workers`` / ``invariant`` shape as the thread and
async interface (including ``count=`` to replicate a worker) and returns the same
:class:`~frontrun.common.InterleavingResult` (``property_holds`` /
``counterexample`` / ``explanation`` / ``assert_holds``).

Two differences are inherent to using processes:

* Workers must be **picklable** (module-level callables), exactly as with
  :mod:`multiprocessing`.
* ``setup()`` returns a **picklable handle** to the external state --- a DB URL,
  a SQLite path, a Redis key namespace --- rather than a live Python object. The
  handle is passed to every ``worker(state)`` and to ``invariant(state)``. State
  lives in the external store (SQL/Redis), not in shared Python memory.

``setup`` and ``invariant`` run in the coordinator process; the workers run in
their own spawned processes.

.. code-block:: python

   import sqlite3
   import frontrun

   # Module-level (picklable) worker: a racy read-modify-write over two statements.
   def increment(db_path):
       conn = sqlite3.connect(db_path, isolation_level=None)
       try:
           val = conn.execute("SELECT val FROM counter WHERE id = 1").fetchone()[0]
           conn.execute("UPDATE counter SET val = ? WHERE id = 1", (val + 1,))
       finally:
           conn.close()

   def read(db_path):
       conn = sqlite3.connect(db_path, isolation_level=None)
       try:
           return conn.execute("SELECT val FROM counter WHERE id = 1").fetchone()[0]
       finally:
           conn.close()

   def test_counter_race(tmp_path):
       db = str(tmp_path / "counter.db")

       def setup():
           conn = sqlite3.connect(db, isolation_level=None)
           conn.execute("CREATE TABLE IF NOT EXISTS counter (id INTEGER PRIMARY KEY, val INTEGER)")
           conn.execute("DELETE FROM counter")
           conn.execute("INSERT INTO counter (id, val) VALUES (1, 0)")
           conn.close()
           return db  # picklable handle passed to each worker(state) and invariant(state)

       result = frontrun.explore(
           setup=setup,
           workers=increment,
           count=2,
           invariant=lambda state: read(state) == 2,
           execution="process",
       )
       assert not result.property_holds        # lost-update race found across processes
       assert result.counterexample is not None

Compare with the same test written against a safe, single-statement increment
(``UPDATE counter SET val = val + 1``): the invariant holds under every
interleaving and ``result.property_holds`` is ``True``.

``execution="process"`` accepts sync ``"dpor"`` only. Async workers and other
strategies raise ``ValueError`` (SQL/Redis state is external, so async worker
support and random scheduling do not apply to the process path). SQLite needs
nothing extra; a Redis worker needs the ``redis`` package and a running server.


``explore_processes()`` --- the lower-level entry
-------------------------------------------------

:func:`frontrun.explore_processes` is the underlying API. Instead of pickled
callables, it spawns each worker as a real OS process running a
``"module:callable"`` target under a fresh interpreter, described by a
:class:`frontrun.Subprocess` spec:

.. code-block:: python

   import frontrun
   from frontrun._dpor_runtime.xproc import _demo_counter

   _TARGET = "frontrun._dpor_runtime.xproc._demo_counter:increment"

   def test_lost_update_across_processes(tmp_path):
       db = str(tmp_path / "counter.db")
       result = frontrun.explore_processes(
           {
               "w0": frontrun.Subprocess(_TARGET, (db,)),
               "w1": frontrun.Subprocess(_TARGET, (db,)),
           },
           setup=lambda: _demo_counter.setup(db),   # resets the DB before each interleaving
           invariant=lambda: _demo_counter.read(db) == 2,  # reads the DB afterwards
           max_iterations=50,
       )
       assert not result.ok
       assert result.failure_kind == "invariant"
       assert result.failing_schedule is not None

``processes`` is a mapping of label → :class:`~frontrun.Subprocess` (labels are
purely for readability) or a plain sequence. ``Subprocess(target, args)`` names a
``"module:callable"`` and its positional ``args``; the args must be
JSON-serialisable, since they are passed to the child through the environment.
Because the child imports the target by name, it must be importable in a fresh
interpreter --- a module-level callable in an installed or on-path module.

Here ``setup`` and ``invariant`` take no arguments and reach the shared store
directly (they run in the coordinator process); ``setup`` resets the external
state before each interleaving and ``invariant`` checks it afterwards.

:func:`~frontrun.explore_processes` returns a ``CrossProcessResult``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``ok``
     - ``True`` if the invariant held under every explored interleaving.
   * - ``failure``
     - Human-readable description of the first violation, or ``None``.
   * - ``failure_kind``
     - One of ``"invariant"``, ``"worker_error"``, ``"deadlock"``, or ``None``.
   * - ``failing_schedule``
     - The interleaving (a list of worker ids) that triggered the failure.
   * - ``iterations``
     - Number of interleavings explored.
   * - ``exhausted``
     - ``True`` if the search space was fully covered.


Strategies and worker reuse
---------------------------

``strategy`` selects the coordinator:

* ``"dpor"`` (default) drives the Rust DPOR engine, pruning equivalent
  interleavings and detecting cross-worker ``SELECT FOR UPDATE`` deadlocks.
  ``max_executions`` and ``preemption_bound`` tune the search.
* ``"exhaustive"`` brute-forces every interleaving at external-access
  granularity, bounded by ``max_iterations``. Useful as a reduction-free
  cross-check that DPOR reaches the same verdict.

``reuse_workers=True`` keeps the worker processes alive across iterations,
re-running the target in place instead of respawning for each interleaving. The
verdict is identical; reuse trades startup cost for the target being run
repeatedly in one process.


Redis workers
-------------

The same interface works against Redis. A worker connects to a Redis server and
performs a racy GET/SET, while the coordinator resets and checks the counter:

.. code-block:: python

   import frontrun
   from frontrun._dpor_runtime.xproc import _demo_redis

   _TARGET = "frontrun._dpor_runtime.xproc._demo_redis:increment"

   def test_redis_lost_update():
       _demo_redis.setup()
       result = frontrun.explore_processes(
           {"w0": frontrun.Subprocess(_TARGET), "w1": frontrun.Subprocess(_TARGET)},
           setup=_demo_redis.setup,
           invariant=lambda: _demo_redis.read() == 2,
       )
       assert not result.ok
       assert result.failure_kind == "invariant"

Redis exploration requires the ``redis`` package and a running server (the demo
workers read ``FRONTRUN_XPROC_REDIS_URL``, defaulting to a local instance). The
atomic variant --- Redis ``INCR`` --- has no race, so ``result.ok`` is ``True``.


Running the tests
-----------------

Cross-process tests spawn real processes and are marked with the pytest ``e2e``
marker, so they are opt-in:

.. code-block:: bash

   make test-e2e-3.14                 # cross-process e2e tests on the 3.14 venv
   pytest -m e2e                      # or select the marker directly

The SQLite tests need nothing beyond the standard library. The Redis tests are
additionally marked ``integration`` and are skipped unless ``redis`` is
installed and a server is reachable.
