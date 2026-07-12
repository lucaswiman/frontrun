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

Because separate processes share no memory, only *external-store* accesses (SQL
statements, Redis commands) are scheduling points --- purely in-memory work
inside a worker is not interleaved the way threads are. This is coarser than
thread scheduling by design: the interesting cross-process races are the ones
that go through the shared database or cache.

Unlike the in-process DPOR path, cross-process exploration does **not** require
the ``frontrun`` CLI wrapper (there is no ``LD_PRELOAD`` C-level I/O to
intercept); a plain ``pytest`` run works, since each worker installs frontrun's
Python-level SQL/Redis interception itself.


Which entry point?
------------------

There are two public entry points; pick by how you want to define workers:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - ``explore(execution="process")``
     - ``explore_processes()``
   * - Worker is
     - any callable — **closures and lambdas too** (serialised with dill)
     - a ``"module:callable"`` **target string** run in a fresh interpreter
   * - Args reach the worker via
     - **dill** serialisation of ``setup()``'s return (tuples, dicts, closures)
     - **JSON** through the environment (scalars/lists/str-keyed dicts only)
   * - Return type
     - :class:`~frontrun.common.InterleavingResult` (like threads/async)
     - :class:`CrossProcessResult` (``ok`` / ``failure`` / ``failure_kind`` ...)
   * - Reach for it when
     - you want a drop-in mirror of the threading API
     - the target must run in a clean interpreter, or richer JSON args suffice

``explore(execution="process")`` is the ergonomic default; ``explore_processes()``
is the lower-level door with explicit per-worker specs.


``execution="process"`` --- the ergonomic mirror
-------------------------------------------------

The simplest way in is :func:`frontrun.explore` with ``execution="process"``. It
has the same ``setup`` / ``workers`` / ``invariant`` shape as the thread and
async interface (including ``count=`` to replicate a worker) and returns the same
:class:`~frontrun.common.InterleavingResult` (``property_holds`` /
``counterexample`` / ``explanation`` / ``assert_holds``).

Install the ``process`` extra (``pip install frontrun[process]``), which pulls in
dill for serialising workers. Two differences are inherent to using processes:

* Workers are **serialised with dill**, so closures and lambdas work as well as
  module-level functions (dill handles more than the stdlib pickle that
  :mod:`multiprocessing` uses). Genuinely unserialisable captures --- an open
  connection, socket, or file handle held by the worker --- still fail, with a
  clear error.
* ``setup()`` returns a **handle** to the external state --- a DB URL, a SQLite
  path, a Redis key namespace --- rather than a live Python object. The handle is
  passed to every ``worker(state)`` and to ``invariant(state)``. State lives in
  the external store (SQL/Redis), not in shared Python memory.

``setup`` and ``invariant`` run in the coordinator process; the workers run in
their own spawned processes.

Because this entry point uses :mod:`multiprocessing` with the ``spawn`` start
method, the parent program must be a file-backed Python module. Running it from
stdin, ``python -c``, or a REPL/notebook cell is rejected with a clear error
before workers start. Put the test in a ``.py`` file, or use
``explore_processes()`` with importable ``"module:callable"`` targets.

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

   _TARGET = "myapp.counter:increment"  # your importable module-level worker

   def test_lost_update_across_processes(tmp_path):
       db = str(tmp_path / "counter.db")
       result = frontrun.explore_processes(
           {
               "w0": frontrun.Subprocess(_TARGET, (db,)),
               "w1": frontrun.Subprocess(_TARGET, (db,)),
           },
           setup=lambda: reset_counter(db),   # your DB reset helper
           invariant=lambda _state: read_counter(db) == 2,
       )
       assert not result.ok
       assert result.failure_kind == "invariant"
       assert result.failing_schedule is not None

``processes`` is a mapping of label → :class:`~frontrun.Subprocess` (labels are
preserved in ``result.worker_labels``) or a plain sequence. ``Subprocess(target, args)`` names a
``"module:callable"`` and its positional ``args``; the args are passed to the
child as **JSON through the environment**, so they must be JSON-serialisable
**and survive a JSON round-trip**: a tuple arrives as a list and a dict with
non-string keys comes back string-keyed. Pass plain scalars, lists, and
string-keyed dicts --- or use ``frontrun.explore(execution="process")`` (which
pickles) when you need richer argument types. Because the child imports the
target by name, it must be importable in a fresh interpreter --- a module-level
callable in an installed or on-path module.

Targets must be synchronous callables. If a target is ``async def`` (or a
plain callable that returns an awaitable), the worker reports a clear error;
cross-process exploration has no asyncio scheduler in the child process.

``setup`` and ``invariant`` both run in the coordinator process. ``setup``
resets the external state before each interleaving and returns a handle to it;
that handle is passed to ``invariant(state)``, which checks the state afterwards
--- mirroring ``explore(execution="process")``. ``invariant`` may ignore
``state`` and reach the shared store directly.

:func:`~frontrun.explore_processes` returns a :class:`frontrun.CrossProcessResult`
(importable from the top-level package):

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
     - One of ``"invariant"``, ``"worker_error"``, ``"deadlock"``,
       ``"timeout"`` (the scheduler's ``deadlock_timeout`` expired — a worker
       blocked outside frontrun's model or a statement outran the budget, so
       the schedule was never driven to completion), ``"nondeterministic"``,
       ``"step_limit"`` (a run exceeded the exhaustive coordinator's
       per-run scheduling-point bound), ``"branch_limit"`` (an execution
       exceeded the DPOR strategy's ``max_branches`` scheduling points and was
       truncated — the reported schedule is the truncated prefix, not a
       verified counterexample), or ``None``.
   * - ``failing_schedule``
     - The interleaving (a list of worker ids) that triggered the failure.
   * - ``worker_labels``
     - For mapping input, the ``{worker_id: label}`` mapping that translates
       numeric ids in schedules and access traces back to the caller's labels;
       otherwise an empty dict.
   * - ``failures``
     - Every failing execution as ``(execution_number, schedule)`` pairs,
       mirroring thread-mode ``InterleavingResult.failures``. Populated by
       both strategies whenever a failure carries a schedule; with
       ``stop_on_first=False`` the DPOR strategy accumulates all failing
       executions here instead of stopping at the first.
   * - ``iterations``
     - Number of interleavings explored.
   * - ``exhausted``
     - ``True`` if the search space was fully covered.  Under
       ``strategy="dpor"`` this requires ``preemption_bound=None``: the
       default bound (2) truncates the search to low-preemption schedules,
       so bounded runs honestly report ``exhausted=False`` even when the
       bounded tree was fully traversed.
   * - ``accesses``
     - The external accesses observed on the failing run, as
       ``(worker_id, resource, "read"|"write")`` tuples, or ``None``.

The :class:`~frontrun.common.InterleavingResult` returned by
``explore(execution="process")`` carries the same structured information:
``exhausted``, ``failure_kind``, and ``failures`` are copied from the
underlying ``CrossProcessResult`` alongside the human-readable
``explanation``.


Strategies and worker reuse
---------------------------

``strategy`` selects the coordinator:

* ``"dpor"`` (default) drives the Rust DPOR engine, pruning equivalent
  interleavings and detecting cross-worker ``SELECT FOR UPDATE`` deadlocks.
  ``max_executions``, ``preemption_bound``, ``max_branches``, and
  ``total_timeout`` bound the search, ``search`` selects the wakeup-tree
  traversal order, and ``stop_on_first=False`` keeps exploring after a
  failure, accumulating every failing execution in ``result.failures``.
* ``"exhaustive"`` brute-forces every interleaving at external-access
  granularity, bounded by ``max_iterations`` and by ``max_steps_per_run`` per
  execution. Useful as a reduction-free cross-check that DPOR reaches the same
  verdict.

Each strategy rejects the other's bounds when passed explicitly:
``max_iterations`` and ``max_steps_per_run`` only apply to ``"exhaustive"``
(bound a DPOR search with ``max_executions`` instead), and the DPOR knobs above
only apply to ``"dpor"``. Detection is value-based, so passing a default value
is indistinguishable from omitting it.

``reuse_workers=True`` keeps the worker processes alive across iterations,
re-running the target in place instead of respawning for each interleaving. The
verdict is identical; reuse trades startup cost for the target being run
repeatedly in one process, so the target's process-global state must be safe to
re-enter (frontrun resets its own per-connection SQL state between iterations).
It is available on both entry points with DPOR ---
``explore_processes(..., reuse_workers=True)`` and
``frontrun.explore(..., execution="process", reuse_workers=True)``. The
lower-level exhaustive strategy rejects worker reuse.

If a reused iteration deadlocks or aborts, frontrun kills and reaps the poisoned
worker processes, then launches a fresh set before continuing the search. This
loses process-global re-entry state at that boundary, but never feeds another
schedule into a desynchronised protocol stream. Thread execution rejects
``reuse_workers=True`` with ``ValueError`` because Python cannot safely kill an
arbitrary stuck thread. Independently, detected thread deadlocks are returned
as failed ``InterleavingResult`` values rather than raised directly.

.. code-block:: python

   result = frontrun.explore_processes(
       {"w0": frontrun.Subprocess(_TARGET, (db,)), "w1": frontrun.Subprocess(_TARGET, (db,))},
       setup=lambda: reset_counter(db),
       invariant=lambda _state: read_counter(db) == 2,
       reuse_workers=True,          # spawn each worker once, re-run per interleaving
   )


Deadlock detection
------------------

When workers take row locks (``SELECT ... FOR UPDATE``) in conflicting orders,
the coordinator's wait-for graph detects the cycle and reports
``failure_kind == "deadlock"`` --- the same machinery as the in-process DPOR
path. This needs a store with real row locks (PostgreSQL/MySQL); SQLite has
none, so the example requires a running Postgres:

.. code-block:: python

   # myapp/transfer.py  (importable target)
   import psycopg2

   def lock_two(dsn, first_id, second_id):
       conn = psycopg2.connect(dsn)
       conn.autocommit = False
       cur = conn.cursor()
       cur.execute("SELECT * FROM accounts WHERE id = %s FOR UPDATE", (first_id,))
       cur.execute("SELECT * FROM accounts WHERE id = %s FOR UPDATE", (second_id,))
       conn.commit()
       conn.close()

Then (``dsn`` and ``reset_accounts`` are your own connection string and
table-reset helper):

.. code-block:: python

   # Two workers lock rows 1 and 2 in opposite order: a classic deadlock.
   result = frontrun.explore_processes(
       {
           "w0": frontrun.Subprocess("myapp.transfer:lock_two", (dsn, 1, 2)),
           "w1": frontrun.Subprocess("myapp.transfer:lock_two", (dsn, 2, 1)),
       },
       setup=lambda: reset_accounts(dsn),
       invariant=lambda _state: True,   # we are looking for the deadlock, not a data race
   )
   assert not result.ok
   assert result.failure_kind == "deadlock"
   assert result.failing_schedule is not None


Redis workers
-------------

The same interface works against Redis. A worker connects to a Redis server and
performs a racy GET/SET, while the coordinator resets and checks the counter:

.. code-block:: python

   import frontrun

   _TARGET = "myapp.redis_counter:increment"  # your importable Redis worker

   def test_redis_lost_update():
       result = frontrun.explore_processes(
           {"w0": frontrun.Subprocess(_TARGET), "w1": frontrun.Subprocess(_TARGET)},
           setup=reset_redis_counter,
           invariant=lambda _state: read_redis_counter() == 2,
       )
       assert not result.ok
       assert result.failure_kind == "invariant"

Redis exploration requires the ``redis`` package and a running server (the demo
workers read ``FRONTRUN_XPROC_REDIS_URL``, defaulting to a local instance). The
atomic variant --- Redis ``INCR`` --- has no race, so ``result.ok`` is ``True``.

The ergonomic ``execution="process"`` door works against Redis too --- the
``setup()`` return value is the picklable handle (here the Redis URL) passed to
each ``worker(state)``:

.. code-block:: python

   import redis
   import frontrun

   REDIS_URL = "redis://127.0.0.1:6379/0"

   def increment(url):
       r = redis.Redis.from_url(url)
       current = int(r.get("counter") or 0)
       r.set("counter", current + 1)

   def setup():
       redis.Redis.from_url(REDIS_URL).set("counter", 0)
       return REDIS_URL  # picklable handle passed to each worker(state)/invariant(state)

   result = frontrun.explore(
       setup=setup,
       workers=increment,
       count=2,
       invariant=lambda url: int(redis.Redis.from_url(url).get("counter")) == 2,
       execution="process",
   )
   assert not result.property_holds   # lost update found across processes


Running the tests
-----------------

Cross-process tests spawn real processes and are marked with the pytest ``e2e``
marker. They run as part of the default suite (``make test-3.14`` applies no
marker filter); the marker exists so they can also be selected on their own:

.. code-block:: bash

   make test-e2e-3.14                 # only the cross-process e2e tests, 3.14 venv
   pytest -m e2e                      # or select the marker directly

The SQLite tests need nothing beyond the standard library. The Redis tests are
additionally marked ``integration`` and are skipped unless ``redis`` is
installed and a server is reachable.
