DPOR in Practice
================

This is a practical guide to using ``explore()`` (or the old ``explore_dpor()``)
for systematic concurrency testing. For the underlying algorithm and theory, see
:doc:`dpor`.

.. note::

   **Prefer** :func:`frontrun.explore` **(0.5+).** The old ``explore_dpor()``
   function is deprecated and will be removed in 0.6. Use
   ``frontrun.explore(strategy='dpor')`` (the default) instead.


What DPOR does
--------------

DPOR and the invariant have separate jobs:

- **DPOR decides which interleavings to run.** It watches every shared-memory
  access, detects which operations *conflict* (access the same object with at
  least one write), and uses that information to skip redundant orderings.
  Two interleavings that only differ in the order of independent operations
  (e.g. two reads, or accesses to different objects) are equivalent, so DPOR
  runs only one representative from each equivalence class.

- **The invariant decides whether a bug occurred.** After all workers finish,
  ``explore()`` calls your invariant on the final state. DPOR has no
  built-in notion of "correct" --- it doesn't know that ``counter == 1`` is
  wrong and ``counter == 2`` is right. You supply that judgement via the
  invariant. Raising ``AssertionError`` in the invariant is treated as a failure
  (the message is included in ``result.explanation``).

(The name "invariant" is standard in tools like this --- loom, CHESS, etc. ---
even though it is technically a *postcondition* checked once after the threads
finish, not a continuously-monitored loop invariant.)

Putting the two together:

.. code-block:: python

   from frontrun import explore

   result = explore(
       setup=MyState,                         # called fresh each execution
       workers=[worker_a, worker_b],          # each receives the state
       invariant=lambda s: s.is_consistent(), # checked after all workers finish
   )
   result.assert_holds()

Old API (deprecated)::

   from frontrun.dpor import explore_dpor

   result = explore_dpor(
       setup=MyState,
       threads=[thread_a, thread_b],
       invariant=lambda s: s.is_consistent(),
   )
   assert result.property_holds, result.explanation

1. DPOR picks an interleaving (based on conflict analysis).
2. ``setup()`` creates fresh state; the threads run under that interleaving.
3. The invariant checks the final state.
4. Repeat until all distinct interleavings are covered.


What it can and cannot find
---------------------------

DPOR explores alternative interleavings only where it detects a *conflict*
--- two threads accessing the same object with at least one write. The
detection mechanisms are layered from fine-grained to coarse:

**Fine-grained detection (precise, no false conflicts):**

- Attribute reads and writes (``self.x``, ``obj.field = ...``)
- Subscript reads and writes (``d[key]``, ``lst[i] = ...``)
- Lock acquire and release (``threading.Lock``, ``threading.RLock``)
- Thread spawn and join
- **Redis commands** --- ``execute_command()`` is intercepted on redis-py
  clients when ``detect_io=True``, which now covers Redis in both sync
  and async contexts.  (The async-only ``detect_redis=True`` kwarg is
  deprecated in 0.5 and will be removed in 0.6.)  Each command is
  classified as a read or write on specific keys; two threads operating
  on *different* keys are independent.  See :doc:`redis`.
- **SQL statements** --- ``cursor.execute()`` is intercepted at the DBAPI
  layer.  Statements are parsed to per-table (or per-row) resource IDs; two
  threads touching *different* tables or rows are independent.  See
  :doc:`sql-technical-details`.

**Coarse endpoint-level detection (can cause combinatorial explosion):**

- **Python socket I/O** (``connect``, ``send``, ``sendall``, ``recv``, etc.)
  --- monkey-patched when ``detect_io=True``.  The resource ID is
  ``socket:<host>:<port>``, so *every* send and recv on the same server
  conflicts with every other send and recv on that server.  Two threads each
  issuing ten Redis commands via the raw socket API would each generate ten
  write (send) and ten read (recv) events, all conflicting --- DPOR would
  explore a combinatorial explosion of orderings, most of them spurious.
  Redis and SQL detection avoid this by suppressing socket-level reporting for
  those connections and replacing it with fine-grained resource IDs.
- **C-level socket I/O via LD_PRELOAD** --- when running under the
  ``frontrun`` CLI, the ``libfrontrun_io.so`` library intercepts libc
  ``send()``/``recv()`` and reports the same ``socket:<host>:<port>`` resource
  IDs.  This catches opaque C drivers (libpq, etc.) that bypass Python's
  ``socket`` module, but with the same coarseness: all traffic to the same
  endpoint is a potential conflict.  For most database and Redis workloads,
  the SQL/Redis key-level detection should be used instead so that the
  endpoint-level reports are suppressed.
- **File opens** (``builtins.open``) --- resource identity is the resolved
  file path; read vs write determined by mode.

**Not tracked (DPOR cannot see these at all):**

- **C-extension shared state.** Shared state modified entirely inside C code
  (NumPy arrays, custom C extensions, etc.) produces no Python bytecode.
  DPOR sees no conflict and explores only one interleaving, regardless of
  whether a race exists.
- **Locks acquired in C without Python wrappers.**  A C extension that manages
  its own mutex without calling through ``threading.Lock`` is invisible to the
  wait-for-graph and the bytecode tracer.

For cases where DPOR cannot see the shared state, two alternatives are available:

- **Bytecode exploration** (``explore_interleavings()``) doesn't need to
  *understand* why a schedule is bad --- it checks an invariant after each run
  and reports when the invariant fails.  If a C extension mutates shared state
  in a way that breaks your invariant, bytecode exploration will often find the
  bad interleaving by chance.  It won't know *why* it's bad (the error trace
  is less interpretable than DPOR's), but it will find it.  See
  :doc:`approaches` for details.

- **Trace markers** let you annotate the points where interleaving matters and
  enumerate the orderings by hand.  This is the most reliable approach when you
  already know the race window.

Beyond invariant violations, DPOR also detects **deadlocks** (via wait-for-graph
cycle detection) and **crashes** (unhandled exceptions in any thread are
re-raised after the execution completes).

Thread functions should also avoid external side effects (writing to files,
sending network requests, modifying global state outside the ``setup`` object).
DPOR replays each interleaving from scratch, so side effects from one replay
will interfere with the next.


Basic usage
-----------

The ``explore_dpor()`` function is the main entry point:

.. code-block:: python

   from frontrun.dpor import explore_dpor

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           temp = self.value
           self.value = temp + 1

   result = explore_dpor(
       setup=Counter,
       threads=[lambda c: c.increment(), lambda c: c.increment()],
       invariant=lambda c: c.value == 2,
   )

   assert result.property_holds, result.explanation

**Parameters:**

``setup``
    A callable that creates fresh shared state. Called once per execution so
    that each interleaving starts from a clean slate.

``threads``
    A list of callables, each receiving the state returned by ``setup``.
    The length of this list determines the number of threads.

``invariant``
    A predicate over the shared state that defines what "correct" means.
    Called after all threads finish each execution. Return ``True`` if the
    state is valid, ``False`` if there is a bug. DPOR decides *which*
    interleavings to try; the invariant decides *whether each one passed*.

``preemption_bound`` *(default: 2)*
    Maximum number of preemptions (context switches away from a runnable
    thread) per execution. A bound of 2 catches the vast majority of real
    bugs. Set to ``None`` for unbounded exploration, but be aware that
    this can be exponentially slower.

``stop_on_first`` *(default: True)*
    If ``True``, stop exploring as soon as the first invariant violation
    is found.  Set to ``False`` to continue exploring and collect all
    failing interleavings.  Use ``False`` when you want an exhaustive
    census of every distinct race in the program.

``max_executions`` *(default: None)*
    Safety cap on total executions. Useful for CI where you want a time
    bound.  When the total number of Mazurkiewicz traces is very large
    (thousands or more), set this to a reasonable budget and use a
    non-DFS search strategy (see ``search`` below) to maximize the
    chance of finding a bug within the budget.

``max_branches`` *(default: 100,000)*
    Maximum scheduling points per execution. Prevents runaway on programs
    with very long traces.

``timeout_per_run`` *(default: 5.0)*
    Timeout in seconds for each individual execution.

``reproduce_on_failure`` *(default: 10)*
    When a counterexample is found, replay the same schedule this many
    times to measure how reproducible the failure is. The reproduction
    count and percentage appear in ``result.explanation``. Set to 0 to
    skip.

``search`` *(default: None)*
    Controls the order in which wakeup-tree branches are explored.
    All strategies visit the same set of Mazurkiewicz trace equivalence
    classes; only the *order* differs.

    - ``None`` or ``"dfs"`` --- classic depth-first search, always picking
      the lowest thread ID.  **Use DFS when the goal is exhaustive
      exploration** (``stop_on_first=False``): it produces the optimal
      (minimum) number of executions because the sleep-set pruning is
      maximally effective under this ordering.

    - ``"bit-reversal"`` or ``"bit-reversal:<seed>"`` --- visit siblings
      in a low-discrepancy bit-reversal permutation.  Maximally spreads
      exploration across diverse conflict points early.

    - ``"round-robin"`` or ``"round-robin:<seed>"`` --- cycle through
      available threads in rotating order.

    - ``"stride"`` or ``"stride:<seed>"`` --- visit every *s*-th sibling
      (*s* coprime to the branching factor, derived from the seed).

    - ``"conflict-first"`` --- reverse of DFS (highest thread ID first).

    **Use a non-DFS strategy when the trace space is large and you have
    a limited execution budget** (``stop_on_first=True``, or a low
    ``max_executions``).  DFS explores traces in a fixed order determined
    by thread IDs, so it may spend many executions on "similar"
    interleavings before reaching the one that triggers a bug.  The
    alternative strategies spread exploration across different conflict
    points earlier, finding bugs faster on average.

    The trade-off: non-DFS strategies may explore a small number of
    redundant trace classes (~5% overhead on complex lock patterns like
    dining philosophers) because changing the sibling ordering can reduce
    sleep-set effectiveness.  For exhaustive exploration where minimizing
    total executions matters, DFS is the best choice.

    See :doc:`search` for a detailed comparison of all strategies.


Interpreting results
--------------------

``explore_dpor()`` returns an ``InterleavingResult`` (the same type used by
``explore_interleavings``):

.. code-block:: python

   @dataclass
   class InterleavingResult:
       property_holds: bool                              # True if invariant held everywhere
       counterexample: list[int] | None = None           # first failing schedule
       num_explored: int = 0                             # total interleavings tried
       unique_interleavings: int = 0                     # distinct schedules (= num_explored for DPOR)
       failures: list[tuple[int, list[int]]] = ...       # all (execution_num, schedule) pairs
       explanation: str | None = None                    # human-readable trace of the race
       reproduction_attempts: int = 0                    # number of replay attempts
       reproduction_successes: int = 0                   # how many replays reproduced the failure
       sql_anomaly: SqlAnomaly | None = None             # classified SQL isolation anomaly (if any)

``counterexample`` is a list of thread IDs representing the order in
which threads were scheduled. For example, ``[0, 0, 1, 1]`` means thread 0
ran for two steps, then thread 1 ran for two steps.

When a race is found, ``explanation`` contains a formatted trace showing the
interleaved source lines, the conflict pattern (lost update, write-write, etc.),
and reproduction statistics. This is the same output for both ``explore_dpor``
and ``explore_interleavings``.

If ``num_explored`` is 1, your threads probably don't share any
state --- the engine saw no conflicts and skipped everything. This is a sign
that either the test is trivially correct or the shared state is not being
accessed in a way the tracer can see (e.g. through a C extension).


Example: verifying that a lock fixes a race
--------------------------------------------

A common pattern is to first show that a race exists, then show that adding
a lock eliminates it:

.. code-block:: python

   import threading
   from frontrun.dpor import explore_dpor

   class UnsafeCounter:
       def __init__(self):
           self.value = 0

       def increment(self):
           temp = self.value
           self.value = temp + 1

   class SafeCounter:
       def __init__(self):
           self.value = 0
           self.lock = threading.Lock()

       def increment(self):
           with self.lock:
               temp = self.value
               self.value = temp + 1

   def test_unsafe_counter_has_race():
       result = explore_dpor(
           setup=UnsafeCounter,
           threads=[lambda c: c.increment(), lambda c: c.increment()],
           invariant=lambda c: c.value == 2,
       )
       assert result.property_holds, result.explanation  # fails — has a race!

   def test_safe_counter_is_correct():
       result = explore_dpor(
           setup=SafeCounter,
           threads=[lambda c: c.increment(), lambda c: c.increment()],
           invariant=lambda c: c.value == 2,
       )
       assert result.property_holds, result.explanation


Example: multiple shared objects
---------------------------------

DPOR tracks objects independently, so races on different attributes are
detected separately:

.. code-block:: python

   from frontrun.dpor import explore_dpor

   class Bank:
       def __init__(self):
           self.a = 100
           self.b = 100

       def transfer(self, amount):
           temp_a = self.a
           temp_b = self.b
           self.a = temp_a - amount
           self.b = temp_b + amount

   def test_concurrent_transfers_conserve_total():
       result = explore_dpor(
           setup=Bank,
           threads=[lambda b: b.transfer(50), lambda b: b.transfer(50)],
           invariant=lambda b: b.a + b.b == 200,
       )
       assert result.property_holds, result.explanation  # fails — total is not conserved


C-level I/O interception via ``LD_PRELOAD``
----------------------------------------------

When run under the ``frontrun`` CLI, the ``LD_PRELOAD`` library
(``libfrontrun_io.so``) intercepts C-level I/O --- including opaque
database drivers like psycopg2/libpq. DPOR consumes these events
automatically when ``detect_io=True`` (the default).

**How it works:**

1. ``explore_dpor()`` starts an ``IOEventDispatcher`` that creates
   an ``os.pipe()`` and passes the write-end FD to the Rust
   ``LD_PRELOAD`` library via the ``FRONTRUN_IO_FD`` environment
   variable. The Rust library writes event records to the pipe for
   every intercepted ``send``, ``recv``, ``read``, ``write``, etc.

2. A ``_PreloadBridge`` maps OS thread IDs (from each event's ``tid``
   field) to DPOR logical thread IDs. Each DPOR-managed thread
   registers its ``threading.get_native_id()`` → ``thread_id`` at
   start.

3. At each DPOR scheduling point, the bridge drains buffered events
   for the current thread and feeds them to the Rust DPOR engine via
   ``engine.report_io_access()``. The engine tracks the shared
   resource (e.g. ``socket:127.0.0.1:5432``) the same way it tracks
   Python-level attribute accesses --- via vector clocks and
   conflict detection.

This is what allows DPOR to detect the SQLAlchemy/psycopg2 lost-update
race (see :doc:`orm_race`): even though psycopg2 calls libc ``send()``
/ ``recv()`` directly (bypassing Python's ``socket`` module), the
``LD_PRELOAD`` library intercepts those calls and DPOR treats the
shared socket endpoint as a conflict point.

.. note::

   The bytecode explorer does *not* consume ``LD_PRELOAD`` events.
   It relies on random scheduling and may still find C-level races
   by chance, but it has no awareness of the underlying I/O conflicts.


ORM helpers: ``django_dpor`` and ``sqlalchemy_dpor``
------------------------------------------------------

Writing correct ``explore_dpor()`` tests against a real database requires
boilerplate: each thread needs its own connection, stale connections from a
previous execution must be closed, and optional lock timeouts must be injected
before the thread runs. The ``frontrun.contrib`` package provides ready-made
wrappers that handle this automatically.

``django_dpor``
~~~~~~~~~~~~~~~

.. code-block:: python

   from frontrun.contrib.django import django_dpor

   result = django_dpor(
       setup=_State,
       threads=[thread_a, thread_b],
       invariant=lambda s: s.login_count == 2,
       lock_timeout=500,  # optional, milliseconds
   )
   assert result.property_holds, result.explanation

``django_dpor`` wraps ``explore_dpor`` and:

* Calls ``connections.close_all()`` before each execution so threads open
  fresh connections (avoids sharing a stale connection across DPOR replays).
* Calls ``conn.close()`` / ``conn.ensure_connection()`` at the start of each
  thread so every thread has its own independent connection.
* Optionally executes ``SET lock_timeout = '<N>ms'`` on each connection,
  converting C-level row-lock blocking into a fast PostgreSQL error rather than
  a hang.

All extra keyword arguments are forwarded to ``explore_dpor``.

``sqlalchemy_dpor``
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from frontrun.contrib.sqlalchemy import sqlalchemy_dpor, get_connection

   result = sqlalchemy_dpor(
       engine=engine,
       setup=_State,
       threads=[thread_a, thread_b],
       invariant=lambda s: _read_count() == 2,
       lock_timeout=500,  # optional, milliseconds
   )
   assert result.property_holds, result.explanation

``sqlalchemy_dpor`` wraps ``explore_dpor`` and:

* Calls ``engine.dispose()`` before each execution to close pooled connections.
* Opens a fresh ``engine.connect()`` connection for each thread and stores it
  in a ``ContextVar`` so the thread can retrieve it with ``get_connection()``.
* Optionally executes ``SET lock_timeout = '<N>ms'`` via
  ``conn.exec_driver_sql()``.

Inside a thread function, retrieve the per-thread connection with:

.. code-block:: python

   from frontrun.contrib.sqlalchemy import get_connection

   def thread_fn(state):
       conn = get_connection()
       result = conn.execute(text("SELECT login_count FROM users WHERE id = 1"))
       ...

Both helpers accept ``detect_io=True`` (the default) and all other
``explore_dpor`` keyword arguments.

Async usage
~~~~~~~~~~~

The same ``django_dpor`` and ``sqlalchemy_dpor`` functions also support
async code.  Pass ``tasks=`` instead of ``threads=`` and ``await`` the
result:

.. code-block:: python

   from frontrun.contrib.django import django_dpor

   result = await django_dpor(
       setup=_State,
       tasks=[task_a, task_b],
       invariant=lambda s: s.login_count == 2,
       lock_timeout=500,
   )
   assert result.property_holds, result.explanation

.. code-block:: python

   from frontrun.contrib.sqlalchemy import sqlalchemy_dpor, get_async_connection

   result = await sqlalchemy_dpor(
       engine=async_engine,
       setup=_State,
       tasks=[task_a, task_b],
       invariant=lambda s: _read_count() == 2,
       lock_timeout=500,
   )

Inside an async task, use ``get_async_connection()`` to retrieve the
per-task connection.


Redis key-level detection
--------------------------

DPOR intercepts ``execute_command()`` on redis-py clients and classifies each
Redis command as a read or write on one or more specific keys.  The DPOR engine
then treats each Redis key as an independent resource, the same way it treats
Python-level attribute accesses.

**Sync usage** (``detect_io=True`` is the default; no extra parameter needed):

.. code-block:: python

   import redis
   from frontrun.dpor import explore_dpor

   def test_redis_lost_update(redis_port):
       class State:
           def __init__(self):
               r = redis.Redis(port=redis_port, decode_responses=True)
               r.set("counter", "0")
               r.close()

       def increment(state):
           r = redis.Redis(port=redis_port, decode_responses=True)
           val = int(r.get("counter"))
           r.set("counter", str(val + 1))
           r.close()

       def invariant(state):
           r = redis.Redis(port=redis_port, decode_responses=True)
           result = int(r.get("counter"))
           r.close()
           return result == 2

       result = explore_dpor(
           setup=State,
           threads=[increment, increment],
           invariant=invariant,
           detect_io=True,          # enables Redis key-level patching (default)
       )
       assert not result.property_holds   # race detected!

**Async usage** (``detect_io=True`` covers Redis from 0.5):

.. code-block:: python

   import asyncio

   import redis.asyncio as aioredis
   from frontrun import explore

   def test_async_redis_check_then_act(redis_port):
       async def maybe_init(state):
           r = aioredis.Redis(port=redis_port, decode_responses=True)
           if not await r.exists("resource"):
               await r.set("resource", "initialized")
           await r.aclose()

       asyncio.run(explore(
           setup=lambda: None,
           workers=maybe_init,
           count=2,
           invariant=lambda s: True,
           detect_io=True,
       ))

DPOR inserts fine-grained scheduling points around each Redis command, so it
can explore the gap between an ``EXISTS`` check and the subsequent ``SET`` ---
the classic TOCTOU (check-then-act) race window.

**What counts as a conflict:**

- Any two operations on the *same* key where at least one is a write.
- Pipeline batches: each command in the pipeline is reported individually.
- ``MULTI``/``EXEC`` transactions: commands between ``MULTI`` and ``EXEC`` are
  buffered and reported atomically.

**What is independent (no conflict, no extra interleavings):**

- Two reads on the same key (e.g. two ``GET``s).
- Any operations on *different* keys, even on the same Redis server.
- Server-level commands (``PING``, ``INFO``, ``CONFIG``, etc.) that carry no
  key-level semantics.

For the full command classification and technical details, see :doc:`redis`.

Prefer ``assert_holds()`` over manual asserts
---------------------------------------------

:meth:`InterleavingResult.assert_holds` is the recommended way to check a DPOR
result in a test.  It raises ``AssertionError`` with the full race explanation
when the invariant failed, and does nothing on success::

   result = explore_dpor(setup, [thread1, thread2], invariant)
   result.assert_holds()          # preferred
   # instead of: assert result.property_holds, result.explanation

Pass ``msg_prefix`` to distinguish multiple assertions in one test::

   result.assert_holds(msg_prefix="transfer race: ")
