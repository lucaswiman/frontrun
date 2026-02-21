DPOR in Practice
================

This is a practical guide to using ``explore_dpor()`` for systematic
concurrency testing. For the underlying algorithm and theory, see
:doc:`dpor`.


What DPOR does
--------------

DPOR and the invariant have separate jobs:

- **DPOR decides which interleavings to run.** It watches every shared-memory
  access, detects which operations *conflict* (access the same object with at
  least one write), and uses that information to skip redundant orderings.
  Two interleavings that only differ in the order of independent operations
  (e.g. two reads, or accesses to different objects) are equivalent, so DPOR
  runs only one representative from each equivalence class.

- **The invariant decides whether a bug occurred.** After all threads finish,
  ``explore_dpor()`` calls your invariant on the final state. DPOR has no
  built-in notion of "correct" --- it doesn't know that ``counter == 1`` is
  wrong and ``counter == 2`` is right. You supply that judgement via the
  invariant.

(The name "invariant" is standard in tools like this --- loom, CHESS, etc. ---
even though it is technically a *postcondition* checked once after the threads
finish, not a continuously-monitored loop invariant.)

Putting the two together:

.. code-block:: python

   from frontrun.dpor import explore_dpor

   result = explore_dpor(
       setup=MyState,                         # called fresh each execution
       threads=[thread_a, thread_b],          # each receives the state
       invariant=lambda s: s.is_consistent(), # checked after all threads finish
   )

   if not result.property_holds:
       print(f"Bug found after {result.executions_explored} executions")

1. DPOR picks an interleaving (based on conflict analysis).
2. ``setup()`` creates fresh state; the threads run under that interleaving.
3. The invariant checks the final state.
4. Repeat until all distinct interleavings are covered.


What it can and cannot find
---------------------------

DPOR explores alternative interleavings only where it detects a *conflict*
--- two threads accessing the same object with at least one write. It detects
conflicts by instrumenting Python bytecode, so only operations that are
visible at the bytecode level register as conflicts.

**Operations DPOR sees (and will explore reorderings of):**

- Attribute reads and writes (``self.x``, ``obj.field = ...``)
- Subscript reads and writes (``d[key]``, ``lst[i] = ...``)
- Lock acquire and release (``threading.Lock``, ``threading.RLock``)
- Thread spawn and join
- **Socket I/O** (``connect``, ``send``, ``sendall``, ``sendto``, ``recv``,
  ``recv_into``, ``recvfrom``) --- detected via automatic monkey-patching
  when ``detect_io=True`` (the default). Two threads accessing the same
  ``host:port`` endpoint conflict; different endpoints are independent.
- **File opens** (``builtins.open``) --- read vs write determined by mode,
  resource identity from the resolved file path.

Beyond invariant violations, DPOR also detects **deadlocks** (via wait-for-graph
cycle detection) and **crashes** (unhandled exceptions in any thread are
re-raised after the execution completes).

**Operations DPOR does not see (and will therefore not explore):**

- **Database operations.** Two threads calling ``cursor.execute("UPDATE ...")``
  on the same row look like independent C function calls to the tracer ---
  DPOR sees no conflict between them and only runs one interleaving.
- **Opaque C-extension I/O.** Database drivers, Redis clients, and other
  libraries that manage sockets entirely in C code bypass the
  monkey-patches, so their operations appear independent.
- **C-extension shared state.** Shared state modified inside C code (NumPy
  arrays, etc.) is not tracked at the bytecode level.

The consequence is not that DPOR "can't run" on such code --- it will run
fine, it just won't explore the interesting schedules. Because the external
operations look independent, DPOR concludes that reordering them cannot change
the outcome and skips all the alternative interleavings where the bugs hide.

For these cases, use :doc:`trace markers <approaches>` with explicit
scheduling instead --- you annotate the points where interleaving matters and
enumerate the orderings by hand.


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

   assert not result.property_holds
   assert result.executions_explored == 2  # only 2 of 6 interleavings needed

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

``max_executions`` *(default: None)*
    Safety cap on total executions. Useful for CI where you want a time
    bound.

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


Interpreting results
--------------------

``explore_dpor()`` returns a ``DporResult``:

.. code-block:: python

   @dataclass
   class DporResult:
       property_holds: bool                              # True if invariant held everywhere
       executions_explored: int = 0                      # total interleavings tried
       counterexample_schedule: list[int] | None = None  # first failing schedule
       failures: list[tuple[int, list[int]]] = ...       # all (execution_num, schedule) pairs
       explanation: str | None = None                    # human-readable trace of the race
       reproduction_attempts: int = 0                    # number of replay attempts
       reproduction_successes: int = 0                   # how many replays reproduced the failure

``counterexample_schedule`` is a list of thread IDs representing the order in
which threads were scheduled. For example, ``[0, 0, 1, 1]`` means thread 0
ran for two steps, then thread 1 ran for two steps.

When a race is found, ``explanation`` contains a formatted trace showing the
interleaved source lines, the conflict pattern (lost update, write-write, etc.),
and reproduction statistics. This is the same output for both ``explore_dpor``
and ``explore_interleavings``.


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
       assert not result.property_holds

   def test_safe_counter_is_correct():
       result = explore_dpor(
           setup=SafeCounter,
           threads=[lambda c: c.increment(), lambda c: c.increment()],
           invariant=lambda c: c.value == 2,
       )
       assert result.property_holds


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
       assert not result.property_holds  # total is not conserved without locking


Tips
----

**Keep thread functions short.** Every bytecode instruction is a potential
scheduling point. Long functions produce deep exploration trees and slow
things down. Extract the concurrent kernel --- the part that actually touches
shared state --- and test that.

**Use ``preemption_bound=2`` (the default).** Empirical research shows this
catches nearly all real bugs. Increasing the bound gives diminishing returns
and exponentially more executions.

**Use ``max_executions`` in CI.** Even with preemption bounding, the
exploration can be large. Setting a cap ensures your test suite has a bounded
runtime. If the cap is hit without finding a bug, the test still provides
useful (though incomplete) coverage.

**Inspect ``executions_explored``.** If DPOR reports that only 1 execution
was explored, your threads probably don't share any state --- the engine
saw no conflicts and skipped everything. This is a sign that either the
test is correct or the shared state is not being accessed in a way the
tracer can see (e.g. through a C extension).

**Avoid external side effects in thread functions.** DPOR replays each
interleaving from scratch. If thread functions write to files, send network
requests, or modify global state outside the ``setup`` object, replays
will interfere with each other.
