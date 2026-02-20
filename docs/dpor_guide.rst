DPOR in Practice
================

This is a practical guide to using ``explore_dpor()`` for systematic
concurrency testing. For the underlying algorithm and theory, see
:doc:`dpor`.


What DPOR does
--------------

``explore_dpor()`` takes a setup function, a list of thread functions, and an
invariant. It runs the threads under every meaningfully different interleaving
and checks the invariant after each one. If the invariant ever fails, it
returns a counterexample.

.. code-block:: python

   from frontrun.dpor import explore_dpor

   result = explore_dpor(
       setup=MyState,                         # called fresh each execution
       threads=[thread_a, thread_b],          # each receives the state
       invariant=lambda s: s.is_consistent(), # checked after all threads finish
   )

   if not result.property_holds:
       print(f"Bug found after {result.executions_explored} executions")

"Meaningfully different" means: two interleavings that only differ in the order
of *independent* operations (e.g. two reads, or accesses to different objects)
are treated as equivalent and only one is explored. This is what makes DPOR
fast --- it skips redundant work that a naive scheduler would repeat.


What it can and cannot find
---------------------------

DPOR works by instrumenting Python bytecode to intercept shared-memory
operations. This determines what is visible to the engine and what is not.

**What DPOR sees:**

- Attribute reads and writes (``self.x``, ``obj.field = ...``)
- Subscript reads and writes (``d[key]``, ``lst[i] = ...``)
- Lock acquire and release (``threading.Lock``, ``threading.RLock``)
- Thread spawn and join

**What DPOR does not see:**

- **Database operations.** A ``SELECT`` followed by an ``UPDATE`` looks like a
  single opaque C function call to the bytecode tracer. The engine cannot
  see that two threads are racing on the same row.
- **File system access.** Reading and writing files goes through C-level I/O
  that the tracer cannot intercept.
- **Network and IPC.** HTTP requests, message queues, Redis commands, etc.
  are all invisible.
- **C extensions.** Any shared state modified inside C code (NumPy arrays,
  database drivers, etc.) is not tracked.

In short: DPOR finds races in *pure Python* shared-memory concurrency. For
races that involve external systems, use :doc:`trace markers <approaches>`
with explicit scheduling instead --- you annotate the points where
interleaving matters and DPOR-style automation is not needed because the
number of interesting orderings is small enough to enumerate by hand.


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
    A predicate over the shared state. Checked after all threads finish.
    Return ``True`` if the state is valid, ``False`` if there is a bug.

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

``cooperative_locks`` *(default: True)*
    Replace ``threading.Lock``, ``threading.Event``, ``queue.Queue``, etc.
    with scheduler-aware versions. This is required for DPOR to control
    scheduling around lock operations. Disable only if your code does not
    use any standard library synchronization primitives.


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

``counterexample_schedule`` is a list of thread IDs representing the order in
which threads were scheduled. For example, ``[0, 0, 1, 1]`` means thread 0
ran for two steps, then thread 1 ran for two steps.


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
