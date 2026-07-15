Frontrun Documentation
======================

Deterministic concurrency testing for Python.

Race conditions are hard to test because they depend on timing. A test that
passes 95% of the time is worse than a test that always fails, because it
breeds false confidence. Frontrun replaces timing-dependent thread interleaving
with deterministic scheduling, so race conditions either always happen or
never happen. Free-threaded Python (3.13t, 3.14t) raises the stakes: races the
GIL kept narrow become real, and frontrun --- which supports free-threaded
builds --- exists to enumerate those interleavings and prove which one is buggy.

Four approaches, in order of decreasing interpretability:

1. **DPOR** --- systematic exploration of every meaningfully different
   interleaving, with causal conflict analysis.
2. **Bytecode exploration** --- random opcode-level schedules that often find
   races very efficiently, including races invisible to DPOR.
3. **Marker schedule exploration** --- exhaustive exploration of all
   interleavings at the ``# frontrun:`` marker level.  Much smaller search
   space than bytecode exploration, with completeness guarantees.
4. **Trace markers** --- comment-based synchronization points for reproducing
   a known race window.

The DPOR engine also drives **cross-process exploration**: the same
``frontrun.explore(...)`` interface applied to separate Python processes that
contend on shared external state (SQL and Redis).

Time itself can be scheduled: with ``clock="virtual"`` sleeps cost zero wall
time and TTL expiry becomes reachable, and with ``clock="explored"`` timer
firings are explored against other operations like any other interleaving
choice — see :doc:`virtual_clock`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   design-principles
   approaches
   case_studies
   dpor_guide
   dpor
   vector-clocks
   search
   examples
   orm_race
   sql-technical-details
   redis
   cross_process
   virtual_clock
   trace_filtering
   internals
   api_reference


Getting Started
---------------

The front door is :func:`frontrun.explore` --- point it at shared state, some
workers, and an invariant, and DPOR explores every meaningfully different
interleaving, no annotations needed:

.. code-block:: python

   import frontrun

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           temp = self.value
           self.value = temp + 1

   def test_increment_is_atomic():
       result = frontrun.explore(
           setup=Counter,
           workers=Counter.increment,
           count=2,
           invariant=lambda c: c.value == 2,
       )
       result.assert_holds()  # fails with the exact racy interleaving

Start with :doc:`quickstart`, then :doc:`dpor_guide` for systematic race
finding. For races frontrun has found in real, unmodified PyPI packages, see
:doc:`case_studies`.


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
