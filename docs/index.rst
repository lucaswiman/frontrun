Frontrun Documentation
======================

Deterministic concurrency testing for Python.

Race conditions are hard to test because they depend on timing. A test that
passes 95% of the time is worse than a test that always fails, because it
breeds false confidence. Frontrun replaces timing-dependent thread interleaving
with deterministic scheduling, so race conditions either always happen or
never happen.

Three approaches, in order of decreasing interpretability:

1. **DPOR** --- systematic exploration of every meaningfully different
   interleaving, with causal conflict analysis.
2. **Bytecode exploration** --- random opcode-level schedules that often find
   races very efficiently, including races invisible to DPOR.
3. **Trace markers** --- comment-based synchronization points for reproducing
   a known race window.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   approaches
   dpor_guide
   dpor
   examples
   orm_race
   internals
   api_reference
   CASE_STUDIES


Getting Started
---------------

The simplest entry point is **trace markers** --- comment-based synchronization
points that let you force a specific execution order:

.. code-block:: python

   from frontrun.common import Schedule, Step
   from frontrun.trace_markers import TraceExecutor

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           temp = self.value  # frontrun: read_value
           temp += 1
           self.value = temp  # frontrun: write_value

   def test_counter_lost_update():
       counter = Counter()
       schedule = Schedule([
           Step("thread1", "read_value"),
           Step("thread2", "read_value"),
           Step("thread1", "write_value"),
           Step("thread2", "write_value"),
       ])

       executor = TraceExecutor(schedule)
       executor.run("thread1", counter.increment)
       executor.run("thread2", counter.increment)
       executor.wait(timeout=5.0)

       assert counter.value == 1  # One increment lost

For automatic race finding without manual markers, see :doc:`dpor_guide` (systematic)
or :doc:`approaches` (random bytecode exploration).


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
