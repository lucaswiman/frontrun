Interlace Documentation
=======================

A library for deterministic concurrency testing that helps you reliably reproduce and test race conditions.

Interlace provides tools for controlling thread interleaving at a fine-grained level, allowing you to deterministically reproduce race conditions in tests and verify that your synchronization primitives work correctly.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   approaches
   api_reference
   examples
   CASE_STUDIES


Key Features
------------

- **Deterministically reproduce race conditions** - Force specific execution ordering to make race conditions happen reliably in tests
- **Test concurrent code exhaustively** - Explore different execution orders to find bugs
- **Verify synchronization correctness** - Ensure that proper locking prevents race conditions
- **Lightweight integration** - No need to modify third-party code when using trace markers

Instead of relying on timing-based race detection (which is unreliable), Interlace lets you control exactly when threads execute, making concurrency testing deterministic and reproducible.


Getting Started
---------------

**Trace Markers** are a lightweight, comment-based approach that requires minimal code changes:

.. code-block:: python

   from interlace.trace_markers import Schedule, Step, TraceExecutor

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           temp = self.value  # interlace: read_value
           temp += 1
           self.value = temp  # interlace: write_value

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

For more information, see :doc:`quickstart` and :doc:`approaches`.


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
