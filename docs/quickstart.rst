Quick Start
===========

This guide will get you up and running with Interlace using **Trace Markers**.


Basic Example: Triggering a Race Condition
-------------------------------------------

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

       # Both threads read before either writes, causing a lost update
       schedule = Schedule([
           Step("thread1", "read_value"),    # T1 reads 0
           Step("thread2", "read_value"),    # T2 reads 0 (both see same value!)
           Step("thread1", "write_value"),   # T1 writes 1
           Step("thread2", "write_value"),   # T2 writes 1 (overwrites T1's update!)
       ])

       executor = TraceExecutor(schedule)
       executor.run("thread1", counter.increment)
       executor.run("thread2", counter.increment)
       executor.wait(timeout=5.0)

       assert counter.value == 1  # One increment lost


Understanding Trace Markers
----------------------------

Trace markers are comments of the form ``# interlace: <name>`` that tell
Interlace where synchronization points are. A marker **gates** the code that
follows it: when a thread reaches a marker, it pauses until the scheduler grants
it a turn. Only then does the gated code execute.

Two placement styles are supported:

1. **Inline with code** (marker on the same line as the operation it gates):

   .. code-block:: python

       def increment(self):
           temp = self.value  # interlace: read_value
           temp += 1
           self.value = temp  # interlace: write_value

   Here ``read_value`` gates the read of ``self.value``, and ``write_value``
   gates the write.

2. **On a separate line** before the operation:

   .. code-block:: python

       def increment(self):
           # interlace: read_value
           temp = self.value
           temp += 1
           # interlace: write_value
           self.value = temp

   The semantics are the same: the marker gates the next executable line.

Name markers after the operation they gate (``read_value``, ``write_balance``,
``acquire_lock``, etc.) rather than using temporal prefixes like ``before_`` or
``after_``.


Creating Schedules
-------------------

A schedule defines the execution order of marked synchronization points:

.. code-block:: python

   from interlace.trace_markers import Schedule, Step

   schedule = Schedule([
       Step("thread1", "marker_name_1"),
       Step("thread1", "marker_name_2"),
       Step("thread2", "marker_name_1"),
       Step("thread2", "marker_name_2"),
   ])

Each ``Step`` specifies:

- The thread/task name
- The marker name to execute at that step


Running with Controlled Interleaving
-------------------------------------

Execute your code with a specific schedule:

.. code-block:: python

   from interlace.trace_markers import TraceExecutor

   executor = TraceExecutor(schedule)

   # Register tasks to run
   executor.run("thread1", task_function_1)
   executor.run("thread2", task_function_2)

   # Wait for completion
   executor.wait(timeout=5.0)


Async Support
-------------

Async trace markers use the same comment-based syntax. Each async task runs in
its own thread (via ``asyncio.run``), with ``sys.settrace`` controlling
interleaving between tasks.

A marker gates the next ``await`` expression. When a task reaches a marker, it
pauses until the scheduler grants it a turn; only then does the gated ``await``
execute. Between two markers the task runs without interruption from other
scheduled tasks.

Here is a complete async example — the same lost-update race as the sync
version, but with ``await`` boundaries:

.. code-block:: python

   from interlace.async_trace_markers import AsyncTraceExecutor
   from interlace.common import Schedule, Step

   class AsyncCounter:
       def __init__(self):
           self.value = 0

       async def get_value(self):
           return self.value

       async def set_value(self, new_value):
           self.value = new_value

       async def increment(self):
           # interlace: read_value
           temp = await self.get_value()
           # interlace: write_value
           await self.set_value(temp + 1)

   def test_async_counter_lost_update():
       counter = AsyncCounter()

       # Both tasks read before either writes — triggers the lost update
       schedule = Schedule([
           Step("task1", "read_value"),
           Step("task2", "read_value"),
           Step("task1", "write_value"),
           Step("task2", "write_value"),
       ])

       executor = AsyncTraceExecutor(schedule)
       executor.run({
           "task1": counter.increment,
           "task2": counter.increment,
       })

       # Both tasks read 0, then both write 1 — one increment is lost
       assert counter.value == 1

   def test_async_counter_serialized():
       counter = AsyncCounter()

       # Serialized: task1 completes before task2 starts
       schedule = Schedule([
           Step("task1", "read_value"),
           Step("task1", "write_value"),
           Step("task2", "read_value"),
           Step("task2", "write_value"),
       ])

       executor = AsyncTraceExecutor(schedule)
       executor.run({
           "task1": counter.increment,
           "task2": counter.increment,
       })

       assert counter.value == 2  # No lost update
