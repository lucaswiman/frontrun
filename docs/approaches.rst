Approaches to Concurrency Control
==================================

Frontrun provides two approaches for controlling thread interleaving.


Trace Markers
--------------

Trace Markers use lightweight comment-based markers to define synchronization
points in your code, requiring no semantic code changes.

**How It Works:**

Each thread is run with a ``sys.settrace`` callback that fires on every source
line. The callback scans each line for ``# frontrun: <name>`` comments using a
``MarkerRegistry`` that caches marker locations per file. When a marker is hit,
the thread calls ``ThreadCoordinator.wait_for_turn()`` which blocks until the
schedule says it's that thread's turn to proceed past that marker. This gives
deterministic control over the order threads reach each synchronization point,
without changing any executable code — markers are just comments.

A marker **gates** the code that follows it. Name markers after the operation
they gate (e.g. ``read_value``, ``write_balance``) rather than with temporal
prefixes like ``before_`` or ``after_``.

.. code-block:: python

   from frontrun.trace_markers import Schedule, Step, TraceExecutor

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           temp = self.value  # frontrun: read_value
           temp += 1
           self.value = temp  # frontrun: write_value


**Async Support:**

Async trace markers use the same comment-based syntax. Each async task runs in
its own thread (via ``asyncio.run``), with the same ``sys.settrace`` mechanism
controlling interleaving between tasks.

The synchronization contract:

- A marker gates the next ``await`` expression (or the line it's on if inline).
  When a task reaches a marker, it pauses until the scheduler grants it a turn.
  Only then does the gated ``await`` execute.
- Between two markers, the task runs without interruption from other scheduled
  tasks. Any intermediate ``await`` calls within that span complete normally.
- Because async code can only interleave at ``await`` points, markers should be
  placed to gate the ``await`` expressions whose ordering you want to control.

.. code-block:: python

   from frontrun.async_trace_markers import AsyncTraceExecutor
   from frontrun.common import Schedule, Step

   class AsyncCounter:
       def __init__(self):
           self.value = 0

       async def get_count(self):
           """Read the counter value (simulates async I/O like database read)."""
           return self.value

       async def set_count(self, value):
           """Write the counter value (simulates async I/O like database write)."""
           self.value = value

       async def increment(self):
           """Increment with a race condition between read and write."""
           # frontrun: read_counter
           current = await self.get_count()
           # frontrun: write_counter
           await self.set_count(current + 1)

   # Alternative: markers inside the async methods themselves
   class AsyncCounterAlt:
       def __init__(self):
           self.value = 0

       async def get_count(self):
           # frontrun: read_counter
           return self.value

       async def set_count(self, value):
           # frontrun: write_counter
           self.value = value

       async def increment(self):
           current = await self.get_count()
           await self.set_count(current + 1)


Bytecode Instrumentation (Experimental)
----------------------------------------

.. warning::

   Bytecode instrumentation is **experimental** and should be used with caution. The API may change, and behavior is not guaranteed to be stable across Python versions.

Bytecode instrumentation automatically inserts checkpoints into functions using Python bytecode rewriting. No manual marker insertion is needed.

**How It Works:**

Each thread is run with a ``sys.settrace`` callback that sets
``f_trace_opcodes = True`` on every frame, so the callback fires at every
*bytecode instruction* rather than every source line. At each opcode the thread
calls ``scheduler.wait_for_turn()``, which blocks until the schedule says it's
that thread's turn. Only user code is traced — stdlib and threading internals
are skipped.

Because the scheduler controls which thread runs each opcode, any blocking call
that happens in C code (like ``threading.Lock.acquire()``) would deadlock — the
blocked thread holds a scheduler turn but can't make progress. To prevent this,
all standard threading and queue primitives (``Lock``, ``RLock``,
``Semaphore``, ``BoundedSemaphore``, ``Event``, ``Condition``, ``Queue``,
``LifoQueue``, ``PriorityQueue``) are monkey-patched with cooperative versions
that spin-yield via the scheduler instead of blocking. The patching is scoped
to each test run: primitives are replaced before ``setup()`` and restored
afterwards.

``explore_interleavings()`` does property-based exploration in the style of
`Hypothesis <https://hypothesis.readthedocs.io/>`_: it generates random
opcode-level schedules and checks that an invariant holds under each one,
returning any counterexample schedule.

.. code-block:: python

   from frontrun.bytecode import explore_interleavings

   class Counter:
       def __init__(self, value=0):
           self.value = value

       def increment(self):
           temp = self.value
           self.value = temp + 1

   def test_counter_increment_is_not_atomic():
       result = explore_interleavings(
           setup=lambda: Counter(value=0),
           threads=[
               lambda c: c.increment(),
               lambda c: c.increment(),
           ],
           invariant=lambda c: c.value == 2,
           max_attempts=200,
           max_ops=200,
           seed=42,
       )

       assert not result.property_holds
       assert result.counterexample.value == 1

**Controlled Interleaving (Internal/Advanced):**

The ``controlled_interleaving`` context manager and ``run_with_schedule`` function allow
running threads under a specific opcode-level schedule. These are primarily intended for
debugging this library or building tooling on top of it, rather than for general use in tests.

.. note::

   Opcode-level schedules are not stable across Python versions. CPython does not guarantee
   that the same source code will compile to the same bytecode between minor releases, so a
   specific schedule that reproduces a race on Python 3.12 may not reproduce the same
   interleaving on 3.13. Counterexample schedules returned by ``explore_interleavings``
   are likewise best treated as ephemeral debugging artifacts rather than long-lived test fixtures.

   The async variant (``frontrun.async_bytecode``) uses ``await_point()`` markers rather
   than opcodes, so its schedules are stable — see that module for details.

**Limitations:**

- Results may vary across Python versions
- Some bytecode patterns may not be instrumented correctly
- Performance impact is higher than trace markers
- Async bytecode instrumentation is also experimental
