Approaches to Concurrency Control
==================================

Interlace provides two approaches for controlling thread interleaving. This document explains the characteristics, trade-offs, and use cases for each.


Trace Markers (Recommended)
---------------------------

Trace Markers are the recommended approach for most use cases. They use lightweight comment-based markers to define synchronization points in your code, requiring no semantic code changes to production code.

**How It Works:**

Each thread is run with a ``sys.settrace`` callback that fires on every source
line. The callback scans each line for ``# interlace: <name>`` comments using a
``MarkerRegistry`` that caches marker locations per file. When a marker is hit,
the thread calls ``ThreadCoordinator.wait_for_turn()`` which blocks until the
schedule says it's that thread's turn to proceed past that marker. This gives
deterministic control over the order threads reach each synchronization point,
without changing any executable code — markers are just comments.

.. code-block:: python

   from interlace.trace_markers import Schedule, Step, TraceExecutor

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           # interlace: after_read
           temp = self.value
           temp += 1
           # interlace: before_write
           self.value = temp

**When to Use:**

- You want explicit control over synchronization points
- You're testing code you can modify slightly (adding comments)
- You need a stable, well-understood approach
- You want minimal performance impact
- You're testing for specific race conditions at known points


**Async Support:**

Async trace markers use the same comment-based syntax as sync trace markers. Race
conditions in async code only occur at ``await`` points, so markers should be
placed before awaits to define synchronization boundaries.

.. code-block:: python

   from interlace.async_trace_markers import AsyncTraceExecutor
   from interlace.common import Schedule, Step

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
           # interlace: read_counter
           current = await self.get_count()
           # interlace: write_counter
           await self.set_count(current + 1)

   # Alternative: markers inside the async methods themselves
   class AsyncCounterAlt:
       def __init__(self):
           self.value = 0

       async def get_count(self):
           # interlace: read_counter
           return self.value

       async def set_count(self, value):
           # interlace: write_counter
           self.value = value

       async def increment(self):
           current = await self.get_count()
           await self.set_count(current + 1)


Bytecode Instrumentation (Experimental)
----------------------------------------

.. warning::

   Bytecode instrumentation is **experimental** and should be used with caution. The API may change, and behavior is not guaranteed to be stable across Python versions.

Bytecode instrumentation automatically inserts checkpoints into functions using Python bytecode rewriting. No manual marker insertion is needed.

**Key Characteristics:**

- **Experimental** - API may change, use with care
- **Automatic** - No manual checkpoint insertion required
- **Unmodified code** - Can test third-party code without changes
- **Property-based exploration** - Can discover edge cases automatically
- **Higher overhead** - Bytecode instrumentation adds runtime cost

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

For property-based exploration, ``explore_interleavings()`` generates random
opcode-level schedules and checks that an invariant holds under each one. If
any schedule violates the invariant, it returns the counterexample schedule for
deterministic reproduction.

.. code-block:: python

   from interlace.bytecode import explore_interleavings

   class Counter:
       def __init__(self, value=0):
           self.value = value

       def increment(self):
           temp = self.value
           self.value = temp + 1

   # Automatically explore different interleavings
   result = explore_interleavings(
       setup=lambda: Counter(value=0),
       threads=[
           lambda c: c.increment(),
           lambda c: c.increment(),
       ],
       invariant=lambda c: c.value == 2,  # Should hold if no races
       max_attempts=200,
       max_ops=200,
       seed=42,
   )

   if not result.property_holds:
       print(f"Race condition found after {result.num_explored} attempts")
       print(f"Expected: 2, Got: {result.counterexample.value}")

**When to Use (with caution):**

- You want to automatically discover race conditions
- You're testing unmodified third-party code
- You're comfortable with experimental features
- You need to explore many possible interleavings
- Performance is not a critical concern

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

   The async variant (``interlace.async_bytecode``) uses ``await_point()`` markers rather
   than opcodes, so its schedules are stable — see that module for details.

**Limitations and Caveats:**

- Results may vary across Python versions
- Some bytecode patterns may not be instrumented correctly
- Performance impact is higher than trace markers
- Less predictable behavior than explicit markers
- Async bytecode instrumentation is also experimental


Comparison Table
----------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Trace Markers
     - Bytecode Instrumentation
   * - Status
     - Stable, recommended
     - Experimental
   * - Manual checkpoint insertion
     - Yes (comments)
     - No (automatic)
   * - Works with unmodified code
     - No (needs comments)
     - Yes
   * - Performance
     - Low overhead
     - Higher overhead
   * - Determinism
     - High
     - Lower (Python version dependent)
   * - Async support
     - Yes, stable
     - Yes, experimental
   * - Property-based exploration
     - Manual schedules
     - Automatic
   * - Learning curve
     - Low
     - Medium
   * - Recommended for most cases
     - ✓
     -


Choosing the Right Approach
---------------------------

**Use Trace Markers if:**

- You want stable, predictable behavior
- You're comfortable adding comments to code
- You have specific synchronization points in mind
- You need deterministic test results
- You're testing code you can modify

**Use Bytecode Instrumentation if:**

- You need to test unmodified third-party code
- You want automatic race condition discovery
- You're comfortable with experimental features
- You have time to handle potential API changes
- Performance is not a concern

**In most cases, start with Trace Markers.** They provide clear, explicit control and stable behavior. Use bytecode instrumentation when you specifically need its capabilities.
