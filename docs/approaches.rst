How Frontrun Works
==================

Frontrun provides three approaches for controlling thread interleaving. They
operate at different levels of granularity and make different trade-offs between
interpretability, coverage, and the kinds of bugs they can find.


DPOR (Systematic Exploration)
------------------------------

DPOR (Dynamic Partial Order Reduction) *systematically* explores every
meaningfully different thread interleaving. Where the bytecode explorer
samples randomly, DPOR guarantees completeness: every distinct interleaving
is tried exactly once and redundant orderings are never re-run.

Like the bytecode explorer, DPOR instruments at the opcode level and needs no
manual markers. It automatically detects attribute reads/writes, subscript
accesses, and lock operations. The exploration engine is written in Rust for
performance.

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

When a race is found, ``result.explanation`` contains an interleaved source-line
trace showing which threads accessed which shared state and the conflict pattern.
DPOR's traces are more interpretable than the bytecode explorer's because DPOR
*knows why* the interleaving matters --- it detected specific conflicting accesses
to the same object and chose to explore the alternative ordering.

**Scope:** DPOR explores alternative schedules where it detects a
conflict --- either two threads accessing the same Python object (with at
least one write) or two threads performing I/O on the same network
endpoint or file path. When run under the ``frontrun`` CLI, a native
``LD_PRELOAD`` library intercepts C-level I/O operations (``send``,
``recv``, ``read``, ``write``, etc.) and feeds them into the DPOR
engine, so even opaque database drivers (libpq, mysqlclient) and Redis
clients (hiredis) are covered --- see :ref:`c-level-io-interception`
below.

DPOR *does* detect many C-level operations: ``list.append``,
``dict.__setitem__``, and similar mutating methods are seen via
``sys.setprofile``; builtins like ``sorted()``, ``sum()``, ``min()``,
``max()``, and ``str.join()`` are registered as passthrough reads on
their arguments; and container constructors (``list()``, ``dict()``,
``enumerate()``, ``zip()``, etc.) are tracked as reads on their inputs.

The gap is **C-level iteration interleaving**.  DPOR treats each C call
as a single atomic operation, but under PEP 703 (free-threaded Python),
C functions that iterate via ``PyIter_Next`` — such as
``list(od.keys())`` — acquire and release the per-object lock on each
element, allowing another thread to mutate the collection between
iterations.  When *both* sides of a race are single C opcodes (e.g.
``list(od.keys())`` vs ``od.move_to_end("a")``), no bytecode-level tool
can expose the interleaving.  See ``PEP-703-REPORT.md`` for details.

DPOR also cannot see shared state managed entirely inside a C extension
without any I/O or Python-visible operations --- for example, in-process
mutations of NumPy arrays or C-level caches with no Python API calls.

For a practical guide see :doc:`dpor_guide`. For the algorithm details and
theory see :doc:`dpor`.


Bytecode Instrumentation
-------------------------

Bytecode instrumentation automatically instruments functions at the opcode
level --- no markers needed.

**How It Works:**

Each thread is run with a ``sys.settrace`` callback that sets
``f_trace_opcodes = True`` on every frame, so the callback fires at every
*bytecode instruction* rather than every source line. At each opcode the thread
calls ``scheduler.wait_for_turn()``, which blocks until the schedule says it's
that thread's turn. Only user code is traced --- stdlib and threading internals
are skipped.

Because the scheduler controls which thread runs each opcode, any blocking call
that happens in C code (like ``threading.Lock.acquire()``) would deadlock --- the
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

   def test_counter_is_atomic():
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

       assert result.property_holds, result.explanation

``explore_interleavings()`` often finds races very quickly --- sometimes on the
first attempt --- because even a single random schedule has a reasonable chance
of interleaving the critical section. It can also catch races that are invisible
to DPOR: if a C extension mutates shared state *without any I/O* (e.g.
in-process C-level mutations), bytecode exploration may stumble into the
bad interleaving through random scheduling even though neither tool can
see the C-level conflict directly.

The trade-off is interpretability. When a race is found, ``result.explanation``
contains an interleaved source-line trace and a best-effort conflict
classification, but the bytecode explorer doesn't *know why* the interleaving
matters the way DPOR does. The ``reproduce_on_failure`` parameter (default 10)
controls how many times the counterexample schedule is replayed to measure
reproducibility.

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
   than opcodes, so its schedules are stable --- see that module for details.


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
without changing any executable code --- markers are just comments.

A marker **gates** the code that follows it. Name markers after the operation
they gate (e.g. ``read_value``, ``write_balance``) rather than with temporal
prefixes like ``before_`` or ``after_``.

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
           return self.value

       async def set_count(self, value):
           self.value = value

       async def increment(self):
           # frontrun: read_counter
           current = await self.get_count()
           # frontrun: write_counter
           await self.set_count(current + 1)


Automatic I/O Detection
-------------------------

Both the bytecode explorer and DPOR automatically detect socket and file
I/O operations. This is enabled by default (``detect_io=True``) and works
by monkey-patching ``socket.socket`` methods and ``builtins.open`` to
report resource accesses to the scheduler.

**Python-level detection (monkey-patching):**

- **Sockets:** ``connect``, ``send``, ``sendall``, ``sendto``, ``recv``,
  ``recv_into``, ``recvfrom``
- **Files:** ``open()`` (read vs write determined by mode)

Resource identity is derived from the socket's peer address
(``host:port``) or the file's resolved path. Two threads accessing the
same endpoint or file are treated as conflicting; different endpoints are
independent.

.. _c-level-io-interception:

C-Level I/O Interception
~~~~~~~~~~~~~~~~~~~~~~~~~~

When run under the ``frontrun`` CLI, a native ``LD_PRELOAD`` library
(``libfrontrun_io.so``) intercepts libc I/O functions directly. This
covers opaque C extensions --- database drivers (libpq, mysqlclient),
Redis clients, HTTP libraries, and anything else that calls libc's
``send()``, ``recv()``, ``read()``, ``write()``, etc.

**Intercepted functions:** ``connect``, ``send``, ``sendto``, ``sendmsg``,
``write``, ``writev``, ``recv``, ``recvfrom``, ``recvmsg``, ``read``,
``readv``, ``close``

The library maintains a process-global file-descriptor → resource map:

.. code-block:: text

   connect(fd, sockaddr{127.0.0.1:5432}, ...)  →  fd=7 → "socket:127.0.0.1:5432"
   send(fd=7, ...)                              →  report write to "socket:127.0.0.1:5432"
   recv(fd=7, ...)                              →  report read from "socket:127.0.0.1:5432"
   close(fd=7)                                  →  remove fd=7 from map

Events are communicated to the Python side via a pipe
(``FRONTRUN_IO_FD``).  An ``IOEventDispatcher`` reads the pipe on a
background thread and delivers events to registered listeners.  When
DPOR is active, a ``_PreloadBridge`` listener routes events to the DPOR
engine for conflict analysis.

**Building:**

.. code-block:: bash

   make build-io    # builds and copies libfrontrun_io.so into the frontrun package

**Usage:**

.. code-block:: bash

   frontrun pytest -vv tests/           # I/O interception + monkey-patching
   frontrun python examples/orm_race.py  # same, for scripts
