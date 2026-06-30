How It Works Under the Hood
===========================

This page walks through the mechanisms Frontrun uses to control thread
interleaving, from Python bytecode up through C-level I/O interception.
The goal is to make the magic legible --- each layer is doing something
specific and the trade-offs follow from how that layer works.


Python bytecode and the interleaving problem
----------------------------------------------

CPython compiles Python source to bytecode instructions (opcodes).  A
line like ``self.value = temp + 1`` compiles to several opcodes:

.. code-block:: text

   # Python 3.10
   LOAD_FAST       1 (temp)
   LOAD_CONST      1 (1)
   BINARY_ADD
   LOAD_FAST       0 (self)
   STORE_ATTR      0 (value)

The GIL (Global Interpreter Lock) ensures that only one thread executes
Python bytecode at a time, but it does *not* guarantee that a thread
runs an entire source line atomically.  CPython can release the GIL
between any two opcodes (and periodically does, to give other threads a
chance to run).  So ``self.value = temp + 1`` is not atomic --- another
thread can execute between ``LOAD_FAST`` and ``STORE_ATTR``.

The free-threaded build (Python 3.13t+) removes the GIL entirely,
allowing true parallel execution.  But even with the GIL, the
interleaving window between opcodes is enough to produce race conditions
in pure Python code.  The classic lost-update bug:

.. code-block:: python

   class Counter:
       def __init__(self):
           self.value = 0

       def increment(self):
           temp = self.value      # LOAD_ATTR → local
           self.value = temp + 1  # BINARY_ADD, STORE_ATTR

Two threads calling ``increment()`` can both execute ``LOAD_ATTR``
(reading the same value) before either executes ``STORE_ATTR``.  One
increment is lost.  This is the kind of bug Frontrun exists to find.

.. note::

   Bytecode is not stable across Python versions.  Python 3.10 uses
   ``BINARY_ADD``; 3.11+ uses ``BINARY_OP``; 3.14 uses
   ``LOAD_FAST_BORROW``.  This is why opcode-level schedules from the
   bytecode explorer are ephemeral debugging artifacts, not long-lived
   test fixtures.


``sys.settrace``: line-level and opcode-level tracing
------------------------------------------------------

Python's ``sys.settrace`` installs a callback that fires on various
execution events.  Frontrun uses two modes:

**Line-level tracing** (used by trace markers):

.. code-block:: python

   def trace_callback(frame, event, arg):
       # event is one of: 'call', 'line', 'return', 'exception'
       if event == 'line':
           # frame.f_lineno tells us which source line is about to execute
           ...
       return trace_callback

   sys.settrace(trace_callback)

The ``'line'`` event fires before each source line executes.  Frontrun's
``MarkerRegistry`` scans the source file for ``# frontrun: <name>``
comments and builds a mapping from line numbers to marker names.  When
the trace callback sees a line event at a marker, it calls
``ThreadCoordinator.wait_for_turn()`` which blocks (via a
``threading.Condition``) until the schedule says it's this thread's turn.

This is lightweight --- the callback fires once per source line, and
the marker-location cache means the per-event overhead is a dict lookup.
The cost is that you need to manually annotate the synchronization points.

**Opcode-level tracing** (used by bytecode exploration and DPOR):

.. code-block:: python

   def trace_callback(frame, event, arg):
       if event == 'call':
           frame.f_trace_opcodes = True   # enable opcode events for this frame
       if event == 'opcode':
           # fires before EACH bytecode instruction
           scheduler.wait_for_turn(thread_id)
       return trace_callback

Setting ``f_trace_opcodes = True`` on a frame causes the trace callback
to fire with ``event='opcode'`` before every bytecode instruction in
that frame.  This gives the scheduler complete control over which thread
runs each instruction --- the fundamental mechanism behind both bytecode
exploration and DPOR.

The per-opcode overhead is substantial (a Python function call for every
single bytecode instruction), which is why both bytecode exploration and
DPOR filter out stdlib and threading internals via
``should_trace_file()`` in ``_tracing.py``.  By default, all code in
``site-packages`` is skipped, but the ``trace_packages`` parameter on
exploration entry points allows widening the filter to include specific
installed packages.  See :doc:`trace_filtering` for details.

On Python 3.12+, ``sys.monitoring`` provides a lower-overhead
alternative to ``sys.settrace`` for opcode-level events.  Frontrun
uses it where available, falling back to ``sys.settrace`` on 3.10--3.11.


``sys.setprofile``: detecting C-level calls
---------------------------------------------

``sys.settrace`` only fires for Python bytecode.  When a C extension
function is *called* from Python, ``sys.settrace`` sees the ``'call'``
event for the Python caller, but the C function itself executes without
any trace events --- it's opaque.

``sys.setprofile`` fills this gap.  It fires ``'c_call'``, ``'c_return'``,
and ``'c_exception'`` events for calls into C code:

.. code-block:: python

   def profile_func(frame, event, arg):
       if event == 'c_call':
           # arg is the C function object (e.g. socket.socket.send)
           qualname = getattr(arg, '__qualname__', '')
           if qualname == 'socket.send':
               # This thread is about to call socket.send() in C
               ...

   sys.setprofile(profile_func)

Frontrun uses this as "Layer 1.5" of I/O detection: it installs a
per-thread profile function that watches for C-level socket calls
(``send``, ``recv``, ``connect``, etc.) and reports them to the
scheduler.  This coexists with ``sys.settrace`` without interference ---
the two mechanisms are independent and both can be active simultaneously.

The limitation is that ``sys.setprofile`` only sees calls *from Python
to C*.  If a C extension calls another C function internally (e.g.
libpq calling libc's ``send()``), the profile callback never fires.
That's where ``LD_PRELOAD`` comes in.


Monkey-patching: cooperative primitives and I/O detection
----------------------------------------------------------

**Threading primitives:**

When the bytecode explorer or DPOR controls thread scheduling at the
opcode level, standard threading primitives become a problem.  If
thread A holds the scheduler's turn and calls ``Lock.acquire()`` on a
lock held by thread B, thread A blocks in C code waiting for the lock.
But thread B can't release the lock because the scheduler hasn't given
it a turn.  Deadlock.

Frontrun solves this by monkey-patching ``threading.Lock``,
``threading.RLock``, ``threading.Semaphore``, ``threading.Event``,
``threading.Condition``, ``queue.Queue``, and related primitives with
*cooperative* versions.  A cooperative lock's ``acquire()`` doesn't
block in C --- it does non-blocking attempts in a loop, yielding its
scheduler turn between each attempt:

.. code-block:: python

   class CooperativeLock:
       def acquire(self, blocking=True, timeout=-1):
           if self._real_lock.acquire(blocking=False):
               return True  # got it immediately
           if not blocking:
               return False
           # Spin-yield: give other threads a chance to run
           while True:
               scheduler.wait_for_turn(thread_id)  # yield to scheduler
               if self._real_lock.acquire(blocking=False):
                   return True

The patching is scoped to each test run: ``patch_locks()`` replaces the
threading module's classes before ``setup()`` runs, and
``unpatch_locks()`` restores them afterward.  Originals are saved at
import time in ``_real_threading.py`` to avoid circular imports.

**I/O detection (Layer 1):**

Socket and file I/O operations are monkey-patched to report resource
accesses to the scheduler:

.. code-block:: python

   # Save the real method
   _real_socket_send = socket.socket.send

   def _traced_send(self, *args, **kwargs):
       # Report the I/O event to the scheduler
       reporter = get_io_reporter()  # per-thread callback from TLS
       if reporter is not None:
           resource_id = f"socket:{self.getpeername()[0]}:{self.getpeername()[1]}"
           reporter(resource_id, "write")
       return _real_socket_send(self, *args, **kwargs)

   # Replace the method on the class
   socket.socket.send = _traced_send

Resource identity is derived from the socket's peer address or the
file's resolved path.  Two threads accessing ``socket:127.0.0.1:5432``
are reported as conflicting; different endpoints are independent.

This works for pure-Python socket usage (e.g. ``httpx``,
``urllib3`` in pure mode).  It does *not* work for C extensions that
manage sockets internally (e.g. ``psycopg2`` calling libpq, which calls
libc ``send()`` directly).


``LD_PRELOAD``: C-level I/O interception
------------------------------------------

The deepest layer.  When Python code calls a C extension, and that C
extension calls libc functions like ``send()`` or ``recv()``, neither
``sys.settrace`` nor ``sys.setprofile`` nor monkey-patching can see it.
The call goes from the C extension directly to libc, bypassing Python
entirely.

``LD_PRELOAD`` (Linux) and ``DYLD_INSERT_LIBRARIES`` (macOS) solve this
by *interposing* a shared library before libc in the dynamic linker's
symbol resolution order.  When any code in the process calls ``send()``,
the dynamic linker finds Frontrun's ``send()`` first:

.. code-block:: c

   // crates/io/src/lib.rs (simplified, shown as C for clarity)

   // Look up the real libc send() once
   static real_send_t real_send = NULL;

   ssize_t send(int fd, const void *buf, size_t len, int flags) {
       if (!real_send) {
           real_send = dlsym(RTLD_NEXT, "send");  // find the NEXT "send" symbol
       }

       // Report the event: "write to socket on fd"
       report_io_event(fd, "write");

       // Call the real libc send()
       return real_send(fd, buf, len, flags);
   }

The actual implementation is in Rust (``crates/io/src/lib.rs``) and
uses ``#[no_mangle]`` with ``extern "C"`` to produce C-compatible
symbol names.  The library maintains a process-global map from file
descriptors to resource IDs:

.. code-block:: text

   connect(fd=7, {127.0.0.1:5432})  →  register fd 7 as "socket:127.0.0.1:5432"
   send(fd=7, ...)                   →  report write to "socket:127.0.0.1:5432"
   recv(fd=7, ...)                   →  report read from "socket:127.0.0.1:5432"
   close(fd=7)                       →  unregister fd 7

Events are communicated to the Python side via one of two channels:

**Pipe transport (preferred):** ``IOEventDispatcher`` in Python creates
an ``os.pipe()`` and passes the write-end file descriptor to the Rust
library via the ``FRONTRUN_IO_FD`` environment variable.  The Rust
library writes event records directly to the pipe.  A Python reader
thread dispatches events to registered callbacks in arrival order.  The
pipe's FIFO semantics provide a natural total order without timestamps.

**Log file transport (debugging only):** ``FRONTRUN_IO_LOG`` points to a
temporary file.  Events are appended per-call (open + write + close
each time) and read back in batch after execution.  This approach is
intended for testing and debugging the frontrun framework itself.  It has
higher overhead than the pipe transport.

The ``frontrun`` CLI sets up the ``LD_PRELOAD`` environment automatically:

.. code-block:: bash

   $ frontrun pytest -v tests/
   frontrun: using preload library /path/to/frontrun/libfrontrun_io.so

This covers opaque C extensions --- database drivers (libpq for
PostgreSQL, mysqlclient, Oracle's thick driver), Redis clients (hiredis),
HTTP libraries, and anything else that calls libc I/O functions.

**Intercepted libc functions:** ``connect``, ``send``, ``sendto``,
``sendmsg``, ``write``, ``writev``, ``recv``, ``recvfrom``, ``recvmsg``,
``read``, ``readv``, ``close``.

**Platform notes:**

- Linux: ``LD_PRELOAD=/path/to/libfrontrun_io.so``
- macOS: ``DYLD_INSERT_LIBRARIES=/path/to/libfrontrun_io.dylib``.
  System Integrity Protection (SIP) strips this variable from
  Apple-signed binaries (``/usr/bin/python3``), so use a Homebrew, pyenv,
  or venv Python.
- Windows: no equivalent mechanism exists.  The ``LD_PRELOAD`` approach
  depends on the Unix dynamic linker's symbol interposition, which has
  no direct Windows analog.


Putting the layers together
----------------------------

Each approach uses a different combination of these mechanisms:

**Trace markers** use ``sys.settrace`` in line-level mode only.  No
monkey-patching, no ``LD_PRELOAD``.  The scheduler controls which thread
proceeds past each marker; between markers, threads run freely.  This is
the lightest-weight approach --- the overhead is one dict lookup per
source line.

**Bytecode exploration** uses ``sys.settrace`` in opcode-level mode,
plus monkey-patched cooperative threading primitives (to prevent
deadlocks) and optionally monkey-patched I/O (to detect socket/file
conflicts).  The scheduler controls every single bytecode instruction.
High overhead, but complete control over interleaving.

**DPOR** uses the same opcode-level ``sys.settrace`` and cooperative
primitives as bytecode exploration.  The difference is the scheduling
policy: DPOR uses a Rust engine (``crates/dpor/``) that tracks
shared-memory accesses via dual vector clocks (``dpor_vv`` for
lock-aware happens-before, ``io_vv`` for lock-oblivious I/O tracking)
and only explores alternative orderings at conflict points.  The engine
implements a hybrid of classic and Optimal DPOR (Abdulla et al., JACM
2017): wakeup trees guide exploration order, sleep set propagation with
step-count-indexed trace caching eliminates equivalent interleavings of
independent operations, and deferred race detection computes ``notdep``
wakeup sequences for better coverage.  Optionally adds
``sys.setprofile`` for C-call detection and monkey-patched I/O.  See
`The DPOR algorithm in detail`_ below for the full description.

**Async shuffler** does *not* use opcode tracing.  It stays entirely in a
single asyncio event loop and uses ``InterleavedLoop`` plus
``contextvars`` to control tasks at natural coroutine suspension points.
``frontrun.explore_async_random()`` generates random schedules over those await
boundaries and checks the invariant afterward.  It does not try to
understand which schedules are equivalent; it simply samples them.

**Async DPOR** also stays in a single asyncio event loop, but the
scheduler wraps each task coroutine so that every *natural* ``await``
becomes a scheduling boundary.  Within each await-delimited block it
installs ``sys.monitoring`` (3.12+) or ``sys.settrace`` opcode events on
user-code frames and reuses the same shadow-stack/opcode processor as
sync DPOR.  The important distinction is that async DPOR does **not**
preempt in the middle of a block; the tracing is used only to tell the
Rust DPOR engine which await-boundary reorderings actually conflict and
which are independent.

**``LD_PRELOAD`` interception** is orthogonal to the Python-level
mechanisms.  It runs whenever the ``frontrun`` CLI is used, regardless
of which approach (or no approach) is active on the Python side.  Events
from the Rust interception library are available via
``IOEventDispatcher`` for any consumer.

.. note::

   DPOR consumes ``LD_PRELOAD`` events when ``detect_io=True`` (the
   default).  ``frontrun.explore()`` starts an ``IOEventDispatcher`` that
   reads the pipe, and a ``_PreloadBridge`` maps OS thread IDs to DPOR
   logical thread IDs and buffers events for draining at each scheduling
   point.  This means C extensions that call libc ``send()``/``recv()``
   directly (e.g. psycopg2 via libpq) are covered --- DPOR treats the
   shared socket endpoint as a conflict and explores alternative
   orderings around the I/O.

   The random bytecode explorer does *not* consume ``LD_PRELOAD`` events.
   It relies on Python-level monkey-patching (and random scheduling) to
   find races involving C-level I/O.

.. list-table:: Mechanism usage by approach
   :header-rows: 1
   :widths: 30 14 14 14 14 14

   * - Mechanism
     - Trace markers
     - Bytecode
     - DPOR
     - LD_PRELOAD
     - sys.setprofile
   * - sys.settrace (line)
     - Yes
     -
     -
     -
     -
   * - sys.settrace (opcode)
     -
     - Yes
     - Yes
     -
     -
   * - Cooperative locks
     -
     - Yes
     - Yes
     -
     -
   * - I/O monkey-patching
     -
     - Optional
     - Optional
     -
     -
   * - C-call profiling
     -
     -
     - Optional
     -
     - Yes
   * - LD_PRELOAD / DYLD
     -
     -
     -
     - Yes
     -

.. list-table:: Mechanism usage by async approach
   :header-rows: 1
   :widths: 34 18 18

   * - Mechanism
     - Async shuffler
     - Async DPOR
   * - ``InterleavedLoop`` / asyncio.Condition gating
     - Yes
     - Yes
   * - Explicit ``await_point()`` markers
     - Yes
     - Optional compatibility only
   * - Automatic scheduling at natural ``await``
     -
     - Yes
   * - Opcode/instruction tracing of Python accesses
     -
     - Yes
   * - DPOR conflict reduction / vector clocks
     -
     - Yes
   * - Async lock monkey-patching / wait-for graph
     -
     - Yes
   * - Async SQL interception
     - Optional
     - Optional


What each layer can and cannot see
------------------------------------

Async approaches: shuffler vs async DPOR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two async implementations share a scheduler foundation but differ in
how much they observe and how they choose schedules.

**Async shuffler** (``frontrun.async_shuffler``):

- User tasks are wrapped so every natural ``await`` becomes a scheduling
  boundary.
- ``await_point()`` is optional; it just adds an extra explicit yield.
- ``AwaitScheduler`` follows a concrete list of task indices, or
  ``frontrun.explore_async_random()`` samples random such lists.
- No Python opcode tracing runs inside the block between two
  await boundaries.
- This makes the implementation simple and the schedules stable across
  Python versions, but it also means the tool does not reduce equivalent
  schedules based on actual conflicts.

**Async DPOR** (``frontrun.async_dpor``):

- User tasks are wrapped in ``_AutoPauseCoroutine``.  Before each step of
  the inner coroutine, the wrapper drives ``scheduler.pause(task_id)``,
  so every natural coroutine suspension becomes a scheduling boundary.
- While a task is running between two such boundaries, opcode-level
  tracing is enabled for user-code frames only.  ``_process_opcode()``
  maintains shadow stacks and reports reads/writes (attributes,
  subscripts, globals, closure cells, selected C-method surrogates, and
  so on) to the Rust DPOR engine.
- When the task reaches the next ``await``, the scheduler asks the DPOR
  engine which task should run next.  The engine has already seen the
  accesses performed in the completed block, so it can backtrack only on
  await-boundary reorderings whose blocks actually conflict.
- Tasks are still atomic *between* awaits from the scheduler's point of
  view.  Async DPOR does not create new mid-block preemption points; the
  tracing only informs reduction.
- ``asyncio.Lock`` is monkey-patched to a cooperative version that adds
  explicit acquisition scheduling points, reports lock acquire/release
  synchronization to the engine, and uses a wait-for graph for deadlock
  detection.
- Async SQL interception buffers I/O/resource accesses and flushes them
  at the next scheduling boundary, so SQL conflicts participate in the
  same DPOR search.

This difference explains the naming split:

- async shuffler: await-point schedule generation/sampling
- async DPOR: await-boundary scheduling plus traced conflict reduction

Understanding these boundaries explains why DPOR misses database-level
races and why bytecode exploration sometimes finds bugs DPOR can't.

**Python attribute access** (e.g. ``self.value``):  Visible to
``sys.settrace`` (opcode events ``LOAD_ATTR``, ``STORE_ATTR``).  DPOR
and bytecode exploration both see these.

**Python-level socket calls** (e.g. ``sock.send(data)``):  Visible to
``sys.settrace`` (the Python call) and to monkey-patched wrappers.
Both DPOR and bytecode exploration can detect these.

**C-extension socket calls from Python** (e.g.
``socket.socket.send(data)``):  Invisible to ``sys.settrace`` (the C
function runs atomically from Python's perspective).  Visible to
``sys.setprofile`` (fires ``'c_call'`` before the C function runs) and
to ``LD_PRELOAD`` (intercepts the underlying libc call).

**C-extension internal calls** (e.g. libpq calling libc ``send()``
inside ``PQexec()``):  Invisible to ``sys.settrace``, ``sys.setprofile``,
and monkey-patching.  Visible *only* to ``LD_PRELOAD``, which intercepts
at the libc level regardless of who called it.  DPOR consumes these
events via ``IOEventDispatcher`` → ``_PreloadBridge`` (see note above).
The bytecode explorer does not consume them but may still find the race
through random scheduling.

**C-level iteration** (e.g. ``list(od.keys())`` while another thread
calls ``od.move_to_end()``):  DPOR treats each C call as a single
atomic operation.  Under PEP 703 (free-threaded Python), C functions
that iterate via ``PyIter_Next`` acquire and release the per-object
lock on each element, so another thread can mutate the collection
between iterations.  When *both* sides of a race are single C opcodes,
no bytecode-level tool can expose the interleaving.  This affects
``itertools`` combinators, ``list()``/``tuple()`` on dict views,
``OrderedDict.move_to_end()`` during iteration, and similar patterns.
See ``PEP-703-REPORT.md`` for worked examples.

**External server state** (e.g. a row in PostgreSQL):  The socket-level
conflict (two threads talking to ``127.0.0.1:5432``) *is* visible to
``LD_PRELOAD``, but DPOR also has a higher-precision layer: the SQL
cursor interception (``_sql_cursor.py``) parses each SQL statement,
extracts table names, row-level predicates, and access kinds, and
reports them as fine-grained resource IDs (e.g. ``sql:users``,
``sql:users:([('id', '1')],)``).  These table-level and row-level
resources participate in DPOR's conflict detection directly, so DPOR
*can* distinguish a ``SELECT`` on table A from an ``UPDATE`` on table B.

When multiple tables are involved (e.g. an ORM pattern that touches
``articles`` then ``versions``), **resource group coalescing** prevents
backtrack explosion: each SQL resource is registered with a table-level
resource group, and the engine skips inline wakeup insertion when
intermediate operations on *unrelated* tables separate the racing
accesses.  Deferred ``notdep`` processing then finds the critical
interleaving efficiently.  See `Resource groups and operation
coalescing`_ below.

**Row-lock deadlocks** are a special case that *is* handled precisely at
the client side, via the row-lock registry described in the next section.


Deadlock detection
--------------------

A deadlock means no thread can make progress — the program is permanently
stuck.  Since the invariant can never be evaluated on a stuck program,
a deadlock is always reported as ``property_holds = False``.

Frontrun detects two kinds of deadlock, both using a directed
**wait-for graph**:

**Cooperative-lock deadlocks (Python-level):**

When ``CooperativeLock`` or ``CooperativeRLock`` would block (the lock is
held by another thread), the slow path registers a waiting edge in the
global ``WaitForGraph`` before entering the spin loop:

.. code-block:: text

    add_waiting(thread_id, lock_id)  →  DFS from ("thread", thread_id)

If the DFS finds a path back to the starting node, a cycle exists.
The waiting edge is immediately removed, ``DeadlockError`` is stored on
``scheduler._error``, and ``SchedulerAbort`` is raised to exit all
managed threads cleanly.  No actual waiting occurs.

**Row-lock deadlocks (``SELECT FOR UPDATE``):**

``SELECT FOR UPDATE`` acquires a row-level exclusive lock inside Postgres.
If thread A holds row lock X and waits for row lock Y while thread B
holds row lock Y and waits for row lock X, both threads will block
indefinitely inside libpq's C code --- invisible to Python tracing.

Frontrun prevents this by maintaining a client-side row-lock registry
(``DporScheduler._active_row_locks``) and intercepting the lock
acquisition *before* the C call:

.. code-block:: text

    SQL cursor detects "SELECT ... FOR UPDATE"
        → identifies row-level resource ID  (e.g. "sql:users:(('id',1))")
        → calls acquire_row_locks(thread_id, [resource_id])

    acquire_row_locks:
        if another thread holds the lock:
            add_waiting(thread_id, row_lock_id, kind="row_lock")
            if WaitForGraph detects a cycle:
                remove_waiting(...)
                scheduler._error = DeadlockError(cycle_description)
                notify_all()
                raise SchedulerAbort
            else:
                condition.wait()   ← Python-level, not C-level

Row-lock nodes use a separate namespace in the wait-for graph
(``("row_lock", counter)`` vs ``("lock", id(obj))`` for cooperative
locks), so their integer IDs cannot collide regardless of pointer values.

**Cross-domain cycles** (a cooperative Python lock and a row lock in a
cycle together) are also detected, because both node types coexist in the
same ``WaitForGraph`` and the DFS traverses edges regardless of kind.

**Error propagation:**

.. code-block:: text

    DeadlockError stored on scheduler._error
    SchedulerAbort raised in detecting thread
    Other threads see scheduler._error and exit their spin loops
    runner.run() returns normally (no TimeoutError)
    explore checks isinstance(scheduler._error, DeadlockError)
        → result.property_holds = False
        → result.explanation = "Deadlock detected ... <cycle>"
        → result.counterexample = schedule trace

Because ``DeadlockError`` is not a subclass of ``TimeoutError``, it is
not swallowed by the ``except TimeoutError: pass`` guard in the main
exploration loop.

**What is NOT covered:**

- Deadlocks that occur entirely inside a C extension without going through
  the row-lock registry (e.g. two threads sharing a single psycopg2
  connection object, which is unsupported anyway).
- ``LOCK TABLE`` or advisory locks --- only ``SELECT FOR UPDATE`` /
  ``SELECT FOR SHARE`` row locks are intercepted by the SQL cursor layer.


The DPOR algorithm in detail
-----------------------------

This section describes the Rust DPOR engine (``crates/dpor/``) and how it
interacts with the Python scheduler to systematically explore thread
interleavings.  The implementation is a hybrid of Algorithms 1 and 2 from
Abdulla et al., "Source Sets: A Foundation for Optimal Dynamic Partial
Order Reduction" (JACM 2017), augmented with a dual vector clock scheme
for I/O conflict detection.


Rust engine / Python scheduler interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rust extension exposes two PyO3 classes: ``DporEngine`` (persistent
across executions) and ``Execution`` (per-execution state).  The Python
scheduler drives exploration via a loop:

.. code-block:: text

   engine = DporEngine(num_threads, preemption_bound, max_branches)
   while True:
       execution = engine.begin_execution()
       # ... start threads ...
       while True:
           thread_id = engine.schedule(execution)      # pick next thread
           if thread_id is None:
               break                                   # deadlock or all done
           # ... run thread until next scheduling point ...
           # opcode tracing reports accesses:
           engine.report_access(execution, tid, obj_id, "read"|"write"|...)
           engine.report_io_access(execution, tid, obj_id, "write")
           engine.report_sync(execution, tid, "lock_acquire", lock_id)
           # ... etc. ...
       # check invariant, record result
       if not engine.next_execution():
           break                                       # exploration complete

Each ``schedule()`` call advances the path position, increments both
vector clocks for the chosen thread, records the scheduling trace, and
returns the thread to run.  During replay (re-executing a prefix), the
engine follows the recorded path; at new branches, it consults the wakeup
tree or defaults to the current thread to minimize preemptions.

``next_execution()`` processes deferred races (computing ``notdep``
sequences), handles deferred lock-release backtracking, then calls
``step()`` to find the next unexplored path in the exploration tree.
``step()`` walks backward through the branch stack, marks explored threads
as ``Visited``, adds them to sleep sets, and picks the next thread from
the wakeup tree (minimum thread ID as a deterministic proxy for the
paper's ordering).


Three vector clocks: ``dpor_vv``, ``io_vv``, ``causality``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. seealso::

   :doc:`vector-clocks` for the full explanation of the ``VersionVec`` data
   structure, all three clocks, synchronization event handling, and worked
   examples showing exactly how races are detected or suppressed.

Each thread maintains three vector clocks (``VersionVec``, indexed by
thread ID):

**``dpor_vv``** --- the primary happens-before clock.  Incremented on
each scheduling step and joined on synchronization events (lock
acquire, lock release, thread spawn, thread join).  Used by
``process_access()`` and ``process_synced_io_access()`` to detect races
between shared-memory accesses.  Two accesses race when their
``dpor_vv`` values are *incomparable* (neither is component-wise less
than or equal to the other).  Because ``dpor_vv`` includes lock edges,
two accesses protected by the same lock will appear ordered and will
*not* be reported as a race.

**``io_vv``** --- the I/O-specific clock.  Incremented on each scheduling
step (same as ``dpor_vv``) and joined on thread spawn/join, but
**not** joined on lock acquire/release.  Used by ``process_io_access()``
for file and socket operations (``LD_PRELOAD`` events, C-call profiling).
Because ``io_vv`` omits lock edges, I/O accesses from different threads
*always* appear potentially concurrent --- even when they occur inside
separate critical sections of the same lock.  This conservative treatment
ensures DPOR explores interleavings around I/O operations and catches
TOCTOU races that lock-aware tracking would suppress.

**``causality``** --- a general causal ordering clock.  Joined on lock
acquire, thread spawn, and thread join (same events as ``dpor_vv``).
It is not incremented on scheduling steps.  It serves as a shared
clock for propagating causal information across synchronization points.

The dual-clock design is a pragmatic extension not in the original paper.
The paper's Algorithms 3--4 (JACM'17 p.27--28) handle locks by tracking
enabled/disabled threads.  Frontrun instead uses the separate ``io_vv``
without lock edges to force exploration at lock boundaries --- conservative
(may over-explore) but catches multi-lock races that pure happens-before
tracking would miss.

**How each access API uses the clocks:**

.. list-table::
   :header-rows: 1
   :widths: 28 16 18 18 20

   * - API
     - Clock used
     - Recording mode
     - Origin tag
     - Typical use
   * - ``report_access``
     - ``dpor_vv``
     - last-write-wins
     - ``PythonMemory``
     - Python attribute/subscript accesses
   * - ``report_first_access``
     - ``dpor_vv``
     - first-write-wins
     - ``PythonMemory``
     - container-level keys (earliest position for wakeup insertion)
   * - ``report_synced_io_access``
     - ``dpor_vv``
     - first-write-wins
     - ``IoDirect``
     - Python-level I/O (Redis, SQL) ordered by Python locks
   * - ``report_io_access``
     - ``io_vv``
     - first-write-wins
     - ``IoDirect``
     - file/socket I/O (``LD_PRELOAD``, C-call detection)

The "last-write-wins" recording mode (``record_access``) overwrites the
per-thread access entry with the latest access, so the race check always
considers the most recent access to each object.  The "first-write-wins"
mode (``record_io_access``) keeps the earliest access per thread, enabling
wakeup tree insertion at the earliest point --- for example, between a
``SELECT`` and ``UPDATE`` in a database transaction.


Conflict detection and race checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two accesses to the same shared object form a **race** (JACM'17 Def 3.3)
when:

1. They are from different threads.
2. At least one is a write (or a weak-write conflicting with a
   non-weak access --- see below).
3. Their vector clocks are incomparable (neither happens-before the other).

The engine tracks per-object, per-thread access history in
``ObjectState``.  Each object maintains four maps (per-thread-read,
per-thread-write, per-thread-weak-write, per-thread-weak-read).  When a
new access arrives, ``dependent_accesses()`` returns all prior accesses
from other threads that conflict with the new access's kind.

Each access also carries an ``AccessOrigin`` tag indicating how it was
generated:

- ``PythonMemory`` --- observed via Python shared-memory tracing
  (``sys.settrace`` opcode events).  Used by ``report_access`` and
  ``report_first_access``.
- ``LockSynthetic`` --- synthetic access for lock acquire/release conflict
  tracking.  Generated in ``process_sync()`` for lock events.
- ``IoDirect`` --- I/O access intercepted from database, Redis, file, or
  socket operations.  Used by ``report_synced_io_access`` and
  ``report_io_access``.

Provenance tags are threaded through the sleep-set propagation layer
(``active_accesses``, ``explored_accesses``, ``propagated_sleep_accesses``,
and the step-count-indexed future access cache all carry
``(AccessKind, AccessOrigin)`` tuples).  When merging accesses with
different origins, the "stronger" origin wins
(``IoDirect`` > ``LockSynthetic`` > ``PythonMemory``).  Currently
the independence check only examines ``AccessKind`` (provenance does not
affect conflict semantics), but the tags enable future per-origin merge
strategies --- for example, relaxing Python-memory merges while keeping
I/O origins conservative.

**Access kinds and conflict rules:**

- ``Write`` conflicts with everything (Read, Write, WeakWrite, WeakRead).
- ``Read`` conflicts with Write and WeakWrite.
- ``WeakWrite`` conflicts with Read and Write, but *not* with other
  WeakWrites or WeakReads.  Used for container subscript writes
  (``STORE_SUBSCR``): two writes to different keys of the same dict should
  not conflict.
- ``WeakRead`` conflicts only with Write.  Used for ``LOAD_ATTR`` on
  mutable containers: loading a container to subscript it should not
  conflict with subscript writes on disjoint keys.

For each conflicting prior access, the engine checks
``prev_access.dpor_vv <= current_dpor_vv`` (or ``io_vv`` for I/O
accesses).  If the check fails, the two accesses are concurrent and a
race is detected.


Backtracking: wakeup trees and deferred races
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a race is detected, the engine responds in two ways (hybrid
approach):

**Inline insertion (Algorithm 1 style, JACM'17 p.16 lines 5--9):**
Immediately inserts the racing thread as a single-element sequence
``[thread_id]`` into the wakeup tree at the position of the first racing
access (``prev_path_id``).  This guarantees that no race is dropped ---
the racing thread is always available for exploration at the conflict
point.  (Exception: for ``process_synced_io_access`` races with
cross-group intermediates, inline insertion is skipped in favor of
deferred processing --- see `Resource groups and operation coalescing`_
below.)

**Deferred ``notdep`` processing (Algorithm 2 style, JACM'17 p.24 lines
1--6):** The race is also stored as a ``PendingRace``.  At the end of the
execution (in ``next_execution()``), each pending race is processed by
computing a ``notdep`` sequence: the threads of events between the two
racing accesses that are *independent* of the first access.  An event is
independent if it is by a different thread AND its recorded accesses do
not conflict with the first event's accesses on any shared object.  The
resulting sequence (independent prefix + racing thread) is inserted into
the wakeup tree, providing a multi-step wakeup path that guides
exploration through independent intermediates.

**Deferred lock-release backtracking:** For multi-lock atomicity bugs,
the engine collects all lock acquire and release events during
execution.  At the end (in ``next_execution()``), for each lock release
by thread T where T later acquires another lock, backtrack opportunities
are inserted at the release position for all other threads.  This allows
another thread to interleave between T's two critical sections.  For
single-lock programs, no thread does acquire-after-release, so no
backtracks are inserted --- giving exact trace counts.


Wakeup trees
~~~~~~~~~~~~~~

A wakeup tree (JACM'17 Def 6.1) is an ordered tree of thread-ID
sequences at each scheduling point.  Each root-to-leaf path represents a
wakeup sequence --- an initial fragment of an execution that reverses a
detected race.

Insertion merges shared prefixes (if thread 0 already exists as a child,
descend into it) and adds new branches at the rightmost position.  The
current implementation uses exact prefix matching rather than the paper's
equivalence checking (``v ~ w``, Lemma 6.2), which is sound but may keep
redundant branches.

When ``step()`` backtracks, it picks the minimum thread ID from the
wakeup tree (deterministic proxy for the paper's ordering), extracts
the subtree for that thread, and stores it as
``pending_wakeup_subtree``.  During the next execution's scheduling, this
subtree guides thread choices at new branches beyond the replay prefix,
ensuring multi-step wakeup sequences are followed correctly.


Sleep sets and trace caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sleep sets prevent re-exploration of equivalent interleavings.  At each
scheduling point, after a thread is explored it is marked ``Visited``
and added to the local sleep set.  A thread in the sleep set is skipped
during wakeup tree insertion --- it will not be re-explored at this
position.

Sleep sets propagate across scheduling points via the independence
relation (JACM'17 Def 3.3): a sleeping thread q stays asleep at position
i+1 if q's recorded accesses are independent of the chosen thread p's
accesses at position i.  Independence means that for every shared object
accessed by both threads, the access kinds are non-conflicting.

To make this precise across executions, the engine maintains a
**step-count-indexed future access cache**.  After each completed
execution, ``step()`` computes per-thread suffix unions:
``future[tid][k]`` is the union of thread ``tid``'s accesses from its
k-th scheduling step onward.  During the next execution, a sleeping
thread that has completed k steps only needs to check its remaining work
(steps k, k+1, ...) against the active thread's accesses.  This prevents
false wakeups when a thread has already finished its conflicting work at
earlier steps.


Exploration structure
~~~~~~~~~~~~~~~~~~~~~~

The exploration tree is a stack of ``Branch`` nodes (one per scheduling
point).  Each branch records:

- Thread statuses (Disabled, Pending, Active, Visited, Blocked, Yield)
- The active (chosen) thread
- The wakeup tree for this position
- The sleep set (local + propagated)
- Per-thread access records for independence checks
- Preemption count (for bounded exploration)

Exploration proceeds as depth-first search.  ``step()`` walks backward
through the stack: at each branch, it marks the current thread as Visited,
removes it from the wakeup tree, and looks for the next unexplored thread.
If the wakeup tree has more branches, it picks the minimum thread and
resets ``pos`` to 0 for a full replay from scratch (stateless model
checking).  If the wakeup tree is empty, it pops the branch and continues
backtracking.

A **preemption bound** limits how many times the scheduler forces a
context switch away from a runnable thread.  When the bound is reached,
wakeup tree insertions that would require a preemption are deferred to an
ancestor branch where the target thread is already active (conservative
wakeup propagation).  This keeps exploration tractable while still
covering most concurrency bugs, which typically require few preemptions.


Resource groups and operation coalescing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When DPOR explores multi-table SQL patterns (e.g. django-reversion's
``create_revision()`` which touches both ``articles`` and ``versions``
tables), intermediate operations on unrelated tables create backtrack
points that cause exploration to explode before reaching the critical
interleaving.  **Resource group coalescing** addresses this.

Each SQL resource reported to the engine can be registered with a
**resource group** (typically the table name, hashed to a ``u64``).
The Python-side ``_io_reporter`` callback in ``dpor.py`` automatically
extracts the table name from SQL resource IDs (e.g. ``sql:versions:...``
→ group ``hash("sql:versions")``) and calls
``engine.register_resource_group(object_key, group_key)``.

When a race is detected in ``process_synced_io_access()``, the engine
checks whether the two racing accesses and the intermediate scheduling
steps involve **cross-group intermediates** --- steps where the active
thread accessed objects in a different resource group than the racing
object.  If cross-group intermediates exist, the inline wakeup insertion
is **skipped**.  The race is still collected as a ``PendingRace`` for
deferred ``notdep`` processing, which computes the independent
intermediate sequence and inserts it into the wakeup tree.

The effect: DPOR no longer creates backtrack points for every pair of
conflicting table accesses.  Instead, it lets the deferred ``notdep``
computation find a wakeup sequence that skips the independent
intermediate table operations and goes straight to the critical
interleaving.  For the django-reversion pattern, this reduces exploration
from exhausting the budget (100+ interleavings without finding the race)
to finding it in ~2 executions.

The ``has_cross_group_intermediates()`` check in ``path.rs`` iterates
over scheduling steps between the two racing positions and checks if any
step's ``active_accesses`` include objects from a resource group different
from the race object's group.  This is conservative: when no resource
groups are registered (non-SQL accesses), all races get inline wakeup
insertion as before.
