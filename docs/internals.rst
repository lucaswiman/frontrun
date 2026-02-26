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
``should_trace_file()`` in ``_tracing.py``:

.. code-block:: python

   def should_trace_file(filename: str) -> bool:
       """Skip stdlib, site-packages, and frontrun internals."""
       if filename.startswith("<"):
           return False
       if filename.startswith(_FRONTRUN_DIR):
           return False
       for skip_dir in _SKIP_DIRS:
           if filename.startswith(skip_dir):
               return False
       return True

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
policy: DPOR uses a Rust engine that tracks shared-memory accesses via
vector clocks and only explores alternative orderings at conflict points.
Optionally adds ``sys.setprofile`` for C-call detection and
monkey-patched I/O.

**``LD_PRELOAD`` interception** is orthogonal to the Python-level
mechanisms.  It runs whenever the ``frontrun`` CLI is used, regardless
of which approach (or no approach) is active on the Python side.  Events
from the Rust interception library are available via
``IOEventDispatcher`` for any consumer.

.. note::

   DPOR consumes ``LD_PRELOAD`` events when ``detect_io=True`` (the
   default).  ``explore_dpor()`` starts an ``IOEventDispatcher`` that
   reads the pipe, and a ``_PreloadBridge`` maps OS thread IDs to DPOR
   logical thread IDs and buffers events for draining at each scheduling
   point.  This means C extensions that call libc ``send()``/``recv()``
   directly (e.g. psycopg2 via libpq) are covered --- DPOR treats the
   shared socket endpoint as a conflict and explores alternative
   orderings around the I/O.

   The bytecode explorer does *not* consume ``LD_PRELOAD`` events.  It
   relies on Python-level monkey-patching (and random scheduling) to
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


What each layer can and cannot see
------------------------------------

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

**External server state** (e.g. a row in PostgreSQL):  The socket-level
conflict (two threads talking to ``127.0.0.1:5432``) *is* visible to
``LD_PRELOAD``.  But the ``LD_PRELOAD`` library can only see that two
threads are using the same socket endpoint --- it can't distinguish a
``SELECT`` on table A from an ``UPDATE`` on table B.  If the preload
events were wired into DPOR, it would explore reorderings of all
database I/O between the two threads, which is a coarse but potentially
useful signal.  The underlying row-level conflict is invisible to any
client-side instrumentation; only the database server knows which rows
are being accessed.  For precise control over database-level races, use
trace markers with manual scheduling.
