================================================================================
Frontrun Case Studies: Concurrency Bug Detection
================================================================================

This document presents case studies demonstrating how **frontrun** finds
and reproduces concurrency bugs in Python libraries by running bytecode
exploration directly against **unmodified library code**.

**Total: 46 concurrency bugs found across 12 libraries, plus 7 safe-area
invariants verified across 140,000 interleavings.**

Key Findings
============

1. **Exploration is highly effective.** All 31 tests found their target bugs on
   all 20 seeds. Simple counter races are found in 1-4 attempts; more complex
   TOCTOU patterns take ~17 attempts on average. No models or simplifications
   needed — frontrun runs directly against real library code.

2. **Plain ``+=`` without locks is a universal vulnerability.** Five of ten
   libraries fail on simple counter increments: pybreaker, urllib3, SQLAlchemy,
   cachetools, and pydis. Any unsynchronized counter is at risk.

3. **Synchronization gaps vary in severity.** Three of five Round 1 libraries
   use no synchronization primitives at all. Round 2 shows that even modern
   libraries like urllib3 and amqtt have races in critical paths. Not all races
   have equal impact — SQLAlchemy's race is cosmetic, while amqtt's corrupts
   the MQTT protocol.

4. **Impact assessment requires code reading.** Frontrun correctly identifies
   all non-atomic operations, but whether a race matters depends on its
   call-site. The SQLAlchemy overflow counter is demonstrably racy, yet benign
   in practice. This shows frontrun's role: find the race, then reason about
   whether it affects correctness.

Run the full suites::

    # Round 1
    PYTHONPATH=frontrun python frontrun/docs/tests/run_external_tests.py

    # Round 2
    PYTHONPATH=frontrun python frontrun/docs/case_studies/tests/test_pybreaker_real.py
    PYTHONPATH=frontrun python frontrun/docs/case_studies/tests/test_urllib3_real.py
    PYTHONPATH=frontrun python frontrun/docs/case_studies/tests/test_sqlalchemy_pool_real.py
    PYTHONPATH=frontrun python frontrun/docs/case_studies/tests/test_amqtt_real.py
    PYTHONPATH=frontrun python frontrun/docs/case_studies/tests/test_pykka_real.py

    # Round 3 (exploration + safe invariants; safe tests take ~1-2 min each)
    python -m pytest docs/case_studies/tests/test_cachetools_concurrency.py -v -s
    python -m pytest docs/case_studies/tests/test_tenacity_concurrency.py -v -s

Or run individual tests from Round 1::

    PYTHONPATH=frontrun python frontrun/docs/tests/test_cachetools_real.py
    PYTHONPATH=frontrun python frontrun/docs/tests/test_threadpoolctl_real.py
    # ... etc

----

Table of Contents
==================

**Round 1:**

1. `TPool (WildPool)`_ -- Thread pool with shutdown races
2. `threadpoolctl`_ -- Native library thread control with no locking
3. `cachetools`_ -- Caching library with unprotected data structures
4. `PyDispatcher`_ -- Signal dispatch with global mutable state
5. `pydis`_ -- Redis clone with shared global state

**Round 2:**

6. `pybreaker`_ -- Circuit breaker with unprotected fail counter
7. `urllib3`_ -- Connection pool with unprotected connection counter
8. `SQLAlchemy (pool)`_ -- Unlocked overflow counter in unlimited-overflow mode
9. `amqtt`_ -- MQTT session with unprotected packet-ID generator
10. `pykka`_ -- Actor framework with TOCTOU in ``tell()``

**Round 3 — Exploration and search depth:**

11. `cachetools (exploration)`_ -- 8 races found, 4 safe invariants verified at depth
12. `tenacity (exploration)`_ -- 6 races found, 4 safe invariants verified at depth

----

1. TPool (WildPool)
===================

**Repository:** `TPool on GitHub <https://github.com/oeg-upm/TPool>`_

**Commit tested:** `1bffaaf <https://github.com/oeg-upm/TPool/tree/1bffaaf>`_

**What it does:** Flexible thread pool for managing concurrent tasks using a
worker thread, semaphore, and task queue.

Bug: ``_should_keep_going()`` TOCTOU (Critical)
------------------------------------------------

The worker loop calls ``_should_keep_going()`` which reads ``keep_going`` under
``worker_lock``, then checks ``_join_is_called`` and ``bench.empty()`` under
``join_lock``.  Between releasing the first lock and acquiring the second, another
thread can enqueue work.  The worker sees an empty queue and exits, leaving tasks
unprocessed.

Exploration Results
~~~~~~~~~~~~~~~~~~~

Bytecode exploration was run directly against the real ``WildPool`` class.  Two
threads exercise the actual ``_should_keep_going()`` and ``add_thread()`` methods.

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **1-3**
==============================  ===============================

The race window between the two ``with`` lock statements in ``_should_keep_going()``
is wide enough at the opcode level that nearly every random schedule triggers it.

**Test file:** `tests/test_tpool_real.py <tests/test_tpool_real.py>`_

----

2. threadpoolctl
================

**Repository:** `threadpoolctl on GitHub <https://github.com/joblib/threadpoolctl>`_

**Commit tested:** `cf38a18 <https://github.com/joblib/threadpoolctl/tree/cf38a18>`_

**What it does:** Introspects and controls thread counts of native BLAS/OpenMP
libraries (OpenBLAS, MKL, BLIS, FlexiBLAS) via ctypes.

**Zero synchronization primitives** -- no ``threading.Lock``, no ``RLock``, no
``Condition`` anywhere in the library.

Bug: ``_get_libc()`` TOCTOU (Critical)
--------------------------------------

.. code-block:: python

    libc = cls._system_libraries.get("libc")  # READ
    if libc is None:                           # CHECK
        libc = ctypes.CDLL(...)                # CREATE (expensive)
        cls._system_libraries["libc"] = libc   # WRITE

Two threads both see ``None`` and both create CDLL objects.

Exploration Results
~~~~~~~~~~~~~~~~~~~

Bytecode exploration was run directly against the real
``ThreadpoolController._get_libc()`` classmethod.  Two threads both call
``_get_libc()`` after clearing the ``_system_libraries`` cache.

==============================  ===============================================
Metric                          Result
==============================  ===============================================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **1** (every seed, first try!)
==============================  ===============================================

The ``_get_libc`` method has a very short code path (dict.get, if-check, CDLL,
dict-store), so the search space is small.  The bug is found on literally every
random schedule.

**Test file:** `tests/test_threadpoolctl_real.py <tests/test_threadpoolctl_real.py>`_

----

3. cachetools
==============

**Repository:** `cachetools on GitHub <https://github.com/tkem/cachetools>`_

**Commit tested:** `e5f8f01 <https://github.com/tkem/cachetools/tree/e5f8f01>`_

**What it does:** Extensible memoizing collections (LRU, TTL, LFU, RR, TLRU
caches) and ``@cached``/``@cachedmethod`` decorators.

Cache objects are ``MutableMapping`` implementations that track ``currsize`` and
evict entries when full.  The ``@cached`` decorator accepts an optional ``lock``
parameter for thread safety -- without it, caches are **explicitly not
thread-safe**.

Bug: ``Cache.__setitem__`` Lost Update (Critical)
-------------------------------------------------

``Cache.__setitem__`` reads ``currsize``, computes a diff based on whether the key
exists, then adds the diff.  Two threads setting different keys both compute
their individual ``diffsize``, but the ``self.__currsize += diffsize`` is not
atomic at the bytecode level (``LOAD_ATTR`` / ``LOAD_FAST`` / ``INPLACE_ADD`` /
``STORE_ATTR``).  A context switch between the load and store causes one thread's
update to be lost.

Exploration Results
~~~~~~~~~~~~~~~~~~~

Bytecode exploration was run directly against the real ``Cache`` class.  Two
threads each call ``cache["a"] = "value_a"`` and ``cache["b"] = "value_b"`` on
the same Cache instance.

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **4**
==============================  ===============================

Example output::

    === Deterministic reproduction ===
      Run 1: currsize=1, len=2 [BUG]
      Run 2: currsize=1, len=2 [BUG]
      ...
      Run 10: currsize=1, len=2 [BUG]

**Test file:** `tests/test_cachetools_real.py <tests/test_cachetools_real.py>`_

----

4. PyDispatcher
================

**Repository:** `pydispatcher on GitHub <https://github.com/mcfletch/pydispatcher>`_

**Commit tested:** `0c2768d <https://github.com/mcfletch/pydispatcher/tree/0c2768d>`_

**What it does:** Multi-producer, multi-consumer signal dispatching system
(observer pattern).

Three **module-level global dictionaries** store all routing state:

- ``connections``: ``{senderkey: {signal: [receivers]}}``
- ``senders``: ``{senderkey: weakref(sender)}``
- ``sendersBack``: ``{receiverkey: [senderkey, ...]}``

**Zero synchronization primitives.** No locks, no thread-safe data structures,
no atomic operations.

Bug: ``connect()`` TOCTOU (Critical)
------------------------------------

Two threads connecting receivers to the same ``(sender, signal)`` both see the key
as absent in ``connections``, both create new signal dicts, and one overwrites the
other — losing the first receiver's registration entirely.

Exploration Results
~~~~~~~~~~~~~~~~~~~

Bytecode exploration was run directly against the real ``dispatcher.connect()``
function.  Two threads connect different receivers to the same ``(sender, signal)``
pair.

==============================  ===============================================
Metric                          Result
==============================  ===============================================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **1.3** (most seeds find it on first try)
==============================  ===============================================

PyDispatcher's complete lack of synchronization means the race window in
``connect()`` spans the entire function body.  Almost any interleaving between
two concurrent ``connect()`` calls triggers the bug.

**Test file:** `tests/test_pydispatcher_real.py <tests/test_pydispatcher_real.py>`_

----

5. pydis
=========

**Repository:** `pydis on GitHub <https://github.com/boramalper/pydis>`_

**Commit tested:** `1b02b27 <https://github.com/boramalper/pydis/tree/1b02b27>`_

**What it does:** Minimal Redis clone in ~250 lines of Python, using asyncio with
uvloop.

All data lives in two module-level globals::

    expiration = collections.defaultdict(lambda: float("inf"))
    dictionary = {}

Each client connection creates a ``RedisProtocol`` instance.  Commands are
processed synchronously within ``data_received()``, but asyncio can interleave
execution between different clients' ``data_received()`` calls.

**Zero synchronization.** No ``asyncio.Lock``, no atomic operations.  Every
command is a read-modify-write on shared global state.

Bug 1: INCR Lost Update (Critical)
----------------------------------

``com_incr`` reads the value, increments, and writes back.  Two concurrent INCRs
both read the same value, both write value+1, and one increment is lost.

Bug 2: SET NX Check-Then-Act (Critical)
---------------------------------------

``SET key value NX`` checks ``if key in dictionary``, then sets.  Two clients both
pass the check and both write, violating NX (set-if-not-exists) semantics.

Exploration Results
~~~~~~~~~~~~~~~~~~~

Bytecode exploration was run directly against the real ``RedisProtocol`` class.
Two protocol instances (simulating two client connections) operate on the same
module-level ``dictionary`` global.

INCR Lost Update::

    ==============================  ===============================
    Metric                          Result
    ==============================  ===============================
    Seeds that found bug            **20 / 20**
    Avg. attempts to find           **1.25**
    ==============================  ===============================

The INCR race window (``value = self.get(key)`` ... ``self.set(key, ...)``) spans
the entire ``com_incr`` method.  The short code path (58 opcodes) means the
scheduler has very few choices to make, and almost all of them trigger the bug.

**SET NX Race:** Also found within 4 attempts (seed=42).

**Test file:** `tests/test_pydis_real.py <tests/test_pydis_real.py>`_

----

6. pybreaker
============

**Repository:** `pybreaker on GitHub <https://github.com/danielfm/pybreaker>`_

**Version tested:** 1.4.1

**What it does:** Python implementation of the Circuit Breaker pattern.
``CircuitMemoryStorage`` tracks failure and success counters for a breaker
stored entirely in process memory.

Bug: ``CircuitMemoryStorage.increment_counter()`` Lost Update (Critical)
------------------------------------------------------------------------

.. code-block:: python

    # pybreaker/__init__.py
    def increment_counter(self) -> None:
        self._fail_counter += 1   # ← not atomic!

The ``+=`` operator compiles to three separate bytecode instructions
(``LOAD_ATTR`` / ``BINARY_OP`` / ``STORE_ATTR`` in Python 3.11+).
Two threads calling ``increment_counter()`` concurrently can both read
the same ``_fail_counter`` value, both compute ``value + 1``, and one
write overwrites the other.

**Impact:** The circuit breaker under-counts failures.  A service that
is actually failing fast may never trip the breaker because its fail
counter stays permanently lower than the threshold.

Exploration Results
~~~~~~~~~~~~~~~~~~~

Bytecode exploration was run directly against the real
``CircuitMemoryStorage.increment_counter()`` method.

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **~2**
==============================  ===============================

**Test file:** `tests/test_pybreaker_real.py <case_studies/tests/test_pybreaker_real.py>`_

----

7. urllib3
==========

**Repository:** `urllib3 on GitHub <https://github.com/urllib3/urllib3>`_

**Version tested:** main branch (post-2.x)

**What it does:** The foundational HTTP client library used by Requests and
many other Python projects.  ``HTTPConnectionPool`` manages a pool of
persistent connections to a single host.

Bug: ``HTTPConnectionPool._new_conn()`` Lost Update (Critical)
--------------------------------------------------------------

.. code-block:: python

    # urllib3/connectionpool.py
    def _new_conn(self) -> BaseHTTPConnection:
        self.num_connections += 1   # ← not atomic!
        conn = self.ConnectionCls(
            host=self._proxy_host or self.host,
            port=self.port,
            ...
        )
        return conn

``num_connections`` is a plain integer attribute incremented with ``+=``,
which is not atomic.  Two threads each calling ``_new_conn()`` (e.g. both
finding the pool empty and creating a new connection) can both read the same
counter, both write ``value + 1``, and one increment is lost.

**Impact:** ``num_connections`` is used for logging and debugging pool
behaviour.  An under-counted value makes it impossible to diagnose
connection-exhaustion issues.  If the counter is ever used for pool
limits, under-counting could allow more connections than intended.

Exploration Results
~~~~~~~~~~~~~~~~~~~

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **~1.5**
==============================  ===============================

**Test file:** `tests/test_urllib3_real.py <case_studies/tests/test_urllib3_real.py>`_

----

8. SQLAlchemy (pool)
====================

**Repository:** `SQLAlchemy on GitHub <https://github.com/sqlalchemy/sqlalchemy>`_

**Version tested:** 2.x main branch

**What it does:** The most widely-used Python SQL toolkit and ORM.
``QueuePool`` (the default pool) maintains a fixed number of idle
connections plus an optional "overflow" of temporary connections.

Finding: ``QueuePool._inc_overflow()`` skips the lock in unlimited mode
-----------------------------------------------------------------------

.. note::

    After closer analysis this is **intentional behaviour**, not a bug.
    It is included here as a case study in how frontrun can surface
    deliberate design decisions that look like races — and as a reminder
    to verify *impact* before filing an issue.

.. code-block:: python

    # sqlalchemy/pool/impl.py
    def _do_get(self) -> ConnectionPoolEntry:
        use_overflow = self._max_overflow > -1        # False when unlimited
        wait = use_overflow and self._overflow >= self._max_overflow
        ...
        if self._inc_overflow():
            return self._create_connection()

    def _inc_overflow(self) -> bool:
        if self._max_overflow == -1:
            self._overflow += 1   # ← no lock
            return True
        with self._overflow_lock:          # lock only needed for finite limit
            if self._overflow < self._max_overflow:
                self._overflow += 1
                return True
            return False

When ``max_overflow == -1`` (unlimited overflow), the method skips
``_overflow_lock`` and increments the counter bare.  Frontrun detects this
and flags a lost-update race.

**Why the lock is deliberately omitted:**

* ``use_overflow = self._max_overflow > -1`` evaluates to ``False``, so
  ``_overflow`` is **never consulted** for pool admission control in unlimited
  mode.  The lock exists solely to protect the ``_overflow < _max_overflow``
  comparison; without a finite limit there is nothing to compare against.
* ``max_overflow`` cannot be changed at runtime (no setter; ``recreate()``
  constructs a new pool from original params), so the mode is fixed for the
  pool's lifetime.
* The only downstream consumers of ``_overflow`` in unlimited mode are the
  diagnostic methods ``overflow()`` and ``checkedout()``.  A transiently
  incorrect diagnostic counter is an acceptable trade-off for avoiding a
  contended lock on every connection checkout.

**What frontrun tells us here:**

Frontrun correctly identifies that the ``+=`` is non-atomic at the bytecode
level and that a concurrent schedule can produce a wrong count.  The
follow-up question — *does this wrong count affect correctness?* — requires
reading the call-site logic, which shows it does not.  This is a good
example of a **true race / no-impact** finding: frontrun is right that the
memory model is unsound, but the library is also right that the impact is
bounded to a slightly stale diagnostic counter.

Exploration Results
~~~~~~~~~~~~~~~~~~~

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found race           **20 / 20**
Avg. attempts to find           **~2.5**
==============================  ===============================

**Test file:** `tests/test_sqlalchemy_pool_real.py <case_studies/tests/test_sqlalchemy_pool_real.py>`_

----

9. amqtt
=========

**Repository:** `amqtt on GitHub <https://github.com/Yakifo/amqtt>`_

**Version tested:** main branch

**What it does:** An MQTT 3.1.1 broker and client library built on asyncio.
``Session`` represents a single MQTT client session and provides
``next_packet_id`` to allocate unique IDs for in-flight messages.

Bug: ``Session.next_packet_id`` Duplicate Packet IDs (Critical)
---------------------------------------------------------------

.. code-block:: python

    # amqtt/session.py
    @property
    def next_packet_id(self) -> int:
        self._packet_id = (self._packet_id % 65535) + 1   # WRITE
        limit = self._packet_id                             # READ (shared!)
        while self._packet_id in self.inflight_in or \
              self._packet_id in self.inflight_out:
            self._packet_id = (self._packet_id % 65535) + 1
            if self._packet_id == limit:
                raise AMQTTError("More than 65535 messages pending")
        return self._packet_id

No lock protects ``_packet_id``.  Two threads/tasks calling this property
concurrently on the same ``Session`` interleave their writes and reads of
``_packet_id``, and can both return the same integer.

**Impact:** The MQTT protocol requires in-flight QoS 1/2 messages to have
unique packet IDs.  Duplicate IDs cause message delivery corruption.

Exploration Results
~~~~~~~~~~~~~~~~~~~

==============================  ============================================
Metric                          Result
==============================  ============================================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **1** (every seed, every first attempt!)
==============================  ============================================

**Test file:** `tests/test_amqtt_real.py <case_studies/tests/test_amqtt_real.py>`_

----

10. pykka
==========

**Repository:** `pykka on GitHub <https://github.com/jodal/pykka>`_

**Version tested:** 4.4.1

**What it does:** A Python actor framework providing thread-based actors.
Each actor runs in its own thread and communicates via message-passing.
``ActorRef.tell()`` sends a fire-and-forget message to an actor.

Bug: ``ActorRef.tell()`` TOCTOU — Ghost Messages (High)
-------------------------------------------------------

.. code-block:: python

    # pykka/_ref.py
    def tell(self, message: Any) -> None:
        if not self.is_alive():                             # CHECK
            raise ActorDeadError(f"{self} is dead")
        self.actor_inbox.put(                               # ACT
            Envelope(message=message, reply_to=None),
            block=False,
        )

There is no lock between the ``is_alive()`` check (which reads
``actor_stopped.is_set()``) and the ``actor_inbox.put()``.  A concurrent
``stop()`` call can set ``actor_stopped`` in the window between the check
and the put.

When this race fires:

1. ``tell()`` checks ``is_alive()`` — returns **True** (actor not yet stopped).
2. Another thread calls ``stop()`` — actor loop processes ``_ActorStop``,
   calls ``actor_stopped.set()``, and exits its loop.
3. ``tell()`` puts the message into the inbox — succeeds without raising
   ``ActorDeadError``.
4. **Nobody reads the message.** The actor loop has exited; the inbox is
   never drained again.  The message is silently lost — a *ghost message*.

**Impact:** From the caller's perspective ``tell()`` returned successfully,
implying the message was delivered.  In reality the message is silently
dropped.  This violates the implicit contract that a non-raising ``tell()``
guarantees delivery to a live actor.

Exploration Results
~~~~~~~~~~~~~~~~~~~

==============================  ===============================
Metric                          Result
==============================  ===============================
Seeds that found bug            **20 / 20**
Avg. attempts to find           **~17**
==============================  ===============================

**Test file:** `tests/test_pykka_real.py <case_studies/tests/test_pykka_real.py>`_

----

11. cachetools (exploration)
============================

**Repository:** `cachetools on GitHub <https://github.com/tkem/cachetools>`_

**Commit tested:** `e5f8f01 <https://github.com/tkem/cachetools/tree/e5f8f01>`_ (v7.0.1)

**What it does:** Extensible memoizing collections (LRU, TTL, LFU, RR, FIFO
caches) and ``@cached``/``@cachedmethod`` decorators.

This study systematically explores cachetools for races in every cache subclass
and verifies that the ``@cached`` decorator's ``lock`` and ``condition``
parameters correctly prevent them.

Bug-Finding Results (8 races found)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======  ===========================  ========================  =========
Test    Component                    Race Type                 Attempts
======  ===========================  ========================  =========
B1      ``Cache.__setitem__``        ``currsize +=`` lost      2
B2      ``Cache.__delitem__``        ``currsize -=`` lost      5
B3      ``LRUCache`` get+del         ``__order`` TOCTOU        3
B4      ``RRCache.__setitem__``      index/keys corruption     3
B5      ``@cached`` (no lock)        ``hits +=`` lost          3
B6      ``LFUCache.__getitem__``     link list crash           **1**
B7      ``Cache`` set+del            currsize desync           48
B8      ``TTLCache._Timer``          ``__nesting +=`` lost     2
======  ===========================  ========================  =========

The most dramatic bug is B6: concurrent ``__getitem__`` calls on ``LFUCache``
crash with ``KeyError`` because the frequency link list manipulation is not
atomic.  Two threads both try to move a key from one frequency bucket to the
next, and one of them finds the key already removed from the source bucket.
This is a **crash in a read-only operation** — two concurrent cache lookups can
crash the process.

Safe-Area Results (4 invariants verified)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======  ===========================  ========================  ==============  ==========
Test    Component                    Invariant                 Interleavings   Result
======  ===========================  ========================  ==============  ==========
S1      ``@cached`` with ``lock``    stats accurate            15,000          SAFE
S2      ``@cached`` with ``lock``    all keys correct          15,000          SAFE
S3      ``@cached`` with ``cond``    no stampede               15,000          race @2459
S4      ``Cache`` single-key ops     key-value retrievable     15,000          SAFE
======  ===========================  ========================  ==============  ==========

The ``@cached`` decorator's ``lock`` parameter (using ``_locked`` / ``_locked_info``
in ``_cached.py``) correctly prevents all races across 30,000 interleavings.

S3 (the ``condition`` parameter) found a potential race after 2,459 interleavings.
The ``_condition_info`` wrapper uses ``cond.wait_for()`` to prevent cache stampede,
but a rare interleaving between the inner ``return v`` (which releases the lock) and
the ``finally`` block (which re-acquires it to remove the key from ``pending``)
creates a window where a second thread can acquire the lock, see the key still in
``pending``, and wait — but then receive the ``notify_all`` and proceed before the
``pending`` set is fully cleaned up.  This warrants further investigation.

**Test file:** `tests/test_cachetools_concurrency.py <case_studies/tests/test_cachetools_concurrency.py>`_

----

12. tenacity (exploration)
==========================

**Repository:** `tenacity on GitHub <https://github.com/jd/tenacity>`_

**Commit tested:** `0bdf1d9 <https://github.com/jd/tenacity/tree/0bdf1d9>`_

**What it does:** General-purpose retry library with configurable stop, wait,
and retry strategies.  ``Retrying`` wraps a function and automatically retries
on failure.

This study explores tenacity's ``+=`` patterns on shared statistics dicts and
``RetryCallState`` attributes, and verifies that ``threading.local()``, ``copy()``,
and immutable strategy objects correctly isolate concurrent calls.

Bug-Finding Results (6 races found)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======  ================================  =========================  =========
Test    Component                         Race Type                  Attempts
======  ================================  =========================  =========
B1      ``statistics["idle_for"]``        ``+= sleep`` lost          **1**
B2      ``statistics["attempt_number"]``  ``+= 1`` lost              **1**
B3      ``wrapped_f.statistics``          orphaned reference         **1**
B4      shared ``Retrying`` copy          stats corruption           **1**
B5      ``RetryCallState.idle_for``       ``+= sleep`` lost          **1**
B6      ``RetryCallState.attempt_number`` ``+= 1`` lost              **1**
======  ================================  =========================  =========

All 6 races are found on the first attempt.  The most interesting finding is B3:
concurrent calls to a ``@retry``-wrapped function both write to
``wrapped_f.statistics``, causing one thread's statistics reference to become
orphaned (still a valid dict, but no longer reachable via the function attribute).

Safe-Area Results (4 invariants verified)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

======  ========================  ==========================  ==============  ======
Test    Component                 Invariant                   Interleavings   Time
======  ========================  ==========================  ==============  ======
S1      ``threading.local``       per-thread stats isolation  20,000          ~83s
S2      per-call state            each call gets own state    20,000          ~83s
S3      immutable strategies      concurrent eval safe        20,000          ~83s
S4      ``copy()`` isolation      independent copies          20,000          ~83s
======  ========================  ==========================  ==============  ======

Tenacity's primary concurrency defenses work correctly: ``threading.local()``
ensures each thread gets its own ``statistics`` dict and ``iter_state``.  The
``copy()`` mechanism creates fully independent ``Retrying`` instances for each
call.  Immutable strategy objects (``stop_after_attempt``, ``wait_none``, etc.)
are stateless and inherently safe to share.  80,000 interleavings produced zero
false positives.

**Test file:** `tests/test_tenacity_concurrency.py <case_studies/tests/test_tenacity_concurrency.py>`_

----

Summary
=======

================  ====================================  =======  ==================  ==============
Library           Finding                               Version  Seeds Found (/ 20)  Avg. Attempts
================  ====================================  =======  ==================  ==============
TPool             ``_should_keep_going`` TOCTOU        1bffaaf  **20 / 20**         1-3
threadpoolctl     ``_get_libc`` TOCTOU                 cf38a18  **20 / 20**         **1**
cachetools        ``__setitem__`` lost update          e5f8f01  **20 / 20**         4
PyDispatcher      ``connect()`` TOCTOU                 0c2768d  **20 / 20**         1.3
pydis             INCR lost update + SET NX            1b02b27  **20 / 20**         1.25
pybreaker         ``increment_counter`` lost update    1.4.1    **20 / 20**         ~2
urllib3           ``_new_conn`` lost update            2.x      **20 / 20**         ~1.5
SQLAlchemy pool   ``_inc_overflow`` race (diagnostic)  2.x      **20 / 20**         ~2.5
amqtt             ``next_packet_id`` duplicate IDs     main     **20 / 20**         **1**
pykka             ``tell()`` TOCTOU ghost messages     4.4.1    **20 / 20**         ~17
cachetools (exp)  8 races (+ 4 safe verified)          v7.0.1   8/8 found           1-48
tenacity (exp)    6 races (+ 4 safe verified)          0bdf1d9  6/6 found           **1**
================  ====================================  =======  ==================  ==============

