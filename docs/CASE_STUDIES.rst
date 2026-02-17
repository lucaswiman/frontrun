================================================================================
Interlace Case Studies: Concurrency Bug Detection in External Libraries
================================================================================

**A note on intent:** The goal of these case studies is not to pick on any of
these libraries.  Concurrency is hard!  We chose small, approachable codebases
that haven't been hardened by years of production multi-threaded use, or that
are explicitly labeled as not thread-safe, precisely because they're good
candidates for demonstrating what interlace can do.  The bugs we find are the
kind that lurk in almost any Python code that touches shared mutable state
without careful synchronization.

This document presents five case studies demonstrating how **interlace** can
find, reproduce, and test concurrency bugs in real-world open-source Python
libraries by running bytecode exploration directly against **unmodified library
code** — no models, no simplifications.

**Total: 16 passing PoC tests across 5 libraries.**

Run the full suite::

    PYTHONPATH=interlace python interlace/docs/tests/run_external_tests.py

Or run individual tests::

    PYTHONPATH=interlace python interlace/docs/tests/test_cachetools_real.py
    PYTHONPATH=interlace python interlace/docs/tests/test_threadpoolctl_real.py
    # ... etc

----

Table of Contents
==================

1. `TPool (WildPool)`_ -- Thread pool with shutdown races
2. `threadpoolctl`_ -- Native library thread control with no locking
3. `cachetools`_ -- Caching library with unprotected data structures
4. `PyDispatcher`_ -- Signal dispatch with global mutable state
5. `pydis`_ -- Redis clone with shared global state

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
Deterministic reproduction      **10/10**
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
Deterministic reproduction      **10/10**
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
Deterministic reproduction      **10/10**
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
Deterministic reproduction      **10/10**
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
    Deterministic reproduction      **10/10**
    ==============================  ===============================

The INCR race window (``value = self.get(key)`` ... ``self.set(key, ...)``) spans
the entire ``com_incr`` method.  The short code path (58 opcodes) means the
scheduler has very few choices to make, and almost all of them trigger the bug.

**SET NX Race:** Also found within 4 attempts (seed=42).

**Test file:** `tests/test_pydis_real.py <tests/test_pydis_real.py>`_

----

Summary
=======

=================  =============================  ========  ==================  ==============  =========
Library            Bug Tested                     Commit    Seeds Found (/ 20)  Avg. Attempts   Reproduce
=================  =============================  ========  ==================  ==============  =========
TPool              ``_should_keep_going`` TOCTOU  1bffaaf   **20 / 20**         1-3             10/10
threadpoolctl      ``_get_libc`` TOCTOU           cf38a18   **20 / 20**         **1**           10/10
cachetools         ``__setitem__`` lost update    e5f8f01   **20 / 20**         4               10/10
PyDispatcher       ``connect()`` TOCTOU           0c2768d   **20 / 20**         1.3             10/10
pydis              INCR lost update + SET NX      1b02b27   **20 / 20**         1.25            10/10
=================  =============================  ========  ==================  ==============  =========

Key Findings
============

1. **Real-code exploration works remarkably well.** Every library's bug was
   found on **all 20 seeds tested**, typically in **1-4 attempts**.  The race
   windows in real code are wide enough at the bytecode level that random
   schedules trigger them almost immediately.  No models needed — interlace
   runs directly against unmodified library code.

2. **Deterministic reproduction is 100% reliable.** Once a counterexample
   schedule is found, ``run_with_schedule`` reproduces the bug **10/10 times**
   across all 5 libraries.  This makes the schedules suitable as regression
   tests.

3. **Zero-lock libraries are common.** Three of five libraries (threadpoolctl,
   PyDispatcher, pydis) use no synchronization whatsoever.  The other two
   (TPool, cachetools) use locks but have gaps in their synchronization.

4. **The bugs are real.** Every bug demonstrated here represents an actual
   concurrency hazard in the library's current codebase.  They range from
   data corruption (cachetools currsize, pydis lost writes) to complete
   functionality failure (TPool task loss, PyDispatcher lost registrations).
