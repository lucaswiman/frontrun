Vector Clocks and Happens-Before
=================================

This page gives a self-contained explanation of the **vector clock**
mechanism that underpins Frontrun's DPOR race detection.  It covers the
theory (the happens-before partial order), the data structure
(``VersionVec``), the two per-thread clocks and why each exists, and
worked examples showing exactly how races are detected --- or suppressed
--- by synchronization.


.. contents:: Contents
   :local:
   :depth: 2


The happens-before relation
---------------------------

Two concurrent operations can only be a **race** if neither is guaranteed
to execute before the other.  The `happens-before relation
<https://en.wikipedia.org/wiki/Happened-before>`_
:math:`\to` formalizes this guarantee as the smallest partial order
satisfying:

.. math::

   \text{(program order)} \quad & e_i \to e_j
     \text{ if } e_i \text{ and } e_j \text{ are by the same thread and }
     i < j \\[4pt]
   \text{(lock synchronization)} \quad & \text{release}(l) \to
     \text{acquire}(l) \text{ for any lock } l \\[4pt]
   \text{(spawn)} \quad & \text{spawn}(t) \to e
     \text{ for the first event } e \text{ of thread } t \\[4pt]
   \text{(join)} \quad & e \to \text{join}(t)
     \text{ for the last event } e \text{ of thread } t \\[4pt]
   \text{(transitivity)} \quad & e_i \to e_k
     \text{ if } e_i \to e_j \text{ and } e_j \to e_k

Two events :math:`e_i` and :math:`e_j` are **concurrent** (written
:math:`e_i \| e_j`) when neither happens-before the other:

.. math::

   e_i \| e_j \;\iff\; e_i \not\to e_j \;\wedge\; e_j \not\to e_i

A **race** is a pair of concurrent events that access the same shared
object with at least one write (JACM'17 Def 3.3, p.13).  Only races need
alternative scheduling --- events that are happens-before ordered will
always execute in that order regardless of the schedule.


The ``VersionVec`` data structure
---------------------------------

A `vector clock <https://en.wikipedia.org/wiki/Vector_clock>`_ is an array
of counters with one entry per thread:

.. math::

   V : \text{ThreadId} \to \mathbb{N}

Frontrun implements this as ``VersionVec`` (``crates/dpor/src/vv.rs``), a
contiguous ``Vec<u32>`` indexed by thread ID.  The four key operations are:

``increment(thread_id)``
    Advance the local component:

    .. math::

       V[t] \leftarrow V[t] + 1

    Called each time a thread is scheduled (each scheduling point).

``join(other)``
    Point-wise maximum:

    .. math::

       V[i] \leftarrow \max(V[i], V'[i]) \quad \forall\, i

    Used when a synchronization event transfers causal knowledge between
    threads --- lock acquire joins the releasing thread's clock, thread
    join joins the joined thread's clock, etc.

``partial_le(other)``
    Returns ``true`` if :math:`V[i] \le V'[i]` for every component.  This
    is the **happens-before test**:

    .. math::

       V \le V' \;\iff\; \forall\, i : V[i] \le V'[i]

    If ``a.partial_le(b)`` is true, then the event that produced clock *a*
    happens-before the event that produced clock *b*.

``concurrent_with(other)``
    Returns ``true`` when neither clock dominates the other:

    .. math::

       V \| V' \;\iff\; V \not\le V' \;\wedge\; V' \not\le V

    Concurrent events are potential races if they also conflict on a shared
    object.

.. note::

   The ``VersionVec`` implementation automatically extends the internal
   ``Vec`` when a thread ID exceeds its length (padding with zeros).
   This means thread IDs need not be known up front.


Two clocks per thread
---------------------

Each thread in the DPOR engine maintains **two** vector clocks:

.. list-table::
   :header-rows: 1
   :widths: 18 20 20 42

   * - Clock
     - Incremented on
     - Joined on
     - Purpose
   * - ``dpor_vv``
     - Every scheduling step
     - Lock acquire, lock release, thread spawn, thread join
     - Primary race detection for shared-memory accesses. Includes lock
       edges, so accesses protected by the same lock appear ordered (no
       false race).
   * - ``io_vv``
     - Every scheduling step
     - Thread spawn, thread join (**not** lock acquire/release)
     - I/O race detection (files, sockets, databases via ``LD_PRELOAD``).
       Omits lock edges so I/O accesses from different threads *always*
       appear potentially concurrent --- catches TOCTOU races that
       lock-aware tracking would suppress.


Why two happens-before clocks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a program where two threads access a shared file, protected by a
Python lock:

.. code-block:: python

   lock = threading.Lock()

   def thread_0():
       with lock:
           f = open("/tmp/data", "w")
           f.write("hello")
           f.close()

   def thread_1():
       with lock:
           f = open("/tmp/data", "r")
           data = f.read()
           f.close()

From Python's perspective, the lock makes these accesses ordered --- T0's
critical section happens-before T1's (or vice versa).  The ``dpor_vv``
clock reflects this ordering, so ``process_access()`` would **not** report
a race on the Python attribute accesses.

But the *file system* doesn't know about the Python lock.  If the program
has a bug where the lock is sometimes missing, or a different code path
accesses the file without the lock, the file-level access is still a
potential TOCTOU (time-of-check-to-time-of-use) race.

The ``io_vv`` clock solves this by **not joining on lock events**.  Two
I/O accesses by different threads always have concurrent ``io_vv`` values
(assuming no thread spawn/join ordering), so ``process_io_access()`` will
flag them as races and explore alternative interleavings around the I/O.
This is conservative (may over-explore) but catches real bugs.


Synchronization events and clock updates
-----------------------------------------

The engine processes four synchronization events, each updating the clocks:

Lock release
~~~~~~~~~~~~

The releasing thread's ``dpor_vv`` clock is stored on the lock:

.. math::

   V_{\text{lock}} \leftarrow V_t

This snapshot captures everything the releasing thread "knows about" ---
its own operations and everything it learned from prior synchronization.

Lock acquire
~~~~~~~~~~~~

The acquiring thread joins the stored clock:

.. math::

   V_t \leftarrow V_t \sqcup V_{\text{lock}}

where :math:`\sqcup` is the component-wise maximum (the ``join``
operation).  After this, the acquirer's clock dominates the releaser's,
establishing that the acquire happens-after the release.

**Critically**, the ``io_vv`` clock is **not** updated on lock
acquire/release.  This is the mechanism that keeps I/O accesses
"concurrent" even when protected by Python locks.

Thread spawn
~~~~~~~~~~~~

The child thread inherits the parent's clocks:

.. math::

   V_{\text{child}} \leftarrow V_{\text{child}} \sqcup V_{\text{parent}}

This applies to both clocks (``dpor_vv`` and ``io_vv``).  All of the
parent's operations before the spawn happen-before the child's first
operation.

Thread join
~~~~~~~~~~~

The joining thread absorbs the joined thread's clocks:

.. math::

   V_t \leftarrow V_t \sqcup V_{t'}

Again, both clocks are updated.  Everything the joined thread did now
happens-before the joiner's subsequent operations.


Worked examples
---------------

These examples trace the vector clock values step by step to show how races
are detected or suppressed.


Example 1: unsynchronized access (race detected)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two threads write to the same shared object with no synchronization:

.. code-block:: python

   shared = Container()

   def thread_0():
       shared.value = 1    # W_0(shared)

   def thread_1():
       shared.value = 2    # W_1(shared)

.. code-block:: text

   Step 0: schedule T0     → T0.dpor_vv = [1, 0]
           W_0(shared): no prior access → no conflict

   Step 1: schedule T1     → T1.dpor_vv = [0, 1]
           W_1(shared): prior write by T0 at step 0 with vv [1, 0]
           Check: [1, 0] ≤ [0, 1]?
             Component 0: 1 ≤ 0?  NO
           Check: [0, 1] ≤ [1, 0]?
             Component 1: 1 ≤ 0?  NO
           → CONCURRENT → RACE DETECTED
           → Insert T1 into wakeup tree at branch 0

T0's clock has component 0 ahead; T1's has component 1 ahead.  Neither
dominates, so the accesses are concurrent.  DPOR will explore the
alternative ordering (T1 before T0).


Example 2: lock-synchronized access (no race)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same program, but with a lock protecting the shared object:

.. code-block:: python

   lock = threading.Lock()
   shared = Container()

   def thread_0():
       with lock:
           shared.value = 1

   def thread_1():
       with lock:
           shared.value = 2

.. code-block:: text

   Step 0: schedule T0         → T0.dpor_vv = [1, 0]
           T0 acquires lock    → no prior release, clock unchanged
           W_0(shared): no prior access → no conflict
           T0 releases lock    → store T0.dpor_vv [1, 0] on lock

   Step 1: schedule T1         → T1.dpor_vv = [0, 1]
           T1 acquires lock    → join with stored [1, 0]
                                  T1.dpor_vv = max([0, 1], [1, 0]) = [1, 1]
           W_1(shared): prior write by T0 at step 0 with vv [1, 0]
           Check: [1, 0] ≤ [1, 1]?
             Component 0: 1 ≤ 1?  YES
             Component 1: 0 ≤ 1?  YES
           → HAPPENS-BEFORE → no race

The lock acquire joined T0's clock into T1, making T0's write
happen-before T1's write.  No alternative ordering is explored.


Example 3: lock-synchronized I/O (TOCTOU race detected)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same lock-protected code, but now the accesses are file I/O reported via
``LD_PRELOAD`` (using ``io_vv``):

.. code-block:: python

   lock = threading.Lock()

   def thread_0():
       with lock:
           open("/tmp/data", "w").write("hello")  # I/O write

   def thread_1():
       with lock:
           data = open("/tmp/data", "r").read()    # I/O read

.. code-block:: text

   Step 0: schedule T0         → T0.io_vv = [1, 0]
           T0 acquires lock    → io_vv NOT joined (lock events skipped)
           I/O write to /tmp/data: no prior I/O access → no conflict
           T0 releases lock    → io_vv NOT updated

   Step 1: schedule T1         → T1.io_vv = [0, 1]
           T1 acquires lock    → io_vv NOT joined
           I/O read from /tmp/data: prior I/O write by T0 with io_vv [1, 0]
           Check: [1, 0] ≤ [0, 1]?
             Component 0: 1 ≤ 0?  NO
           → CONCURRENT under io_vv → RACE DETECTED

Even though the Python lock orders the operations, the ``io_vv`` clock
ignores the lock, so the I/O accesses appear concurrent.  DPOR will
explore the alternative ordering.  This catches bugs where:

- The lock is sometimes missing on a different code path
- A third thread accesses the file without the lock
- The program relies on file-system state that isn't actually protected


Example 4: thread spawn establishes ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   shared = Container()
   shared.value = 0

   def parent():
       shared.value = 1          # W_parent(shared)
       t = threading.Thread(target=child)
       t.start()                 # spawn

   def child():
       x = shared.value          # R_child(shared)

.. code-block:: text

   Step 0: schedule parent   → parent.dpor_vv = [1, 0]
           W_parent(shared): no prior access → no conflict

   Step 1: spawn child       → child.dpor_vv = join([0, 0], [1, 0]) = [1, 0]

   Step 2: schedule child    → child.dpor_vv = [1, 1]
           R_child(shared): prior write by parent at step 0 with vv [1, 0]
           Check: [1, 0] ≤ [1, 1]?
             Component 0: 1 ≤ 1?  YES
             Component 1: 0 ≤ 1?  YES
           → HAPPENS-BEFORE → no race

The spawn transfers the parent's clock to the child, so the parent's write
is ordered before the child's read.


Example 5: concurrent access after spawn (race detected)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   shared = Container()

   def parent():
       t = threading.Thread(target=child)
       t.start()
       shared.value = 1          # W_parent(shared) — AFTER spawn

   def child():
       shared.value = 2          # W_child(shared)

.. code-block:: text

   Step 0: schedule parent   → parent.dpor_vv = [1, 0]
           spawn child       → child.dpor_vv = join([0, 0], [1, 0]) = [1, 0]

   Step 1: schedule parent   → parent.dpor_vv = [2, 0]
           W_parent(shared): no prior access → no conflict

   Step 2: schedule child    → child.dpor_vv = [1, 1]
           W_child(shared): prior write by parent at step 1 with vv [2, 0]
           Check: [2, 0] ≤ [1, 1]?
             Component 0: 2 ≤ 1?  NO
           → CONCURRENT → RACE DETECTED

The parent wrote *after* the spawn, so the spawn's clock transfer doesn't
cover it.  The parent's post-spawn write and the child's write are
genuinely concurrent.


How race detection drives DPOR
-------------------------------

When a race is detected, the engine performs two actions:

1. **Inline insertion**: insert a single-element wakeup sequence
   ``[racing_thread]`` into the wakeup tree at the branch where the first
   racing access occurred.  This guarantees the race is never lost.

2. **Deferred notdep processing**: store the race as a ``PendingRace``.
   At the end of the execution, compute the **notdep sequence** ---
   threads between the two racing accesses whose work is independent of
   the first access --- and insert the full multi-step sequence.  This
   guides future exploration through independent intermediates, reducing
   unnecessary branching.

The vector clock comparison (``partial_le`` / ``concurrent_with``) is the
*sole decision point* for whether a race exists.  Everything else in the
DPOR algorithm --- wakeup trees, sleep sets, notdep sequences --- flows
from this check.  See :doc:`dpor` for the full algorithm description and
:doc:`search` for how search strategies affect exploration order.


Complexity
----------

**Per scheduling point**: :math:`O(T)` where :math:`T` is the number of
threads, dominated by the vector-clock comparison (one ``partial_le`` call
iterates over all components).

**Per synchronization event**: :math:`O(T)` for the ``join`` operation
(component-wise maximum of two vectors).

**Space per thread**: :math:`O(T)` for each of the two vector clocks
(two ``Vec<u32>`` of length :math:`T`).

For programs with many threads, the per-access cost scales linearly.  In
practice, concurrency tests rarely exceed ~10 threads, making the overhead
negligible compared to Python interpretation costs.


Further reading
---------------

- `Vector clock (Wikipedia) <https://en.wikipedia.org/wiki/Vector_clock>`_
- `Happened-before (Wikipedia)
  <https://en.wikipedia.org/wiki/Happened-before>`_
- Abdulla et al., `"Source Sets: A Foundation for Optimal Dynamic Partial
  Order Reduction" <https://doi.org/10.1145/3073408>`_, JACM 2017 ---
  Def 3.2 (happens-before, p.12--13), Section 10 (implementation with
  vector clocks, p.34--35).
- See :doc:`dpor` for the full DPOR algorithm, :doc:`search` for search
  strategies, and :doc:`internals` for how the engine integrates with
  Python tracing.
