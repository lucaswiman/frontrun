DPOR: Dynamic Partial Order Reduction
======================================

Frontrun includes a built-in DPOR engine for *systematic* concurrency testing.
Where the bytecode explorer samples random interleavings hoping to hit a bug,
DPOR guarantees that every meaningfully different interleaving is explored
exactly once --- and nothing redundant is ever re-run.

The engine is written in Rust for performance and exposed to Python via PyO3.


Why DPOR?
---------

Consider two threads, each executing three shared-memory operations. There are
``6! / (3! * 3!) = 20`` possible orderings --- but most of them produce the same
observable outcome. If the threads access disjoint variables, *every* ordering
is equivalent and only one execution is needed.

DPOR exploits this insight. It tracks which operations actually *conflict*
(access the same object with at least one write) and only explores alternative
orderings at those conflict points. For programs with mostly thread-local work
this collapses an exponential search space down to a handful of executions.


Algorithm overview
------------------

Frontrun implements the classic DPOR algorithm from Flanagan and Godefroid
(POPL 2005) with optional preemption bounding. The algorithm works in three
phases that repeat until no unexplored interleavings remain:

1. **Execute** the program under a deterministic schedule, recording every
   shared-memory access and synchronization event.
2. **Detect dependencies** --- pairs of concurrent accesses to the same object
   where at least one is a write. For each dependency, insert a *backtrack
   point* in the exploration tree so that the alternative ordering will be tried
   in a future execution.
3. **Advance** to the next unexplored path by backtracking through the
   exploration tree in depth-first order.


Happens-before and vector clocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two operations are *concurrent* if neither happens before the other.
Happens-before is the standard partial order induced by program order within a
thread and by synchronization across threads (lock acquire/release, thread
spawn/join).

Frontrun tracks happens-before using **vector clocks** (``VersionVec``). A
vector clock is a dense array of counters, one per thread. The key operations
are:

``increment(thread_id)``
    Advance the local component. Called each time a thread is scheduled.

``join(other)``
    Point-wise maximum: ``self[i] = max(self[i], other[i])``. Used when
    synchronization transfers causal knowledge between threads.

``partial_le(other)``
    Returns true if ``self[i] <= other[i]`` for every component. If
    ``a.partial_le(b)`` then *a* happens before *b*.

``concurrent_with(other)``
    True when neither clock dominates the other --- the operations could have
    executed in either order.

Each thread carries *two* vector clocks:

``causality``
    Tracks the program's semantic happens-before relation. Updated when
    synchronization primitives (locks, joins, spawns) transfer ordering
    information between threads.

``dpor_vv``
    Tracks the scheduler's branch decisions. Incremented each time the thread
    is scheduled and joined on synchronization the same way as ``causality``.
    This is the clock used for DPOR dependency detection --- it tells us
    whether two scheduling decisions were causally ordered or concurrent.


Conflict detection
~~~~~~~~~~~~~~~~~~~

Every shared-memory access is reported to the engine with a thread ID, an
object ID, and a kind (read or write). The engine maintains an ``ObjectState``
for each object, recording the last access of any kind and the last write
access separately.

When a new access arrives the engine asks: *what was the last dependent access
to this object?*

- A **read** depends on the last **write** (two reads never conflict).
- A **write** depends on the last access of **any** kind.

If such a prior access exists and its ``dpor_vv`` is *not*
``partial_le`` of the current thread's ``dpor_vv``, the two accesses are
concurrent and could race. The engine inserts a backtrack point at the
exploration-tree branch where the prior access was made, marking the current
thread for exploration there. This ensures a future execution will try
scheduling the current thread at that earlier point, reversing the order of the
two conflicting operations.


Synchronization events
~~~~~~~~~~~~~~~~~~~~~~~

Synchronization primitives update the ``causality`` (and ``dpor_vv``) clocks so
that accesses ordered by proper synchronization are not flagged as conflicts:

**Lock acquire**
    The acquiring thread joins the vector clock that was stored when the lock
    was last released. This establishes that the acquire happens after the
    previous release.

**Lock release**
    The releasing thread's current causality clock is stored on the lock for
    future acquirers.

**Thread join**
    The joining thread joins both the causality and DPOR clocks of the joined
    thread. All of the joined thread's operations now happen before the
    joiner's subsequent operations.

**Thread spawn**
    The child thread inherits the parent's causality and DPOR clocks. The
    parent's operations before the spawn happen before the child's operations.


The exploration tree
--------------------

The engine maintains a ``Path`` --- a sequence of ``Branch`` nodes, one per
scheduling decision. Each branch records:

- The **status** of every thread at that point (disabled, pending, active,
  backtrack, visited, blocked, or yielded).
- Which thread was **chosen** (the ``active_thread``).
- The cumulative **preemption count** (how many times a runnable thread was
  preempted in favor of a different thread up to this point).

Scheduling
~~~~~~~~~~~

When the engine needs to pick the next thread:

1. If we are **replaying** a previously recorded path (``pos < branches.len``),
   return the same choice as before. This is how the engine deterministically
   re-executes the shared prefix leading to a backtrack point.

2. Otherwise this is a **new** scheduling decision. The engine prefers the
   currently active thread (to minimize preemptions) and creates a new
   ``Branch`` recording the decision and the status of all threads.

Backtracking
~~~~~~~~~~~~~

When the engine calls ``backtrack(path_id, thread_id)`` it marks ``thread_id``
for future exploration at branch ``path_id``. The thread's status at that
branch changes from ``Pending`` to ``Backtrack``.

When the current execution finishes, ``step()`` walks backward through the
branch list:

1. Mark the current branch's active thread as ``Visited``.
2. Look for any thread marked ``Backtrack`` in this branch.
3. If found, promote it to ``Active``, set it as the branch's new choice, and
   reset the replay position. The next execution will replay up to this branch
   and then diverge.
4. If no backtrack thread is found, pop the branch and continue walking
   backward.
5. If all branches are exhausted, exploration is complete.

This is a standard depth-first search over the tree of scheduling choices,
pruned by DPOR so that only branches with genuine conflicts are explored.


Preemption bounding
--------------------

Real programs often have far more conflicts than can feasibly be explored.
*Preemption bounding* limits exploration to executions with at most *k*
preemptions (context switches away from a runnable thread). Since most
concurrency bugs surface with 1--2 preemptions, a small bound drastically
cuts the search space while still catching the vast majority of bugs.

When a backtrack point would create a preemption that exceeds the bound, the
engine falls back to ``add_conservative_backtrack``: it walks backward through
earlier branches looking for a point where the same thread can be explored
without exceeding the preemption budget. This maintains soundness within the
bounded exploration --- every execution with at most *k* preemptions that
differs in a dependent operation will still be explored.


Data structures
---------------

The implementation is split across six Rust modules in ``frontrun-dpor/src/``:

``vv.rs`` --- Vector clocks
    ``VersionVec``: a dense ``Vec<u32>`` indexed by thread ID with
    ``increment``, ``join``, ``partial_le``, and ``concurrent_with`` operations.

``access.rs`` --- Access records
    ``AccessKind`` (``Read`` | ``Write``) and ``Access``, which stores the
    ``path_id`` (branch index where the access occurred), the thread's
    ``dpor_vv`` at that moment, and the ``thread_id``.

``object.rs`` --- Shared object state
    ``ObjectState`` tracks the last access and last write access to each object.
    ``last_dependent_access(kind)`` returns the relevant prior access for
    conflict detection.

``thread.rs`` --- Thread state
    ``Thread`` holds the two vector clocks (``causality`` and ``dpor_vv``) and
    the ``finished``/``blocked`` flags. ``ThreadStatus`` is the per-branch
    status enum used by the exploration tree.

``path.rs`` --- Exploration tree
    ``Branch`` and ``Path``. ``Path`` drives scheduling, backtracking, and
    depth-first advancement through the exploration tree.

``engine.rs`` --- Orchestration
    ``DporEngine`` ties everything together. ``Execution`` holds per-run state
    (threads, objects, lock release clocks, schedule trace). The engine
    processes accesses and syncs, inserts backtrack points, and advances to the
    next execution.


Python API
----------

The Rust engine is exposed to Python via PyO3 as the ``frontrun_dpor`` native
module. The two Python-visible classes are ``PyDporEngine`` and
``PyExecution``.

.. code-block:: python

   from frontrun_dpor import PyDporEngine, PyExecution

   engine = PyDporEngine(
       num_threads=2,
       preemption_bound=2,       # optional; None = unbounded
       max_branches=100_000,     # safety limit per execution
       max_executions=None,      # optional cap on total executions
   )

   while True:
       execution = engine.begin_execution()

       while True:
           thread_id = engine.schedule(execution)
           if thread_id is None:
               break  # deadlock or branch limit

           # ... run thread_id until it performs a shared access ...

           engine.report_access(execution, thread_id, object_id, "write")
           # or: engine.report_sync(execution, thread_id, "lock_acquire", lock_id)

           execution.finish_thread(thread_id)

       # check invariants on this execution's final state ...

       if not engine.next_execution():
           break  # all interleavings explored

``report_access(execution, thread_id, object_id, kind)``
    Report a shared-memory access. ``kind`` is ``"read"`` or ``"write"``.
    ``object_id`` is an opaque ``u64`` that uniquely identifies the shared
    object (e.g., ``id(obj)``).

``report_sync(execution, thread_id, event_type, sync_id)``
    Report a synchronization event. ``event_type`` is one of
    ``"lock_acquire"``, ``"lock_release"``, ``"thread_join"``,
    ``"thread_spawn"``. ``sync_id`` identifies the lock or thread.

``next_execution()``
    Advance to the next unexplored path. Returns ``False`` when exploration
    is complete.

``execution.finish_thread(thread_id)``
    Mark a thread as finished (no more operations).

``execution.block_thread(thread_id)`` / ``execution.unblock_thread(thread_id)``
    Mark a thread as blocked or unblocked (e.g., waiting on a lock).

Properties: ``engine.executions_completed``, ``engine.tree_depth``,
``engine.num_threads``, ``execution.schedule_trace``, ``execution.aborted``.


Worked example: lost update
----------------------------

Consider the classic lost-update bug: two threads each read a shared counter,
increment locally, and write back.

.. code-block:: text

   Thread 0            Thread 1
   --------            --------
   local = counter     local = counter
   counter = local+1   counter = local+1

With DPOR exploration:

**Execution 1** --- Thread 0 runs first, then Thread 1:

.. code-block:: text

   T0: read counter    (object 0, read)
   T0: write counter   (object 0, write)
   T1: read counter    (object 0, read)    <- depends on T0's write, but ordered
   T1: write counter   (object 0, write)   <- depends on T0's write, but ordered

All accesses are ordered (T0 finishes before T1 starts). Final counter = 2.
No backtrack points needed for this execution --- but the engine detects that
T1's read of the counter conflicts with T0's write, and the two are concurrent
in the *scheduling* sense (T1 could have been scheduled before T0 finished).
A backtrack point is inserted.

**Execution 2** --- Thread 1's read happens before Thread 0's write:

.. code-block:: text

   T0: read counter
   T1: read counter    <- both read 0
   T0: write counter   <- writes 1
   T1: write counter   <- writes 1 (lost update!)

Final counter = 1. The bug is found.

DPOR only needed 2 executions. A naive exploration of all ``4! / (2! * 2!) = 6``
orderings would have run 6.


Complexity
----------

**Per access:** O(*T*) where *T* is the number of threads, dominated by the
vector-clock comparison.

**Space:** O(*D* x *T* + *O*) where *D* is the exploration tree depth and *O*
is the number of unique shared objects. Only the last two accesses per object
are retained.

**Executions:** In the worst case exponential in the number of dependent
operations, but in practice DPOR prunes the vast majority of redundant
interleavings. With preemption bounding, the explored subset is polynomial in
the program length for a fixed bound *k*.
