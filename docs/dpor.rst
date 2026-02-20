DPOR: Dynamic Partial Order Reduction
======================================

Frontrun includes a built-in `DPOR
<https://en.wikipedia.org/wiki/Partial_order_reduction>`_ engine for
*systematic* concurrency testing. Where the bytecode explorer samples random
interleavings hoping to hit a bug, DPOR guarantees that every meaningfully
different interleaving is explored exactly once --- and nothing redundant is
ever re-run.

The engine is written in Rust for performance and exposed to Python via PyO3.
Its design is inspired by the `loom <https://github.com/tokio-rs/loom>`_
library for Rust, which pioneered the idea of embedding a DPOR-based model
checker directly into a language's test harness.


Why DPOR?
---------

Consider two threads, each executing *n* shared-memory operations. The total
number of possible interleavings is:

.. math::

   \binom{2n}{n} = \frac{(2n)!}{(n!)^2}

For *n* = 3 that is 20 interleavings; for *n* = 10 it is 184,756. The number
grows exponentially --- yet most of these interleavings produce the same
observable outcome. If the threads access disjoint variables, *every* ordering
is equivalent and only one execution is needed.

DPOR exploits this insight. It tracks which operations actually *conflict*
(access the same object with at least one write) and only explores alternative
orderings at those conflict points. For programs with mostly thread-local work
this collapses an exponential search space down to a handful of executions.


Algorithm overview
------------------

Frontrun implements the classic DPOR algorithm from `Flanagan and Godefroid
(POPL 2005) <https://dl.acm.org/doi/10.1145/1040305.1040315>`_ with optional
`preemption bounding
<https://www.microsoft.com/en-us/research/publication/iterative-context-bounding-for-systematic-testing-of-multithreaded-programs/>`_.
The algorithm works in three phases that repeat until no unexplored
interleavings remain:

1. **Execute** the program under a deterministic schedule, recording every
   shared-memory access and synchronization event.
2. **Detect dependencies** --- pairs of concurrent accesses to the same object
   where at least one is a write. For each dependency, insert a *backtrack
   point* in the exploration tree so that the alternative ordering will be tried
   in a future execution.
3. **Advance** to the next unexplored path by backtracking through the
   exploration tree in depth-first order.


Key concepts
~~~~~~~~~~~~

Before diving into the details, here are the core ideas in abstract form.

A **trace** is a sequence of operations (events) performed by all threads
during one execution:

.. math::

   \sigma = e_1, e_2, \ldots, e_k

Two operations :math:`e_i` and :math:`e_j` are **dependent** if they access
the same shared object and at least one is a write. Operations that are
independent can be freely reordered without changing the outcome.

Two operations are **co-enabled** if both threads could have been scheduled at
that point. The DPOR algorithm identifies pairs of operations that are both
dependent *and* co-enabled --- these are the only points where exploring a
different scheduling order could produce a different result.

For example, consider a trace of four events on two threads accessing objects
*x* and *y*:

.. math::

   \sigma = \underbrace{W_0(x)}_{e_1},\;
            \underbrace{R_0(y)}_{e_2},\;
            \underbrace{W_1(x)}_{e_3},\;
            \underbrace{R_1(y)}_{e_4}

Here :math:`W_t(o)` means thread *t* writes object *o* and :math:`R_t(o)`
means thread *t* reads object *o*. Events :math:`e_1` and :math:`e_3` are
dependent (both access *x*, one is a write). Events :math:`e_2` and
:math:`e_4` are independent (both are reads of *y*). DPOR would insert a
backtrack point to try scheduling thread 1 before :math:`e_1`, producing a
trace where :math:`W_1(x)` precedes :math:`W_0(x)`.


.. _dpor-happens-before:

Happens-before and vector clocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two operations are *concurrent* if neither `happens before
<https://en.wikipedia.org/wiki/Happened-before>`_ the other. The happens-before
relation :math:`\to` is the smallest partial order satisfying:

.. math::

   \text{(program order)} \quad & e_i \to e_j
     \text{ if } e_i \text{ and } e_j \text{ are by the same thread and }
     i < j \\
   \text{(synchronization)} \quad & \text{release}(l) \to \text{acquire}(l)
     \text{ for lock } l \\
   \text{(spawn)} \quad & \text{spawn}(t) \to e
     \text{ for any first event } e \text{ of thread } t \\
   \text{(join)} \quad & e \to \text{join}(t)
     \text{ for any last event } e \text{ of thread } t \\
   \text{(transitivity)} \quad & e_i \to e_k
     \text{ if } e_i \to e_j \text{ and } e_j \to e_k

Two events :math:`e_i` and :math:`e_j` are **concurrent** (written
:math:`e_i \| e_j`) when :math:`e_i \not\to e_j` and :math:`e_j \not\to e_i`.

Frontrun tracks happens-before using `vector clocks
<https://en.wikipedia.org/wiki/Vector_clock>`_ (``VersionVec``). A vector clock
is an array of counters with one entry per thread (stored as a contiguous
``Vec<u32>`` indexed by thread ID):

.. math::

   V : \text{ThreadId} \to \mathbb{N}

The key operations are:

``increment(thread_id)``
    Advance the local component: :math:`V[t] \leftarrow V[t] + 1`. Called each
    time a thread is scheduled.

``join(other)``
    Point-wise maximum: :math:`V[i] \leftarrow \max(V[i], V'[i])` for all *i*.
    Used when synchronization transfers causal knowledge between threads.

``partial_le(other)``
    Returns true if :math:`V[i] \le V'[i]` for every component. If
    ``a.partial_le(b)`` then *a* happens before *b*:

    .. math::

       V \le V' \iff \forall\, i : V[i] \le V'[i]

``concurrent_with(other)``
    True when neither clock dominates the other:

    .. math::

       V \| V' \iff V \not\le V' \;\wedge\; V' \not\le V

Each thread carries *two* vector clocks:

``causality``
    Tracks the program's semantic happens-before relation. Updated when
    synchronization primitives (locks, joins, spawns) transfer ordering
    information between threads.

``dpor_vv``
    Tracks the scheduler's branch decisions. Incremented each time the thread
    is scheduled, and merged (via the same component-wise-max ``join``
    operation) on synchronization events just like ``causality``. This is the
    clock used for DPOR dependency detection --- it tells us whether two
    scheduling decisions were causally ordered or concurrent.


Conflict detection
~~~~~~~~~~~~~~~~~~~

Every shared-memory access is reported to the engine with a thread ID, an
object ID, and a kind (read or write). The engine maintains an ``ObjectState``
for each object, recording the last access of any kind and the last write
access separately.

When a new access arrives the engine asks: *what was the last dependent access
to this object?*  The dependency (or **conflict**) relation is:

.. math::

   \text{dep}(e_i, e_j) \iff
     \text{obj}(e_i) = \text{obj}(e_j) \;\wedge\;
     \bigl(\text{kind}(e_i) = W \;\lor\; \text{kind}(e_j) = W\bigr)

In the implementation:

- A **read** depends on the last **write** (two reads never conflict).
- A **write** depends on the last access of **any** kind.

If such a prior access :math:`e_p` exists and its ``dpor_vv`` is *not*
``partial_le`` of the current thread's ``dpor_vv``, the two accesses are
concurrent and could race:

.. math::

   \text{dep}(e_p, e_c) \;\wedge\; e_p \| e_c
   \implies \text{insert backtrack at branch}(e_p)

The engine inserts a backtrack point at the exploration-tree branch where the
prior access was made, marking the current thread for exploration there. This
ensures a future execution will try scheduling the current thread at that
earlier point, reversing the order of the two conflicting operations.


Synchronization events
~~~~~~~~~~~~~~~~~~~~~~~

Synchronization primitives update the ``causality`` (and ``dpor_vv``) clocks so
that accesses ordered by proper synchronization are not flagged as conflicts:

**Lock acquire**
    The acquiring thread joins the vector clock that was stored when the lock
    was last released:

    .. math::

       V_t \leftarrow V_t \sqcup V_{\text{lock}}

    This establishes that the acquire happens after the previous release.

**Lock release**
    The releasing thread's current causality clock is stored on the lock for
    future acquirers:

    .. math::

       V_{\text{lock}} \leftarrow V_t

**Thread join**
    The joining thread joins both the causality and DPOR clocks of the joined
    thread. All of the joined thread's operations now happen before the
    joiner's subsequent operations:

    .. math::

       V_t \leftarrow V_t \sqcup V_{t'}

**Thread spawn**
    The child thread inherits the parent's causality and DPOR clocks. The
    parent's operations before the spawn happen before the child's operations:

    .. math::

       V_{\text{child}} \leftarrow V_{\text{child}} \sqcup V_{\text{parent}}


The exploration tree
--------------------

The engine maintains a ``Path`` --- a sequence of ``Branch`` nodes, one per
scheduling decision. Each branch records:

- The **status** of every thread at that point (disabled, pending, active,
  backtrack, visited, blocked, or yielded).
- Which thread was **chosen** (the ``active_thread``).
- The cumulative **preemption count** (how many times a runnable thread was
  preempted in favor of a different thread up to this point).

To illustrate, consider a two-thread program where DPOR finds one conflict.
The exploration tree looks like:

.. code-block:: text

   Branch 0              Branch 1            Branch 2            Branch 3
   ┌─────────────┐       ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
   │ T0: Active  │  -->  │ T0: Active  │ --> │ T1: Active  │ --> │ T1: Active   │
   │ T1: Pending │       │ T1: Pending │     │ T0: Pending │     │ T0: Disabled │ 
   └─────────────┘       └─────────────┘     └─────────────┘     └──────────────┘
                            ^
                            │
                       CONFLICT DETECTED between
                       T0's write here and T1's
                       later write: mark T1 as
                       Backtrack at Branch 1

After the first execution completes, ``step()`` walks backward, finds the
backtrack at Branch 1, and schedules T1 there instead:

.. code-block:: text

   Branch 0              Branch 1 (replayed, different choice)
   ┌─────────────┐       ┌─────────────┐
   │ T0: Active  │  -->  │ T1: Active  │ --> ...
   │ T1: Pending │       │ T0: Visited │
   └─────────────┘       └─────────────┘

The prefix up to Branch 0 is replayed identically; only the decision at
Branch 1 changes.

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
preemptions (context switches away from a runnable thread). Empirical research
has shown that most concurrency bugs surface with very few preemptions ---
`Musuvathi and Qadeer (2007)
<https://www.microsoft.com/en-us/research/publication/iterative-context-bounding-for-systematic-testing-of-multithreaded-programs/>`_
found that a bound of 2 is sufficient to catch the vast majority of bugs in
practice.

When a backtrack point would create a preemption that exceeds the bound, the
engine falls back to ``add_conservative_backtrack``: it walks backward through
earlier branches looking for a point where the same thread can be explored
without exceeding the preemption budget. This maintains soundness within the
bounded exploration --- every execution with at most *k* preemptions that
differs in a dependent operation will still be explored.

With preemption bounding, the number of explored executions is polynomial in
the program length for a fixed bound *k*:

.. math::

   O\!\left(\binom{n}{k}\right)

where *n* is the number of scheduling points, versus the exponential
:math:`O(t^n)` for *t* threads without bounding.


Abstract reduction example
---------------------------

To see how DPOR prunes the search space, consider three events by two threads:

.. math::

   e_1 = W_0(x), \quad e_2 = R_1(y), \quad e_3 = W_1(x)

Events :math:`e_1` and :math:`e_3` are **dependent** (both access *x*, at
least one is a write). Events :math:`e_1` and :math:`e_2` are **independent**
(different objects). Events :math:`e_2` and :math:`e_3` are **independent**
(same thread --- program-ordered).

A naive scheduler would explore all interleavings:

.. math::

   &(1)\; e_1, e_2, e_3 \qquad
    (2)\; e_1, e_3, e_2 \qquad
    (3)\; e_2, e_1, e_3 \\
   &(4)\; e_2, e_3, e_1 \qquad
    (5)\; e_3, e_1, e_2 \qquad
    (6)\; e_3, e_2, e_1

But interleavings that only swap independent events produce the same result.
For example, (1) :math:`e_1, e_2, e_3` and (3) :math:`e_2, e_1, e_3` differ
only in the order of :math:`e_1` and :math:`e_2`, which are independent --- so
they are equivalent. The equivalence classes (called `Mazurkiewicz traces
<https://en.wikipedia.org/wiki/Trace_theory>`_) are:

.. math::

   \{(1), (3)\} \quad \text{and} \quad \{(4), (5), (6), (2)\}

DPOR explores **one representative per class** --- here just 2 executions
instead of 6. The reduction is even more dramatic with more threads and more
thread-local work.


Worked example: concurrent counter
------------------------------------

Consider the classic lost-update bug: two threads each read a shared counter,
increment locally, and write back.

.. code-block:: python

   counter = 0

   def thread_0():
       local = counter    # R_0(counter)
       counter = local+1  # W_0(counter)

   def thread_1():
       local = counter    # R_1(counter)
       counter = local+1  # W_1(counter)

There are four shared-memory operations, giving
:math:`\binom{4}{2} = 6` possible interleavings in total. Let's trace
how DPOR explores them.

Execution 1: T0 runs to completion, then T1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The engine's default scheduling preference is to keep running the current
thread, producing:

.. code-block:: text

   Step   Thread   Operation        Object    Kind
   ──────────────────────────────────────────────────
    0      T0      R_0(counter)     counter   read
    1      T0      W_0(counter)     counter   write
    2      T1      R_1(counter)     counter   read
    3      T1      W_1(counter)     counter   write

   Result: counter = 2  (correct)

Vector clocks evolve as follows (showing ``dpor_vv`` as ``[T0, T1]``):

.. code-block:: text

   Step 0:  T0 scheduled  →  T0.dpor_vv = [1, 0]
            R_0(counter): no prior access → no conflict

   Step 1:  T0 scheduled  →  T0.dpor_vv = [2, 0]
            W_0(counter): last dependent = R_0 at step 0 with vv [1, 0]
            [1, 0] ≤ [2, 0]?  YES → happens-before, no backtrack

   Step 2:  T1 scheduled  →  T1.dpor_vv = [0, 1]
            R_1(counter): last dependent = W_0 at step 1 with vv [2, 0]
            [2, 0] ≤ [0, 1]?  NO → CONCURRENT!
            ⟹ backtrack: mark T1 for exploration at branch 1

   Step 3:  T1 scheduled  →  T1.dpor_vv = [0, 2]
            W_1(counter): last dependent = W_0 at step 1 with vv [2, 0]
            [2, 0] ≤ [0, 2]?  NO → concurrent, but T1 already marked

One backtrack point was inserted at branch 1 (step 1), marking T1 for
exploration there.

Execution 2: T0 does one step, then T1 runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The engine replays the prefix through branch 0 (T0 at step 0), then at
branch 1 schedules T1 instead of T0:

.. code-block:: text

   Step   Thread   Operation        Object    Kind
   ──────────────────────────────────────────────────
    0      T0      R_0(counter)     counter   read     (replayed)
    1      T1      R_1(counter)     counter   read     ← different choice
    2      T0      W_0(counter)     counter   write
    3      T1      W_1(counter)     counter   write

   Result: counter = 1  (BUG! lost update)

Both threads read ``counter = 0``, then both write ``1``. The engine has
found the bug in just 2 executions out of the 6 possible interleavings.

Why not more executions?
~~~~~~~~~~~~~~~~~~~~~~~~~

DPOR did not need to explore orderings like T1-first-then-T0 because they
would be reached by the same backtrack mechanism from the other direction. The
key insight is that only the *relative order of dependent operations* matters.
Swapping two independent events (like :math:`R_0(x)` and :math:`R_1(y)`)
produces an equivalent trace, so DPOR skips it.


Reduction in practice
~~~~~~~~~~~~~~~~~~~~~~

The following table shows how DPOR reduces the exploration space as programs
grow:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 20

   * - Scenario
     - Naive
     - DPOR
     - Reduction
   * - 2 threads, 2 ops each, same object
     - 6
     - 2
     - 3x
   * - 2 threads, 5 ops each, same object
     - 252
     - 2
     - 126x
   * - 2 threads, 2 ops each, disjoint objects
     - 6
     - 1
     - 6x
   * - 3 threads, 2 ops each, one shared object
     - 90
     - 6
     - 15x

For threads accessing disjoint objects the reduction is total (one execution
regardless of operation count). When threads share objects, the explored set
corresponds to the `Mazurkiewicz traces
<https://en.wikipedia.org/wiki/Trace_theory>`_ --- equivalence classes of
interleavings under independent-operation reordering.


Data structures
---------------

The implementation is split across six Rust modules in ``frontrun-dpor/src/``:

``vv.rs`` --- Vector clocks
    ``VersionVec``: a contiguous ``Vec<u32>`` indexed by thread ID with
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


Complexity
----------

**Per access:** :math:`O(T)` where :math:`T` is the number of threads,
dominated by the vector-clock comparison.

**Space:** :math:`O(D \cdot T + O)` where :math:`D` is the exploration tree
depth and :math:`O` is the number of unique shared objects. Only the last two
accesses per object are retained.

**Executions:** In the worst case exponential in the number of dependent
operations, but in practice DPOR prunes the vast majority of redundant
interleavings. With preemption bounding at bound :math:`k`, the explored
subset is :math:`O\!\left(\binom{n}{k}\right)` which is polynomial in program length *n*
for fixed *k*.


Further reading
---------------

- Cormac Flanagan and Patrice Godefroid, `"Dynamic Partial-Order Reduction for
  Model Checking Software"
  <https://dl.acm.org/doi/10.1145/1040305.1040315>`_, POPL 2005 --- the
  original DPOR paper.
- Madanlal Musuvathi and Shaz Qadeer, `"Iterative Context Bounding for
  Systematic Testing of Multithreaded Programs"
  <https://www.microsoft.com/en-us/research/publication/iterative-context-bounding-for-systematic-testing-of-multithreaded-programs/>`_,
  PLDI 2007 --- preemption bounding.
- `Partial order reduction (Wikipedia)
  <https://en.wikipedia.org/wiki/Partial_order_reduction>`_
- `Vector clock (Wikipedia)
  <https://en.wikipedia.org/wiki/Vector_clock>`_
- `Mazurkiewicz trace theory (Wikipedia)
  <https://en.wikipedia.org/wiki/Trace_theory>`_
- `loom <https://github.com/tokio-rs/loom>`_ --- Rust concurrency testing
  library that inspired this implementation.
