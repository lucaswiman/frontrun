Search Strategies
=================

Frontrun's DPOR engine supports multiple **search strategies** that control
the order in which wakeup-tree branches are explored.  All strategies visit
the same set of `Mazurkiewicz trace
<https://en.wikipedia.org/wiki/Trace_theory>`_ equivalence classes ---
soundness and completeness are preserved.  Only the *exploration order*
differs, which affects how quickly bugs are found when ``stop_on_first=True``.

This page first reviews the wakeup-tree algorithm that underlies every
strategy, then describes each strategy in detail and discusses their
soundness and optimality trade-offs.


.. contents:: Contents
   :local:
   :depth: 2


The Optimal DPOR Algorithm
--------------------------

Frontrun implements a hybrid of Algorithms 1 and 2 from Abdulla et al.,
`"Source Sets: A Foundation for Optimal Dynamic Partial Order Reduction"
<https://doi.org/10.1145/3073408>`_ (JACM 2017).  The goal is to explore
exactly **one execution per Mazurkiewicz trace equivalence class** --- two
executions are equivalent when they differ only in the ordering of
independent (non-conflicting) operations.


Core data structures
~~~~~~~~~~~~~~~~~~~~

**Wakeup tree** (``WakeupTree``, Definition 6.1, JACM'17 p.21--22).
An ordered tree of thread-ID sequences.  Each root-to-leaf path represents
a sequence of scheduling decisions to explore.  Shared prefixes are merged
and duplicate sequences deduplicated on insertion.  At each branch in the
exploration tree, the wakeup tree records *which alternative interleavings
still need to be explored*.

Two properties make the tree optimal:

- **Property 1** (no sleep-set--blocked leaves): for each leaf *w*,
  :math:`\mathit{WI}[E](w) \cap P = \emptyset`.  No wakeup sequence leads
  to an execution that would be blocked by the sleep set.

- **Property 2** (sleep-set removal across siblings): for siblings
  :math:`u.p \prec u.w` where :math:`u.w` is a leaf,
  :math:`p \notin \mathit{WI}[E.u](w)`.  Exploring the earlier sibling
  :math:`u.p` adds *p* to the sleep set, but the later sibling :math:`u.w`
  removes *p* --- so both orderings are eventually covered.

Together these guarantee that each explored execution represents a
*distinct* Mazurkiewicz trace.

**Sleep set**.
At each scheduling point, threads whose remaining work is independent of
the active thread's work are *asleep*.  A sleeping thread will not be added
to the wakeup tree because an equivalent execution starting with it has
already been (or will be) explored.  Sleep sets are propagated forward
across scheduling points using access-kind independence checks:

.. math::

   \mathit{Sleep}' = \{q \in \mathit{sleep}(E) \mid E \vdash p \mathbin{\diamond} q\}

where :math:`p \mathbin{\diamond} q` means *p*'s and *q*'s accesses are
independent (JACM'17 Def 3.3, p.13).

**Branch**.
One node per scheduling decision, holding: per-thread status, the active
thread, a wakeup tree, a sleep set, the preemption count, and per-thread
access records for independence checks.


The exploration loop
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   repeat:
     1. EXECUTE under a deterministic schedule.
        At each shared-memory access, report it to the engine.

     2. DETECT RACES INLINE (Algorithm 1 style).
        When access e' by thread T conflicts with prior access e by
        another thread and e ∦ e' (concurrent, not happens-before
        ordered), immediately insert [T] into the wakeup tree at
        branch(e).

     3. COLLECT DEFERRED RACES.
        Also store each race as a PendingRace for post-execution
        processing.

     4. PROCESS DEFERRED RACES (Algorithm 2 style).
        For each race (e, e'):
          a. Compute notdep(e, E).e' — the threads between e and e'
             whose accesses are independent of e, followed by the
             racing thread.
          b. Insert the full notdep sequence into the wakeup tree at
             branch(e).  Multi-step sequences guide future exploration
             through independent intermediates.

     5. BACKTRACK (step()).
        Walk backward through branches:
          - Mark the active thread as Visited; add to sleep set.
          - Remove its branch from the wakeup tree.
          - Pick the next unexplored thread using the search strategy.
          - Extract the subtree for wakeup guidance at deeper levels.
          - If the wakeup tree is empty, pop the branch and continue.

     6. REPLAY PREFIX.
        The next execution replays the shared prefix identically,
        propagating sleep sets via independence checks, then diverges
        at the backtrack point.

   until no unexplored branches remain.


Notdep sequences
~~~~~~~~~~~~~~~~

The **notdep** sequence for a race between events *e* (at position *i*) and
*e'* (at position *j*, by thread *T*) is:

.. math::

   \mathit{notdep}(e, E) = \langle t_k \mid i < k < j,\;
   \mathit{thread}(e_k) \neq \mathit{thread}(e),\;
   \mathit{accesses}(e_k) \mathbin{\diamond} \mathit{accesses}(e) \rangle
   \cdot T

It contains the threads of events between *e* and *e'* that are
**independent** of *e* (different thread, no conflicting accesses),
followed by the racing thread *T*.  Inserting this multi-step sequence
into the wakeup tree tells the scheduler: "replay these independent
events first, *then* run *T* before *e*."


Soundness and optimality
~~~~~~~~~~~~~~~~~~~~~~~~~

**Soundness** (every distinct trace class is explored):

- Inline wakeup insertion ensures no race is ever lost --- even if the
  notdep computation is infeasible, the single-thread insertion
  ``[T]`` is always performed.
- Vector clocks (``dpor_vv``) precisely track happens-before via lock
  acquire/release, thread spawn/join, ensuring only genuinely concurrent
  accesses are flagged as races.
- Sleep sets only prune threads whose *remaining* work is independent
  (position-sensitive future cache prevents false omissions).

**Optimality** (at most one execution per trace class):

- Wakeup tree Properties 1 and 2 guarantee that no sleep-set--blocked
  exploration occurs and that every sleep-set entry is eventually woken.
- Notdep sequences provide precise multi-step guidance, reducing the
  branching factor compared to single-thread backtracking.
- *Caveat*: the insert operation uses exact prefix matching rather than
  the paper's equivalence check (:math:`v \sim_{[E]} w`, Lemma 6.2).
  This is sound but may retain redundant branches, causing minor
  over-exploration relative to the theoretical optimum.


Happens-before tracking
~~~~~~~~~~~~~~~~~~~~~~~~

The engine tracks happens-before using **vector clocks** --- one
``VersionVec`` per thread, incremented on each scheduling step and joined
on synchronization events (lock acquire/release, thread spawn/join).  Two
accesses race when they are to the same object, at least one is a write,
and their vector clocks are *concurrent* (neither dominates).

Each thread actually carries three clocks: ``dpor_vv`` (lock-aware, for
shared-memory races), ``io_vv`` (lock-oblivious, for TOCTOU/I/O races),
and ``causality`` (general causal ordering).  See :doc:`vector-clocks` for
the full explanation with worked examples.


Search Strategies
-----------------

All strategies operate on the **sorted** list of root-level thread IDs in
the wakeup tree and pick one to explore next.  The choice determines which
Mazurkiewicz trace classes are reached first.


DFS (depth-first search)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   strategy = "dfs"

Always picks the **minimum** thread ID among wakeup tree roots.
Deterministic, no seed parameter.  This is the default from Algorithm 2 of
the JACM'17 paper.

**Characteristics:**

- Fully deterministic and reproducible across runs.
- Explores traces in a predictable lexicographic order over thread IDs.
- Best sleep-set effectiveness: the DFS ordering matches the paper's
  :math:`\prec` relation exactly, so wakeup tree Properties 1 and 2 are
  maximally effective.
- Best for full-space exploration (``stop_on_first=False``) where minimizing
  total executions matters.


BitReversal
~~~~~~~~~~~

.. code-block:: python

   strategy = "bit_reversal"
   search_seed = 42

Uses a `van der Corput
<https://en.wikipedia.org/wiki/Van_der_Corput_sequence>`_-style
**bit-reversal permutation** over sibling positions.  Given *n* siblings,
the first two picks are maximally spread (distance *n*/2 apart), and
subsequent picks fill in the gaps --- a low-discrepancy sequence over the
wakeup tree.

.. math::

   \text{idx} = \text{bitrev}(\text{counter} + \text{seed}) \mod n

Different seeds rotate the permutation, producing different deterministic
orderings that each cover the full tree exactly once (for power-of-two *n*).

**Characteristics:**

- Maximally spreads exploration across distinct conflict points early.
- Best for ``stop_on_first=True`` when the bug could be at any depth ---
  diverse early coverage increases the chance of hitting it quickly.
- Deterministic given the same seed.


RoundRobin
~~~~~~~~~~

.. code-block:: python

   strategy = "round_robin"
   search_seed = 0

Cycles through available threads with a rotating offset:

.. math::

   \text{idx} = (\text{counter} + \text{seed}) \bmod n

Each consecutive pick at the same wakeup-tree level chooses a different
thread.

**Characteristics:**

- Simple, predictable diversity across threads.
- Avoids the "always-thread-0-first" bias of DFS.
- Good general-purpose choice for ``stop_on_first=True``.


Stride
~~~~~~

.. code-block:: python

   strategy = "stride"
   search_seed = 7

Picks every *s*-th sibling, where *s* is derived from the seed and chosen
**coprime** to the branching factor *n*:

.. math::

   s = \text{coprime\_stride}(\text{seed}, n) \qquad
   \text{idx} = (\text{counter} \cdot s) \bmod n

Because :math:`\gcd(s, n) = 1`, the sequence visits all *n* positions
exactly once before repeating.

**Characteristics:**

- Different seeds produce qualitatively different traversal patterns.
- Fuller coverage than RoundRobin for non-uniform branching factors.
- Deterministic given the same seed.


ConflictFirst
~~~~~~~~~~~~~

.. code-block:: python

   strategy = "conflict_first"

Always picks the **maximum** thread ID --- the reverse of DFS.

**Characteristics:**

- Prioritizes higher-numbered threads, which in many programs are the
  "later" threads involved in deeper wakeup sequences.
- Races reversed at shallower depths (by higher-priority threads) get
  explored first.
- A simple heuristic for bug-finding when conflicts cluster around
  higher thread IDs.


Comparison and Trade-offs
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 12 12 12 46

   * - Strategy
     - Deterministic
     - Seed
     - Sleep-set quality
     - Best for
   * - DFS
     - Yes
     - —
     - Optimal
     - Full exploration, reproducibility
   * - BitReversal
     - Yes
     - Yes
     - Reduced
     - ``stop_on_first``, diverse early coverage
   * - RoundRobin
     - Yes
     - Yes
     - Reduced
     - ``stop_on_first``, thread diversity
   * - Stride
     - Yes
     - Yes
     - Reduced
     - ``stop_on_first``, non-uniform branching
   * - ConflictFirst
     - Yes
     - —
     - Reduced
     - ``stop_on_first``, deep-conflict programs


Why DFS is optimal and the others are not
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wakeup tree optimality depends on the **sibling ordering** :math:`\prec`.
The DFS strategy uses ``min_thread()`` as a deterministic proxy for
:math:`\prec`, which matches the paper's Algorithm 2 default.  Under this
ordering, the two wakeup tree properties hold precisely:

1. No leaf leads to a sleep-set--blocked execution (Property 1).
2. For consecutive siblings :math:`p \prec q`, exploring *p* first adds it
   to the sleep set, and *q*'s exploration removes it (Property 2).

Non-DFS strategies reorder siblings, effectively using a different
:math:`\prec`.  This can violate Property 2: a sibling explored early under
BitReversal might not be "removed" by the sibling explored next, because
the second sibling's wakeup sequence was designed (by the wakeup tree
insertion algorithm) for the DFS ordering.  The consequence is:

- **No races are missed** --- soundness is preserved because races are
  detected during execution and inserted into the wakeup tree regardless of
  exploration order.
- **Some redundant traces may be explored** --- the sleep set may fail to
  prune equivalent interleavings that DFS would have caught, because the
  non-DFS ordering breaks the sleep-set removal guarantee.

In practice, the overhead is small.  The non-DFS strategies trade a modest
increase in total executions for significantly faster time-to-first-bug
when used with ``stop_on_first=True``.


API usage
---------

The search strategy is set via the ``search_strategy`` and ``search_seed``
parameters to ``frontrun.explore()``:

.. code-block:: python

   import frontrun

   # Default: DFS
   results = frontrun.explore(fn, workers=[t0, t1])

   # BitReversal with seed
   results = frontrun.explore(
       fn, workers=[t0, t1],
       search_strategy="bit_reversal",
       search_seed=42,
   )

   # Stop on first bug with RoundRobin
   results = frontrun.explore(
       fn, workers=[t0, t1],
       search_strategy="round_robin",
       search_seed=0,
       stop_on_first=True,
   )


Further reading
---------------

- Abdulla et al., `"Source Sets: A Foundation for Optimal Dynamic Partial
  Order Reduction" <https://doi.org/10.1145/3073408>`_, JACM 2017 ---
  Algorithms 1--2, wakeup trees (Definition 6.1), notdep sequences, sleep
  sets.
- Flanagan and Godefroid, `"Dynamic Partial-Order Reduction for Model
  Checking Software"
  <https://dl.acm.org/doi/10.1145/1040305.1040315>`_, POPL 2005 ---
  the original DPOR algorithm.
- Musuvathi and Qadeer, `"Iterative Context Bounding for Systematic Testing
  of Multithreaded Programs"
  <https://www.microsoft.com/en-us/research/publication/iterative-context-bounding-for-systematic-testing-of-multithreaded-programs/>`_,
  PLDI 2007 --- preemption bounding.
- See :doc:`dpor` for a comprehensive overview of the DPOR engine, vector
  clocks, conflict detection, and data structures.
