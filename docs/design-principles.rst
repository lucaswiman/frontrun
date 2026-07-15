Design Principles
=================

Frontrun is a **white-box** concurrency tester: it runs *your* code and
controls its scheduling from the inside.  The deliverable is a
**constructive proof** --- a deterministic, replayable, DPOR-minimized
schedule that says "run these operations in this exact order and the
invariant fails."  Everything below follows from taking that deliverable
seriously.  Anything that weakens *determinism*, *replayability*, or
*exactness of the counterexample* is working against the core, no matter
how much convenience it buys.

These principles are binding on contributors (human and agent alike):
when a change conflicts with one of them, the change is wrong, not the
principle.


The deliverable is a constructive proof
---------------------------------------

A frontrun failure is not "the test got flaky" --- it is a specific
interleaving, expressed as a schedule, that deterministically violates
the invariant.  That is the entire value proposition: it distinguishes a
real logic bug from a timeout artifact, and it hands the developer the
canonical minimal interleaving that causes the bug.

Consequences:

* Exploration must be **deterministic**: the same inputs and the same
  schedule must produce the same execution.  Sources of nondeterminism
  (wall-clock time, unordered iteration, PRNGs) are either controlled or
  excluded from the scheduling-relevant path.
* Counterexamples must be **replayable**: a reported failing schedule is
  a claim that re-running it reproduces the failure.
* Counterexamples should be **minimal**: DPOR's partial-order reduction
  exists so the reported schedule contains only scheduling decisions
  that matter.


White-box, not black-box
------------------------

Frontrun instruments code from the inside --- workers import frontrun's
patching, which is what preserves *semantic access identity* (knowing
that two operations touch the same lock, row, or key) and
*step-completion* (knowing an operation finished before scheduling the
next).  Cross-process SQL/Redis exploration extends the same model to
contention on shared rows rather than shared memory; the workers are
still instrumented Python processes.

Scheduling *unmodified or non-Python* workers via wire-level
interception is explicitly out of scope.  That is a black-box,
Jepsen-shaped problem --- observing histories for consistency violations
rather than proving a specific interleaving --- and belongs in a
separate tool (which could reuse the Rust DPOR engine, but not
frontrun's product shape).  Don't grow frontrun toward it.


Fail closed: a pass is a certificate, not a default
---------------------------------------------------

``property_holds=True`` is a **positive claim**: "the explored
interleavings ran, the invariant was checked, and it held."  It must
never be the value a result takes by default, by omission, or by an
error path that forgot to report.

A pass certificate requires positive evidence:

* at least one interleaving actually executed (a run that explored zero
  interleavings is vacuous, and a vacuous pass is a frontrun bug, not a
  pass);
* the worker bodies actually executed --- a worker that silently never
  ran cannot certify anything about the code it was supposed to run;
* no coverage-degrading event occurred (tracer failures, suppressed
  exceptions, instrumentation gaps) without being reflected in the
  result.

When frontrun cannot tell whether exploration was sound, the honest
answer is *inconclusive* or a demoted claim --- never a clean pass.
False failures are annoying; false passes are the one thing a testing
tool must not produce.


Dependencies: over-approximate, never under-approximate
-------------------------------------------------------

DPOR is sound only if every real dependency between operations is
visible to the engine.  The governing rule for semantic access identity
(which rows a SQL statement touches, which keys a Redis command reads or
writes, which object a lock guards) is:

   **Over-merging is allowed; under-merging is forbidden.**

Treating two independent operations as conflicting (over-merging) costs
performance: the engine explores interleavings that were in fact
equivalent.  Treating two conflicting operations as independent
(under-merging) costs soundness: a real race is pruned from the search
space and frontrun reports a false pass.

Consequences:

* Identity extraction is **coarse by default**.  An unrecognized command,
  an unparseable statement, or an ambiguous scope is modeled as a
  conservative global write --- it conflicts with everything.
* **Narrowing requires positive evidence.**  An access is scoped down to
  a specific row, key, or table only when a command-table entry or a
  successful parse affirmatively says the operation touches nothing
  else.  "Probably only touches this key" is not evidence.
* The command tables themselves are tested for **scope completeness**:
  every command a client can send either has an explicit, audited scope
  entry or falls through to the coarse default.


Honest exhaustiveness
---------------------

``exhausted=True`` means "the search space was fully covered under the
stated model" --- it upgrades "no failure found" to "no failure exists
(at this granularity)."  That claim is only as good as its weakest
link, so any event that degrades coverage demotes it:

* hitting an iteration, depth, or time bound;
* divergence between the schedule the engine planned and the execution
  that physically occurred (e.g. a worker redirected by a row-lock
  conflict mid-step);
* any skipped, truncated, or abandoned branch.

A demoted result still reports what was explored --- "no failure found
within bounds" is useful and honest.  Claiming exhaustion after a
truncated search is a false proof.


Oracles over review
-------------------

Soundness is defended by **mechanisms and differential oracles**, not by
rounds of code review.  Review samples the bug population; an oracle
checks an invariant on every run.  The oracles frontrun maintains:

* **DPOR vs. exhaustive enumeration** --- on small generated programs,
  the DPOR engine must find exactly the set of outcomes that brute-force
  enumeration finds, and the number of explored interleavings must match
  the Mazurkiewicz trace count.
* **Virtual vs. real clock differential** --- a program certified as
  pass under the virtual clock must not fail under the real clock.
  (The reverse is fine: the virtual clock legitimately reaches states,
  like TTL expiry, that real time rarely hits.)
* **Mutation and fault-injection tests on the certification gate** ---
  each historically observed soundness bug (vacuous pass, over-claimed
  exhaustion, silently unexecuted worker) is resurrected as a mutation
  the test suite must kill.  A soundness fix without a test that fails
  on the old behavior is not done.

When a new class of soundness bug is found, the fix includes extending
an oracle or adding a mutation test --- otherwise the class will
reappear.


One owner per low-level facility
--------------------------------

Global interpreter facilities are owned by exactly one module, and
everyone else goes through its API.  ``sys.settrace`` /
``sys.monitoring`` / ``f_trace_opcodes`` belong to
``frontrun/_opcode_observer.py``; new exploration strategies register as
``Strategy`` / ``AsyncStrategy`` adapters in ``frontrun/_strategy.py``
rather than adding dispatch branches.  Single ownership is what keeps
version-specific choices (like 3.14t routing through ``sys.monitoring``)
in one place and keeps two subsystems from fighting over the same hook.
