"""Defect #22: access-anchor labels collide across instances of one class.

Access-anchored replay (defect #20) records the failing execution's accesses
to the racing objects as ``TypeName.attr`` labels and gates the replay run's
accesses on that recorded order.  The label carries no instance identity, so
accesses to a *different* instance of the same class are indistinguishable
from accesses to the racing object.

When such a sibling instance is touched a run-varying number of times — a
per-thread status/result object polled in a timing-dependent loop, the exact
drift shape defect #20 exists to survive — its accesses contaminate the
anchor stream: a replay run with a different sibling-access count silently
lets the *racing* read or write consume an anchor recorded for a *sibling*
access (or vice versa), releasing the racing access at the wrong point.  The
recorded counterexample then reproduces only by luck of the positional drift
(observed 35-37/80 across batches, worst 0/10), with no gate timeout and no
degrade — the misattribution is silent.  Renaming the sibling's class (no
collision) reproduces 10/10, so the label collision is the sole cause.

The fix anchors setup-reachable objects by their pre-registered stable ID
(``TypeName.attr#<id>``): replay pre-registers the fresh ``setup()`` graph in
the same deterministic order as exploration, so the racing instance and its
siblings get distinct, run-stable labels.  Mid-run objects keep the plain
``TypeName.attr`` best-effort label.
"""

from __future__ import annotations

import random

import frontrun


class Holder:
    """Carrier for the racing attribute; also the sibling instances' class."""

    def __init__(self) -> None:
        self.cb: int | None = None


class State:
    def __init__(self) -> None:
        self.shared = Holder()  # the RACING instance
        self.mine = [Holder(), Holder()]  # per-thread siblings: never raced
        self.results: list[int | None] = []


def _variable_work() -> int:
    # Run-varying traced-opcode count so every replay run drifts positionally
    # (same modeling as defect #20's regression).
    acc = 0
    for _ in range(random.randint(50, 800)):
        acc += 1
    return acc


def _make_worker(tid: int):
    def worker(s: State) -> None:
        s.shared.cb = tid  # racing WRITE
        _variable_work()
        helper = s.mine[tid]  # this thread's sibling: NOT raced
        for _ in range(random.randint(1, 6)):
            # Run-varying read-modify-writes on the colliding label
            # ``Holder.cb`` (poll/retry-counter shape).  Alternating
            # read/write kinds defeat consecutive-duplicate collapsing,
            # so each iteration lands in the anchor stream.
            helper.cb = (helper.cb or 0) + 1
        _variable_work()
        s.results.append(s.shared.cb)  # racing READ

    return worker


def _invariant(s: State) -> bool:
    # Violated when one thread's write serviced both reads (the recorded
    # counterexample: both writes before either read).
    return not (len(s.results) == 2 and len(set(s.results)) == 1)


def test_anchor_labels_distinguish_instances_of_same_class():
    result = frontrun.explore(
        setup=State,
        workers=[_make_worker(0), _make_worker(1)],
        invariant=_invariant,
        detect_io=False,
        reproduce_on_failure=10,
    )
    assert not result.property_holds, "DPOR failed to detect the write-read race"
    assert result.reproduction_successes >= 8, (
        f"replay reproduced only {result.reproduction_successes}/"
        f"{result.reproduction_attempts} — the sibling Holder instances' "
        "accesses collide with the racing object's anchors (defect #22)"
    )
