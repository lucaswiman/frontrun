"""Regression test for cross-execution stability of ``StableObjectIds``.

DPOR carries object-ID-keyed state (sleep sets, trace caches) ACROSS
executions inside the Rust ``Path``.  ``StableObjectIds`` originally assigned
IDs in *first-touch order during worker execution* and cleared the map every
execution.  After a DPOR backtrack changes the schedule, the same logical
object is touched in a different order, so it gets a *different* ID than last
execution and old IDs are reassigned to different objects.

The sleep-set propagation then compares last-execution IDs against the
current execution's IDs.  With permuted IDs, a sleeping thread's cached
future accesses can appear independent of a running thread's dependent
access — so the thread stays asleep and a genuinely distinct Mazurkiewicz
trace is silently pruned (unsound: missed interleavings).

These tests assert that the same logical object receives the same stable ID
across two executions whose *post-divergence touch order differs*, which the
first-touch-order implementation fails.
"""

from __future__ import annotations

from frontrun._opcode_observer import StableObjectIds


class _Cell:
    """A distinct logical object identified by its ``name``."""

    def __init__(self, name: str) -> None:
        self.name = name


def _setup() -> dict[str, _Cell]:
    """Deterministically rebuild the same logical-object graph each execution."""
    return {"X": _Cell("X"), "C": _Cell("C")}


def _pre_register(sids: StableObjectIds, root: object) -> None:
    """Pre-register *root*'s object graph if the API exists (no-op otherwise).

    Kept tolerant so the failing assertion is the *permutation* of IDs (the
    actual bug), not a missing-method error.
    """
    fn = getattr(sids, "pre_register", None)
    if fn is not None:
        fn(root)


def test_stable_id_consistent_across_permuted_touch_order() -> None:
    """The same logical object must map to the same ID across executions.

    Execution 1 touches X then C (T0 runs first).
    Execution 2 touches C then X (T1 runs first after a backtrack).

    A correct implementation gives X the same ID in both, and C the same ID
    in both.  The first-touch-counter implementation instead gives X id 0 then
    id 1 (and C id 1 then id 0) — permuted, which corrupts the Rust sleep-set
    comparison.
    """
    sids = StableObjectIds()

    # --- Execution 1: state built, then touched in order X, C ---
    sids.reset_for_execution()
    state1 = _setup()
    _pre_register(sids, state1)
    x1 = sids.get(state1["X"])
    c1 = sids.get(state1["C"])

    # --- Execution 2: fresh state, touched in the OPPOSITE order C, X ---
    sids.reset_for_execution()
    state2 = _setup()
    _pre_register(sids, state2)
    c2 = sids.get(state2["C"])
    x2 = sids.get(state2["X"])

    assert x1 == x2, (
        f"Logical object X got id {x1} in exec 1 but id {x2} in exec 2 — "
        f"permuted IDs corrupt cross-execution sleep-set comparison"
    )
    assert c1 == c2, (
        f"Logical object C got id {c1} in exec 1 but id {c2} in exec 2 — "
        f"permuted IDs corrupt cross-execution sleep-set comparison"
    )
    # X and C must remain distinguishable (no collapse).
    assert x1 != c1, "Distinct logical objects must keep distinct IDs"


def test_stable_id_distinguishes_instances_of_same_type() -> None:
    """Two distinct instances of the same type must get distinct, stable IDs.

    Guards against an over-coarse fix that hashes purely by type and collapses
    all instances together (which would over-serialize unrelated objects).
    """
    sids = StableObjectIds()

    sids.reset_for_execution()
    s1 = _setup()
    _pre_register(sids, s1)
    a1 = sids.get(s1["X"])
    b1 = sids.get(s1["C"])

    sids.reset_for_execution()
    s2 = _setup()
    _pre_register(sids, s2)
    a2 = sids.get(s2["X"])
    b2 = sids.get(s2["C"])

    assert a1 != b1, "Two distinct _Cell instances collapsed to one ID"
    assert a1 == a2 and b1 == b2, "Per-instance IDs must be stable across executions"
