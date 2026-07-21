"""Verification of the theory in theory.md.

Two layers:
  1. Exhaustive: every union-closed family on a universe of <= 4 elements
     (all 2^16 candidate families are examined for n = 4).
  2. Property-based fuzzing (hypothesis): random generator families on larger
     universes; hypothesis shrinks any counterexample it finds.

Run directly (``python3 test_theory.py``) or via pytest.
"""

from __future__ import annotations

import itertools
import random

from frankl import (
    close_under_union,
    critical_elements,
    element_counts,
    frankl_holds,
    irreducibles,
    is_union_closed,
    product_family,
    universe_mask,
)

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAVE_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    HAVE_HYPOTHESIS = False


def all_union_closed_families(n: int):
    """Yield every union-closed family on universe [n] containing a nonempty set."""
    subsets = list(range(1 << n))
    for bits in range(1, 1 << len(subsets)):
        fam = [s for i, s in enumerate(subsets) if bits >> i & 1]
        if all(s == 0 for s in fam):
            continue
        if is_union_closed(fam):
            yield fam


# --------------------------------------------------------------------------
# Exhaustive layer (n <= 4 for the conjecture itself, n <= 3 for the heavier
# per-family lemma checks; n = 4 spot-checks lemmas on a random sample).
# --------------------------------------------------------------------------


def _check_family_lemmas(fam: list[int], n: int) -> None:
    m = len(fam)
    counts = element_counts(fam, n)

    # Conjecture holds (n <= 12 is known in the literature; we recheck).
    assert frankl_holds(fam, n), f"conjecture fails on {fam}"

    # Critical elements: definition forces |F| odd.
    crits = critical_elements(fam, n)
    if crits:
        assert m % 2 == 1
        for x in crits:
            assert 2 * counts[x] == m - 1

    # Singleton lemma: {a} in F  =>  a abundant.
    for i in range(n):
        if (1 << i) in fam:
            assert 2 * counts[i] >= m

    # Doubleton theorem: {a,b} in F  =>  a or b abundant.
    for i, j in itertools.combinations(range(n), 2):
        if (1 << i) | (1 << j) in fam:
            assert 2 * counts[i] >= m or 2 * counts[j] >= m

    # Irreducible-removal lemma: removing ANY subset of irreducibles
    # preserves union-closure (check singletons, pairs, and the whole set).
    irr = sorted(irreducibles(fam))
    removals = [[s] for s in irr] + [list(p) for p in itertools.combinations(irr, 2)] + [irr]
    for rem in removals:
        rest = [s for s in fam if s not in rem]
        assert is_union_closed(rest), f"removing {rem} from {fam} broke closure"

    # Cross-check the irreducibility implementation against the binary
    # definition: s is reducible iff s = t | v for members t, v != s.
    for s in fam:
        binary_reducible = any(t | v == s for t in fam for v in fam if t != s and v != s)
        assert (s not in irr) == binary_reducible

    # Every member is a union of irreducibles; every universe element is in
    # some irreducible.
    for s in fam:
        u = 0
        for t in irr:
            if t | s == s:
                u |= t
        assert u == s
    assert universe_mask(irr) == universe_mask(fam)


def test_exhaustive_small_universes() -> None:
    for n in (1, 2, 3):
        for fam in all_union_closed_families(n):
            _check_family_lemmas(fam, n)


def test_exhaustive_n4_conjecture_and_sampled_lemmas() -> None:
    rng = random.Random(0)
    count = 0
    for fam in all_union_closed_families(4):
        count += 1
        assert frankl_holds(fam, 4), f"conjecture fails on {fam}"
        if rng.random() < 0.01:
            _check_family_lemmas(fam, 4)
    assert count > 1000  # sanity: we actually enumerated something


# --------------------------------------------------------------------------
# Product theorem: the original statement is FALSE without the empty set in
# both factors; the corrected statement holds.
# --------------------------------------------------------------------------


def test_product_theorem_original_statement_fails() -> None:
    # F0 = {{a}}, F1 = {{b}} are union-closed with disjoint universes, but
    # C(F0 u F1) = {emptyset, {a}, {b}, {a,b}} under the closure-with-empty-
    # union definition (or {{a},{b},{a,b}} under pairwise closure) -- neither
    # has size |F0| * |F1| = 1, and E_a drops from 1 to 1/2 (or 2/3).
    f0, f1 = [0b1], [0b1]
    t = close_under_union([f0[0], f1[0] << 1])
    assert t is not None
    t_with_empty = t | {0}
    assert len(t_with_empty) == 4 != len(f0) * len(f1)
    counts = element_counts(t_with_empty, 2)
    assert 2 * counts[0] == len(t_with_empty)  # E_a = 1/2, not E_a(F0) = 1


def _random_uc_family(rng: random.Random, n: int, ngens: int, with_empty: bool) -> list[int]:
    gens = [rng.randrange(1, 1 << n) for _ in range(ngens)]
    fam = close_under_union(gens)
    assert fam is not None
    if with_empty:
        fam.add(0)
    return sorted(fam)


def test_product_theorem_corrected() -> None:
    # With emptyset in BOTH factors: |T| = |F0|*|F1| and proportions of every
    # element are preserved exactly.
    rng = random.Random(1)
    for _ in range(200):
        n0, n1 = rng.randint(1, 5), rng.randint(1, 5)
        f0 = _random_uc_family(rng, n0, rng.randint(1, 4), with_empty=True)
        f1 = _random_uc_family(rng, n1, rng.randint(1, 4), with_empty=True)
        t = sorted(product_family(f0, n0, f1))
        # t IS the union-closure of f0 u shifted(f1) here:
        direct = close_under_union(list(f0) + [s << n0 for s in f1])
        assert direct is not None
        assert set(t) == direct | {0}
        assert len(t) == len(f0) * len(f1)
        c0, ct = element_counts(f0, n0), element_counts(t, n0 + n1)
        for x in range(n0):
            # E_x(T) == E_x(F0): c_t[x] * |F0| == c_0[x] * |T| with |T| = |F0||F1|
            assert ct[x] * len(f0) == c0[x] * len(t)
        c1 = element_counts(f1, n1)
        for x in range(n1):
            assert ct[n0 + x] * len(f1) == c1[x] * len(t)


# --------------------------------------------------------------------------
# Fuzzing layer (hypothesis shrinks counterexamples -- the "frontrun spirit").
# --------------------------------------------------------------------------

if HAVE_HYPOTHESIS:
    N_FUZZ = 9

    @settings(max_examples=2000, deadline=None)
    @given(st.lists(st.integers(min_value=1, max_value=(1 << N_FUZZ) - 1), min_size=1, max_size=8))
    def test_fuzz_conjecture(gens: list[int]) -> None:
        fam = close_under_union(gens)
        assert fam is not None
        assert frankl_holds(fam, N_FUZZ), f"COUNTEREXAMPLE? gens={gens} family={sorted(fam)}"

    @settings(max_examples=500, deadline=None)
    @given(
        st.lists(st.integers(min_value=0, max_value=(1 << N_FUZZ) - 1), min_size=1, max_size=8),
        st.randoms(use_true_random=False),
    )
    def test_fuzz_irreducible_removal(gens: list[int], rng: random.Random) -> None:
        fam = close_under_union(gens)
        assert fam is not None
        irr = sorted(irreducibles(fam))
        rem = [s for s in irr if rng.random() < 0.5]
        rest = [s for s in fam if s not in rem]
        assert is_union_closed(rest)

    @settings(max_examples=500, deadline=None)
    @given(st.lists(st.integers(min_value=1, max_value=(1 << N_FUZZ) - 1), min_size=1, max_size=8))
    def test_fuzz_removal_abundance_transfer(gens: list[int]) -> None:
        """The engine of the critical-element theorems: if F is union-closed,
        s irreducible, and x abundant in F - {s}, then x is abundant in F or
        x is 'one short' (2|F_x| = |F| - 1, x not in s)."""
        fam = close_under_union(gens)
        assert fam is not None
        m = len(fam)
        for s in irreducibles(fam):
            rest = [t for t in fam if t != s]
            if not rest or all(t == 0 for t in rest):
                continue
            counts_rest = element_counts(rest, N_FUZZ)
            counts_full = element_counts(fam, N_FUZZ)
            for x in range(N_FUZZ):
                if 2 * counts_rest[x] >= m - 1 and counts_rest[x] > 0:
                    in_s = bool(s >> x & 1)
                    if in_s:
                        assert 2 * counts_full[x] >= m
                    else:
                        assert 2 * counts_full[x] >= m - 1


def test_triple_below_half_construction() -> None:
    from constructions import triple_below_half

    for w, d in [(3, 1), (6, 2), (6, 1), (9, 3), (8, 2)]:
        fam, n = triple_below_half(w, d)
        m = len(fam)
        assert is_union_closed(fam)
        assert 0b111 in fam  # the 3-set {0,1,2} is a member
        counts = element_counts(fam, n)
        for x in range(3):
            assert 2 * counts[x] < m, f"(w={w},d={d}): element {x} not below 1/2"
        assert frankl_holds(fam, n)  # the conjecture still holds globally
    fam, n = triple_below_half(6, 2)
    counts = element_counts(fam, n)
    assert len(fam) == 188 and counts[0] == counts[1] == counts[2] == 88


def main() -> None:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        print(f"{name} ... ", end="", flush=True)
        fn()
        print("ok")
    print(f"\nall {len(tests)} checks passed (hypothesis={'on' if HAVE_HYPOTHESIS else 'OFF'})")


if __name__ == "__main__":
    main()
