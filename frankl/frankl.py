"""Core utilities for exploring the union-closed sets (Frankl) conjecture.

Sets over a universe [n] = {0, ..., n-1} are represented as int bitmasks;
a family is a collection of distinct bitmasks. All proportion arithmetic is
exact (integers / fractions.Fraction) -- never floats -- because the whole
game is played on the knife edge at 1/2.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from fractions import Fraction

Mask = int
Family = frozenset[Mask]


def close_under_union(gens: Iterable[Mask], cap: int | None = None) -> set[Mask] | None:
    """Union-closure of ``gens``: all unions of *nonempty* subfamilies.

    The empty set is a member iff 0 is a generator.  Returns None if the
    closure would exceed ``cap`` members (search cutoff, fail-closed).
    """
    closure: set[Mask] = set()
    for g in gens:
        if g in closure:
            continue
        closure |= {g} | {g | s for s in closure}
        if cap is not None and len(closure) > cap:
            return None
    return closure


def is_union_closed(family: Collection[Mask]) -> bool:
    fam = set(family)
    return all(a | b in fam for a in fam for b in fam)


def universe_mask(family: Iterable[Mask]) -> Mask:
    u = 0
    for s in family:
        u |= s
    return u


def element_counts(family: Iterable[Mask], n: int) -> list[int]:
    """counts[i] = number of members containing element i."""
    counts = [0] * n
    for s in family:
        m = s
        while m:
            b = m & -m
            counts[b.bit_length() - 1] += 1
            m ^= b
    return counts


def abundant_elements(family: Collection[Mask], n: int) -> list[int]:
    """Elements in at least half the members (the conjecture: nonempty)."""
    m = len(family)
    return [i for i, c in enumerate(element_counts(family, n)) if 2 * c >= m and c > 0]


def critical_elements(family: Collection[Mask], n: int) -> list[int]:
    """x with |F_x|/|F| < 1/2 but |F_x|/(|F|-1) >= 1/2, i.e. 2c = |F| - 1."""
    m = len(family)
    return [i for i, c in enumerate(element_counts(family, n)) if 2 * c < m and 2 * c >= m - 1]


def max_proportion(family: Collection[Mask], n: int) -> Fraction:
    """max_x |F_x| / |F| (exact)."""
    m = len(family)
    counts = element_counts(family, n)
    best = max((c for c in counts), default=0)
    return Fraction(best, m)


def frankl_holds(family: Collection[Mask], n: int) -> bool:
    """True iff some element lies in >= half the members.

    Only meaningful for a union-closed family containing a nonempty set.
    """
    return bool(abundant_elements(family, n))


def irreducibles(family: Collection[Mask]) -> set[Mask]:
    """Members not expressible as the union of two *other* members.

    In a union-closed family, s != 0 is reducible iff the union of its proper
    subsets within the family equals s (see theory.md, Lemma on removal).
    The empty set is always irreducible (0 = a|b forces a = b = 0).
    """
    fam = set(family)
    irr: set[Mask] = set()
    for s in fam:
        if s == 0:
            irr.add(s)
            continue
        u = 0
        for t in fam:
            if t != s and t | s == s:  # t is a proper subset of s
                u |= t
        if u != s:
            irr.add(s)
    return irr


def product_family(f0: Collection[Mask], n0: int, f1: Collection[Mask]) -> set[Mask]:
    """{ s0 | (s1 << n0) } -- disjoint-universe product of two families."""
    return {a | (b << n0) for a in f0 for b in f1}


def powerset_family(n: int) -> set[Mask]:
    return set(range(1 << n))


def popcount_members(family: Iterable[Mask]) -> list[int]:
    return sorted(s.bit_count() for s in family)
