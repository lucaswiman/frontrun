"""Explicit constructions.

`triple_below_half(w, d)` builds a union-closed family containing the 3-set
{0,1,2} in which NONE of 0, 1, 2 is abundant -- the Sarvate-Renaud
phenomenon, from scratch, beating what local search finds (search.py's
triple mode plateaus at exactly 1/2).

Construction.  Universe = {0,1,2} + W with |W| = w.  Fix disjoint
d-subsets D_0, D_1, D_2 of W.  Describe the family by layers
H_T = { s ∩ W : s ∈ F, s ∩ {0,1,2} = T } for T ⊆ {0,1,2}:

    H_∅ = H_{012} = P(W),   H_{i} = up(D_i),   H_{ij} = up(D_i ∪ D_j)

where up(D) = { A ⊆ W : A ⊇ D }.  Union-closure reduces to the layer
condition  H_{T1} ⊎ H_{T2} ⊆ H_{T1 ∪ T2}  (element-wise unions), which
up-sets satisfy: up(D) ⊎ up(D') ⊆ up(D ∪ D'), and everything lands in
P(W) at the top.  ∅ ∈ H_{012} makes {0,1,2} a member.

Counting (q = |P(W)| = 2^w, s = 2^(w-d), p = 2^(w-2d)):
    |F_0|  = q + 2p + s          (layers 012, 01, 02, 0)
    |F|    = 2q + 3p + 3s
so 2|F_0| - |F| = p - s < 0 whenever d >= 1: elements 0, 1, 2 all sit
strictly below 1/2.  For (w, d) = (6, 2): proportion 88/188 = 22/47 ≈ 0.468
on a 9-element universe with 188 member sets.
"""

from __future__ import annotations

from frankl import Mask


def triple_below_half(w: int = 6, d: int = 2) -> tuple[set[Mask], int]:
    """Returns (family, n) with n = 3 + w.  Requires 1 <= d and 3*d <= w."""
    if not (1 <= d and 3 * d <= w):
        raise ValueError("need 1 <= d and 3*d <= w")
    n = 3 + w
    dmask = [((1 << d) - 1) << (3 + i * d) for i in range(3)]
    wall = ((1 << w) - 1) << 3
    powerset_w = [m << 3 for m in range(1 << w)]

    def up(dm: Mask) -> list[Mask]:
        return [a for a in powerset_w if a & dm == dm]

    layers: dict[int, list[Mask]] = {0b000: powerset_w, 0b111: powerset_w}
    for i in range(3):
        layers[1 << i] = up(dmask[i])
    for i in range(3):
        for j in range(i + 1, 3):
            layers[(1 << i) | (1 << j)] = up(dmask[i] | dmask[j])

    fam = {t | a for t, sets in layers.items() for a in sets}
    assert wall  # universe sanity
    return fam, n
