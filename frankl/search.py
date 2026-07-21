"""Counterexample search for the union-closed sets conjecture.

Two modes:

  anneal  -- simulated annealing over generator families on universe [n]
             (the empty set is allowed as a generator), minimizing the
             *margin* max_x (2|F_x| - |F|) of the closure.  margin >= 0 iff
             some element is in at least half the sets, so a counterexample
             is exactly a family with margin <= -1.  Powersets including
             the empty set achieve margin 0 (the conjecture is tight there);
             we search for margin < 0.  Note: for families WITHOUT the empty
             set the conjectured floor is margin >= 1, since adding the empty
             set to a margin-0 family would produce a counterexample.

  triple  -- Sarvate-Renaud mode: force the 3-set {0,1,2} to be a member and
             minimize the max proportion among elements 0, 1, 2 only.  This
             reproduces the known phenomenon that a smallest set of size 3
             need not contain an abundant element, and measures how far below
             1/2 its elements can simultaneously be pushed.

All comparisons are exact (integer margin / Fraction), never floats.
Deterministic per seed.
"""

from __future__ import annotations

import argparse
import random
import sys
from fractions import Fraction

from frankl import close_under_union, element_counts

CLOSURE_CAP = 6000


def margin(fam: set[int], n: int) -> int:
    """max_x (2|F_x| - |F|) over x in the universe.  >= 0 iff Frankl holds."""
    m = len(fam)
    counts = element_counts(fam, n)
    return max(2 * c - m for c in counts if c > 0)


def triple_objective(fam: set[int], n: int) -> Fraction:
    m = len(fam)
    counts = element_counts(fam, n)
    return max(Fraction(counts[i], m) for i in range(3))


def mutate(gens: list[int], n: int, rng: random.Random, fixed: list[int]) -> list[int]:
    new = list(gens)
    op = rng.random()
    if op < 0.6 or len(new) <= 2:
        i = rng.randrange(len(new))
        new[i] ^= 1 << rng.randrange(n)
    elif op < 0.8 and len(new) < 2 * n:
        new.append(rng.randrange(0, 1 << n))
    else:
        del new[rng.randrange(len(new))]
    out = fixed + [g for g in new if g not in fixed]
    if not any(out):  # keep at least one nonempty member
        out.append(1 << rng.randrange(n))
    return out


def anneal(
    n: int,
    steps: int,
    rng: random.Random,
    mode: str,
) -> tuple[object, list[int], int]:
    fixed = [0b111] if mode == "triple" else []
    ngens = rng.randint(3, n)
    gens = fixed + [rng.randrange(1, 1 << n) for _ in range(ngens)]

    def score(g: list[int]):
        fam = close_under_union(g, cap=CLOSURE_CAP)
        if fam is None:
            return None, 0
        if mode == "triple":
            return triple_objective(fam, n), len(fam)
        return margin(fam, n), len(fam)

    cur, cur_size = score(gens)
    while cur is None:
        gens = fixed + [rng.randrange(1, 1 << n) for _ in range(ngens)]
        cur, cur_size = score(gens)
    best, best_gens, best_size = cur, list(gens), cur_size

    temp_hi, temp_lo = 2.0, 0.01
    for step in range(steps):
        temp = temp_hi * (temp_lo / temp_hi) ** (step / max(steps - 1, 1))
        cand = mutate(gens, n, rng, fixed)
        val, size = score(cand)
        if val is None:
            continue
        delta = float(val - cur)
        if delta <= 0 or rng.random() < 2.718 ** (-delta / temp):
            gens, cur, cur_size = cand, val, size
            if (cur, -cur_size) < (best, -best_size):
                best, best_gens, best_size = cur, list(gens), cur_size
    return best, best_gens, best_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["anneal", "triple"], default="anneal")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    global_best = None
    for r in range(args.restarts):
        best, gens, size = anneal(args.n, args.steps, rng, args.mode)
        record = (best, size, gens)
        if global_best is None or (best, -size) < (global_best[0], -global_best[1]):
            global_best = record
            print(f"[restart {r}] new best: objective={best} |F|={size} gens={sorted(set(gens))}", flush=True)
        else:
            print(f"[restart {r}] best this restart: {best} (|F|={size})", flush=True)

    assert global_best is not None
    best, size, gens = global_best
    print(f"\nmode={args.mode} n={args.n}: global best objective = {best} with |F| = {size}")
    print(f"generators: {sorted(set(gens))}")
    if args.mode == "anneal":
        if isinstance(best, int) and best < 0:
            print("!!! MARGIN < 0: POTENTIAL COUNTEREXAMPLE -- verify independently !!!")
            sys.exit(2)
        print("(margin 0 = conjecture exactly tight, as for powersets; never went below)")
    else:
        print(f"(triple max-proportion {best} = {float(best):.4f}; < 1/2 reproduces Sarvate-Renaud)")


if __name__ == "__main__":
    main()
