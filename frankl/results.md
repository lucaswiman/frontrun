# Search results

All runs deterministic per seed; commands reproduce exactly.

## Margin minimization (`--mode anneal`)

Objective: minimize the margin `mu(F) = max_x (2|F_x| - |F|)` over
union-closures of generator families (empty set allowed as generator).
A counterexample to the conjecture is exactly `mu <= -1`.

| command | best margin | note |
|---|---|---|
| `python3 search.py --mode anneal --n 10 --restarts 3 --steps 800 --seed 7` | **0** | floor reached immediately |
| `python3 search.py --mode anneal --n 12 --restarts 20 --steps 5000 --seed 2` | **0** | 20/20 restarts hit 0, never below |

The landscape plateaus at the conjectured floor: `mu = 0` is realized by
many small families (e.g. generators `{∅, {e4}, {e2, e11}}`), and no
configuration with `mu < 0` was ever seen. Consistent with the conjecture
being *tight but true*: the extremal surface is everywhere reachable, and
nothing pokes below it. (Per theory.md Prop 2.12, a search restricted to
∅-free families must instead target `mu <= 0`; its conjectured floor is 1.)

## Sarvate–Renaud triple mode (`--mode triple`)

Objective: with the 3-set {0,1,2} forced as a member, minimize
`max(E_0, E_1, E_2)` (exact fractions).

| command | best | note |
|---|---|---|
| `python3 search.py --mode triple --n 8 --restarts 3 --steps 800 --seed 3` | 13/25 | above 1/2 |
| `python3 search.py --mode triple --n 9 --restarts 30 --steps 5000 --seed 1` | **1/2** (\|F\| = 50) | plateau at exactly 1/2 |

Local search never got below 1/2. The explicit layered up-set construction
(`constructions.py::triple_below_half`) does:

| (w, d) | universe | \|F\| | E_a = E_b = E_c |
|---|---|---|---|
| (3, 1) | 6 | 34 | 8/17 ≈ 0.4706 |
| **(6, 2)** | **9** | **188** | **22/47 ≈ 0.4681** |
| (9, 3) | 12 | 1240 | 74/155 ≈ 0.4774 |

General formula: proportion `(q + 2p + s) / (2q + 3p + 3s)` with
`q = 2^w, s = 2^(w-d), p = 2^(w-2d)`; below 1/2 by exactly `(s-p)/2` sets
for every `d >= 1`. Within this family (w, d) = (6, 2) is optimal.

The gap between what annealing finds (1/2) and what the construction
achieves (22/47) is the empirical takeaway: near the conjecture's boundary
the interesting objects are *structured*, and unguided local search does
not find them. Any serious counterexample hunt should search the
constrained incidence structures of theory.md §2, not raw generator space.
