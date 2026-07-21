# Frankl union-closed sets conjecture — exploration

Standalone exploration (no dependency on the frontrun package) extending a
"minimal counterexample has to be weird" program: verified structure
theorems, corrections to the base notes, an explicit Sarvate–Renaud-type
construction, and a counterexample search harness.

**Start with [`theory.md`](theory.md)** — the main writeup: status of the
conjecture as of July 2026, two corrections to the base notes (the product
theorem is false as stated without ∅ in both factors; two fixes in the
critical-elements proof), and new structure theorems for a minimal
counterexample, the headline being **at least three critical elements**
(up from two) via a removal calculus for union-irreducible members.

Files:

- `frankl.py` — core library: bitmask families, union-closure, exact
  proportions (`fractions.Fraction`, never floats), irreducibles, critical
  elements, products.
- `test_theory.py` — machine verification: exhaustive over *all*
  union-closed families on universes ≤ 4, hypothesis fuzzing (with
  shrinking) on 9 elements, both directions of the product-theorem
  correction, and the explicit construction. Run `python3 test_theory.py`
  (or pytest). Requires only stdlib; `hypothesis` optional but recommended.
- `constructions.py` — a 188-member union-closed family on 9 elements
  containing a 3-set none of whose elements reaches proportion 1/2
  (22/47 each) — the Sarvate–Renaud phenomenon built from layered up-sets.
- `search.py` — simulated-annealing counterexample search
  (margin-minimization and triple modes), exact arithmetic, deterministic
  per seed.
- `results.md` — what the searches found (spoiler: the conjectured floors,
  and a case where explicit construction beats local search).
