# The union-closed sets conjecture: minimal-counterexample structure, corrections, and search

Notes extending the "a minimal counterexample has to be weird" program.
Everything here that is checkable by machine is checked in `test_theory.py`
(exhaustively for universes of ≤ 4 elements, by hypothesis-fuzzing above that).

**Conventions.** Families are finite families of finite sets containing at
least one nonempty set. $m = |F|$. Membership of $\emptyset$ *matters* (it
counts in the denominator of every proportion) and is tracked explicitly
throughout — this is where the one real bug below comes from.
$\mathcal{E}_x(F) = |F_x|/|F|$ as in the base notes. *Union-closed* means
closed under pairwise (equivalently finite nonempty) unions. An element $x$
is *abundant* if $\mathcal{E}_x(F)\ge 1/2$; a counterexample is a family with
no abundant element. An element is *critical* if
$|F_x|/|F| < 1/2 \le |F_x|/(|F|-1)$, equivalently $2|F_x| = |F|-1$ (which
forces $|F|$ odd, as shown in the base notes).

**Status of the conjecture (checked July 2026).** Open. Gilmer's 2022
entropy breakthrough gave the first constant bound (an element in ≥ 1% of
the sets); the constant was pushed to $(3-\sqrt5)/2 \approx 0.38197$ by
Alweiss–Huang–Sellke, Chase–Lovett, Sawin and Pebody, and by refined
couplings to ≈ 0.38234 (Sawin, evaluated by Yu and Cambie) and ≈ 0.38271
(Liu). Crucially, the *approximate* version of the problem that all these
methods actually bound is **genuinely tight near 0.382** — there are
"almost union-closed" families with no element above $(3-\sqrt5)/2$ — so the
entropy/coupling route provably cannot reach 1/2 by itself. Any proof (or
refutation) must exploit exact closure. Separately, exhaustive/structural
work shows a counterexample needs a universe of ≥ 13 elements
(Bošnjak–Marković for 11, Vučković–Živković for 12) and at least 47 member
sets (Roberts–Simpson: a counterexample on a minimal universe of $q$
elements has ≥ $4q-1$ members). See references at the end.

---

## 1. Corrections to the base notes

### 1.1 The product theorem is false as stated

**Claimed:** for union-closed $F_0, F_1$ with disjoint universes and
$T = \mathcal{C}(F_0\cup F_1)$: (1) $|T| = |F_0|\cdot|F_1|$, and (2)
proportions of every element are inherited from its factor.

**Counterexample:** $F_0 = \{\{a\}\}$, $F_1 = \{\{b\}\}$. Then
$\mathcal{C}(F_0\cup F_1) \supseteq \{\{a\},\{b\},\{a,b\}\}$ (four members
if the empty union is admitted, three if not) — not $1 = |F_0|\cdot|F_1|$ —
and $\mathcal{E}_a$ drops from $1$ to $1/2$ (or $2/3$), not preserved.

**What went wrong:** the map $u(s_0,s_1) = s_0\cup s_1$ is always
*injective* (recover $s_0 = u \cap \mathcal{U}(F_0)$), but it is only
*surjective* onto $\mathcal{C}(F_0\cup F_1)$ when each pure-$F_0$ member
$s_0$ can be written as $s_0 \cup s_1$ — i.e. when $\emptyset \in F_1$, and
symmetrically $\emptyset \in F_0$.

**Corrected statement.** *Let $F_0, F_1$ be union-closed with disjoint
universes and $\emptyset \in F_0$, $\emptyset \in F_1$. Then
$T = \mathcal{C}(F_0\cup F_1) = \{s_0\cup s_1 : s_0\in F_0,\, s_1\in F_1\}$,
$|T| = |F_0|\cdot|F_1|$, and $\mathcal{E}_x(T) = \mathcal{E}_x(F_i)$ for
every $x\in\mathcal{U}(F_i)$.* The original proof then goes through
verbatim. (Verified both directions in
`test_theory.py::test_product_theorem_*`.)

The corollary you presumably wanted survives, because adjoining $\emptyset$
to a counterexample only shrinks proportions:

**Corollary 1.1.** If $F_0, F_1$ are counterexamples with disjoint
universes, then $(F_0\cup\{\emptyset\}) \times (F_1\cup\{\emptyset\})$ (in
the product-of-unions sense above) is a counterexample. Hence if one
counterexample exists, infinitely many do.

### 1.2 Two fixes in the critical-elements theorem, part (2)

* The parenthetical "choose $s'$ such that $s'$ has **no non-empty subsets
  in $F$** (for example, the set of smallest cardinality containing $x$)"
  claims too much: the minimum-cardinality set containing $x$ may well have
  nonempty subsets in $F$ — subsets *not containing $x$*. What the argument
  actually needs is that $s'$ is not the union of two *other* members
  (**union-irreducible**), and that is what minimum cardinality among
  $x$-containing sets gives you: if $s' = t\cup v$ with $t,v \ne s'$, then
  $t,v \subsetneq s'$ and one of them contains $x$, contradicting
  minimality.
* The final line "$x\ne x'$ since $x\in s$ and $x'\notin s'$" has a typo —
  part (1) proved $x\notin s$. The correct reason is $x\in s'$ but
  $x'\notin s'$.
* (Pedantic) both parts implicitly need $F\smallsetminus\{s\}$ to still be a
  legal instance (nonempty, not $\{\emptyset\}$); families that small are
  directly checked not to be counterexamples.

With those repairs both parts are correct — and part (2) becomes a
one-line corollary of Theorem 3 below.

---

## 2. New structure theorems

Throughout: $F$ is a **minimal counterexample** (minimum $m = |F|$ among all
counterexamples), $X$ its set of critical elements,
$\mathrm{Irr}(F)$ its set of union-irreducible members ($s$ with no
decomposition $s = t\cup v$, $t,v\in F\smallsetminus\{s\}$).

**Lemma 2.1 (irreducibles).** In any finite union-closed $F$:
(a) every member is a union of irreducible members; hence every element of
$\mathcal{U}(F)$ lies in some irreducible;
(b) an irreducible $s$ is not the union of *any* subfamily of
$F\smallsetminus\{s\}$;
(c) **for every $R\subseteq \mathrm{Irr}(F)$, $F\smallsetminus R$ is
union-closed.**

*Proof.* (a) Strong induction on $|s|$: a reducible $s = t\cup v$ has
$t,v\subsetneq s$. (b) Induction on $|T|$ for $\bigcup T = s$,
$T\subseteq F\smallsetminus\{s\}$: pick $t\in T$; $w=\bigcup(T\smallsetminus\{t\})\in F$;
if $w\ne s$ then $s=t\cup w$ contradicts irreducibility; if $w=s$ recurse.
(c) If $t,v\in F\smallsetminus R$ and $t\cup v = s\in R$, then $t,v\ne s$
contradicts irreducibility of $s$. $\square$

(c) is the engine of everything below: *any* set of irreducibles can be
deleted simultaneously, and what remains is a smaller union-closed family
to which minimality applies. Checked exhaustively and by fuzzing.

**Theorem 2.2 (deficiency counting).** Let $R\subseteq\mathrm{Irr}(F)$,
$|R| = k\ge 1$. Then there is an element $y$ with

1. $|(F\smallsetminus R)_y| \ge (m-k)/2$ (abundant in the reduced family),
2. $y$ belongs to at most $\lfloor (k-1)/2\rfloor$ members of $R$.

*Proof.* $F\smallsetminus R$ is union-closed (2.1c) and smaller, so by
minimality it has an abundant $y$: $|(F\smallsetminus R)_y|\ge (m-k)/2$.
With $j = |\{s\in R : y\in s\}|$ we get
$|F_y| = |(F\smallsetminus R)_y| + j \ge (m-k)/2 + j$, while
$|F_y| < m/2$ since $F$ is a counterexample; so $j < k/2$. $\square$

**Theorem 2.3 (a critical element avoids every irreducible).** For **every**
$s\in\mathrm{Irr}(F)$ there is a critical $x\notin s$. Consequently $m$ is
odd, and no irreducible member contains all critical elements.

*Proof.* $k=1$ in Theorem 2.2: $j=0$, so $x\notin s$ and
$|F_x| = |(F\smallsetminus\{s\})_x| \ge (m-1)/2$; with $|F_x| < m/2$ this
forces $2|F_x| = m-1$ (and $m$ odd): $x$ is critical. $\square$

This strengthens part (1)–(2) of the base notes from *one particular*
subset-minimal $s$ to all irreducibles at once, and gives the two-critical
result instantly: pick an irreducible $s_1$ containing the critical $x_1$
(exists by 2.1a); Theorem 2.3 hands a critical $x_2\notin s_1 \ni x_1$.

**Theorem 2.4 (pair avoidance).** For every pair of distinct
$s,t\in\mathrm{Irr}(F)$ there is a *critical* $x$ with $x\notin s$ and
$x\notin t$.

*Proof.* $k=2$ in Theorem 2.2: $j\le 0$, so $y$ avoids both, and
$|F_y| \ge \lceil (m-2)/2\rceil = (m-1)/2$ since $m$ is odd; again
$2|F_y| = m-1$. $\square$

**Theorem 2.5 (at least THREE critical elements).** $|X| \ge 3$.

*Proof.* First, $|\mathrm{Irr}(F)|\ge 2$: if $\mathrm{Irr} = \{s\}$
(possibly plus $\emptyset$), then by 2.1a $F\subseteq\{\emptyset, s\}$ and
neither $\{s\}$ nor $\{\emptyset,s\}$ is a counterexample.
Now suppose $|X|\le 2$, say $X = \{x_1, x_2\}$ (possibly $x_1 = x_2$; $X\ne\emptyset$
by 2.3). By 2.1a choose irreducibles $s_1\ni x_1$ and $s_2\ni x_2$, taking
$s_1 = s_2$ whenever possible. If $s_1\ne s_2$, the pair $\{s_1,s_2\}$
violates Theorem 2.4: its avoider must be critical, but $x_1\in s_1$ and
$x_2\in s_2$. If $s_1 = s_2 = s^{*}$, pick any other irreducible
$u\ne s^{*}$; the pair $\{s^{*}, u\}$ needs a critical avoider outside
$s^{*}$, but $x_1, x_2 \in s^{*}$. Contradiction either way. $\square$

**Theorem 2.6 (rigidity when $|X| = 3$).** If $F$ has exactly three
critical elements, then no irreducible member contains two of them — the
irreducibles containing critical elements are partitioned into three
"colour classes", one per critical element.

*Proof.* Suppose $s\in\mathrm{Irr}$ contains $x_1, x_2$. By 2.3,
$x_3\notin s$. Choose an irreducible $t\ni x_3$; then $t \ne s$, and the
pair $\{s,t\}$ has no critical avoider: $x_1,x_2\in s$, $x_3\in t$,
contradicting 2.4. $\square$

**Theorem 2.7 (a minimal counterexample barely fails).** Every critical
element lies in exactly $(m-1)/2$ members, every element lies in at most
$(m-1)/2$ members, and the maximum is attained (2.3). Hence
$$\max_x \mathcal{E}_x(F) \;=\; \frac12 - \frac{1}{2m} \;\ge\; \frac12 - \frac1{94} \approx 0.4894,$$
using $m\ge 47$. A minimal counterexample misses the conjectured bound by
less than $1.1\%$ — it is an extremal object balanced on a knife edge, not
a "wild" family.

**Proposition 2.8 (WLOG separating).** If $x\ne y$ lie in exactly the same
members ("twins"), deleting $y$ from every member is a size- and
proportion-preserving isomorphism onto a union-closed family. Hence a
*minimal* counterexample may be taken **separating** (no twins), with
universe size $\le$ the number of distinct sets $F_x$.

*Proof.* $s\mapsto s\smallsetminus\{y\}$ collides only on pairs
$s, s\cup\{y\}$ with $y\notin s$; twin-ness makes $x$ a member of exactly
one of them and not the other — impossible since they agree off $y$.
Closure and counts are visibly preserved. $\square$

**Proposition 2.9 (indecomposability).** A minimal counterexample is not a
disjoint-universe product: there are no union-closed $F_0, F_1$ containing
$\emptyset$, on disjoint nonempty universes, with
$F = \{s_0\cup s_1 : s_i \in F_i\}$.

*Proof.* By the corrected product theorem, every $x\in\mathcal{U}(F_i)$ has
$\mathcal{E}_x(F_i) = \mathcal{E}_x(F) < 1/2$, so each $F_i$ is itself a
counterexample; but $|F_i| \ge 2$ (it contains $\emptyset$ and a nonempty
set), so $|F_0| = |F|/|F_1| < |F|$, contradicting minimality. $\square$

**Proposition 2.10 (no small members).** Any counterexample has all members
of size ≥ 3: a singleton $\{a\}\in F$ makes $a$ abundant
($s\mapsto s\cup\{a\}$ injects the $a$-free members into the $a$-members),
and the doubleton theorem of the base notes handles size 2. This is where
the "local" road ends: Sarvate–Renaud found union-closed families whose
unique minimal member is a 3-set none of whose elements is abundant, and
Poonen characterized exactly which local configurations force abundance.
`constructions.py` builds the phenomenon from scratch: a 188-member
union-closed family on 9 elements containing the 3-set $\{a,b,c\}$ with
$\mathcal{E}_a = \mathcal{E}_b = \mathcal{E}_c = 22/47 \approx 0.468$
(layered up-set construction, verified in `test_theory.py`). Notably, the
simulated annealer in `search.py --mode triple` plateaus at exactly $1/2$
and never finds this — structure-guided construction beats local search, a
data point for the search-strategy discussion in §4.

**Observation 2.11 (bookkeeping).** Since every member is a union of
nonempty irreducibles, $m \le 2^{|\mathrm{Irr}(F)|}$; with $m \ge 47$ a
minimal counterexample has at least 6 nonempty irreducible members — so by
2.4 there are ≥ 15 irreducible pairs, every one of which must be dodged by
one of the ≥ 3 critical elements.

**Proposition 2.12 (the ∅-free reformulation behind the search).** The
conjecture is equivalent to: *every union-closed family **not** containing
$\emptyset$ has an element in **strictly more** than half of its members.*
Define the margin $\mu(F) = \max_x (2|F_x| - |F|)$. Then the conjecture
says $\mu \ge 0$ for all families, and $\mu \ge 1$ for ∅-free families —
if some ∅-free $F$ had $\mu(F) = 0$, then $F\cup\{\emptyset\}$ would be a
counterexample.

*Proof.* Adjoining/removing $\emptyset$ changes no $|F_x|$ and shifts $m$
by one; $2|F_x|\ge m+1$ iff $2|F_x| \ge m'$ for $m' = m+1$. $\square$

This matters for search design: a harness that (like ours, initially)
never generates $\emptyset$ must hunt for margin $\le 0$, not margin
$\le -1$, or it silently weakens the target — a nice instance of the
fail-closed lesson from frontrun's own design principles.

### Where this leaves the program

Stacking everything up, a minimal counterexample must: have odd $m \ge 47$
on a universe of $n\ge 13$ elements; be separating and indecomposable;
contain no member of size < 3; have ≥ 6 irreducibles and ≥ 3 critical
elements, each critical sitting at *exactly* $(m-1)/2$ memberships; have
every irreducible (indeed every irreducible *pair*) avoided by a critical
element; and if it has exactly 3 criticals, no irreducible sees two of
them. None of this is yet close to a contradiction — but 2.2 is a
machine: other removal sets $R$ (e.g. all irreducibles containing a fixed
critical element) give further inequalities, and pushing $|X|\ge 3$ to
$|X|\ge 4$ via the $k=3$ case of 2.2 plus 2.6-style colouring is a
concrete, plausibly-tractable next step.

---

## 3. What the computation shows (`test_theory.py`, `search.py`)

* Exhaustive over all union-closed families on universes of ≤ 4 elements
  (all $2^{16}$ candidate families for $n=4$): the conjecture, the
  singleton/doubleton lemmas, Lemma 2.1 (all removal subsets), and the
  irreducibility characterizations all hold.
* Hypothesis fuzzing on $n = 9$ (closures of random generator families,
  with shrinking): conjecture + removal lemma + the abundance-transfer
  step that powers Theorems 2.2–2.4. No violations.
* `search.py --mode anneal`: simulated annealing minimizing the margin
  $\mu$ over generator families ($\emptyset$ allowed). Reaches the
  conjectured floor $\mu = 0$ quickly and never went below, across seeds
  and universe sizes up to 13.
* `search.py --mode triple`: forces a 3-set member and minimizes the max
  proportion over its three elements. Local search reached exactly 1/2 but
  never below; the explicit construction in `constructions.py` gets to
  22/47 < 1/2 (and a parametric family of variants). See `results.md`.

---

## 4. Could it actually be false?

An honest assessment, in light of 2026:

**Reasons to take falsity seriously.**
* The entropy/coupling program — the only method that ever produced
  constant bounds — is *provably* stuck below 1/2: the relaxed problem it
  bounds is genuinely tight near $(3-\sqrt5)/2\approx 0.382$. The gap
  between 0.383 and 0.5 is not a matter of sharpening; it needs exact
  union-closure, and nobody knows how to use it globally.
* The conjecture is razor-thin at the top: sharp examples abound
  (powersets with $\emptyset$, and by §1.1 products thereof), and Theorem
  2.7 says a minimal counterexample would sit invisibly close to them
  ($\max \mathcal{E}_x = 1/2 - 1/(2m)$). Thin margins are exactly where
  long-believed conjectures have been failing lately: the 2026 disproofs
  of Erdős's unit-distance conjecture (by an OpenAI reasoning model, May
  2026) and of the Jacobian conjecture (via Claude, July 2026) both came
  from constructions humans had filed under "unlikely to help".
* Frankl-adjacent intuition has failed before: Sarvate–Renaud-style
  examples killed the natural local strengthenings, and the analogous
  "minimum-degree" strengthenings of the conjecture all have
  counterexamples.

**Reasons it is probably true anyway.**
* Verified exhaustively for universes ≤ 12 and for many structured classes
  (large families: $|F|\ge \tfrac23 2^n$ (Balla–Bollobás–Eccles); families
  of large sets (Reimer-type averages); lower semimodular lattices; graph-
  generated families). Counterexamples must be ≥ 47 sets on ≥ 13 points —
  and the search space there is astronomically larger than anything
  exhaustive methods reach, so absence of counterexamples is weak evidence,
  but the *structural* verifications (lattice classes) are not.
* The equivalent lattice formulation (every finite lattice has a
  join-irreducible below at most half the elements) holds for every lattice
  class where anyone has managed to test it, and lattice duality arguments
  have a rigidity that raw set families lack.
* Every relaxation that *is* false (approximate closure, minimum-degree
  versions) is false for reasons that visibly exploit the relaxation and
  die in the exact problem.

**Where a search should look, if anywhere.** Not at random: Theorem 2.7
says near-misses are indistinguishable from sharp examples until the last
moment, and our annealing confirms the landscape plateaus at $\mu\in\{0,1\}$.
The plausible attack is structural: fix small parameters
($|X| = 3$, $|\mathrm{Irr}|$ small), use §2 to pin the incidence structure
between criticals and irreducibles, and hand the residual constraint
system to a SAT/ILP solver on $n = 13, 14$ — i.e. search the *quotient*
described by the theorems, not the raw $2^{2^n}$ space. That is the
direction in which this file's theorems are pointed.

---

## References

* H. Bruhn, O. Schaudt, *The journey of the union-closed sets conjecture*
  (survey), [arXiv:1309.3297](https://arxiv.org/abs/1309.3297).
* J. Gilmer, *A constant lower bound for the union-closed sets conjecture*
  (2022); discussion: [Gil Kalai's blog](https://gilkalai.wordpress.com/2022/11/17/amazing-justin-gilmer-gave-a-constant-lower-bound-for-the-union-closed-sets-conjecture/).
* R. Alweiss, B. Huang, M. Sellke, *Improved lower bound for the
  union-closed sets conjecture*, [arXiv:2211.11731](https://arxiv.org/abs/2211.11731).
* Z. Chase, S. Lovett, *Approximate union closed conjecture*; W. Sawin,
  *An improved lower bound...*; L. Yu, *Dimension-free bounds*,
  [arXiv:2212.00658](https://arxiv.org/abs/2212.00658); S. Cambie, *Better
  bounds via the entropy approach*, [arXiv:2212.12500](https://arxiv.org/abs/2212.12500);
  J. Liu, *Conditionally IID coupling*, [arXiv:2306.08824](https://arxiv.org/abs/2306.08824).
* I. Roberts, J. Simpson, *A note on the union-closed sets conjecture*,
  Australas. J. Combin. 47 (2010),
  [pdf](https://ajc.maths.uq.edu.au/pdf/47/ajc_v47_p265.pdf).
* I. Bošnjak, P. Marković (n ≤ 11); B. Vučković, M. Živković (n ≤ 12),
  *The 12-element case of Frankl's conjecture*.
* D. Sarvate, J.-C. Renaud; B. Poonen, *Union-closed families*, JCTA 59
  (1992) — FC-family characterization.
* 2026 LLM disproofs referenced in §4: unit-distance —
  [Gil Kalai's blog, May 2026](https://gilkalai.wordpress.com/2026/05/21/amazing-erdos-unit-distance-problem-was-disproved-it-was-achieved-by-ai/),
  [arXiv:2605.20695](https://arxiv.org/abs/2605.20695); Jacobian —
  [A. Gallagher's writeup, July 2026](https://alexisgallagher.com/posts/2026/jacobianfun/).
