# Randomized Trace Exploration for DPOR

## Problem statement

Our current DPOR implementation systematically explores Mazurkiewicz trace
equivalence classes using Optimal DPOR (Abdulla et al., JACM 2017). This is
**complete** — it will eventually visit every equivalence class — but exploration
order is deterministic (deepest-backtrack-first DFS). If bugs cluster in a
region of the trace space that DFS reaches late, we waste time exploring "boring"
neighborhoods first.

The concern: when races are **dense** (many equivalence classes, e.g., dining
philosophers with 4,076 traces), and we use `stop_on_first=True`, our DFS
ordering may systematically miss the buggy traces. We'd like to explore more
**varied** traces early without losing the efficiency of trace caching and sleep
set propagation.

This document surveys the literature on randomized/hybrid approaches to
partial-order exploration and proposes concrete strategies for frontrun.

---

## Literature survey

### 1. PCT — Probabilistic Concurrency Testing (Burckhardt et al., ASPLOS 2010)

**Core idea**: Assign random priorities to threads. At `d-1` randomly chosen
"priority change points", reassign a thread's priority. With `n` threads and
`k` steps, a bug of depth `d` is found with probability >= `1/(n * k^(d-1))`.

**Key insight**: Bug **depth** — the minimum number of scheduling constraints
needed to expose a bug — is usually small (1-2 for atomicity violations, 2 for
deadlocks). PCT exploits this by using very few random decisions.

**Relevance to us**: PCT is completely orthogonal to DPOR — it samples
individual schedules, not equivalence classes. But the depth characterization
is useful: if we can bias our DPOR exploration toward traces that differ in
"deep" ways (different lock orderings, different read-write interleavings),
we'd cover the interesting space faster.

**Ref**: [A Randomized Scheduler with Probabilistic Guarantees of Finding Bugs](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos277-pct.pdf)

### 2. POS — Partial Order Sampling (Yuan et al., CAV 2018)

**Core idea**: Assign each event a random priority. At each step, execute the
highest-priority enabled event. When event `e1` races with `e2`, **reassign**
`e2` a fresh random priority. This ensures that each partial order (trace) is
sampled with non-trivial probability.

**Why reassignment matters**: Without it, once a priority ordering is chosen,
racing events always resolve the same way. Reassignment after each race means
future races are resolved independently of past ones.

**Probability bound**: For a program with `M` Mazurkiewicz traces, each trace
is sampled with probability >= `1/M` (uniform over traces, not schedules).

**Relevance to us**: POS achieves partial-order-aware sampling. The priority
reassignment mechanism could be adapted to randomize which wakeup tree branch
we explore next in our DPOR engine, without losing soundness.

**Ref**: [Partial Order Aware Concurrency Sampling](https://link.springer.com/chapter/10.1007/978-3-319-96142-2_20)

### 3. taPCT — Trace-Aware PCT (Ozkan et al., OOPSLA 2019)

**Core idea**: Combine PCT's depth-bounded sampling with POS's partial-order
awareness. Instead of choosing priority change points from ALL events, choose
them only from **racy events** (events involved in data races). This gives
probability >= `1/(n_racy^(d-1))` instead of `1/(k^(d-1))`, a huge improvement
since `n_racy << k`.

**Relevance to us**: We already track racing events (via the Rust engine's
vector clocks). We could use race information to guide which wakeup tree
branches to explore first — branches that reverse races involving more
"interesting" events (lock acquisitions, shared writes) get higher priority.

**Ref**: [Trace Aware Random Testing for Distributed Systems](https://dl.acm.org/doi/10.1145/3360606)

### 4. RFF — Reads-From Fuzzer (Wolff et al., ASPLOS 2024)

**Core idea**: Apply greybox fuzzing to concurrency testing. Use the
**reads-from relation** (which write does each read see?) as the coverage
metric. Mutate abstract schedules (sets of reads-from constraints) and use
feedback (did this mutation produce a new reads-from equivalence class?) to
guide exploration.

**Key mechanisms**:
- **Abstract schedules**: Sets of positive/negative reads-from constraints
  (e.g., "read R must see write W1, must NOT see write W2")
- **Mutation**: Flip, add, or remove individual constraints
- **Feedback**: Track which reads-from equivalence classes have been seen;
  prioritize mutations that produce new classes
- **Full schedule control**: Unlike stress testing, RFF controls every
  scheduling decision, enabling fine-grained interleaving

**Relevance to us**: The "reads-from relation as coverage" idea maps directly
to our object access tracking. We could compute a fingerprint of each trace's
reads-from relation and use it to detect when DPOR is producing "similar"
traces, then bias exploration toward different neighborhoods.

**Ref**: [Greybox Fuzzing for Concurrency Testing](https://dl.acm.org/doi/10.1145/3620665.3640389)

### 5. SURW — Selectively Uniform Random Walk (Zhao et al., ASPLOS 2025)

**Core idea**: At each scheduling point, pick a thread with probability
proportional to its **remaining event count**. This achieves approximately
uniform sampling over interleavings for a selected subset of "interesting"
events.

**Key mechanism**: Take a pre-defined set of interesting events Δ (e.g., shared
memory accesses, lock operations) and per-thread counts. Weight each thread by
its remaining interesting events, so threads with more work left get scheduled
more often early on.

**Relevance to us**: The "interesting events" selection maps to our
`active_accesses` / `explored_accesses` infrastructure. We could weight wakeup
tree branches by the number of unseen scheduling decisions they'd exercise.

**Ref**: [Selectively Uniform Concurrency Testing](https://dl.acm.org/doi/10.1145/3669940.3707214)

### 6. POP — Parsimonious Optimal DPOR (Agarwal et al., CAV 2024)

**Core idea**: Optimal DPOR with polynomial (not exponential) memory. Uses a
compact "conflict detector" representation instead of explicit sleep sets.
Eagerly explores new executions from race reversals instead of accumulating
them.

**Relevance to us**: Not directly about randomization, but the eager
exploration strategy could combine well with randomized ordering — explore
races as they're discovered, in random order, rather than accumulating a
worklist.

**Ref**: [Parsimonious Optimal Dynamic Partial Order Reduction](https://arxiv.org/html/2405.11128)

---

## Proposals for frontrun

### Proposal A: Randomized wakeup tree ordering (low effort, high value)

**Idea**: Randomize the order in which wakeup tree branches are explored.
Currently, `step()` picks the min-index thread from the wakeup tree
(deterministic DFS). Instead, pick a random branch, optionally weighted by
heuristics.

**Implementation**:
1. Add a `randomize: bool` (or `seed: int | None`) parameter to `frontrun.explore()`
2. In `Path::step()`, instead of `wakeup.min_thread()`, use
   `wakeup.random_thread(rng)` — pick uniformly at random from the available
   wakeup tree children
3. Sleep sets and trace caching remain **unchanged** — they're orthogonal to
   exploration order. Each explored trace is still a distinct Mazurkiewicz
   equivalence class
4. With `stop_on_first=True`, this gives us randomized early termination:
   different seeds explore different parts of the trace space first

**Properties**:
- **Soundness**: Unaffected. We still only explore valid wakeup tree branches;
  sleep sets still prune equivalent traces. We just visit them in a different
  order.
- **Completeness**: If we explore all branches (no `stop_on_first`), we still
  visit every equivalence class. With `stop_on_first`, different seeds give
  different coverage.
- **Effort**: Small — only `step()` and `schedule()` need changes in the Rust
  engine. No Python-side changes.

**Heuristic weights** (optional, for biased exploration):
- **Race density**: Weight branches by the number of races they reverse.
  Branches that reverse more races are more likely to produce different behavior.
- **Thread diversity**: Prefer branches that schedule a thread we haven't
  scheduled recently. Avoids the "always schedule thread 0 first" bias.
- **Access novelty**: Weight by the number of unseen object access patterns in
  the branch's prefix. Uses trace cache data.

**Analogy**: This is like randomized DFS in a graph — same nodes visited
eventually, but the order varies, which matters when you stop early.

### Proposal B: Trace fingerprinting with coverage feedback (medium effort, high value)

**Idea**: Compute a lightweight fingerprint of each explored trace (based on
the reads-from relation or access ordering) and use it as coverage feedback.
If recent traces have similar fingerprints, inject randomness to break out
of the neighborhood.

**Implementation**:
1. After each execution, compute a **trace fingerprint**: hash of the sequence
   `(thread_id, object_key, access_kind)` at each scheduling point. This is
   essentially a reads-from signature.
2. Maintain a **coverage map**: set of seen fingerprints (or a bloom filter for
   memory efficiency).
3. Track a **novelty score**: fraction of recent traces (last N) that produced
   new fingerprints.
4. When novelty drops below a threshold:
   - **Shuffle wakeup tree order** (as in Proposal A)
   - **Skip the deepest backtrack** and try a shallower one (explore a
     different part of the tree)
   - **Insert random "chaos" branches**: at the current backtrack point, instead
     of the wakeup tree's suggestion, pick a random enabled thread and explore
     from there (this may re-explore an equivalence class, but breaks out of
     the neighborhood)

**Properties**:
- **Soundness**: The chaos branches may revisit equivalence classes (not
  optimal), but will never miss a bug. The fingerprint-guided skipping is a
  heuristic — if we skip a subtree, we lose completeness for that subtree but
  gain diversity.
- **Trade-off**: Completeness vs. diversity. With `stop_on_first=True`, this
  is usually the right trade-off.
- **Effort**: Medium. Fingerprinting is easy (hash the access trace). The
  coverage feedback loop requires changes to `next_execution()` in the Rust
  engine.

**Inspired by**: RFF's reads-from coverage feedback, adapted to work within
DPOR's systematic framework.

### Proposal C: Hybrid DPOR + random sampling (medium effort, medium value)

**Idea**: Interleave systematic DPOR exploration with random sampling runs.
Every K-th execution, instead of following the wakeup tree, run a POS-style
random schedule. Use the random run's races to seed new wakeup tree entries.

**Implementation**:
1. Every `K` executions (e.g., K=10), run a **random execution**:
   - Use POS-style priority scheduling: each thread gets a random priority,
     reassign on races
   - Report accesses to the engine as usual
   - Collect races from the random run
2. For each race found in the random run that would produce a new equivalence
   class (not in the sleep set), **inject** it into the wakeup tree at the
   appropriate position
3. Continue systematic DPOR exploration, now with the injected branches

**Properties**:
- **Completeness**: The systematic DPOR part remains complete. Random runs may
  discover races that DFS would reach late, effectively "teleporting" to
  distant parts of the trace space.
- **Overhead**: The random runs may re-explore known equivalence classes
  (wasted work). Mitigated by the sleep set check before injection.
- **Effort**: Medium. Requires a second scheduler mode (POS) alongside the
  DPOR scheduler. The injection mechanism needs care to maintain wakeup tree
  invariants.

**Inspired by**: Morpheus (ASPLOS 2020), which keeps a concise summary of
explored executions and uses it to guide future random explorations.

### Proposal D: Depth-biased backtrack selection (low effort, medium value)

**Idea**: Instead of always exploring the deepest backtrack point (DFS), use
a weighted selection that sometimes picks shallower backtracks. Shallower
backtracks produce more "globally different" traces (they diverge earlier),
while deeper backtracks produce locally different traces.

**Implementation**:
1. In `Path::step()`, when selecting which backtrack point to explore next,
   instead of always picking the deepest (current behavior):
   - Compute a weight for each backtrack point: `weight(pos) = alpha^(max_pos - pos)`
     where `alpha < 1` biases toward shallow, `alpha > 1` biases toward deep,
     `alpha = 1` is uniform
   - Sample from this distribution
2. Default `alpha` could adapt based on progress: start with `alpha > 1`
   (deep-first, like current DFS) and decrease over time to explore more diverse
   traces as the search matures

**Properties**:
- **Soundness**: Unaffected — all backtrack points are valid
- **Completeness**: If all branches are eventually explored, still complete
- **Effort**: Low — only the backtrack selection in `step()` needs changes

**Analogy**: This is like the explore/exploit trade-off in multi-armed bandits.
Early on, exploit (deep DFS, find bugs near the first trace quickly). Later,
explore (shallow backtracks, find bugs in distant parts of the space).

---

## Recommendation

**Start with Proposal A** (randomized wakeup tree ordering). It's the lowest
effort, doesn't compromise any DPOR guarantees, and directly addresses the
concern about DFS ordering bias. A `seed` parameter on `frontrun.explore()` would
let users run multiple randomized explorations in parallel for even better
coverage.

**Then add Proposal D** (depth-biased backtrack selection) as a complement.
Together, A+D randomize both "which thread to run at a backtrack point" and
"which backtrack point to visit next", giving good diversity across the entire
trace space.

**Proposal B** (trace fingerprinting) is the most interesting for the dense-race
case but requires more infrastructure. It would be a good follow-up once A+D
are in place and we can measure whether diversity is still insufficient.

**Proposal C** (hybrid DPOR+random) is the most disruptive and should only be
considered if the other approaches prove insufficient. The POS-style random
scheduler would need its own implementation and the injection mechanism is
subtle.

---

## Relation to existing `frontrun.explore_random()`

We already have a pure-random bytecode explorer (`frontrun.explore_random()` in
`bytecode.py`) that generates random round-robin schedules via Hypothesis.
This is complementary to — not competitive with — the proposals above:

- `frontrun.explore_random()` explores **random schedules** (many schedules per
  equivalence class, no trace-level deduplication)
- `frontrun.explore(strategy="dpor")` explores **equivalence classes** (one schedule per
  class, systematic)
- The proposals here add **randomized ordering** to the systematic exploration,
  getting the best of both worlds

The fingerprinting from Proposal B could potentially be shared: use the same
trace fingerprint to detect when `frontrun.explore_random()` is stuck in a
neighborhood, and inject a Hypothesis-generated schedule that targets a
different region.

---

## References

- Burckhardt et al., "A Randomized Scheduler with Probabilistic Guarantees of Finding Bugs", ASPLOS 2010
  ([PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/asplos277-pct.pdf))
- Yuan et al., "Partial Order Aware Concurrency Sampling", CAV 2018
  ([Springer](https://link.springer.com/chapter/10.1007/978-3-319-96142-2_20))
- Ozkan et al., "Trace Aware Random Testing for Distributed Systems", OOPSLA 2019
  ([ACM DL](https://dl.acm.org/doi/10.1145/3360606))
- Wolff et al., "Greybox Fuzzing for Concurrency Testing", ASPLOS 2024
  ([ACM DL](https://dl.acm.org/doi/10.1145/3620665.3640389),
   [PDF](https://abhikrc.com/pdf/ASPLOS24.pdf))
- Zhao et al., "Selectively Uniform Concurrency Testing", ASPLOS 2025
  ([ACM DL](https://dl.acm.org/doi/10.1145/3669940.3707214),
   [PDF](https://www.comp.nus.edu.sg/~umathur/papers/surw-asplos25.pdf))
- Agarwal et al., "Parsimonious Optimal Dynamic Partial Order Reduction", CAV 2024
  ([arXiv](https://arxiv.org/html/2405.11128))
- Sen, "Effective Random Testing of Concurrent Programs", ASE 2007 (RAPOS)
  ([PDF](https://people.eecs.berkeley.edu/~ksen/papers/fuzzpar.pdf))
- Yuan et al., "Effective Concurrency Testing for Distributed Systems", ASPLOS 2020 (Morpheus)
  ([PDF](https://www.cs.columbia.edu/~junfeng/papers/morpheus-asplos20.pdf))
- Abdulla et al., "Source Sets: A Foundation for Optimal DPOR", JACM 2017
  (see `ideas/abdulla_jacm2017/`)
