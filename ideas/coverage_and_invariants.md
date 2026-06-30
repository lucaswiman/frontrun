# Coverage Analysis, Syntax Analysis, and Invariant Discovery for frontrun

**Status:** Research/proposal document.

This document investigates two questions:
1. How can syntax analysis or coverage analysis supplement DPOR and bytecode fuzzing to find more bugs faster?
2. What invariants beyond "does not crash" and "does not deadlock" can be automatically checked?

---

## Part 1: Coverage and Syntax Analysis for Exploration Guidance

### 1.1 The problem: wasted exploration

Both `frontrun.explore(strategy="dpor")` and `frontrun.explore_random()` can spend many executions
exploring interleavings that exercise the same program behavior. DPOR already
deduplicates at the Mazurkiewicz trace level (same partial order = same trace),
but two *distinct* traces can still produce identical program states if the
racing accesses happen to be commutative at the application level. Bytecode
fuzzing is worse: it has no deduplication at all, relying on Hypothesis to
generate diverse schedules by chance.

**Concrete example:** A program with 3 threads each incrementing a counter
under a lock has many distinct Mazurkiewicz traces (different lock acquisition
orders), but they all produce the same final state. DPOR explores all of them;
coverage guidance could recognize they're equivalent and skip redundant ones.

### 1.2 Coverage metrics for concurrency testing

Standard code coverage (line, branch) is nearly useless for concurrency testing —
the same lines execute under every interleaving. We need **concurrency-aware
coverage metrics**:

#### 1.2.1 Reads-from coverage (from RFF, ASPLOS 2024)

**What it captures:** For each read of a shared variable, which write did it
observe? The set of (read, observed-write) pairs forms the "reads-from
relation." Two executions with identical reads-from relations will compute
identical results (assuming deterministic thread-local computation).

**How to implement in frontrun:** The DPOR engine already tracks per-object
access histories (`ObjectState` in `crates/dpor/src/object.rs`). After each
execution, compute a fingerprint:

```
fingerprint = hash(sorted([
    (thread_id, object_key, "read", last_writer_thread, last_writer_step)
    for each read access
]))
```

The `last_writer_thread` and `last_writer_step` are available from the vector
clock infrastructure — the engine knows which write each read "sees" because
it tracks the most recent write to each object per thread.

**Impact:** Use the fingerprint as a coverage signal:
- In DPOR: when recent traces produce duplicate fingerprints, bias wakeup tree
  exploration toward branches that reverse different races (complements the
  search strategies in `crates/dpor/src/path.rs`).
- In bytecode fuzzing: feed fingerprint novelty back to Hypothesis as a
  coverage target. Hypothesis's `target()` function accepts a float score;
  use `len(new_fingerprints) / len(all_fingerprints)` as the score.

**Expected benefit:** The RFF paper reports 1.5-3x more bugs found per unit
time compared to PCT and POS, primarily because fingerprint-guided exploration
avoids re-exploring behaviorally equivalent interleavings.

#### 1.2.2 Conflict-pair coverage

**What it captures:** Which pairs of conflicting accesses have been explored in
both orderings? A conflict pair is `(thread_A writes X, thread_B reads X)` or
`(thread_A writes X, thread_B writes X)`. DPOR guarantees that every
*reachable* conflict pair ordering is eventually explored, but it doesn't
prioritize unexplored pairs.

**How to implement:** Maintain a set of explored conflict pair orderings:

```rust
// In engine.rs, after each execution:
explored_pairs: HashSet<(ObjectId, usize, usize, AccessKind, AccessKind)>
// (object, first_thread, second_thread, first_kind, second_kind)
```

When selecting which wakeup tree branch to explore next, prefer branches whose
race reversal involves an unexplored conflict pair ordering. This is a
refinement of the `conflict-first` search strategy already in `path.rs`.

**Expected benefit:** Focuses exploration on the most "novel" parts of the
interleaving space. Particularly useful with `stop_on_first=True` and bounded
execution budgets.

#### 1.2.3 Synchronization-pattern coverage

**What it captures:** The sequence of synchronization operations (lock acquire,
lock release, condition wait/notify, I/O) per thread, abstracted away from
specific data values. Two executions with the same sync pattern but different
data accesses between syncs are unlikely to find new bugs.

**How to implement:** The `SyncEvent` enum in `engine.rs` already categorizes
events. Build a per-execution sync signature:

```
sync_sig = hash(tuple(
    (thread_id, event_type, lock_id_or_resource)
    for each sync event in schedule order
))
```

**Expected benefit:** Cheap to compute, catches the common case where DPOR
explores many interleavings that differ only in the ordering of independent
non-conflicting operations between synchronization points.

### 1.3 Syntax analysis for smarter access tracking

#### 1.3.1 Static race detection to seed DPOR

**Idea:** Before running any executions, statically analyze the thread
functions to identify *potential* shared accesses. This can:

1. **Pre-populate the object set** so the first DPOR execution already knows
   which objects to watch, reducing the "warm-up" cost.
2. **Identify definitely-local variables** that can be excluded from tracing,
   reducing overhead.
3. **Estimate conflict density** to choose the right search strategy
   automatically.

**Implementation approach — AST analysis:**

```python
import ast

def find_shared_accesses(func) -> set[str]:
    """Find attribute names that could be shared accesses."""
    tree = ast.parse(inspect.getsource(func))
    accesses = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            # self.x patterns where self is a parameter
            if isinstance(node.value, ast.Name) and node.value.id == func.__code__.co_varnames[0]:
                accesses.add(node.attr)
    return accesses
```

This is sound (conservative) for the common case where shared state is accessed
through `self.attr` on the setup object. It misses aliased accesses
(`x = shared_obj; x.value`) but those are caught at runtime by the shadow
stack.

**What this buys us:**
- If static analysis finds that two threads access disjoint attribute sets,
  we know there are zero races and can skip DPOR entirely (report
  `property_holds=True` immediately).
- If it finds that all shared accesses are behind lock acquisitions
  (detectable via `with lock:` context managers in the AST), we can narrow
  DPOR to explore only lock-ordering interleavings.
- The analysis runs once per test, taking microseconds, vs. milliseconds per
  DPOR execution.

#### 1.3.2 Bytecode analysis for access classification

The shadow stack in `dpor.py` (lines 114-135) works by mirroring the CPython
evaluation stack to track which object is being accessed by `LOAD_ATTR` /
`STORE_ATTR`. This is a runtime cost paid at every opcode.

**Optimization via static bytecode analysis:** Pre-analyze each code object
to classify instructions:

```python
# Already partially done in _get_instructions() (dpor.py:153-173)
# Extend to classify:
INTERESTING_OPS = {
    "LOAD_ATTR", "STORE_ATTR",           # shared memory
    "LOAD_GLOBAL", "STORE_GLOBAL",       # module-level shared state
    "STORE_SUBSCR", "DELETE_SUBSCR",      # container mutations
    "CALL_FUNCTION", "CALL_METHOD",       # potential side effects
}
```

For each code object, build a bitmap of which offsets contain interesting ops.
The trace callback can then skip the shadow stack update for non-interesting
instructions, reducing overhead by 60-80% (most instructions are arithmetic,
locals, or control flow).

**This is already partially implemented** — the `_get_instructions` cache in
`dpor.py` maps offsets to `dis.Instruction`. The optimization is to precompute
a fast-path "skip" set and avoid the dict lookup for boring opcodes entirely.

#### 1.3.3 Call-graph analysis for I/O prediction

The `_io_detection.py` module patches `socket.socket` and `builtins.open` to
detect I/O at runtime. Static call-graph analysis could predict which threads
perform I/O before execution:

- Parse thread functions' ASTs for `import` statements and function calls
- Follow the call graph to identify functions that reach `socket.*`,
  `open()`, `requests.*`, `psycopg2.*`, etc.
- Tag threads as "I/O-performing" vs "compute-only"

**Benefit:** I/O-performing threads are more likely to have high-latency
operations that interact with the LD_PRELOAD interception layer. Knowing this
in advance lets the scheduler allocate more generous timeouts for those threads
and prioritize exploring I/O interleavings (which are often where production
bugs lurk).

### 1.4 Coverage-guided bytecode fuzzing

The random bytecode explorer (`frontrun.explore_random()` in `bytecode.py`) uses
Hypothesis to generate random round-robin schedules. Currently, schedule
generation is blind — Hypothesis has no feedback about which schedules are
"interesting."

**Proposal: greybox concurrency fuzzing via `hypothesis.target()`:**

```python
from hypothesis import target

# Inside explore_random, after each execution:
fingerprint = compute_reads_from_fingerprint(execution_trace)
if fingerprint not in seen_fingerprints:
    seen_fingerprints.add(fingerprint)
    target(len(seen_fingerprints), label="unique_behaviors")
```

Hypothesis will then bias schedule generation toward schedules that produce
new fingerprints, effectively turning blind random testing into greybox fuzzing.

**Complementary signals to feed back:**
- Number of distinct conflict pairs exercised (more = better coverage)
- Maximum stack depth at context switches (deeper = more complex interleavings)
- Whether a new synchronization pattern was observed
- Whether a thread blocked on a lock (indicates contention, which is where bugs hide)

**Expected improvement:** The literature (RFF, ASPLOS 2024) shows 2-5x
improvement in bug-finding rate when coverage feedback is added to random
concurrency testing. The improvement is largest for programs with many
interleavings but few distinct behaviors.

---

## Part 2: Invariant Discovery — Beyond "Does Not Crash" and "Does Not Deadlock"

frontrun currently checks two implicit invariants (no crash, no deadlock via
`WaitForGraph` cycle detection in `_deadlock.py`) plus one explicit invariant
(the user-supplied `invariant` callable). For SQL workloads, it additionally
classifies anomalies via the DSG in `_sql_anomaly.py` (lost update, write
skew, dirty read, non-repeatable read, write-write conflict).

The problem: writing correct invariants is hard. Users often write trivially
weak invariants (`lambda state: True`) or overly specific ones that encode
implementation details. We should provide **general-purpose invariants** that
catch real bugs without requiring domain knowledge.

### 2.1 Invariants we can check automatically (no user input)

#### 2.1.1 Serializability (schedule equivalence)

**Invariant:** The final state under any interleaving must equal the final
state produced by *some* sequential execution of the same threads.

**How it works:** Before exploring interleavings, run each thread function
sequentially in every possible order (N! orderings for N threads; feasible for
2-4 threads). Collect the set of "valid" final states. Then, for each
interleaved execution, check if the final state matches any valid sequential
state.

```python
def check_serializability(setup, threads, state_hash):
    """Check that interleaved result matches some sequential execution."""
    from itertools import permutations
    valid_states = set()
    for perm in permutations(range(len(threads))):
        s = setup()
        for i in perm:
            threads[i](s)
        valid_states.add(state_hash(s))
    return valid_states  # interleaved result must be in this set
```

**Why this is powerful:** This is the gold standard concurrency correctness
criterion. It catches atomicity violations, lost updates, and ordering bugs
without the user needing to know the expected final value. The user only needs
to provide a `state_hash` function (or we can use `repr()` / `pickle` as a
default).

**Cost:** N! sequential executions up front (6 for 3 threads, 24 for 4). This
is cheap compared to the exponential interleaving space that DPOR explores.

**Limitation:** Only works when thread functions are deterministic. If a thread
function has internal randomness or time-dependence, the sequential baseline
varies. Detectable: run each sequential ordering twice and check consistency.

#### 2.1.2 Linearizability (for data structure methods)

**Invariant:** Each operation on a concurrent data structure must appear to
take effect atomically at some point between its invocation and response.

**How it works:** Record the invocation/response history of each method call:

```
Thread 0: invoke push(1) at t=0, respond at t=5
Thread 1: invoke push(2) at t=2, respond at t=7
Thread 0: invoke pop() at t=8, respond with 2 at t=10
```

Then check if there exists a sequential ordering of these operations that
(a) is consistent with the real-time ordering (if op A completed before op B
started, A must come before B), and (b) is a valid sequential execution of the
data structure.

**Implementation:** This is the Wing-Gong linearizability checker (1993). For
small histories (< 20 operations), brute-force enumeration works. For larger
ones, use the Lowe approach (2017) or the P-compositionality optimization.

**Where it fits in frontrun:** Add a `LinearizabilityChecker` that wraps a
data structure, records method calls via `__getattr__` interception, and
checks at the end of each execution. This would be an opt-in invariant:

```python
import frontrun

result = frontrun.explore(
    setup=lambda: LinearizabilityChecker(ConcurrentQueue()),
    workers=[lambda q: q.push(1), lambda q: q.push(2)],
    invariant=lambda q: q.is_linearizable(),
)
```

#### 2.1.3 Determinism (same input → same output)

**Invariant:** All interleavings produce the same final state.

This is a stronger (and often too strong) version of serializability, but it's
trivially checkable: hash the state after the first execution, then compare on
all subsequent ones. Useful for programs that *should* be fully synchronized.

```python
# Inside the exploration loop:
if first_state_hash is None:
    first_state_hash = state_hash(state)
elif state_hash(state) != first_state_hash:
    # Nondeterminism detected — two interleavings produce different results
    report_bug(...)
```

**When to use:** Good default for testing idempotent operations, caches, and
read-only shared state. Not suitable when different orderings legitimately
produce different results (e.g., concurrent counter without synchronization
is *expected* to have multiple valid outcomes).

#### 2.1.4 Absence of data races (even "benign" ones)

**Invariant:** No two threads access the same memory location concurrently
where at least one access is a write, unless mediated by synchronization.

The DPOR engine already detects races (that's how it builds the wakeup tree).
But currently races are used only for exploration guidance, not reported as
bugs. Many codebases treat *any* unsynchronized shared access as a defect
(the C/C++ memory model makes data races undefined behavior; Python is more
forgiving but races still indicate logic errors).

**Proposal:** Add a `detect_races=True` option to `frontrun.explore(strategy="dpor")` that
reports all detected races as findings, even if the user-supplied invariant
passes. Each race would include:
- The two conflicting accesses (thread, object, attribute, read/write)
- Source locations (from the trace recorder)
- Whether the race is "benign" (both orderings produce the same state) or
  "harmful" (different orderings produce different states)

This is essentially ThreadSanitizer (TSan) for Python, but using the
deterministic DPOR scheduler instead of sampling. Since DPOR already tracks
this information via vector clocks, the implementation cost is minimal.

#### 2.1.5 Lock discipline violations

**Invariant:** Locks are always acquired in a consistent global order across
all threads.

The `WaitForGraph` in `_deadlock.py` already detects deadlocks (cycles in the
wait-for graph). But lock-ordering violations can be detected *before* a
deadlock occurs: if thread A acquires lock 1 then lock 2, and thread B acquires
lock 2 then lock 1, that's a potential deadlock even if it doesn't happen in
this particular interleaving.

**Implementation:** Track the per-thread lock acquisition order across all
executions. Build a global lock-order graph. If the graph contains a cycle,
report it as a potential deadlock, even if no execution deadlocked.

```python
# lock_order_graph: dict[int, set[int]]  # lock_id -> set of locks acquired while holding it
for thread_lock_sequence in all_thread_lock_sequences:
    for i, lock_a in enumerate(thread_lock_sequence):
        for lock_b in thread_lock_sequence[i+1:]:
            lock_order_graph.setdefault(lock_a, set()).add(lock_b)

# Check for cycles in lock_order_graph
```

**Benefit:** Catches potential deadlocks that DPOR might not trigger because
they require specific preemption patterns. This is what Java's `jcstress` and
Go's race detector do.

### 2.2 Domain-specific invariants (with minimal user input)

#### 2.2.1 Conservation laws

**Invariant:** A numeric quantity is conserved across all threads.

Example: in a bank transfer, the total balance across all accounts must remain
constant. The user specifies a `conserved_quantity` function:

```python
import frontrun

result = frontrun.explore(
    setup=lambda: BankAccounts(a=100, b=100),
    workers=[
        lambda s: transfer(s, from_='a', to='b', amount=50),
        lambda s: transfer(s, from_='b', to='a', amount=30),
    ],
    invariant=lambda s: s.balance('a') + s.balance('b') == 200,
)
```

This is already expressible as a regular invariant, but we could provide a
`conserved(lambda s: s.total_balance())` helper that:
1. Computes the quantity before and after each execution
2. Reports the exact moment the conservation law breaks (which instruction
   caused the total to change), not just the final state

#### 2.2.2 Monotonicity

**Invariant:** A value only increases (or only decreases) over time.

Useful for: sequence numbers, timestamps, log-structured data, version
counters. Detectable by instrumenting attribute writes and checking that the
new value compares correctly to the old value.

```python
import frontrun

result = frontrun.explore(
    setup=lambda: VersionedStore(),
    workers=[...],
    invariant=monotonic(lambda s: s.version, direction="increasing"),
)
```

#### 2.2.3 No stale reads

**Invariant:** A thread that reads a value written by another thread always
sees the most recent write (not an older one).

This catches the "ABA problem" and stale-cache bugs. Detectable by
instrumenting reads to record the observed value and comparing against the
write history maintained by the DPOR engine.

#### 2.2.4 Idempotency

**Invariant:** Executing a function twice produces the same state as executing
it once.

Useful for testing retry logic, at-least-once delivery handlers, and
crash-recovery code. Implementable as a meta-invariant:

```python
def idempotent(func):
    def check(state):
        s1 = copy.deepcopy(state)
        func(s1)
        s2 = copy.deepcopy(state)
        func(s2); func(s2)
        return state_eq(s1, s2)
    return check
```

### 2.3 Inferred invariants (fully automatic)

#### 2.3.1 Daikon-style likely invariants

**Idea:** Run the sequential executions (from §2.1.1) and use dynamic invariant
detection (Daikon, Goldstein et al. 2024) to infer likely properties of the
shared state:

- `counter.value >= 0` (non-negativity)
- `len(queue.items) <= queue.max_size` (capacity bound)
- `account.balance == old_balance - transfer_amount` (relational)

Then check these inferred invariants under concurrent execution. Any invariant
that holds sequentially but fails concurrently is a concurrency bug.

**Implementation path:**
1. Instrument the shared state object with `__setattr__` / `__getattr__`
   hooks to record all value transitions
2. After sequential runs, feed the traces to a simple invariant inferrer
   (check: non-negative, non-null, bounded, monotonic, value-in-set)
3. Add inferred invariants as additional checks during DPOR exploration

**Practical considerations:**
- False positives: an inferred invariant might not be a true invariant, just
  something that happened to hold in sequential runs. Mitigation: require the
  invariant to hold across all N! sequential orderings, not just one.
- Performance: invariant checking adds overhead per execution. Keep invariants
  simple (comparison operators, set membership) so checking is O(1).

#### 2.3.2 Differential testing across interleavings

**Idea:** Instead of checking a fixed invariant, compare the *distribution*
of outcomes across interleavings. Flag outlier outcomes.

For example, if 99 out of 100 interleavings produce `counter.value == 2` and
one produces `counter.value == 1`, the outlier is likely a bug — even without
an explicit invariant saying the value should be 2.

**Implementation:**

```python
from collections import Counter
outcomes = Counter()
for execution in dpor_executions:
    outcomes[state_hash(execution.final_state)] += 1

# Flag outcomes that appear in < 5% of executions
for state, count in outcomes.items():
    if count / total < 0.05:
        report_suspicious_outcome(state, count, total)
```

**When this works well:** Programs where the "correct" behavior is the common
case and bugs are rare interleavings. This is true for most well-synchronized
programs with a few race conditions.

**When it fails:** Programs with many valid outcomes (e.g., concurrent
insertions into an unordered set — many final orderings are valid). Mitigation:
combine with serializability checking (§2.1.1) so only outcomes that differ
from all sequential orderings are flagged.

#### 2.3.3 Thread-local invariant inference

**Idea:** Each thread should see a consistent view of shared state within
each "critical section" (region between synchronization points). Detect
violations by recording the values read by each thread and checking for
internal consistency.

Example: if thread A reads `account.balance = 100` and then
`account.pending = 5`, the values should be consistent (balance + pending =
total). If another thread modifies `balance` between the two reads, thread A
sees an inconsistent snapshot.

**Implementation:** At each scheduling point, snapshot the values visible to
the active thread. Check that within a synchronization-free region, the
snapshot is consistent (values don't change between reads unless the thread
itself wrote them).

---

## Part 3: Synthesis — Putting It Together

### 3.1 Recommended implementation priority

| Priority | Feature | Effort | Impact | Depends on |
|----------|---------|--------|--------|------------|
| **P0** | Reads-from fingerprinting (§1.2.1) | Medium | High | Engine changes |
| **P0** | ~~Serializability checking (§2.1.1)~~ | Low | High | None | **DONE** |
| **P1** | ~~Race reporting mode (§2.1.4)~~ | Low | Medium | Already tracked | **DONE** |
| **P1** | Coverage-guided Hypothesis (§1.4) | Low | Medium | §1.2.1 |
| **P1** | Lock-order violation detection (§2.1.5) | Low | Medium | WaitForGraph exists |
| **P2** | Static race pre-analysis (§1.3.1) | Medium | Medium | AST parsing |
| **P2** | Differential outcome testing (§2.3.2) | Low | Medium | State hashing |
| **P2** | Bytecode skip-set optimization (§1.3.2) | Low | Perf only | Instruction cache exists |
| **P3** | Linearizability checking (§2.1.2) | High | High | Method interception |
| **P3** | Daikon-style inference (§2.3.1) | High | Medium | Instrumentation |

### 3.2 How these compose

The coverage and invariant improvements are orthogonal and composable:

```
                    ┌─────────────────────┐
                    │   User's test case   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Static pre-analysis │  (§1.3.1: estimate conflicts,
                    │  (AST + bytecode)    │   choose strategy, skip locals)
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
    ┌─────────▼──────┐  ┌─────▼───────┐  ┌─────▼───────────┐
    │ Sequential runs │  │    DPOR     │  │ Bytecode fuzzing │
    │ (all N! orders) │  │  (engine)   │  │  (Hypothesis)    │
    └─────────┬──────┘  └─────┬───────┘  └─────┬───────────┘
              │               │                 │
              │         ┌─────▼───────┐   ┌─────▼───────────┐
              │         │  Coverage   │   │  Coverage        │
              │         │  feedback   │   │  feedback via    │
              │         │ (§1.2.1-3)  │   │  target() (§1.4) │
              │         └─────┬───────┘   └─────┬───────────┘
              │               │                 │
              └───────────────┼─────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Invariant checking │
                    │                   │
                    │ • Serializability  │ (compare to sequential baselines)
                    │ • Race reporting   │ (from DPOR vector clocks)
                    │ • Lock ordering    │ (from WaitForGraph)
                    │ • User invariant   │ (existing)
                    │ • SQL anomaly      │ (existing DSG)
                    │ • Differential     │ (outlier detection)
                    └───────────────────┘
```

### 3.3 Key insight: coverage reduces traces, invariants increase bugs found

These two improvements attack different parts of the problem:

- **Coverage analysis** reduces the number of executions needed to find a bug
  by avoiding redundant exploration. It doesn't find new *types* of bugs, but
  finds the same bugs faster.

- **Invariant enrichment** increases the number of bugs found per execution
  by checking more properties. A single execution that passes the user's
  invariant might still violate serializability, contain a data race, or
  exhibit a lock-ordering violation.

Together, they multiply: fewer executions × more bugs per execution = much
better bug-finding ROI.

---

## Implementation status

### Implemented

**Serializability checking (§2.1.1)** — `serializable_invariant` parameter added to all
exploration entry points (`frontrun.explore` for both sync and async DPOR, `frontrun.explore_random` and
`frontrun.explore_async_random` for the bytecode shufflers). When enabled, all N! sequential
orderings are run before exploration to compute valid final states. Each interleaved
execution's final state is checked against this set. Accepts `True` (uses `repr()` as
hash) or a callable `state_hash` function. Default off.

**Race reporting / error_on_any_race (§2.1.4)** — `error_on_any_race` parameter added to
the DPOR exploration paths (`frontrun.explore(strategy="dpor")`, sync and async). When enabled, any
unsynchronized data race detected by the DPOR vector clock engine is treated as a test
failure, even if the user-supplied invariant passes. Filters out container-level
(`report_first_access`) and lock-synthetic races to avoid false positives — only
attribute-level and I/O races are flagged. For non-DPOR shufflers, passing
`error_on_any_race=True` raises `ValueError` since they lack race detection. Default off.

Both options set `result.races_detected` on `InterleavingResult` when races are found.

## References

- Wolff et al., "Greybox Fuzzing for Concurrency Testing", ASPLOS 2024
- Wing & Gong, "Testing and Verifying Concurrent Objects", JPDC 1993
- Lowe, "Testing for Linearizability", CPP 2017
- Ernst et al., "Dynamically Discovering Likely Program Invariants" (Daikon), IEEE TSE 2001
- Goldstein et al., "The Sparse Synchronization Model for Spec Inference", OOPSLA 2024
- Adve & Hill, "A Unified Formalization of Four Shared-Memory Models", IEEE TPDS 1993
- Serebryany & Iskhodzhanov, "ThreadSanitizer — Data Race Detection in Practice", WBIA 2009
