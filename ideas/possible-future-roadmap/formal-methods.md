# TLA+/Quint Integration Roadmap

## Status
No TLA+/Quint workflows are currently implemented. SQL conflict detection has TLA+ specs (38,016 states verified) but lacks runtime integration.

## Key Insight: Coding Agents as the Bridge

Coding agents (LLM-based) fundamentally change the formal methods equation. They can:
- Generate and maintain TLA+ specs from Python code (syncing as code evolves)
- Parse specs and translate counterexamples into reproducible frontrun schedules
- Translate TLA+ invariants into Python assertions
- Run TLC, interpret results, and report bugs without user ever seeing TLA+

**Result:** TLA+ becomes internal plumbing that an agent uses, like a compiler uses SSA. The developer-facing API is simply "test my concurrent code more thoroughly."

## Recommended Starting Path

### Priority 1: Foundation (Low effort, enables everything else)

#### 1.1 **Spec/Code Correspondence Linter** (Effort: Low)
Cross-reference a TLA+ spec against Python source and flag mismatches:
- Every TLA+ action has a `# frontrun:` marker
- Every frontrun marker has a TLA+ action
- TLA+ variables covered by state extractors
- PlusCal processes map to thread names

**Why:** Quick drift detection. Foundation for all other workflows. Works for both agent-generated and manually-written specs.

#### 1.2 **Invariant Assertion Bridge** (Effort: Low)
Translate TLA+ invariants (e.g., `TypeOK`, `MutualExclusion`) to Python assertion functions decorated with `@frontrun_invariant`. Frontrun calls these after every step, catching violations with full trace.

**Why:** Low barrier to entry. Immediate value even without full integration. Agents can mechanically translate common patterns (type invariants, mutual exclusion, deadlock freedom).

#### 1.3 **TLA+ Parsing Infrastructure** (Effort: Low)
Use `tree-sitter-tlaplus` (pip-installable, no JVM) to extract:
- Actions, processes, labels, variables, invariants
- Build `SpecMapping` data structure mapping TLA+ entities to frontrun annotations

**Why:** Programmatic foundation for all downstream workflows. Agent-facing plumbing. PlusCal specs map especially well (labels → markers, processes → threads).

### Priority 2: Core Integration (Medium effort, high impact)

#### 2.1 **Counterexample Replay** (Effort: Low–Medium)
When TLC finds an invariant violation, convert the trace to a frontrun `Schedule` and replay it against real Python code:
1. TLC finds counterexample trace (action sequence + states)
2. Agent maps TLA+ actions to frontrun markers
3. Build and execute `Schedule` with real Python executors
4. Confirm or refute bug in actual code

**Why:** Turns TLC's abstract violations into concrete reproduction tests. Full agent pipeline: spec → TLC → counterexample → frontrun schedule → bug confirmation → regression test.

#### 2.2 **Trace Validation** (Effort: Medium)
Emit frontrun execution traces and validate them against TLA+ specs:
1. Run frontrun interleaving, record trace of `(thread, marker, state_snapshot)` tuples
2. Translate to TLA+ `Trace` module (Cirstea/Kuppe/Merz framework, SEFM 2024)
3. TLC validates trace is a valid spec behavior
4. If validation fails, diagnose spec vs. code mismatch

**Why:** Closes the loop between formal model and real execution. Provides spec-guided test coverage guarantees: "if TLC verifies all behaviors up to depth N, and our traces match, we have real confidence."

#### 2.3 **Spec-Guided Schedule Generation** (Effort: Medium)
Replace random exploration with TLC enumeration:
1. Agent maintains TLA+ spec (or PlusCal, which is more ergonomic)
2. TLC enumerates all distinct behaviors (exhaustive BFS)
3. Convert each behavior to a frontrun `Schedule`
4. Execute all schedules against real Python code
5. Compare actual state at each step against TLC's predicted state

**Why:** Solves the "coverage gap" problem. TLC's symmetry reduction and state-space pruning are more sophisticated than basic DPOR. With agent maintaining the spec, essentially free exhaustive coverage.

**Relationship to DPOR:** TLC explores the model's state space (smaller, structured). DPOR explores the implementation's state space (larger, noisier). Outsourcing to TLC is a faster path to exhaustive coverage.

### Priority 3: Advanced (High effort, enables formal guarantees)

#### 3.1 **Refinement Checking** (Effort: High)
Verify the Python implementation refines the TLA+ spec:
1. Write high-level spec (coarse-grained actions like `Read`, `Write`)
2. Write lower-level spec refining it (fine-grained, matching frontrun markers)
3. TLC verifies lower spec refines higher (Refinement Mapping)
4. Frontrun trace validation confirms Python code conforms to lower spec
5. By transitivity: Python ⊨ high-level spec

**Why:** Gold standard of formal verification applied to real code. Separates concerns: high-level spec captures algorithm, low-level spec captures implementation granularity, frontrun confirms code matches. Agents maintain both as code evolves.

## Supporting Tools & Implementation

### TLA+ Parsing
- **Recommended:** `tree-sitter-tlaplus` (pip install, no JVM, error-tolerant CST)
- **Alternative:** Apalache `parse --output=result.json` (full semantic analysis, JVM required)

### TLC Integration
- **TLC installation:** Already available in most TLA+ environments
- **Invocation:** Subprocess calls or `modelator-py` wrapper
- **Output parsing:** Extract action sequences and state diffs from `.out` files or ITF (Informal Trace Format) JSON

### PlusCal Sweet Spot
PlusCal specs map to frontrun better than raw TLA+:
- PlusCal labels (atomic units) → frontrun markers
- PlusCal processes → frontrun threads
- `pc` variable (program counter) → schedule position

Agents can generate PlusCal more naturally than raw TLA+ because its imperative process/label structure mirrors actual concurrent Python code.

## Sequencing

Foundation first (1.1 linter + 1.3 parsing, then 1.2 invariant bridge), then counterexample
replay (2.1), trace validation (2.2), spec-guided schedules (2.3). Refinement checking (3.1)
only once 2.2/2.3 are solid.

## Expected Outcomes

- **Spec/code drift detection:** Automatic (linter)
- **Exhaustive coverage:** TLC-guided instead of random
- **Deterministic bug reproduction:** Counterexamples → frontrun schedules
- **Agent-driven workflow:** Developer writes Python + markers, agent handles TLA+
- **Formal guarantees:** With refinement checking, mathematical proof that code implements spec

## References

- Cirstea et al. ["Validating Traces of Distributed Programs Against TLA+ Specifications"](https://arxiv.org/abs/2404.16075) (SEFM 2024)
- Howard et al. ["Smart Casual Verification of CCF's Distributed Consensus"](https://arxiv.org/html/2406.17455v1) (NSDI 2025)
- [tree-sitter-tlaplus](https://github.com/tlaplus-community/tree-sitter-tlaplus)
- [modelator-py](https://github.com/informalsystems/modelator-py)
