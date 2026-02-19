# Frontrun + TLA+: Integration Brainstorm

## Overview

Frontrun controls and records concurrent execution schedules in Python. TLA+ is a
formal specification language for modeling and verifying concurrent/distributed systems.
The two are natural complements: TLA+ reasons about *what interleavings are possible
and correct*, while frontrun *forces specific interleavings in real code*.

### The Key Insight: Coding Agents as the Bridge

The traditional objection to formal methods is maintenance cost — keeping a TLA+ spec
in sync with evolving code is expensive human labor. **Coding agents fundamentally
change this equation.** An LLM-based agent can:

- Read Python code and generate a corresponding TLA+ spec
- Read a TLA+ spec and generate frontrun-annotated Python tests
- Keep specs and code in sync as either changes
- Run TLC, parse counterexamples, and translate them into frontrun schedules
- Translate TLA+ invariants into Python assertion functions

The developer never needs to learn or touch TLA+. TLA+ becomes an **internal reasoning
tool that the agent uses** — like how a compiler uses SSA, the user doesn't need to
know it's there. The human-facing API is just "test my concurrent code more
thoroughly."

This reframes every workflow below. Where it says "user writes a TLA+ spec," read
"agent generates and maintains a TLA+ spec." Where it says "convert counterexample to
schedule," read "agent does this automatically." The parsing infrastructure in section
9 is agent-facing plumbing, not a human API.

### The Natural Mapping

Frontrun markers are the Rosetta Stone — they're already named actions at
interleaving-relevant points, which is exactly what TLA+ models. The mapping between
`Step("t1", "read_value")` and the TLA+ action `Read(t1)` is almost syntactic.

---

## 1. Trace Validation: Check frontrun executions against TLA+ specs

**The idea:** Frontrun already controls and records execution schedules. Emit those
schedules as TLA+ traces and validate them against a spec using TLC.

**How it would work:**
1. Agent generates a TLA+ spec from the Python code and its frontrun markers
2. Frontrun runs an interleaving, recording a trace:
   `[(thread, marker, state_snapshot), ...]`
3. A `frontrun.tlaplus` module translates that trace into a TLA+ `Trace` module
   (following the Cirstea/Kuppe/Merz framework from SEFM 2024)
4. TLC validates that the trace is a valid behavior of the spec
5. If validation fails, the agent diagnoses whether the spec or code is wrong

**Why it's a natural fit:**
- Frontrun markers already name "actions" — these map directly to TLA+ actions
- The `Schedule` / `Step` data structures already encode the interleaving order
- `InterleavingResult` could be extended to capture state snapshots at each step
- An agent can generate the spec, run the validation, and interpret results without
  the developer ever seeing TLA+

**Sketch of the API:**
```python
from frontrun.tlaplus import TLAPlusValidator

validator = TLAPlusValidator(
    spec_file="Counter.tla",
    marker_to_action={
        "read_value": "Read",
        "write_value": "Write",
    },
    state_extractor=lambda obj: {"counter": obj.value},
)

# Run all interleavings and validate each trace against the spec
results = validator.explore_and_validate(
    executors={"t1": counter.increment, "t2": counter.increment},
    num_runs=100,
)
# results.violations -> list of traces that diverged from spec
```

**Effort:** Medium. Needs: trace serialization to TLA+, TLC invocation (via
`modelator-py` or subprocess), result parsing.

**Key references:**
- Cirstea, Kuppe, Loillier, Merz. ["Validating Traces of Distributed Programs Against TLA+ Specifications"](https://arxiv.org/abs/2404.16075) (SEFM 2024)
- MongoDB's [repl-trace-checker](https://github.com/mongodb-labs/repl-trace-checker) — validates server logs against `RaftMongo.tla`
- Microsoft CCF's [trace validation](https://arxiv.org/html/2406.17455v1) — prevented 6 bugs

---

## 2. Spec-Guided Schedule Generation: Use TLC to generate interesting interleavings

**The idea:** Instead of random exploration (`explore_interleavings`), use TLC to
enumerate *all* distinct behaviors of a TLA+ spec, then replay each one as a frontrun
`Schedule`.

**How it would work:**
1. Agent generates/maintains a TLA+ spec from the Python code
2. TLC model-checks the spec and emits all distinct traces (or traces violating an
   invariant)
3. Each trace is a sequence of `(action, state)` pairs
4. Agent maps actions back to `(thread, marker)` pairs to build frontrun `Schedule`s
5. Frontrun executes each schedule against real Python code
6. Agent compares actual state at each step against TLC's predicted state

**Why this is the killer feature:**
- Solves frontrun's "coverage gap" problem (noted in FUTURE_WORK.md) — TLC's
  exhaustive BFS replaces random exploration
- Provides *formal* coverage guarantees: if TLC finds no invariant violations for all
  behaviors up to depth N, and the implementation matches, you have real confidence
- This is essentially DPOR (also in FUTURE_WORK.md) but outsourced to TLC, which
  already does it well
- **With an agent maintaining the spec, the cost of this approach drops to near zero**
  — the developer just runs their tests and gets exhaustive coverage for free

**Relationship to DPOR:** TLC's symmetry reduction and state-space pruning are more
sophisticated than basic DPOR. With an agent maintaining the spec cheaply, this is a
faster path to exhaustive coverage than implementing DPOR natively in frontrun. DPOR
explores the *implementation's* state space (large, noisy). TLC explores the *model's*
state space (smaller, more structured). The agent bridges the two.

---

## 3. Refinement Checking: Verify the Python implementation refines the TLA+ spec

**The idea:** A TLA+ spec is "abstract" — it might model a counter with atomic `Read`
and `Write` actions. The Python implementation is "concrete" — it has bytecodes between
the read and write. Frontrun can bridge the refinement gap.

**How it would work:**
1. Agent writes a high-level TLA+ spec with coarse actions
2. Agent writes a lower-level TLA+ spec that refines it (with finer actions matching
   frontrun markers)
3. TLC verifies the refinement (`LowSpec => HighSpec`)
4. Frontrun trace validation (workflow 1) verifies the Python code conforms to
   `LowSpec`
5. By transitivity: Python code refines the high-level spec

**Why it matters:** This is the "gold standard" of formal verification applied to real
code. It separates concerns cleanly: the high-level spec captures what the algorithm
*should* do, the low-level spec captures the implementation's atomicity grain, and
frontrun confirms the real code matches. An agent can maintain both spec levels as the
code evolves.

---

## 4. Invariant Assertion Bridge: Express TLA+ invariants as Python assertions

**The idea:** TLA+ invariants are predicates over state. Translate them to Python and
check them at every marker during frontrun execution.

**How it would work:**
```python
# TLA+ invariant: \A t1, t2 \in Threads:
#   pc[t1] = "write" /\ pc[t2] = "write" => FALSE
# (No two threads in the write section simultaneously)

# Agent-generated Python translation:
@frontrun_invariant
def mutual_exclusion(state):
    writers = [t for t in state.threads if state.pc[t] == "write"]
    assert len(writers) <= 1, "Mutual exclusion violated!"
```

Frontrun could call these invariants after every step in the schedule, catching
violations immediately with a full trace for reproduction.

**Agent workflow:** The agent reads the TLA+ spec (which it also wrote), extracts
invariants, and generates Python assertion functions. When the spec changes, the agent
regenerates the assertions. For common invariant patterns (type invariants, mutual
exclusion, no-deadlock), the translation is mechanical. For complex temporal properties,
the agent can approximate or flag for human review.

---

## 5. Bidirectional State Machine DSL

**The idea:** Create a shared Python DSL that generates *both* a TLA+ spec and
frontrun-instrumented Python code from the same source of truth.

**Sketch:**
```python
from frontrun.formal import StateMachine, Action, Invariant

counter = StateMachine("Counter", variables={"count": 0})

@counter.action("Increment")
def increment(state):
    temp = state.count          # frontrun: read
    state.count = temp + 1      # frontrun: write

@counter.invariant
def count_non_negative(state):
    return state.count >= 0

# Generate TLA+ spec
counter.to_tla("Counter.tla")

# Generate frontrun test
counter.to_frontrun_test(threads=2, runs=100)
```

**Re-evaluation with agents in mind:** A full DSL that generates both TLA+ and Python
may be over-engineered when an agent can maintain the spec directly. But the
*sync-checking* aspect of this idea is independently valuable, even without a DSL.

### The Practical Version: A Spec/Code Correspondence Linter

Rather than a DSL, a lighter-weight tool that cross-references a TLA+ spec against
Python source files and flags mismatches:

```
$ frontrun lint --spec Counter.tla --source counter.py

WARNING: TLA+ action "Init" has no corresponding frontrun marker
WARNING: Frontrun marker "cleanup" in counter.py:42 has no corresponding TLA+ action
WARNING: TLA+ variable "locked" not covered by state_extractor
OK: Read <-> read_value, Write <-> write_value
OK: Processes {1, 2} <-> threads {t1, t2}
```

**What the linter checks:**
- Every TLA+ action (or PlusCal label) has a corresponding `# frontrun:` marker
- Every frontrun marker in the Python source appears in the spec
- TLA+ variables have corresponding fields in the `state_extractor`
- PlusCal processes map to frontrun execution names
- Invariants in the spec have corresponding Python assertion functions

**Implementation:** This is straightforward with `tree-sitter-tlaplus` for the spec
side and frontrun's existing `MarkerRegistry` for the Python side. The linter just
compares the two sets and reports differences.

**Agent workflow:** The linter serves as a fast, deterministic check that the agent
can run after updating either the spec or the code. When the linter reports
mismatches, the agent decides how to resolve them — maybe the spec needs a new action,
maybe the code needs a new marker, maybe something was intentionally removed. The
linter catches the drift; the agent handles the judgment calls.

This also works without an agent — a developer maintaining their own TLA+ spec gets
the same drift detection. And it could run in CI as a consistency check.

---

## 6. PlusCal-to-Schedule Compiler

**The idea:** PlusCal is a pseudocode language that transpiles to TLA+. Its `process`
and `label` concepts map almost 1:1 to frontrun threads and markers.

```
process Incrementer \in {1, 2}
begin
  read:    temp := count;
  write:   count := temp + 1;
end process;
```

Labels (`read`, `write`) become frontrun markers. Processes become threads. The
compiler reads PlusCal and generates:
- A `Schedule` strategy that covers all TLC-discovered behaviors
- A test template with the right markers and thread structure

**Agent angle:** PlusCal is an especially good target for agent-generated specs because
its imperative style is closer to how agents (and developers) think about algorithms.
An agent can translate Python concurrent code to PlusCal more naturally than to raw
TLA+, and the label/process structure directly produces the frontrun scaffolding.

---

## 7. Counterexample Replay

**The idea:** When TLC finds an invariant violation, it produces a counterexample trace
(a sequence of states). Convert this trace into an frontrun `Schedule` and replay it
against the Python implementation to confirm (or refute) the bug in real code.

```python
from frontrun.tlaplus import replay_counterexample

# TLC found: Init -> Read(t1) -> Read(t2) -> Write(t1) -> Write(t2) violates CounterCorrect
schedule = replay_counterexample(
    "MC_Counter.out",  # TLC output
    action_to_marker={"Read": "read_value", "Write": "write_value"},
    process_to_thread={"1": "t1", "2": "t2"},
)

# Now run it against real code
executor = TraceExecutor(schedule)
executor.run("t1", counter.increment)
executor.run("t2", counter.increment)
result = executor.finish()
assert result.had_error, "Bug confirmed in implementation!"
```

**Why it's powerful with agents:** The full pipeline becomes automatic:
1. Agent generates TLA+ spec from Python code
2. Agent runs TLC, which finds an invariant violation
3. Agent converts the counterexample into an frontrun `Schedule`
4. Agent runs the schedule against real code to confirm the bug
5. Agent reports the bug to the developer with a concrete reproduction test
6. Agent adds the test to the suite as a regression test

The developer sees: "I found a race condition in your counter. Here's a test that
reproduces it deterministically." They never see the TLA+ spec that found it.

---

## 8. Quint as the Ergonomic Middle Ground

Instead of raw TLA+, use [Quint](https://quint-lang.org/) (TypeScript-like syntax,
compiles to TLA logic, uses Apalache). Quint's JSON output format would be easier to
parse from Python than TLA+ output.

**Agent consideration:** An agent can work with either TLA+ or Quint. TLA+ has more
training data and community resources. Quint has better toolchain ergonomics (JSON IR,
static types, REPL). For agent-generated specs that humans rarely read, TLA+ is fine.
If developers want to inspect/edit the specs, Quint's more readable syntax is an
advantage.

---

## 9. Parsing TLA+ Specs from Python: Extracting Entity Mappings

Several integration workflows above depend on reading a TLA+ spec and extracting its
structure — actions, processes, variables, labels, invariants — to automatically map
them to frontrun annotations. **With agents in the loop, this parsing infrastructure
is primarily agent-facing plumbing**: the agent generates a spec, then parses it back
to produce frontrun data structures. The human rarely interacts with this layer
directly.

### Available Parsers

#### `tree-sitter-tlaplus` (Recommended)

The most practical option for Python integration. Pip-installable, no JVM required,
actively maintained, and parses both TLA+ and PlusCal.

- **PyPI**: `pip install tree-sitter tree-sitter-tlaplus`
- **GitHub**: [tlaplus-community/tree-sitter-tlaplus](https://github.com/tlaplus-community/tree-sitter-tlaplus)
- **Maintained by**: Andrew Helwer (also involved in official TLA+ tooling)
- **Output**: Concrete syntax tree (CST) — a full parse tree of typed nodes with
  positions, children, and type strings like `module`, `operator_definition`,
  `bounded_quantification`, etc.
- **Coverage**: Full TLA+ language including Unicode symbols and PlusCal blocks.
  Error-tolerant (incremental parsing).
- **Used by**: GitHub (`.tla` syntax highlighting), Neovim, [Spectacle/tla-web](https://github.com/will62794/spectacle)

**Basic usage:**
```python
import tree_sitter_tlaplus as tstla
from tree_sitter import Language, Parser

TLA_LANGUAGE = Language(tstla.language())
parser = Parser(TLA_LANGUAGE)

spec = b"""
---- MODULE Counter ----
VARIABLES count, pc

Read(t) == /\\ pc[t] = "read"
           /\\ pc' = [pc EXCEPT ![t] = "write"]

Write(t) == /\\ pc[t] = "write"
            /\\ count' = count + 1
            /\\ pc' = [pc EXCEPT ![t] = "done"]

Next == \\E t \\in Threads : Read(t) \\/ Write(t)
====
"""

tree = parser.parse(spec)
root = tree.root_node

# Walk the CST to extract operator definitions (= TLA+ actions)
def find_operator_defs(node):
    if node.type == "operator_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            yield name_node.text.decode()
    for child in node.children:
        yield from find_operator_defs(child)

actions = list(find_operator_defs(root))
# -> ["Read", "Write", "Next"]
```

**Caveat**: tree-sitter gives a *concrete* syntax tree, not a semantic AST. You get
the parse structure but not name resolution, type information, or module instantiation.
For extracting action names, variable declarations, process structures, and labels
this is sufficient. For deeper semantic analysis, see Apalache below.

**Agent note:** An agent could also just read the TLA+ source text directly and extract
structure without a formal parser — LLMs parse TLA+ well enough for mapping extraction.
The tree-sitter parser is valuable for **programmatic** pipelines where the agent isn't
in the loop (CI, automated test generation).

#### Apalache JSON IR

If you need a fully-resolved, semantically-analyzed AST and can tolerate a JVM
dependency, Apalache's `parse` command exports structured JSON.

- **Install**: [apalache-mc.org](https://apalache-mc.org/) (Scala, requires JVM)
- **Usage**: `apalache-mc parse --output=result.json MySpec.tla`
- **Output**: JSON serialization of Apalache's internal IR (documented in ADR-005).
  Includes typed representations of modules, operators, variables, expressions, and
  type annotations.
- **Also exports**: ITF (Informal Trace Format, ADR-015) — `.itf.json` files for
  counterexample traces with `vars`, `states`, and `loop` fields.

```python
import subprocess, json

subprocess.run(["apalache-mc", "parse", "--output=result.json", "Counter.tla"], check=True)
with open("result.json") as f:
    ir = json.load(f)

# ir contains fully-resolved operator definitions, variable declarations, etc.
```

**Advantages over tree-sitter**: Handles module instantiation/flattening, name
resolution, and Apalache type annotations. **Disadvantages**: JVM startup cost,
heavier dependency, Apalache-specific IR rather than generic TLA+ AST.

#### SANY XML Export (Reference parser, hard to consume)

SANY is the official Java parser used by TLC. It has an
[XMLExporter](https://github.com/tlaplus/tlaplus/blob/master/tlatools/org.lamport.tlatools/src/tla2sany/xml/XMLExporter.java)
but the format is poorly documented. The TLA+ community itself considers it difficult —
[tlaplus/rfcs#16](https://github.com/tlaplus/rfcs/issues/16) notes it "requires a lot
of implicit and undocumented knowledge." Not recommended for new integrations.

#### `tla` PyPI Package (Pure Python, dormant)

A Python port of the TLAPM OCaml parser using PLY. Provides native Python AST objects
with a visitor pattern (`tla.ast`, `tla.visit.NodeTransformer`). At version 0.0.1,
appears unmaintained. Related forks:
[ivan-gavran/tla_python_parser](https://github.com/ivan-gavran/tla_python_parser),
[g302ge/tla_python](https://github.com/g302ge/tla_python).

#### Quint JSON IR

[Quint](https://quint-lang.org/) parses its own language (not TLA+ directly) and
exports JSON IR via `quint parse --out=result.json`. Useful if adopting Quint as the
specification language instead of raw TLA+. TypeScript-based, no JVM needed.

### What to Extract: TLA+ Entities That Map to Frontrun

The core goal is to parse a TLA+ spec and produce a mapping that frontrun can use.
Here are the key entities and how they correspond:

| TLA+ Entity | Frontrun Concept | How to Extract |
|-------------|-------------------|----------------|
| **Action** (operator like `Read(t)`, `Write(t)`) | **Marker name** (`# frontrun: read`, `# frontrun: write`) | tree-sitter: `operator_definition` nodes at the top level of the spec |
| **Process** (PlusCal `process P \in S`) | **Thread / execution name** (`executor.run("P", ...)`) | tree-sitter: `pcal_process` nodes inside `pcal_algorithm` |
| **Label** (PlusCal `read:`, `write:`) | **Marker name** (directly — labels are the atomic units in PlusCal) | tree-sitter: `pcal_label` nodes within a process body |
| **Variable** (`VARIABLES count, pc`) | **State snapshot keys** (`state_extractor` return keys) | tree-sitter: `variable_declaration` nodes |
| **Invariant** (`TypeOK`, `CounterCorrect`) | **Python assertion function** (`@frontrun_invariant`) | tree-sitter: operator defs referenced in `INVARIANT` config or spec property |
| **`pc` variable** (program counter) | **Current marker per thread** (implicit in frontrun schedule) | Convention: the `pc` variable maps thread IDs to label names |
| **Fairness** (`WF_vars(Action)`) | **Schedule constraints** (every thread eventually progresses) | tree-sitter: `fairness_constraint` nodes in temporal formulas |

### Concrete Design: `SpecMapping` Class

A `SpecMapping` would parse a TLA+ file and produce the data structures frontrun
needs. This is primarily consumed by agents and automated pipelines:

```python
from dataclasses import dataclass
import tree_sitter_tlaplus as tstla
from tree_sitter import Language, Parser

@dataclass
class SpecMapping:
    """Mapping between a TLA+ spec and frontrun annotations."""
    actions: list[str]              # e.g. ["Read", "Write"]
    processes: list[str]            # e.g. ["Incrementer"] (PlusCal only)
    labels: dict[str, list[str]]    # process -> labels, e.g. {"Incrementer": ["read", "write"]}
    variables: list[str]            # e.g. ["count", "pc"]
    invariants: list[str]           # e.g. ["TypeOK", "CounterCorrect"]

    # Derived mappings for frontrun
    def action_to_marker(self) -> dict[str, str]:
        """Map TLA+ action names to frontrun marker names (lowercase/snake_case)."""
        return {a: a.lower() for a in self.actions}

    def label_to_marker(self) -> dict[str, str]:
        """Map PlusCal labels to frontrun marker names (identity — labels are already good names)."""
        all_labels = [l for labels in self.labels.values() for l in labels]
        return {l: l for l in all_labels}

    def process_to_thread(self) -> dict[str, str]:
        """Map PlusCal process names to frontrun thread names."""
        return {p: p.lower() for p in self.processes}

    @classmethod
    def from_tla_file(cls, path: str) -> "SpecMapping":
        tla_lang = Language(tstla.language())
        parser = Parser(tla_lang)
        with open(path, "rb") as f:
            tree = parser.parse(f.read())
        return cls._extract(tree.root_node)

    @classmethod
    def _extract(cls, root) -> "SpecMapping":
        actions, processes, labels, variables, invariants = [], [], {}, [], []

        def walk(node):
            if node.type == "operator_definition":
                name = node.child_by_field_name("name")
                if name:
                    actions.append(name.text.decode())
            elif node.type == "variable_declaration":
                for child in node.children:
                    if child.type == "identifier":
                        variables.append(child.text.decode())
            elif node.type == "pcal_process":
                # Extract process name and its labels
                name_node = node.child_by_field_name("name")
                if name_node:
                    pname = name_node.text.decode()
                    processes.append(pname)
                    labels[pname] = []
                    for desc in _descendants(node):
                        if desc.type == "pcal_label":
                            lbl = desc.children[0].text.decode()
                            labels[pname].append(lbl)
            for child in node.children:
                walk(child)

        walk(root)
        return cls(actions=actions, processes=processes, labels=labels,
                   variables=variables, invariants=invariants)


def _descendants(node):
    """Yield all descendant nodes."""
    for child in node.children:
        yield child
        yield from _descendants(child)
```

### PlusCal Is the Sweet Spot

PlusCal specs map to frontrun more naturally than raw TLA+ because:

1. **Labels = markers**: PlusCal labels define the atomic grain — each label is one
   indivisible step. This is exactly what an frontrun marker represents.
2. **Processes = threads**: PlusCal `process` declarations correspond directly to
   frontrun execution units.
3. **`pc` = schedule position**: PlusCal's auto-generated `pc` variable tracks which
   label each process is at, which is the same information frontrun's `Schedule`
   encodes.

A PlusCal spec like:
```
--algorithm Counter {
    variables count = 0;

    process (Incrementer \in {1, 2}) {
        read:   temp := count;
        write:  count := temp + 1;
    }
}
```

Directly implies:
```python
schedule = Schedule(steps=[
    Step("incrementer_1", "read"),
    Step("incrementer_2", "read"),
    Step("incrementer_1", "write"),
    Step("incrementer_2", "write"),
])
```

The `tree-sitter-tlaplus` grammar parses PlusCal blocks (both C-syntax and P-syntax),
so extracting processes and labels is straightforward CST walking.

**Agent advantage:** An agent can generate PlusCal from Python code more naturally than
raw TLA+, because PlusCal's imperative process/label structure mirrors how concurrent
Python code is actually written. The agent translates idioms it already understands
(threads, shared state, critical sections) into PlusCal's nearly-identical vocabulary.

### End-to-End Workflow: Agent-Driven Spec Testing

The full agent-mediated pipeline, from Python code to formally-guided tests:

```
Developer writes concurrent Python code with frontrun markers
    |
    v
Agent reads code, generates PlusCal/TLA+ spec
    |
    v
Agent runs TLC on the spec
    |
    +---> No violations found for all behaviors up to depth N
    |     Agent reports: "All interleavings verified up to N steps"
    |
    +---> TLC finds counterexample trace
          |
          v
    Agent converts counterexample to frontrun Schedule
          |
          v
    Agent runs Schedule against real Python code
          |
          +---> Bug reproduced: Agent reports bug + deterministic test
          |
          +---> Bug not reproduced: Spec-implementation mismatch,
                agent diagnoses the discrepancy
```

**Programmatic version:**

```python
from frontrun.tlaplus import SpecMapping, replay_counterexample
from frontrun.trace_markers import TraceExecutor
from frontrun.common import Schedule, Step

# 1. Parse the spec (agent-generated)
mapping = SpecMapping.from_tla_file("Counter.tla")
# mapping.labels = {"Incrementer": ["read", "write"]}
# mapping.variables = ["count", "pc"]

# 2. Run TLC to find counterexamples (or enumerate behaviors)
#    (via subprocess or modelator-py)

# 3. Convert a TLC counterexample into an frontrun Schedule
schedule = replay_counterexample(
    "MC_Counter.out",
    action_to_marker=mapping.label_to_marker(),
    process_to_thread=mapping.process_to_thread(),
)

# 4. Execute against real Python code
executor = TraceExecutor(schedule)
executor.run("incrementer_1", counter.increment)
executor.run("incrementer_2", counter.increment)
result = executor.finish()

# 5. Compare actual state against TLC's predicted state
assert counter.value == expected_from_tlc
```

### Open Questions

- **Granularity mismatch**: A TLA+ action may correspond to multiple Python source
  lines. Should one TLA+ action map to one marker, or should markers be placed at
  sub-action granularity? The PlusCal label approach avoids this since labels are
  already the atomic grain. An agent can make this judgment call per-codebase.
- **State projection**: TLA+ state includes all variables; Python state may be spread
  across object attributes, locals, globals. The `state_extractor` callback handles
  this but requires authoring. An agent can generate the extractor from the spec's
  variable list and the Python class structure.
- **Module instantiation**: TLA+ specs often use `INSTANCE` to compose modules. This
  is invisible to tree-sitter (it's a semantic operation). For specs with complex
  module structure, Apalache's flattened JSON IR would be needed.
- **Symmetry**: TLA+ processes are often symmetric (`\in {1, 2}`). Frontrun could
  exploit this to reduce the number of schedules to test, mirroring TLC's symmetry
  reduction.
- **Spec correctness**: If the agent generates the spec, who validates the spec? One
  answer: the trace validation workflow (1) serves as a bidirectional check — if the
  spec and code disagree, that's useful signal regardless of which is "right."
- **When the agent isn't available**: The parsing and mapping infrastructure should
  also work as a standalone library for developers who *do* write TLA+ specs manually.
  Agent-friendliness and human-friendliness aren't mutually exclusive.

---

## Assessment: Where to Start

| Workflow | Impact | Effort | Dependencies |
|----------|--------|--------|-------------|
| **7. Counterexample Replay** | High | Low | TLC installed, parse output |
| **1. Trace Validation** | High | Medium | `modelator-py` or TLC subprocess |
| **4. Invariant Bridge** | Medium | Low | Pure Python, no external tools |
| **2. Spec-Guided Schedules** | Very High | Medium | TLC trace enumeration |
| **6. PlusCal Compiler** | High | Medium | PlusCal parser |
| **3. Refinement Checking** | Very High | High | Workflows 1+2 as foundation |
| **5. Spec/Code Linter** | High | Low | tree-sitter + MarkerRegistry |

The suggested starting path: **Counterexample Replay** (7) and **Invariant Bridge**
(4) are low-effort and immediately useful. Then build toward **Trace Validation** (1)
and **Spec-Guided Schedules** (2), which together give the full agent-driven pipeline:
TLC finds bugs in the model, frontrun confirms them in real code.

**Note on the Linter (5):** The spec/code correspondence linter is low-effort, useful
to both agents and humans, and provides the foundation for all other workflows — if
the spec and code don't agree on what the actions are, nothing else works. It also
naturally falls out of the `SpecMapping` + `MarkerRegistry` infrastructure needed for
the other workflows.

---

## Relevant Tools and References

### TLA+ Parsing Tools
- [tree-sitter-tlaplus](https://github.com/tlaplus-community/tree-sitter-tlaplus) — pip-installable TLA+/PlusCal parser with Python bindings (recommended)
- [Apalache](https://apalache-mc.org/) — symbolic model checker with `parse --output=result.json` for JSON IR export
- [tla (PyPI)](https://pypi.org/project/tla/) — pure Python TLA+ parser (v0.0.1, dormant)
- [Quint](https://quint-lang.org/) — engineer-friendly spec language with JSON IR export

### Python-TLA+ Tools
- [modelator-py](https://github.com/informalsystems/modelator-py) — Python wrapper for TLC and Apalache
- [modelator](https://github.com/informalsystems/modelator) — Rust/Go tool for TLA+ model-based testing

### Concurrency Testing Tools (Prior Art)
- [Coyote](https://microsoft.github.io/coyote/) (Microsoft, C#) — binary rewriting + controlled scheduling
- [Loom](https://github.com/tokio-rs/loom) (Rust) — exhaustive concurrency testing, CDSChecker-based
- [Shuttle](https://github.com/awslabs/shuttle) (AWS, Rust) — randomized concurrency testing with PCT
- CHESS (Microsoft Research) — foundational work on preemption bounding

### Key Papers
- Cirstea et al. ["Validating Traces of Distributed Programs Against TLA+ Specifications"](https://arxiv.org/abs/2404.16075) (SEFM 2024)
- Howard et al. ["Smart Casual Verification of CCF's Distributed Consensus"](https://arxiv.org/html/2406.17455v1) (NSDI 2025)
- Kuprianov & Konnov. ["Model-based testing with TLA+ and Apalache"](http://conf.tlapl.us/2020/09-Kuprianov_and_Konnov-Model-based_testing_with_TLA_+_and_Apalache.pdf) (TLA+ Conference 2020)
- Musuvathi & Qadeer. ["Iterative Context Bounding for Systematic Testing"](https://dl.acm.org/doi/10.1145/1273442.1250785) (PLDI 2007)
