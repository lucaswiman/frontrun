# Comprehensible Race Condition Error Messages

## Problem

When frontrun finds a race condition, the counterexample is a **list of thread indices**:

```
[0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, ...]
```

This is an opcode-level schedule — one entry per bytecode instruction, per thread. For a
typical 3-line function, the schedule might be 50+ entries long, most of which are
irrelevant internal opcodes (LOAD_FAST, PUSH_NULL, PRECALL, etc.). The user sees a wall of
integers and has no way to connect them back to the source code that actually raced.

The trace markers approach is better (steps are named), but the bytecode and DPOR approaches
— the ones most users will reach for — give almost no explanatory power.

**Goal:** When a test fails, tell the user a *story* about what happened, in terms of their
source code, not bytecode indices.

---

## What information is available at failure time?

### Currently captured

| Data | Bytecode | DPOR | Trace Markers |
|------|----------|------|---------------|
| Schedule (thread indices) | ✓ | ✓ | ✓ (named steps) |
| Thread functions (callables) | ✓ | ✓ | ✓ |
| Invariant that failed | ✓ | ✓ | N/A |
| Shared state object | ✓ | ✓ | N/A |
| Seed / execution number | ✓ | ✓ | N/A |

### Available but not currently captured

| Data | How to get it | Cost |
|------|---------------|------|
| Frame objects (filename, lineno, function name) | `sys.settrace` / `sys.monitoring` already has them | Nearly free — just record |
| Opcode names at each step | `dis.get_instructions(frame.f_code)` + `frame.f_lasti` | Already done in DPOR's `_process_opcode` |
| Source lines | `linecache.getline(filename, lineno)` | Cheap |
| Object + attribute names for shared accesses | Shadow stack already tracks for DPOR | Already done for DPOR; need to add for bytecode |
| Read vs. write classification | DPOR reports to engine already | Already done for DPOR |
| Lock acquire/release events | Cooperative lock patches report these | Already done |
| Which specific accesses conflicted | DPOR engine knows which accesses caused backtracking | Would need Rust engine to export |

**Key insight:** The trace callbacks already see the frame object at every opcode. We just
need to *record* the interesting ones (source-level operations on shared state) and *discard*
the noise (stack manipulation, control flow, internal ops).

---

## Display Options

### Option 1: Interleaved Source Line Trace

Show the relevant source lines from each thread, interleaved in execution order, with
thread labels. Filter out lines that don't touch shared state.

```
Race condition found after 3 interleavings.

  Thread 0 (increment)  │ counter.py:8   temp = self.value        # read self.value → 0
  Thread 1 (increment)  │ counter.py:8   temp = self.value        # read self.value → 0
  Thread 0 (increment)  │ counter.py:9   self.value = temp + 1    # write self.value ← 1
  Thread 1 (increment)  │ counter.py:9   self.value = temp + 1    # write self.value ← 1

Invariant violated: counter.value == 2 (actual: 1)
```

**Pros:**
- Reads like a story: "thread 0 read, then thread 1 read the *same stale value*, then both wrote"
- Directly shows the classic lost-update pattern
- Compact — only the lines that matter

**Cons:**
- Requires source line deduplication (same line executes many opcodes; show it once per "visit")
- Need heuristics to decide which lines are "interesting"
- Multi-line expressions could be tricky

### Option 2: Two-Column Side-by-Side (à la TLA+ error traces)

Show each thread's execution as a column, with time flowing downward. Mark the conflicting
operations.

```
Race condition found after 3 interleavings.

  Time │ Thread 0 (increment)              │ Thread 1 (increment)
  ─────┼────────────────────────────────────┼────────────────────────────────────
    1  │ temp = self.value        [read]   │
    2  │                                    │ temp = self.value        [read]
    3  │ self.value = temp + 1    [write]  │
    4  │                                    │ self.value = temp + 1    [write]
  ─────┼────────────────────────────────────┼────────────────────────────────────

  ⚠ Both threads read self.value before either wrote.
  Invariant violated: counter.value == 2 (actual: 1)
```

**Pros:**
- Visual structure makes the interleaving obvious at a glance
- Familiar to anyone who's read a TLA+ or SPIN error trace
- Empty cells show where a thread was *not running* — makes preemptions visible

**Cons:**
- Gets wide with 3+ threads
- Terminal width constraints may truncate
- ASCII art is fragile in different terminal/font configurations

### Option 3: Grouped "Chapters" per Thread, with Conflict Annotations

Instead of interleaving, show each thread's execution sequentially, then annotate which
operations conflicted with which.

```
Race condition found after 3 interleavings.

  Thread 0 (increment):
    counter.py:8   temp = self.value        # read self.value → 0       ← (A)
    counter.py:9   self.value = temp + 1    # write self.value ← 1      ← (B)

  Thread 1 (increment):
    counter.py:8   temp = self.value        # read self.value → 0       ← (C)
    counter.py:9   self.value = temp + 1    # write self.value ← 1      ← (D)

  Execution order: A → C → B → D
  Conflict: (A) and (C) both read self.value before (B) or (D) wrote it.
  Invariant violated: counter.value == 2 (actual: 1)
```

**Pros:**
- Easy to see what each thread *intended* to do
- Conflict annotation tells you the "why" — which ordering broke things
- Scales to many threads without width explosion

**Cons:**
- The execution order line `A → C → B → D` can be hard to mentally replay
- Loses the visceral "they were interleaved!" feeling of options 1/2
- Letters/labels add cognitive overhead

### Option 4: Minimal "Diff from Correct" Explanation

Show the correct (serialized) execution, then highlight what was different in the
failing interleaving — i.e., what moved.

```
Race condition found after 3 interleavings.

  Correct (serial) execution:
    T0: temp = self.value       →  0
    T0: self.value = temp + 1   →  1
    T1: temp = self.value       →  1
    T1: self.value = temp + 1   →  2    ✓ counter.value == 2

  Failing interleaving (T1 preempted T0 between read and write):
    T0: temp = self.value       →  0
    T1: temp = self.value       →  0    ← T1 read stale value
    T0: self.value = temp + 1   →  1
    T1: self.value = temp + 1   →  1    ← overwrote T0's write

  Invariant violated: counter.value == 2 (actual: 1)
```

**Pros:**
- Directly answers "what went wrong?" by showing the expected vs actual
- The "stale value" and "overwrote" annotations tell the story
- Maps well to how developers debug: "it should have been X, but was Y"

**Cons:**
- Requires running the serial execution as a baseline (extra work, but cheap)
- Two sections is more output to read
- "Correct" execution assumes serial is always correct (true for most invariants, but
  not all)

### Option 5: Annotated Schedule with Source Context (Minimal Change)

Keep the schedule, but annotate each *interesting* step with source context. The simplest
incremental improvement.

```
Race condition found after 3 interleavings.
Schedule (showing shared-state operations only):

  step  12: T0  counter.py:8   LOAD_ATTR self.value        (read)
  step  31: T1  counter.py:8   LOAD_ATTR self.value        (read)
  step  47: T0  counter.py:9   STORE_ATTR self.value       (write)
  step  58: T1  counter.py:9   STORE_ATTR self.value       (write)

  Full schedule (62 opcodes): [0, 1, 0, 1, 0, 0, 0, 1, ...]
  Invariant violated: counter.value == 2 (actual: 1)
```

**Pros:**
- Minimal implementation effort — just filter and annotate the existing schedule
- Still shows the raw schedule for advanced users
- Naturally extends to DPOR (which already classifies accesses)

**Cons:**
- Opcode names (LOAD_ATTR, STORE_ATTR) are less friendly than source lines
- Still somewhat low-level

---

## Connecting Back to a "Story": Design Principles

Whatever display format we choose, the core challenge is the same: **filtering 50-200 opcode
steps down to the 2-6 operations that actually matter, then explaining** ***why*** **that
ordering is wrong.**

### Step 1: Record an execution trace (not just a schedule)

During the failing run, record a list of `TraceEvent` structs:

```python
@dataclass
class TraceEvent:
    step_index: int           # position in the opcode schedule
    thread_id: int            # which thread
    filename: str             # source file
    lineno: int               # source line number
    function_name: str        # enclosing function
    source_line: str          # actual source text
    opcode: str               # LOAD_ATTR, STORE_ATTR, etc.
    object_repr: str | None   # "self.value", "account.balance", etc.
    access_type: str | None   # "read", "write", or None
```

This is straightforward: the trace callback already has `frame` (which gives filename,
lineno, f_code.co_name), and for DPOR we already classify accesses. For bytecode mode,
we'd add lightweight access detection (just LOAD_ATTR/STORE_ATTR on the frame's
f_locals/f_globals, no full shadow stack needed).

### Step 2: Filter to "interesting" events

An event is interesting if:
1. It accesses shared mutable state (LOAD_ATTR/STORE_ATTR on objects reachable from the
   setup state, or on objects passed to multiple threads), **OR**
2. It's a synchronization operation (lock acquire/release, event wait/set), **OR**
3. It's the last operation before a thread yields (preemption point)

Everything else (local variable manipulation, control flow, function call setup) is noise.

For the simple case, "shared state" can be approximated as: any attribute access on an
object that was returned by `setup()` or is reachable from it. DPOR's shadow stack already
tracks this precisely.

### Step 3: Deduplicate by source line

Multiple opcodes execute for a single source line. Group consecutive events from the same
thread on the same source line into a single "line execution." Show the line once, with a
summary:

```
counter.py:8   temp = self.value    →  read self.value (= 0)
```

This collapses `LOAD_FAST self`, `LOAD_ATTR value`, `STORE_FAST temp` into one human-readable
line.

### Step 4: Identify the conflict pattern

Look at the filtered trace and classify the bug:

| Pattern | Shape | Name |
|---------|-------|------|
| R₀ R₁ W₀ W₁ | Both read before either writes | **Lost update** (atomicity violation) |
| W₀ R₁ | Write happens before expected read | **Order violation** |
| R₁ W₀ | Read sees pre-write value | **Stale read** |
| Lock₀ Lock₁ Lock₁ Lock₀ | Circular lock acquisition | **Deadlock** |

Add a one-line summary like:
```
⚠ Lost update: both threads read self.value (= 0) before either wrote.
```

### Step 5: Present the trace + summary

Combine the filtered trace (in one of the display formats above) with the pattern
classification. The "story" is:

1. **What happened** (the interleaved trace)
2. **Why it's wrong** (the conflict pattern + invariant violation)
3. **How to reproduce** (the full schedule, for `run_with_schedule()`)

---

## Implementation Strategy: Cheapest Path to High Value

### Phase 1: Record and filter (works for both bytecode and DPOR)

- Add a `TraceRecorder` that accumulates `TraceEvent` objects during a run
- In bytecode mode: record at each opcode callback (only for user-code frames)
- In DPOR mode: record alongside `_process_opcode` (already has all the info)
- Only record when replaying a counterexample (not during exploration — too expensive)
- After the failing run, filter to shared-state accesses

### Phase 2: Source-line trace display (Option 1)

- Deduplicate events to source lines
- Format as the interleaved source line trace
- This is the highest-value, most-readable format
- Attach to the `InterleavingResult` / `DporResult` as a `.explanation: str` field

### Phase 3: Conflict classification

- Pattern-match the filtered trace to identify lost update, order violation, etc.
- Add a one-line summary to the output
- This is the "punch line" that tells the user what class of bug they have

### Phase 4: Rich terminal output (optional)

- Use ANSI colors: red for writes, blue for reads, dim for thread labels
- Highlight the conflicting pair of operations
- Add a `--no-color` flag for CI
- Consider rich/click for terminal width detection

---

## Trace Markers: Already Pretty Good

Trace markers already have named steps (`Step("thread1", "read_value")`), so the schedule
is inherently readable. But we could still improve:

- Show the source line next to each step name
- When a schedule stalls, explain *which* thread was expected at *which* marker, and show
  what each thread was actually doing (blocked on lock? not yet reached that line?)

---

## Integration with Hypothesis

When using `schedule_strategy()` with `@given`, Hypothesis controls the test execution. On
failure, Hypothesis prints the counterexample schedule. We could:

1. **Custom `note()` call:** Inside the test body, after detecting failure, call
   `hypothesis.note(formatted_trace)` to include the trace in Hypothesis's output.

2. **Custom `repr` for schedules:** Wrap the schedule in a class with a `__repr__` that
   includes the trace. Hypothesis prints `repr()` of failing examples.

3. **`event()` annotations:** Use `hypothesis.event()` during exploration to tag
   interleavings by conflict type, giving coverage statistics in Hypothesis's output.

---

## Open Questions

1. **Performance of recording:** Recording every opcode during counterexample replay should
   be cheap (it's one run, already slow due to tracing). But should we record during
   exploration too, to avoid needing a replay step?

2. **Multi-function threads:** When thread functions call helpers, the trace spans multiple
   files/functions. How deep do we go? Probably: show the top-level call + any function that
   actually touches shared state.

3. **Value capture:** Showing `self.value → 0` requires reading the value at trace time.
   This is a snapshot that could be stale by the time we display it. For the replay run
   this is fine (deterministic), but we should document the caveat.

4. **Object identity vs. name:** DPOR tracks `(id(obj), attr_name)`. For display, we need
   a human-readable name. Could use `f_locals` variable name + attribute, or the `repr()` of
   small objects. Need heuristics to avoid huge repr strings.

5. **How much to show for large programs:** A real codebase might have 20 shared-state
   accesses in the failing trace. Showing all 20 is too much. Should we:
   - Show only the first conflict pair + context?
   - Show all, but highlight the conflict?
   - Let the user control verbosity?
