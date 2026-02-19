# Property-Based Testing of Schedules for Comment Annotations

## The Problem

Frontrun currently has two modes that sit at opposite ends of a spectrum:

1. **Trace markers** (`# frontrun: marker_name`) — You manually write a `Schedule`
   of `Step` objects that specifies the exact execution order. Precise and
   deterministic, but requires knowing the bug-triggering interleaving in advance.

2. **Bytecode exploration** (`explore_interleavings`) — Generates random schedules
   at the **opcode level**, exploring every possible context switch between bytecode
   instructions. Fully automatic, but the search space is enormous (hundreds of
   opcodes per thread, so `num_threads ^ num_opcodes` possible schedules).

There's a natural middle ground: **property-based testing at the marker level**.

## The Idea

Use the `# frontrun: marker_name` comments as the vocabulary for schedule
generation instead of individual opcodes. A Hypothesis strategy generates
sequences of `(thread_name, marker_name)` steps, and the trace marker executor
runs each generated schedule.

```python
# Instead of this (opcode-level, huge search space):
schedule = [0, 0, 1, 1, 0, 1, 0, 0, 1, ...]  # 300 opcode decisions

# Generate this (marker-level, tiny search space):
schedule = Schedule([
    Step("thread1", "read"),
    Step("thread2", "read"),
    Step("thread1", "write"),
    Step("thread2", "write"),
])
```

If a function has 3 markers and there are 2 threads, the marker-level search
space is roughly `(2 * 3)^(2 * 3) = 46656` permutations — versus billions at
the opcode level. In practice, valid schedules are a small fraction of this
(each thread must hit its markers in order), reducing it even further.

## Proposed API

### `marker_schedule_strategy`

A Hypothesis strategy that generates valid `Schedule` objects from declared markers:

```python
from hypothesis import given
from frontrun.trace_markers import marker_schedule_strategy, TraceExecutor

@given(schedule=marker_schedule_strategy(
    threads={
        "worker": ["read_queue", "check_flag", "process"],
        "main":   ["enqueue", "set_flag"],
    }
))
def test_no_lost_tasks(schedule):
    state = MyState()
    executor = TraceExecutor(schedule)
    executor.run("worker", lambda: state.worker())
    executor.run("main", lambda: state.main_thread())
    executor.wait(timeout=5.0)
    assert not state.task_lost
```

The strategy would:
1. Take a dict of `{thread_name: [marker1, marker2, ...]}` (markers in source order)
2. Generate all valid interleavings where each thread's markers appear in order
3. Use Hypothesis's shrinking to minimize counterexamples to the shortest schedule

### `explore_marker_interleavings`

A convenience function mirroring `explore_interleavings` but at marker granularity:

```python
from frontrun.trace_markers import explore_marker_interleavings

result = explore_marker_interleavings(
    setup=lambda: MyState(),
    threads={
        "worker": (lambda s: s.worker(), ["read_queue", "check_flag", "process"]),
        "main":   (lambda s: s.main_thread(), ["enqueue", "set_flag"]),
    },
    invariant=lambda s: not s.task_lost,
    max_attempts=1000,
    seed=42,
)
```

## Why This Is Valuable

### 1. Dramatic Search Space Reduction

The case study results show that bytecode exploration finds bugs quickly in
practice (1-9 attempts). But this is partly because the tested code paths are
short. For longer functions with many opcodes, the search space grows
exponentially. Marker-level scheduling keeps the space manageable regardless
of function length.

| Approach | Search Space (2 threads, 100 opcodes each) | Search Space (2 threads, 5 markers each) |
|----------|---------------------------------------------|------------------------------------------|
| Bytecode | `2^200` ≈ 10^60 | — |
| Markers  | — | `C(10,5)` = 252 valid orderings |

### 2. Markers as Documentation

The `# frontrun: marker_name` comments serve double duty:
- **For humans**: They mark the interesting interleave points in the code,
  documenting where context switches matter
- **For the tool**: They define the vocabulary for schedule generation

This is a form of "specification by annotation" — the developer identifies
the critical sections, and the tool exhaustively explores all orderings.

### 3. Exhaustive Exploration Becomes Feasible

With 2 threads and 5 markers each, there are only 252 valid interleavings
(the number of ways to merge two ordered sequences of length 5). The tool
can explore ALL of them, not just a random sample. This gives **completeness
guarantees** that random bytecode exploration cannot.

```python
result = explore_marker_interleavings(
    ...,
    exhaustive=True,  # explore ALL valid orderings
)
assert result.property_holds  # proven correct for all marker-level interleavings!
```

### 4. Better Shrinking

Hypothesis can shrink marker-level schedules to minimal counterexamples.
A bytecode schedule might be 200 opcode decisions; its marker-level equivalent
might be 6 steps. The minimal counterexample is immediately human-readable:

```
Counterexample:
  Step("worker", "read_queue")
  Step("main", "enqueue")
  Step("main", "set_flag")
  Step("worker", "check_flag")   ← worker sees flag but missed the enqueue
```

### 5. Composable with Existing Infrastructure

The trace marker executor (`TraceExecutor`) already handles schedule-driven
execution. The only new piece is the schedule *generation* — everything else
is reusable.

## Implementation Sketch

### Schedule Generation

A valid marker schedule is an **interleaving of ordered sequences**. For threads
`A = [a1, a2, a3]` and `B = [b1, b2]`, valid schedules include:

```
[a1, a2, a3, b1, b2]     # A runs first
[a1, b1, a2, b2, a3]     # interleaved
[b1, b2, a1, a2, a3]     # B runs first
[a1, b1, b2, a2, a3]     # etc.
```

The constraint is that each thread's markers appear in their original order.
This is equivalent to choosing positions for one thread's markers in the
combined sequence — a well-studied combinatorial problem.

```python
from hypothesis import strategies as st
from frontrun.common import Schedule, Step

def marker_schedule_strategy(threads: dict[str, list[str]]):
    """Generate valid Schedule objects from thread marker declarations.

    A valid schedule interleaves each thread's markers while preserving
    their relative order within each thread.
    """
    thread_items = list(threads.items())
    total_steps = sum(len(markers) for _, markers in thread_items)

    # Strategy: generate a permutation of step indices, then filter
    # to only those that preserve per-thread marker order.
    # More efficient: generate by choosing which thread goes next.

    @st.composite
    def gen(draw):
        remaining = {name: list(markers) for name, markers in thread_items}
        steps = []

        while any(remaining.values()):
            # Choose from threads that still have markers
            available = [name for name, m in remaining.items() if m]
            thread = draw(st.sampled_from(available))
            marker = remaining[thread].pop(0)
            steps.append(Step(thread, marker))

        return Schedule(steps)

    return gen()
```

### Exhaustive Enumeration

For small marker sets, enumerate all valid interleavings:

```python
from itertools import combinations

def all_marker_schedules(threads: dict[str, list[str]]) -> list[Schedule]:
    """Enumerate ALL valid interleavings of thread markers."""
    thread_items = list(threads.items())

    if len(thread_items) == 2:
        # Two threads: choose positions for thread A's markers
        name_a, markers_a = thread_items[0]
        name_b, markers_b = thread_items[1]
        n = len(markers_a) + len(markers_b)

        schedules = []
        for positions in combinations(range(n), len(markers_a)):
            steps = [None] * n
            a_idx, b_idx = 0, 0
            for i in range(n):
                if i in positions:
                    steps[i] = Step(name_a, markers_a[a_idx])
                    a_idx += 1
                else:
                    steps[i] = Step(name_b, markers_b[b_idx])
                    b_idx += 1
            schedules.append(Schedule(steps))

        return schedules

    # General case: recursive
    ...
```

### Partial Schedules

Sometimes you want markers at only the *interesting* points, not everywhere.
The schedule controls execution order at markers; between markers, threads
run freely. This is already how `TraceExecutor` works — it only synchronizes
at marker points.

This means you can add markers surgically:

```python
def _should_keep_going(self):
    with self.worker_lock:                    # frontrun: acquire_worker_lock
        keep_going = self.keep_going
    # ← race window is HERE, between the two locks
    with self.join_lock:                      # frontrun: acquire_join_lock
        if self._join_is_called and self.bench.empty():
            return False
    return keep_going
```

Just two markers, but they capture the exact race window.

## Comparison with Bytecode Exploration

| Feature | Bytecode Exploration | Marker Schedules |
|---------|---------------------|------------------|
| Setup effort | Zero (automatic) | Add `# frontrun:` comments |
| Search space | Enormous (opcode-level) | Small (marker-level) |
| Exhaustive | No (random sampling) | Yes (for small marker sets) |
| Counterexample size | 50-300 opcode steps | 3-10 marker steps |
| Human-readable | No (opcode indices) | Yes (named markers) |
| Works with any code | Yes | Requires annotation |
| Finds subtle bugs | Sometimes misses narrow windows | Guarantees finding them if markers bracket the window |

The two approaches are complementary:
- Use **bytecode exploration** first to discover bugs automatically
- Then add **markers** at the identified race windows for regression testing
- Use **marker schedule strategies** to verify that the fix actually eliminates
  ALL problematic interleavings, not just the one counterexample

## Relation to the Case Studies

In the case studies, the real-code exploration tests found every bug on 20/20
seeds. This suggests that for the specific code paths tested, the race windows
are wide enough that random bytecode schedules hit them easily.

But consider a more subtle bug where the race window is exactly 2 opcodes wide
in a 500-opcode function. The probability of the random scheduler placing a
context switch in that 2-opcode window is low. With markers at the window
boundaries, the search becomes trivial.

The PyDispatcher case study already demonstrated this dual approach:
1. **Bytecode exploration** found the send-during-disconnect bug
2. **Trace markers** reproduced it with an exact, human-readable schedule

The missing piece is the **automatic generation** of marker schedules — which
is what this proposal adds.
