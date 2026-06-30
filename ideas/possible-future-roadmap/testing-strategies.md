# Remaining Testing Strategy Extensions

Last reviewed: 2026-06-12.

The core property-based marker schedule infrastructure is **fully implemented**:
`marker_schedule_strategy()`, `all_marker_schedules()`, `explore_marker_interleavings()`,
bytecode-level exploration, DPOR, and the pytest plugin (`frontrun/pytest_plugin.py`,
registered via the `pytest11` entry point: cooperative lock patching + skip guard for
runs outside the `frontrun` CLI).

Previously-listed extensions that were dropped as not worth doing (agent-duplicative
convenience layers or low-value analysis): adaptive marker placement, Hypothesis
convenience profiles/decorators, schedule filtering constraints, distribution analysis,
multi-level markers, `@pytest.mark.frontrun_markers` parametrization sugar, comparative
benchmarking. See git history for the original write-ups.

---

## Extension: Hybrid Marker + Bytecode Exploration

**Status:** Not implemented
**Complexity:** Medium

Combine marker-level and bytecode-level exploration for cases where markers bracket a
race but the exact bytecode interleaving within that window matters.

### Idea

1. User places markers at high-level race windows (e.g., lock acquire/release boundaries)
2. Tool explores all marker-level interleavings
3. For each marker-level schedule, run bytecode-level exploration *within* that schedule
4. This creates a two-level search: coarse (marker) + fine (bytecode)

### Implementation approach

```python
def explore_hybrid_interleavings(
    setup: Callable,
    threads: dict[str, tuple[Callable, list[str]]],
    invariant: Callable,
    bytecode_per_marker: bool = True,  # fine-grained search within each marker schedule
    bytecode_attempts: int = 100,
):
    """Explore marker schedules, then bytecode-level within each."""
```

Register as a `Strategy` adapter in `frontrun/_strategy.py` so `frontrun.explore(strategy=...)`
picks it up.

### Value

- Captures bugs that require both marker-level ordering AND specific bytecode interleaving
- Reduces search space vs. pure bytecode exploration
- Guarantees coverage of marker-level interleavings
- Best current answer to the state-explosion ceiling on realistic code

---

## Extension: Marker Coverage and Regression Tracking

**Status:** Not implemented
**Complexity:** Low

Track which marker-level interleavings have actually been exercised and flag gaps.

### Idea

After running marker-based tests, generate a report of:
- Which marker-level interleavings were actually executed
- Which were missed or underexplored
- Recommendations for additional test cases

### Implementation approach

- Instrument `explore_marker_interleavings` to log executed schedules
- Compare against `all_marker_schedules()` to identify gaps
- Generate human-readable coverage report

### Value

- Helps teams know whether they've covered the interesting cases
- Detects regressions where a bug fix is only tested on one interleaving
- Natural fit for CI summary output via the existing pytest plugin
