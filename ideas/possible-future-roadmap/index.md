# Frontrun Roadmap

Last reviewed: 2026-07-06.

## Documents

| Document | Scope |
|----------|-------|
| [../cross_process_exploration.md](../cross_process_exploration.md) | Deterministic interleaving of OS processes contending on shared SQL/Redis state |
| [../fault_injection.md](../fault_injection.md) | Injected exceptions / async cancellation as schedulable events |
| [../virtual_clock.md](../virtual_clock.md) | Virtualized time: timeout, retry, and TTL races as explorable schedules |
| [virtual-clock-transparency.md](virtual-clock-transparency.md) | Follow-up work to make virtual-clock behavior more transparent |
| [virtual-clock-hardening-deferred.md](virtual-clock-hardening-deferred.md) | Deferred fixes from the virtual-clock hardening wave (replay accounting, loop-timer diagnostics, HB edges, cleanups) |
| [dpor-improvements.md](dpor-improvements.md) | Wakeup tree equivalence, redundant opcode suppression (all low priority) |
| [integrations-and-detection.md](integrations-and-detection.md) | SQL/Redis/resource detection layers, FK analysis |
| [formal-methods.md](formal-methods.md) | TLA+/Quint integration, spec-guided exploration, counterexample replay |
| [testing-strategies.md](testing-strategies.md) | Hybrid marker+bytecode exploration, marker coverage tracking |
| [../random_dpor.md](../random_dpor.md) | Literature survey + proposals for randomized/hybrid DPOR exploration |
| [../coverage_and_invariants.md](../coverage_and_invariants.md) | Research report: coverage signals + automatic invariant discovery |

## Implemented (removed from the lists below)

Refactoring roadmap phases 1–5 (`_dpor_runtime/`, `_async_autopause.py`,
`contrib/*/_shared.py`, patch registries, `_threaded_runner.py`), all 5 DPOR search
strategies (`crates/dpor/src/path.rs`), pytest plugin (`frontrun/pytest_plugin.py`),
position-sensitive future access cache, provenance-tagged access summaries,
WeakWrite+WeakRead merge, per-step independence check, SQL resource grouping
(Defect #15 Approach 2), async/await marker support.

**Cross-process exploration** (`frontrun.explore(execution="process")`,
`_dpor_runtime/xproc/`). Multiprocessing/subprocess Python workers running
frontrun's SQL/Redis patching, coordinated over a socket. Phase 1
(plumbing/exhaustive), Phase 2 (engine-driven DPOR reduction via per-worker relay
threads reusing `DporScheduler`; cross-process row-lock deadlock detection; Redis)
and Phase 3 (persistent worker reuse) are done and covered by unit + functional +
e2e tests (SQLite/Redis subprocesses).

Scheduling *unmodified / non-Python* workers is **out of scope for frontrun by
design**, and this is a scope decision, not a missing feature. frontrun is a
white-box tester: it instruments *your* code from the inside to produce a
deterministic, replayable, minimized proof of the exact interleaving that breaks
an invariant (see "Design goals & scope" in `CLAUDE.md`). Reaching unmodified
workers means intercepting at the wire — and that is a fundamentally black-box,
*Jepsen-shaped* problem (observe histories, check them for consistency
violations) rather than a white-box one (control the schedule, prove one
interleaving). Different product.

Two interception layers were considered and neither belongs in frontrun. The
**LD_PRELOAD** syscall PoC (Phase 4, prototyped and removed) could block-until-grant,
but can't supply the semantic access identity DPOR needs (which row/key, read vs
write), can't get step-completion (it hooked `send` but not `recv`, so grant order
!= effect order), and misses the common case (SQLite is in-process file I/O with no
wire protocol; Go/static binaries bypass libc). A **DSN-level protocol-aware proxy**
(a process in front of Postgres/Redis, so "unmodified worker" becomes "point your
connection string at the proxy") would actually work — full-duplex framing gives
identity + quiescence — but it is a *separate tool*: a per-protocol wire state
machine, months of work, Jepsen-adjacent in shape. If ever built, it should be its
own library/service that **reuses frontrun's Rust DPOR engine** (the reusable seam:
vector clocks, sleep sets, wakeup tree, row-lock reasoning), not a feature grown
inside frontrun. The PG parser at `crates/io/src/sql_extract.rs` would be a starting
seed for such a tool. Meanwhile the supported path is unchanged: if the workers are
Python, run frontrun inside them.

**Virtual clock** (`frontrun.explore(clock="virtual"|"explored")`) is implemented
for sync and async workers under DPOR and random strategies. It covers
scheduler-visible Python timing (`time.*`, `time.sleep`, `asyncio.sleep`, sync
cooperative timed waits) and keeps timer firings replayable via autojump or the
DPOR clock actor. Follow-up transparency work, such as virtual `asyncio.wait_for`,
`datetime` support, and captured-reference diagnostics, is tracked in
`virtual-clock-transparency.md`.

## Dropped (2026-06-12 cleanup)

Removed as not worth doing — superseded by existing machinery, low value, or duplicative
of what a coding agent does on demand: adaptive marker placement, Hypothesis convenience
profiles, schedule filtering constraints, distribution analysis, multi-level markers,
comparative benchmarking, `__class__`-reassignment taint propagation, `gc.get_referrers()`
resource discovery, record/replay of external state, import-hook library registry,
one-line resource decorator, frame-local variable poisoning, counterexample → regression
test generation. Originals in git history.

## Priority overview

### P1 — New directions (endorsed 2026-06)

1. **Cross-process exploration** (cross_process_exploration.md) — ✅ implemented
   for Python multiprocessing/subprocess workers (Phases 1–3); see the Implemented
   section above. Non-Python / unmodified-worker scheduling is out of scope by
   design (white-box vs. Jepsen-shaped; see above) — a separate tool, if ever.
2. **Fault-point exploration** (fault_injection.md) — start with the async cancellation
   sweep (v1), which needs no new exploration machinery.
3. **Virtual clock transparency follow-ups** (virtual-clock-transparency.md) —
   virtual `asyncio.wait_for`, async Queue/Condition visibility, `datetime`
   support, and diagnostics for captured real-time references.

### P1 — Carried over

4. **Cross-table FK analysis** (integrations-and-detection) — schema introspection for
   foreign-key dependencies; catches referential-integrity races. ~150 LOC + 25 tests.
5. **Randomized wakeup tree ordering** (random_dpor: Proposal A) — different seeds explore
   different trace-space regions. Low effort, high value for `stop_on_first=True`.
6. **Hybrid marker + bytecode exploration** (testing-strategies) — two-level search:
   coarse markers, fine bytecode within each window. Best current answer to state
   explosion on realistic code.
7. **Counterexample replay from TLC** (formal-methods: 2.1) and **invariant assertion
   bridge** (formal-methods: 1.2) — agent-driven TLA+ pipeline entry points.

### P2 — Nice to have, lower effort

8. **RETURNING clause injection** (integrations-and-detection) — captures autoincrement
   IDs from PostgreSQL INSERTs; also needed by cross-process exploration.
9. **Marker coverage tracking** (testing-strategies) — report which interleavings were
   actually exercised.
10. **Wakeup tree equivalence checking** (dpor-improvements: Phase 4c) — sound
    optimization; benefit depends on workload.
11. **sys.addaudithook integration** / **sys.monitoring CALL events**
    (integrations-and-detection) — zero-config I/O safety net; lower-overhead detection
    on Python 3.12+.

### P3 — Deferred / exploratory

12. **Spec-guided schedule generation from TLC** (formal-methods: 2.3) — replace random
    exploration with TLC-enumerated behaviors.
13. **Refinement checking** (formal-methods: 3.1) — requires the P1 formal-methods items.
14. **Wire-protocol parsing at LD_PRELOAD level** (integrations-and-detection) — revisit
    if cross-process phase 4 (unmodified/non-Python workers) is pursued.
15. **Trace fingerprinting with coverage feedback** (random_dpor: Proposal B) — hash
    reads-from relations; adaptively skip stale backtrack points.
16. **Depth-biased backtrack selection** (random_dpor: Proposal D) — explore/exploit
    trade-off across search phases.
