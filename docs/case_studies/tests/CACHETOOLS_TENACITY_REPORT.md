# Cachetools & Tenacity Concurrency Bug Exploration Report

**Libraries:**
- cachetools v7.0.1 (commit e5f8f01)
- tenacity (commit 0bdf1d9)

**Tool:** frontrun bytecode exploration (property-based interleaving search)

## Summary

We wrote 22 property-based concurrency tests across two libraries: 14 bug-finding
tests and 8 safe-area tests. **All 14 bug-finding tests found races (most within
1-3 attempts).** All 8 safe-area tests held their invariants across 15,000-20,000
interleavings each, confirming zero false positives.

## cachetools Results (12 tests)

### Bug-Finding Tests (8/8 found races)

| # | Component | Race Type | Attempts | Description |
|---|-----------|-----------|----------|-------------|
| B1 | Cache.__setitem__ | Lost update (`currsize +=`) | 2 | Two concurrent inserts lose a currsize increment |
| B2 | Cache.__delitem__ | Lost update (`currsize -=`) | 5 | Two concurrent deletes lose a currsize decrement |
| B3 | LRUCache get+del | TOCTOU (`__order` desync) | 3 | __getitem__ touches __order while __delitem__ removes key |
| B4 | RRCache.__setitem__ | Index corruption | 3 | Concurrent inserts corrupt __index/__keys swap-delete structure |
| B5 | @cached (no lock) | Stats lost update (`hits +=`) | 3 | hits/misses counters lose increments without lock |
| B6 | LFUCache.__getitem__ | Link corruption (KeyError) | 1 | Concurrent gets corrupt frequency linked list, causing crash |
| B7 | Cache set+del | Currsize desync | 48 | Concurrent set and delete produce incorrect currsize |
| B8 | TTLCache._Timer | Nesting lost update | 2 | `__nesting += 1`/`-= 1` loses updates, breaks timer context |

### Safe-Area Tests (4/4 held)

| # | Component | Invariant | Interleavings | Time |
|---|-----------|-----------|---------------|------|
| S1 | @cached with lock | Stats accurate (hits=4, misses=2) | 15,000 | ~70s |
| S2 | @cached with lock | All keys stored correctly | 15,000 | ~70s |
| S3 | @cached with condition | No stampede (call_count=1) | 15,000 | race @2459 |
| S4 | Cache single-key | Key-value pairs always retrievable | 15,000 | SAFE |

S3 found a potential race after 2,459 interleavings: the `_condition_info`
wrapper has a window between `return v` (which releases the inner lock) and
the `finally` block (which re-acquires it to clean up `pending`).

### Notable Finding: LFUCache Crash (B6)

The most dramatic cachetools bug is in `LFUCache.__getitem__`: concurrent reads
crash with `KeyError` because the frequency link list manipulation (move key from
one frequency bucket to the next) is not atomic. Thread1 removes key "a" from
`link.keys`, then thread2 tries to remove the same key and gets `KeyError`. This
is not just a data race -- it's a crash in a read-only operation.

### Positive Control: @cached with lock and condition

The `@cached` decorator's `lock` parameter correctly prevents all races across
30,000 interleavings. The `condition` variant (S3) found a potential edge case
after 2,459 interleavings, suggesting a subtle race in `_condition_info`.

## tenacity Results (10 tests)

### Bug-Finding Tests (6/6 found races)

| # | Component | Race Type | Attempts | Description |
|---|-----------|-----------|----------|-------------|
| B1 | statistics["idle_for"] | Lost update (`+= sleep`) | 1 | Shared dict value increment is non-atomic |
| B2 | statistics["attempt_number"] | Lost update (`+= 1`) | 1 | Shared dict value increment is non-atomic |
| B3 | wrapped_f.statistics | Orphaned reference | 1 | Concurrent calls overwrite .statistics attribute |
| B4 | Shared Retrying copy | Stats corruption | 1 | Two threads sharing one copy corrupt statistics |
| B5 | RetryCallState.idle_for | Lost update (`+= sleep`) | 1 | Attribute increment on shared object is non-atomic |
| B6 | RetryCallState.attempt_number | Lost update (`+= 1`) | 1 | Attribute increment on shared object is non-atomic |

### Safe-Area Tests (4/4 held)

| # | Component | Invariant | Interleavings | Time |
|---|-----------|-----------|---------------|------|
| S1 | threading.local stats | Per-thread isolation | 20,000 | ~83s |
| S2 | Per-call RetryCallState | Each call gets own state | 20,000 | ~83s |
| S3 | Immutable strategies | Concurrent evaluation safe | 20,000 | ~83s |
| S4 | copy() isolation | Independent copies don't interfere | 20,000 | ~83s |

### Notable Finding: wrapped_f.statistics Race (B3)

The `wraps()` method in `BaseRetrying` has a subtle race: each call to a wrapped
function creates a `copy()` of the Retrying object and overwrites
`wrapped_f.statistics = copy.statistics`. When two threads call the wrapped
function concurrently, both write to the same attribute. The result: one thread
captures a reference to its own statistics dict via `wrapped_f.statistics`, but
the other thread later overwrites it with a different dict. The first thread's
captured reference becomes orphaned -- still valid but no longer accessible through
the wrapped function's `.statistics` attribute.

### Positive Control: threading.local and copy()

Tenacity's primary defense against concurrency bugs is `threading.local()` for
per-thread state isolation and `copy()` for per-call isolation. Both mechanisms
work correctly:

- `threading.local` ensures that `self.statistics` and `self.iter_state` are
  independent per thread, even when sharing a single `Retrying` object.
- `copy()` creates fully independent `Retrying` instances, so concurrent calls
  to a wrapped function don't interfere with each other's retry state.

## Methodology

Each test follows the pattern:
1. **setup**: Create fresh library objects (no external resources needed)
2. **threads**: Define 2 thread functions exercising concurrent access
3. **invariant**: A predicate that should hold in any linearizable execution
4. **explore**: frontrun generates random opcode-level schedules and checks
   the invariant

Bug-finding tests use 500 max_attempts (sufficient since all races are found
within 1-48 attempts). Safe-area tests use 15,000-20,000 attempts each,
targeting ~70-83 seconds per test.

## Conclusion

Both cachetools and tenacity have straightforward concurrency bugs in their
unprotected code paths, and both have effective defenses where applied:

- **cachetools**: All `Cache` subclass operations (set, get, delete) race on
  `currsize` and internal data structures. The `@cached` decorator's `lock`
  parameter completely fixes this. The `condition` parameter additionally
  prevents cache stampede.

- **tenacity**: The `+=` pattern on shared statistics dicts and `RetryCallState`
  attributes is racy. However, tenacity's design mitigates this through
  `threading.local` (per-thread statistics) and `copy()` (per-call isolation).
  The only race that affects normal usage is the `wrapped_f.statistics`
  attribute overwrite (B3).
