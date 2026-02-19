# Explicit Cooperative Primitives (Loom-Style)

Alternative to monkey-patching for production use.

## Idea

An alternative to monkey-patching is providing explicit cooperative replacements
that users opt into, inspired by the Rust
[loom](https://github.com/tokio-rs/loom) library.

In loom, instead of `std::sync::Mutex`, you use `loom::sync::Mutex`. Under test,
loom's Mutex cooperates with loom's scheduler. In production, it compiles down to
the real Mutex. The user changes imports, not runtime behavior.

The equivalent for frontrun:

```python
# Instead of:
import threading
lock = threading.Lock()

# User writes:
from frontrun.sync import Lock
lock = Lock()
```

Under `controlled_interleaving`, `frontrun.sync.Lock` behaves like
`_CooperativeLock`. Outside of it, it delegates to `threading.Lock` with zero
overhead.

## Advantages over monkey-patching

- **No global state mutation**: Patching `threading.Lock` affects the entire
  process, including the scheduler, unrelated libraries, and other test threads.
  Explicit imports are scoped to the code that uses them.
- **No recursion risk**: The scheduler uses real `threading.Condition` by
  construction -- there's nothing to accidentally patch.
- **Clearer intent**: Reading the code makes it obvious which primitives are
  under scheduler control.
- **Composable**: Multiple test harnesses can coexist without conflicting patches.

## Disadvantages

- **Requires import changes**: Can't test unmodified code. This is the main
  drawback -- the whole point of the bytecode approach is testing code as-is.
- **Dual-import maintenance**: `frontrun.sync` must mirror the `threading` and
  `queue` APIs exactly, and stay in sync as CPython evolves.

## Why this matters

The monkey-patching approach works for demos and simple cases, but the fragility
issues described in [known_issues.md](known_issues.md) (global state, internal
resolution leaks, parallel test collisions) make it a poor fit for production
test suites. The loom-style approach would be more robust for users willing to
change imports. Since all the cooperative wrapper implementations already exist,
`frontrun.sync` would just be thin dispatch wrappers over them.

## Implementation

- [ ] Design `frontrun.sync` module API (mirror `threading` and `queue`)
- [ ] Implement thin dispatch wrappers over existing cooperative primitives
- [ ] Add mode detection (controlled vs. production)
