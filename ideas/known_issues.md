# Known Issues and Limitations

## Monkey-Patching Fragility (Partially Fixed)

The bytecode approach patches `threading.Lock`, `threading.Semaphore`, etc. at the
module level, which is global mutable state. This creates several problems:

1. **Internal resolution leaks**: When stdlib code resolves names from
   `threading`'s module globals, it picks up the patched cooperative versions
   instead of the real ones. For example, `BoundedSemaphore.__init__` resolves
   `Semaphore` from `threading`'s module globals, getting our patched version.
   Every new primitive risks similar interactions. **Mitigated**: cooperative
   primitives check TLS scheduler context and fall back to real behaviour when
   no scheduler is active, so stdlib code on unmanaged threads works correctly.

2. **Import-time lock creation**: Libraries that create locks at import time
   (before patching) will hold real locks. **Mitigated**: the pytest plugin
   calls ``patch_locks()`` in ``pytest_configure`` — before test collection
   imports any modules — so that module-level ``threading.Lock()`` calls
   in the code under test create cooperative locks.  Patching is on by
   default; disable with ``--no-frontrun-patch-locks``.

## Random Exploration Lacks Coverage Guarantees (Improved)

`explore_interleavings()` generates random schedules, which provides no feedback
about how much of the interleaving space has been covered. For simple programs
(a few opcodes, 2 threads), random works well. For anything with loops or
complex synchronization, you might need thousands of attempts to hit the one bad
interleaving, with no way to know if you've missed it. See
[dpor_spec.md](dpor_spec.md) for the principled solution.

**Improved.** `InterleavingResult` now includes a `unique_interleavings` field
that reports how many distinct schedule orderings were actually observed during
exploration.  This provides a lower bound on coverage — if `unique_interleavings`
is much less than `num_explored`, the exploration is converging and additional
attempts are unlikely to find new behaviour.  Both the sync and async
`explore_interleavings()` functions populate this field.
