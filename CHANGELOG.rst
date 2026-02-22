Changelog
=========

All releases: https://github.com/lucaswiman/frontrun/releases

0.1.0 (Unreleased)
-------------------

**DPOR (Dynamic Partial Order Reduction)**

Systematic interleaving exploration via a Rust engine (``frontrun._dpor``,
built with PyO3/maturin).  Unlike the bytecode explorer which samples random
schedules, DPOR guarantees that every meaningfully distinct interleaving is
tried exactly once.  Shared-memory conflicts are detected automatically at the
bytecode level using a shadow stack; vector clocks prune redundant orderings.

**Automatic I/O detection**

Both the bytecode explorer and DPOR now detect socket and file I/O
automatically (``detect_io=True`` by default).  When two threads touch the same
network endpoint or file path the scheduler treats the operations as
conflicting and explores their reorderings.

**C-level I/O interception**

A new ``frontrun`` CLI wraps any command with an ``LD_PRELOAD`` library
(``libfrontrun_io.so``, built from ``crates/io/``) that intercepts libc I/O
functions (``connect``, ``send``, ``recv``, ``read``, ``write``, etc.).  This
covers opaque C extensions such as database drivers, Redis clients, and HTTP
libraries.

**Interpretable error messages**

When a race condition is found, ``result.explanation`` now contains a
human-readable trace showing interleaved source lines, the conflict pattern
(lost update, write–write, etc.), and reproduction statistics.

**Other changes**

- Cooperative threading primitives (``Lock``, ``RLock``, ``Semaphore``,
  ``Event``, ``Condition``, ``Queue``, etc.) extracted to a shared module so
  both the bytecode explorer and DPOR use the same wrappers.
- Deadlock detection via wait-for graph cycle detection.
- ``--frontrun-patch-locks`` pytest plugin for early cooperative patching;
  tests that need the frontrun environment are auto-skipped when it is absent.
- Free-threaded Python (3.13t, 3.14t) support, including a fix for a PyO3
  "Already borrowed" panic.
- Multi-version test matrix: Python 3.10, 3.14, 3.14t.
- ``DporResult`` merged into ``InterleavingResult``; all three ``explore_*``
  functions return the same type.
- Improved ``_dpor`` import error with build instructions.

0.0.2 (2026-02-17)
-------------------

Rename library from interlace to frontrun.

0.0.1 (2026-02-17)
-------------------

Initial release (as "interlace").  Includes trace markers, bytecode
exploration, and async variants.
