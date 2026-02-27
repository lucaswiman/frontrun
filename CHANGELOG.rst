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

**LD_PRELOAD events wired into DPOR**

When run under the ``frontrun`` CLI with ``detect_io=True`` (the default),
``explore_dpor()`` now consumes C-level I/O events from the ``LD_PRELOAD``
library via ``IOEventDispatcher`` → ``_PreloadBridge``.  This means DPOR
detects races involving opaque C extensions (e.g. psycopg2/libpq calling
libc ``send()``/``recv()`` directly) that previously went unnoticed.

**Improved DPOR race detection**

- Global variable and module-level attribute accesses are now tracked as
  shared-memory conflicts.
- C-level container mutations (``list.append``, ``dict.__setitem__``, etc.)
  detected via ``sys.setprofile``.
- Closure variable (``LOAD_DEREF`` / ``STORE_DEREF``) accesses tracked.
- Builtin function calls that mutate containers (e.g. ``sorted()``,
  ``list()``) treated as reads on their arguments.
- Container iteration (``GET_ITER`` / ``FOR_ITER``) tracked as reads.
- Known limitation: C-level iteration interleaving (e.g.
  ``list(od.keys())`` vs ``OrderedDict.move_to_end()``, or ``itertools``
  combinators racing with mutations) is undetectable at the opcode level.
  See ``PEP-703-REPORT.md``.

**Other changes**

- Cooperative threading primitives (``Lock``, ``RLock``, ``Semaphore``,
  ``Event``, ``Condition``, ``Queue``, etc.) extracted to a shared module so
  both the bytecode explorer and DPOR use the same wrappers.
- Deadlock detection via wait-for graph cycle detection.
- ``--frontrun-patch-locks`` pytest plugin for early cooperative patching;
  tests that need the frontrun environment are auto-skipped when it is absent.
- Free-threaded Python (3.13t, 3.14t) support, including a fix for a PyO3
  "Already borrowed" panic and intermittent hangs in cooperative lock
  patching on 3.14t.
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
