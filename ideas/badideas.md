# Salvageable Metaprogramming for Resource Detection

Techniques from the dark arts that are actually practical. Evaluated against the
existing DPOR shadow-stack architecture (`dpor.py`) and `_tracing.py` filtering.

---

## `__class__` Reassignment (Body-Snatching)

Python lets you swap an object's type at runtime. The object keeps all its state
(`__dict__`, C-level fields) but dispatches through the new type. Unlike a proxy
wrapper, `isinstance()` still works — the new type is a subclass.

```python
class InstrumentedConnection(type(conn)):
    def execute(self, *args, **kwargs):
        report_access(id(self), AccessKind.WRITE)
        return super().execute(*args, **kwargs)

conn.__class__ = InstrumentedConnection
```

**Why it works here:** DPOR already has `PY_RETURN` handlers (`handle_py_return`
at `dpor.py:581`). Intercept return values and class-swap anything that looks
like a resource:

```python
def handle_py_return(code, instruction_offset, retval):
    if _is_resource_like(retval) and not _already_instrumented(retval):
        retval.__class__ = _make_instrumented_subclass(type(retval))
```

The `_is_resource_like` check can be a duck-type test (`hasattr(retval,
"execute") or hasattr(retval, "send")`) or check against known base classes.
Once swapped, the object self-reports every method call to the DPOR engine —
no further tracing overhead needed for that object.

**Limitation:** `__class__` assignment only works on heap types with compatible
`__slots__`/C layouts. Pure-Python library objects (SQLAlchemy `Session`,
`Connection`, etc.) — fine. C extension objects like `sqlite3.Cursor` —
raises `TypeError`. For those, fall back to `sys.setprofile` (below).

**Verified:** See `ideas/experiments/test_class_reassignment.py`. `isinstance()`
preserved, attributes preserved, thread-safe (10 concurrent swaps, 0 errors).
Auto-swap from `sys.settrace` return event works. Confirmed failure on
`socket.socket`, `sqlite3.Cursor`, and `io.StringIO` (all C extension types).

---

## `sys.setprofile` / `sys.monitoring` C_CALL Events

`sys.settrace` doesn't fire for C function calls. `_tracing.py:should_trace_file`
skips all library/site-packages code, so the DPOR shadow stack goes dark inside
C extensions. But `sys.setprofile` fires `c_call`/`c_return` events for those
exact calls:

```python
def profile_func(frame, event, arg):
    if event == "c_call":
        # arg is the C function object being called
        if arg is socket.socket.send:
            report_resource_access("socket", AccessKind.WRITE)
        elif arg is socket.socket.recv:
            report_resource_access("socket", AccessKind.READ)
    return profile_func

sys.setprofile(profile_func)
```

**Integration with DPOR:** On the `sys.settrace` path (3.10-3.11), add
`sys.setprofile` right next to it in `_run_thread_settrace`. On the
`sys.monitoring` path (3.12+), PEP 669 already defines `C_RAISE` and
`C_RETURN` event types — add them to the `set_events` bitmask. The profile
callback calls `_report_read`/`_report_write` with a synthetic `object_key`,
feeding directly into the existing Rust engine conflict model. No new Rust code
needed.

**Interaction with `sys.settrace`:** Not a problem. Profile fires around C
calls, trace fires around opcodes. They don't overlap — they fire at different
times on different events.

**Verified:** See `ideas/experiments/test_setprofile_c_calls.py`. `socket.connect`,
`socket.sendall`, `socket.recv` all detected. Per-thread profiles are independent.
Coexists with `sys.settrace` — both fire simultaneously. On 3.13 via
`sys.monitoring`, INSTRUCTION + CALL events coexist on the same tool ID
(see `test_monitoring_c_call.py`).

---

## `gc.get_referrers()` One-Shot Resource Discovery

Not for continuous use — the cost per call is high. But as a **one-shot resource
identification** mechanism triggered by an audit hook or profile callback:

1. `sys.addaudithook` fires on `socket.connect(("localhost", 5432))`
2. You have the socket object. Call `gc.get_referrers(sock)` once.
3. Walk up: socket → `Connection._sock` → `Connection` → `Engine._pool` → `Engine`
4. Now `id(engine)` is the resource identity for endpoint `("localhost", 5432)`.
5. Cache that mapping. Never walk again.

```python
def find_owner(obj, max_depth=10):
    """Walk gc.get_referrers upward to find the high-level resource owner."""
    current = obj
    for _ in range(max_depth):
        referrers = [
            r for r in gc.get_referrers(current)
            if not isinstance(r, (types.FrameType, dict, list))
            and r is not current
        ]
        if not referrers:
            return current
        current = referrers[0]
    return current
```

This is a **startup-time operation**, not per-opcode. Run once when the first
I/O to a new endpoint is detected, cache the result. GC non-determinism doesn't
matter because you're just using it for resource *identification* — the actual
conflict detection still goes through deterministic vector clocks in the Rust
engine.

**Verified:** See `ideas/experiments/test_gc_referrers.py`. Successfully walks
`socket → DBConnection → ConnectionPool → Engine`. Key finding:
`gc.get_referrers(obj)` returns `__dict__` (a plain dict), not the owning
object — must walk *through* dicts by checking `getattr(dr, "__dict__") is c`.
Two connections from the same pool both resolve to the same `id(engine)`.
Cold walk: ~1.5ms (trivial graph) to ~25ms (27K objects). Cache lookup: ~1µs
(1300x speedup).

---

## Frame-Local Variable Poisoning (Narrow Use)

On 3.13+ `frame.f_locals` is natively writable (PEP 667). On older versions,
`ctypes.pythonapi.PyFrame_LocalsToFast` does the same thing (used by pdb,
pydevd). This lets you swap a function's arguments with proxies before the
function body executes:

```python
def trace_func(frame, event, arg):
    if event == "call":
        for name, val in frame.f_locals.items():
            if looks_like_resource(val):
                frame.f_locals[name] = ResourceProxy(val, resource_id=name)
                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(frame), ctypes.c_int(0))
    return trace_func
```

**When to prefer this over `__class__` reassignment:** When the object is a C
extension type that doesn't support `__class__` swapping (e.g. `sqlite3.Cursor`).
The proxy approach works on anything, at the cost of breaking `isinstance()`
checks within the wrapped function.

**Limitation:** Only works for local variables/arguments, not attributes or
globals. Version-sensitive (`PyFrame_LocalsToFast` is CPython-specific). Prefer
`__class__` swapping or `sys.setprofile` when possible.

---

## Trace-Time Lock `acquire`/`release` Interception (Lovecraftian Horror)

Instead of monkeypatching `threading.Lock` globally, intercept `lock.acquire()`
and `lock.release()` calls *inside the trace callback* — the same `sys.settrace`
/ `sys.monitoring` machinery that already runs for every opcode.

**The idea:** when the tracer sees a `CALL` opcode targeting a bound method like
`lock.acquire`, replace the blocking call with a cooperative spin-yield loop
before the real instruction executes. No global mutation, perfectly scoped to
traced runs, works for locks created at any time.

**Why it doesn't work cleanly:** trace callbacks fire *around* opcodes, not
*instead of* them. You cannot prevent the real `CALL` instruction from executing.
Options for working around this, each worse than the last:

1. **Pre-acquire with `lock.acquire(blocking=False)` in the trace callback,**
   spinning and yielding scheduler turns until it succeeds. Then the real `CALL`
   instruction's blocking acquire succeeds instantly. But now you've acquired
   twice — you need to release once to compensate, creating a window where the
   lock is unheld between your compensating release and the real acquire. Race
   condition in your race condition detector.

2. **Replace the bound method on the eval stack** before `CALL` executes, using
   `ctypes` to poke at the frame's value stack. Undocumented, changes across
   CPython versions, and the eval stack layout differs between 3.10, 3.11, 3.12,
   3.13, and 3.14. Five layouts to reverse-engineer and maintain.

3. **Swap `f_locals` to point at a wrapper** via PEP 667 / `PyFrame_LocalsToFast`.
   Only works when the lock is a local variable, not `self._lock.acquire()`.

4. **Use `sys.monitoring` `CALL` events** (3.12+). PEP 669 gives you the callee
   but no way to replace it or skip the call.

**Where it *could* be useful without the horror:** for DPOR conflict detection
(not interleaving control), the trace callback could detect `lock.acquire` calls
and report them to the Rust engine as sync events — improving happens-before
tracking for real locks without needing to control the interleaving. This is a
read-only observation, so none of the above problems apply. But it still requires
shadow-stack tracking through `LOAD_ATTR` to identify which object the `acquire`
is being called on, and `CALL` opcodes don't appear in DPOR's `_process_opcode`
explicit handling (they fall through to `dis.stack_effect`).

**Verdict:** Early monkey-patching via the pytest plugin (on by default) covers
the realistic cases. This approach is preserved here as a future possibility if
someone wants to eliminate global lock patching entirely and doesn't mind mass
`ctypes` eval-stack surgery.

---

## `gc.get_objects()` Lock Instance Replacement (`--frontrun-patch-locks=aggressive`)

Walk all live objects via `gc.get_objects()`, find real `_thread.lock` and
`_thread.RLock` instances, then use `gc.get_referrers()` to locate their
containers (dicts, lists, `__dict__`) and swap in cooperative wrappers.

```python
for lock in gc.get_objects():
    if type(lock) is real_lock_type:
        wrapper = CooperativeLock.__new__(CooperativeLock)
        wrapper._lock = lock
        for referrer in gc.get_referrers(lock):
            _try_replace(referrer, lock, wrapper)
```

**The appeal:** catches locks created before `patch_locks()` ran — e.g. stdlib
module-level locks, third-party libraries that create locks at import time.
Monkey-patching `threading.Lock` only affects *future* calls; this retroactively
fixes *existing* instances.

**Why it's problematic:**

1. **Recursive wrapping.** The wrapper's `_lock` attribute refers to the real
   lock. But `gc.get_referrers(real_lock)` also finds the wrapper's `__dict__`
   as a referrer, so `_try_replace` wraps the wrapper — `CooperativeLock` whose
   `_lock` is another `CooperativeLock` whose `_lock` is the real lock. Every
   method call recurses. The fix is to skip referrers that are already
   cooperative wrappers, but the check is fragile.

2. **Stdlib internal locks.** Python's logging, threading, importlib, and io
   modules all hold real locks in module-level or instance variables. Replacing
   them with cooperative wrappers means `logging.Handler.acquire()` now calls
   cooperative `acquire()` which imports `frontrun._deadlock` which may trigger
   logging… infinite recursion at shutdown. The `_is_frontrun_internal` guard
   only skips frontrun's own modules, not stdlib.

3. **Best-effort coverage.** Locks in tuples, frozensets, C extension structs,
   closure cells, and `__slots__` cannot be replaced. The user gets silent
   partial coverage with no way to know which locks were missed.

4. **Non-deterministic.** `gc.get_objects()` order depends on allocation
   history. Different test runs may replace different subsets of locks depending
   on import order, gc timing, and which objects have been collected.

5. **Wrapper construction via `__new__`.** Bypassing `__init__` requires
   manually initializing every field. When `CooperativeRLock` gains a new field,
   the `__new__` path silently produces broken instances (as happened with
   `_owner` and `_count`).

**Current status:** Removed. The gc-scanning code has been deleted from
`pytest_plugin.py`. The problems above (especially recursive wrapping and
stdlib lock interference) made it unreliable. Early monkey-patching (now on
by default) covers the realistic cases without these risks.

---

## Import Hook Bytecode Rewriting (Long-Term Option)

A standard technique used by coverage.py and crosshair. Register a custom
finder/loader on `sys.meta_path` that rewrites bytecode at import time to insert
instrumentation callbacks around specific operations.

**Where this would shine:** Eliminating `should_trace_file` entirely. Instead of
tracing every opcode in user code, rewrite bytecodes at import time to insert
callbacks only around `CALL` instructions whose target is a resource method. Zero
overhead on non-resource code paths.

**Why it's a long-term option, not an immediate one:** The shadow stack code in
`_process_opcode` already demonstrates the pain of supporting bytecodes across
Python 3.10 through 3.14t (~200 lines of version-specific opcode handling).
Bytecode rewriting would need to handle all the same version differences. For the
specific goal of "detect resource accesses," `sys.setprofile` achieves the same
thing with ~20 lines.
