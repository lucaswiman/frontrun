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
