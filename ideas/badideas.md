# Terrible Metaprogramming Ideas for Detecting Stateful Resources

The dark arts of Python runtime manipulation, ranked by cursedness.

---

## 1. `__class__` Reassignment Drive-By

Python lets you swap an object's type *at runtime*:

```python
class InstrumentedConnection(type(conn)):
    def execute(self, *args, **kwargs):
        report_access(id(self), AccessKind.WRITE)
        return super().execute(*args, **kwargs)

conn.__class__ = InstrumentedConnection
```

The object keeps all its state (`__dict__`, C-level fields) but now dispatches through your instrumented type. No proxy, no wrapper, the object *becomes* instrumented. Works on any heap type (so not `int` or `str`, but most library objects).

**The terrible part:** Do it from `sys.settrace`. When you see a `RETURN_VALUE` and the return value has a `.execute` method, silently reassign its class:

```python
def trace_func(frame, event, arg):
    if event == "return" and hasattr(arg, "execute"):
        # You just called session.cursor()? That cursor is ours now.
        arg.__class__ = make_instrumented_subclass(type(arg))
    return trace_func
```

Every cursor, every connection, every file handle that passes through a return value gets body-snatched.

---

## 2. `ctypes.pythonapi` Type Slot Surgery

Can't monkey-patch `socket.socket.send` because it's a C builtin? CPython stores method implementations in *type slots* — C function pointers in the `PyTypeObject` struct. You can overwrite them with `ctypes`:

```python
import ctypes

# PyTypeObject layout (simplified, version-dependent)
class PyTypeObject(ctypes.Structure):
    _fields_ = [
        # ... many fields ...
        ("tp_getattro", ctypes.CFUNCTYPE(
            ctypes.py_object, ctypes.py_object, ctypes.py_object)),
    ]

# Get the actual type object's address
type_obj = PyTypeObject.from_address(id(socket.socket))

# Save original
original_getattro = type_obj.tp_getattro

# Install our version
@ctypes.CFUNCTYPE(ctypes.py_object, ctypes.py_object, ctypes.py_object)
def evil_getattro(self_ptr, name_ptr):
    result = original_getattro(self_ptr, name_ptr)
    # ... instrument here ...
    return result

type_obj.tp_getattro = evil_getattro
```

You just patched attribute access on a C extension type. Every `sock.send`, `sock.recv`, `sock.connect` now goes through your code. **On every socket object in the entire process.** Including ones in the standard library's `http.client`, `urllib`, `smtplib`...

**Segfault probability:** nonzero. **Fun factor:** immeasurable.

---

## 3. Frame-Local Variable Poisoning via `sys.settrace`

The trace function can read `frame.f_locals`. On CPython, you can also *write* to it with `ctypes` (or on 3.13+ with `frame.f_locals` being writable):

```python
def trace_func(frame, event, arg):
    if event == "call":
        for name, val in frame.f_locals.items():
            if looks_like_resource(val):
                # Replace the local variable with a proxy
                frame.f_locals[name] = ResourceProxy(
                    val, resource_id=f"{name}@{frame.f_code.co_filename}")
                # Force the frame to re-read f_locals into fastlocals
                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(frame), ctypes.c_int(0)
                )
    return trace_func
```

The function *thinks* it has a normal connection object. It actually has a proxy. You replaced its local variables while it wasn't looking. The function never consented to this.

Works at function entry (swap arguments), at every line (swap anything that appeared), even at returns (swap before the caller sees the result).

---

## 4. Import Hook Bytecode Rewriting

Register a custom finder/loader on `sys.meta_path` that intercepts every import and rewrites the bytecode before the module executes:

```python
import dis, types

class BytecodeRewriter:
    def find_module(self, name, path=None):
        return self  # we'll handle everything, thanks

    def load_module(self, name):
        # Let the real import happen
        spec = importlib.util.find_spec(name)
        source = spec.loader.get_source(name)
        code = compile(source, spec.origin, "exec")

        # Rewrite the bytecode
        code = rewrite_calls(code)

        mod = types.ModuleType(name)
        exec(code, mod.__dict__)
        sys.modules[name] = mod
        return mod

def rewrite_calls(code):
    """Replace CALL instructions with instrumented versions."""
    # Walk the bytecode, find every CALL_FUNCTION/CALL_METHOD
    # Inject a CALL to our instrumentation before each one
    # Recursively rewrite code objects in co_consts (nested functions)
    ...
```

Every function call in every imported module now goes through your instrumentation layer. You can check if the callee is a resource method, log it, intercept it, whatever.

**The truly terrible variant:** Don't just instrument calls. Rewrite `STORE_ATTR` to `CALL_FUNCTION(instrumented_setattr, obj, name, value)`. Now you've turned every attribute write into a function call you control. You've reimplemented `sys.settrace` but worse and in bytecode.

---

## 5. Metaclass Virus via `__init_subclass__`

```python
_original_init_subclass = object.__init_subclass__

def evil_init_subclass(cls, **kwargs):
    _original_init_subclass(**kwargs)
    # Wrap every method that looks stateful
    for name in list(vars(cls)):
        if name in ("execute", "send", "recv", "write", "read", "commit", "flush"):
            original = vars(cls)[name]
            if callable(original):
                @functools.wraps(original)
                def wrapper(*args, _orig=original, _name=name, **kw):
                    report_resource_access(_name)
                    return _orig(*args, **kw)
                setattr(cls, name, wrapper)

# Install on object itself (requires ctypes slot manipulation from #2)
patch_slot(object, "__init_subclass__", classmethod(evil_init_subclass))
```

Every class defined from this point forward that has an `execute`, `send`, `write`, etc. method gets automatically wrapped. The instrumentation *propagates through inheritance*. Define a new DB adapter class? Already instrumented. Subclass `io.BufferedWriter`? Instrumented. It's a metaclass virus.

---

## 6. `gc.get_referrers()` Chain Walking (Ambient Object Graph Analysis)

Don't instrument anything. Instead, periodically scan the entire object graph:

```python
import gc, socket

def find_resource_owners():
    """Walk backwards from every socket to find who owns it."""
    for obj in gc.get_objects():
        if isinstance(obj, socket.socket):
            chain = []
            current = obj
            for _ in range(10):  # max depth
                referrers = gc.get_referrers(current)
                # Filter out frames, dicts, and this function's locals
                referrers = [
                    r for r in referrers
                    if not isinstance(r, (types.FrameType, dict))
                ]
                if not referrers:
                    break
                current = referrers[0]
                chain.append(current)
            # chain is now [cursor, session, engine, module, ...]
            yield obj, chain
```

Call this from your `sys.settrace` callback periodically. When a socket does I/O (detected via audit hooks), walk `gc.get_referrers` to find the SQLAlchemy `Engine` that owns it. Resource identity = `id(engine)`, not `id(socket)`. Two threads using different sockets but the same engine → conflict detected.

**Runtime cost:** yes. **Correctness:** surprisingly decent because Python's GC actually maintains the referrer graph accurately. **Determinism:** absolutely not, because GC is non-deterministic, so you might find different referrer chains on different runs.

---

## 7. `code.replace()` Live Code Object Swapping

Python 3.8+ lets you create modified copies of code objects:

```python
def instrument_function(func):
    """Replace a function's code object with an instrumented version."""
    old_code = func.__code__

    # Build a wrapper code object that calls our hook then the original
    hook_code = compile(
        "__frontrun_hook__(); __frontrun_original__()",
        old_code.co_filename,
        "exec"
    )

    # Or more surgically: prepend instructions to the existing bytecode
    new_bytecode = (
        bytes([LOAD_GLOBAL, hook_name_index, CALL_FUNCTION, 0, POP_TOP])
        + old_code.co_code
    )

    func.__code__ = old_code.replace(co_code=new_bytecode)
```

You just modified a live function's bytecode. Every existing reference to this function (closures, bound methods, callbacks registered in C code) now runs your instrumented version because they all share the same function object. No need to find all call sites.

**The worst variant:** Do it to every Python function in the process that *calls* resource-like methods:

```python
for obj in gc.get_objects():
    if isinstance(obj, types.FunctionType):
        if any(name in obj.__code__.co_names
               for name in ("send", "execute", "write")):
            instrument_function(obj)
```

Walk every function object in the process, check if its bytecode references resource-like names, and rewrite it if so. On-the-fly. While threads are running.

---

## 8. `sys.setprofile` for C Function Call Interception

`sys.settrace` doesn't fire for C function calls. But `sys.setprofile` does:

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

`c_call` events fire *before* the C function executes. `c_return` fires after. You get full visibility into C extension calls. Combined with the DPOR scheduler, you can pause a thread right before it does `socket.send` and switch to another thread.

**The catch:** `sys.setprofile` and `sys.settrace` interact poorly. You can have both installed, but the profile function fires at different granularity. The DPOR shadow stack (which relies on `sys.settrace` for opcode events) would need to coordinate with the profile function (which sees C calls). It's two instrumentation systems running on the same frame simultaneously.

---

## 9. `__getattribute__` on `object` via Forbidden Fruit

The [`forbiddenfruit`](https://pypi.org/project/forbiddenfruit/) library (or raw ctypes) lets you patch methods on builtin types:

```python
from forbiddenfruit import curse

original_getattribute = object.__getattribute__

def cursed_getattribute(self, name):
    result = original_getattribute(self, name)
    if name in ("execute", "send", "write", "commit") and callable(result):
        @functools.wraps(result)
        def wrapper(*args, **kwargs):
            report_resource_access(type(self).__name__, name)
            return result(*args, **kwargs)
        return wrapper
    return result

curse(object, "__getattribute__", cursed_getattribute)
```

You just patched `object.__getattribute__`. **Every attribute access on every object in the entire Python process** now goes through your function. Every `.append()` on a list. Every `.format()` on a string. Every `.execute()` on a cursor.

**Performance impact:** catastrophic. Every single attribute lookup in the process now has Python-level overhead instead of a C fast path. Your program will be approximately 100-1000x slower.

**But:** you will *never miss a resource access*. Ever.

---

## 10. The Unholy Synthesis: `eval` Frame Injection

```python
def trace_func(frame, event, arg):
    if event == "call" and frame.f_code.co_filename.endswith("psycopg2/cursor.py"):
        # We're entering psycopg2's cursor code
        # Inject a variable into the frame that reports back to us
        import ctypes
        frame.f_locals["__frontrun_spy__"] = lambda: report_resource_access("postgres")
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), 0)

        # Now rewrite the code object to call __frontrun_spy__() at entry
        old_code = frame.f_code
        spy_bytecode = compile("__frontrun_spy__()", "<frontrun>", "exec").co_code
        frame.f_code = old_code.replace(
            co_code=spy_bytecode + old_code.co_code
        )
```

You detected entry into a library function via `sys.settrace`, injected a spy variable into its local scope via `ctypes`, then rewrote its bytecode to call the spy. The library function has been compromised from the inside. It will report its own execution to your scheduler without knowing it.

---

## Terrible Ideas Tier List

| Tier | Technique | Segfault Risk | Magic Level |
|------|-----------|--------------|-------------|
| S | `ctypes` type slot surgery | HIGH | Eldritch |
| S | `object.__getattribute__` curse | LOW | Omniscient |
| A | Frame local variable poisoning | MEDIUM | Parasite |
| A | Import hook bytecode rewriting | LOW | Compiler |
| A | `__class__` reassignment drive-by | LOW | Body-snatcher |
| B | `gc.get_referrers` chain walking | LOW | Stalker |
| B | `code.replace()` live swapping | MEDIUM | Surgeon |
| B | `sys.setprofile` C call interception | LOW | Wiretap |
| C | `__init_subclass__` metaclass virus | LOW-MEDIUM | Pandemic |
| C | `eval` frame injection | HIGH | Possession |
