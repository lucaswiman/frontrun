# Detecting Stateful Resource Access at Runtime

## The Core Problem

DPOR currently tracks conflicts via Python object attribute accesses (`LOAD_ATTR`/`STORE_ATTR` on `id(obj)`). But when shared state lives *outside* the process — in a database, on a filesystem, behind a socket — two threads can both call `cursor.execute("UPDATE ...")` using *different* Python cursor objects. DPOR sees no conflict because the Python objects are distinct. The shared state is invisible.

The question is: how do you detect these external-resource accesses with minimal (or zero) configuration, and feed them into the existing interleaving machinery?

---

## Layer 0: `sys.addaudithook` (The Closest Thing to Magic)

Python 3.8+ has [audit hooks](https://docs.python.org/3/library/sys.html#sys.addaudithook) that fire on security-sensitive operations — and they fire **from C code**, not just Python. Relevant events:

| Audit event | What it catches |
|---|---|
| `open` | All file opens (including `sqlite3` DB files) |
| `socket.connect` | TCP/Unix socket connections (DB drivers, HTTP clients) |
| `socket.sendmsg` | Data sent over sockets |
| `socket.bind` / `socket.listen` | Server sockets |
| `sqlite3.connect` | SQLite specifically |
| `subprocess.Popen` | Subprocess creation |

```python
import sys

def resource_hook(event, args):
    if event == "socket.connect":
        sock, address = args  # address is (host, port)
        # Now we know this thread is talking to a specific endpoint
    elif event == "open":
        filename, mode, flags = args
        # File I/O — filename is the resource identity

sys.addaudithook(resource_hook)
```

**Why this is powerful for frontrun:**

1. **Zero configuration.** No library-specific knowledge needed. Works with psycopg2, redis-py, pymongo, urllib3 — anything that eventually hits Python's socket/file layer.

2. **Natural resource identity.** The `(host, port)` tuple from `socket.connect` is a perfect "resource ID" for DPOR's conflict model. Two threads writing to the same `(localhost, 5432)` → conflict. Two threads writing to different endpoints → independent.

3. **Fires inside C extensions.** The audit hook lives in CPython's C implementation of `socket_connect`, `builtin_open`, etc. Even if a C extension calls `PyObject_Call(socket_connect, ...)`, the hook fires.

**Limitations:**
- C extensions that bypass Python's socket module entirely (e.g., libpq doing raw `connect()` syscalls) won't trigger it. In practice, most Python DB drivers go through Python's socket layer at least for connection setup.
- Granularity is coarse: you know "thread X sent data to postgres" but not "thread X updated table `accounts`". Everything to the same endpoint looks like the same resource.
- Audit hooks can't be removed once installed (by design). Need to gate on "are we in a frontrun test run?" via a flag.

**Integration with DPOR:** Extend the `object_id` → `ObjectState` conflict model to also track `endpoint_id` → `EndpointState`. When the audit hook fires `socket.sendmsg` for endpoint `(host, port)`, report it to the Rust engine as a write access on a virtual "object" representing that endpoint.

---

## Layer 1: Socket/FD Monkey-Patching (More Control Than Audit Hooks)

Patch `socket.socket` methods directly. Unlike audit hooks, patches can be installed/removed per test and can carry richer context:

```python
_real_send = socket.socket.send

def _traced_send(self, data, *args):
    endpoint = self.getpeername()  # (host, port)
    scheduler.report_resource_access(
        resource_id=("socket", endpoint),
        kind=AccessKind.WRITE,
        metadata=data[:100],  # first 100 bytes for debugging
    )
    return _real_send(self, data, *args)
```

**Advantages over audit hooks:**
- Removable (restore original methods after test)
- Can inspect the data being sent (parse SQL out of the wire protocol?)
- Can attach to specific socket instances rather than globally
- Can intercept `recv` too, distinguishing reads from writes

**For file I/O:** Patch `builtins.open`, `os.read`, `os.write`. Resource identity = `os.path.realpath(filename)`. Mode `"r"` → read access, `"w"`/`"a"` → write access.

**This is what `_cooperative.py` already does for locks.** The pattern is established in the codebase — `frontrun/_cooperative.py` already monkey-patches `threading.Lock`, `queue.Queue`, etc. Extending this to `socket.socket` and `builtins.open` is a natural fit.

---

## Layer 1.5: `sys.setprofile` / `sys.monitoring` C_CALL Events

`sys.settrace` doesn't fire for C function calls. `_tracing.py:should_trace_file` skips all library/site-packages code, so DPOR's shadow stack goes dark inside C extensions like `psycopg2` or `sqlite3`. But `sys.setprofile` fires `c_call`/`c_return` events for those exact calls:

```python
def profile_func(frame, event, arg):
    if event == "c_call":
        if arg is socket.socket.send:
            report_resource_access("socket", AccessKind.WRITE)
        elif arg is socket.socket.recv:
            report_resource_access("socket", AccessKind.READ)

sys.setprofile(profile_func)
```

**Integration path:** On the `sys.settrace` path (3.10-3.11), install `sys.setprofile` alongside it in `_run_thread_settrace`. On the `sys.monitoring` path (3.12+), PEP 669 already defines `C_RAISE` and `C_RETURN` event types — add them to the `set_events` bitmask. The callback calls `_report_read`/`_report_write` with a synthetic `object_key`, feeding directly into the existing Rust engine. No new Rust code needed.

`sys.setprofile` and `sys.settrace` don't conflict: profile fires around C calls, trace fires around opcodes. Different events, different times.

---

## Layer 2: Taint Propagation (Proxy or `__class__` Reassignment)

The insight: if you can identify the *root* resource object (an `Engine`, a `Connection`, a file handle), you can make everything derived from it self-reporting.

### Option A: `__class__` Reassignment (preferred)

Swap the object's type at runtime. The object keeps all its state but dispatches through an instrumented subclass. Unlike a proxy, `isinstance()` still works:

```python
class InstrumentedConnection(type(conn)):
    def execute(self, *args, **kwargs):
        report_access(id(self), AccessKind.WRITE)
        return super().execute(*args, **kwargs)

conn.__class__ = InstrumentedConnection
```

DPOR already has `PY_RETURN` handlers (`dpor.py:581`). Intercept return values there:

```python
def handle_py_return(code, instruction_offset, retval):
    if _is_resource_like(retval) and not _already_instrumented(retval):
        retval.__class__ = _make_instrumented_subclass(type(retval))
```

Every cursor, connection, or file handle that passes through a return value gets automatically instrumented. Once swapped, the object self-reports every method call — no further tracing overhead.

**Limitation:** Only works on heap types with compatible `__slots__`/C layout. Pure-Python library objects — fine. C extension types like `sqlite3.Cursor` — raises `TypeError`. For those, fall back to Option B or `sys.setprofile`.

### Option B: Proxy wrapper (explicit config)

```python
class ResourceProxy:
    """Transparent proxy that taints all return values."""
    def __init__(self, wrapped, resource_id):
        object.__setattr__(self, '_wrapped', wrapped)
        object.__setattr__(self, '_resource_id', resource_id)

    def __getattr__(self, name):
        val = getattr(self._wrapped, name)
        if callable(val):
            @functools.wraps(val)
            def traced_call(*args, **kwargs):
                scheduler.report_resource_access(
                    resource_id=self._resource_id,
                    kind=AccessKind.WRITE,  # conservative
                )
                result = val(*args, **kwargs)
                # Taint propagation: wrap return values too
                if hasattr(result, '__class__') and not isinstance(result, (int, str, float, bool, type(None))):
                    return ResourceProxy(result, self._resource_id)
                return result
            return traced_call
        return val
```

Now `engine.session().cursor().execute(...)` works:

```python
engine = ResourceProxy(create_engine("sqlite:///test.db"), resource_id="main-db")
# engine.session() → ResourceProxy(session, "main-db")
# .cursor()         → ResourceProxy(cursor, "main-db")
# .execute(...)     → reports access to "main-db", returns result
```

**Requires one line of config** (wrapping the root resource), but then propagates automatically through arbitrary call chains. Breaks `isinstance()` checks, so prefer `__class__` reassignment when possible.

### Auto-detecting roots with `gc.get_referrers()`

When an audit hook fires on `socket.connect(("localhost", 5432))`, walk `gc.get_referrers(sock)` up the reference chain to find the owning high-level object:

```python
def find_owner(obj, max_depth=10):
    """One-shot walk up gc.get_referrers to find the resource owner."""
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

socket → `Connection._sock` → `Connection` → `Engine._pool` → `Engine`. Cache the mapping `id(engine) → endpoint`. This is a startup-time operation, not per-opcode. GC non-determinism doesn't matter because you're just using it for resource *identification*; actual conflict detection still goes through deterministic vector clocks.

---

## Layer 3: `sys.monitoring` CALL Events (Python 3.12+)

`sys.monitoring` can fire on CALL events with much lower overhead than `sys.settrace`. You could monitor calls to known I/O functions:

```python
import sys

SENTINEL_FUNCTIONS = {
    socket.socket.send, socket.socket.recv, socket.socket.connect,
    builtins.open, os.read, os.write,
    # Could also detect by method name pattern:
    # anything named .execute(), .commit(), .rollback()
}

def call_handler(code, instruction_offset, callable, arg0):
    if callable in SENTINEL_FUNCTIONS:
        report_resource_access(...)

sys.monitoring.register_callback(
    sys.monitoring.TOOL_ID,
    sys.monitoring.events.CALL,
    call_handler
)
```

**The duck-typing variant:** Instead of a fixed set, detect by method name:

```python
RESOURCE_METHOD_NAMES = {"execute", "commit", "rollback", "send", "recv", "write", "read", "flush"}

def call_handler(code, offset, callable, arg0):
    name = getattr(callable, '__name__', '')
    if name in RESOURCE_METHOD_NAMES:
        # Heuristic: this looks like a stateful resource operation
        resource_id = id(arg0) if arg0 is not None else id(callable)
        report_resource_access(resource_id, ...)
```

This is heuristic and imprecise, but catches a *lot* with zero config. A `.execute()` call on any DB cursor, a `.send()` on any socket, a `.write()` on any file-like object — all detected automatically.

---

## Layer 4: Import Hooks + Known-Library Registry (Plugin System)

Use `sys.meta_path` to detect when specific libraries are imported, then install targeted instrumentation:

```python
class ResourceInstrumentor:
    """Automatically instruments known libraries on import."""

    KNOWN_LIBRARIES = {
        "sqlite3": lambda mod: patch_methods(mod.Cursor, ["execute", "executemany"]),
        "psycopg2": lambda mod: patch_methods(mod.extensions.cursor, ["execute"]),
        "redis": lambda mod: patch_methods(mod.StrictRedis, ["set", "get", "delete", ...]),
        "pymongo": lambda mod: patch_methods(mod.collection.Collection, ["insert_one", "find", ...]),
        "sqlalchemy": lambda mod: patch_methods(mod.engine.Engine, ["execute", "connect"]),
        "httpx": lambda mod: patch_methods(mod.Client, ["get", "post", "put", "delete"]),
    }

    def find_module(self, name, path=None):
        if name in self.KNOWN_LIBRARIES:
            return self
        return None

    def load_module(self, name):
        # Let the real import happen, then patch
        mod = importlib.import_module(name)
        self.KNOWN_LIBRARIES[name](mod)
        return mod
```

**This is the "plugin" approach.** Ship a registry of known libraries, let users extend it:

```python
@frontrun.register_resource("my_custom_db")
def instrument_my_db(mod):
    patch_methods(mod.Connection, ["query", "mutate"])
```

Advantage: high precision (you know exactly which methods are stateful). Disadvantage: requires maintenance and doesn't catch unknown libraries.

**Could be combined with Layer 0/1 as a fallback:** Use the plugin registry for known libraries (high precision), fall back to socket-level detection for everything else (low precision but complete).

---

## Layer 5: Deterministic Replay of External State

Instead of detecting *which* operations touch external state, **intercept all I/O and make it deterministic:**

```python
class DeterministicSocketLayer:
    """Records I/O on first run, replays on subsequent runs."""

    def __init__(self):
        self.recordings = {}  # (thread_id, sequence_num) → bytes

    def send(self, sock, data):
        # Always record what was sent (for conflict detection)
        key = (current_thread_id(), self.next_seq())
        self.recordings[key] = ("send", sock.getpeername(), data)
        # Actually send on first run; replay from recording on reruns
        if self.mode == "record":
            return real_send(sock, data)
        else:
            return self.replayed_response(key)
```

This is how [rr](https://rr-project.org/) works for system-level record/replay. At the Python level, you'd intercept at the socket/file layer and:

1. **Record run:** Execute normally, record all I/O operations with their thread ID and vector clock
2. **Analyze conflicts:** Two I/O operations to the same endpoint conflict if at least one is a write (send) and they're not ordered by happens-before
3. **Replay runs:** Re-execute with different interleavings, replaying recorded responses

**This solves a problem the other approaches don't:** external state changes between runs. If thread A inserts a row and thread B reads it, the DB state depends on ordering. By recording and replaying, you get deterministic behavior regardless of external state.

**Major complexity cost.** This is essentially building a VCR/test double layer. But for the DPOR use case, it's the only way to truly replay different interleavings of external operations.

---

## Layer 6: One-Line Decorator Annotation (Minimal Config, Maximum Precision)

A middle ground between "zero config" and "full plugin system":

```python
@frontrun.resource("database")
def transfer(from_acct, to_acct, amount):
    # Everything in this function is treated as accessing "database"
    cursor.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, from_acct))
    cursor.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, to_acct))
```

Or at a finer grain:

```python
def transfer(from_acct, to_acct, amount):
    with frontrun.accessing("database"):
        cursor.execute(...)
    with frontrun.accessing("database"):
        cursor.execute(...)
```

This is analogous to the existing `# frontrun:` trace markers but for resource identity rather than scheduling points. Minimal config (one decorator or context manager) with perfect precision (the user says exactly what's a resource).

---

## Synthesis: A Layered Approach

These aren't mutually exclusive. The most practical design is probably layered:

```
┌─────────────────────────────────────────────────────┐
│  Layer 6: User annotations                           │  ← highest precision
│  @frontrun.resource("db") / with accessing("db")     │
├─────────────────────────────────────────────────────┤
│  Layer 4: Known-library plugins                      │  ← auto-installed on import
│  sqlalchemy, redis, psycopg2, ...                    │
├─────────────────────────────────────────────────────┤
│  Layer 3: Duck-typing heuristics                     │  ← .execute(), .send(), .write()
│  sys.monitoring CALL events                          │
├─────────────────────────────────────────────────────┤
│  Layer 2: Taint propagation                          │  ← auto or manual
│  __class__ reassignment / proxy + gc.get_referrers   │
├─────────────────────────────────────────────────────┤
│  Layer 1.5: sys.setprofile / sys.monitoring C_CALL   │  ← C extension visibility
│  Fires on C function calls invisible to sys.settrace │
├─────────────────────────────────────────────────────┤
│  Layer 1: Socket/file monkey-patches                 │  ← catch-all I/O
│  socket.send, builtins.open, os.write                │
├─────────────────────────────────────────────────────┤
│  Layer 0: sys.addaudithook                           │  ← zero-config safety net
│  Fires even from C extensions                        │
└─────────────────────────────────────────────────────┘
```

Higher layers override lower ones (if a plugin provides precise per-table tracking for SQLAlchemy, don't also report the raw socket I/O from the same operation). Lower layers catch everything the higher layers miss.

---

## How This Feeds Into the Three Approaches

**DPOR (most impactful):** Extend the Rust engine's conflict model. Currently `object_id` → `ObjectState`. Add `resource_id` → `ResourceState` with the same read/write/vector-clock logic. A `socket.sendmsg` to `(localhost, 5432)` becomes a write access on resource `("socket", "localhost", 5432)`. Two threads writing to the same resource → DPOR backtracks and explores both orderings.

**Bytecode fuzzing:** Resource accesses become mandatory scheduling points. Instead of treating every opcode equally, weight resource-accessing opcodes higher — always consider a thread switch around I/O operations. This dramatically improves the signal-to-noise ratio of random exploration.

**Trace markers:** Could auto-generate markers. When a resource access is detected, automatically insert a virtual marker named after the resource. The user gets `# frontrun:` style scheduling control without writing the comments.

---

## Wild Ideas Worth Exploring

**SQL wire protocol parsing:** At the socket layer, parse the first few bytes of data sent to known ports (5432=postgres, 3306=mysql, 6379=redis). Extract the SQL command or Redis command. Now you know not just "thread X talked to postgres" but "thread X did `UPDATE accounts`". Resource identity becomes `("postgres", "accounts")` — per-table conflict detection with zero config.

**`LD_PRELOAD` for C extensions:** For libraries that bypass Python's socket layer entirely, ship a small C shared library that intercepts `send`/`recv`/`write`/`read` at the libc level and calls back into Python. This catches *everything*, including pure-C database drivers. Pairs naturally with the Rust DPOR engine (Rust ↔ C interop is trivial).

**Frame introspection at I/O points:** When a resource access is detected (via any layer), inspect `sys._getframe()` to capture the call stack. Use the innermost user-code frame as the "operation identity." This lets you show the user *where* in their code the conflicting resource accesses happen, even without trace markers.

**Import hook bytecode rewriting (long-term):** Eliminate `should_trace_file` filtering entirely. Rewrite bytecodes at import time to insert callbacks only around `CALL` instructions targeting resource methods. Zero overhead on non-resource code. This is a legitimate technique (coverage.py does it), but the implementation cost is high: the shadow stack code in `_process_opcode` already demonstrates ~200 lines of version-specific opcode handling across 3.10–3.14t. For the specific goal of "detect resource accesses," `sys.setprofile` achieves the same thing with ~20 lines.

---

## Experimental Findings

Toy experiments in `ideas/experiments/` verified each technique. Key results:

### `sys.setprofile` — **IMPLEMENTED** (see `frontrun/_io_detection.py`)

Validated in experiment, now implemented in production code. `make_io_profile_func()` and `install_io_profile()` provide per-thread C-call detection for socket and file I/O, coexisting with `sys.settrace`.

### `sys.monitoring` C_CALL events (test_monitoring_c_call.py, Python 3.13)

- **PEP 669 exposes `CALL`, `C_RAISE`, `C_RETURN` events** — `CALL` fires for both Python and C functions.
- **INSTRUCTION + CALL coexist on the same tool ID:** 89 instruction events and 7 call events fired during a single function that does socket I/O. DPOR could use one tool for INSTRUCTION (existing) and add CALL events to the same bitmask.
- **Python vs C calls distinguishable:** C functions lack `__code__` attribute, Python functions have it. `isinstance(callable_obj, type)` catches type constructors.

### `__class__` reassignment (test_class_reassignment.py)

- **Works perfectly on pure-Python objects:** `isinstance()` preserved, all attributes preserved, method dispatch goes through instrumented subclass. Thread-safe (10 concurrent swaps, zero errors).
- **Fails on C extension types:** `socket.socket` → `TypeError: __class__ assignment: object layout differs`. `sqlite3.Cursor` → `TypeError: only supported for mutable types`. `io.StringIO` → same failure.
- **Auto-swap from `sys.settrace` return event works:** The trace function can intercept return values and swap classes on the fly. Duck-type check (`hasattr(retval, "execute")`) correctly identifies resource-like objects.
- **Practical implication:** Use for pure-Python library objects (SQLAlchemy Session, Connection, etc.). Fall back to `sys.setprofile` for C extension types (sqlite3, raw socket).

### `gc.get_referrers()` one-shot (test_gc_referrers.py)

- **Successfully walks from socket to Engine:** `socket → DBConnection → ConnectionPool → Engine` chain correctly traversed.
- **Key implementation detail:** `gc.get_referrers(obj)` returns `__dict__` (a plain dict), not the owning object directly. Must walk *through* dicts: find the dict, then find which object's `__dict__` it is (`getattr(dr, "__dict__", None) is c`).
- **Same-pool connections resolve to same Engine:** Two connections from the same pool both walk up to `id(engine)` — exactly the resource identity we need.
- **Different engines resolve to different identities:** Confirmed.
- **Cost:** ~1.5ms cold walk (trivial object graph), ~25ms with 27K objects. Cache lookup: ~1µs. **1300x speedup** after caching. Acceptable for one-shot startup-time use.

### `sys.addaudithook` (test_audit_hook.py)

- **`socket.connect` fires with address tuple:** `('127.0.0.1', port)` — perfect resource identity.
- **`socket.sendmsg` does NOT fire** (at least not for `sendall` on this Python version). The audit event set is more limited than expected. `socket.__new__` and `socket.connect` are the reliable events.
- **`open` fires with path and mode:** Both write and read opens captured. Mode is available for distinguishing read vs write access.
- **`sqlite3.connect` fires with path and connection handle:** The handle is the actual `sqlite3.Connection` object — can be used as the resource root for `gc.get_referrers` or `__class__` swapping.
- **Socket object identity preserved in audit args:** `args[0] is test_sock` → `True`. Can feed directly to `gc.get_referrers()`.
- **No per-thread filtering:** Audit hooks fire globally. Must call `threading.current_thread()` inside the hook to attribute events to threads.

### Recommended implementation order

Based on experiments, the practical stack (from simplest to most complex):

1. **`sys.setprofile`** (~30 lines): catches C-level I/O, per-thread, coexists with `sys.settrace`. Works today on 3.10+.
2. **`sys.addaudithook`** (~20 lines): zero-config safety net for `socket.connect` and `open`. Limited event set but catches connection establishment.
3. **`gc.get_referrers` one-shot** (~40 lines): maps low-level I/O objects to high-level resource owners. Must walk through `__dict__` dicts. Cache after first walk.
4. **`__class__` reassignment** (~50 lines): makes pure-Python resource objects self-reporting. Auto-trigger from `sys.settrace` return events. Falls back gracefully on C types.
