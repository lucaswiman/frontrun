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

## Layer 2: Taint Propagation via Proxy Objects

The insight: if you can identify the *root* resource object (an `Engine`, a `Connection`, a file handle), you can wrap it in a proxy that *taints* everything derived from it.

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

**Requires one line of config** (wrapping the root resource), but then propagates automatically through arbitrary call chains. This directly addresses the concern about `engine.session().cursor().execute(...)`.

**Enhancement — auto-detect roots:** Combine with audit hooks. When `socket.connect` fires, walk `gc.get_referrers(sock)` up the reference chain to find the "owning" high-level object. Tag that object as a resource root. Next time any thread touches it (detected via DPOR's normal attribute tracking), we know it's a resource access.

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
┌─────────────────────────────────────────────────┐
│  Layer 6: User annotations                       │  ← highest precision
│  @frontrun.resource("db") / with accessing("db") │
├─────────────────────────────────────────────────┤
│  Layer 4: Known-library plugins                  │  ← auto-installed on import
│  sqlalchemy, redis, psycopg2, ...               │
├─────────────────────────────────────────────────┤
│  Layer 3: Duck-typing heuristics                 │  ← .execute(), .send(), .write()
│  sys.monitoring CALL events                      │
├─────────────────────────────────────────────────┤
│  Layer 1: Socket/file monkey-patches             │  ← catch-all I/O
│  socket.send, builtins.open, os.write            │
├─────────────────────────────────────────────────┤
│  Layer 0: sys.addaudithook                       │  ← zero-config safety net
│  Fires even from C extensions                    │
└─────────────────────────────────────────────────┘
```

Higher layers override lower ones (if a plugin provides precise per-table tracking for SQLAlchemy, don't also report the raw socket I/O from the same operation). Lower layers catch everything the higher layers miss.

---

## How This Feeds Into the Three Approaches

**DPOR (most impactful):** Extend the Rust engine's conflict model. Currently `object_id` → `ObjectState`. Add `resource_id` → `ResourceState` with the same read/write/vector-clock logic. A `socket.sendmsg` to `(localhost, 5432)` becomes a write access on resource `("socket", "localhost", 5432)`. Two threads writing to the same resource → DPOR backtracks and explores both orderings.

**Bytecode fuzzing:** Resource accesses become mandatory scheduling points. Instead of treating every opcode equally, weight resource-accessing opcodes higher — always consider a thread switch around I/O operations. This dramatically improves the signal-to-noise ratio of random exploration.

**Trace markers:** Could auto-generate markers. When a resource access is detected, automatically insert a virtual marker named after the resource. The user gets `# frontrun:` style scheduling control without writing the comments.

---

## Wild Ideas Worth Exploring

**`gc.get_referrers` chain walking:** When `socket.connect` fires, walk `gc.get_referrers(sock)` upward to find the high-level object that owns this socket. If it's a `sqlalchemy.engine.Engine`, use that as the resource identity. Completely automatic library detection without a plugin registry.

**SQL wire protocol parsing:** At the socket layer, parse the first few bytes of data sent to known ports (5432=postgres, 3306=mysql, 6379=redis). Extract the SQL command or Redis command. Now you know not just "thread X talked to postgres" but "thread X did `UPDATE accounts`". Resource identity becomes `("postgres", "accounts")` — per-table conflict detection with zero config.

**`LD_PRELOAD` for C extensions:** For libraries that bypass Python's socket layer entirely, ship a small C shared library that intercepts `send`/`recv`/`write`/`read` at the libc level and calls back into Python. This catches *everything*, including pure-C database drivers. Pairs naturally with the Rust DPOR engine (Rust ↔ C interop is trivial).

**Frame introspection at I/O points:** When a resource access is detected (via any layer), inspect `sys._getframe()` to capture the call stack. Use the innermost user-code frame as the "operation identity." This lets you show the user *where* in their code the conflicting resource accesses happen, even without trace markers.
