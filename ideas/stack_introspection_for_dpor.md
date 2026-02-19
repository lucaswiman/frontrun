# Stack Introspection for Automatic Shared Access Detection in DPOR

## Problem

The DPOR prototype (`dpor_prototype.py`) requires users to manually decompose their
code into `Step` objects with explicit `object_id` and `AccessKind` annotations:

```python
Step(object_id=0, kind=AccessKind.READ,
     apply=lambda s: s.__setitem__("local", [s["counter"], s["local"][1]]))
```

This is fine for toy examples but completely impractical for real code. If finding
concurrency bugs in open source libraries is a primary use case, we need to
**automatically detect shared object accesses** from unmodified Python code.

Separately, the DPOR prototype defines its own `CooperativeLock` and `SharedVar`
primitives, which duplicate the existing monkey-patched primitives in `bytecode.py`
and wouldn't work with any external library that uses `threading.Lock` directly.

This document investigates two questions:

1. **Can we use stack/bytecode introspection to automatically detect shared object
   attribute accesses?** (Yes, with caveats.)
2. **Can we reuse the existing monkey-patched threading primitives from `bytecode.py`
   for DPOR?** (Yes, straightforwardly.)

---

## Background: How CPython Executes Python Code

### The compilation pipeline

When you write Python code like `self.value = temp + 1`, CPython doesn't interpret
the text directly. It first compiles it to **bytecode** -- a sequence of low-level
instructions for a virtual machine. You can see the bytecodes for any function using
the `dis` module:

```python
import dis

class Counter:
    def increment(self):
        temp = self.value
        self.value = temp + 1

dis.dis(Counter.increment)
```

Output:
```
  # Line: temp = self.value
  LOAD_FAST     0 (self)      # Push local variable 'self' onto the stack
  LOAD_ATTR     0 (value)     # Pop object from stack, push object.value
  STORE_FAST    1 (temp)      # Pop top of stack, store into local 'temp'

  # Line: self.value = temp + 1
  LOAD_FAST     1 (temp)      # Push local 'temp'
  LOAD_CONST    1 (1)         # Push the constant 1
  BINARY_OP     0 (+)         # Pop two values, push their sum
  LOAD_FAST     0 (self)      # Push local 'self'
  STORE_ATTR    0 (value)     # Pop value and object, set object.value = value
```

### The evaluation stack

CPython is a **stack-based** virtual machine. Most bytecode instructions work by
pushing values onto or popping values from an **evaluation stack** (also called the
operand stack). Think of it like a stack of plates:

- `LOAD_FAST self` puts the object `self` on top of the plate stack
- `LOAD_ATTR value` takes the top plate (an object), looks up `.value` on it,
  and puts the result back as a new top plate
- `BINARY_OP +` takes the top two plates, adds them, puts the result back

This is the key data structure for understanding what objects are being accessed
during execution.

### What `sys.settrace` gives us

Python's `sys.settrace` lets us install a callback that CPython calls during
execution. With `frame.f_trace_opcodes = True`, the callback fires **before every
single bytecode instruction**. At each call, we can inspect:

- **`frame.f_code`** -- the code object containing the bytecodes
- **`frame.f_lasti`** -- the byte offset of the instruction about to execute
- **`frame.f_locals`** -- a dict of all local variables and their current values
- **`frame.f_globals`** -- a dict of global/module-level variables

What we **cannot** inspect: **the evaluation stack itself**. CPython does not expose
it to Python code. This is the central challenge this document addresses.

### Key bytecodes for shared state access

These are the bytecodes that matter for detecting when threads access shared state:

| Bytecode | What it does | Stack effect | Why it matters |
|----------|-------------|--------------|----------------|
| `LOAD_FAST x` | Push local variable `x` onto the stack | +1 (pushes) | Loads objects that may be shared |
| `LOAD_GLOBAL x` | Push global variable `x` onto the stack | +1 (pushes) | Module-level shared state |
| `LOAD_DEREF x` | Push a closure variable `x` onto the stack | +1 (pushes) | Captured shared objects |
| `LOAD_CONST c` | Push a constant (number, string, etc.) | +1 (pushes) | Not shared, but needed for stack tracking |
| **`LOAD_ATTR name`** | Pop object from stack, push `object.name` | 0 (pop 1, push 1) | **Shared read: `self.value`** |
| **`STORE_ATTR name`** | Pop value and object, set `object.name = value` | -2 (pop 2) | **Shared write: `self.value = x`** |
| `STORE_FAST x` | Pop top of stack, store into local `x` | -1 (pops) | Needed for stack tracking |
| `BINARY_OP op` | Pop two values, push result of operation | -1 (pop 2, push 1) | Needed for stack tracking |
| `COPY n` | Push a copy of the n-th item from the top | +1 (pushes) | Used in `+=` patterns |
| `SWAP n` | Swap top of stack with the n-th item | 0 | Used in `+=` patterns |
| `BINARY_SUBSCR` | Pop key and container, push `container[key]` | -1 (pop 2, push 1) | **Dict/list reads** |
| `STORE_SUBSCR` | Pop key, container, value; set `container[key] = value` | -3 (pop 3) | **Dict/list writes** |

### The `+=` compilation pattern

Augmented assignment (`self.count += 1`) compiles to a surprisingly complex
bytecode sequence because it needs to:
1. Load the object (`self`) -- it'll need it twice (once for reading, once for writing)
2. Read the old value (`self.count`)
3. Compute the new value (`old + 1`)
4. Store the new value back (`self.count = new`)

```
LOAD_FAST    self       # Stack: [self]
COPY         1          # Stack: [self, self]  -- duplicate for later STORE_ATTR
LOAD_ATTR    count      # Stack: [self, self.count]  -- READ access
LOAD_CONST   1          # Stack: [self, self.count, 1]
BINARY_OP    +=         # Stack: [self, result]
SWAP         2          # Stack: [result, self]  -- reorder for STORE_ATTR
STORE_ATTR   count      # Stack: []  -- WRITE access, pops both
```

This is important because the simple "look at the previous instruction" approach
fails here -- the instruction before `LOAD_ATTR` is `COPY`, not `LOAD_FAST`, and
the instruction before `STORE_ATTR` is `SWAP`, not `LOAD_FAST`.

---

## Approach 1: Look-Back Analysis (Simple, Limited)

For `LOAD_ATTR`/`STORE_ATTR` instructions, look at the **preceding instruction** to
identify what object is being accessed.

### How it works

Pre-analyze each code object to build a map from `(code_id, byte_offset)` to access
info, recording what the preceding instruction was:

```python
# Pre-compute once per code object, cache by code id.
# This avoids calling dis.get_instructions on every opcode event.
_access_cache = {}

def _analyze_code(code):
    """Walk the bytecodes of a code object and record LOAD_ATTR/STORE_ATTR
    instructions along with their preceding instruction.
    """
    prev = None
    for instr in dis.get_instructions(code):
        key = (id(code), instr.offset)
        if instr.opname in ('LOAD_ATTR', 'STORE_ATTR'):
            kind = 'READ' if instr.opname == 'LOAD_ATTR' else 'WRITE'
            # Record: what kind of access, the attribute name, and what
            # instruction came immediately before (to identify the object)
            _access_cache[key] = (kind, instr.argval,
                                  prev.opname if prev else None,
                                  prev.argval if prev else None)
        else:
            _access_cache[key] = None
        prev = instr
```

At trace time, use the preceding instruction to resolve the object from `f_locals`
or `f_globals`:

```python
def trace(frame, event, arg):
    if event == 'opcode':
        info = _access_cache.get((id(frame.f_code), frame.f_lasti))
        if info is not None:
            kind, attr_name, prev_op, prev_arg = info
            obj = None
            # If the preceding instruction was LOAD_FAST, we know the
            # object is a local variable and can read it from f_locals:
            if prev_op == 'LOAD_FAST':
                obj = frame.f_locals.get(prev_arg)
            # Similarly for globals:
            elif prev_op == 'LOAD_GLOBAL':
                obj = frame.f_globals.get(prev_arg)
            # Closure variables also appear in f_locals:
            elif prev_op == 'LOAD_DEREF':
                obj = frame.f_locals.get(prev_arg)
            if obj is not None:
                engine.process_access(thread_id, (id(obj), attr_name), kind)
        return trace
```

### What it handles

| Pattern | Bytecodes | Resolved? |
|---------|-----------|-----------|
| `self.x` (read) | `LOAD_FAST self` -> `LOAD_ATTR x` | Yes |
| `self.x = val` (write) | `LOAD_FAST self` -> `STORE_ATTR x` | Yes |
| `global_obj.x` | `LOAD_GLOBAL g` -> `LOAD_ATTR x` | Yes |
| Closure `obj.x` | `LOAD_DEREF obj` -> `LOAD_ATTR x` | Yes |

### What it misses

| Pattern | Why it fails |
|---------|-------------|
| `self.x += 1` | `LOAD_ATTR` is preceded by `COPY`, not `LOAD_FAST` |
| `self.a.b` (chained) | Second `LOAD_ATTR` is preceded by first `LOAD_ATTR` |
| `get_obj().x` | `LOAD_ATTR` is preceded by `CALL` |

### Performance

With the code pre-analysis cached, look-back adds ~2x overhead over base opcode
tracing (which is already ~25-50x over untraced execution). For testing workloads,
the absolute overhead is acceptable.

### Verdict

**Feasible for simple patterns, insufficient for real code.** The `+=` pattern alone
is common enough to make this approach unreliable as a sole detection mechanism.

---

## Approach 2: Shadow Stack (Comprehensive, Recommended)

Since CPython doesn't expose its evaluation stack, we build our own **shadow stack**
-- a Python list that we update in parallel with CPython's real stack, tracking what
objects are at each position.

### What is a shadow stack?

A shadow stack is a data structure that **mirrors** the CPython evaluation stack.
Every time CPython pushes a value onto its internal stack, we push a corresponding
value onto our shadow stack. Every time CPython pops, we pop. This way, when
CPython is about to execute `LOAD_ATTR`, and we know that instruction pops the
top-of-stack to get the target object, we can look at **our** shadow stack's top
element to identify that object.

The shadow stack doesn't need to be perfectly accurate for every value. We only
care about identifying the objects involved in `LOAD_ATTR`/`STORE_ATTR`. For
other values (like arithmetic results), we can push `None` as a placeholder
meaning "we don't know what this value is."

### How it works

```python
class ShadowStack:
    """Mirrors CPython's evaluation stack to track object identity.

    CPython doesn't expose its internal evaluation stack to Python code.
    This class maintains a parallel stack by simulating the push/pop
    effects of each bytecode instruction. When we need to know what
    object is being accessed by LOAD_ATTR or STORE_ATTR, we look at
    our shadow stack instead of CPython's real stack.

    Values on the shadow stack are either:
    - A real Python object (when we know what it is, e.g. from LOAD_FAST)
    - None (when the value is unknown, e.g. the result of BINARY_OP)
    """
    def __init__(self):
        self.stack = []

    def push(self, val):
        """Push a value onto the shadow stack (mirrors CPython push)."""
        self.stack.append(val)

    def pop(self):
        """Pop and return top value (mirrors CPython pop). Returns None on underflow."""
        return self.stack.pop() if self.stack else None

    def peek(self, n=0):
        """Look at the n-th item from the top without popping.
        peek(0) = TOS, peek(1) = second from top, etc."""
        idx = -(n + 1)
        return self.stack[idx] if abs(idx) <= len(self.stack) else None
```

For each opcode, update the shadow stack to stay in sync with CPython:

```python
def process_opcode(self, frame, thread_id, engine, execution):
    """Called before each bytecode instruction executes.

    Looks at which instruction is about to execute (via frame.f_lasti),
    updates the shadow stack to mirror what CPython will do to its real
    stack, and reports any shared object accesses to the DPOR engine.
    """
    instr = self._get_instr(frame.f_code, frame.f_lasti)
    if instr is None:
        return

    shadow = self._shadow(frame)
    op = instr.opname

    # === Instructions that LOAD values onto the stack ===

    if op == 'LOAD_FAST':
        # LOAD_FAST pushes a local variable onto the stack.
        # Example: `self` in `self.value` -- LOAD_FAST puts the self
        # object on the stack so the next instruction can use it.
        # We read it from f_locals (CPython's local variable dict).
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)

    elif op == 'LOAD_GLOBAL':
        # LOAD_GLOBAL pushes a module-level variable.
        # Example: `shared_dict` in `shared_dict['key']`
        val = frame.f_globals.get(instr.argval)
        shadow.push(val)

    elif op == 'LOAD_DEREF':
        # LOAD_DEREF pushes a variable captured from an enclosing scope
        # (a "closure variable"). Example:
        #     def make_worker(obj):
        #         def worker():
        #             obj.count += 1   # 'obj' is accessed via LOAD_DEREF
        #         return worker
        # Despite being a closure variable, it appears in f_locals.
        val = frame.f_locals.get(instr.argval)
        shadow.push(val)

    elif op == 'LOAD_CONST':
        # LOAD_CONST pushes a compile-time constant (number, string, None).
        # Not relevant for shared state, but needed to keep the shadow
        # stack in sync. Example: the `1` in `self.count += 1`
        shadow.push(instr.argval)

    # === Stack manipulation instructions ===

    elif op == 'COPY':
        # COPY n: Duplicate the n-th item from the top of the stack.
        # COPY 1 duplicates TOS (top of stack).
        #
        # This instruction appears in augmented assignment (`+=`):
        #   LOAD_FAST self     -> stack: [self]
        #   COPY 1             -> stack: [self, self]  (save for STORE_ATTR later)
        #   LOAD_ATTR count    -> stack: [self, self.count]
        #   ...
        n = instr.arg
        if len(shadow.stack) >= n:
            shadow.push(shadow.stack[-n])
        else:
            shadow.push(None)

    elif op == 'SWAP':
        # SWAP n: Swap TOS (top of stack) with the n-th item from top.
        # SWAP 2 swaps the top two items.
        #
        # This also appears in augmented assignment (`+=`):
        #   ... BINARY_OP +=   -> stack: [self, result]
        #   SWAP 2             -> stack: [result, self]
        #   STORE_ATTR count   -> stack: []  (sets self.count = result)
        n = instr.arg
        if len(shadow.stack) >= n:
            shadow.stack[-1], shadow.stack[-n] = shadow.stack[-n], shadow.stack[-1]

    # === ATTRIBUTE ACCESS -- the instructions we care about most ===

    elif op == 'LOAD_ATTR':
        # LOAD_ATTR pops an object from the stack, looks up an attribute
        # on it, and pushes the result.
        #
        # Example: `self.value` compiles to LOAD_FAST self, LOAD_ATTR value
        # When LOAD_ATTR executes:
        #   - It pops `self` from the stack
        #   - It pushes `self.value` (the result of the attribute lookup)
        #
        # For DPOR, this is a READ access on the shared object `self`
        # for attribute `value`. We report it to the engine.
        obj = shadow.pop()
        attr = instr.argval
        if obj is not None:
            # Report this read to the DPOR engine.
            # The object_key uniquely identifies "attribute X on object Y"
            object_key = (id(obj), attr)
            engine.process_access(
                execution, thread_id, hash(object_key), AccessKind.READ
            )
            # Push the attribute value so chained accesses work.
            # For `self.a.b`, after LOAD_ATTR a we push self.a,
            # then LOAD_ATTR b will pop self.a and read .b from it.
            try:
                shadow.push(getattr(obj, attr))
            except Exception:
                shadow.push(None)  # attribute lookup failed
        else:
            shadow.push(None)  # unknown object, can't track

    elif op == 'STORE_ATTR':
        # STORE_ATTR pops an object (TOS) and a value (TOS1), then sets
        # object.attr = value.
        #
        # Example: `self.value = temp + 1` -> ... LOAD_FAST self, STORE_ATTR value
        # When STORE_ATTR executes:
        #   - TOS = the object to set the attribute on (self)
        #   - TOS1 = the value to store (temp + 1)
        #   - It pops both and performs self.value = (temp + 1)
        #
        # For DPOR, this is a WRITE access on the shared object.
        obj = shadow.pop()   # TOS = object being written to
        _val = shadow.pop()  # TOS1 = value being stored
        if obj is not None:
            object_key = (id(obj), instr.argval)
            engine.process_access(
                execution, thread_id, hash(object_key), AccessKind.WRITE
            )

    # === SUBSCRIPT access (dict/list operations) ===

    elif op == 'BINARY_SUBSCR':
        # BINARY_SUBSCR pops a key (TOS) and container (TOS1), pushes
        # container[key]. Example: `d['key']` -> LOAD_FAST d, LOAD_CONST 'key',
        # BINARY_SUBSCR
        key = shadow.pop()
        container = shadow.pop()
        if container is not None:
            object_key = (id(container), f'[{key!r}]')
            engine.process_access(
                execution, thread_id, hash(object_key), AccessKind.READ
            )
        shadow.push(None)  # result of container[key] is unknown

    elif op == 'STORE_SUBSCR':
        # STORE_SUBSCR pops key (TOS), container (TOS1), and value (TOS2),
        # then sets container[key] = value.
        key = shadow.pop()
        container = shadow.pop()
        _val = shadow.pop()
        if container is not None:
            object_key = (id(container), f'[{key!r}]')
            engine.process_access(
                execution, thread_id, hash(object_key), AccessKind.WRITE
            )

    # === Arithmetic and other operations ===

    elif op == 'BINARY_OP':
        # BINARY_OP pops two values, pushes the result of the operation
        # (e.g., addition, subtraction). We don't know the result at this
        # point (we'd have to execute the operation ourselves), so we
        # push None as a placeholder.
        shadow.pop()
        shadow.pop()
        shadow.push(None)

    elif op in ('STORE_FAST', 'STORE_GLOBAL', 'STORE_DEREF'):
        # These instructions pop TOS and store it into a variable.
        # No shared access to report, just keep the stack in sync.
        shadow.pop()

    elif op == 'RETURN_VALUE':
        # Pops TOS as the return value.
        shadow.pop()

    elif op == 'POP_TOP':
        # Discards TOS. Common after function calls whose return value
        # is ignored (e.g., `self.items.append(x)` -- the None return
        # from append is discarded).
        shadow.pop()

    else:
        # === Fallback for all other instructions ===
        #
        # dis.stack_effect() tells us how many items an instruction
        # pushes/pops. We use this as a fallback for instructions we
        # don't explicitly handle. The pushed values are unknown (None).
        #
        # Example: CALL pops the callable + arguments, pushes 1 return
        # value. stack_effect tells us the net change.
        try:
            effect = dis.stack_effect(instr.opcode, instr.arg or 0)
            for _ in range(max(0, -effect)):
                shadow.pop()
            for _ in range(max(0, effect)):
                shadow.push(None)
        except (ValueError, TypeError):
            # If stack_effect fails (e.g., for opcodes without args),
            # we've lost track of the stack. Clear it to avoid reporting
            # wrong objects. The stack resets at the next function call
            # boundary anyway.
            shadow.stack.clear()
```

### Verified behaviors (tested on CPython 3.11)

| Python Pattern | Bytecode Sequence | Shadow Stack Result |
|---------|-------------------|---------------------|
| `self.x` (read) | `LOAD_FAST self` -> `LOAD_ATTR x` | Correctly identifies `self` object |
| `self.x = val` (write) | `LOAD_FAST self` -> `STORE_ATTR x` | Correctly identifies `self` object |
| `self.x += 1` | `LOAD_FAST` -> `COPY 1` -> `LOAD_ATTR` -> ... -> `SWAP 2` -> `STORE_ATTR` | Correctly identifies `self` for **both** READ and WRITE |
| Closure `obj.x` | `LOAD_DEREF obj` -> `LOAD_ATTR x` | Correctly resolves via `f_locals` |
| Global `g.x` | `LOAD_GLOBAL g` -> `LOAD_ATTR x` | Correctly resolves via `f_globals` |
| Chained `self.a.b` | `LOAD_FAST self` -> `LOAD_ATTR a` -> `LOAD_ATTR b` | Identifies `self` for `.a`; identifies `self.a` result for `.b` via `getattr` |
| Multi-thread on same object | Two threads calling `obj.increment()` | Both correctly resolve to same `id(obj)` |

### Limitations

1. **`getattr` during tracing**: The shadow stack calls `getattr(obj, attr)` after
   `LOAD_ATTR` to push the result value (needed for chained access like `self.a.b`).
   This re-executes the attribute access, which:
   - Could have side effects (properties, `__getattribute__` overrides)
   - May not match the actual result if another thread modified the object
   - Could raise exceptions

   **Mitigation**: Use `object.__getattribute__` directly, or accept `None` (unknown)
   for complex cases. The object identity for the `LOAD_ATTR` (the thing we pop) is
   the critical piece; the pushed result only matters for chained accesses.

2. **Stack desynchronization**: If the shadow stack gets out of sync with CPython's
   real stack (due to unhandled opcodes, exceptions, or `dis.stack_effect` errors),
   subsequent access detection for that frame becomes unreliable.

   **Mitigation**: Clear the shadow stack on unhandled opcodes or exceptions.
   Function call/return boundaries naturally reset the stack (each frame gets its own
   shadow stack).

3. **Version-specific bytecodes**: The opcode set changes between CPython versions.
   Python 3.12 merges `LOAD_METHOD` into `LOAD_ATTR`, removes `PRECALL`, etc.
   Python 3.13 makes further changes.

   **Mitigation**: The core instructions (`LOAD_FAST`, `LOAD_ATTR`, `STORE_ATTR`,
   `COPY`, `SWAP`) are stable across 3.11-3.13. Use `dis.stack_effect` as fallback
   for version-specific opcodes.

4. **Function calls** (`CALL`/`PRECALL`): These consume arguments from the stack and
   push a return value. The return value is always `None` (unknown) in our shadow
   stack. This means method calls like `self.items.append(x)` will track the `.items`
   access on `self` but not what happens inside `append`.

### Performance

Benchmarks on CPython 3.11 (100 iterations of a 100-iteration counter increment):

| Mode | Time | Overhead vs. baseline |
|------|------|----------------------|
| Baseline (no tracing) | 0.3ms | 1x |
| Opcode tracing only (`sys.settrace`) | 7-15ms | ~25-50x |
| Look-back access tracking (cached) | 31ms | ~100x |
| Shadow stack (estimated) | 40-60ms | ~130-200x |

This overhead is comparable to the existing opcode scheduling in `bytecode.py` and
is acceptable for testing workloads. The trace function is called for every bytecode
instruction regardless (that's the 25-50x overhead); the shadow stack adds ~2x on
top of that.

### Verdict

**Feasible and sufficient for DPOR integration.** The shadow stack correctly handles
all common access patterns including `+=`. The main engineering cost is handling
~15 opcodes explicitly (with `dis.stack_effect` as a fallback).

---

## Approach 3: `__getattribute__` / `__setattr__` Proxies

Wrap shared objects in proxy objects that intercept attribute access.

```python
class TrackedProxy:
    """Wraps an object to intercept attribute access. NOT recommended."""
    def __init__(self, wrapped, object_id, engine, thread_id):
        object.__setattr__(self, '_wrapped', wrapped)
        # ...

    def __getattr__(self, name):
        self._engine.process_access(self._thread_id, self._object_id, name, 'READ')
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        self._engine.process_access(self._thread_id, self._object_id, name, 'WRITE')
        setattr(self._wrapped, name, value)
```

### Verdict

**Not feasible for external libraries.** Requires wrapping every shared object,
doesn't work for C-extension types, and changes object identity (`proxy is not obj`).
The `SharedVar` approach in `dpor_prototype.py` is essentially this, and it has the
same fundamental problem: it can't be used on unmodified code.

---

## Reusing Monkey-Patched Threading Primitives for DPOR

### Current situation

**In `bytecode.py`**: Cooperative threading primitives (`_CooperativeLock`,
`_CooperativeRLock`, etc.) that:

1. Are installed via monkey-patching (`threading.Lock = _CooperativeLock`)
2. Use thread-local storage (`_active_scheduler`) to find the opcode scheduler
3. Spin-yield instead of blocking, giving the scheduler control over thread execution

**In `dpor_prototype.py`**: A separate `CooperativeLock` that:

1. Must be explicitly instantiated by the user (can't use `threading.Lock()`)
2. Reports acquire/release to the DPOR engine
3. Is completely separate from the monkey-patching infrastructure

The DPOR prototype's approach is a nonstarter for external libraries. If a library
uses `threading.Lock()`, it creates a real lock that the DPOR engine knows nothing
about. The monkey-patching approach in `bytecode.py` solves this -- when we replace
`threading.Lock` with `_CooperativeLock`, any library that calls `threading.Lock()`
gets a cooperative lock automatically.

### Integration design

The existing cooperative primitives just need to be extended to **also report to a
DPOR engine** when one is active. The thread-local storage already provides the
plumbing:

```python
# In bytecode.py, the thread-local context already stores:
_active_scheduler = threading.local()
# .scheduler -- the OpcodeScheduler instance
# .thread_id -- which thread we are

# We add two more fields:
# .dpor_engine    -- the DporEngine instance (or None when not doing DPOR)
# .dpor_execution -- the current Execution state (or None)


class _CooperativeLock:
    """Modified to also report to DPOR engine."""

    def acquire(self, blocking=True, timeout=-1):
        # ... existing spin-yield logic unchanged ...

        # NEW: after successful acquire, report to DPOR engine.
        # If no DPOR engine is active, dpor_engine will be None and
        # we skip this entirely (zero cost).
        engine = getattr(_active_scheduler, 'dpor_engine', None)
        execution = getattr(_active_scheduler, 'dpor_execution', None)
        if engine and execution:
            tid = getattr(_active_scheduler, 'thread_id', None)
            engine.process_sync(
                execution, tid,
                LockAcquire(lock_id=id(self))
            )
        return True

    def release(self):
        # NEW: report release to DPOR engine before actually releasing.
        engine = getattr(_active_scheduler, 'dpor_engine', None)
        execution = getattr(_active_scheduler, 'dpor_execution', None)
        if engine and execution:
            tid = getattr(_active_scheduler, 'thread_id', None)
            engine.process_sync(
                execution, tid,
                LockRelease(lock_id=id(self))
            )
        self._lock.release()
```

### Why this works

1. **Zero changes to external libraries**: Libraries use `threading.Lock()` as
   normal. The monkey-patch makes them cooperative and DPOR-aware transparently.

2. **Reuses existing code**: The cooperative primitives already handle all the
   edge cases (spin-yield, schedule exhaustion, timeout fallback). We only add
   the DPOR reporting on top.

3. **Backward compatible**: When no DPOR engine is active (`dpor_engine is None`),
   `getattr` returns `None` and the reporting is skipped. Existing tests pass
   unchanged.

4. **Consistent with opcode scheduling**: The DPOR engine replaces the random
   schedule generator while reusing the same cooperative primitive infrastructure.

### Verified in prototype

Tested with the following scenario:

```python
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()  # becomes DporCooperativeLock after patching

    def safe_increment(self):
        with self.lock:
            temp = self.value
            self.value = temp + 1
```

Running two threads, the DPOR engine received these events (in order):
```
Thread 0: acquire lock 0x7f...
Thread 0: READ  .value on Counter
Thread 0: WRITE .value on Counter
Thread 0: release lock 0x7f...
Thread 1: acquire lock 0x7f...
Thread 1: READ  .value on Counter
Thread 1: WRITE .value on Counter
Thread 1: release lock 0x7f...
```

The DPOR engine can see that the two threads' accesses to `.value` are ordered by
the lock acquire/release (happens-before), so it knows this particular interleaving
is safe. If the lock were removed, the READ/WRITE accesses would be concurrent, and
the engine would insert backtrack points to explore both orderings.

---

## Identifying "Shared" Objects

Detecting accesses is only half the problem. The DPOR engine needs to know which
accesses are to **shared** objects (accessed by multiple threads). Options:

### Option A: Track Everything, Filter Later

Record all attribute accesses from all threads. After each execution, identify
objects accessed by multiple threads as shared. Simple but generates many spurious
DPOR backtrack points for thread-local state (each thread's own local objects).

### Option B: Pre-declare Shared Objects

The user identifies shared objects upfront:

```python
explore_dpor(
    shared_objects=[account_a, account_b],
    threads=[...],
)
```

Only accesses to these objects (matched by `id()`) are reported. Simple and precise
but requires the user to know which objects are shared.

### Option C: Heuristic Detection

Track `(object_id, attr_name)` pairs. An object becomes "shared" the first time
it's accessed by a second thread. Before that, accesses are ignored.

This requires a two-pass approach: run once to detect shared objects, then re-run
with DPOR using the detected sharing information.

### Option D: Conservative (Recommended for prototype)

Treat all attribute accesses on objects reachable from function arguments or closure
variables as potentially shared. This over-approximates but is safe: DPOR with extra
backtrack points is slower (explores more interleavings than necessary) but still
correct.

For the prototype, this is the right tradeoff: correctness over performance.
Optimization can come later via escape analysis or user hints.

---

## Python Version Considerations

### Python 3.11 (tested)
- `f_trace_opcodes` works as documented
- Key opcodes: `LOAD_FAST`, `LOAD_GLOBAL`, `LOAD_DEREF`, `LOAD_ATTR`, `STORE_ATTR`,
  `COPY`, `SWAP`, `BINARY_OP`, `PRECALL`, `CALL`, `LOAD_METHOD`

### Python 3.12+
- `LOAD_ATTR` now also handles method loading (replaces `LOAD_METHOD`)
- `PRECALL` removed
- `sys.monitoring` (PEP 669) available as lower-overhead alternative to `sys.settrace`
- `INSTRUCTION` event in `sys.monitoring` fires per-opcode like `f_trace_opcodes`

### Python 3.13+
- Further bytecode changes (specialized instructions)
- `sys.monitoring` is the preferred API

### Recommendation

Target Python 3.11+ initially. The shadow stack needs version-specific opcode
handlers, but the core set is small and stable. Use `dis.stack_effect` as fallback
for any opcodes not explicitly handled.

---

## Summary

| Approach | Feasibility | Handles `+=` | External libs | Complexity |
|----------|-------------|--------------|---------------|------------|
| Look-back analysis | Partial | No | Yes | Low |
| **Shadow stack** | **Yes** | **Yes** | **Yes** | **Medium** |
| `__getattribute__` proxy | No (invasive) | N/A | No | Low |
| Monkey-patched locks for DPOR | Yes | N/A | Yes | Low |

### Recommended architecture

```
                        ┌──────────────────────┐
                        │   User's unmodified  │
                        │    threaded code     │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┼───────────────┐
                    │              │               │
            ┌───────▼───────┐  ┌──▼──────────┐  ┌──▼───────────┐
            │ Monkey-patched│  │sys.settrace │  │  DPOR Engine │
            │ threading.Lock│  │  + shadow   │  │  (scheduling │
            │ (sync events) │  │   stack     │  │   decisions) │
            └───────┬───────┘  │  (access    │  └──────▲───────┘
                    │          │  detection) │         │
                    │          └──────┬──────┘         │
                    │                 │                │
                    └────────►  process_sync  ◄────────┘
                              process_access
```

1. **Monkey-patched primitives** (existing code, extended) report lock/event/semaphore
   operations to the DPOR engine as sync events, establishing happens-before edges.
2. **Shadow stack tracker** (new code) inspects bytecodes during opcode tracing to
   detect attribute accesses on shared objects, reporting them to the DPOR engine.
3. **DPOR engine** (existing `dpor_prototype.py`) uses both sync events and access
   reports to compute happens-before relations and guide scheduling decisions.

### Next steps

1. Extend `_active_scheduler` thread-local in `bytecode.py` to carry DPOR engine
   and execution references.
2. Add `process_sync` calls to existing cooperative primitives.
3. Implement `ShadowStackTracker` as a standalone module.
4. Create `DporBytecodeShuffler` that combines the opcode scheduler, shadow stack
   tracker, and DPOR engine into a single test runner.
5. Handle `LOAD_METHOD`/`CALL` patterns for method call tracking.
6. Add version-specific opcode handling for Python 3.12+.
