# PEP 703 Race Conditions: The Accidental Atomicity Gap

PEP 703 removes the Global Interpreter Lock (GIL) from CPython while preserving
thread-safety for individual operations on built-in types through per-object
locks. This report documents a class of race conditions that PEP 703 newly
introduces — races that were impossible under the GIL due to its coarse-grained
protection of C-level compound operations.

## Background: How the GIL Accidentally Protects Compound Operations

Under the GIL, a thread holds the lock continuously while executing C code.
The GIL is only checked (and potentially released) when control returns to the
Python bytecode evaluation loop. This means C functions that iterate a
collection in a tight loop — calling `tp_iternext` repeatedly without
re-entering the eval loop — execute **atomically** with respect to other
threads.

PEP 703 replaces this global lock with fine-grained, per-object locks. Each
`__next__()` call on an iterator acquires the source object's lock, reads one
element, and releases the lock. Between iterations, another thread can freely
acquire the same lock and mutate the object.

The result: compound C operations that were accidentally atomic under the GIL
become interruptible under PEP 703. Pure Python code that was safe (if
accidentally so) can begin exhibiting races.

## The Pattern

All the races in this report share the same structure:

```
Thread 1: C-level function iterates a collection
          (each element access individually locked)

Thread 2: Performs multiple correlated mutations on the same collection
          (each mutation is a separate Python bytecode)
```

Under the GIL, Thread 1's C function reads all elements atomically — it sees
either the fully pre-mutation state or the fully post-mutation state. Under
PEP 703, Thread 1 can read some elements before Thread 2's mutations and
others after, observing a state that **never existed as a whole**.

## Race 1: `functools.reduce()` Over a Mutating List

```python
import functools

items = [1, 2, 3]

# Thread 1
accumulated = functools.reduce(lambda a, b: a + b, items)

# Thread 2
items[0] = 10
items[2] = 30
```

`functools.reduce` iterates using `PyIter_Next` in a C loop. Under the GIL,
the iteration completes without interruption — the result is either
`1+2+3 = 6` (all original) or `10+2+30 = 42` (all mutated).

Under PEP 703, reduce can read `items[0]` between Thread 2's two mutations:

1. Thread 2 stores `items[0] = 10`
2. Thread 1's reduce reads `items[0]=10`, `items[1]=2`, `items[2]=3` → **15**
3. Thread 2 stores `items[2] = 30`

The result 15 corresponds to the snapshot `[10, 2, 3]`, which never existed as
a coherent state. Similarly, reduce could see `[1, 2, 30]` → **33**.

**Invariant violated:** `accumulated in (6, 42)` — only pre- or post-mutation
totals should be observable.

## Race 2: `zip()` Across Correlated Lists

```python
keys = ["a", "b", "c"]
values = [1, 2, 3]

# Thread 1
result = dict(zip(keys, values))

# Thread 2
keys[0] = "z"
values[0] = 99
```

`zip_next` calls `PyIter_Next` on each sub-iterator sequentially. Under the
GIL, both reads for a single tuple happen atomically. Under PEP 703, the key
and value reads for the same position can straddle a mutation:

1. Thread 2 stores `keys[0] = "z"`
2. Thread 1's zip reads `keys[0]="z"`, then `values[0]=1` → tuple `("z", 1)`
3. Thread 2 stores `values[0] = 99`

Result: `{"z": 1, "b": 2, "c": 3}`. Neither `("a", 1)` nor `("z", 99)` — the
first entry pairs a post-mutation key with a pre-mutation value.

**Invariant violated:** `result.get("a") == 1 or result.get("z") == 99` — the
first entry should be internally consistent.

## Race 3: `sum()` Over Mutating Dict Values

```python
data = {"a": 1, "b": 2, "c": 3}

# Thread 1
value_sum = sum(data.values())

# Thread 2
data["a"] = 10
data["c"] = 30
```

`sum()` iterates using `PyIter_Next` in a C loop. Iterating `dict.values()`
acquires the dict's per-object lock for each `__next__` call independently.
Value mutations (without key insertion/deletion) do not trigger the dict's
size-change RuntimeError guard. Under PEP 703:

1. Thread 2 stores `data["a"] = 10`
2. Thread 1's sum reads values `10, 2, 3` → **15**
3. Thread 2 stores `data["c"] = 30`

**Invariant violated:** `value_sum in (6, 42)` — the sum reflects a partial
update that never existed atomically.

## Race 4: `list()` on OrderedDict Keys During Reordering

```python
from collections import OrderedDict

od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])

# Thread 1
keys = list(od.keys())
first_key = keys[0]
last_key = keys[-1]

# Thread 2
od.move_to_end("a")
```

`list()` on a dict keys view iterates with `PyIter_Next` in a tight C loop
(`list_init`). Under the GIL, this loop doesn't release the GIL between
iterations, so it completes atomically. Under PEP 703:

1. Thread 1 reads key `"a"` (position 0)
2. Thread 2 calls `move_to_end("a")` — order becomes `["b", "c", "a"]`
3. Thread 1 continues reading: `"b"`, `"c"`, `"a"`

Result: `keys = ["a", "b", "c", "a"]` — four elements from a three-element
dict! The iterator can see the moved key at both its old and new positions.

**Invariant violated:**
`(first_key == "a" and last_key == "c") or (first_key == "b" and last_key == "a")`
— neither the pre- nor post-reorder snapshot.

This race is unique among the examples because it **cannot be detected at the
Python bytecode level**. The entire `list()` call is a single opcode. The
interleaving happens inside C, between individually-locked `__next__` calls
within `list_init`'s tight loop. Bytecode-level tools can detect Races 1–3
because the *mutations* span multiple bytecodes (two separate `STORE_SUBSCR`
operations), but Race 4 has a single-bytecode mutation (`move_to_end`) and
requires interleaving within the single-bytecode `list()` call.

## What Is NOT a PEP 703 Race

Two patterns that appear similar but are not PEP 703 regressions:

### Individually Atomic Dict Operations

```python
data = {"a": 1, "b": 2}

# Thread 1
popped = data.pop("a", None)

# Thread 2
data.update({"a": 99})
```

Both `dict.pop` and `dict.update` are single C operations protected by the
dict's per-object lock. Under PEP 703, they are serialized — one completes
fully before the other begins. Both orderings produce valid, consistent states.
No invariant can be violated because the operations are genuinely atomic.

The key distinction: PEP 703 races arise from **compound C operations**
(iteration loops) losing atomicity, not from individual operations becoming
unsafe.

### Generator `.send()` With Per-Generator Locks

```python
def counter():
    total = 0
    while True:
        val = yield total
        total += val

gen = counter()
next(gen)

# Thread 1 and Thread 2 both do:
result = gen.send(1)
state.accumulated = result
```

Under the GIL, `gen.send()` runs the generator's Python frame with the GIL
held — the generator's `gi_running` flag prevents concurrent entry. CPython
3.13+ (PEP 703 builds) added **per-generator locks** that provide the same
protection without the GIL.

The `state.accumulated` write race (last writer wins) exists under both the GIL
and PEP 703 — it's a garden-variety bytecode-level race, not a PEP 703
regression.

## The Atomicity Spectrum

These races reveal a spectrum of atomicity guarantees in PEP 703:

| Operation | GIL Atomicity | PEP 703 Atomicity |
|-----------|--------------|-------------------|
| `dict[k] = v`, `list.append(x)` | Atomic (single bytecode) | Atomic (per-object lock) |
| `dict.pop()`, `dict.update()` | Atomic (single C call) | Atomic (per-object lock held for duration) |
| `dict.copy()`, `sorted(list)` | Atomic (copies under lock) | Atomic (lock held during copy) |
| `reduce(f, list)`, `sum(iter)` | **Atomic** (tight C loop) | **NOT atomic** (lock per `__next__`) |
| `list(view)`, `dict(zip(...))` | **Atomic** (tight C loop) | **NOT atomic** (lock per `__next__`) |

The critical boundary is between operations that hold a lock for their entire
duration versus operations that acquire and release per-element. Under the GIL,
this distinction was invisible — the GIL provided a superset of both behaviors.
Under PEP 703, it becomes the difference between safe and racy code.

## Mitigations

1. **External synchronization.** Protect correlated reads and writes with a
   `threading.Lock`:
   ```python
   lock = threading.Lock()

   # Thread 1
   with lock:
       total = sum(data.values())

   # Thread 2
   with lock:
       data["a"] = 10
       data["c"] = 30
   ```

2. **Atomic snapshots.** Copy the collection before iterating:
   ```python
   # Thread 1 — safe under PEP 703
   snapshot = dict(data)       # dict() copies atomically under per-object lock
   total = sum(snapshot.values())
   ```

3. **Immutable intermediaries.** Use `tuple(list)` or `frozenset(set)` to
   capture a consistent snapshot before passing to C functions:
   ```python
   # Thread 1
   frozen = tuple(items)       # tuple() on a list copies atomically
   accumulated = functools.reduce(lambda a, b: a + b, frozen)
   ```

4. **Awareness.** Any C function that iterates a mutable collection using
   `PyIter_Next` is potentially affected. The `sum`, `min`, `max`, `reduce`,
   `str.join`, `zip`, `map`, `filter`, `any`, `all`, and `list()/tuple()` on
   non-list iterables are all candidates. Code review should flag concurrent
   access to collections consumed by these functions.
