# PEP 703 Race Conditions: The Accidental Atomicity Gap

PEP 703 removes the GIL from CPython and replaces it with per-object locks.
Individual operations on built-in types (`dict[k] = v`, `list.append(x)`)
remain atomic. But C functions that **iterate** a collection — `reduce`,
`sum`, `zip`, `list()` on views — used to run atomically under the GIL
because their tight C loops never released it. Under PEP 703, each
`__next__()` call acquires and releases the object lock independently,
so another thread can mutate the collection between iterations.

This means a C-level iteration can observe a state that **never existed as
a whole** — some elements from before a mutation, others from after.

## Example 1: `functools.reduce()` sees a partial mutation

```python
items = [1, 2, 3]

# Thread 1                              # Thread 2
accumulated = functools.reduce(          items[0] = 10
    lambda a, b: a + b, items)           items[2] = 30
```

Under the GIL, the result is always 6 (`1+2+3`) or 42 (`10+2+30`). Under
PEP 703, `reduce` can read between the two stores:

```
Thread 2: items[0] = 10
Thread 1: reduce reads [10, 2, 3] → 15     # snapshot [10, 2, 3] never existed
Thread 2: items[2] = 30
```

## Example 2: `zip()` tears a correlated pair

```python
keys = ["a", "b", "c"]
values = [1, 2, 3]

# Thread 1                              # Thread 2
result = dict(zip(keys, values))         keys[0] = "z"
                                         values[0] = 99
```

`zip` reads `keys[0]` then `values[0]` in separate locked calls. Between them:

```
Thread 2: keys[0] = "z"
Thread 1: zip reads keys[0]="z", values[0]=1 → ("z", 1)    # pairs new key with old value
Thread 2: values[0] = 99
```

Result: `{"z": 1, ...}` — neither the old pair `("a", 1)` nor the new `("z", 99)`.

## Example 3: `sum()` over dict values sees a partial update

```python
data = {"a": 1, "b": 2, "c": 3}

# Thread 1                              # Thread 2
total = sum(data.values())               data["a"] = 10
                                         data["c"] = 30
```

Dict value mutations don't change the dict's size, so no `RuntimeError` is
raised. `sum` iterates with per-element locking:

```
Thread 2: data["a"] = 10
Thread 1: sum reads 10, 2, 3 → 15       # {a:10, b:2, c:3} never existed
Thread 2: data["c"] = 30
```

## Example 4: `list()` on OrderedDict keys sees duplicates

```python
od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])

# Thread 1                              # Thread 2
keys = list(od.keys())                   od.move_to_end("a")
```

`list()` calls `__next__` in a tight C loop. Under the GIL this loop is
atomic. Under PEP 703:

```
Thread 1: reads "a"
Thread 2: move_to_end("a") → order is now [b, c, a]
Thread 1: reads "b", "c", "a"
```

Result: `keys = ["a", "b", "c", "a"]` — four elements from a three-element
dict.

## What is NOT a PEP 703 race

**Single atomic operations** like `dict.pop()` vs `dict.update()` are still
serialized by the per-object dict lock. Both orderings produce valid states.

**Generator `.send()`** has per-generator locks in CPython 3.13+, preserving
the GIL-era guarantee that only one thread executes a generator at a time.

## Affected functions

Any C function iterating via `PyIter_Next`: `sum`, `min`, `max`, `reduce`,
`str.join`, `zip`, `map`, `filter`, `any`, `all`, and `list()`/`tuple()` on
non-list iterables (list/tuple inputs are copied atomically under the object
lock).

## Mitigations

Snapshot before iterating:

```python
total = sum(dict(data).values())                   # dict() copies atomically
result = functools.reduce(op.add, tuple(items))    # tuple(list) copies atomically
```

Or protect with a lock:

```python
with lock:
    total = sum(data.values())
```
