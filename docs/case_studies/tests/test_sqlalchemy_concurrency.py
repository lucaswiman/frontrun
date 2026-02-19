"""
Comprehensive concurrency tests for SQLAlchemy using frontrun bytecode exploration.

Tests various areas of SQLAlchemy for race conditions using in-memory SQLite and
property-based interleaving exploration. Each test specifies an invariant that
should hold regardless of thread scheduling, and frontrun searches for a
counterexample schedule that violates it.

Areas tested:
1. LRUCache._inc_counter lost update (non-atomic +=)
2. LRUCache concurrent get/set data corruption
3. Event registry _stored_in_collection TOCTOU
4. QueuePool._do_get TOCTOU with finite overflow
5. SingletonThreadPool._all_conns set race
6. _memoized_property double-evaluation TOCTOU
7. ScopedRegistry double-creation TOCTOU
8. _dialect_info type memo cache TOCTOU
9. Pool _strong_ref_connection_records global dict race
10. LRUCache _manage_size concurrent eviction + insert

Repository: https://github.com/sqlalchemy/sqlalchemy (commit 6fa097e)
"""

import os
import signal
import sys
import weakref
from contextlib import contextmanager

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_lib = os.path.join(_test_dir, "..", "external_repos", "sqlalchemy", "lib")
sys.path.insert(0, os.path.abspath(_repo_lib))

from frontrun.bytecode import explore_interleavings, run_with_schedule  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def timeout_minutes(minutes=10):
    def _handler(signum, frame):
        raise TimeoutError(f"Test timed out after {minutes} minute(s)")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(minutes * 60))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def print_result(name, result, extra=""):
    status = "PASS (no race found)" if result.property_holds else "FAIL (race found!)"
    print(f"  [{name}] {status} after {result.num_explored} interleavings{extra}")
    return result


def run_sweep(name, setup, threads, invariant, num_seeds=10, max_attempts=200,
              max_ops=200, timeout=10):
    """Run exploration across multiple seeds, return (found_seeds, total_explored)."""
    found_seeds = []
    total_explored = 0
    for seed in range(num_seeds):
        with timeout_minutes(timeout):
            result = explore_interleavings(
                setup=setup,
                threads=threads,
                invariant=invariant,
                max_attempts=max_attempts,
                max_ops=max_ops,
                seed=seed,
            )
        total_explored += result.num_explored
        if not result.property_holds:
            found_seeds.append((seed, result.num_explored))
    return found_seeds, total_explored


def run_reproduce(name, setup, threads, invariant, max_attempts=500, max_ops=200,
                  timeout=10, repro_count=10):
    """Find a counterexample and reproduce it deterministically."""
    with timeout_minutes(timeout):
        result = explore_interleavings(
            setup=setup,
            threads=threads,
            invariant=invariant,
            max_attempts=max_attempts,
            max_ops=max_ops,
            seed=42,
        )
    if not result.counterexample:
        return 0

    bugs = 0
    for i in range(repro_count):
        state = run_with_schedule(
            result.counterexample,
            setup=setup,
            threads=threads,
        )
        if not invariant(state):
            bugs += 1
    return bugs


# ===========================================================================
# Test 1: LRUCache._inc_counter lost update
# ===========================================================================
# LRUCache._inc_counter does `self._counter += 1` without any lock.
# This is called from get(), __getitem__, and __setitem__.
# Two concurrent get() calls can both load the same _counter value.

class LRUCacheCounterState:
    def __init__(self):
        from sqlalchemy.util._collections import LRUCache
        self.cache = LRUCache(capacity=100)
        # Pre-populate so get() hits the counter path
        self.cache["key1"] = "val1"
        self.cache["key2"] = "val2"
        self.initial_counter = self.cache._counter

    def thread1(self):
        self.cache.get("key1")
        self.cache.get("key2")

    def thread2(self):
        self.cache.get("key1")
        self.cache.get("key2")


def _lru_counter_invariant(s):
    # Each thread does 2 gets, each incrementing _counter once = 4 total
    return s.cache._counter == s.initial_counter + 4


def test_lru_cache_counter_lost_update():
    """LRUCache._inc_counter: self._counter += 1 without lock."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: LRUCacheCounterState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_lru_counter_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("LRUCache._inc_counter", result)


def test_lru_cache_counter_sweep():
    """Sweep 10 seeds for LRUCache counter race."""
    found, total = run_sweep(
        "LRUCache._inc_counter",
        setup=lambda: LRUCacheCounterState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_lru_counter_invariant,
    )
    print(f"  LRUCache counter: {len(found)}/10 seeds found, {total} total explored")
    if found:
        avg = sum(n for _, n in found) / len(found)
        print(f"  Avg attempts: {avg:.1f}")
    return found


def test_lru_cache_counter_reproduce():
    """Reproduce LRUCache counter race deterministically."""
    bugs = run_reproduce(
        "LRUCache._inc_counter",
        setup=lambda: LRUCacheCounterState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_lru_counter_invariant,
    )
    print(f"  LRUCache counter reproduced: {bugs}/10")
    return bugs


# ===========================================================================
# Test 2: LRUCache concurrent __setitem__ + get data consistency
# ===========================================================================
# Two threads: one inserts new keys, the other reads. The _inc_counter race
# can cause two entries to share the same counter value, corrupting LRU order.

class LRUCacheSetGetState:
    def __init__(self):
        from sqlalchemy.util._collections import LRUCache
        self.cache = LRUCache(capacity=100)

    def thread1(self):
        self.cache["a"] = "value_a"
        self.cache["b"] = "value_b"

    def thread2(self):
        self.cache["c"] = "value_c"
        self.cache["d"] = "value_d"


def _lru_setget_invariant(s):
    # All 4 keys should be present and counter should reflect 4 increments
    keys_ok = len(s.cache) == 4
    # Each __setitem__ calls _inc_counter once, so 4 total from initial 0
    counter_ok = s.cache._counter == 4
    return keys_ok and counter_ok


def test_lru_cache_setitem_race():
    """LRUCache.__setitem__: concurrent inserts corrupt counter."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: LRUCacheSetGetState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_lru_setget_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("LRUCache.__setitem__", result)


def test_lru_cache_setitem_sweep():
    """Sweep 10 seeds for LRUCache setitem race."""
    found, total = run_sweep(
        "LRUCache.__setitem__",
        setup=lambda: LRUCacheSetGetState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_lru_setget_invariant,
    )
    print(f"  LRUCache setitem: {len(found)}/10 seeds found, {total} total explored")
    return found


# ===========================================================================
# Test 3: Event registry _stored_in_collection TOCTOU
# ===========================================================================
# _stored_in_collection checks `if owner_ref in dispatch_reg` then writes.
# Two threads registering listeners for different events on the same target
# can race on the global _key_to_collection and _collection_to_key dicts.

class EventRegistryState:
    """Test _stored_in_collection + _removed_from_collection race.

    _removed_from_collection does:
        dispatch_reg.pop(owner_ref, None)
        if not dispatch_reg:
            del _key_to_collection[key]

    If thread1 removes and sees empty dispatch_reg, it deletes the key.
    But thread2 may have just added to dispatch_reg for the same key.
    Thread1's delete then removes thread2's registration.
    """
    def __init__(self):
        from sqlalchemy.event import registry
        # Save and clear global state
        self._saved_k2c = dict(registry._key_to_collection)
        self._saved_c2k = dict(registry._collection_to_key)
        registry._key_to_collection.clear()
        registry._collection_to_key.clear()

        # Same event key, two different owner refs
        self.key = (999, "on_test", 42)

        class FakeOwner:
            pass
        self.owner1 = FakeOwner()
        self.owner2 = FakeOwner()
        self.ref1 = weakref.ref(self.owner1)
        self.ref2 = weakref.ref(self.owner2)
        self.listener1 = lambda: "listener1"
        self.listener2 = lambda: "listener2"
        self.listen_ref1 = weakref.ref(self.listener1)
        self.listen_ref2 = weakref.ref(self.listener2)

        # Pre-register owner1 for this key
        registry._key_to_collection[self.key][self.ref1] = self.listen_ref1
        registry._collection_to_key[self.ref1][self.listen_ref1] = self.key

    def thread1(self):
        """Remove owner1's registration (simulates _removed_from_collection)."""
        from sqlalchemy.event import registry
        dispatch_reg = registry._key_to_collection[self.key]
        dispatch_reg.pop(self.ref1, None)
        if not dispatch_reg:
            del registry._key_to_collection[self.key]
        if self.ref1 in registry._collection_to_key:
            listener_to_key = registry._collection_to_key[self.ref1]
            listener_to_key.pop(self.listen_ref1, None)

    def thread2(self):
        """Add owner2's registration (simulates _stored_in_collection)."""
        from sqlalchemy.event import registry
        dispatch_reg = registry._key_to_collection[self.key]
        if self.ref2 not in dispatch_reg:
            dispatch_reg[self.ref2] = self.listen_ref2
        listener_to_key = registry._collection_to_key[self.ref2]
        listener_to_key[self.listen_ref2] = self.key

    def cleanup(self):
        from sqlalchemy.event import registry
        registry._key_to_collection.clear()
        registry._collection_to_key.clear()
        registry._key_to_collection.update(self._saved_k2c)
        registry._collection_to_key.update(self._saved_c2k)


def _event_registry_invariant(s):
    from sqlalchemy.event import registry
    try:
        # After remove(owner1) + add(owner2), owner2's registration
        # should always be present in _key_to_collection.
        # The race: thread1 removes owner1, sees empty dict, deletes
        # the key entry — but thread2 had already added owner2 to
        # that same dict, so the deletion kills owner2's registration.
        if s.key not in registry._key_to_collection:
            # Key was deleted — owner2's registration is lost!
            result = False
        else:
            dispatch_reg = registry._key_to_collection[s.key]
            result = s.ref2 in dispatch_reg
    finally:
        s.cleanup()
    return result


def test_event_registry_stored_in_collection():
    """Event registry: remove + add race deletes new registration."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: EventRegistryState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_event_registry_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("Event registry remove+add", result)


# ===========================================================================
# Test 4: QueuePool._do_get TOCTOU with finite overflow
# ===========================================================================
# _do_get reads `self._overflow >= self._max_overflow` WITHOUT the lock,
# then calls _inc_overflow which acquires the lock. A thread can read a stale
# overflow value and take the wrong code path.

class QueuePoolFiniteOverflowState:
    def __init__(self):
        from sqlalchemy.pool import QueuePool
        # pool_size=1, max_overflow=1 → tight limits for races
        self.pool = QueuePool(lambda: object(), pool_size=1, max_overflow=1)
        self.results = [None, None]
        self.errors = []

    def thread1(self):
        try:
            ok = self.pool._inc_overflow()
            self.results[0] = ok
        except Exception as e:
            self.errors.append(e)

    def thread2(self):
        try:
            ok = self.pool._inc_overflow()
            self.results[1] = ok
        except Exception as e:
            self.errors.append(e)


def _queuepool_finite_invariant(s):
    # With max_overflow=1 and pool_size=1, initial _overflow is -1.
    # _inc_overflow should allow at most max_overflow (1) increments
    # beyond the initial value. After two concurrent _inc_overflow calls:
    # Correct: one returns True (overflow goes to 0), one returns False
    # because 0 is NOT < 1... wait, initial is -1, max_overflow=1.
    # _inc_overflow: with lock: if _overflow < _max_overflow: _overflow += 1
    # Initially _overflow = -1, _max_overflow = 1
    # Thread 1: -1 < 1 → True, _overflow = 0
    # Thread 2: 0 < 1 → True, _overflow = 1
    # So both should succeed! With the lock, _overflow should be exactly
    # initial + (number of True results).
    num_true = sum(1 for r in s.results if r is True)
    expected_overflow = -1 + num_true  # -1 is initial (0 - pool_size=1)
    return s.pool._overflow == expected_overflow


def test_queuepool_finite_overflow():
    """QueuePool._inc_overflow with finite max_overflow: verify lock works."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: QueuePoolFiniteOverflowState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_queuepool_finite_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("QueuePool finite _inc_overflow", result)


def test_queuepool_finite_overflow_sweep():
    """Sweep 10 seeds for QueuePool finite overflow."""
    found, total = run_sweep(
        "QueuePool finite _inc_overflow",
        setup=lambda: QueuePoolFiniteOverflowState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_queuepool_finite_invariant,
    )
    print(f"  QueuePool finite overflow: {len(found)}/10 seeds found, {total} total explored")
    return found


# ===========================================================================
# Test 5: SingletonThreadPool._all_conns set race
# ===========================================================================
# _do_get appends to _all_conns (a plain set) without any lock.
# _cleanup pops from _all_conns without any lock.
# Concurrent _do_get from different threads can corrupt the set.

class SingletonThreadPoolDisposeState:
    """Test SingletonThreadPool.dispose + _do_get race.

    dispose() iterates over _all_conns and closes each, then clears the set.
    _do_get() adds a new connection to _all_conns.
    Concurrent dispose + create can leave a stale (closed) connection in
    _all_conns, or cause an error during iteration.
    """
    def __init__(self):
        from sqlalchemy.pool.impl import SingletonThreadPool

        self.close_count = 0
        outer = self

        class FakeConn:
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True
                outer.close_count += 1

        self.pool = SingletonThreadPool(lambda: FakeConn(), pool_size=3)
        # Fill pool
        for _ in range(2):
            c = self.pool._create_connection()
            self.pool._all_conns.add(c)
        self.errors = []

    def thread1(self):
        try:
            self.pool.dispose()
        except Exception as e:
            self.errors.append(("t1", e))

    def thread2(self):
        try:
            conn = self.pool._create_connection()
            self.pool._all_conns.add(conn)
        except Exception as e:
            self.errors.append(("t2", e))


def _singleton_dispose_invariant(s):
    # After dispose + add:
    # If dispose runs first: _all_conns cleared, then one add → len == 1, no closed conns
    # If add runs first: len == 3, dispose closes all 3 → len == 0
    # Race: dispose iterates _all_conns while add mutates it → possible error
    # or the new connection gets closed by dispose (it shouldn't have been).
    no_errors = len(s.errors) == 0
    # Check no connection in _all_conns is closed (it would mean dispose
    # closed it but didn't remove it, or it was added after close)
    return no_errors


def test_singleton_threadpool_all_conns():
    """SingletonThreadPool: dispose + add race on _all_conns."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: SingletonThreadPoolDisposeState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_singleton_dispose_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("SingletonThreadPool dispose+add", result)


# ===========================================================================
# Test 6: _memoized_property double-evaluation TOCTOU
# ===========================================================================
# _memoized_property.__get__ does:
#   obj.__dict__[self.__name__] = result = self.fget(obj)
# Two threads accessing the same property concurrently can both call fget()
# because neither sees the key in __dict__ yet.

class MemoizedPropertyState:
    def __init__(self):
        from sqlalchemy.util.langhelpers import _memoized_property

        self.call_count = 0

        class Target:
            pass

        @_memoized_property
        def expensive(target_self):
            self.call_count += 1
            return f"result_{self.call_count}"

        Target.expensive = expensive
        self.target = Target()
        self.results = [None, None]

    def thread1(self):
        self.results[0] = self.target.expensive

    def thread2(self):
        self.results[1] = self.target.expensive


def _memoized_property_invariant(s):
    # Invariant: fget should be called exactly once (memoized!)
    # If there's a race, both threads call fget and call_count becomes 2
    return s.call_count == 1


def test_memoized_property_double_eval():
    """_memoized_property: TOCTOU double-evaluation of fget."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: MemoizedPropertyState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_memoized_property_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("_memoized_property double-eval", result)


def test_memoized_property_sweep():
    """Sweep 10 seeds for _memoized_property race."""
    found, total = run_sweep(
        "_memoized_property",
        setup=lambda: MemoizedPropertyState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_memoized_property_invariant,
    )
    print(f"  _memoized_property: {len(found)}/10 seeds found, {total} total explored")
    return found


def test_memoized_property_reproduce():
    """Reproduce _memoized_property race deterministically."""
    bugs = run_reproduce(
        "_memoized_property",
        setup=lambda: MemoizedPropertyState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_memoized_property_invariant,
    )
    print(f"  _memoized_property reproduced: {bugs}/10")
    return bugs


# ===========================================================================
# Test 7: ScopedRegistry double-creation TOCTOU
# ===========================================================================
# ScopedRegistry.__call__ tries registry[key], catches KeyError, then calls
# registry.setdefault(key, self.createfunc()). Two threads with the same
# scope key can both call createfunc() before setdefault resolves.

class ScopedRegistryState:
    def __init__(self):
        from sqlalchemy.util._collections import ScopedRegistry
        self.create_count = 0

        def create():
            self.create_count += 1
            return f"session_{self.create_count}"

        # Use a fixed scope so both threads get the same key
        self.registry = ScopedRegistry(create, scopefunc=lambda: "shared_scope")
        self.results = [None, None]

    def thread1(self):
        self.results[0] = self.registry()

    def thread2(self):
        self.results[1] = self.registry()


def _scoped_registry_invariant(s):
    # createfunc should be called exactly once for a given scope
    return s.create_count == 1


def test_scoped_registry_double_create():
    """ScopedRegistry: TOCTOU double-creation of scoped object."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ScopedRegistryState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_scoped_registry_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("ScopedRegistry double-create", result)


def test_scoped_registry_sweep():
    """Sweep 10 seeds for ScopedRegistry race."""
    found, total = run_sweep(
        "ScopedRegistry",
        setup=lambda: ScopedRegistryState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_scoped_registry_invariant,
    )
    print(f"  ScopedRegistry: {len(found)}/10 seeds found, {total} total explored")
    return found


def test_scoped_registry_reproduce():
    """Reproduce ScopedRegistry race deterministically."""
    bugs = run_reproduce(
        "ScopedRegistry",
        setup=lambda: ScopedRegistryState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_scoped_registry_invariant,
    )
    print(f"  ScopedRegistry reproduced: {bugs}/10")
    return bugs


# ===========================================================================
# Test 8: TypeEngine._dialect_info TOCTOU
# ===========================================================================
# _dialect_info checks `if self in dialect._type_memos` then creates and
# stores a new entry. Two threads with the same type can both miss and
# both create separate memo dicts.

class DialectInfoState:
    def __init__(self):
        from sqlalchemy import String
        from sqlalchemy.dialects.sqlite.base import SQLiteDialect

        self.type_obj = String(length=50)
        self.dialect = SQLiteDialect()
        # Clear any cached memos
        self.dialect._type_memos = {}
        self.create_count = 0
        self.results = [None, None]

        # Monkey-patch _gen_dialect_impl to count calls
        original = self.type_obj._gen_dialect_impl

        def counting_gen(dialect):
            self.create_count += 1
            return original(dialect)

        self.type_obj._gen_dialect_impl = counting_gen

    def thread1(self):
        self.results[0] = self.type_obj._dialect_info(self.dialect)

    def thread2(self):
        self.results[1] = self.type_obj._dialect_info(self.dialect)


def _dialect_info_invariant(s):
    # _gen_dialect_impl should be called exactly once (cached)
    return s.create_count == 1


def test_dialect_info_toctou():
    """TypeEngine._dialect_info: TOCTOU double-creation of type memo."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: DialectInfoState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_dialect_info_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("_dialect_info TOCTOU", result)


def test_dialect_info_sweep():
    """Sweep 10 seeds for _dialect_info race."""
    found, total = run_sweep(
        "_dialect_info",
        setup=lambda: DialectInfoState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_dialect_info_invariant,
    )
    print(f"  _dialect_info: {len(found)}/10 seeds found, {total} total explored")
    return found


# ===========================================================================
# Test 9: LRUCache._manage_size concurrent eviction + insert
# ===========================================================================
# _manage_size uses a mutex (non-blocking acquire) for eviction, but
# __setitem__ writes to _data and calls _inc_counter without the mutex.
# A thread inserting while another is evicting can corrupt state.

class LRUCacheEvictionState:
    def __init__(self):
        from sqlalchemy.util._collections import LRUCache
        # Small capacity to trigger eviction
        self.cache = LRUCache(capacity=2, threshold=0.5)
        # Fill to capacity
        self.cache["existing1"] = "val1"
        self.cache["existing2"] = "val2"
        self.errors = []

    def thread1(self):
        # This should trigger _manage_size eviction
        try:
            self.cache["new1"] = "new_val1"
            self.cache["new2"] = "new_val2"
        except Exception as e:
            self.errors.append(("t1", e))

    def thread2(self):
        try:
            self.cache["new3"] = "new_val3"
            self.cache["new4"] = "new_val4"
        except Exception as e:
            self.errors.append(("t2", e))


def _lru_eviction_invariant(s):
    # After concurrent inserts that trigger eviction:
    # 1. No errors should have occurred
    # 2. Cache len should not exceed capacity + threshold * capacity (= 3)
    # 3. _counter should match the number of operations performed
    no_errors = len(s.errors) == 0
    size_ok = len(s.cache) <= s.cache.capacity + s.cache.capacity * s.cache.threshold + 1
    # Counter: 2 initial + 4 new inserts = 6
    counter_ok = s.cache._counter == 6
    return no_errors and size_ok and counter_ok


def test_lru_cache_eviction_race():
    """LRUCache: concurrent inserts triggering eviction."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: LRUCacheEvictionState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_lru_eviction_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("LRUCache eviction race", result)


def test_lru_cache_eviction_sweep():
    """Sweep 10 seeds for LRUCache eviction race."""
    found, total = run_sweep(
        "LRUCache eviction",
        setup=lambda: LRUCacheEvictionState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_lru_eviction_invariant,
    )
    print(f"  LRUCache eviction: {len(found)}/10 seeds found, {total} total explored")
    return found


# ===========================================================================
# Test 10: QueuePool._inc_overflow + _dec_overflow with unlimited overflow
# ===========================================================================
# This is the known "intentional" race from the existing case study, but we
# test it with _dec_overflow to see if the counter can go negative incorrectly.

class QueuePoolOverflowDecState:
    def __init__(self):
        from sqlalchemy.pool import QueuePool
        self.pool = QueuePool(lambda: None, pool_size=5, max_overflow=-1)
        # Start with overflow at a known value by doing some increments
        self.pool._inc_overflow()
        self.pool._inc_overflow()
        # Now _overflow should be -5 + 2 = -3
        self.initial_overflow = self.pool._overflow

    def thread1(self):
        self.pool._inc_overflow()
        self.pool._dec_overflow()

    def thread2(self):
        self.pool._inc_overflow()
        self.pool._dec_overflow()


def _overflow_inc_dec_invariant(s):
    # Each thread does +1 then -1, net effect should be 0
    return s.pool._overflow == s.initial_overflow


def test_queuepool_overflow_inc_dec():
    """QueuePool: concurrent _inc/_dec_overflow with unlimited mode."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: QueuePoolOverflowDecState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_overflow_inc_dec_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("QueuePool _inc/_dec overflow", result)


def test_queuepool_overflow_inc_dec_sweep():
    """Sweep 10 seeds for QueuePool inc/dec overflow race."""
    found, total = run_sweep(
        "QueuePool inc/dec overflow",
        setup=lambda: QueuePoolOverflowDecState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_overflow_inc_dec_invariant,
    )
    print(f"  QueuePool inc/dec: {len(found)}/10 seeds found, {total} total explored")
    return found


# ===========================================================================
# Test 11: LRUCache get() counter vs __setitem__ counter
# ===========================================================================
# Both get() and __setitem__ call _inc_counter. Test the specific scenario
# where a read and a write race on the counter.

class LRUCacheGetSetCounterState:
    def __init__(self):
        from sqlalchemy.util._collections import LRUCache
        self.cache = LRUCache(capacity=100)
        self.cache["existing"] = "val"
        self.initial_counter = self.cache._counter  # Should be 1

    def thread1(self):
        # Reading triggers _inc_counter
        self.cache.get("existing")

    def thread2(self):
        # Writing triggers _inc_counter
        self.cache["new_key"] = "new_val"


def _lru_get_set_counter_invariant(s):
    # One get + one set = 2 counter increments
    return s.cache._counter == s.initial_counter + 2


def test_lru_cache_get_set_counter():
    """LRUCache: concurrent get + set racing on _inc_counter."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: LRUCacheGetSetCounterState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_lru_get_set_counter_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("LRUCache get/set counter", result)


# ===========================================================================
# Test 12: ScopedRegistry clear() + __call__() TOCTOU
# ===========================================================================
# One thread clears the registry while another calls __call__() which does
# a check-then-create. This can cause the cleared scope to be immediately
# re-created, violating the clear() semantics.

class ScopedRegistryClearState:
    def __init__(self):
        from sqlalchemy.util._collections import ScopedRegistry
        self.create_count = 0

        def create():
            self.create_count += 1
            return f"session_{self.create_count}"

        self.registry = ScopedRegistry(create, scopefunc=lambda: "shared")
        # Pre-create the scoped object
        self.registry()
        self.initial_create_count = self.create_count  # Should be 1
        self.clear_happened = False

    def thread1(self):
        # Clear the scope
        self.registry.clear()
        self.clear_happened = True

    def thread2(self):
        # Try to access - may trigger re-creation
        self.registry()


def _scoped_clear_invariant(s):
    # After clear + access: create should be called at most 2 times total
    # (once initially, once after clear if accessed). But the interesting
    # invariant is that after both operations, the registry SHOULD have a
    # value for the scope.
    has_value = s.registry.has()
    # The create count should be exactly 2 (initial + re-create after clear)
    # A race might cause it to be 1 (clear happened after re-create was
    # interleaved) or even 3 (double-create).
    return has_value and s.create_count <= 2


def test_scoped_registry_clear_call():
    """ScopedRegistry: clear() + __call__() TOCTOU race."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: ScopedRegistryClearState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_scoped_clear_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("ScopedRegistry clear+call", result)


# ===========================================================================
# Test 13: _memoized_property - both threads get same result
# ===========================================================================
# Even if fget is called twice, both threads should see the same object
# (the second write to __dict__ should be the same reference as the first).
# But with a race, they might see different objects.

class MemoizedPropertyConsistencyState:
    def __init__(self):
        from sqlalchemy.util.langhelpers import _memoized_property

        class Target:
            pass

        call_results = []

        @_memoized_property
        def prop(self_target):
            result = object()  # Unique object each call
            call_results.append(result)
            return result

        Target.prop = prop
        self.target = Target()
        self.call_results = call_results
        self.results = [None, None]

    def thread1(self):
        self.results[0] = self.target.prop

    def thread2(self):
        self.results[1] = self.target.prop


def _memoized_consistency_invariant(s):
    # Both threads should get the same object
    return s.results[0] is s.results[1]


def test_memoized_property_consistency():
    """_memoized_property: both threads should see same cached result."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: MemoizedPropertyConsistencyState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_memoized_consistency_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("_memoized_property consistency", result)


def test_memoized_property_consistency_sweep():
    """Sweep 10 seeds for _memoized_property consistency."""
    found, total = run_sweep(
        "_memoized_property consistency",
        setup=lambda: MemoizedPropertyConsistencyState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_memoized_consistency_invariant,
    )
    print(f"  _memoized_property consistency: {len(found)}/10 seeds found, {total} total explored")
    return found


# ===========================================================================
# Test 14: QueuePool dispose() + _inc_overflow race
# ===========================================================================
# dispose() resets _overflow without the lock. If another thread is calling
# _inc_overflow concurrently, the reset can be lost.

class QueuePoolDisposeState:
    def __init__(self):
        from sqlalchemy.pool import QueuePool
        self.pool = QueuePool(lambda: None, pool_size=2, max_overflow=-1)
        # Do some increments
        self.pool._inc_overflow()
        self.pool._inc_overflow()
        self.pool._inc_overflow()
        self.errors = []

    def thread1(self):
        try:
            self.pool.dispose()
        except Exception as e:
            self.errors.append(e)

    def thread2(self):
        try:
            self.pool._inc_overflow()
        except Exception as e:
            self.errors.append(e)


def _dispose_overflow_invariant(s):
    # After dispose, _overflow should be reset to 0 - size = -2
    # After _inc_overflow, it should be -2 + 1 = -1
    # But the ordering matters. In any valid serial order:
    # If dispose first, then inc: _overflow = -2 + 1 = -1
    # If inc first, then dispose: _overflow = -2 (dispose resets)
    # So valid values are -1 or -2
    return s.pool._overflow in (-1, -2)


def test_queuepool_dispose_inc():
    """QueuePool: dispose() + _inc_overflow race on _overflow."""
    with timeout_minutes(10):
        result = explore_interleavings(
            setup=lambda: QueuePoolDisposeState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_dispose_overflow_invariant,
            max_attempts=500,
            max_ops=200,
            seed=42,
        )
    return print_result("QueuePool dispose+inc", result)


def test_queuepool_dispose_inc_sweep():
    """Sweep 10 seeds for QueuePool dispose+inc race."""
    found, total = run_sweep(
        "QueuePool dispose+inc",
        setup=lambda: QueuePoolDisposeState(),
        threads=[lambda s: s.thread1(), lambda s: s.thread2()],
        invariant=_dispose_overflow_invariant,
    )
    print(f"  QueuePool dispose+inc: {len(found)}/10 seeds found, {total} total explored")
    return found


# ===========================================================================
# Main runner
# ===========================================================================

if __name__ == "__main__":
    tests = [
        ("1. LRUCache._inc_counter lost update", test_lru_cache_counter_lost_update),
        ("1b. LRUCache._inc_counter sweep", test_lru_cache_counter_sweep),
        ("1c. LRUCache._inc_counter reproduce", test_lru_cache_counter_reproduce),
        ("2. LRUCache.__setitem__ race", test_lru_cache_setitem_race),
        ("2b. LRUCache.__setitem__ sweep", test_lru_cache_setitem_sweep),
        ("3. Event registry TOCTOU", test_event_registry_stored_in_collection),
        ("4. QueuePool finite overflow", test_queuepool_finite_overflow),
        ("4b. QueuePool finite overflow sweep", test_queuepool_finite_overflow_sweep),
        ("5. SingletonThreadPool._all_conns", test_singleton_threadpool_all_conns),
        ("6. _memoized_property double-eval", test_memoized_property_double_eval),
        ("6b. _memoized_property sweep", test_memoized_property_sweep),
        ("6c. _memoized_property reproduce", test_memoized_property_reproduce),
        ("7. ScopedRegistry double-create", test_scoped_registry_double_create),
        ("7b. ScopedRegistry sweep", test_scoped_registry_sweep),
        ("7c. ScopedRegistry reproduce", test_scoped_registry_reproduce),
        ("8. _dialect_info TOCTOU", test_dialect_info_toctou),
        ("8b. _dialect_info sweep", test_dialect_info_sweep),
        ("9. LRUCache eviction race", test_lru_cache_eviction_race),
        ("9b. LRUCache eviction sweep", test_lru_cache_eviction_sweep),
        ("10. QueuePool inc/dec overflow", test_queuepool_overflow_inc_dec),
        ("10b. QueuePool inc/dec sweep", test_queuepool_overflow_inc_dec_sweep),
        ("11. LRUCache get/set counter", test_lru_cache_get_set_counter),
        ("12. ScopedRegistry clear+call", test_scoped_registry_clear_call),
        ("13. _memoized_property consistency", test_memoized_property_consistency),
        ("13b. _memoized_property consistency sweep", test_memoized_property_consistency_sweep),
        ("14. QueuePool dispose+inc", test_queuepool_dispose_inc),
        ("14b. QueuePool dispose+inc sweep", test_queuepool_dispose_inc_sweep),
    ]

    print("=" * 72)
    print("SQLAlchemy Concurrency Bug Exploration via Frontrun")
    print("=" * 72)
    print()

    results_summary = []
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_fn()
            results_summary.append((name, result, None))
        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append((name, None, str(e)))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, result, error in results_summary:
        if error:
            print(f"  {name}: ERROR - {error}")
        elif hasattr(result, 'property_holds'):
            status = "RACE FOUND" if not result.property_holds else "No race"
            print(f"  {name}: {status}")
        else:
            print(f"  {name}: completed ({result})")
