"""
Concurrency tests for cachetools using frontrun bytecode exploration.

Bug-finding tests target areas WITHOUT proper synchronization.
Safe-area tests target areas WITH locks or inherent thread-safety.

Repository: https://github.com/tkem/cachetools (commit e5f8f01, v7.0.1)
"""

import os
import signal
import sys
from contextlib import contextmanager

_test_dir = os.path.dirname(os.path.abspath(__file__))
_repo_src = os.path.join(_test_dir, "..", "external_repos", "cachetools", "src")
sys.path.insert(0, os.path.abspath(_repo_src))

from frontrun.bytecode import explore_interleavings  # noqa: E402


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


def print_result(name, result):
    if result.property_holds:
        print(f"  [{name}] SAFE: invariant held across {result.num_explored} interleavings")
    else:
        print(f"  [{name}] RACE FOUND after {result.num_explored} interleavings!")
    return result


# ===========================================================================
# BUG-FINDING TESTS (areas that should race)
# ===========================================================================


# --- B1: Cache.__setitem__ currsize lost update ---

class CacheCurrsizeState:
    def __init__(self):
        from cachetools import Cache
        self.cache = Cache(maxsize=100)

    def thread1(self):
        self.cache["a"] = "value_a"
        self.cache["b"] = "value_b"

    def thread2(self):
        self.cache["c"] = "value_c"
        self.cache["d"] = "value_d"


def _currsize_invariant(s):
    return s.cache.currsize == len(s.cache)


def test_b1_cache_currsize(max_attempts=500, max_ops=200):
    """Cache.__setitem__: currsize lost update."""
    with timeout_minutes(5):
        return print_result("Cache currsize", explore_interleavings(
            setup=lambda: CacheCurrsizeState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_currsize_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B2: Cache.__delitem__ currsize lost update ---

class CacheDelCurrsizeState:
    def __init__(self):
        from cachetools import Cache
        self.cache = Cache(maxsize=100)
        for i in range(4):
            self.cache[f"k{i}"] = f"v{i}"

    def thread1(self):
        del self.cache["k0"]

    def thread2(self):
        del self.cache["k1"]


def _del_currsize_invariant(s):
    return s.cache.currsize == len(s.cache)


def test_b2_cache_del_currsize(max_attempts=500, max_ops=200):
    """Cache.__delitem__: currsize lost update on concurrent delete."""
    with timeout_minutes(5):
        return print_result("Cache del currsize", explore_interleavings(
            setup=lambda: CacheDelCurrsizeState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_del_currsize_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B3: LRUCache __getitem__ + __delitem__ TOCTOU ---

class LRUGetDelState:
    def __init__(self):
        from cachetools import LRUCache
        self.cache = LRUCache(maxsize=10)
        self.cache["target"] = "value"
        self.cache["other"] = "other_val"
        self.errors = []

    def thread1(self):
        try:
            _ = self.cache["target"]  # triggers __touch -> move_to_end
        except KeyError:
            pass  # acceptable if deleted

    def thread2(self):
        try:
            del self.cache["target"]
        except KeyError:
            pass


def _lru_get_del_invariant(s):
    # No unexpected exceptions should have leaked out
    # After operations, internal __order should match __data
    order_keys = set(s.cache._LRUCache__order.keys())
    data_keys = set(s.cache.keys())
    return order_keys == data_keys


def test_b3_lru_get_del(max_attempts=500, max_ops=200):
    """LRUCache: concurrent get + delete TOCTOU on __order."""
    with timeout_minutes(5):
        return print_result("LRU get+del TOCTOU", explore_interleavings(
            setup=lambda: LRUGetDelState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_lru_get_del_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B4: RRCache __setitem__ index collision ---

class RRCacheIndexState:
    def __init__(self):
        from cachetools import RRCache
        self.cache = RRCache(maxsize=100)
        self.errors = []

    def thread1(self):
        try:
            self.cache["a"] = 1
            self.cache["b"] = 2
        except Exception as e:
            self.errors.append(e)

    def thread2(self):
        try:
            self.cache["c"] = 3
            self.cache["d"] = 4
        except Exception as e:
            self.errors.append(e)


def _rr_index_invariant(s):
    # All indices must be unique and point to correct keys
    index = s.cache._RRCache__index
    keys = s.cache._RRCache__keys
    if len(index) != len(keys):
        return False
    for key, idx in index.items():
        if idx >= len(keys) or keys[idx] != key:
            return False
    return True


def test_b4_rrcache_index(max_attempts=500, max_ops=200):
    """RRCache.__setitem__: concurrent inserts corrupt index."""
    with timeout_minutes(5):
        return print_result("RRCache index collision", explore_interleavings(
            setup=lambda: RRCacheIndexState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_rr_index_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B5: @cached without lock - hits/misses lost update ---

class CachedStatsState:
    def __init__(self):
        from cachetools import LRUCache, cached
        self.cache = LRUCache(maxsize=100)

        @cached(self.cache, info=True)
        def compute(x):
            return x * 2

        self.compute = compute
        # Pre-populate cache so both threads get hits
        self.compute(1)
        self.compute(2)

    def thread1(self):
        self.compute(1)
        self.compute(2)

    def thread2(self):
        self.compute(1)
        self.compute(2)


def _cached_stats_invariant(s):
    info = s.compute.cache_info()
    # 2 initial misses + 4 hits (2 per thread) = total 6 calls
    # hits should be 4, misses should be 2
    return info.hits == 4 and info.misses == 2


def test_b5_cached_stats(max_attempts=500, max_ops=200):
    """@cached without lock: hits/misses lost update."""
    with timeout_minutes(5):
        return print_result("@cached stats lost update", explore_interleavings(
            setup=lambda: CachedStatsState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_cached_stats_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B6: LFUCache concurrent get corrupts frequency links ---

class LFUGetState:
    def __init__(self):
        from cachetools import LFUCache
        self.cache = LFUCache(maxsize=10)
        self.cache["a"] = 1
        self.cache["b"] = 2
        self.cache["c"] = 3
        self.crashed = False

    def thread1(self):
        try:
            _ = self.cache["a"]
            _ = self.cache["b"]
        except (KeyError, AttributeError):
            self.crashed = True

    def thread2(self):
        try:
            _ = self.cache["a"]
            _ = self.cache["c"]
        except (KeyError, AttributeError):
            self.crashed = True


def _lfu_links_invariant(s):
    # A crash IS a race (KeyError in link.keys.remove)
    if s.crashed:
        return False
    # Every key in __links should exist in cache, and vice versa
    links = s.cache._LFUCache__links
    data_keys = set(s.cache.keys())
    link_keys = set(links.keys())
    return link_keys == data_keys


def test_b6_lfu_links(max_attempts=500, max_ops=200):
    """LFUCache: concurrent gets corrupt frequency link structure."""
    with timeout_minutes(5):
        return print_result("LFU link corruption", explore_interleavings(
            setup=lambda: LFUGetState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_lfu_links_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B7: Cache.__setitem__ + __delitem__ concurrent currsize ---

class CacheSetDelState:
    def __init__(self):
        from cachetools import Cache
        self.cache = Cache(maxsize=100)
        self.cache["existing"] = "val"

    def thread1(self):
        self.cache["new"] = "new_val"

    def thread2(self):
        try:
            del self.cache["existing"]
        except KeyError:
            pass


def _set_del_currsize_invariant(s):
    return s.cache.currsize == len(s.cache)


def test_b7_cache_set_del(max_attempts=500, max_ops=200):
    """Cache: concurrent set + delete corrupts currsize."""
    with timeout_minutes(5):
        return print_result("Cache set+del currsize", explore_interleavings(
            setup=lambda: CacheSetDelState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_set_del_currsize_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- B8: TTLCache _Timer.__nesting lost update ---

class TimerNestingState:
    def __init__(self):
        from cachetools import TTLCache
        self.cache = TTLCache(maxsize=10, ttl=600)
        self.cache["a"] = 1
        self.cache["b"] = 2

    def thread1(self):
        # get() uses `with self.__timer:` which does __nesting += 1 / -= 1
        self.cache.get("a")
        self.cache.get("b")

    def thread2(self):
        self.cache.get("a")
        self.cache.get("b")


def _timer_nesting_invariant(s):
    # After both threads finish, __nesting should be back to 0
    timer = s.cache.timer
    nesting = timer._Timer__nesting
    return nesting == 0


def test_b8_timer_nesting(max_attempts=500, max_ops=200):
    """TTLCache._Timer: __nesting lost update from concurrent get()."""
    with timeout_minutes(5):
        return print_result("TTLCache timer nesting", explore_interleavings(
            setup=lambda: TimerNestingState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_timer_nesting_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# ===========================================================================
# SAFE-AREA TESTS (areas that should NOT race)
# ===========================================================================


# --- S1: @cached with lock - stats should be accurate ---

class CachedLockedStatsState:
    def __init__(self):
        import threading
        from cachetools import LRUCache, cached
        self.cache = LRUCache(maxsize=100)
        self.lock = threading.Lock()

        @cached(self.cache, lock=self.lock, info=True)
        def compute(x):
            return x * 2

        self.compute = compute
        # Pre-populate
        self.compute(1)
        self.compute(2)

    def thread1(self):
        self.compute(1)
        self.compute(2)

    def thread2(self):
        self.compute(1)
        self.compute(2)


def _locked_stats_invariant(s):
    info = s.compute.cache_info()
    return info.hits == 4 and info.misses == 2


def test_s1_cached_locked_stats(max_attempts=15000, max_ops=400):
    """@cached with lock: stats should be accurate."""
    with timeout_minutes(5):
        return print_result("@cached locked stats", explore_interleavings(
            setup=lambda: CachedLockedStatsState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_locked_stats_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- S2: @cached with lock - all keys stored correctly ---

class CachedLockedDataState:
    def __init__(self):
        import threading
        from cachetools import LRUCache, cached
        self.cache = LRUCache(maxsize=100)
        self.lock = threading.Lock()

        @cached(self.cache, lock=self.lock)
        def compute(x):
            return x * 2

        self.compute = compute

    def thread1(self):
        self.compute(1)
        self.compute(2)

    def thread2(self):
        self.compute(3)
        self.compute(4)


def _locked_data_invariant(s):
    # All 4 distinct keys should be in cache with correct values
    from cachetools.keys import hashkey
    return (
        s.cache[hashkey(1)] == 2
        and s.cache[hashkey(2)] == 4
        and s.cache[hashkey(3)] == 6
        and s.cache[hashkey(4)] == 8
    )


def test_s2_cached_locked_data(max_attempts=15000, max_ops=400):
    """@cached with lock: all computed values stored correctly."""
    with timeout_minutes(5):
        return print_result("@cached locked data", explore_interleavings(
            setup=lambda: CachedLockedDataState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_locked_data_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- S3: @cached with lock+condition - no stampede, accurate stats ---

class CachedConditionState:
    def __init__(self):
        import threading
        from cachetools import LRUCache, cached
        self.cache = LRUCache(maxsize=100)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.call_count = 0
        outer = self

        @cached(self.cache, lock=self.lock, condition=self.cond, info=True)
        def compute(x):
            outer.call_count += 1
            return x * 2

        self.compute = compute

    def thread1(self):
        self.compute(42)

    def thread2(self):
        self.compute(42)


def _condition_invariant(s):
    # With condition, only ONE thread should compute for key 42
    # The other waits and gets the cached result
    info = s.compute.cache_info()
    return s.call_count == 1 and info.misses == 1 and info.hits == 1


def test_s3_cached_condition(max_attempts=15000, max_ops=400):
    """@cached with condition: prevents stampede, accurate stats."""
    with timeout_minutes(5):
        return print_result("@cached condition", explore_interleavings(
            setup=lambda: CachedConditionState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_condition_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# --- S4: Cache single-thread data integrity ---
# Even though currsize races, individual key-value pairs should be
# retrievable from the same thread that inserted them.

class CacheSingleKeyState:
    def __init__(self):
        from cachetools import Cache
        self.cache = Cache(maxsize=100)
        self.t1_ok = False
        self.t2_ok = False

    def thread1(self):
        self.cache["t1_key"] = "t1_val"
        self.t1_ok = self.cache["t1_key"] == "t1_val"

    def thread2(self):
        self.cache["t2_key"] = "t2_val"
        self.t2_ok = self.cache["t2_key"] == "t2_val"


def _single_key_invariant(s):
    return s.t1_ok and s.t2_ok


def test_s4_cache_single_key(max_attempts=15000, max_ops=400):
    """Cache: individual key-value pairs are always retrievable."""
    with timeout_minutes(5):
        return print_result("Cache single-key integrity", explore_interleavings(
            setup=lambda: CacheSingleKeyState(),
            threads=[lambda s: s.thread1(), lambda s: s.thread2()],
            invariant=_single_key_invariant,
            max_attempts=max_attempts, max_ops=max_ops, seed=42,
        ))


# ===========================================================================
# Main runner
# ===========================================================================

BUG_TESTS = [
    ("B1: Cache currsize lost update", test_b1_cache_currsize, 500, 200),
    ("B2: Cache del currsize", test_b2_cache_del_currsize, 500, 200),
    ("B3: LRU get+del TOCTOU", test_b3_lru_get_del, 500, 200),
    ("B4: RRCache index collision", test_b4_rrcache_index, 500, 200),
    ("B5: @cached stats lost update", test_b5_cached_stats, 500, 200),
    ("B6: LFU link corruption", test_b6_lfu_links, 500, 200),
    ("B7: Cache set+del currsize", test_b7_cache_set_del, 500, 200),
    ("B8: TTLCache timer nesting", test_b8_timer_nesting, 500, 200),
]

SAFE_TESTS = [
    ("S1: @cached locked stats", test_s1_cached_locked_stats, 15000, 500),
    ("S2: @cached locked data", test_s2_cached_locked_data, 15000, 500),
    ("S3: @cached condition", test_s3_cached_condition, 15000, 500),
    ("S4: Cache single-key integrity", test_s4_cache_single_key, 20000, 500),
]

ALL_TESTS = BUG_TESTS + SAFE_TESTS

if __name__ == "__main__":
    import time as _time

    print("=" * 72)
    print("cachetools Concurrency Tests via Frontrun")
    print("=" * 72)

    for name, fn, attempts, ops in ALL_TESTS:
        print(f"\n--- {name} ---")
        t0 = _time.monotonic()
        try:
            fn(max_attempts=attempts, max_ops=ops)
            print(f"  Time: {_time.monotonic() - t0:.1f}s")
        except Exception as e:
            print(f"  ERROR ({_time.monotonic() - t0:.1f}s): {e}")
