"""
Tests for threading primitives race conditions.

These tests demonstrate race conditions that occur when threading primitives
(RLock, Semaphore, Event, Condition, Queue) are not properly wrapped with
cooperative implementations.

These tests are expected to expose race conditions and may fail or pass
depending on whether the scheduler hits the race condition in a particular run.
Some tests may timeout/deadlock, demonstrating the need for cooperative wrappers.
"""

import queue
import threading

import pytest

import frontrun

# ---------------------------------------------------------------------------
# Test: threading.RLock
# ---------------------------------------------------------------------------


@pytest.mark.intentionally_leaves_dangling_threads
def test_dpor_queue_join_yields_to_consumer() -> None:
    """Queue.join() must not retain the scheduler turn needed by task_done()."""
    invariant_calls: list[bool] = []

    class State:
        def __init__(self) -> None:
            self.queue: queue.Queue[int] = queue.Queue()
            self.queue.put(1)
            self.join_returned = False
            self.consumed = False

    def joiner(state: State) -> None:
        state.queue.join()
        state.join_returned = True

    def consumer(state: State) -> None:
        state.queue.get()
        state.consumed = True
        state.queue.task_done()

    def invariant(state: State) -> bool:
        invariant_calls.append(True)
        return state.join_returned and state.consumed

    result = frontrun.explore(
        setup=State,
        workers=[joiner, consumer],
        invariant=invariant,
        detect_io=False,
        timeout_per_run=0.05,
        max_executions=1,
        reproduce_on_failure=0,
    )

    assert invariant_calls == [True], "queue join wedged, so the completed-state invariant was never evaluated"
    assert result.property_holds, result.explanation


class RLockCounter:
    """Counter using RLock that can be acquired multiple times by same thread."""

    def __init__(self):
        self.value = 0
        self._lock = threading.RLock()

    def increment_with_reentry(self):
        """Acquire lock, call helper that also acquires it."""
        with self._lock:
            self._increment_helper()

    def _increment_helper(self):
        """Helper that also acquires the same RLock."""
        with self._lock:
            temp = self.value
            self.value = temp + 1


def test_rlock_race_condition():
    """Cooperative RLock serializes reentrant increments so the counter reaches 2.

    frontrun patches ``threading.RLock`` with its cooperative version, so the
    reentrant acquire() yields to the scheduler instead of deadlocking in C.
    Mutual exclusion holds under every explored interleaving, so the invariant
    is never violated.
    """
    result = frontrun.explore_random(
        setup=lambda: RLockCounter(),
        threads=[
            lambda c: c.increment_with_reentry(),
            lambda c: c.increment_with_reentry(),
        ],
        invariant=lambda c: c.value == 2,
        max_attempts=30,
        max_ops=300,
        seed=42,
    )

    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Test: threading.Semaphore
# ---------------------------------------------------------------------------


class SemaphoreResource:
    """Resource pool protected by a Semaphore."""

    def __init__(self, max_resources=2):
        self.in_use = 0
        self.max_in_use = 0
        self.semaphore = threading.Semaphore(max_resources)
        self._lock = threading.Lock()  # For tracking stats

    def use_resource(self):
        """Acquire resource, use it, then release."""
        self.semaphore.acquire()

        # Track usage
        with self._lock:
            self.in_use += 1
            if self.in_use > self.max_in_use:
                self.max_in_use = self.in_use

        # Simulate some work
        temp = self.in_use

        # Release resource
        with self._lock:
            self.in_use -= 1

        self.semaphore.release()


def test_semaphore_race_condition():
    """Cooperative Semaphore(1) keeps at most one thread in the critical section.

    frontrun patches ``threading.Semaphore`` with its cooperative version, so an
    exhausted acquire() yields to the scheduler instead of deadlocking. The
    semaphore bound is respected under every explored interleaving, so
    ``max_in_use`` never exceeds 1.
    """
    result = frontrun.explore_random(
        setup=lambda: SemaphoreResource(max_resources=1),
        threads=[
            lambda r: r.use_resource(),
            lambda r: r.use_resource(),
        ],
        invariant=lambda r: r.max_in_use <= 1,  # Should never exceed semaphore limit
        max_attempts=20,
        max_ops=100,
        seed=42,
    )

    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Test: threading.BoundedSemaphore
# ---------------------------------------------------------------------------


class BoundedSemaphoreResource:
    """Resource with strict bounds on acquire/release pairs."""

    def __init__(self):
        self.semaphore = threading.BoundedSemaphore(2)
        self.acquired_count = 0
        self._lock = threading.Lock()

    def acquire_and_release(self):
        """Properly acquire and release the bounded semaphore."""
        self.semaphore.acquire()

        with self._lock:
            self.acquired_count += 1

        self.semaphore.release()


def test_bounded_semaphore_race_condition():
    """Cooperative BoundedSemaphore lets all three acquire/release pairs complete.

    frontrun patches ``threading.BoundedSemaphore`` with its cooperative
    version, so blocking acquire() yields instead of deadlocking. Every explored
    interleaving completes all three acquire+increment+release cycles, so
    ``acquired_count`` reaches 3.
    """
    result = frontrun.explore_random(
        setup=lambda: BoundedSemaphoreResource(),
        threads=[
            lambda r: r.acquire_and_release(),
            lambda r: r.acquire_and_release(),
            lambda r: r.acquire_and_release(),
        ],
        invariant=lambda r: r.acquired_count == 3,
        max_attempts=20,
        max_ops=150,
        seed=42,
    )

    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Direct unit tests for CooperativeSemaphore (unmanaged thread path)
# ---------------------------------------------------------------------------


def test_cooperative_semaphore_nonblocking_success():
    """Non-blocking acquire returns True when available, False when exhausted."""
    from frontrun._cooperative import CooperativeSemaphore

    sem = CooperativeSemaphore(1)
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is False
    sem.release()
    assert sem.acquire(blocking=False) is True


def test_cooperative_semaphore_timeout_expires_unmanaged():
    """Blocking acquire with timeout returns False after deadline."""
    import time

    from frontrun._cooperative import CooperativeSemaphore

    sem = CooperativeSemaphore(1)
    assert sem.acquire() is True  # take the one slot

    start = time.monotonic()
    result = sem.acquire(timeout=0.05)
    elapsed = time.monotonic() - start

    assert result is False
    assert elapsed >= 0.04
    assert elapsed < 1.0


def test_cooperative_semaphore_timeout_acquires_when_released():
    """Blocking acquire with timeout returns True if released before deadline."""
    import threading as real_threading
    import time

    from frontrun._cooperative import CooperativeSemaphore

    sem = CooperativeSemaphore(1)
    sem.acquire()

    def _releaser():
        time.sleep(0.05)
        sem.release()

    t = real_threading.Thread(target=_releaser)
    t.start()
    try:
        assert sem.acquire(timeout=2.0) is True
    finally:
        t.join()


def test_cooperative_semaphore_no_timeout_acquires_when_released():
    """Blocking acquire with no timeout waits until released (unmanaged path)."""
    import threading as real_threading
    import time

    from frontrun._cooperative import CooperativeSemaphore

    sem = CooperativeSemaphore(1)
    sem.acquire()

    def _releaser():
        time.sleep(0.05)
        sem.release()

    t = real_threading.Thread(target=_releaser)
    t.start()
    try:
        assert sem.acquire() is True
    finally:
        t.join()


# ---------------------------------------------------------------------------
# Test: threading.Event
# ---------------------------------------------------------------------------


class EventCoordinator:
    """Coordinates threads using an Event."""

    def __init__(self):
        self.event = threading.Event()
        self.ready_count = 0
        self.proceeded_count = 0
        self._lock = threading.Lock()

    def waiter(self):
        """Wait for event to be set."""
        with self._lock:
            self.ready_count += 1

        self.event.wait()  # Block until event is set

        with self._lock:
            self.proceeded_count += 1

    def setter(self):
        """Set the event after a delay."""
        # Give waiters time to start waiting
        temp = self.ready_count

        self.event.set()


def test_event_race_condition():
    """Cooperative Event lets the waiter proceed once the setter fires.

    frontrun patches ``threading.Event`` with its cooperative version, so
    ``wait()`` yields to the scheduler instead of blocking in C. The setter can
    run to set the event under every explored interleaving, so the single waiter
    always proceeds exactly once.
    """
    result = frontrun.explore_random(
        setup=lambda: EventCoordinator(),
        threads=[
            lambda e: e.waiter(),
            lambda e: e.setter(),
        ],
        invariant=lambda e: e.proceeded_count == 1,
        max_attempts=30,
        max_ops=300,
        seed=42,
    )

    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Test: threading.Condition
# ---------------------------------------------------------------------------


class ConditionQueue:
    """Simple queue using Condition for wait/notify."""

    def __init__(self):
        self.items = []
        self.condition = threading.Condition()
        self.get_count = 0
        self.put_count = 0

    def put(self, item):
        """Add item and notify waiters."""
        with self.condition:
            self.items.append(item)
            self.put_count += 1
            self.condition.notify()

    def get(self):
        """Wait for item and retrieve it."""
        with self.condition:
            while not self.items:
                self.condition.wait()  # Block until notified

            item = self.items.pop(0)
            self.get_count += 1
            return item


def test_condition_race_condition():
    """Cooperative Condition wakes the waiting getter after the putter notifies.

    frontrun patches ``threading.Condition`` with its cooperative version, so
    ``wait()`` yields to the scheduler instead of blocking in C. The putter can
    run to notify under every explored interleaving, so both the put and the get
    complete exactly once.
    """
    result = frontrun.explore_random(
        setup=lambda: ConditionQueue(),
        threads=[
            lambda q: q.put("item1"),
            lambda q: q.get(),
        ],
        invariant=lambda q: q.get_count == 1 and q.put_count == 1,
        max_attempts=20,
        max_ops=100,
        seed=42,
    )

    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Test: queue.Queue (get operation)
# ---------------------------------------------------------------------------


class QueueConsumer:
    """Consumer that gets items from a queue."""

    def __init__(self):
        self.queue = queue.Queue(maxsize=2)
        self.consumed = []
        self._lock = threading.Lock()

    def produce(self, item):
        """Put item in queue."""
        self.queue.put(item)

    def consume(self):
        """Get item from queue."""
        item = self.queue.get()  # Blocks if queue is empty
        with self._lock:
            self.consumed.append(item)


def test_queue_get_race_condition():
    """Cooperative Queue.get() blocks then resumes once the producer puts.

    frontrun patches ``queue.Queue`` with its cooperative version, so ``get()``
    on an empty queue yields to the scheduler instead of blocking in C. The
    producer can run to add an item under every explored interleaving, so the
    consumer receives exactly one item.
    """
    result = frontrun.explore_random(
        setup=lambda: QueueConsumer(),
        threads=[
            lambda q: q.produce("item1"),
            lambda q: q.consume(),
        ],
        invariant=lambda q: len(q.consumed) == 1,
        max_attempts=20,
        max_ops=100,
        seed=42,
    )

    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Test: queue.Queue (put operation)
# ---------------------------------------------------------------------------


class QueueProducer:
    """Producer that puts items in a bounded queue."""

    def __init__(self):
        self.queue = queue.Queue(maxsize=1)  # Small queue to force blocking
        self.produced = []
        self.consumed = []
        self._lock = threading.Lock()

    def produce(self, item):
        """Put item in queue (blocks if full)."""
        self.queue.put(item)  # Blocks if queue is full
        with self._lock:
            self.produced.append(item)

    def consume(self):
        """Get item from queue to make space."""
        item = self.queue.get()
        with self._lock:
            self.consumed.append(item)


def test_queue_put_race_condition():
    """Cooperative Queue.put() blocks on a full queue then resumes after a get.

    frontrun patches ``queue.Queue`` with its cooperative version, so ``put()``
    on a full queue yields to the scheduler instead of blocking in C. The
    consumer can run to make space under every explored interleaving, so both
    producers complete and the consumer receives one item.
    """
    result = frontrun.explore_random(
        setup=lambda: QueueProducer(),
        threads=[
            lambda q: q.produce("item1"),
            lambda q: q.produce("item2"),
            lambda q: q.consume(),
        ],
        invariant=lambda q: len(q.produced) == 2 and len(q.consumed) == 1,
        max_attempts=30,
        max_ops=400,
        seed=42,
    )

    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Test: Multiple primitives interacting
# ---------------------------------------------------------------------------


class MultiPrimitiveSystem:
    """System using multiple threading primitives together."""

    def __init__(self):
        self.lock = threading.RLock()
        self.event = threading.Event()
        self.queue = queue.Queue()
        self.value = 0

    def producer(self):
        """Produce value and signal."""
        with self.lock:
            self.value += 1

        self.queue.put(self.value)
        self.event.set()

    def consumer(self):
        """Wait for signal and consume."""
        self.event.wait()
        item = self.queue.get()

        with self.lock:
            self.value += item


def test_multiple_primitives_race_condition():
    """Cooperative RLock + Event + Queue compose so the consumer sums the value.

    This test combines RLock, Event, and Queue. With all three patched to their
    cooperative versions, the consumer waits for the producer's signal, consumes
    the queued value, and adds it under every explored interleaving, so the
    final value is 2.
    """
    result = frontrun.explore_random(
        setup=lambda: MultiPrimitiveSystem(),
        threads=[
            lambda s: s.producer(),
            lambda s: s.consumer(),
        ],
        invariant=lambda s: s.value == 2,  # 1 from producer, +1 from consumer
        max_attempts=20,
        max_ops=150,
        seed=42,
    )

    assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# Test: CooperativeCondition.notify(n) with negative n
# ---------------------------------------------------------------------------


def test_cooperative_condition_notify_negative_n_does_not_corrupt_state():
    """notify(n) with n < 0 must not decrement _served counter.

    Without a guard, min(n, unserved) returns the negative n and
    _served += actual decrements it, corrupting the ticket system.
    """
    from frontrun._cooperative import CooperativeCondition, CooperativeLock

    lock = CooperativeLock()
    cond = CooperativeCondition(lock)

    lock.acquire()
    served_before = cond._served
    cond.notify(-1)
    served_after = cond._served
    lock.release()

    assert served_after >= served_before, f"notify(-1) corrupted _served: was {served_before}, now {served_after}"
