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

from frontrun.bytecode import explore_interleavings

# ---------------------------------------------------------------------------
# Test: threading.RLock
# ---------------------------------------------------------------------------


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
    """Test that RLock without cooperative wrapper can cause issues.

    Without a cooperative RLock, the blocking acquire() in C code will
    deadlock the scheduler. This test demonstrates the race condition.
    """
    result = explore_interleavings(
        setup=lambda: RLockCounter(),
        threads=[
            lambda c: c.increment_with_reentry(),
            lambda c: c.increment_with_reentry(),
        ],
        invariant=lambda c: c.value == 2,
        max_attempts=100,
        max_ops=300,
        seed=42,
    )

    # The test may pass or fail depending on scheduling, but should not crash
    # Once cooperative RLock is implemented, this should consistently pass


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
    """Test that Semaphore without cooperative wrapper can cause issues.

    Without a cooperative Semaphore, blocking acquire() will deadlock
    the scheduler when the semaphore is exhausted.
    """
    result = explore_interleavings(
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

    # May pass or fail, but demonstrates the need for cooperative Semaphore


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
    """Test that BoundedSemaphore without cooperative wrapper can cause issues.

    BoundedSemaphore is like Semaphore but raises on over-release.
    Without cooperative wrapper, blocking acquire() will deadlock.
    """
    result = explore_interleavings(
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

    # Demonstrates need for cooperative BoundedSemaphore


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
    """Test that Event.wait() without cooperative wrapper can cause issues.

    Without a cooperative Event, wait() blocks in C code and deadlocks
    the scheduler. The setter thread can't run to actually set the event.
    """
    result = explore_interleavings(
        setup=lambda: EventCoordinator(),
        threads=[
            lambda e: e.waiter(),
            lambda e: e.setter(),
        ],
        invariant=lambda e: e.proceeded_count == 1,
        max_attempts=100,
        max_ops=300,
        seed=42,
    )

    # Demonstrates need for cooperative Event


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
    """Test that Condition.wait() without cooperative wrapper can cause issues.

    Without a cooperative Condition, wait() blocks in C code and deadlocks
    the scheduler. The putter thread can't run to notify.
    """
    result = explore_interleavings(
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

    # Demonstrates need for cooperative Condition


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
    """Test that Queue.get() without cooperative wrapper can cause issues.

    Without a cooperative Queue, get() blocks in C code when queue is empty,
    deadlocking the scheduler. The producer can't run to add items.
    """
    result = explore_interleavings(
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

    # Demonstrates need for cooperative Queue.get()


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
    """Test that Queue.put() without cooperative wrapper can cause issues.

    Without a cooperative Queue, put() blocks in C code when queue is full,
    deadlocking the scheduler. The consumer can't run to make space.
    """
    result = explore_interleavings(
        setup=lambda: QueueProducer(),
        threads=[
            lambda q: q.produce("item1"),
            lambda q: q.produce("item2"),
            lambda q: q.consume(),
        ],
        invariant=lambda q: len(q.produced) == 2 and len(q.consumed) == 1,
        max_attempts=100,
        max_ops=400,
        seed=42,
    )

    # Demonstrates need for cooperative Queue.put()


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
    """Test interaction of multiple threading primitives.

    This test combines RLock, Event, and Queue to demonstrate
    complex race conditions when primitives are not cooperative.
    """
    result = explore_interleavings(
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

    # Demonstrates need for all cooperative primitives working together
