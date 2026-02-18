"""
Buggy concurrent programs for testing frontrun's ability to detect common concurrency bugs.

This module contains intentionally buggy implementations of concurrent data structures
and operations to demonstrate four classes of concurrency bugs:

1. Atomicity Violation - Read-modify-write without proper synchronization
2. Order Violation - Thread uses uninitialized resource
3. Deadlock - Circular lock dependency
4. Async Suspension-Point Race - Async await between read and write
"""

import threading

# ============================================================================
# Bug Class 1: Atomicity Violation (threading)
# ============================================================================


class BuggyCounter:
    """
    Counter with atomicity violation - increment is not atomic.

    Bug: The increment operation reads self.value, computes new_value,
    then writes back. Two threads can interleave and cause a lost update.

    Version for use with trace_markers - has # frontrun: comments.
    """

    def __init__(self):
        self.value = 0

    def increment(self):
        # Read current value
        current = self.value  # frontrun: read_value

        # Compute new value
        new_value = current + 1

        # Write back (this can get interleaved with another thread)
        self.value = new_value  # frontrun: write_value


class BuggyCounterBytecode:
    """
    Counter with atomicity violation for bytecode-level testing.

    Same bug as BuggyCounter but without trace markers.
    The bytecode module will control interleaving at opcode level.
    """

    def __init__(self):
        self.value = 0

    def increment(self):
        # Read-modify-write without locking
        current = self.value
        new_value = current + 1
        self.value = new_value


# ============================================================================
# Bug Class 2: Order Violation (threading)
# ============================================================================


class BuggyResourceManager:
    """
    Resource manager with order violation.

    Bug: The use_resource() method assumes init_resource() has already
    been called, but there's no synchronization ensuring this order.
    If use_resource() runs before init_resource(), it will see None.

    Version for use with trace_markers - has # frontrun: comments.
    """

    def __init__(self):
        self.resource = None
        self.used_before_init = False

    def init_resource(self, value):
        """Initialize the resource - should be called before use."""
        self.resource = value  # frontrun: init_resource

    def use_resource(self):
        """Use the resource - assumes it's been initialized."""
        val = self.resource  # frontrun: use_resource
        if val is None:
            self.used_before_init = True
            return None
        return val.upper()


class BuggyResourceManagerBytecode:
    """
    Resource manager with order violation for bytecode-level testing.

    Same bug as BuggyResourceManager but without trace markers.
    """

    def __init__(self):
        self.resource = None
        self.used_before_init = False

    def init_resource(self, value):
        """Initialize the resource - should be called before use."""
        self.resource = value

    def use_resource(self):
        """Use the resource - assumes it's been initialized."""
        # Check if resource is None (order violation)
        if self.resource is None:
            self.used_before_init = True
            return None
        result = self.resource.upper()
        return result


# ============================================================================
# Bug Class 3: Deadlock (threading)
# ============================================================================


class BuggyBankWithDeadlock:
    """
    Bank with two accounts that can deadlock during transfers.

    Bug: transfer_a_to_b acquires lock_a then lock_b.
         transfer_b_to_a acquires lock_b then lock_a.
         If these run concurrently, they can deadlock.

    Version for use with trace_markers - has # frontrun: comments.
    """

    def __init__(self):
        self.account_a = 100
        self.account_b = 100
        self.lock_a = threading.Lock()
        self.lock_b = threading.Lock()

    def transfer_a_to_b(self, amount):
        """Transfer from account A to account B."""
        self.lock_a.acquire()  # frontrun: acquire_lock_a
        try:
            self.lock_b.acquire()  # frontrun: acquire_lock_b
            try:
                self.account_a -= amount
                self.account_b += amount
            finally:
                self.lock_b.release()
        finally:
            self.lock_a.release()

    def transfer_b_to_a(self, amount):
        """Transfer from account B to account A."""
        self.lock_b.acquire()  # frontrun: acquire_lock_b_reverse
        try:
            self.lock_a.acquire()  # frontrun: acquire_lock_a_reverse
            try:
                self.account_b -= amount
                self.account_a += amount
            finally:
                self.lock_a.release()
        finally:
            self.lock_b.release()


class BuggyBankWithDeadlockBytecode:
    """
    Bank with two accounts that can deadlock during transfers.

    Same bug as BuggyBankWithDeadlock: two locks acquired in opposite order.
    For bytecode-level testing. When run with cooperative_locks=False, real
    deadlocks will occur (manifesting as TimeoutError, caught by run_with_schedule).
    The `completed` flag lets the invariant check whether both transfers finished.
    """

    def __init__(self):
        self.account_a = 100
        self.account_b = 100
        self.lock_a = threading.Lock()
        self.lock_b = threading.Lock()
        self.transfer_a_to_b_completed = False
        self.transfer_b_to_a_completed = False

    def transfer_a_to_b(self, amount):
        """Transfer from account A to account B. Acquires lock_a then lock_b."""
        with self.lock_a, self.lock_b:
            self.account_a -= amount
            self.account_b += amount
            self.transfer_a_to_b_completed = True

    def transfer_b_to_a(self, amount):
        """Transfer from account B to account A. Acquires lock_b then lock_a (opposite order!)."""
        with self.lock_b, self.lock_a:
            self.account_b -= amount
            self.account_a += amount
            self.transfer_b_to_a_completed = True

    @property
    def completed(self):
        """Both transfers completed successfully."""
        return self.transfer_a_to_b_completed and self.transfer_b_to_a_completed


# ============================================================================
# Bug Class 4: Async Suspension-Point Race
# ============================================================================


class AsyncBuggyCounter:
    """
    Async counter with suspension-point race condition.

    Bug: Between reading self.value and writing it back, there's an await.
    Another task can interleave and cause a lost update.

    Version for use with async_trace_markers - uses # frontrun: comments.
    """

    def __init__(self):
        self.value = 0

    async def get_value(self):
        """Get counter value (simulates async I/O)."""
        return self.value

    async def set_value(self, value):
        """Set counter value (simulates async I/O)."""
        self.value = value

    async def increment(self):
        """Increment the counter (not atomic due to await in the middle)."""
        # frontrun: read_value
        current = await self.get_value()
        # frontrun: write_value
        await self.set_value(current + 1)


class AsyncBuggyCounterBytecode:
    """
    Async counter with suspension-point race for async_bytecode testing.

    Same bug as AsyncBuggyCounter but uses await_point() instead of mark().
    """

    def __init__(self):
        self.value = 0

    async def increment(self):
        """Increment the counter with explicit suspension point."""
        # Import here to avoid import errors if async_bytecode not available
        from frontrun.async_bytecode import await_point

        current = self.value
        await await_point()

        new_value = current + 1
        await await_point()

        self.value = new_value


class AsyncBuggyResourceManager:
    """
    Async resource manager with order violation.

    Bug: use_resource() assumes init_resource() has completed,
    but there's no synchronization between async tasks.

    Version for use with async_trace_markers - uses # frontrun: comments.
    """

    def __init__(self):
        self.resource = None
        self.used_before_init = False

    async def _write_resource(self, value):
        """Write resource value (simulates async I/O)."""
        self.resource = value

    async def _read_resource(self):
        """Read resource value (simulates async I/O)."""
        return self.resource

    async def init_resource(self, value):
        """Initialize the resource - should be called before use."""
        # frontrun: init_resource
        await self._write_resource(value)

    async def use_resource(self):
        """Use the resource - assumes it's been initialized."""
        # frontrun: use_resource
        val = await self._read_resource()
        if val is None:
            self.used_before_init = True
            return None
        return val.upper()


class AsyncBuggyResourceManagerBytecode:
    """
    Async resource manager with order violation for async_bytecode testing.
    """

    def __init__(self):
        self.resource = None
        self.used_before_init = False

    async def init_resource(self, value):
        """Initialize the resource."""
        from frontrun.async_bytecode import await_point

        await await_point()
        self.resource = value

    async def use_resource(self):
        """Use the resource - assumes it's been initialized."""
        from frontrun.async_bytecode import await_point

        await await_point()
        # Check if resource is None (order violation)
        if self.resource is None:
            self.used_before_init = True
            return None
        result = self.resource.upper()
        return result
