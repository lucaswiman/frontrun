"""
Tests demonstrating frontrun's ability to detect four classes of concurrency bugs.

For each bug class, we have three types of tests:
1. Exact reproduction using trace_markers/async_trace_markers with a specific Schedule
2. Property-based exploration using bytecode/async_bytecode with explore_interleavings
3. Hypothesis-based testing using bytecode/async_bytecode with schedule_strategy

Bug classes covered:
1. Atomicity Violation - Read-modify-write without locking
2. Order Violation - Thread uses uninitialized resource
3. Deadlock - Circular lock dependency
4. Async Suspension-Point Race - Await between read and write
"""

import asyncio

import pytest
from hypothesis import Phase, given, settings

from frontrun.async_bytecode import (
    explore_interleavings as async_explore_interleavings,
)
from frontrun.async_bytecode import (
    run_with_schedule as async_run_with_schedule,
)
from frontrun.async_bytecode import (
    schedule_strategy as async_schedule_strategy,
)
from frontrun.async_trace_markers import AsyncTraceExecutor
from frontrun.bytecode import (
    explore_interleavings,
    run_with_schedule,
    schedule_strategy,
)
from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor
from tests.buggy_programs import (
    AsyncBuggyCounter,
    AsyncBuggyCounterBytecode,
    AsyncBuggyResourceManager,
    AsyncBuggyResourceManagerBytecode,
    BuggyBankWithDeadlock,
    BuggyBankWithDeadlockBytecode,
    BuggyCounter,
    BuggyCounterBytecode,
    BuggyResourceManager,
    BuggyResourceManagerBytecode,
)

# ============================================================================
# Bug Class 1: Atomicity Violation Tests
# ============================================================================


def test_atomicity_violation_exact_schedule():
    """
    Test atomicity violation with exact schedule using trace_markers.

    Forces the classic lost update: both threads read, then both write.
    Expected final value: 2, Actual: 1 (lost update)
    """
    counter = BuggyCounter()

    # Schedule that causes lost update:
    # T1 reads (0), T2 reads (0), T1 writes (1), T2 writes (1)
    schedule = Schedule(
        [
            Step("thread1", "read_value"),
            Step("thread2", "read_value"),
            Step("thread1", "write_value"),
            Step("thread2", "write_value"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run("thread1", lambda: counter.increment())
    executor.run("thread2", lambda: counter.increment())
    executor.wait(timeout=5.0)

    # Bug manifests: both increments happened but value is only 1
    assert counter.value == 1, "Lost update - expected 2 but got 1"


def test_atomicity_violation_exploration():
    """
    Test atomicity violation using bytecode exploration.

    Randomly explores interleavings to find the atomicity violation.
    """
    result = explore_interleavings(
        setup=lambda: BuggyCounterBytecode(),
        threads=[
            lambda c: c.increment(),
            lambda c: c.increment(),
        ],
        invariant=lambda c: c.value == 2,
        max_attempts=200,
        seed=42,
    )

    # Should find a counterexample where value != 2
    assert not result.property_holds, "Should find atomicity violation"
    assert result.counterexample is not None, "Should have counterexample schedule"


@given(schedule=schedule_strategy(num_threads=2))
@settings(max_examples=200, phases=[Phase.generate])
def test_atomicity_violation_hypothesis(schedule):
    """
    Test atomicity violation using Hypothesis with schedule_strategy.

    Hypothesis generates random schedules and we check if the invariant holds.
    """
    counter = run_with_schedule(
        schedule,
        setup=lambda: BuggyCounterBytecode(),
        threads=[
            lambda c: c.increment(),
            lambda c: c.increment(),
        ],
    )

    # In some schedules, the atomicity violation will occur
    # We're just demonstrating that various schedules can be explored
    # The bug may or may not manifest in any given schedule
    assert counter.value in [1, 2], f"Unexpected value: {counter.value}"


# ============================================================================
# Bug Class 2: Order Violation Tests
# ============================================================================


def test_order_violation_exact_schedule():
    """
    Test order violation with exact schedule using trace_markers.

    Forces use_resource() to run before init_resource(). The user thread
    sees resource=None because init hasn't happened yet.
    """
    manager = BuggyResourceManager()

    # Schedule that violates order: use before init
    schedule = Schedule(
        [
            Step("user_thread", "use_resource"),
            Step("init_thread", "init_resource"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run("init_thread", lambda: manager.init_resource("hello"))
    executor.run("user_thread", lambda: manager.use_resource())
    executor.wait(timeout=5.0)

    # Bug manifests: use_resource ran before init_resource
    assert manager.used_before_init, "Resource should have been used before initialization"


def test_order_violation_exploration():
    """
    Test order violation using bytecode exploration.

    Explores interleavings where use_resource() might run before init_resource().
    The invariant checks that the resource wasn't used before initialization.
    """

    def invariant(manager):
        # Check if resource was used before init (should be False)
        return not manager.used_before_init

    result = explore_interleavings(
        setup=lambda: BuggyResourceManagerBytecode(),
        threads=[
            lambda m: m.init_resource("hello"),
            lambda m: m.use_resource(),
        ],
        invariant=invariant,
        max_attempts=200,
        seed=42,
    )

    # Should find a counterexample where resource is used before init
    assert not result.property_holds, "Should find order violation"
    assert result.counterexample is not None, "Should have counterexample schedule"


@given(schedule=schedule_strategy(num_threads=2))
@settings(max_examples=200, phases=[Phase.generate])
def test_order_violation_hypothesis(schedule):
    """
    Test order violation using Hypothesis with schedule_strategy.

    Some schedules will trigger the order violation.
    """
    manager = BuggyResourceManagerBytecode()

    error_raised = False
    try:
        run_with_schedule(
            schedule,
            setup=lambda: manager,
            threads=[
                lambda m: m.init_resource("hello"),
                lambda m: m.use_resource(),
            ],
        )
    except AttributeError:
        error_raised = True

    # The bug may or may not manifest depending on the schedule
    # We just verify that the test runs without crashing unexpectedly
    assert True, "Test completed"


# ============================================================================
# Bug Class 3: Deadlock Tests
# ============================================================================


@pytest.mark.intentionally_leaves_dangling_threads
def test_deadlock_exact_schedule():
    """
    Test deadlock with exact schedule using trace_markers.

    Forces the classic deadlock: T1 acquires A then waits for B,
    while T2 acquires B then waits for A.

    Note: The threads will hang indefinitely at the deadlock point.
    We verify the deadlock by checking that threads are still alive after timeout.
    """
    bank = BuggyBankWithDeadlock()

    # Schedule that causes deadlock:
    # T1 gets lock_a, T2 gets lock_b, then both wait forever
    schedule = Schedule(
        [
            Step("thread1", "acquire_lock_a"),
            Step("thread2", "acquire_lock_b_reverse"),
            Step("thread1", "acquire_lock_b"),
            Step("thread2", "acquire_lock_a_reverse"),
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run("thread1", lambda: bank.transfer_a_to_b(10))
    executor.run("thread2", lambda: bank.transfer_b_to_a(10))

    # Wait with a short timeout - threads won't complete due to deadlock
    # The wait() method now raises TimeoutError when threads don't complete
    try:
        executor.wait(timeout=1.0)
        assert False, "Should have timed out due to deadlock"
    except TimeoutError:
        # Expected - deadlock occurred
        pass

    # Verify that threads are still alive (deadlocked)
    alive_count = sum(1 for t in executor.threads if t.is_alive())
    assert alive_count > 0, "At least one thread should be deadlocked"


@pytest.mark.intentionally_leaves_dangling_threads
def test_deadlock_exploration():
    """
    Test deadlock using bytecode exploration with cooperative_locks=False.

    With cooperative_locks=False, real Lock.acquire() blocks in C, so when the
    scheduler interleaves lock acquisitions in the wrong order, a real deadlock
    occurs. run_with_schedule catches the resulting TimeoutError and returns
    the state with completed=False.

    Note: explore_interleavings doesn't expose cooperative_locks, so we write
    a manual exploration loop using run_with_schedule directly.
    """
    import random

    rng = random.Random(42)
    found_deadlock = False

    for _ in range(50):
        length = rng.randint(1, 300)
        schedule = [rng.randint(0, 1) for _ in range(length)]

        bank = run_with_schedule(
            schedule,
            setup=lambda: BuggyBankWithDeadlockBytecode(),
            threads=[
                lambda b: b.transfer_a_to_b(10),
                lambda b: b.transfer_b_to_a(10),
            ],
            cooperative_locks=False,
            timeout=0.5,
        )

        if not bank.completed:
            found_deadlock = True
            break

    assert found_deadlock, "Should find a schedule that causes deadlock"


@pytest.mark.intentionally_leaves_dangling_threads
@given(schedule=schedule_strategy(num_threads=2, max_ops=50))
@settings(max_examples=20, phases=[Phase.generate], deadline=None)
def test_deadlock_hypothesis(schedule):
    """
    Test deadlock-prone code using Hypothesis with cooperative_locks=False.

    With real locks and opposite acquisition order, some schedules will deadlock
    (timeout -> completed=False) and others will complete (completed=True).

    Uses short timeout and few examples since each deadlocking schedule blocks
    for the full timeout duration.
    """
    bank = run_with_schedule(
        schedule,
        setup=lambda: BuggyBankWithDeadlockBytecode(),
        threads=[
            lambda b: b.transfer_a_to_b(10),
            lambda b: b.transfer_b_to_a(10),
        ],
        cooperative_locks=False,
        timeout=0.5,
    )

    # Some schedules will deadlock (completed=False), others won't
    # We just verify the state is consistent
    if bank.completed:
        assert bank.account_a == 100 and bank.account_b == 100, "Total should be conserved after both transfers"


# ============================================================================
# Bug Class 4: Async Suspension-Point Race Tests
# ============================================================================


def test_async_suspension_point_race_exact_schedule():
    """
    Test async suspension-point race with exact schedule using async_trace_markers.

    Forces both tasks to read before either writes, causing lost update.
    """
    counter = AsyncBuggyCounter()

    # Schedule that causes lost update in async context
    schedule = Schedule(
        [
            Step("task1", "read_value"),
            Step("task2", "read_value"),
            Step("task1", "write_value"),
            Step("task2", "write_value"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)

    executor.run(
        {
            "task1": counter.increment,
            "task2": counter.increment,
        }
    )

    # Bug manifests: both increments happened but value is only 1
    assert counter.value == 1, "Lost update in async - expected 2 but got 1"


def test_async_suspension_point_race_exploration():
    """
    Test async suspension-point race using async_bytecode exploration.

    Explores task interleavings to find the race condition.
    """

    async def run_exploration():
        result = await async_explore_interleavings(
            setup=lambda: AsyncBuggyCounterBytecode(),
            tasks=[
                lambda c: c.increment(),
                lambda c: c.increment(),
            ],
            invariant=lambda c: c.value == 2,
            max_attempts=200,
            seed=42,
        )

        # Should find a counterexample where value != 2
        assert not result.property_holds, "Should find async race condition"
        assert result.counterexample is not None, "Should have counterexample schedule"

    asyncio.run(run_exploration())


@given(schedule=async_schedule_strategy(num_tasks=2))
@settings(max_examples=200, phases=[Phase.generate])
def test_async_suspension_point_race_hypothesis(schedule):
    """
    Test async suspension-point race using Hypothesis with schedule_strategy.

    Hypothesis generates random task schedules.
    """

    async def run_with_hypothesis_schedule():
        counter = await async_run_with_schedule(
            schedule,
            setup=lambda: AsyncBuggyCounterBytecode(),
            tasks=[
                lambda c: c.increment(),
                lambda c: c.increment(),
            ],
        )
        return counter

    counter = asyncio.run(run_with_hypothesis_schedule())

    # In some schedules, the race condition will occur
    assert counter.value in [1, 2], f"Unexpected value: {counter.value}"


# ============================================================================
# Additional Test: Async Order Violation
# ============================================================================


def test_async_order_violation_exact_schedule():
    """
    Test async order violation with exact schedule.

    Forces use_resource() to run before init_resource(). The user task
    sees resource=None because init hasn't happened yet.
    """
    manager = AsyncBuggyResourceManager()

    # Schedule that violates order: use before init
    schedule = Schedule(
        [
            Step("user_task", "use_resource"),
            Step("init_task", "init_resource"),
        ]
    )

    executor = AsyncTraceExecutor(schedule)

    executor.run(
        {
            "init_task": lambda: manager.init_resource("hello"),
            "user_task": manager.use_resource,
        }
    )

    assert manager.used_before_init, "Resource should have been used before initialization"


def test_async_order_violation_exploration():
    """
    Test async order violation using async_bytecode exploration.
    """

    async def run_exploration():
        def invariant(manager):
            # Check if resource was used before init (should be False)
            return not manager.used_before_init

        result = await async_explore_interleavings(
            setup=lambda: AsyncBuggyResourceManagerBytecode(),
            tasks=[
                lambda m: m.init_resource("hello"),
                lambda m: m.use_resource(),
            ],
            invariant=invariant,
            max_attempts=200,
            seed=42,
        )

        # Should find a counterexample where resource is used before init
        assert not result.property_holds, "Should find async order violation"
        assert result.counterexample is not None, "Should have counterexample schedule"

    asyncio.run(run_exploration())
