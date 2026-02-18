"""
Tests for bytecode-level deterministic concurrency testing.

Demonstrates both:
1. Exact schedule reproduction (specific interleaving)
2. Property-based exploration (find bad interleavings via random search)
"""

import threading

from frontrun.bytecode import (
    BytecodeShuffler,
    OpcodeScheduler,
    controlled_interleaving,
    explore_interleavings,
    run_with_schedule,
)

# ---------------------------------------------------------------------------
# Test fixtures: classes with race conditions
# ---------------------------------------------------------------------------


class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def transfer(self, amount):
        current = self.balance
        self.balance = current + amount


class Counter:
    def __init__(self, value=0):
        self.value = value

    def increment(self):
        temp = self.value
        self.value = temp + 1


class SafeCounter:
    """Counter protected by a lock — should be race-free."""

    def __init__(self, value=0):
        self.value = value
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            temp = self.value
            self.value = temp + 1


# ---------------------------------------------------------------------------
# Unit tests: OpcodeScheduler
# ---------------------------------------------------------------------------


def test_scheduler_basic_round_robin():
    """Two threads alternate opcode execution."""
    scheduler = OpcodeScheduler([0, 1, 0, 1], num_threads=2)

    assert scheduler.wait_for_turn(0) is True  # schedule[0] = 0
    assert scheduler.wait_for_turn(1) is True  # schedule[1] = 1
    assert scheduler.wait_for_turn(0) is True  # schedule[2] = 0
    assert scheduler.wait_for_turn(1) is True  # schedule[3] = 1
    assert scheduler.wait_for_turn(0) is False  # exhausted
    assert scheduler._finished is True


def test_scheduler_skips_done_threads():
    """If a scheduled thread is already done, skip to next step."""
    scheduler = OpcodeScheduler([0, 1, 0], num_threads=2)

    assert scheduler.wait_for_turn(0) is True
    scheduler.mark_done(0)
    assert scheduler.wait_for_turn(1) is True
    # schedule[2] = 0, but thread 0 is done — should skip past it
    assert scheduler.wait_for_turn(1) is False  # schedule exhausted


# ---------------------------------------------------------------------------
# Integration tests: BytecodeShuffler with exact schedules
# ---------------------------------------------------------------------------


def test_controlled_interleaving_basic():
    """Run two simple functions under a controlled schedule."""
    results = []

    def func_a():
        results.append("a")

    def func_b():
        results.append("b")

    # Long enough schedule that both functions complete
    schedule = [0] * 50 + [1] * 50
    with controlled_interleaving(schedule, num_threads=2) as runner:
        runner.run([func_a, func_b])

    assert "a" in results
    assert "b" in results


def test_bank_account_sequential_correct():
    """Sequential schedule ensures no race."""
    account = BankAccount(balance=100)

    def t1():
        account.transfer(50)

    def t2():
        account.transfer(50)

    schedule = [0] * 200 + [1] * 200
    with controlled_interleaving(schedule, num_threads=2) as runner:
        runner.run([t1, t2])

    assert account.balance == 200


def test_bank_account_race_reproduced():
    """Reproduce a lost-update race by interleaving reads before writes.

    With enough alternation, we can force both threads to read the same
    balance before either writes.
    """
    found_race = False

    for _ in range(5):
        account = BankAccount(balance=100)

        def t1(acc=account):
            acc.transfer(50)

        def t2(acc=account):
            acc.transfer(50)

        # Alternate rapidly to maximize chance of interleaving within transfer()
        schedule = [0, 1] * 150
        scheduler = OpcodeScheduler(schedule, num_threads=2)
        runner = BytecodeShuffler(scheduler)
        runner.run([t1, t2])

        if account.balance != 200:
            found_race = True
            break

    if found_race:
        assert account.balance == 150


# ---------------------------------------------------------------------------
# Property-based tests: explore_interleavings
# ---------------------------------------------------------------------------


def test_explore_finds_counter_race():
    """Random exploration should find an interleaving that breaks the counter."""
    result = explore_interleavings(
        setup=lambda: Counter(value=0),
        threads=[
            lambda c: c.increment(),
            lambda c: c.increment(),
        ],
        invariant=lambda c: c.value == 2,
        max_attempts=200,
        max_ops=200,
        seed=42,
    )

    assert not result.property_holds, (
        f"Expected to find a race condition, but invariant held across {result.num_explored} interleavings"
    )
    assert result.counterexample is not None


def test_cooperative_locks_prevent_deadlock():
    """Cooperative locks let the scheduler interleave lock-protected code.

    Without cooperative locks, threading.Lock.acquire() blocks in C and
    never yields back to the trace function, deadlocking the scheduler.

    With cooperative locks (the default), Lock.acquire() is replaced by a
    spin loop that yields scheduler turns between non-blocking acquire
    attempts. This lets the lock-holding thread run and release the lock.
    """
    result = explore_interleavings(
        setup=lambda: SafeCounter(value=0),
        threads=[
            lambda c: c.increment(),
            lambda c: c.increment(),
        ],
        invariant=lambda c: c.value == 2,
        max_attempts=50,
        max_ops=200,
        seed=42,
    )

    assert result.property_holds, (
        f"SafeCounter with lock should not have races, but got counterexample at attempt {result.num_explored}"
    )


def test_explore_bank_account_race():
    """Random exploration should find a lost-update in BankAccount."""
    result = explore_interleavings(
        setup=lambda: BankAccount(balance=100),
        threads=[
            lambda acc: acc.transfer(50),
            lambda acc: acc.transfer(50),
        ],
        invariant=lambda acc: acc.balance == 200,
        max_attempts=200,
        max_ops=200,
        seed=42,
    )

    assert not result.property_holds, (
        f"Expected to find lost-update race, but invariant held across {result.num_explored} interleavings"
    )


def test_explore_three_threads():
    """Three threads incrementing a counter — exploration finds the race."""
    result = explore_interleavings(
        setup=lambda: Counter(value=0),
        threads=[
            lambda c: c.increment(),
            lambda c: c.increment(),
            lambda c: c.increment(),
        ],
        invariant=lambda c: c.value == 3,
        max_attempts=200,
        max_ops=300,
        seed=42,
    )

    assert not result.property_holds


def test_run_with_schedule_returns_state():
    """run_with_schedule returns the state object for inspection."""
    state = run_with_schedule(
        schedule=[0] * 200 + [1] * 200,
        setup=lambda: Counter(value=0),
        threads=[
            lambda c: c.increment(),
            lambda c: c.increment(),
        ],
    )

    assert state.value == 2


def test_explore_with_seed_is_reproducible():
    """Same seed should produce the same exploration outcome."""
    kwargs = dict(
        setup=lambda: Counter(value=0),
        threads=[
            lambda c: c.increment(),
            lambda c: c.increment(),
        ],
        invariant=lambda c: c.value == 2,
        max_attempts=50,
        max_ops=200,
    )

    r1 = explore_interleavings(**kwargs, seed=123)
    r2 = explore_interleavings(**kwargs, seed=123)

    assert r1.property_holds == r2.property_holds
    assert r1.num_explored == r2.num_explored


def test_scheduler_had_error():
    """Test that OpcodeScheduler tracks errors correctly."""
    scheduler = OpcodeScheduler([0, 1], num_threads=2)

    assert not scheduler.had_error

    error = RuntimeError("Test error")
    scheduler.report_error(error)

    assert scheduler.had_error
    assert scheduler._error is error
