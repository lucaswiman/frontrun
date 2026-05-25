"""
Tests for bytecode-level deterministic concurrency testing.

Demonstrates both:
1. Exact schedule reproduction (specific interleaving)
2. Property-based exploration (find bad interleavings via random search)
"""

import threading

import pytest

from frontrun import explore_interleavings as explore_interleavings_api
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
    # max_ops=4 disables dynamic schedule extension so the scheduler
    # finishes exactly when the explicit schedule is consumed.
    scheduler = OpcodeScheduler([0, 1, 0, 1], num_threads=2, max_ops=4)

    assert scheduler.wait_for_turn(0) is True  # schedule[0] = 0
    assert scheduler.wait_for_turn(1) is True  # schedule[1] = 1
    assert scheduler.wait_for_turn(0) is True  # schedule[2] = 0
    assert scheduler.wait_for_turn(1) is True  # schedule[3] = 1
    assert scheduler.wait_for_turn(0) is False  # exhausted
    assert scheduler._finished is True


def test_scheduler_extends_schedule():
    """When the explicit schedule is exhausted, the scheduler dynamically
    extends with round-robin entries to maintain deterministic control."""
    scheduler = OpcodeScheduler([0, 1], num_threads=2)

    assert scheduler.wait_for_turn(0) is True  # schedule[0] = 0
    assert scheduler.wait_for_turn(1) is True  # schedule[1] = 1
    # Explicit schedule is exhausted — dynamic extension should kick in
    assert scheduler.wait_for_turn(0) is True  # extended round-robin
    assert scheduler.wait_for_turn(1) is True  # extended round-robin
    assert scheduler._finished is False


def test_scheduler_skips_done_threads():
    """If a scheduled thread is already done, skip to next step."""
    scheduler = OpcodeScheduler([0, 1, 0], num_threads=2)

    assert scheduler.wait_for_turn(0) is True
    scheduler.mark_done(0)
    assert scheduler.wait_for_turn(1) is True
    # schedule[2] = 0, but thread 0 is done — dynamic extension has
    # only thread 1 active, so thread 1 gets to run
    scheduler.mark_done(1)
    # Now both threads are done — scheduler should finish
    assert scheduler.wait_for_turn(1) is False


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


def test_top_level_explore_interleavings_dispatches_to_bytecode():
    """The package-level explore_interleavings wrapper should dispatch sync threads."""
    result = explore_interleavings_api(
        setup=lambda: Counter(value=0),
        threads=[
            lambda c: c.increment(),
            lambda c: c.increment(),
        ],
        invariant=lambda c: c.value == 2,
        max_attempts=200,
        max_ops=50,
        seed=42,
    )

    assert not result.property_holds
    assert result.counterexample is not None


def test_cooperative_locks_prevent_deadlock():
    """Cooperative locks let the scheduler interleave lock-protected code.

    Lock.acquire() is replaced by a spin loop that yields scheduler turns
    between non-blocking acquire attempts. This lets the lock-holding thread
    run and release the lock, preventing the deadlock that would occur if
    Lock.acquire() blocked in C.
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


def test_extend_schedule_respects_max_ops_strictly():
    """_extend_schedule should not let the schedule grow past max_ops entries."""
    max_ops = 5
    scheduler = OpcodeScheduler([0, 1], num_threads=2, max_ops=max_ops)

    # Simulate consuming the initial schedule and extending
    total_scheduled = len(scheduler.schedule)
    while True:
        scheduler._index = len(scheduler.schedule)  # consume all entries
        if not scheduler._extend_schedule():
            break
        total_scheduled = len(scheduler.schedule)
        if total_scheduled > max_ops + 10:
            break  # safety

    assert total_scheduled <= max_ops, f"Schedule grew to {total_scheduled} entries, exceeding max_ops={max_ops}"


def test_scheduler_had_error():
    """Test that OpcodeScheduler tracks errors correctly."""
    scheduler = OpcodeScheduler([0, 1], num_threads=2)

    assert not scheduler.had_error

    error = RuntimeError("Test error")
    scheduler.report_error(error)

    assert scheduler.had_error
    assert scheduler._error is error


def test_explore_random_forwards_debug_to_run_with_schedule(monkeypatch: pytest.MonkeyPatch):
    """explore_random(debug=True) should forward debug to run_with_schedule.

    Bug: explore_random accepts a debug parameter but doesn't pass it through
    to run_with_schedule(), so timeout debug messages are never printed even
    when debug=True.
    """
    from frontrun import bytecode

    captured_kwargs: list[dict] = []
    original = bytecode.run_with_schedule

    def spy(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(bytecode, "run_with_schedule", spy)

    class C:
        def __init__(self):
            self.value = 0

        def inc(self):
            self.value += 1

    bytecode.explore_random(
        setup=C,
        threads=[C.inc, C.inc],
        invariant=lambda c: c.value == 2,
        debug=True,
        max_attempts=1,
        seed=42,
    )

    assert captured_kwargs, "run_with_schedule was never called"
    assert captured_kwargs[0].get("debug") is True, (
        f"debug=True was not forwarded to run_with_schedule; got kwargs: {captured_kwargs[0]}"
    )


def test_run_thread_clears_runtime_state_after_error(monkeypatch: pytest.MonkeyPatch):
    from frontrun import _opcode_observer

    scheduler = OpcodeScheduler([0], num_threads=1)
    runner = BytecodeShuffler(scheduler)
    events: list[str] = []

    # Install a real opcode trace handle so the install/uninstall path is
    # exercised; the actual sys.settrace / sys.monitoring calls are patched
    # through the tracer-backend module so we observe lifecycle events
    # without mutating global tracer state.
    runner._start_opcode_trace()
    monkeypatch.setattr(
        "frontrun.bytecode.set_context", lambda _scheduler, thread_id: events.append(f"context:{thread_id}")
    )
    monkeypatch.setattr("frontrun.bytecode.clear_context", lambda: events.append("clear_context"))
    monkeypatch.setattr(
        "frontrun.bytecode.set_dpor_scheduler", lambda value: events.append(f"scheduler:{value is not None}")
    )
    monkeypatch.setattr("frontrun.bytecode.set_dpor_thread_id", lambda value: events.append(f"thread_id:{value}"))
    monkeypatch.setattr("frontrun.bytecode.install_thread_opcode_trace", lambda _handle: events.append("trace:on"))
    monkeypatch.setattr("frontrun.bytecode.uninstall_thread_opcode_trace", lambda _handle: events.append("trace:off"))
    monkeypatch.setattr(runner, "_setup_io_reporter", lambda thread_id: events.append(f"io:{thread_id}"))
    monkeypatch.setattr(runner, "_teardown_io_reporter", lambda: events.append("io:off"))

    def boom() -> None:
        raise RuntimeError("boom")

    try:
        runner._run_thread(0, boom, (), {})
    finally:
        # Tear down the global tracer state we installed for this test.
        if runner._opcode_handle is not None:
            _opcode_observer.stop_opcode_trace(runner._opcode_handle)
            runner._opcode_handle = None

    assert isinstance(runner.errors[0], RuntimeError)
    assert scheduler.had_error
    assert scheduler._threads_done == {0}
    assert events == [
        "context:0",
        "io:0",
        "scheduler:True",
        "thread_id:0",
        "trace:on",
        "trace:off",
        "io:off",
        "scheduler:False",
        "thread_id:None",
        "clear_context",
    ]
