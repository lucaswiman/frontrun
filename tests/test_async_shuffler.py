"""
Tests for await-point-level deterministic async concurrency testing.

Demonstrates both:
1. Exact schedule reproduction (specific interleaving)
2. Property-based exploration (find bad interleavings via random search)
"""

import asyncio

import pytest

import frontrun
from frontrun.async_shuffler import (
    AsyncShuffler,
    AwaitScheduler,
    _task_id_var,
    await_point,
    controlled_interleaving,
    explore_async_random,
    run_with_schedule,
)

# ---------------------------------------------------------------------------
# Test fixtures: classes with race conditions
# ---------------------------------------------------------------------------


class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    async def transfer(self, amount):
        current = self.balance
        await await_point()
        self.balance = current + amount


class Counter:
    def __init__(self, value=0):
        self.value = value

    async def increment(self):
        temp = self.value
        await await_point()
        self.value = temp + 1


class SafeCounter:
    """Counter protected by asyncio.Lock — should be race-free."""

    def __init__(self, value=0):
        self.value = value
        self._lock = asyncio.Lock()

    async def increment(self):
        async with self._lock:
            temp = self.value
            await await_point()
            self.value = temp + 1


class NaturalCounter:
    """Counter that uses a natural await instead of explicit await_point()."""

    def __init__(self, value=0):
        self.value = value

    async def increment(self):
        temp = self.value
        await asyncio.sleep(0)
        self.value = temp + 1


# ---------------------------------------------------------------------------
# Unit tests: AwaitScheduler
# ---------------------------------------------------------------------------


def test_scheduler_basic_round_robin():
    """Two tasks alternate await-point execution."""

    async def _test():
        scheduler = AwaitScheduler([0, 1, 0, 1], num_tasks=2)

        # Set up context vars for task 0
        _task_id_var.set(0)
        await scheduler.pause(0)  # schedule[0] = 0
        assert scheduler._index == 1

        # Set up context vars for task 1
        _task_id_var.set(1)
        await scheduler.pause(1)  # schedule[1] = 1
        assert scheduler._index == 2

        # Back to task 0
        _task_id_var.set(0)
        await scheduler.pause(0)  # schedule[2] = 0
        assert scheduler._index == 3

        # Back to task 1
        _task_id_var.set(1)
        await scheduler.pause(1)  # schedule[3] = 1
        assert scheduler._index == 4

        # Schedule exhausted
        _task_id_var.set(0)
        await scheduler.pause(0)
        assert scheduler._finished is True

    asyncio.run(_test())


def test_scheduler_skips_done_tasks():
    """If a scheduled task is already done, skip to next step."""

    async def _test():
        scheduler = AwaitScheduler([0, 1, 0], num_tasks=2)

        # Task 0 runs
        _task_id_var.set(0)
        await scheduler.pause(0)
        assert scheduler._index == 1

        # Mark task 0 as done
        await scheduler._mark_done(0)

        # Task 1 runs
        _task_id_var.set(1)
        await scheduler.pause(1)
        assert scheduler._index == 2

        # schedule[2] = 0, but task 0 is done — should skip past it
        await scheduler.pause(1)  # Still task 1
        assert scheduler._finished is True

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# Integration tests: AsyncShuffler with exact schedules
# ---------------------------------------------------------------------------


def test_controlled_interleaving_basic():
    """Run two simple async functions under a controlled schedule."""

    async def _test():
        results = []

        async def func_a():
            results.append("a")
            await await_point()

        async def func_b():
            results.append("b")
            await await_point()

        # Schedule that allows both functions to complete
        schedule = [0] * 10 + [1] * 10
        async with controlled_interleaving(schedule, num_tasks=2) as runner:
            await runner.run([func_a, func_b])

        assert "a" in results
        assert "b" in results

    asyncio.run(_test())


def test_bank_account_sequential_correct():
    """Sequential schedule ensures no race."""

    async def _test():
        account = BankAccount(balance=100)

        async def t1():
            await account.transfer(50)

        async def t2():
            await account.transfer(50)

        # Run task 0 fully, then task 1 fully
        schedule = [0] * 20 + [1] * 20
        async with controlled_interleaving(schedule, num_tasks=2) as runner:
            await runner.run([t1, t2])

        assert account.balance == 200

    asyncio.run(_test())


def test_bank_account_race_reproduced():
    """Reproduce a lost-update race by interleaving reads before writes.

    With alternation, we can force both tasks to read the same
    balance before either writes.
    """

    async def _test():
        found_race = False

        for _ in range(5):
            account = BankAccount(balance=100)

            async def t1():
                await account.transfer(50)

            async def t2():
                await account.transfer(50)

            # Alternate rapidly to maximize chance of interleaving within transfer()
            schedule = [0, 1] * 15
            scheduler = AwaitScheduler(schedule, num_tasks=2)
            runner = AsyncShuffler(scheduler)
            await runner.run([t1, t2])

            if account.balance != 200:
                found_race = True
                break

        if found_race:
            assert account.balance == 150

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# Property-based tests: explore_async_random
# ---------------------------------------------------------------------------


def test_explore_finds_counter_race():
    """Random exploration should find an interleaving that breaks the counter."""

    async def _test():
        result = await explore_async_random(
            setup=lambda: Counter(value=0),
            tasks=[
                lambda c: c.increment(),
                lambda c: c.increment(),
            ],
            invariant=lambda c: c.value == 2,
            max_attempts=200,
            max_ops=50,
            seed=42,
        )

        assert not result.property_holds, (
            f"Expected to find a race condition, but invariant held across {result.num_explored} interleavings"
        )
        assert result.counterexample is not None

    asyncio.run(_test())


def test_explore_reports_task_crash_instead_of_propagating():
    """A task raising under some interleaving is a counterexample, not a fatal
    crash.  The random shuffler must surface it as ``property_holds=False``
    (mirroring the DPOR path's "Task crash ..." handling) rather than letting
    the exception escape and abort the whole exploration.
    """

    async def _test():
        async def raising_task(state):
            await await_point()
            raise ValueError("boom in task body")

        result = await explore_async_random(
            setup=lambda: {"n": 0},
            tasks=[raising_task, raising_task],
            invariant=lambda s: True,
            max_attempts=5,
            max_ops=20,
            seed=1,
        )

        assert not result.property_holds
        assert result.counterexample is not None
        assert result.explanation is not None
        assert "ValueError" in result.explanation
        assert "boom in task body" in result.explanation

    asyncio.run(_test())


def test_explore_bank_account_race():
    """Random exploration should find a lost-update in BankAccount."""

    async def _test():
        result = await explore_async_random(
            setup=lambda: BankAccount(balance=100),
            tasks=[
                lambda acc: acc.transfer(50),
                lambda acc: acc.transfer(50),
            ],
            invariant=lambda acc: acc.balance == 200,
            max_attempts=200,
            max_ops=50,
            seed=42,
        )

        assert not result.property_holds, (
            f"Expected to find lost-update race, but invariant held across {result.num_explored} interleavings"
        )

    asyncio.run(_test())


def test_explore_finds_natural_await_race():
    """Random exploration should treat natural awaits as scheduling points."""

    async def _test():
        result = await explore_async_random(
            setup=lambda: NaturalCounter(value=0),
            tasks=[
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

    asyncio.run(_test())


def test_top_level_explore_async_random_dispatches_to_async_shuffler():
    """The package-level explore_async_random wrapper should dispatch async tasks."""

    async def _test():
        result = await frontrun.explore_async_random(
            setup=lambda: NaturalCounter(value=0),
            tasks=[
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

    asyncio.run(_test())


def test_explore_three_tasks():
    """Three tasks incrementing a counter — exploration finds the race."""

    async def _test():
        result = await explore_async_random(
            setup=lambda: Counter(value=0),
            tasks=[
                lambda c: c.increment(),
                lambda c: c.increment(),
                lambda c: c.increment(),
            ],
            invariant=lambda c: c.value == 3,
            max_attempts=200,
            max_ops=50,
            seed=42,
        )

        assert not result.property_holds

    asyncio.run(_test())


def test_run_with_schedule_returns_state():
    """run_with_schedule returns the state object for inspection."""

    async def _test():
        state = await run_with_schedule(
            schedule=[0] * 20 + [1] * 20,
            setup=lambda: Counter(value=0),
            tasks=[
                lambda c: c.increment(),
                lambda c: c.increment(),
            ],
        )

        assert state.value == 2

    asyncio.run(_test())


def test_explore_async_random_unwinds_patch_stack_in_reverse_order(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []

    def patch_sql() -> None:
        events.append("patch_sql")

    def unpatch_sql() -> None:
        events.append("unpatch_sql")

    def patch_sleep() -> None:
        events.append("patch_sleep")

    def unpatch_sleep() -> None:
        events.append("unpatch_sleep")

    monkeypatch.setattr("frontrun.async_shuffler._sql_async_available", True)
    monkeypatch.setattr("frontrun.async_shuffler.patch_sql_async", patch_sql)
    monkeypatch.setattr("frontrun.async_shuffler.unpatch_sql_async", unpatch_sql)
    monkeypatch.setattr("frontrun.async_dpor._patch_asyncio_sleep", patch_sleep)
    monkeypatch.setattr("frontrun.async_dpor._unpatch_asyncio_sleep", unpatch_sleep)

    async def _test() -> None:
        with pytest.raises(RuntimeError, match="setup boom"):
            await explore_async_random(
                setup=lambda: (_ for _ in ()).throw(RuntimeError("setup boom")),
                tasks=[lambda _state: asyncio.sleep(0)],
                invariant=lambda _state: True,
                max_attempts=1,
                detect_sql=True,
                patch_sleep=True,
            )

    asyncio.run(_test())

    assert events == ["patch_sql", "patch_sleep", "unpatch_sleep", "unpatch_sql"]


def test_explore_with_seed_is_reproducible():
    """Same seed should produce the same exploration outcome."""

    async def _test():
        kwargs = dict(
            setup=lambda: Counter(value=0),
            tasks=[
                lambda c: c.increment(),
                lambda c: c.increment(),
            ],
            invariant=lambda c: c.value == 2,
            max_attempts=50,
            max_ops=50,
        )

        r1 = await explore_async_random(**kwargs, seed=123)
        r2 = await explore_async_random(**kwargs, seed=123)

        assert r1.property_holds == r2.property_holds
        assert r1.num_explored == r2.num_explored

    asyncio.run(_test())


def test_scheduler_had_error():
    """Test that AwaitScheduler tracks errors correctly."""

    async def _test():
        scheduler = AwaitScheduler([0, 1], num_tasks=2)

        assert not scheduler.had_error

        error = RuntimeError("Test error")
        await scheduler._report_error(error)

        assert scheduler.had_error
        assert scheduler._error is error

    asyncio.run(_test())
