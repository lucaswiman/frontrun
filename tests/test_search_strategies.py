"""Tests for DPOR search strategy exploration ordering.

Measures how many executions each search strategy needs to find the first
invariant violation across a variety of race condition scenarios.
"""

from __future__ import annotations

import threading

import pytest

import frontrun

STRATEGIES = [
    "dfs",
    "bit-reversal",
    "bit-reversal:42",
    "round-robin",
    "round-robin:7",
    "stride",
    "stride:3",
    "conflict-first",
]


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


class Counter:
    def __init__(self) -> None:
        self.value = 0


class BankAccount:
    def __init__(self, balance: int) -> None:
        self.balance = balance


class Accounts:
    def __init__(self) -> None:
        self.a = BankAccount(100)
        self.b = BankAccount(100)


# ---------------------------------------------------------------------------
# Scenarios that MUST find a bug (property_holds=False)
# ---------------------------------------------------------------------------


def scenario_lost_update():
    """Two threads doing non-atomic read-modify-write on a shared counter."""

    def t(s: Counter) -> None:
        temp = s.value
        s.value = temp + 1

    return Counter, [t, t], lambda s: s.value == 2, "lost_update"


def scenario_three_thread_counter():
    """Three threads doing non-atomic increment."""

    def t(s: Counter) -> None:
        temp = s.value
        s.value = temp + 1

    return Counter, [t, t, t], lambda s: s.value == 3, "three_thread_counter"


def scenario_bank_transfer():
    """Two non-atomic transfers from account a to account b."""

    def transfer(s: Accounts) -> None:
        temp_a = s.a.balance
        temp_b = s.b.balance
        s.a.balance = temp_a - 10
        s.b.balance = temp_b + 10

    return Accounts, [transfer, transfer], lambda s: s.a.balance + s.b.balance == 200, "bank_transfer"


def scenario_write_order():
    """Two threads writing different values; invariant requires specific order."""

    class State:
        def __init__(self) -> None:
            self.x = 0

    def t0(s: State) -> None:
        s.x = 1

    def t1(s: State) -> None:
        s.x = 2

    # Fails when t0 writes last (x==1)
    return State, [t0, t1], lambda s: s.x != 1, "write_order"


def scenario_dining_philosophers_3():
    """Three dining philosophers - deadlock detection."""
    n = 3

    class State:
        def __init__(self) -> None:
            self.forks = [threading.Lock() for _ in range(n)]

    def make_phil(i: int):  # noqa: ANN202
        def phil(s: State) -> None:
            left = i
            right = (i + 1) % n
            with s.forks[left]:
                with s.forks[right]:
                    pass

        return phil

    return State, [make_phil(i) for i in range(n)], lambda s: True, "dining_philosophers_3"


def scenario_augmented_assign():
    """Two threads using += on shared counter (non-atomic)."""

    def t(s: Counter) -> None:
        s.value += 1

    return Counter, [t, t], lambda s: s.value == 2, "augmented_assign"


def scenario_four_thread_counter():
    """Four threads doing non-atomic increment — many more traces."""

    def t(s: Counter) -> None:
        temp = s.value
        s.value = temp + 1

    return Counter, [t, t, t, t], lambda s: s.value == 4, "four_thread_counter"


def scenario_dining_philosophers_4():
    """Four dining philosophers - deadlock detection."""
    n = 4

    class State:
        def __init__(self) -> None:
            self.forks = [threading.Lock() for _ in range(n)]

    def make_phil(i: int):  # noqa: ANN202
        def phil(s: State) -> None:
            left = i
            right = (i + 1) % n
            with s.forks[left]:
                with s.forks[right]:
                    pass

        return phil

    return State, [make_phil(i) for i in range(n)], lambda s: True, "dining_philosophers_4"


def scenario_three_writers():
    """Three threads writing different values to the same field.

    Invariant: x must be 3 (last writer wins for thread 2).
    Fails when thread 2 doesn't write last.
    """

    class State:
        def __init__(self) -> None:
            self.x = 0

    def t0(s: State) -> None:
        s.x = 1

    def t1(s: State) -> None:
        s.x = 2

    def t2(s: State) -> None:
        s.x = 3

    return State, [t0, t1, t2], lambda s: s.x == 3, "three_writers"


def scenario_four_writers():
    """Four threads each writing a distinct value.

    Invariant: x must be 4. Fails unless thread 3 writes last.
    """

    class State:
        def __init__(self) -> None:
            self.x = 0

    def make_writer(val: int):  # noqa: ANN202
        def writer(s: State) -> None:
            s.x = val

        return writer

    threads = [make_writer(i + 1) for i in range(4)]
    return State, threads, lambda s: s.x == 4, "four_writers"


def scenario_three_thread_two_vars():
    """Three threads racing on two shared variables — richer conflict space."""

    class State:
        def __init__(self) -> None:
            self.x = 0
            self.y = 0

    def t0(s: State) -> None:
        s.x += 1
        s.y += 1

    def t1(s: State) -> None:
        s.x += 1
        s.y += 1

    def t2(s: State) -> None:
        s.x += 1
        s.y += 1

    return State, [t0, t1, t2], lambda s: s.x == 3 and s.y == 3, "three_thread_two_vars"


SCENARIOS = [
    scenario_lost_update,
    scenario_three_thread_counter,
    scenario_bank_transfer,
    scenario_write_order,
    scenario_dining_philosophers_3,
    scenario_augmented_assign,
    scenario_four_thread_counter,
    scenario_dining_philosophers_4,
    scenario_three_writers,
    scenario_four_writers,
    scenario_three_thread_two_vars,
]


# ---------------------------------------------------------------------------
# TDD: test that the search parameter is accepted and all strategies find bugs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_search_strategy_accepted(strategy: str) -> None:
    """Each search strategy string is accepted without error."""
    setup, threads, invariant, _ = scenario_lost_update()
    result = frontrun.explore(
        setup=setup,
        workers=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search=strategy,
        detect_io=False,
    )
    # Must find the bug regardless of strategy
    assert not result.property_holds, f"strategy={strategy} should find lost update bug"
    assert result.num_explored >= 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_strategies_find_bank_transfer_bug(strategy: str) -> None:
    """All strategies find the bank transfer race."""
    setup, threads, invariant, _ = scenario_bank_transfer()
    result = frontrun.explore(
        setup=setup,
        workers=threads,
        invariant=invariant,
        max_executions=500,
        stop_on_first=True,
        search=strategy,
        detect_io=False,
    )
    assert not result.property_holds, f"strategy={strategy} should find bank transfer bug"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_strategies_find_three_thread_bug(strategy: str) -> None:
    """All strategies find the three-thread counter race."""
    setup, threads, invariant, _ = scenario_three_thread_counter()
    result = frontrun.explore(
        setup=setup,
        workers=threads,
        invariant=invariant,
        max_executions=1000,
        stop_on_first=True,
        search=strategy,
        detect_io=False,
    )
    assert not result.property_holds, f"strategy={strategy} should find three-thread counter bug"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_strategies_find_dining_deadlock(strategy: str) -> None:
    """All strategies find the dining philosophers deadlock."""
    setup, threads, invariant, _ = scenario_dining_philosophers_3()
    result = frontrun.explore(
        setup=setup,
        workers=threads,
        invariant=invariant,
        max_executions=5000,
        preemption_bound=2,
        stop_on_first=True,
        search=strategy,
        detect_io=False,
        deadlock_timeout=2.0,
    )
    assert not result.property_holds, f"strategy={strategy} should find dining philosophers deadlock"


def test_invalid_strategy_rejected() -> None:
    """Unknown strategy strings raise ValueError."""
    setup, threads, invariant, _ = scenario_lost_update()
    with pytest.raises(Exception, match="unknown search strategy"):
        frontrun.explore(
            setup=setup,
            workers=threads,
            invariant=invariant,
            max_executions=10,
            search="bogus",
            detect_io=False,
        )


# ---------------------------------------------------------------------------
# Benchmark: explorations to first failure across all strategies x scenarios
# ---------------------------------------------------------------------------


def test_search_strategy_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    """Benchmark: how many executions does each strategy need to find the first bug?

    This test always passes but prints a comparison table.
    """
    results: dict[str, dict[str, int | str]] = {}

    for scenario_fn in SCENARIOS:
        setup_fn, threads, invariant, name = scenario_fn()
        is_deadlock = name == "dining_philosophers_3"

        for strategy in STRATEGIES:
            kwargs: dict[str, object] = {
                "setup": setup_fn,
                "workers": threads,
                "invariant": invariant,
                "max_executions": 5000,
                "stop_on_first": True,
                "search": strategy,
                "detect_io": False,
            }
            if is_deadlock:
                kwargs["deadlock_timeout"] = 2.0
                kwargs["preemption_bound"] = 2

            result = frontrun.explore(**kwargs)  # type: ignore[arg-type]

            key = f"{name}/{strategy}"
            if result.property_holds:
                results[key] = {"explorations": result.num_explored, "found": "NO"}
            else:
                results[key] = {"explorations": result.num_explored, "found": "YES"}

    # Print table
    with capsys.disabled():
        # Collect unique scenario names and strategy names
        scenario_names = [fn()[3] for fn in SCENARIOS]

        # Header
        col_width = max(len(s) for s in STRATEGIES) + 2
        header = f"{'scenario':<30}" + "".join(f"{s:>{col_width}}" for s in STRATEGIES)
        print()
        print("=" * len(header))
        print("EXPLORATIONS TO FIRST FAILURE (lower is better)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for scenario_name in scenario_names:
            row = f"{scenario_name:<30}"
            for strategy in STRATEGIES:
                key = f"{scenario_name}/{strategy}"
                info = results[key]
                val = str(info["explorations"])
                if info["found"] == "NO":
                    val = f"{val}*"
                row += f"{val:>{col_width}}"
            print(row)

        print("-" * len(header))
        print("* = bug not found within max_executions")
        print("=" * len(header))
