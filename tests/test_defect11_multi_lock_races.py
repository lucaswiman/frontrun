"""Defect #11 (FIXED): DPOR detects races between two separate lock objects.

Problem statement
-----------------

A common real-world race pattern involves a **compound operation** that
updates two separately-locked values in sequence, where a concurrent reader
observes the values *between* the two updates and sees an inconsistent
snapshot:

    # Thread A (writer — "observe"):
    with count_lock:                 # Lock 1
        count._value += 1
    # << race window: count incremented, sum not yet >>
    with sum_lock:                   # Lock 2
        sum._value += amount

    # Thread B (reader — "collect"):
    with count_lock:                 # Lock 1
        c = count._value             # sees count incremented
    with sum_lock:                   # Lock 2
        s = sum._value               # but sum not yet incremented → c > s

This is the pattern used by ``prometheus_client.Summary.observe()``.

How DPOR detects this
---------------------

The fix has two parts:

1. **Lock-based happens-before fixed**: Lock release now stores the
   releasing thread's ``dpor_vv`` (previously stored ``causality``, which
   was never incremented and always zero).  This ensures data accesses
   inside the same critical section are correctly ordered by HB.

2. **Lock operations create backtrack points**: Lock acquire is reported
   as a Write I/O access to a virtual lock object, using ``io_vv`` (which
   doesn't include lock-based HB) and first-access semantics.  This makes
   lock acquires on the same lock by different threads appear concurrent,
   creating backtrack points at lock boundaries.

Together, these changes let DPOR explore the ordering where Thread B runs
its entire count_lock + sum_lock sequence *between* Thread A's two
critical sections — finding the count=1, sum=0 invariant violation.

Running
-------
::

    make test-3.14 PYTEST_ARGS="-v -k test_defect11_multi"
"""

from __future__ import annotations

import threading

from frontrun import explore


class TestMultiLockRaces:
    """Defect #11 (FIXED): DPOR detects races that span two separately-locked objects.

    Lock acquire is now reported as an I/O access, creating backtrack points
    at lock boundaries.  Combined with fixed lock-based happens-before, DPOR
    explores the interleaving where a reader runs between a writer's two
    critical sections.
    """

    def test_prometheus_summary_observe_pattern(self) -> None:
        """Two separately-locked counters updated non-atomically.

        Mimics prometheus_client's Summary.observe():
          - Thread A increments count (lock 1), then increments sum (lock 2)
          - Thread B reads count (lock 1), then reads sum (lock 2)

        The race: Thread B reads count AFTER Thread A increments it, but
        reads sum BEFORE Thread A increments it → sees count > sum.

        DPOR should report property_holds=False (the race is real and
        confirmed by stress testing), but it reports True because it
        never explores the interleaving where Thread B runs between
        Thread A's two locked sections.
        """

        class State:
            def __init__(self) -> None:
                self.count_lock = threading.Lock()
                self.sum_lock = threading.Lock()
                self.count = 0
                self.total = 0
                self.observed_count: int | None = None
                self.observed_total: int | None = None

        def writer(state: State) -> None:
            # Increment count and sum in two separately-locked sections.
            # The race window is between these two blocks.
            with state.count_lock:
                state.count += 1
            # << Thread B can interleave here and see count=1, total=0 >>
            with state.sum_lock:
                state.total += 10

        def reader(state: State) -> None:
            # Read both values — if we interleave in the gap, we see
            # count=1 but total=0.
            with state.count_lock:
                state.observed_count = state.count
            with state.sum_lock:
                state.observed_total = state.total

        def invariant(state: State) -> bool:
            # If reader ran, count and total should be consistent:
            # either both 0 (reader ran first) or count=1 and total=10
            # (writer finished first).  count=1 and total=0 means the
            # reader interleaved between the two locked sections.
            if state.observed_count is None:
                return True  # reader didn't run yet
            if state.observed_count == 0:
                return True  # reader ran before writer
            # count > 0: writer's first lock section ran.
            # total should also be updated.
            return state.observed_total == state.observed_count * 10

        result = explore(
            setup=State,
            workers=[writer, reader],
            invariant=invariant,
            detect_io=False,
            max_executions=200,
            preemption_bound=None,  # Full exploration — no bound
            deadlock_timeout=10.0,
        )

        assert not result.property_holds, (
            f"DPOR should detect the multi-lock race but missed it.  "
            f"Explored {result.num_explored} interleavings.  "
            f"The race exists: a reader can see count=1, total=0 when it "
            f"interleaves between the writer's two locked sections."
        )

    def test_transfer_between_two_locked_accounts(self) -> None:
        """Transfer between two separately-locked accounts is non-atomic.

        A common pattern in financial systems: debit one account (lock 1),
        then credit another (lock 2).  A concurrent reader can observe
        the intermediate state where money has been debited but not yet
        credited — the total across both accounts is temporarily wrong.
        """

        class State:
            def __init__(self) -> None:
                self.lock_a = threading.Lock()
                self.lock_b = threading.Lock()
                self.balance_a = 100
                self.balance_b = 100
                self.observed_total: int | None = None

        def transfer(state: State) -> None:
            # Debit A, then credit B — NOT atomic across both accounts.
            with state.lock_a:
                state.balance_a -= 50
            # << race window: A debited, B not yet credited >>
            with state.lock_b:
                state.balance_b += 50

        def auditor(state: State) -> None:
            # Read both balances — should always sum to 200.
            with state.lock_a:
                a = state.balance_a
            with state.lock_b:
                b = state.balance_b
            state.observed_total = a + b

        def invariant(state: State) -> bool:
            if state.observed_total is None:
                return True
            # Total should always be 200 (conservation of money).
            return state.observed_total == 200

        result = explore(
            setup=State,
            workers=[transfer, auditor],
            invariant=invariant,
            detect_io=False,
            max_executions=200,
            preemption_bound=None,
            deadlock_timeout=10.0,
        )

        assert not result.property_holds, (
            f"DPOR should detect the transfer atomicity violation but missed "
            f"it.  Explored {result.num_explored} interleavings.  "
            f"The auditor can observe balance_a=50, balance_b=100 (total=150) "
            f"when it interleaves between the debit and credit."
        )
