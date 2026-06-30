"""Tests for DPOR behavior with transaction.atomic() and property access.

Bug 1 (design flaw): DporScheduler._report_and_wait skips ALL scheduling
inside explicit SQL transactions, not just SQL operations. This means DPOR
cannot find races on non-SQL shared state mutated inside transaction.atomic()
blocks. The fix: remove the _in_transaction early return — SQL events are
already buffered by _tx_buffer and flushed atomically at COMMIT.

Bug 2: _process_opcode uses getattr(obj, attr) for LOAD_ATTR, which triggers
property descriptors. Properties that access the DB cause recursive lock deadlock.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from frontrun import explore

# ---------------------------------------------------------------------------
# Bug 1: DPOR must still schedule non-SQL opcodes inside transactions
# ---------------------------------------------------------------------------


class TestDporSchedulesInsideTransactions:
    """DPOR must detect races on non-SQL shared state inside transaction.atomic().

    The old behavior skipped ALL scheduling inside explicit transactions,
    making DPOR blind to races on Python objects modified within a
    transaction body. SQL atomicity is separately handled by _tx_buffer.
    """

    def test_dpor_finds_race_on_shared_state_inside_transaction(self) -> None:
        """DPOR must detect a lost-update race on a Python object that is
        read and written inside a transaction.atomic() block.

        Scenario: two threads each BEGIN, read a shared counter, write
        counter + 1, then COMMIT. The counter update is a classic
        lost-update race on the Python object — the SQL transaction
        doesn't protect it.

        With the old _in_transaction early return, DPOR never interleaves
        between the read and write of shared_state.value, so it cannot
        find this race.
        """

        class State:
            def __init__(self) -> None:
                self.value = 0

        uri = "file:dpor_tx_race?mode=memory&cache=shared"
        # Set up the shared DB
        setup_conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        setup_conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v INTEGER)")
        setup_conn.execute("DELETE FROM t")
        setup_conn.execute("INSERT INTO t VALUES (1, 0)")
        setup_conn.commit()

        def setup() -> State:
            # Reset DB state for each exploration
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            conn.execute("UPDATE t SET v = 0 WHERE id = 1")
            conn.commit()
            conn.close()
            return State()

        def thread_fn(state: State) -> None:
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
            cur = conn.cursor()
            cur.execute("BEGIN")
            # Read and write shared Python state INSIDE the transaction.
            # The SQL transaction does NOT protect this Python object.
            temp = state.value
            state.value = temp + 1
            cur.execute("COMMIT")
            conn.close()

        result = explore(
            setup=setup,
            workers=[thread_fn, thread_fn],
            invariant=lambda s: s.value == 2,
            detect_io=True,
            reproduce_on_failure=0,
            max_executions=50,
            preemption_bound=2,
        )

        assert not result.property_holds, (
            "DPOR should find the lost-update race on shared Python state "
            "inside transaction.atomic(), but it reported the property holds. "
            "This means DPOR is still skipping all scheduling inside explicit "
            "transactions, hiding non-SQL races."
        )

        setup_conn.close()

    def test_dpor_finds_race_on_shared_list_inside_transaction(self) -> None:
        """Variant: race on a shared list (append is not atomic with read)."""

        class State:
            def __init__(self) -> None:
                self.items: list[int] = []
                self.seen: list[int] = []

        uri = "file:dpor_tx_list_race?mode=memory&cache=shared"
        setup_conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        setup_conn.execute("CREATE TABLE IF NOT EXISTS t2 (id INTEGER PRIMARY KEY, v INTEGER)")
        setup_conn.execute("DELETE FROM t2")
        setup_conn.execute("INSERT INTO t2 VALUES (1, 0)")
        setup_conn.commit()

        def setup() -> State:
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            conn.execute("UPDATE t2 SET v = 0 WHERE id = 1")
            conn.commit()
            conn.close()
            return State()

        def thread_fn(state: State, thread_id: int) -> None:
            def _inner(s: State) -> None:
                conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
                cur = conn.cursor()
                cur.execute("BEGIN")
                # Read shared state inside transaction
                count = len(s.items)
                s.seen.append(count)
                s.items.append(thread_id)
                cur.execute("COMMIT")
                conn.close()

            return _inner

        result = explore(
            setup=setup,
            workers=[thread_fn(State(), 0), thread_fn(State(), 1)],  # type: ignore[arg-type]
            invariant=lambda s: len(s.seen) < 2 or s.seen[0] != s.seen[1],
            detect_io=True,
            reproduce_on_failure=0,
            max_executions=50,
            preemption_bound=2,
        )

        # Both threads could see count=0 if interleaved between len() and append()
        assert not result.property_holds, (
            "DPOR should find the race where both threads see len(items)==0 "
            "inside their transactions. If this fails, DPOR is still skipping "
            "scheduling inside explicit transactions."
        )

        setup_conn.close()


# ---------------------------------------------------------------------------
# Reproduction: schedules generated by DPOR should replay correctly
# ---------------------------------------------------------------------------


class TestDporTransactionReproduction:
    """After fixing DPOR to schedule inside transactions, reproduction
    should work because DPOR and OpcodeScheduler now count the same opcodes."""

    def test_counterexample_reproduces_with_transactions(self) -> None:
        """When DPOR finds a race inside a transaction, replay must succeed."""

        class State:
            def __init__(self) -> None:
                self.value = 0

        uri = "file:dpor_tx_repro?mode=memory&cache=shared"
        setup_conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        setup_conn.execute("CREATE TABLE IF NOT EXISTS t3 (id INTEGER PRIMARY KEY, v INTEGER)")
        setup_conn.execute("DELETE FROM t3")
        setup_conn.execute("INSERT INTO t3 VALUES (1, 0)")
        setup_conn.commit()

        def setup() -> State:
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            conn.execute("UPDATE t3 SET v = 0 WHERE id = 1")
            conn.commit()
            conn.close()
            return State()

        def thread_fn(state: State) -> None:
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
            cur = conn.cursor()
            cur.execute("BEGIN")
            temp = state.value
            state.value = temp + 1
            cur.execute("COMMIT")
            conn.close()

        result = explore(
            setup=setup,
            workers=[thread_fn, thread_fn],
            invariant=lambda s: s.value == 2,
            detect_io=True,
            reproduce_on_failure=5,
            max_executions=50,
            preemption_bound=2,
        )

        if not result.property_holds:
            assert result.reproduction_successes > 0, (
                f"DPOR found the race but reproduction failed: "
                f"{result.reproduction_successes}/{result.reproduction_attempts}. "
                f"Schedule length: {len(result.counterexample or [])}. "
                "OpcodeScheduler and DporScheduler likely disagree on step counts."
            )

        setup_conn.close()


# ---------------------------------------------------------------------------
# Bug 2: Property access in _process_opcode causes recursive lock deadlock
# ---------------------------------------------------------------------------


class TestPropertyAccessDeadlock:
    """_process_opcode must not trigger property descriptors via getattr().

    When LOAD_ATTR encounters a property whose getter does I/O (e.g. a
    Django model property that queries the DB), getattr() triggers the
    getter. If the getter tries to acquire the scheduler lock (via
    cursor.execute → _intercept_execute → report_and_wait), we get a
    recursive lock deadlock because _process_opcode is already called
    from within the locked section of report_and_wait.
    """

    def test_load_attr_uses_safe_getattr(self) -> None:
        """Verify _process_opcode uses _safe_getattr instead of bare
        getattr for LOAD_ATTR, avoiding property descriptor triggers."""
        import inspect

        from frontrun.dpor import _process_opcode

        source = inspect.getsource(_process_opcode)

        # Find the LOAD_ATTR section
        load_attr_idx = source.index("LOAD_ATTR")
        # Find the next elif/else to bound the section
        next_section = source.index("elif op ==", load_attr_idx + 1)
        load_attr_section = source[load_attr_idx:next_section]

        # The LOAD_ATTR section must use _safe_getattr (which bypasses
        # property descriptors) instead of bare getattr(obj, attr).
        assert "_safe_getattr" in load_attr_section, (
            "_process_opcode LOAD_ATTR handler does not use _safe_getattr. "
            "Bare getattr(obj, attr) triggers property descriptors, causing "
            "recursive lock deadlock when a property does DB access."
        )

    def test_property_not_triggered_during_dpor(self) -> None:
        """End-to-end: accessing an object with a property must not trigger
        the property getter during DPOR shadow-stack tracking."""
        call_count = [0]

        class ModelWithDBProperty:
            def __init__(self) -> None:
                self.name = "test"

            @property
            def expensive(self) -> str:
                call_count[0] += 1
                return "computed"

        obj = ModelWithDBProperty()

        def thread_fn(_state: Any) -> None:
            # Access the property — the opcode tracer will see LOAD_ATTR
            # for 'expensive' and push a value onto the shadow stack.
            # With the fix, _safe_getattr checks the instance dict and
            # class MRO without triggering the property getter.
            _ = obj.expensive  # noqa: F841

        call_count[0] = 0

        result = explore(
            setup=lambda: None,
            workers=[thread_fn],
            invariant=lambda _: True,
            max_executions=1,
            reproduce_on_failure=0,
        )

        # The getter is called once by the actual thread execution.
        # _process_opcode must NOT call it again via getattr().
        assert result.num_explored >= 1
        assert call_count[0] == 1, (
            f"Property getter was called {call_count[0]} times, expected exactly 1. "
            f"_process_opcode is likely triggering the property getter via getattr()."
        )

    def test_load_method_uses_safe_getattr(self) -> None:
        """LOAD_METHOD handler (Python 3.10) should also use _safe_getattr."""
        import inspect

        from frontrun.dpor import _process_opcode

        source = inspect.getsource(_process_opcode)

        # Find the LOAD_METHOD section
        if "LOAD_METHOD" not in source:
            pytest.skip("LOAD_METHOD not in _process_opcode (Python 3.11+)")

        load_method_idx = source.index('"LOAD_METHOD"')
        next_section = source.index("elif op ==", load_method_idx + 1)
        load_method_section = source[load_method_idx:next_section]

        assert "_safe_getattr" in load_method_section, (
            "_process_opcode LOAD_METHOD handler does not use _safe_getattr. "
            "Bare getattr(obj, attr) triggers property descriptors, causing "
            "recursive lock deadlock when a property does DB access."
        )
