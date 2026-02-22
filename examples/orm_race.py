"""
SQLAlchemy ORM lost-update race condition — real Postgres
=========================================================

SQLAlchemy's ORM computes new column values in Python.  When two
concurrent sessions read the same row, compute a new value from the
stale read, and write it back, the second commit silently overwrites
the first — a classic **lost update**.

Scenario
--------
A ``users`` table with a ``login_count`` column (initially 0).
Two concurrent login-event handlers each do::

    user = session.get(User, 1)          # SELECT
    user.login_count = user.login_count + 1   # Python-side increment
    session.commit()                     # UPDATE … SET login_count=<n>

Race window::

    Handler A:  SELECT → login_count = 0
    Handler B:  SELECT → login_count = 0
    Handler A:  UPDATE SET login_count = 1, COMMIT   ← correct
    Handler B:  UPDATE SET login_count = 1, COMMIT   ← stale! should be 2

Fix: use a SQL expression ``User.login_count + 1`` so the increment
happens server-side, or ``SELECT … FOR UPDATE`` to serialize access.

This example demonstrates detection using frontrun:

1. **Trace markers** (deterministic) — exact schedule forces the race
2. **Bytecode exploration** (automatic) — random opcode-level schedules
3. **Naive threading** — shows intermittent failure rate

Requirements::

    make build-examples-3.14t   # or build-examples-3.10
    createdb frontrun_test      # or: sudo -u postgres createdb frontrun_test

Running::

    .venv-3.14t/bin/python examples/orm_race.py
"""

from __future__ import annotations

import random
import threading
import time

from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from frontrun.bytecode import explore_interleavings
from frontrun.common import Schedule, Step
from frontrun.trace_markers import TraceExecutor

_DB_NAME = "frontrun_test"
_SEP = "=" * 70


# ---------------------------------------------------------------------------
# SQLAlchemy model
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    login_count: Mapped[int] = mapped_column(default=0)


_engine = create_engine(f"postgresql:///{_DB_NAME}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_table() -> None:
    """(Re-)create the users table and seed one row."""
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)
    with Session(_engine) as session:
        session.add(User(id=1, name="Alice", login_count=0))
        session.commit()


def _reset_row() -> None:
    """Reset login_count to 0 between attempts."""
    with Session(_engine) as session:
        user = session.get(User, 1)
        assert user is not None
        user.login_count = 0
        session.commit()


def _read_count() -> int:
    """Read the current login_count."""
    with Session(_engine) as session:
        user = session.get(User, 1)
        assert user is not None
        return user.login_count


# ============================================================================
# Demo 1: Exact reproduction with trace markers
# ============================================================================
#
# Each handler opens its own SQLAlchemy session (separate Postgres
# connection).  The # frontrun: markers on session.get() and
# session.commit() let TraceExecutor control when each SQL statement
# fires.
#
# Schedule:
#   1. Handler A: SELECT (reads login_count=0)
#   2. Handler B: SELECT (reads login_count=0 — stale)
#   3. Handler A: UPDATE SET login_count=1, COMMIT
#   4. Handler B: UPDATE SET login_count=1, COMMIT  ← lost update


def demo_trace_markers() -> None:
    """Reproduce the lost update deterministically against real Postgres."""
    print(_SEP)
    print("Demo 1: SQLAlchemy lost update  (TraceExecutor — deterministic)")
    print(_SEP)
    print()
    print("  Scenario:")
    print("    Two handlers each read login_count and increment it in Python.")
    print("    Expected final login_count: 2")
    print()

    _setup_table()

    errors: dict[str, str] = {}

    def handler_a() -> None:
        try:
            with Session(_engine) as session:
                user = session.get(User, 1)  # frontrun: orm_read
                assert user is not None
                user.login_count = user.login_count + 1
                session.commit()  # frontrun: orm_write
        except Exception as exc:
            errors["a"] = str(exc).strip()

    def handler_b() -> None:
        try:
            with Session(_engine) as session:
                user = session.get(User, 1)  # frontrun: orm_read
                assert user is not None
                user.login_count = user.login_count + 1
                session.commit()  # frontrun: orm_write
        except Exception as exc:
            errors["b"] = str(exc).strip()

    # Force both SELECTs before either UPDATE
    schedule = Schedule(
        [
            Step("a", "orm_read"),  # Handler A does SELECT (login_count=0)
            Step("b", "orm_read"),  # Handler B does SELECT (login_count=0, stale)
            Step("a", "orm_write"),  # Handler A commits (login_count=1)
            Step("b", "orm_write"),  # Handler B commits (login_count=1, should be 2!)
        ]
    )

    executor = TraceExecutor(schedule)
    executor.run("a", handler_a)
    executor.run("b", handler_b)
    executor.wait(timeout=10.0)

    if errors:
        for label, msg in errors.items():
            print(f"  {label} error: {msg}")
        print()

    count = _read_count()

    print(f"  Final login_count: {count}  (expected 2)")
    print()

    if count == 1:
        print("  LOST UPDATE confirmed: one increment was silently lost.")
        print("  Handler B's commit wrote login_count=1 based on a stale")
        print("  read, overwriting handler A's increment.")
        print()
        print("  Reproducibility: 100% — the Schedule deterministically forces")
        print("  both SELECTs to run before either UPDATE on every execution.")
    elif count == 2:
        print("  No lost update (both increments persisted).")
    else:
        print(f"  Unexpected login_count={count}.")
    print()


# ============================================================================
# Demo 2: Automatic detection with bytecode exploration
# ============================================================================
#
# explore_interleavings generates random opcode-level schedules and runs
# both handlers under controlled interleaving.  psycopg2 (used under the
# hood by SQLAlchemy) releases the GIL during C-level I/O, so the
# scheduler has real interleaving leverage.


def demo_bytecode_exploration() -> None:
    """Find the lost update automatically with random bytecode schedules."""
    print(_SEP)
    print("Demo 2: SQLAlchemy lost update  (bytecode exploration — automatic)")
    print(_SEP)
    print()
    print("  Generating random opcode-level schedules and running both")
    print("  handlers against real Postgres.  Checking whether both")
    print("  increments persist after each interleaving.")
    print()

    _setup_table()

    class _State:
        """Per-attempt state: reset login_count to 0."""

        def __init__(self) -> None:
            with Session(_engine) as session:
                user = session.get(User, 1)
                assert user is not None
                user.login_count = 0
                session.commit()

    def _thread_fn(_state: _State) -> None:
        with Session(_engine) as session:
            user = session.get(User, 1)
            assert user is not None
            user.login_count = user.login_count + 1
            session.commit()

    def _invariant(_state: _State) -> bool:
        with Session(_engine) as session:
            user = session.get(User, 1)
            assert user is not None
            return user.login_count == 2

    result = explore_interleavings(
        setup=_State,
        threads=[_thread_fn, _thread_fn],
        invariant=_invariant,
        max_attempts=50,
        seed=42,
        detect_io=False,  # psycopg2 uses C-level sockets
        deadlock_timeout=15.0,
        timeout_per_run=60.0,
        reproduce_on_failure=5,
    )

    print(f"  property_holds    : {result.property_holds}")
    print(f"  attempts_explored : {result.num_explored}")
    if result.counterexample is not None:
        repro = ""
        if result.reproduction_attempts:
            repro = f"  ({result.reproduction_successes}/{result.reproduction_attempts} reproductions)"
        print(f"  counterexample found after {result.num_explored} attempt(s){repro}")
    print()
    if result.explanation:
        for line in result.explanation.splitlines():
            print("  " + line)
        print()
    if not result.property_holds:
        print("  LOST UPDATE confirmed via bytecode exploration.")
    else:
        print("  No lost update found in the explored interleavings.")
    print()


# ============================================================================
# Demo 3: Naive threading — intermittent failure
# ============================================================================
#
# In production, two login events for the same user don't arrive at the
# exact same instant.  The race only manifests when the second handler's
# SELECT lands inside the first handler's SELECT→COMMIT window (typically
# <1ms on localhost).  We model realistic request arrival by staggering
# thread starts with a random 0–15ms offset, which gives roughly 10%
# collision rate on localhost.
#
# If both threads started at the exact same instant (offset=0), the race
# would reproduce ~95% of the time because psycopg2 releases the GIL
# during I/O and both threads hit their SELECTs simultaneously.


def demo_naive_threading(trials: int = 500) -> None:
    """Show the intermittent nature of the race with plain threads + Postgres."""
    print(_SEP)
    print(f"Demo 3: Naive threading + SQLAlchemy  ({trials} trials)")
    print(_SEP)
    print()
    print("  Running both handlers in plain threads against real Postgres.")
    print("  Threads start with a random 0-15ms offset to model realistic")
    print("  request arrival timing.  Counting how often the race manifests...")
    print()

    _setup_table()

    failures = 0
    rng = random.Random(42)

    for _ in range(trials):
        _reset_row()

        def handler() -> None:
            with Session(_engine) as session:
                user = session.get(User, 1)
                assert user is not None
                user.login_count = user.login_count + 1
                session.commit()

        t1 = threading.Thread(target=handler)
        t2 = threading.Thread(target=handler)
        t1.start()
        # Random offset models realistic request arrival timing.
        time.sleep(rng.uniform(0, 0.015))
        t2.start()
        t1.join()
        t2.join()

        if _read_count() != 2:
            failures += 1

    rate = failures / trials * 100
    print(f"  Trials:   {trials}")
    print(f"  Failures: {failures}")
    print(f"  Rate:     {rate:.1f}%")
    print()
    if failures > 0:
        print(f"  The race manifested in {failures}/{trials} trials ({rate:.1f}%).")
        print("  frontrun makes it 100% reproducible.")
    else:
        print("  No race observed in this run.")
    print()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    demo_trace_markers()
    demo_bytecode_exploration()
    demo_naive_threading()
