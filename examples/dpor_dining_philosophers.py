"""Dining philosophers (3 philosophers): lock-ordering deadlock.

Three philosophers sit at a round table.  Each needs both the left and right
fork (a ``threading.Lock``) to eat.  They always pick up the left fork first,
then try for the right — a classic circular wait that deadlocks.

There is no shared state to update; the invariant is always ``True``.  DPOR
finds the deadlock directly via its wait-for-graph cycle detector.

The exploration can produce up to ``max_executions`` interleavings before
stopping.  With ``stop_on_first=False`` and a generous limit you can see both
deadlocking and non-deadlocking schedules in the report.

Usage::

    python examples/dpor_dining_philosophers.py [report.html]
"""

from __future__ import annotations

import sys
import threading


class Table:
    """The three forks on the table."""

    def __init__(self) -> None:
        self.forks = [threading.Lock() for _ in range(3)]


def make_philosopher(i: int):
    """Return a philosopher function for seat *i*."""

    def philosopher(table: Table) -> None:
        left = i
        right = (i + 1) % 3
        with table.forks[left]:
            with table.forks[right]:
                pass  # eating

    philosopher.__name__ = f"philosopher_{i}"
    return philosopher


def run_exploration(report_path: str | None = None) -> None:
    """Run DPOR exploration, optionally writing an HTML report to *report_path*."""
    import frontrun._report
    from frontrun import explore

    frontrun._report._global_report_path = report_path
    try:
        result = explore(
            setup=Table,
            workers=[make_philosopher(i) for i in range(3)],
            invariant=lambda _: True,
            preemption_bound=2,
            stop_on_first=False,
            detect_io=False,
            deadlock_timeout=2.0,
            max_executions=1000,
        )
        print(result.explanation)
    finally:
        frontrun._report._global_report_path = None


if __name__ == "__main__":
    report_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_exploration(report_path)
