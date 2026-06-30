"""Corrected bank transfer: a single lock makes all interleavings safe.

This is the locked counterpart to ``dpor_bank_transfer.py``.  Adding a
shared ``threading.Lock`` around each transfer makes the operations atomic:
DPOR can only reorder the two lock acquisitions (T1 first vs T2 first), so
it explores far fewer interleavings — all of which leave the total at 300.

Compare against ``dpor_bank_transfer.py`` to see how locking reduces both
the number of explored paths and the number of failures.

Usage::

    python examples/dpor_bank_transfer_locked.py [report.html]
"""

from __future__ import annotations

import sys
import threading


class Accounts:
    """Three bank accounts protected by a single shared lock."""

    def __init__(self) -> None:
        self.a = 100
        self.b = 100
        self.c = 100
        self.lock = threading.Lock()


def transfer_a_to_b(accounts: Accounts) -> None:
    """Atomic transfer of 60 from A to B (lock held for the entire operation)."""
    with accounts.lock:
        if accounts.a >= 60:
            accounts.a -= 60
            accounts.b += 60


def transfer_b_to_c(accounts: Accounts) -> None:
    """Atomic transfer of 80 from B to C (lock held for the entire operation)."""
    with accounts.lock:
        if accounts.b >= 80:
            accounts.b -= 80
            accounts.c += 80


def run_exploration(report_path: str | None = None) -> None:
    """Run DPOR exploration, optionally writing an HTML report to *report_path*."""
    import frontrun
    import frontrun._report

    frontrun._report._global_report_path = report_path
    try:
        result = frontrun.explore(
            setup=Accounts,
            workers=[transfer_a_to_b, transfer_b_to_c],
            invariant=lambda accs: accs.a + accs.b + accs.c == 300,
            preemption_bound=2,
            stop_on_first=False,
        )
        print(result.explanation)
    finally:
        frontrun._report._global_report_path = None


if __name__ == "__main__":
    report_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_exploration(report_path)
