"""Bank-account fund transfer: a mildly complicated DPOR race example.

Two concurrent non-atomic transfers share account B::

    Thread 1: transfer 60 from A to B
    Thread 2: transfer 80 from B to C

Both threads read-modify-write ``accounts.b`` without any locking.  In some
interleavings one write overwrites the other, leaving the total balance
different from 300.

Usage — run exploration and print the result::

    python examples/dpor_bank_transfer.py

Generate an interactive HTML exploration report::

    python examples/dpor_bank_transfer.py dpor_bank_transfer.html
"""

from __future__ import annotations

import sys


class Accounts:
    """Three bank accounts used to demonstrate a fund-transfer race."""

    def __init__(self) -> None:
        self.a = 100
        self.b = 100
        self.c = 100


def transfer_a_to_b(accounts: Accounts) -> None:
    """Non-atomic transfer of 60 from account A to account B."""
    if accounts.a >= 60:
        accounts.a -= 60
        accounts.b += 60


def transfer_b_to_c(accounts: Accounts) -> None:
    """Non-atomic transfer of 80 from account B to account C."""
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
