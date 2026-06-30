"""SQLite counter: application-level lost update with a real database.

Two threads each read a counter from SQLite, increment it in Python, then
write it back.  Because neither thread uses an atomic ``UPDATE counter = counter + 1``
the second writer silently overwrites the first — a classic lost update.

DPOR intercepts the SQL operations via its built-in SQL cursor patching,
detects the read/write conflict on the ``counter`` table, and explores all
meaningful interleavings.

The fixed variant uses a single atomic SQL statement instead of a
read-modify-write round-trip; DPOR finds no violations.

Usage::

    python examples/dpor_sqlite_counter.py [report.html]          # racy
    python examples/dpor_sqlite_counter.py [report.html] fixed    # corrected
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile


def _make_db(path: str) -> None:
    """Create (or reset) the counter table at *path*."""
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE IF EXISTS counter")
    conn.execute("CREATE TABLE counter (value INTEGER NOT NULL)")
    conn.execute("INSERT INTO counter VALUES (0)")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Racy version: read-modify-write in Python
# ---------------------------------------------------------------------------

class State:
    """Per-execution shared state: a fresh SQLite file reset for every run."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        _make_db(db_path)


def racy_increment(state: State) -> None:
    """Read the counter, increment in Python, write back — not atomic."""
    conn = sqlite3.connect(state.db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("SELECT value FROM counter")
    value = cur.fetchone()[0]
    value += 1
    cur.execute("UPDATE counter SET value = ?", (value,))
    conn.commit()
    conn.close()


def atomic_increment(state: State) -> None:
    """Increment the counter with a single atomic SQL statement."""
    conn = sqlite3.connect(state.db_path, check_same_thread=False)
    conn.execute("UPDATE counter SET value = value + 1")
    conn.commit()
    conn.close()


def _read_counter(db_path: str) -> int:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    value = conn.execute("SELECT value FROM counter").fetchone()[0]
    conn.close()
    return value


# ---------------------------------------------------------------------------
# Exploration entry points
# ---------------------------------------------------------------------------

def run_exploration(report_path: str | None = None) -> None:
    """Racy read-modify-write: DPOR should find an invariant violation."""
    import frontrun._report
    from frontrun import explore

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        frontrun._report._global_report_path = report_path
        try:
            result = explore(
                setup=lambda: State(db_path),
                workers=[racy_increment, racy_increment],
                invariant=lambda s: _read_counter(db_path) == 2,
                detect_io=True,
                preemption_bound=2,
                stop_on_first=False,
                deadlock_timeout=5.0,
            )
            print(result.explanation)
        finally:
            frontrun._report._global_report_path = None
    finally:
        os.unlink(db_path)


def run_exploration_fixed(report_path: str | None = None) -> None:
    """Atomic SQL update: DPOR should find no violations."""
    import frontrun._report
    from frontrun import explore

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        frontrun._report._global_report_path = report_path
        try:
            result = explore(
                setup=lambda: State(db_path),
                workers=[atomic_increment, atomic_increment],
                invariant=lambda s: _read_counter(db_path) == 2,
                detect_io=True,
                preemption_bound=2,
                stop_on_first=False,
                deadlock_timeout=5.0,
            )
            print(result.explanation)
        finally:
            frontrun._report._global_report_path = None
    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    report_path = sys.argv[1] if len(sys.argv) > 1 else None
    mode = sys.argv[2] if len(sys.argv) > 2 else "racy"
    if mode == "fixed":
        run_exploration_fixed(report_path)
    else:
        run_exploration(report_path)
