"""A tiny SQLite counter used to demonstrate/e2e-test cross-process exploration.

Lives inside the package so it is importable by ``module:callable`` name in a
freshly spawned worker regardless of the working directory. ``increment`` is the
racy target (read-modify-write across two statements); ``increment_atomic`` is
the safe variant (a single ``UPDATE ... = val + 1``). ``setup`` and ``read`` run
in the coordinator process to reset and check the shared database.

Connections use ``isolation_level=None`` (autocommit) so each statement is its
own transaction — the coordinator controls ordering at statement granularity and
SQLite never blocks one worker's write behind another's open transaction.
"""

from __future__ import annotations

import sqlite3


def setup(db_path: str) -> None:
    """(Re)create the counter table with a single row ``val = 0``."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS counter (id INTEGER PRIMARY KEY, val INTEGER)")
        conn.execute("DELETE FROM counter")
        conn.execute("INSERT INTO counter (id, val) VALUES (1, 0)")
    finally:
        conn.close()


def read(db_path: str) -> int | None:
    """Return the current counter value (coordinator-side invariant check)."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        row = conn.execute("SELECT val FROM counter WHERE id = 1").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def increment(db_path: str) -> None:
    """Racy read-modify-write: SELECT then UPDATE (two scheduling points)."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        val = conn.execute("SELECT val FROM counter WHERE id = 1").fetchone()[0]
        conn.execute("UPDATE counter SET val = ? WHERE id = 1", (val + 1,))
    finally:
        conn.close()


def increment_atomic(db_path: str) -> None:
    """Safe increment: a single atomic UPDATE (one scheduling point)."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("UPDATE counter SET val = val + 1 WHERE id = 1")
    finally:
        conn.close()
