"""A tiny SQLite counter used by cross-process e2e tests.

The module lives under ``tests`` so subprocess targets remain importable during
the test suite without shipping demo code in the runtime package.
"""

from __future__ import annotations

import sqlite3
import threading


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
    """Return the current counter value."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        row = conn.execute("SELECT val FROM counter WHERE id = 1").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def read_in_joined_thread(db_path: str) -> None:
    """Exercise SQL from a sequentially joined child thread."""
    thread = threading.Thread(target=read, args=(db_path,))
    thread.start()
    thread.join()


def increment(db_path: str) -> None:
    """Racy read-modify-write: SELECT then UPDATE."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        val = conn.execute("SELECT val FROM counter WHERE id = 1").fetchone()[0]
        conn.execute("UPDATE counter SET val = ? WHERE id = 1", (val + 1,))
    finally:
        conn.close()


def increment_atomic(db_path: str) -> None:
    """Safe increment: a single atomic UPDATE."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("UPDATE counter SET val = val + 1 WHERE id = 1")
    finally:
        conn.close()


def setup_snapshot(db_path: str) -> None:
    """(Re)create the account + per-worker snapshot tables."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS acct (id INTEGER PRIMARY KEY, v INTEGER)")
        conn.execute("CREATE TABLE IF NOT EXISTS snap (wid INTEGER PRIMARY KEY, seen INTEGER)")
        conn.execute("DELETE FROM acct")
        conn.execute("DELETE FROM snap")
        conn.execute("INSERT INTO acct VALUES (1, 0)")
        conn.execute("INSERT INTO snap VALUES (0, -1), (1, -1)")
    finally:
        conn.close()


def snapshot_values(db_path: str) -> list[int]:
    """Return the per-worker snapshot values ordered by worker id."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        return [row[0] for row in conn.execute("SELECT seen FROM snap ORDER BY wid")]
    finally:
        conn.close()


def update_then_snapshot(db_path: str, wid: int) -> None:
    """Write-then-read-back: increment, read own write, record what was seen."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute("UPDATE acct SET v = v + 1 WHERE id = 1")
        seen = conn.execute("SELECT v FROM acct WHERE id = 1").fetchone()[0]
        conn.execute("UPDATE snap SET seen = ? WHERE wid = ?", (seen, wid))
    finally:
        conn.close()
