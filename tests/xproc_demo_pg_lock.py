"""Importable Postgres worker for cross-process row-lock redirect tests.

Used by ``tests/test_row_lock_redirect_coverage.py`` via
``frontrun.explore_processes`` — each worker runs in its own subprocess
(with frontrun's SQL patching bootstrapped by ``worker_main``) and contends
on the same Postgres row with ``SELECT ... FOR UPDATE``, which drives the
coordinator's modeled row-lock arbitration into the redirect seam that
SQLite-backed workers never reach (issue #250).
"""

from __future__ import annotations

import psycopg2

TABLE = "xproc_redirect_accounts"


def locked_increment(dsn: str) -> None:
    """Read-modify-write of the shared balance under a row lock."""
    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute(f"SELECT balance FROM {TABLE} WHERE id = 1 FOR UPDATE")
        row = cur.fetchone()
        assert row is not None
        cur.execute(f"UPDATE {TABLE} SET balance = %s WHERE id = 1", (row[0] + 100,))
        conn.commit()
    finally:
        conn.close()
