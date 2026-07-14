"""Regression: DPOR must explore writes landing between a write and its read-back.

Two workers each run the common write-then-read-back SQL shape (sqlite,
autocommit, one connection per worker):

    UPDATE acct SET v = v + 1 WHERE id = 1
    SELECT v FROM acct WHERE id = 1          -- read own write back
    UPDATE snap SET seen = ? WHERE wid = ?   -- record what was seen

The interleaving "both UPDATEs, then both SELECTs" makes both workers see
v == 2 and violates the invariant.  The DPOR engine used to record only the
*first* access per (thread, kind) per object, so each worker's read-back
SELECT on ``acct`` was invisible to race detection: the only seeded branches
were the two serial orders, and the run was certified as a false pass with
``exhausted=True`` even under ``preemption_bound=None`` — while
``strategy="exhaustive"`` found the violation on the same workload.
"""

from __future__ import annotations

import sqlite3

import frontrun


def _setup_db(db: str) -> str:
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS acct (id INTEGER PRIMARY KEY, v INTEGER)")
        conn.execute("CREATE TABLE IF NOT EXISTS snap (wid INTEGER PRIMARY KEY, seen INTEGER)")
        conn.execute("DELETE FROM acct")
        conn.execute("DELETE FROM snap")
        conn.execute("INSERT INTO acct VALUES (1, 0)")
        conn.execute("INSERT INTO snap VALUES (0, -1), (1, -1)")
    finally:
        conn.close()
    return db


def _snap_values(db: str) -> list[int]:
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        return [row[0] for row in conn.execute("SELECT seen FROM snap ORDER BY wid")]
    finally:
        conn.close()


def test_write_then_read_back_race_is_found_by_thread_dpor(tmp_path) -> None:
    db = str(tmp_path / "write_read_back.db")

    def make_worker(wid: int):
        def worker(dbp: str) -> None:
            conn = sqlite3.connect(dbp, isolation_level=None)
            try:
                conn.execute("UPDATE acct SET v = v + 1 WHERE id = 1")
                seen = conn.execute("SELECT v FROM acct WHERE id = 1").fetchone()[0]
                conn.execute("UPDATE snap SET seen = ? WHERE wid = ?", (seen, wid))
            finally:
                conn.close()

        return worker

    result = frontrun.explore(
        setup=lambda: _setup_db(db),
        workers=[make_worker(0), make_worker(1)],
        invariant=lambda _dbp: _snap_values(db) != [2, 2],
        execution="thread",
        preemption_bound=None,
    )

    assert not result.property_holds, (
        "DPOR certified a pass without exploring the interleaving where both "
        "UPDATEs land before either read-back SELECT"
    )
