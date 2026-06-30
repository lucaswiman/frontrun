import sqlite3

from frontrun import explore
from frontrun._io_detection import set_io_reporter
from frontrun._sql_cursor import patch_sql, unpatch_sql


def test_cross_column_conflict_fix():
    """Verify that DPOR finds conflicts between different column sets on the same table."""
    uri = "file:defect1_fix?mode=memory&cache=shared"
    # Keep one connection open to keep the database alive
    master = sqlite3.connect(uri, uri=True)
    master.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, active INT)")
    master.execute("DELETE FROM users")
    master.execute("INSERT INTO users VALUES (1, 'alice', 0)")
    master.commit()

    class State:
        def __init__(self):
            conn = sqlite3.connect(uri, uri=True)
            conn.execute("UPDATE users SET active = 0 WHERE id = 1")
            conn.commit()
            conn.close()
            self.activated = 0

    def thread_fn(s):
        conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        c = conn.cursor()
        # Thread 1/2: SELECT by username (e.g. Django natural key lookup)
        c.execute("SELECT active FROM users WHERE username = 'alice'")
        row = c.fetchone()
        if row and not row[0]:
            # Thread 1/2: UPDATE by id (e.g. Django .save() after lookup)
            # This SHOULD conflict with the SELECT above in the other thread.
            c.execute("UPDATE users SET active = 1 WHERE id = 1")
            s.activated += 1
        conn.close()

    # We expect DPOR to find the interleaving where both SELECT before either UPDATE.
    # This leads to s.activated == 2, which fails the invariant.
    res = explore(
        setup=State,
        workers=[thread_fn, thread_fn],
        invariant=lambda s: s.activated < 2,
        detect_io=True,
    )

    # Bug fixed: res.property_holds should be False (race found)
    assert not res.property_holds, "DPOR missed the cross-column race condition!"
    master.close()


def test_row_level_preserved_for_primary_colset():
    """Primary-key writes should stay row-granular, not collapse to table writes."""
    events: list[tuple[str, str]] = []

    def reporter(resource_id: str, kind: str) -> None:
        events.append((resource_id, kind))

    set_io_reporter(reporter)
    patch_sql()
    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE counters (id INTEGER PRIMARY KEY, val INT)")
        conn.execute("INSERT INTO counters VALUES (1, 0), (2, 0)")

        events.clear()
        conn.execute("UPDATE counters SET val = 1 WHERE id = 1")
        first_update = list(events)

        events.clear()
        conn.execute("UPDATE counters SET val = 1 WHERE id = 2")
        second_update = list(events)
    finally:
        unpatch_sql()
        set_io_reporter(None)
        if conn is not None:
            conn.close()

    first_writes = [res for res, kind in first_update if kind == "write"]
    second_writes = [res for res, kind in second_update if kind == "write"]

    assert first_writes, first_update
    assert second_writes, second_update
    assert "sql:counters" not in first_writes, first_update
    assert "sql:counters" not in second_writes, second_update
    assert set(first_writes).isdisjoint(second_writes), (first_update, second_update)
