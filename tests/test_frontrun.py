"""
Basic tests for frontrun library.
"""

import re

import frontrun


def test_import():
    """Test that frontrun module can be imported."""
    assert frontrun is not None


def test_version():
    """__version__ is a valid version string."""
    assert isinstance(frontrun.__version__, str)
    assert re.match(r"^\d+\.\d+", frontrun.__version__), f"Invalid version format: {frontrun.__version__}"


def test_cooperative_lock_release_in_dpor_machinery_clears_owner():
    """Releasing a CooperativeLock inside DPOR machinery must still clear _owner_thread_id."""
    from frontrun._cooperative import CooperativeLock, _scheduler_tls

    lock = CooperativeLock()
    lock.acquire()
    lock._owner_thread_id = 42

    _scheduler_tls._in_dpor_machinery = True
    try:
        lock.release()
        assert lock._owner_thread_id is None, (
            "_owner_thread_id must be cleared even when release() runs inside DPOR machinery"
        )
    finally:
        _scheduler_tls._in_dpor_machinery = False


def test_wrap_sync_thread_exit_receives_exception_info_on_lock_timeout_failure():
    """conn_ctx.__exit__ must receive real exception info so SQLAlchemy rolls back."""
    import contextvars
    from unittest.mock import MagicMock

    from frontrun.contrib.sqlalchemy._shared import wrap_sync_thread

    conn = MagicMock()
    conn.exec_driver_sql.side_effect = RuntimeError("lock_timeout failed")

    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)

    engine = MagicMock()
    engine.connect.return_value = conn_ctx

    current_connection: contextvars.ContextVar[object] = contextvars.ContextVar("test_conn")

    wrapper = wrap_sync_thread(engine, current_connection, lock_timeout=100, fn=lambda _s: None)

    try:
        wrapper(None)
    except RuntimeError:
        pass

    conn_ctx.__exit__.assert_called_once()
    exit_args = conn_ctx.__exit__.call_args[0]
    assert exit_args[0] is RuntimeError, f"Expected RuntimeError type, got {exit_args[0]}"
    assert isinstance(exit_args[1], RuntimeError), f"Expected RuntimeError instance, got {exit_args[1]}"
    assert exit_args[2] is not None, "Expected a traceback, got None"


def test_sqlite3_custom_factory_traced():
    """sqlite3.connect(factory=CustomConnection) must still trace SQL."""
    import sqlite3

    from frontrun._sql_cursor import _ORIGINAL_METHODS, patch_sql, unpatch_sql

    class CustomConnection(sqlite3.Connection):
        custom_attr = True

    try:
        patch_sql()
        conn = sqlite3.connect(":memory:", factory=CustomConnection)

        assert isinstance(conn, CustomConnection), "Connection should be an instance of CustomConnection"
        assert getattr(conn, "custom_attr", False), "Custom attribute should be preserved"

        cur = conn.cursor()
        traced_execute = type(cur).execute
        original_execute = _ORIGINAL_METHODS.get((sqlite3.Cursor, "execute"))
        assert original_execute is not None, "_ORIGINAL_METHODS should have sqlite3.Cursor.execute"
        assert traced_execute is not original_execute, (
            "Cursor.execute should be the traced version, not the original — custom factory bypassed tracing"
        )

        conn.close()
    finally:
        unpatch_sql()


def test_find_cycle_from_resets_visited_per_outer_neighbor():
    """Visited/path state must be independent per outer-loop iteration.

    The fix resets ``visited`` and ``path`` for each top-level neighbor in
    ``_find_cycle_from`` so that nodes explored via one neighbor cannot
    shadow reachable paths through a later neighbor.  This test validates
    the cycle is found regardless of which neighbor is explored first.
    """
    from frontrun._deadlock import WaitForGraph

    g = WaitForGraph()

    start = ("thread", 1)
    n1 = ("lock", 1)
    n2 = ("lock", 2)
    b = ("thread", 2)
    c = ("lock", 3)

    g._edges[start] = {n1, n2}
    g._edges[n1] = {b}
    g._edges[n2] = {b}
    g._edges[b] = {c}
    g._edges[c] = {start}

    cycle = g._find_cycle_from(start)
    assert cycle is not None, "Expected a cycle (start -> ... -> start) but got None"
    assert cycle[0] == cycle[-1] == start
