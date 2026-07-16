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


def test_sqlalchemy_sync_wrapper_preserves_deferred_worker_return():
    """The core runner must see coroutine/generator returns and fail closed."""
    import contextvars
    from unittest.mock import MagicMock

    from frontrun.contrib.sqlalchemy._shared import wrap_sync_thread

    conn = MagicMock()
    conn_ctx = MagicMock()
    conn_ctx.__enter__ = MagicMock(return_value=conn)
    conn_ctx.__exit__ = MagicMock(return_value=False)
    engine = MagicMock()
    engine.connect.return_value = conn_ctx
    current_connection: contextvars.ContextVar[object] = contextvars.ContextVar("test_deferred_conn")
    deferred = object()

    wrapper = wrap_sync_thread(engine, current_connection, lock_timeout=None, fn=lambda _state: deferred)

    assert wrapper(None) is deferred


def test_django_sync_wrapper_preserves_deferred_worker_return():
    """Django connection cleanup must not discard the worker return value."""
    from frontrun.contrib.django._shared import wrap_sync_thread

    class Connection:
        def close(self) -> None:
            pass

        def ensure_connection(self) -> None:
            pass

    deferred = object()
    wrapper = wrap_sync_thread(
        lambda _state: deferred,
        connections={"default": Connection()},
        db_alias="default",
        lock_timeout=None,
    )

    assert wrapper(None) is deferred


def test_sqlalchemy_async_setup_recycles_pool_without_sync_closing() -> None:
    """Async-driver connections cannot be closed through sync_engine.dispose()."""
    from types import SimpleNamespace

    from frontrun.contrib.sqlalchemy._shared import wrap_async_setup

    calls: list[bool] = []

    def dispose(*, close: bool = True) -> None:
        calls.append(close)
        if close:
            raise RuntimeError("MissingGreenlet")

    engine = SimpleNamespace(sync_engine=SimpleNamespace(dispose=dispose))
    wrapped = wrap_async_setup(engine, lambda: "state")

    assert wrapped() == "state"
    assert calls == [False]


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
