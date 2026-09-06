"""Tests for bugs and simplification seams in contrib wrappers."""

from __future__ import annotations

import inspect

import pytest


class _FakeCursor:
    def __init__(self, execute_raises: bool) -> None:
        self._execute_raises = execute_raises

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str) -> None:
        if self._execute_raises:
            raise RuntimeError(f"backend rejected: {sql}")


class _FakeConnection:
    """Minimal stand-in for a Django connection, tracking close() calls."""

    def __init__(self, *, ensure_raises: bool = False, execute_raises: bool = False) -> None:
        self.close_calls = 0
        self.ensure_calls = 0
        self._ensure_raises = ensure_raises
        self._execute_raises = execute_raises

    def close(self) -> None:
        self.close_calls += 1

    def ensure_connection(self) -> None:
        self.ensure_calls += 1
        if self._ensure_raises:
            raise RuntimeError("could not connect")

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._execute_raises)


class TestDjangoFreshConnectionCleanup:
    """Setup failures must not leak the fresh connection."""

    def test_connection_closed_when_lock_timeout_setup_raises(self) -> None:
        from frontrun.contrib.django._shared import _fresh_connection

        conn = _FakeConnection(execute_raises=True)
        connections = {"default": conn}

        with pytest.raises(RuntimeError):  # noqa: PT012
            with _fresh_connection(connections, "default", lock_timeout=5000):
                pass

        assert conn.close_calls == 2

    def test_connection_closed_when_ensure_connection_raises(self) -> None:
        from frontrun.contrib.django._shared import _fresh_connection

        conn = _FakeConnection(ensure_raises=True)
        connections = {"default": conn}

        with pytest.raises(RuntimeError):  # noqa: PT012
            with _fresh_connection(connections, "default", lock_timeout=None):
                pass

        assert conn.close_calls == 2


class TestDjangoSharedConnectionWrapper:
    """Django sync/async wrappers should share one connection helper."""

    def test_wrappers_use_shared_connection_helper(self) -> None:
        """Both Django wrappers should route through one shared helper."""
        from frontrun.contrib.django import _shared as django_shared

        source = inspect.getsource(django_shared)
        assert "_fresh_connection" in source, "Expected a shared Django connection helper"
        assert "with _fresh_connection(" in inspect.getsource(django_shared.wrap_sync_thread)
        assert "with _fresh_connection(" in inspect.getsource(django_shared.wrap_async_task)


class TestSqlalchemyAsyncLockTimeoutForwarding:
    """async_sqlalchemy_dpor should forward lock_timeout to _explore_async_dpor."""

    def test_lock_timeout_forwarded(self) -> None:
        """lock_timeout should be passed through to _explore_async_dpor.

        Regression: lock_timeout is consumed by the function signature and
        NOT present in **kwargs, so it was never forwarded (unlike the sync
        version which explicitly passes lock_timeout=lock_timeout).
        """
        from frontrun.contrib.sqlalchemy._async import async_sqlalchemy_dpor

        source = inspect.getsource(async_sqlalchemy_dpor)
        assert "lock_timeout=lock_timeout" in source, (
            "async_sqlalchemy_dpor does not forward lock_timeout to _explore_async_dpor. "
            "The sync version (sqlalchemy_dpor) passes lock_timeout=lock_timeout explicitly."
        )


class TestSqlalchemySyncExitExceptionInfo:
    """SQLAlchemy sync wrapper should pass exception info to __exit__."""

    @pytest.mark.parametrize("failure", [None, "enter", "setup", "wrapping", "body", "exit"])
    def test_connection_lifetime(self, failure: str | None) -> None:
        from contextvars import ContextVar
        from unittest.mock import MagicMock, PropertyMock

        from frontrun._cooperative import is_sync_suppressed
        from frontrun.contrib.sqlalchemy._shared import wrap_sync_thread

        conn, context, engine = MagicMock(), MagicMock(), MagicMock()
        current: ContextVar[object] = ContextVar("test_connection", default=None)
        error = RuntimeError(failure)
        engine.connect.return_value = context

        def stage(name: str) -> None:
            assert is_sync_suppressed() == (name != "body")
            if name == "body":
                assert current.get() is conn
            if failure == name:
                raise error

        context.__enter__.side_effect = lambda: (stage("enter"), conn)[1]
        context.__exit__.side_effect = lambda *exc: stage("exit")
        conn.exec_driver_sql.side_effect = lambda sql: stage("setup")
        if failure == "wrapping":
            type(conn).execute = PropertyMock(side_effect=error)
        wrapped = wrap_sync_thread(engine, current, 5000, lambda state: stage("body"))
        if failure is None:
            wrapped(None)
        else:
            with pytest.raises(RuntimeError) as caught:
                wrapped(None)
            assert caught.value is error

        assert not is_sync_suppressed()
        assert current.get() is None
        if failure == "enter":
            context.__exit__.assert_not_called()
        else:
            context.__exit__.assert_called_once()
            assert context.__exit__.call_args.args[1] is (error if failure in {"setup", "wrapping", "body"} else None)


class TestSqlalchemyConnectionHelpers:
    """SQLAlchemy wrappers should share connection-scope plumbing."""

    def test_wrappers_use_shared_connection_scope(self) -> None:
        """Both SQLAlchemy wrappers should route token handling through one helper."""
        from frontrun.contrib.sqlalchemy import _shared as sa_shared

        source = inspect.getsource(sa_shared)
        assert "_current_connection_scope" in source, "Expected a shared SQLAlchemy connection-scope helper"
        assert "_lock_timeout_statement" in source, "Expected a shared SQLAlchemy lock_timeout helper"


class TestSqlalchemyAsyncSetupSuppression:
    """Async SQLAlchemy setup should suppress reporting during engine.dispose()."""

    def test_async_setup_suppresses_reporting(self) -> None:
        """wrap_async_setup should suppress cooperative reporting during dispose.

        The sync version (wrap_sync_setup) correctly wraps engine.dispose() in
        suppress_sync_reporting/unsuppress_sync_reporting, but wrap_async_setup
        calls engine.sync_engine.dispose() without suppression. This can cause
        internal SQLAlchemy/psycopg2 lock events to leak into DPOR reporting
        during setup.
        """
        from frontrun.contrib.sqlalchemy._shared import wrap_async_setup

        source = inspect.getsource(wrap_async_setup)
        assert "suppress_sync_reporting" in source, (
            "wrap_async_setup should suppress cooperative reporting during dispose(), "
            "like wrap_sync_setup does. Internal engine locks are implementation details."
        )


class TestRedisAsyncReportedBranch:
    """Async Redis interceptors should share reported-command handling."""

    def test_async_interceptors_use_shared_reported_branch(self) -> None:
        """Both async Redis interceptors should route the reported branch through one helper."""
        from frontrun import _redis_client_async

        source = inspect.getsource(_redis_client_async)
        assert "_dispatch_async" in source, "Expected a shared async Redis dispatch helper"
        assert "_dispatch_async(" in inspect.getsource(_redis_client_async._intercept_execute_command_async)
        assert "_dispatch_async(" in inspect.getsource(_redis_client_async._intercept_pipeline_execute_async)
