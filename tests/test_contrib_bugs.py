"""Tests for bugs and simplification seams in contrib wrappers."""

from __future__ import annotations

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

    @pytest.mark.parametrize("failure", ["ensure", "execute"])
    def test_connection_closed_when_setup_raises(self, failure: str) -> None:
        from frontrun.contrib.django._shared import _fresh_connection

        conn = _FakeConnection(ensure_raises=failure == "ensure", execute_raises=failure == "execute")
        connections = {"default": conn}

        with pytest.raises(RuntimeError):  # noqa: PT012
            with _fresh_connection(connections, "default", lock_timeout=5000):
                pass

        assert conn.close_calls == 2


class TestSqlalchemyAsyncLockTimeoutForwarding:
    """async_sqlalchemy_dpor should forward lock_timeout to _explore_async_dpor."""

    def test_lock_timeout_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from frontrun.contrib.sqlalchemy._async import async_sqlalchemy_dpor

        explore = AsyncMock()
        monkeypatch.setattr("frontrun.async_dpor._explore_async_dpor", explore)
        result = asyncio.run(
            async_sqlalchemy_dpor(MagicMock(), object, [AsyncMock()], lambda state: True, lock_timeout=500)
        )
        assert result is explore.return_value
        assert explore.await_args.kwargs["lock_timeout"] == 500


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
            if name == "exit":
                assert current.get() is (None if failure in {"setup", "wrapping"} else conn)
            if failure == name:
                raise error

        context.__enter__.side_effect = lambda: (stage("enter"), conn)[1]
        context.__exit__.side_effect = lambda *exc: (stage("exit"), True)[1]
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


@pytest.mark.parametrize("async_engine", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_sqlalchemy_setup_suppression(async_engine: bool, fails: bool) -> None:
    from unittest.mock import MagicMock

    from frontrun._cooperative import is_sync_suppressed
    from frontrun.contrib.sqlalchemy._shared import wrap_async_setup, wrap_sync_setup

    def operation(*args: object, **kwargs: object) -> str:
        assert is_sync_suppressed()
        if fails:
            raise RuntimeError("disposal failed")
        return "state"

    engine = MagicMock()
    engine.dispose.side_effect = engine.sync_engine.dispose.side_effect = operation
    wrapper = (wrap_async_setup if async_engine else wrap_sync_setup)(engine, operation)
    if fails:
        with pytest.raises(RuntimeError, match="disposal failed"):
            wrapper()
    else:
        assert wrapper() == "state"
    assert not is_sync_suppressed()
