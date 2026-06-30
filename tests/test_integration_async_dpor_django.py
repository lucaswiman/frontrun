"""Integration tests: async DPOR with Django async ORM against real Postgres.

Tests that ``explore(strategy="dpor")`` can detect race conditions using
Django's async ORM, with await_point() as the scheduling granularity.

Requires a running Postgres with a ``frontrun_test`` database::

    createdb frontrun_test
"""

from __future__ import annotations

import asyncio
import os

import pytest

# Allow sync Django ORM from async context in tests
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")

try:
    import django
    from django.conf import settings
except ImportError:
    pytest.skip("django not installed", allow_module_level=True)

pytestmark = pytest.mark.integration

try:
    import psycopg2  # noqa: F401
except ImportError:
    pytest.skip("psycopg2 not installed", allow_module_level=True)

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")

if not settings.configured:
    settings.configure(
        DATABASES={"default": {"ENGINE": "django.db.backends.postgresql", "NAME": _DB_NAME}},
        INSTALLED_APPS=["django.contrib.contenttypes", "django.contrib.auth"],
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from django.db import connection, connections  # noqa: E402

from frontrun import explore  # noqa: E402
from frontrun.async_dpor import await_point  # noqa: E402
from frontrun.contrib.django import async_django_dpor  # noqa: E402

User = get_user_model()


@pytest.fixture(scope="module")
def _pg_available():
    """Ensure Postgres is available and setup test tables."""
    try:
        connection.ensure_connection()
    except Exception:
        pytest.skip(f"PostgreSQL not available at {_DB_NAME}")

    with connection.cursor() as cur:
        for tbl in [
            "auth_user_groups",
            "auth_user_user_permissions",
            "auth_user",
            "auth_group_permissions",
            "auth_group",
            "auth_permission",
            "django_content_type",
        ]:
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    with connection.schema_editor() as editor:
        from django.contrib.auth.models import Group, Permission
        from django.contrib.contenttypes.models import ContentType

        editor.create_model(ContentType)
        editor.create_model(Permission)
        editor.create_model(Group)
        editor.create_model(User)
    yield
    with connection.cursor() as cur:
        for tbl in [
            "auth_user_groups",
            "auth_user_user_permissions",
            "auth_user",
            "auth_group_permissions",
            "auth_group",
            "auth_permission",
            "django_content_type",
        ]:
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")


class TestAsyncDporDjango:
    """Async DPOR integration tests with Django async ORM."""

    def test_lock_timeout_is_forwarded_and_cursor_exits(self, _pg_available, monkeypatch) -> None:
        """async_django_dpor should apply lock_timeout using a real Django cursor.

        This uses the real Django connection and database, but wraps the
        cursor so we can verify the SQL that was executed and that the cursor
        context manager exits cleanly.
        """

        executed_sql: list[str] = []
        cursor_exit_called = False
        conn = connections["default"]
        original_cursor = conn.cursor

        class CursorProxy:
            def __init__(self, cursor):
                self._cursor = cursor
                self._entered_cursor = None

            def __enter__(self):
                self._entered_cursor = self._cursor.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                nonlocal cursor_exit_called
                cursor_exit_called = True
                return self._cursor.__exit__(exc_type, exc, tb)

            def execute(self, sql, *args, **kwargs):
                executed_sql.append(sql)
                target = self._entered_cursor or self._cursor
                return target.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                target = self._entered_cursor or self._cursor
                return getattr(target, name)

        def cursor_wrapper(*args, **kwargs):
            return CursorProxy(original_cursor(*args, **kwargs))

        monkeypatch.setattr(conn, "cursor", cursor_wrapper)

        class _State:
            pass

        async def noop(state: _State) -> None:
            await await_point()

        async def run_test():
            return await async_django_dpor(
                setup=_State,
                tasks=[noop],
                invariant=lambda s: True,
                lock_timeout=500,
                detect_sql=True,
            )

        result = asyncio.run(run_test())

        assert result.property_holds
        assert "SET lock_timeout = '500ms'" in executed_sql
        assert cursor_exit_called

    @pytest.mark.intentionally_leaves_dangling_threads
    def test_activation_race(self, _pg_available) -> None:
        """Async DPOR should detect a double-activation race using Django async ORM."""

        class _State:
            def __init__(self) -> None:
                User.objects.filter(username="async_testuser").delete()
                User.objects.create_user(username="async_testuser", is_active=False)
                self.results: list[str | None] = [None, None]

        async def activate(state: _State, idx: int) -> None:
            from asgiref.sync import sync_to_async

            user = await sync_to_async(User.objects.get)(username="async_testuser")
            is_active = user.is_active
            await await_point()
            if not is_active:
                user.is_active = True
                await sync_to_async(user.save)()
                state.results[idx] = "activated"
            else:
                state.results[idx] = "already_active"

        def make_task(idx: int):
            async def task(state: _State) -> None:
                await activate(state, idx)

            return task

        def invariant(state: _State) -> bool:
            return not (state.results[0] == "activated" and state.results[1] == "activated")

        async def run_test():
            return await explore(
                setup=_State,
                workers=[make_task(0), make_task(1)],
                invariant=invariant,
                detect_io=True,
                deadlock_timeout=10.0,
                timeout_per_run=15.0,
            )

        result = asyncio.run(run_test())
        assert not result.property_holds, (
            f"Async DPOR should detect the double-activation race. Explored {result.num_explored} interleavings."
        )

    def test_exploration_completes(self, _pg_available) -> None:
        """Verify exploration completes without deadlock with Django async."""

        class _State:
            def __init__(self) -> None:
                User.objects.filter(username="async_testuser2").delete()
                User.objects.create_user(username="async_testuser2", is_active=False)

        async def read_only(state: _State) -> None:
            from asgiref.sync import sync_to_async

            await sync_to_async(User.objects.get)(username="async_testuser2")
            await await_point()

        async def run_test():
            return await explore(
                setup=_State,
                workers=[read_only, read_only],
                invariant=lambda s: True,
                detect_io=True,
                deadlock_timeout=10.0,
            )

        result = asyncio.run(run_test())
        assert result.property_holds
        assert result.num_explored >= 1
