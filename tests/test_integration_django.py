"""Integration tests for Django: DPOR exploration with real Postgres.

Includes a regression test for the pytest plugin deadlock when processing
LD_PRELOAD events during DPOR execution.
"""

from __future__ import annotations

import os

import pytest

try:
    import django
    from django.conf import settings
except ImportError:
    pytest.skip("django not installed", allow_module_level=True)

try:
    import psycopg2  # noqa: F401
except ImportError:
    pytest.skip("psycopg2 not installed", allow_module_level=True)

from frontrun.cli import require_active

pytestmark = pytest.mark.integration

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")

if not settings.configured:
    settings.configure(
        DATABASES={"default": {"ENGINE": "django.db.backends.postgresql", "NAME": _DB_NAME}},
        INSTALLED_APPS=["django.contrib.contenttypes", "django.contrib.auth"],
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()

from django.contrib.auth import get_user_model  # noqa: E402
from django.db import connection  # noqa: E402

from frontrun.contrib.django import django_dpor  # noqa: E402

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


class TestDjangoIntegration:
    """Integration tests for Django and DPOR."""

    def test_dpor_activation_race(self, _pg_available) -> None:
        """Verify that DPOR finds a Django user activation race without deadlocking."""
        require_active("test_dpor_activation_race")

        class _State:
            def __init__(self) -> None:
                User.objects.filter(username="testuser").delete()
                User.objects.create_user(username="testuser", is_active=False)
                self.results: list[str | None] = [None, None]

        def _make_fn(i: int):
            def fn(state: _State) -> None:
                try:
                    user = User.objects.get(username="testuser")
                    if user.is_active:
                        state.results[i] = "already_active"
                        return
                    user.is_active = True
                    user.save()
                    state.results[i] = "activated"
                except Exception as exc:
                    state.results[i] = f"error: {exc}"

            return fn

        def _invariant(state: _State) -> bool:
            return not (state.results[0] == "activated" and state.results[1] == "activated")

        result = django_dpor(
            setup=_State,
            threads=[_make_fn(0), _make_fn(1)],
            invariant=_invariant,
            deadlock_timeout=15.0,
            timeout_per_run=30.0,
        )

        assert not result.property_holds, "DPOR should find the double-activation race"
        assert result.num_explored > 0
