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
from tests.django_test_helpers import configure_django

pytest_plugins = ("tests.django_test_helpers",)

pytestmark = pytest.mark.integration

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")

configure_django(django, settings, _DB_NAME)

from django.contrib.auth import get_user_model  # noqa: E402

from frontrun.contrib.django import django_dpor  # noqa: E402

User = get_user_model()


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
