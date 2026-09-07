"""Opt-in Django integration-test setup.

This module is imported only by Django integration tests and has no Django
imports at module load time; it must not be imported from global conftest.
"""

import pytest


def configure_django(django, settings, db_name: str) -> None:
    """Configure Django for a module without importing Django eagerly."""
    if not settings.configured:
        settings.configure(
            DATABASES={"default": {"ENGINE": "django.db.backends.postgresql", "NAME": db_name}},
            INSTALLED_APPS=["django.contrib.contenttypes", "django.contrib.auth"],
            DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        )
        django.setup()


@pytest.fixture(scope="module", name="_pg_available")
def pg_available():
    """Ensure Postgres is available and own setup/teardown of Django tables."""
    from django.conf import settings
    from django.contrib.auth import get_user_model
    from django.db import connection

    db_name = settings.DATABASES["default"]["NAME"]
    try:
        connection.ensure_connection()
    except Exception:
        pytest.skip(f"PostgreSQL not available at {db_name}")

    tables = [
        "auth_user_groups",
        "auth_user_user_permissions",
        "auth_user",
        "auth_group_permissions",
        "auth_group",
        "auth_permission",
        "django_content_type",
    ]
    with connection.cursor() as cur:
        for table in tables:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    with connection.schema_editor() as editor:
        from django.contrib.auth.models import Group, Permission
        from django.contrib.contenttypes.models import ContentType

        editor.create_model(ContentType)
        editor.create_model(Permission)
        editor.create_model(Group)
        editor.create_model(get_user_model())
    try:
        yield
    finally:
        with connection.cursor() as cur:
            for table in tables:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
