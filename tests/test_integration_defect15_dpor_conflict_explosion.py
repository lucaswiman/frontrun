"""Defect #15: DPOR cannot find races in complex check-then-insert patterns.

DPOR finds simple check-then-insert races easily (SELECT + INSERT, 2 ops
per thread).  But when intermediate SQL operations separate the check
from the insert — as happens in real ORMs like django-reversion — the
conflict explosion prevents DPOR from reaching the critical interleaving.
"""

from __future__ import annotations

import os

import pytest

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    pytest.skip("psycopg2 not installed", allow_module_level=True)

from frontrun.cli import require_active

pytestmark = pytest.mark.integration
import frontrun

_DB_NAME = os.environ.get("FRONTRUN_TEST_DB", "frontrun_test")
_DSN = os.environ.get("DATABASE_URL", f"dbname={_DB_NAME}")


@pytest.fixture(scope="module")
def pg_tables():
    """Create test tables mimicking a simplified ORM pattern."""
    try:
        conn = psycopg2.connect(_DSN)
    except Exception:
        pytest.skip(f"Postgres not available at {_DSN}")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS defect15_versions CASCADE")
        cur.execute("DROP TABLE IF EXISTS defect15_revisions CASCADE")
        cur.execute("DROP TABLE IF EXISTS defect15_articles CASCADE")
        cur.execute("CREATE TABLE defect15_articles (  id SERIAL PRIMARY KEY,  content TEXT NOT NULL)")
        cur.execute("CREATE TABLE defect15_revisions (  id SERIAL PRIMARY KEY,  comment TEXT NOT NULL)")
        cur.execute(
            "CREATE TABLE defect15_versions ("
            "  id SERIAL PRIMARY KEY,"
            "  revision_id INTEGER REFERENCES defect15_revisions(id),"
            "  article_id INTEGER REFERENCES defect15_articles(id),"
            "  content TEXT NOT NULL"
            ")"
        )
    conn.close()
    yield
    conn = psycopg2.connect(_DSN)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS defect15_versions CASCADE")
        cur.execute("DROP TABLE IF EXISTS defect15_revisions CASCADE")
        cur.execute("DROP TABLE IF EXISTS defect15_articles CASCADE")
    conn.close()


class _State:
    def __init__(self) -> None:
        conn = psycopg2.connect(_DSN)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("DELETE FROM defect15_versions")
            cur.execute("DELETE FROM defect15_revisions")
            cur.execute("DELETE FROM defect15_articles")
            # Reset sequences so auto-increment IDs don't collide with seed data
            cur.execute("ALTER SEQUENCE defect15_articles_id_seq RESTART WITH 100")
            cur.execute("ALTER SEQUENCE defect15_revisions_id_seq RESTART WITH 100")
            cur.execute("ALTER SEQUENCE defect15_versions_id_seq RESTART WITH 100")
            cur.execute("INSERT INTO defect15_articles (id, content) VALUES (1, 'initial')")
            # Seed version (like django-reversion's initial revision)
            cur.execute("INSERT INTO defect15_revisions (id, comment) VALUES (1, 'seed')")
            cur.execute("INSERT INTO defect15_versions (revision_id, article_id, content) VALUES (1, 1, 'initial')")
        conn.close()
        self.results: list[str | None] = [None, None]


def _make_orm_style_thread(idx: int):
    """Mimics django-reversion's create_revision() pattern.

    The SQL flow mirrors what the ORM generates:
    1. SELECT article (model.objects.get)
    2. SELECT article again (model validation / related lookups)
    3. SELECT version for dedup check (ignore_duplicates)
    4. INSERT revision
    5. INSERT version

    The race: both threads' dedup check (step 3) sees no matching
    'updated' version because neither has inserted yet.  Both proceed
    to INSERT, creating duplicate versions.

    Note: We use SELECT (not UPDATE) for the article operations to
    avoid PostgreSQL row locks that would serialise the two threads
    and make the race impossible at the database level.  The
    intermediate SELECTs still create DPOR conflict points on the
    articles table that exercise the same code path.
    """

    def thread_fn(state: _State) -> None:
        conn = psycopg2.connect(_DSN)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                # 1. SELECT article (ORM .get())
                cur.execute("SELECT id, content FROM defect15_articles WHERE id = 1")
                cur.fetchone()

                # 2. SELECT article again (related model load / validation)
                cur.execute("SELECT content FROM defect15_articles WHERE id = 1")
                cur.fetchone()

                # 3. Dedup check (ignore_duplicates SELECT)
                cur.execute("SELECT id FROM defect15_versions WHERE article_id = 1 ORDER BY id DESC LIMIT 1")
                latest = cur.fetchone()
                if latest is not None:
                    cur.execute(
                        "SELECT content FROM defect15_versions WHERE id = %s",
                        (latest[0],),
                    )
                    row = cur.fetchone()
                    if row and row[0] == "updated":
                        # Duplicate — skip (ignore_duplicates behavior)
                        conn.commit()
                        state.results[idx] = "skipped"
                        return

                # 4. INSERT revision
                cur.execute(
                    "INSERT INTO defect15_revisions (comment) VALUES (%s) RETURNING id",
                    (f"thread_{idx}",),
                )
                rev_id = cur.fetchone()[0]

                # 5. INSERT version
                cur.execute(
                    "INSERT INTO defect15_versions (revision_id, article_id, content) VALUES (%s, 1, %s)",
                    (rev_id, "updated"),
                )

            conn.commit()
            state.results[idx] = "inserted"
        except Exception as exc:
            conn.rollback()
            state.results[idx] = f"error: {type(exc).__name__}: {exc}"
        finally:
            conn.close()

    return thread_fn


def _invariant(state: _State) -> bool:
    """At most 2 versions should exist (seed + one update)."""
    conn = psycopg2.connect(_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM defect15_versions WHERE article_id = 1")
        count = cur.fetchone()[0]
    conn.close()
    return count <= 2


def test_simple_check_then_insert_is_found(pg_tables) -> None:
    """Baseline: DPOR finds the race with a simple SELECT + INSERT pattern."""
    require_active("test_simple_check_then_insert_is_found")

    def make_simple_thread(idx: int):
        def thread_fn(state: _State) -> None:
            conn = psycopg2.connect(_DSN)
            conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM defect15_versions WHERE article_id = 1 AND content = 'updated'")
                    if cur.fetchone() is not None:
                        conn.rollback()
                        state.results[idx] = "skipped"
                        return
                    cur.execute(
                        "INSERT INTO defect15_revisions (comment) VALUES (%s) RETURNING id",
                        (f"thread_{idx}",),
                    )
                    rev_id = cur.fetchone()[0]
                    cur.execute(
                        "INSERT INTO defect15_versions (revision_id, article_id, content) VALUES (%s, 1, %s)",
                        (rev_id, "updated"),
                    )
                conn.commit()
                state.results[idx] = "inserted"
            except Exception as exc:
                conn.rollback()
                state.results[idx] = f"error: {type(exc).__name__}: {exc}"
            finally:
                conn.close()

        return thread_fn

    result = frontrun.explore(
        setup=_State,
        workers=[make_simple_thread(0), make_simple_thread(1)],
        invariant=_invariant,
        detect_io=True,
        timeout_per_run=10.0,
        deadlock_timeout=10.0,
        max_executions=50,
        preemption_bound=None,
        reproduce_on_failure=0,
    )

    assert not result.property_holds, (
        f"Expected DPOR to find the simple check-then-insert race. num_explored={result.num_explored}"
    )


def test_orm_style_check_then_insert_race(pg_tables) -> None:
    """ORM-style pattern with intermediate ops between check and insert.

    Same race as the simple case, but with SELECT article operations
    between transaction start and the dedup check.  These extra
    operations on a second table create additional DPOR conflict points.
    With resource group-aware deferred notdep processing (Defect #15
    Approach 2: Operation Coalescing for Unrelated Tables), DPOR skips
    inline wakeup insertion for SQL resources and relies on deferred
    notdep sequences that naturally capture cross-table independence,
    finding the critical interleaving efficiently.
    """
    require_active("test_orm_style_check_then_insert_race")

    result = frontrun.explore(
        setup=_State,
        workers=[_make_orm_style_thread(0), _make_orm_style_thread(1)],
        invariant=_invariant,
        detect_io=True,
        timeout_per_run=10.0,
        deadlock_timeout=10.0,
        max_executions=100,
        preemption_bound=None,
        reproduce_on_failure=0,
    )

    assert not result.property_holds, (
        f"DPOR should find the ORM-style check-then-insert race but "
        f"explored {result.num_explored} interleavings without finding it. "
        f"The intermediate SELECT+UPDATE on articles create conflict "
        f"points that prevent reaching the critical interleaving."
    )
