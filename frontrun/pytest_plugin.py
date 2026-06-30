"""Pytest plugin for frontrun concurrency testing.

Register via the ``pytest11`` entry point.  Two key behaviors:

1. **Cooperative lock patching** — replaces ``threading.Lock``,
   ``queue.Queue``, etc. with cooperative versions that yield scheduler
   turns instead of blocking in C.  This is required for bytecode-level
   and DPOR exploration.

   Patching is **on by default only when running under the ``frontrun``
   CLI** (i.e. ``FRONTRUN_ACTIVE=1``).  When running plain ``pytest``
   without the CLI wrapper, patching is **off** unless explicitly
   requested with ``--frontrun-patch-locks``.

2. **Skip guard** — ``frontrun.explore`` and related entry points
   raise ``pytest.skip`` when called outside the ``frontrun`` CLI
   environment.  This prevents confusing failures when tests are run
   without the required monkey-patching and LD_PRELOAD setup.

Usage::

    # Preferred: run pytest through the frontrun CLI
    frontrun pytest -vv tests/

    # Or explicitly enable patching without the CLI wrapper
    pytest --frontrun-patch-locks
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("frontrun", "Frontrun concurrency testing")
    group.addoption(
        "--frontrun-patch-locks",
        action="store_true",
        default=False,
        help="Enable cooperative lock patching (auto-enabled under `frontrun` CLI).",
    )
    group.addoption(
        "--no-frontrun-patch-locks",
        action="store_true",
        default=False,
        help="Disable cooperative lock patching even when running under `frontrun` CLI.",
    )
    group.addoption(
        "--frontrun-report",
        default=None,
        metavar="PATH",
        help="Generate interactive HTML report of DPOR exploration at PATH.",
    )


def _should_patch(config: pytest.Config) -> bool:
    """Determine whether to apply cooperative lock patching."""
    # Explicit disable always wins
    if config.getoption("--no-frontrun-patch-locks", default=False):
        return False

    # Explicit enable
    if config.getoption("--frontrun-patch-locks", default=False):
        return True

    # Auto-enable when running under the frontrun CLI
    from frontrun.cli import is_active

    return is_active()


_PATCHED_ATTR = "_frontrun_patched"


def pytest_configure(config: pytest.Config) -> None:
    # Set up report path if requested
    report_path = config.getoption("--frontrun-report", default=None)
    if report_path:
        import frontrun._report

        frontrun._report._global_report_path = report_path

    if not _should_patch(config):
        return

    # Warm SQL parsers BEFORE patching locks.  sqlglot.dialects uses a
    # module-level _import_lock = threading.RLock() for lazy dialect loading.
    # If patch_locks() runs first, that lock becomes a CooperativeRLock which
    # can deadlock during counterexample reproduction (no scheduler context).
    from frontrun._sql_cursor import _warm_sql_parsers

    _warm_sql_parsers()

    from frontrun._cooperative import patch_locks

    patch_locks()
    # Record that we patched, so pytest_unconfigure can unpatch even if
    # the environment changes during the test session (e.g. a test clears
    # FRONTRUN_ACTIVE).
    config._frontrun_patched = True  # type: ignore[attr-defined]


def pytest_unconfigure(config: pytest.Config) -> None:
    # Clear report path
    import frontrun._report

    frontrun._report._global_report_path = None

    if not getattr(config, "_frontrun_patched", False):
        return

    from frontrun._cooperative import unpatch_locks

    unpatch_locks()
