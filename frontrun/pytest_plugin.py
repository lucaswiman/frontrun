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

2. **Skip guard** — ``explore_interleavings`` and ``explore_dpor``
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


def pytest_configure(config: pytest.Config) -> None:
    if not _should_patch(config):
        return

    from frontrun._cooperative import patch_locks

    patch_locks()


def pytest_unconfigure(config: pytest.Config) -> None:
    if not _should_patch(config):
        return

    from frontrun._cooperative import unpatch_locks

    unpatch_locks()
