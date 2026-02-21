"""Pytest plugin that patches threading primitives before test collection.

Register via the ``pytest11`` entry point so that ``patch_locks()`` runs
in ``pytest_configure`` — *before* pytest imports any test modules or
conftest files.  This ensures that libraries which create module-level
``threading.Lock()`` instances at import time get cooperative versions
instead of real ones.

Patching is **on by default**.  To disable::

    pytest --no-frontrun-patch-locks
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("frontrun", "Frontrun concurrency testing")
    group.addoption(
        "--no-frontrun-patch-locks",
        action="store_true",
        default=False,
        help="Disable automatic cooperative lock patching (enabled by default).",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--no-frontrun-patch-locks", default=False):
        return

    from frontrun._cooperative import patch_locks

    patch_locks()


def pytest_unconfigure(config: pytest.Config) -> None:
    if config.getoption("--no-frontrun-patch-locks", default=False):
        return

    from frontrun._cooperative import unpatch_locks

    unpatch_locks()
