"""Pytest plugin that patches threading primitives before test collection.

Register via the ``pytest11`` entry point so that ``patch_locks()`` runs
in ``pytest_configure`` — *before* pytest imports any test modules or
conftest files.  This ensures that libraries which create module-level
``threading.Lock()`` instances at import time get cooperative versions
instead of real ones.

Usage::

    # Just install frontrun — the plugin auto-registers.
    # Then enable early patching with the CLI flag:
    pytest --frontrun-patch-locks            # same as =monkey
    pytest --frontrun-patch-locks=monkey     # monkeypatch threading.Lock etc.
    pytest --frontrun-patch-locks=aggressive # also gc-scan for pre-existing instances
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_MODES = ("monkey", "aggressive")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("frontrun", "Frontrun concurrency testing")
    group.addoption(
        "--frontrun-patch-locks",
        nargs="?",
        const="monkey",
        default=None,
        choices=_MODES,
        help="Patch threading/queue primitives with cooperative versions before test collection. "
        "'monkey' (default) replaces the classes so new Lock() calls return cooperative locks. "
        "'aggressive' additionally walks gc.get_objects() to replace pre-existing lock instances.",
    )


def pytest_configure(config: pytest.Config) -> None:
    mode = config.getoption("--frontrun-patch-locks", default=None)
    if mode is None:
        return

    from frontrun._cooperative import patch_locks

    patch_locks()

    if mode == "aggressive":
        _replace_preexisting_locks()


def pytest_unconfigure(config: pytest.Config) -> None:
    mode = config.getoption("--frontrun-patch-locks", default=None)
    if mode is None:
        return

    from frontrun._cooperative import unpatch_locks

    unpatch_locks()


def _replace_preexisting_locks() -> None:
    """Walk ``gc.get_objects()`` and replace real Lock/RLock instances.

    For each real ``threading.Lock`` (``_thread.lock``) or
    ``threading.RLock`` (``_thread.RLock``) instance found, we look at
    its referrers to find mutable containers (dicts, lists, objects with
    ``__dict__``) and swap the reference to a cooperative wrapper that
    delegates to the original lock.

    This is inherently best-effort — locks stored in tuples, C structs,
    or closures cannot be replaced.  But it covers the common case of
    ``_lock = threading.Lock()`` stored as a module or instance attribute.
    """
    import _thread

    from frontrun._cooperative import CooperativeLock, CooperativeRLock

    # Identify the C types for real locks
    real_lock_type = type(_thread.allocate_lock())
    real_rlock_type = type(_thread.RLock())

    # Collect all real lock instances first (snapshot to avoid mutating
    # the object list while iterating).
    real_locks: list[object] = [
        obj for obj in gc.get_objects() if type(obj) is real_lock_type or type(obj) is real_rlock_type
    ]

    replaced = 0
    for lock in real_locks:
        is_rlock = type(lock) is real_rlock_type
        # Build a cooperative wrapper around the existing real lock
        if is_rlock:
            wrapper = CooperativeRLock.__new__(CooperativeRLock)
            wrapper._lock = lock  # type: ignore[attr-defined]
            wrapper._object_id = id(wrapper)  # type: ignore[attr-defined]
            wrapper._owner_thread_id = None  # type: ignore[attr-defined]
        else:
            wrapper = CooperativeLock.__new__(CooperativeLock)
            wrapper._lock = lock  # type: ignore[attr-defined]
            wrapper._object_id = id(wrapper)  # type: ignore[attr-defined]
            wrapper._owner_thread_id = None  # type: ignore[attr-defined]

        # Walk referrers and replace the reference
        for referrer in gc.get_referrers(lock):
            # Skip our own locals / the real_locks list
            if referrer is real_locks:
                continue
            # Skip frontrun's own internal real-lock references
            if _is_frontrun_internal(referrer):
                continue
            _try_replace(referrer, lock, wrapper)
            replaced += 1

    if replaced:
        # Collect to clean up any orphaned real lock refs
        gc.collect()


def _is_frontrun_internal(referrer: object) -> bool:
    """Return True if *referrer* is a frontrun-internal dict we shouldn't touch."""
    if not isinstance(referrer, dict):
        return False
    # Module __dict__ for frontrun internals
    mod_name = referrer.get("__name__", "")
    if isinstance(mod_name, str) and mod_name.startswith("frontrun"):
        return True
    # The _real_threading module's dict
    if referrer.get("__file__", ""):
        file = referrer["__file__"]
        if isinstance(file, str) and "_real_threading" in file:
            return True
    return False


def _try_replace(container: object, old: object, new: object) -> None:
    """Best-effort replacement of *old* with *new* inside *container*."""
    if isinstance(container, dict):
        for key, value in list(container.items()):
            if value is old:
                try:
                    container[key] = new
                except TypeError:
                    pass  # frozen / immutable dict
    elif isinstance(container, list):
        for i, value in enumerate(container):
            if value is old:
                container[i] = new
    elif hasattr(container, "__dict__"):
        d = container.__dict__
        if not isinstance(d, dict):  # type: ignore[reportUnnecessaryIsInstance]  # __dict__ may be mappingproxy at runtime
            return
        for key, value in list(d.items()):
            if value is old:
                try:
                    setattr(container, key, new)
                except (AttributeError, TypeError):
                    pass  # read-only attribute
