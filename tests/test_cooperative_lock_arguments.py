"""CPython-compatible argument validation for cooperative locks."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from frontrun._cooperative import CooperativeLock, CooperativeRLock


@pytest.mark.parametrize("lock_factory", [CooperativeLock, CooperativeRLock])
@pytest.mark.parametrize(
    "acquire",
    [
        lambda lock: lock.acquire(blocking=False, timeout=-1),
        lambda lock: lock.acquire(blocking=False, timeout=0),
        lambda lock: lock.acquire(timeout=-2),
    ],
)
def test_invalid_acquire_argument_combinations_raise(
    lock_factory: Callable[[], CooperativeLock | CooperativeRLock],
    acquire: Callable[[CooperativeLock | CooperativeRLock], bool],
) -> None:
    """Cooperative wrappers reject every combination rejected by CPython."""
    with pytest.raises(ValueError):
        acquire(lock_factory())
