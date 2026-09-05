"""CPython-compatible argument validation for cooperative locks."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from frontrun._cooperative import CooperativeLock, CooperativeRLock


@pytest.mark.parametrize("lock_factory", [CooperativeLock, CooperativeRLock])
@pytest.mark.parametrize(
    "acquire",
    [
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


@pytest.mark.parametrize("lock_factory", [CooperativeLock, CooperativeRLock])
def test_default_timeout_is_valid_for_nonblocking_lock_acquire(
    lock_factory: Callable[[], CooperativeLock | CooperativeRLock],
) -> None:
    """CPython accepts its -1 timeout sentinel with blocking=False."""
    assert lock_factory().acquire(blocking=False, timeout=-1)


def test_semaphore_rejects_nonblocking_timeout() -> None:
    from frontrun._cooperative import CooperativeSemaphore

    with pytest.raises(ValueError):
        CooperativeSemaphore().acquire(blocking=False, timeout=0)
