"""Finding 9b/9e: PatchScope.close robustness and scheduler error preservation."""

from __future__ import annotations

import threading

import pytest

from frontrun._real_threading import condition as _real_condition
from frontrun._real_threading import lock as _real_lock
from frontrun._threaded_runner import PatchScope, instrumentation_scope, notify_scheduler_timeout, run_thread_group


def test_patch_scope_runs_all_cleanups_even_if_one_raises():
    """One raising unpatch must not skip the remaining LIFO cleanups (9e).

    Otherwise a failure tearing down one patch leaves threading primitives
    patched process-wide.
    """
    calls: list[str] = []
    scope = PatchScope()

    scope.add(lambda: None, lambda: calls.append("first"))

    def _raises() -> None:
        calls.append("second")
        raise RuntimeError("boom")

    scope.add(lambda: None, _raises)
    scope.add(lambda: None, lambda: calls.append("third"))

    with pytest.raises(RuntimeError, match="boom"):
        scope.close()

    # LIFO order: third, second (raises), first — all must run.
    assert calls == ["third", "second", "first"], calls


class _FakeScheduler:
    def __init__(self) -> None:
        self._error: Exception | None = None
        self._lock = _real_lock()
        self._condition = _real_condition(self._lock)


def test_notify_scheduler_timeout_preserves_first_error():
    """A pre-existing DeadlockError must not be clobbered by the timeout (9b)."""

    class DeadlockError(Exception):
        pass

    sched = _FakeScheduler()
    original = DeadlockError("real cause of the hang")
    sched._error = original

    notify_scheduler_timeout(sched, [])

    assert sched._error is original, f"notify_scheduler_timeout overwrote the first error with {sched._error!r}"


def test_instrumentation_scope_is_lazy_and_rolls_back_partial_install():
    calls: list[str] = []

    def fail() -> None:
        calls.append("patch-sql")
        raise RuntimeError("install failed")

    scope = instrumentation_scope(
        [(lambda: calls.append("patch-io"), lambda: calls.append("unpatch-io")), (fail, lambda: None)]
    )
    assert calls == []

    with pytest.raises(RuntimeError, match="install failed"):
        with scope:
            pass

    assert calls == ["patch-io", "patch-sql", "unpatch-io"]


def test_run_thread_group_only_starts_new_threads():
    calls: list[int] = []
    store: list[threading.Thread] = []

    def make_target(i: int, func, args):
        return lambda: func(*args)

    run_thread_group(
        funcs=[calls.append],
        args=[(1,)],
        make_thread_target=make_target,
        name_prefix="test",
        timeout=1.0,
        thread_store=store,
    )
    run_thread_group(
        funcs=[calls.append],
        args=[(2,)],
        make_thread_target=make_target,
        name_prefix="test",
        timeout=1.0,
        thread_store=store,
    )

    assert calls == [1, 2]
