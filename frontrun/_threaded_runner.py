from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

from frontrun._dpor_core.worker import WorkerTarget
from frontrun._virtual_clock import real_monotonic

_POST_TIMEOUT_CLEANUP_JOIN_SECONDS = 0.5


class PatchScope:
    """Apply runner patch/unpatch pairs with LIFO teardown."""

    def __init__(self) -> None:
        self._cleanup: list[Callable[[], None]] = []

    def add(
        self,
        patch: Callable[[], Any],
        unpatch: Callable[[], None],
        *,
        enabled: bool = True,
    ) -> None:
        if not enabled:
            return
        patch()
        self._cleanup.append(unpatch)

    def close(self) -> None:
        # Run ALL cleanups in LIFO order even if some raise; otherwise a single
        # failing unpatch would leave the remaining threading primitives patched
        # process-wide.  Collect errors and re-raise the first afterwards.
        errors: list[BaseException] = []
        while self._cleanup:
            try:
                self._cleanup.pop()()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> PatchScope:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


@contextmanager
def instrumentation_scope(
    patches: Sequence[tuple[Callable[[], Any], Callable[[], None], bool]],
):
    """Install instrumentation transactionally after entering the context."""
    with PatchScope() as scope:
        for patch, unpatch, enabled in patches:
            scope.add(patch, unpatch, enabled=enabled)
        yield


def join_threads_with_deadline(
    threads: Sequence[threading.Thread],
    timeout: float | None,
) -> list[threading.Thread]:
    """Join threads against a shared deadline and return any still alive."""
    # This deadline governs the runner itself, not explored code.  In virtual
    # clock mode time.monotonic() is patched and may jump by seconds at zero
    # wall cost, which must not consume the worker join budget.
    deadline = real_monotonic() + timeout if timeout is not None else None
    for thread in threads:
        if deadline is not None:
            remaining = max(0.0, deadline - real_monotonic())
            thread.join(timeout=remaining)
        else:
            thread.join()
    return [thread for thread in threads if thread.is_alive()]


def notify_scheduler_timeout(scheduler: Any, alive: list[threading.Thread]) -> None:
    """Report a thread-group timeout to a scheduler and wake waiters.

    Preserves any pre-existing ``scheduler._error`` (e.g. a DeadlockError that
    explains *why* the threads hung); only sets the generic TimeoutError when no
    error has been recorded yet, mirroring ``report_error``'s first-error-wins
    semantics (finding 9b).
    """
    if scheduler._error is None:
        scheduler._error = TimeoutError(f"Timed out waiting for {len(alive)} thread(s) to complete")
    with scheduler._condition:
        scheduler._condition.notify_all()


class ThreadWorkerSet:
    """Concrete in-process launcher shared by systematic and random runners."""

    def __init__(self, *, name_prefix: str = "dpor", thread_store: list[threading.Thread] | None = None) -> None:
        self.name_prefix = name_prefix
        self.threads = thread_store if thread_store is not None else []

    def launch(
        self,
        targets: Sequence[WorkerTarget],
        *,
        on_partial_start: Callable[[list[threading.Thread]], None] | None = None,
    ) -> list[threading.Thread]:
        threads: list[threading.Thread] = []
        for target in targets:
            if target.func is None:
                raise TypeError("ThreadWorkerSet requires WorkerTarget.func")
            thread = threading.Thread(
                target=target.func,
                args=target.args,
                name=f"{self.name_prefix}-{target.worker_id}",
                daemon=True,
            )
            self.threads.append(thread)
            threads.append(thread)
        started: list[threading.Thread] = []
        try:
            for thread in threads:
                thread.start()
                started.append(thread)
        except BaseException:
            alive = [thread for thread in started if thread.is_alive()]
            if alive and on_partial_start is not None:
                on_partial_start(alive)
            join_threads_with_deadline(alive, _POST_TIMEOUT_CLEANUP_JOIN_SECONDS)
            raise
        return threads

    def join(self, handles: Sequence[threading.Thread], timeout: float) -> list[threading.Thread]:
        return join_threads_with_deadline(handles, timeout)

    def run(
        self,
        targets: Sequence[WorkerTarget],
        *,
        timeout: float,
        on_timeout: Callable[[list[threading.Thread]], None] | None = None,
        teardown: Callable[[], None] | None = None,
    ) -> None:
        try:
            handles = self.launch(targets, on_partial_start=on_timeout)
            alive = self.join(handles, timeout)
            if alive and on_timeout is not None:
                on_timeout(alive)
                self.join(alive, _POST_TIMEOUT_CLEANUP_JOIN_SECONDS)
        finally:
            if teardown is not None:
                teardown()
