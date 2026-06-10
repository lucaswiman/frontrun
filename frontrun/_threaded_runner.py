from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any


class PatchScope:
    """Apply runner patch/unpatch pairs with LIFO teardown."""

    def __init__(self) -> None:
        self._cleanup: list[Callable[[], None]] = []

    def add(
        self,
        patch: Callable[[], None],
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


def join_threads_with_deadline(
    threads: Sequence[threading.Thread],
    timeout: float | None,
) -> list[threading.Thread]:
    """Join threads against a shared deadline and return any still alive."""
    deadline = time.monotonic() + timeout if timeout is not None else None
    for thread in threads:
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
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
    for thread in alive:
        thread.join(timeout=0.5)


def run_thread_group(
    *,
    funcs: Sequence[Callable[..., None]],
    args: Sequence[tuple[Any, ...]],
    make_thread_target: Callable[[int, Callable[..., None], tuple[Any, ...]], Callable[[], None]],
    name_prefix: str,
    timeout: float,
    thread_store: list[threading.Thread],
    setup: Callable[[], None] | None = None,
    teardown: Callable[[], None] | None = None,
    on_timeout: Callable[[list[threading.Thread]], None] | None = None,
) -> list[threading.Thread]:
    """Start a thread group, join it against a deadline, and return alive threads."""
    if setup is not None:
        setup()
    try:
        for i, (func, thread_args) in enumerate(zip(funcs, args)):
            thread = threading.Thread(
                target=make_thread_target(i, func, thread_args),
                name=f"{name_prefix}-{i}",
                daemon=True,
            )
            thread_store.append(thread)

        for thread in thread_store:
            thread.start()

        alive = join_threads_with_deadline(thread_store, timeout)
        if alive and on_timeout is not None:
            on_timeout(alive)
        return alive
    finally:
        if teardown is not None:
            teardown()
