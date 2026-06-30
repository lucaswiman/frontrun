"""In-process (OS-thread) implementation of the ``WorkerSet`` port.

Thin wrapper over :func:`frontrun._threaded_runner.run_thread_group` that
adapts it to the backend-agnostic :class:`frontrun._dpor_core.worker.WorkerSet`
contract. The planned cross-process backend supplies a subprocess-based
implementation of the same Protocol.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from frontrun._dpor_core.worker import WorkerTarget
from frontrun._threaded_runner import run_thread_group


class ThreadWorkerSet:
    """Run each :class:`WorkerTarget` on its own daemon ``threading.Thread``."""

    def __init__(
        self,
        *,
        name_prefix: str = "dpor",
        thread_store: list[threading.Thread] | None = None,
    ) -> None:
        self.name_prefix = name_prefix
        # The runner exposes the live threads for inspection/timeout handling,
        # so allow it to share its own list as the backing store.
        self.threads: list[threading.Thread] = thread_store if thread_store is not None else []

    def run(
        self,
        targets: Sequence[WorkerTarget],
        run_one: Callable[[WorkerTarget], None],
        *,
        timeout: float,
        on_timeout: Callable[[list[Any]], None] | None = None,
        teardown: Callable[[], None] | None = None,
    ) -> None:
        targets = list(targets)
        funcs = [t.func for t in targets]
        args = [t.args for t in targets]

        def make_thread_target(
            index: int,
            func: Callable[..., None],
            thread_args: tuple[Any, ...],
        ) -> Callable[[], None]:
            target = targets[index]

            def body() -> None:
                run_one(target)

            return body

        run_thread_group(
            funcs=funcs,
            args=args,
            make_thread_target=make_thread_target,
            name_prefix=self.name_prefix,
            timeout=timeout,
            thread_store=self.threads,
            teardown=teardown,
            on_timeout=on_timeout,
        )
