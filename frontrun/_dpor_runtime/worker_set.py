"""In-process (OS-thread) implementation of the ``WorkerSet`` port.

This backend runs each :class:`frontrun._dpor_core.worker.WorkerTarget` as a
daemon ``threading.Thread``. Cross-process backends implement the same
``launch`` / ``join`` port with process handles.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from frontrun._dpor_core.worker import WorkerTarget
from frontrun._threaded_runner import join_threads_with_deadline


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

    def launch(self, targets: Sequence[WorkerTarget]) -> list[threading.Thread]:
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
        for thread in threads:
            thread.start()
        return threads

    def join(self, handles: Any, timeout: float) -> list[threading.Thread]:
        return join_threads_with_deadline(handles, timeout)

    def run(
        self,
        targets: Sequence[WorkerTarget],
        *,
        timeout: float,
        on_timeout: Callable[[list[Any]], None] | None = None,
        teardown: Callable[[], None] | None = None,
    ) -> None:
        try:
            # launch() must be inside the try: if it raises partway (e.g.
            # thread.start() hits "can't start new thread"), teardown — the
            # runner's opcode-tracer uninstall — must still run, or the
            # process-wide tracer leaks and mis-traces every later execution.
            handles = self.launch(targets)
            alive = self.join(handles, timeout)
            if alive and on_timeout is not None:
                on_timeout(alive)
        finally:
            if teardown is not None:
                teardown()
