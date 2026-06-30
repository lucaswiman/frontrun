"""
Frontrun: Deterministic concurrency testing for Python.

Trace markers (sync)::

    from frontrun.common import Schedule, Step
    from frontrun import TraceExecutor

Async trace markers (pass a dict of task names to coroutines)::

    from frontrun import TraceExecutor
    from frontrun.common import Schedule, Step
    executor = TraceExecutor(schedule)
    executor.run({"task1": coro_factory1, "task2": coro_factory2})

Unified exploration entry point (recommended)::

    import frontrun

    # Sync DPOR (default)
    result = frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    result.assert_holds()

    # Async — detected automatically
    async def worker(state): ...
    result = await frontrun.explore(setup=make_state, workers=worker, count=2, invariant=...)

    # Strategy selection
    result = frontrun.explore(..., strategy="dpor")    # default
    result = frontrun.explore(..., strategy="random")

Bytecode (random) exploration::

    import frontrun
    frontrun.explore_random(...)

Async shuffler (random) exploration::

    import frontrun
    await frontrun.explore_async_random(...)

Contrib helpers (use threads= for sync, tasks= for async)::

    from frontrun.contrib.django import django_dpor
    from frontrun.contrib.sqlalchemy import sqlalchemy_dpor, get_connection, get_async_connection
"""

import importlib
from importlib.metadata import version as _metadata_version
from typing import TYPE_CHECKING, Any

from frontrun.common import NondeterministicSQLError
from frontrun.explore import explore
from frontrun.trace_markers import TraceExecutor

if TYPE_CHECKING:
    from frontrun.async_shuffler import explore_async_random as explore_async_random
    from frontrun.bytecode import explore_random as explore_random

try:
    __version__: str = _metadata_version("frontrun")
except Exception:
    __version__ = "0.0.0"


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "explore_random": ("frontrun.bytecode", "explore_random"),
    "explore_async_random": ("frontrun.async_shuffler", "explore_async_random"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module 'frontrun' has no attribute {name!r}")


__all__ = [
    "NondeterministicSQLError",
    "TraceExecutor",
    "__version__",
    "explore",
    "explore_async_random",
    "explore_random",
]
