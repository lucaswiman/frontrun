"""Shared keyword-only constructor for :class:`PyDporEngine`."""

from __future__ import annotations

from typing import Any

try:
    from frontrun._dpor import PyDporEngine  # type: ignore[reportAttributeAccessIssue]
except ModuleNotFoundError as _err:
    raise ModuleNotFoundError(
        "frontrun._dpor_core requires the frontrun._dpor Rust extension.\n"
        "Build it with:  make build-dpor-3.14   (or build-dpor-3.10 / build-dpor-3.14t)\n"
        "Or install from source:  pip install -e ."
    ) from _err


def make_dpor_engine(
    *,
    num_threads: int,
    preemption_bound: int | None,
    max_branches: int,
    max_executions: int | None,
    search: str | None = None,
) -> Any:
    if max_executions is not None and max_executions <= 0:
        raise ValueError("max_executions must be positive or None")
    if max_branches <= 0:
        raise ValueError("max_branches must be positive")
    if preemption_bound is not None and preemption_bound < 0:
        raise ValueError("preemption_bound must be non-negative or None")
    return PyDporEngine(
        num_threads=num_threads,
        preemption_bound=preemption_bound,
        max_branches=max_branches,
        max_executions=max_executions,
        search=search,
    )
