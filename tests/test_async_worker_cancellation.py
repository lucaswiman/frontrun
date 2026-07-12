"""Regression tests for user-initiated async worker cancellation."""

from __future__ import annotations

import asyncio

import pytest

import frontrun
from frontrun.async_shuffler import run_with_schedule


async def _self_cancel(state: list[str]) -> None:
    state.append("partial")
    raise asyncio.CancelledError("worker cancelled itself")


@pytest.mark.parametrize("strategy", ["dpor", "random"])
def test_self_cancelled_worker_is_not_a_successful_exploration(strategy: str) -> None:
    """A cancelled worker leaves partial state and cannot prove the property."""
    options: dict[str, object]
    if strategy == "dpor":
        options = {"max_executions": 1, "reproduce_on_failure": 0, "detect_io": False}
    else:
        options = {"max_attempts": 1, "max_ops": 2, "seed": 1}

    result = asyncio.run(
        frontrun.explore(
            setup=list,
            workers=[_self_cancel],
            invariant=lambda state: state == ["partial"],
            strategy=strategy,
            **options,
        )
    )

    assert not result.property_holds
    assert result.explanation is not None
    assert "cancel" in result.explanation.lower()


def test_run_with_schedule_propagates_worker_cancellation() -> None:
    """Exact replay must not return state from a cancelled worker."""

    async def replay() -> None:
        with pytest.raises(asyncio.CancelledError, match="worker cancelled itself"):
            await run_with_schedule([0, 0], list, [_self_cancel])

    asyncio.run(replay())
