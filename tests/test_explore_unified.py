"""Tests for the unified frontrun.explore() entry point and related API changes.

Covers:
  (a) explore() dispatcher — sync DPOR path
  (b) explore() dispatcher — async path
  (c) workers=fn, count=N shorthand
  (d) AssertionError in invariant → explanation
  (f) detect_io in async DPOR covers Redis (detect_redis)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

import frontrun
from frontrun.common import InterleavingResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@dataclass
class Counter:
    value: int = 0

    def increment(self) -> None:
        v = self.value
        # No artificial yield — DPOR explores all bytecode interleavings
        self.value = v + 1


def counter_invariant(c: Counter) -> bool:
    return c.value == 2


# ---------------------------------------------------------------------------
# (a) explore() dispatcher — sync DPOR path
# ---------------------------------------------------------------------------


def test_explore_sync_dpor_finds_race():
    """explore() with sync workers uses DPOR and finds the lost-update race."""
    result = frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=counter_invariant,
    )
    assert isinstance(result, InterleavingResult)
    # Counter with no locking has a race; DPOR must find it
    assert not result.property_holds
    assert result.explanation is not None


def test_explore_sync_dpor_passes_for_correct_code():
    """explore() reports property_holds=True when there is no race."""

    @dataclass
    class LockedCounter:
        import threading

        value: int = 0
        _lock: object = field(default_factory=lambda: __import__("threading").Lock())

        def increment(self) -> None:
            with self._lock:  # type: ignore[attr-defined]
                self.value += 1

    result = frontrun.explore(
        setup=LockedCounter,
        workers=LockedCounter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    assert isinstance(result, InterleavingResult)
    assert result.property_holds


def test_explore_rejects_deferred_setup_body() -> None:
    """A sync exploration must not certify without executing async setup."""

    async def setup() -> Counter:
        return Counter()

    with pytest.raises(TypeError, match="setup.*deferred body was not executed"):
        frontrun.explore(
            setup=setup,
            workers=[lambda _state: None],
            invariant=lambda _state: True,
            strategy="dpor",
            max_executions=1,
        )


def test_explore_rejects_deferred_invariant_body() -> None:
    """An async invariant result must not be treated as truthy without awaiting."""

    async def invariant(_state: Counter) -> bool:
        return False

    with pytest.raises(TypeError, match="invariant.*deferred body was not executed"):
        frontrun.explore(
            setup=Counter,
            workers=[lambda _state: None],
            invariant=invariant,
            strategy="dpor",
            max_executions=1,
        )


def test_explore_sync_random_strategy():
    """explore(strategy='random') finds the lost-update race."""
    result = frontrun.explore(
        setup=Counter,
        workers=[Counter.increment, Counter.increment],
        invariant=counter_invariant,
        strategy="random",
        max_attempts=200,
        seed=42,
    )
    assert isinstance(result, InterleavingResult)
    assert not result.property_holds


def test_explore_unknown_strategy_raises():
    """Unknown strategy value raises ValueError."""
    with pytest.raises(ValueError, match="unknown strategy"):
        frontrun.explore(
            setup=Counter,
            workers=[Counter.increment],
            invariant=counter_invariant,
            strategy="bananas",
        )


def test_explore_preemption_bound_none_passthrough(monkeypatch):
    """explore(preemption_bound=None) should pass None through to _explore_dpor."""
    from frontrun._dpor_runtime import explore as explore_mod

    captured: dict = {}
    original = explore_mod._explore_dpor

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(explore_mod, "_explore_dpor", spy)
    frontrun.explore(
        setup=Counter,
        workers=[Counter.increment, Counter.increment],
        invariant=counter_invariant,
        preemption_bound=None,
    )
    assert "preemption_bound" in captured, "preemption_bound=None should be forwarded, not stripped"
    assert captured["preemption_bound"] is None


# ---------------------------------------------------------------------------
# (b) explore() dispatcher — async path
# ---------------------------------------------------------------------------


@dataclass
class AsyncCounter:
    value: int = 0

    async def increment(self) -> None:
        v = self.value
        await asyncio.sleep(0)  # yield to scheduler
        self.value = v + 1


def test_explore_async_returns_coroutine():
    """explore() with async workers returns a coroutine (not an InterleavingResult)."""
    import inspect

    coro = frontrun.explore(
        setup=AsyncCounter,
        workers=AsyncCounter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
        strategy="dpor",
    )
    assert inspect.iscoroutine(coro), "expected a coroutine for async workers"
    coro.close()  # avoid ResourceWarning


def test_explore_async_dpor_finds_race():
    """explore() with async workers (DPOR) finds the lost-update race."""
    result = asyncio.run(
        frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=lambda c: c.value == 2,
            strategy="dpor",
        )
    )
    assert not result.property_holds


def test_explore_async_random_finds_race():
    """explore() with async workers (random) finds the lost-update race."""
    result = asyncio.run(
        frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=lambda c: c.value == 2,
            strategy="random",
            max_attempts=200,
            seed=42,
        )
    )
    assert not result.property_holds


def test_explore_routes_async_callable_instances_to_async():
    """Workers passed as class instances with an async ``__call__`` must route
    to the async engine.

    ``any_async`` used ``inspect.iscoroutinefunction`` directly, which returns
    False for a callable object whose ``__call__`` is async, so such workers
    were misrouted to the synchronous strategy and their coroutines were never
    awaited (a silent false ``property_holds=True``).
    """
    import inspect

    class AsyncWorker:
        async def __call__(self, c: AsyncCounter) -> None:
            v = c.value
            await asyncio.sleep(0)  # yield to scheduler
            c.value = v + 1

    coro = frontrun.explore(
        setup=AsyncCounter,
        workers=[AsyncWorker(), AsyncWorker()],
        invariant=lambda c: c.value == 2,
        strategy="dpor",
    )
    assert inspect.iscoroutine(coro), "async callable instances must route to the async engine"
    result = asyncio.run(coro)
    assert not result.property_holds


# ---------------------------------------------------------------------------
# (c) workers=fn, count=N shorthand
# ---------------------------------------------------------------------------


def test_count_shorthand_expands_workers():
    """workers=fn + count=N is equivalent to workers=[fn, fn, ..., fn]."""
    result = frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=counter_invariant,
    )
    assert isinstance(result, InterleavingResult)
    # Same as passing [Counter.increment, Counter.increment] — race exists
    assert not result.property_holds


def test_count_shorthand_count_one():
    """count=1 with a single callable works (trivial, no races)."""
    result = frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=1,
        invariant=lambda c: c.value == 1,
    )
    assert result.property_holds


def test_count_with_list_raises():
    """Providing count AND a list raises ValueError."""
    with pytest.raises(ValueError, match="'count' cannot be used"):
        frontrun.explore(
            setup=Counter,
            workers=[Counter.increment, Counter.increment],
            invariant=counter_invariant,
            count=2,
        )


def test_count_zero_raises():
    """count=0 raises ValueError."""
    with pytest.raises(ValueError, match="count must be a positive integer"):
        frontrun.explore(
            setup=Counter,
            workers=Counter.increment,
            invariant=counter_invariant,
            count=0,
        )


def test_count_negative_raises():
    """count=-1 raises ValueError."""
    with pytest.raises(ValueError, match="count must be a positive integer"):
        frontrun.explore(
            setup=Counter,
            workers=Counter.increment,
            invariant=counter_invariant,
            count=-1,
        )


def test_workers_list_without_count():
    """workers as a plain list works (no count needed)."""
    result = frontrun.explore(
        setup=Counter,
        workers=[Counter.increment, Counter.increment],
        invariant=counter_invariant,
    )
    assert isinstance(result, InterleavingResult)


def test_workers_tuple_without_count():
    """workers as a tuple works (no count needed)."""
    result = frontrun.explore(
        setup=Counter,
        workers=(Counter.increment, Counter.increment),
        invariant=counter_invariant,
    )
    assert isinstance(result, InterleavingResult)


# ---------------------------------------------------------------------------
# (d) AssertionError in invariant → explanation
# ---------------------------------------------------------------------------


def assert_invariant_with_message(c: Counter) -> bool:
    assert c.value == 2, f"expected 2, got {c.value}"
    return True


def test_assertion_error_in_invariant_dpor():
    """AssertionError in invariant is treated as failure; message in explanation."""
    result = frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=assert_invariant_with_message,
    )
    assert not result.property_holds
    assert result.explanation is not None
    assert "AssertionError" in result.explanation
    assert "expected 2" in result.explanation


def test_assertion_error_in_invariant_random():
    """AssertionError in invariant (random strategy) is treated as failure."""
    result = frontrun.explore(
        setup=Counter,
        workers=Counter.increment,
        count=2,
        invariant=assert_invariant_with_message,
        strategy="random",
        max_attempts=200,
        seed=42,
    )
    assert not result.property_holds
    assert result.explanation is not None
    assert "AssertionError" in result.explanation


def test_assertion_error_async_dpor():
    """AssertionError in async DPOR invariant is treated as failure."""

    def assert_inv(c: AsyncCounter) -> bool:
        assert c.value == 2, f"async: expected 2, got {c.value}"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=assert_inv,
            strategy="dpor",
        )
    )
    assert not result.property_holds
    assert result.explanation is not None
    assert "AssertionError" in result.explanation


def test_assertion_error_async_random():
    """AssertionError in async random invariant is treated as failure."""

    def assert_inv(c: AsyncCounter) -> bool:
        assert c.value == 2, f"async-random: expected 2, got {c.value}"
        return True

    result = asyncio.run(
        frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=assert_inv,
            strategy="random",
            max_attempts=200,
            seed=42,
        )
    )
    assert not result.property_holds
    assert result.explanation is not None
    assert "AssertionError" in result.explanation


# ---------------------------------------------------------------------------
# (e) Canonical random APIs work
# ---------------------------------------------------------------------------


def test_explore_random_works():
    """explore_random (canonical name) works without warning."""
    result = frontrun.explore_random(
        setup=Counter,
        threads=[Counter.increment, Counter.increment],
        invariant=counter_invariant,
        max_attempts=100,
        seed=42,
    )
    assert isinstance(result, InterleavingResult)
    assert not result.property_holds


def test_explore_async_random_works():
    """explore_async_random (canonical name) works without warning."""
    result = asyncio.run(
        frontrun.explore_async_random(
            setup=AsyncCounter,
            tasks=[AsyncCounter.increment, AsyncCounter.increment],
            invariant=lambda c: c.value == 2,
            max_attempts=100,
            seed=42,
        )
    )
    assert isinstance(result, InterleavingResult)
    assert not result.property_holds


# ---------------------------------------------------------------------------
# (f) detect_io in async DPOR covers Redis
# ---------------------------------------------------------------------------


def test_explore_async_random_detect_io_propagates_to_detect_sql(monkeypatch):
    """detect_io=True must activate detect_sql=True in the async random path.

    The async DPOR path correctly uses ``detect_sql = ... or detect_io``, but
    the async random path uses ``setdefault`` which is a no-op when
    detect_sql=False is already present from _select_kwargs.
    """
    import frontrun.async_shuffler as _shuffler_mod

    captured_kwargs: dict[str, object] = {}

    async def _spy(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return InterleavingResult(property_holds=True, num_explored=1)

    monkeypatch.setattr(_shuffler_mod, "explore_async_random", _spy)

    asyncio.run(
        frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=lambda c: c.value == 2,
            strategy="random",
            detect_io=True,
        )
    )
    assert captured_kwargs.get("detect_sql") is True, (
        f"detect_io=True should propagate to detect_sql=True in async random path, "
        f"but got detect_sql={captured_kwargs.get('detect_sql')!r}"
    )


def test_explore_unified_detect_io_async_dpor():
    """frontrun.explore(detect_io=True) with async workers doesn't raise."""
    result = asyncio.run(
        frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=lambda c: c.value == 2,
            strategy="dpor",
            detect_io=True,
        )
    )
    assert isinstance(result, InterleavingResult)


# ---------------------------------------------------------------------------
# (i) assert_holds() edge cases
# ---------------------------------------------------------------------------


def test_assert_holds_with_prefix_and_none_explanation():
    """assert_holds(msg_prefix=...) must not produce 'prefix: None' when explanation is None.

    When property_holds=False and explanation is None, the f-string interpolation
    of None produces the literal string "None", which is misleading.
    """
    result = InterleavingResult(property_holds=False, explanation=None)
    with pytest.raises(AssertionError) as exc_info:
        result.assert_holds(msg_prefix="Test case 1: ")
    msg = str(exc_info.value)
    assert "None" not in msg, f"assert_holds() produced misleading message containing literal 'None': {msg!r}"


def test_assert_holds_with_prefix_and_explanation():
    """assert_holds(msg_prefix=...) should prepend prefix to the explanation."""
    result = InterleavingResult(property_holds=False, explanation="lost update on x")
    with pytest.raises(AssertionError, match="Test: lost update on x"):
        result.assert_holds(msg_prefix="Test: ")


# ---------------------------------------------------------------------------
# (j) explore() error message uses registry keys
# ---------------------------------------------------------------------------


def test_explore_unknown_strategy_error_message_reflects_registry():
    """The error message for unknown strategy should derive from the registry, not be hardcoded.

    If a new strategy is added to STRATEGIES, the error message must automatically
    include it without updating the error string manually.
    """
    from frontrun._strategy import STRATEGIES

    class _FakeStrategy:
        allowed_keys: frozenset[str] = frozenset()

        def run(self, **kwargs): ...

    STRATEGIES["experimental"] = _FakeStrategy()
    try:
        with pytest.raises(ValueError) as exc_info:
            frontrun.explore(
                setup=Counter,
                workers=[Counter.increment],
                invariant=counter_invariant,
                strategy="bananas",
            )
        msg = str(exc_info.value)
        assert "experimental" in msg, (
            f"Error message should dynamically include new registry key 'experimental' but got: {msg!r}"
        )
    finally:
        del STRATEGIES["experimental"]


# ---------------------------------------------------------------------------
# (k) explore() strategy validation uses correct registry for async workers
# ---------------------------------------------------------------------------


def test_explore_validates_strategy_against_async_registry():
    """explore() with async workers should validate strategy against ASYNC_STRATEGIES.

    Bug: explore() validates strategy against the sync STRATEGIES dict even when
    async workers are detected. If a strategy exists only in ASYNC_STRATEGIES,
    it would be wrongly rejected. The error message should also reflect the
    async registry's keys.
    """
    from frontrun._strategy import ASYNC_STRATEGIES

    class _FakeAsyncStrategy:
        allowed_keys: frozenset[str] = frozenset()

        async def run(self, **kwargs):
            from frontrun.common import InterleavingResult

            return InterleavingResult(property_holds=True)

    ASYNC_STRATEGIES["async_only"] = _FakeAsyncStrategy()  # type: ignore[assignment]
    try:
        # This should NOT raise — "async_only" is a valid async strategy
        coro = frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=lambda c: c.value == 2,
            strategy="async_only",
        )
        # Clean up the coroutine
        coro.close()
    finally:
        del ASYNC_STRATEGIES["async_only"]


# (l) explore() rejects explicitly-passed options the strategy does not support
# ---------------------------------------------------------------------------

# (strategy, kwargs) combos that must raise for sync workers: each option is
# silently dropped by the adapter's allowlist today, which is the same
# correctness footgun the process branch already rejects.
_SYNC_REJECTED_COMBOS = [
    ("dpor", {"seed": 42}),
    ("dpor", {"max_attempts": 50}),
    ("dpor", {"max_ops": 10}),
    ("dpor", {"debug": True}),
    ("dpor", {"detect_sql": True}),
    ("random", {"preemption_bound": 5}),
    ("random", {"preemption_bound": None}),
    ("random", {"max_executions": 3}),
    ("random", {"max_branches": 10}),
    ("random", {"stop_on_first": False}),
    ("random", {"lock_timeout": 100}),
    ("random", {"track_dunder_dict_accesses": True}),
    ("random", {"search": "random"}),
    ("random", {"detect_sql": True}),
]


@pytest.mark.parametrize(("strategy", "kwargs"), _SYNC_REJECTED_COMBOS, ids=lambda p: str(p))
def test_explore_sync_rejects_unsupported_option(strategy, kwargs):
    """Sync explore() must raise for options the selected strategy ignores."""
    with pytest.raises(ValueError, match="not supported"):
        frontrun.explore(
            setup=Counter,
            workers=[Counter.increment, Counter.increment],
            invariant=counter_invariant,
            strategy=strategy,
            **kwargs,
        )


_ASYNC_REJECTED_COMBOS = [
    ("dpor", {"seed": 42}),
    ("dpor", {"max_attempts": 50}),
    ("dpor", {"debug": True}),
    ("dpor", {"search": "random"}),
    ("dpor", {"track_dunder_dict_accesses": True}),
    ("random", {"reproduce_on_failure": 3}),
    ("random", {"warn_nondeterministic_sql": False}),
    ("random", {"debug": True}),
    ("random", {"stop_on_first": False}),
    ("random", {"preemption_bound": 5}),
    ("random", {"lock_timeout": 100}),
]


@pytest.mark.parametrize(("strategy", "kwargs"), _ASYNC_REJECTED_COMBOS, ids=lambda p: str(p))
def test_explore_async_rejects_unsupported_option(strategy, kwargs):
    """Async explore() must raise synchronously for unsupported options."""
    with pytest.raises(ValueError, match="not supported"):
        coro = frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=lambda c: c.value == 2,
            strategy=strategy,
            **kwargs,
        )
        coro.close()  # only reached when validation failed to raise


def test_explore_rejection_error_names_option_and_strategy():
    """The error must name the offending option, the strategy, and point at
    the strategy that does support the option."""
    with pytest.raises(ValueError) as exc_info:
        frontrun.explore(
            setup=Counter,
            workers=[Counter.increment, Counter.increment],
            invariant=counter_invariant,
            strategy="dpor",
            seed=42,
        )
    msg = str(exc_info.value)
    assert "seed" in msg
    assert "dpor" in msg
    assert "random" in msg, f"error should point at the strategy that supports seed=: {msg!r}"


def test_explore_rejection_error_for_sync_only_option_with_async_workers():
    """debug= is supported by no async strategy; the error should say so."""
    with pytest.raises(ValueError) as exc_info:
        coro = frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=lambda c: c.value == 2,
            strategy="random",
            debug=True,
        )
        coro.close()
    msg = str(exc_info.value)
    assert "debug" in msg
    assert "sync" in msg, f"error should say debug= needs sync workers: {msg!r}"


def test_explore_accepts_explicitly_passed_defaults():
    """Passing an unsupported option at its default value is indistinguishable
    from not passing it, and must not raise (it is a no-op either way)."""
    result = frontrun.explore(
        setup=Counter,
        workers=[Counter.increment, Counter.increment],
        invariant=counter_invariant,
        strategy="dpor",
        seed=None,
        max_attempts=200,
        max_ops=None,
        debug=False,
        detect_sql=False,
    )
    assert isinstance(result, InterleavingResult)
    assert not result.property_holds


def test_explore_async_accepts_detect_sql():
    """detect_sql= is consumed by both async adapters (folded into detect_sql
    of the underlying implementation) and must not be rejected."""
    import inspect

    for strategy in ("dpor", "random"):
        coro = frontrun.explore(
            setup=AsyncCounter,
            workers=AsyncCounter.increment,
            count=2,
            invariant=lambda c: c.value == 2,
            strategy=strategy,
            detect_sql=True,
        )
        assert inspect.iscoroutine(coro)
        coro.close()


# ---------------------------------------------------------------------------
# (m) mixed sync/async workers are diagnosed; awaitable-returning callables work
# ---------------------------------------------------------------------------


def _sync_increment(c: AsyncCounter) -> None:
    c.value += 1


def test_explore_mixed_sync_async_workers_fail_with_actionable_diagnosis():
    """A genuinely sync worker in an async run must fail with a named,
    actionable diagnostic on the first execution.

    Static rejection is impossible: a plain callable returning an awaitable is
    a *valid* async worker (async_dpor's task contract), statically identical
    to a sync worker. So the mix is diagnosed at call time instead of
    surfacing as an opaque "can't be used in 'await' expression".
    """
    result = asyncio.run(
        frontrun.explore(
            setup=AsyncCounter,
            workers=[_sync_increment, AsyncCounter.increment],
            invariant=lambda c: c.value == 2,
            reproduce_on_failure=0,
        )
    )
    assert not result.property_holds
    assert result.explanation is not None
    assert "mixes async and sync callables" in result.explanation
    assert "_sync_increment" in result.explanation


def test_explore_accepts_plain_callable_returning_awaitable():
    """A non-``async def`` callable returning a coroutine is a valid async
    worker and must not be misdiagnosed as a mixed sync worker."""
    result = asyncio.run(
        frontrun.explore(
            setup=AsyncCounter,
            workers=[AsyncCounter.increment, lambda c: asyncio.sleep(0)],
            invariant=lambda c: c.value == 1,
        )
    )
    assert result.property_holds, result.explanation


def test_explore_all_plain_awaitable_workers_fail_closed() -> None:
    """An ambiguous all-plain worker list must not discard returned coroutines."""

    def returns_awaitable(c: AsyncCounter):
        return c.increment()

    with pytest.raises(TypeError, match="returned an awaitable"):
        frontrun.explore(
            setup=AsyncCounter,
            workers=[returns_awaitable],
            invariant=lambda c: c.value == 1,
            reproduce_on_failure=0,
        )


@pytest.mark.parametrize("strategy", ["dpor", "random"])
def test_async_worker_baseexception_is_not_certified_as_a_pass(strategy: str) -> None:
    async def exits(_state: object) -> None:
        raise GeneratorExit("worker exited")

    limits = {"max_executions": 1, "reproduce_on_failure": 0} if strategy == "dpor" else {"max_attempts": 1}
    with pytest.raises(GeneratorExit, match="worker exited"):
        asyncio.run(
            frontrun.explore(
                setup=object,
                workers=[exits],
                invariant=lambda _state: True,
                strategy=strategy,
                **limits,
            )
        )


def test_tiny_positive_total_timeout_still_runs_one_execution() -> None:
    result = frontrun.explore(
        setup=object,
        workers=[lambda _state: None],
        invariant=lambda _state: False,
        total_timeout=1e-12,
        reproduce_on_failure=0,
    )

    assert result.num_explored == 1
    assert not result.property_holds


def test_random_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts must be positive"):
        frontrun.explore(
            setup=object,
            workers=[lambda _state: None],
            invariant=lambda _state: True,
            strategy="random",
            max_attempts=0,
        )


def test_explore_process_rejects_async_workers_in_mixed_list_eagerly():
    """execution='process' runs sync code only, so a worker list containing
    any async worker (mixed or not) is rejected eagerly."""
    with pytest.raises(ValueError, match="async workers are not supported"):
        frontrun.explore(
            setup=AsyncCounter,
            workers=[_sync_increment, AsyncCounter.increment],
            invariant=lambda c: c.value == 2,
            execution="process",
        )


# ---------------------------------------------------------------------------
# (n) reuse_workers is a process-only option
# ---------------------------------------------------------------------------


def test_explore_thread_rejects_reuse_workers():
    """reuse_workers=True with thread execution was a silent no-op; it must
    raise like every other explicitly-passed unsupported option."""
    with pytest.raises(ValueError, match="reuse_workers"):
        frontrun.explore(
            setup=Counter,
            workers=[Counter.increment, Counter.increment],
            invariant=counter_invariant,
            reuse_workers=True,
        )


def test_explore_thread_accepts_reuse_workers_default():
    """Passing reuse_workers=False (the default) is indistinguishable from
    omitting it and must not raise."""
    result = frontrun.explore(
        setup=Counter,
        workers=[Counter.increment, Counter.increment],
        invariant=counter_invariant,
        reuse_workers=False,
    )
    assert isinstance(result, InterleavingResult)


# ---------------------------------------------------------------------------
# (o) process-branch rejection messages match the thread-branch style
# ---------------------------------------------------------------------------


def test_process_unsupported_option_message_shape():
    """The process-branch rejection must use the thread-branch sentence shape:
    name the option (with '=' suffix), say it is 'not supported with' the mode,
    and point at what to do instead."""
    with pytest.raises(ValueError) as exc_info:
        frontrun.explore(
            setup=Counter,
            workers=[Counter.increment, Counter.increment],
            invariant=counter_invariant,
            execution="process",
            seed=42,
        )
    msg = str(exc_info.value)
    assert "seed=" in msg
    assert "not supported with execution='process'" in msg
    assert "execution='thread'" in msg


# ---------------------------------------------------------------------------
# explore_async_random: total_timeout support
# ---------------------------------------------------------------------------


def test_explore_async_random_respects_total_timeout():
    """explore_async_random must honor total_timeout to bound exploration time."""

    @dataclass
    class SlowCounter:
        value: int = 0

    async def slow_increment(c: SlowCounter) -> None:
        v = c.value
        await asyncio.sleep(0)
        c.value = v + 1

    result = asyncio.run(
        frontrun.explore_async_random(
            setup=SlowCounter,
            tasks=[slow_increment, slow_increment],
            invariant=lambda c: c.value == 2,
            max_attempts=10_000,
            total_timeout=0.01,
        )
    )
    assert result.num_explored < 10_000, (
        f"total_timeout=0.01 should have stopped exploration early, but ran {result.num_explored} attempts"
    )
