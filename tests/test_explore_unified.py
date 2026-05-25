"""Tests for the unified frontrun.explore() entry point and related API changes.

Covers:
  (a) explore() dispatcher — sync DPOR path
  (b) explore() dispatcher — async path
  (c) workers=fn, count=N shorthand
  (d) AssertionError in invariant → explanation
  (e) Deprecation shims warn and still work
  (f) detect_io in async DPOR covers Redis (detect_redis)
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass, field

import pytest

from frontrun import explore, explore_async_random, explore_random
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
    result = explore(
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

    result = explore(
        setup=LockedCounter,
        workers=LockedCounter.increment,
        count=2,
        invariant=lambda c: c.value == 2,
    )
    assert isinstance(result, InterleavingResult)
    assert result.property_holds


def test_explore_sync_random_strategy():
    """explore(strategy='random') finds the lost-update race."""
    result = explore(
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
        explore(
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
    explore(
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

    coro = explore(
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
        explore(
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
        explore(
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


# ---------------------------------------------------------------------------
# (c) workers=fn, count=N shorthand
# ---------------------------------------------------------------------------


def test_count_shorthand_expands_workers():
    """workers=fn + count=N is equivalent to workers=[fn, fn, ..., fn]."""
    result = explore(
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
    result = explore(
        setup=Counter,
        workers=Counter.increment,
        count=1,
        invariant=lambda c: c.value == 1,
    )
    assert result.property_holds


def test_count_with_list_raises():
    """Providing count AND a list raises ValueError."""
    with pytest.raises(ValueError, match="'count' cannot be used"):
        explore(
            setup=Counter,
            workers=[Counter.increment, Counter.increment],
            invariant=counter_invariant,
            count=2,
        )


def test_count_zero_raises():
    """count=0 raises ValueError."""
    with pytest.raises(ValueError, match="count must be a positive integer"):
        explore(
            setup=Counter,
            workers=Counter.increment,
            invariant=counter_invariant,
            count=0,
        )


def test_count_negative_raises():
    """count=-1 raises ValueError."""
    with pytest.raises(ValueError, match="count must be a positive integer"):
        explore(
            setup=Counter,
            workers=Counter.increment,
            invariant=counter_invariant,
            count=-1,
        )


def test_workers_list_without_count():
    """workers as a plain list works (no count needed)."""
    result = explore(
        setup=Counter,
        workers=[Counter.increment, Counter.increment],
        invariant=counter_invariant,
    )
    assert isinstance(result, InterleavingResult)


def test_workers_tuple_without_count():
    """workers as a tuple works (no count needed)."""
    result = explore(
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
    result = explore(
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
    result = explore(
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
        explore(
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
        explore(
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
# (e) Deprecation shims warn and still work
# ---------------------------------------------------------------------------


def test_explore_dpor_deprecated_warns():
    """Importing explore_dpor from frontrun emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import frontrun

        _ = frontrun.explore_dpor  # noqa: F841 — triggers __getattr__
    assert any(issubclass(w.category, DeprecationWarning) and "explore_dpor" in str(w.message) for w in caught)


def test_explore_interleavings_deprecated_warns():
    """Calling explore_interleavings from bytecode emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from frontrun.bytecode import explore_interleavings

        explore_interleavings(
            setup=Counter,
            threads=[Counter.increment, Counter.increment],
            invariant=counter_invariant,
            max_attempts=5,
            seed=42,
        )
    assert any(issubclass(w.category, DeprecationWarning) and "explore_interleavings" in str(w.message) for w in caught)


def test_explore_async_interleavings_deprecated_warns():
    """Calling explore_interleavings from async_shuffler emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from frontrun.async_shuffler import explore_interleavings as _deprecated

        asyncio.run(
            _deprecated(
                setup=AsyncCounter,
                tasks=[AsyncCounter.increment, AsyncCounter.increment],
                invariant=lambda c: c.value == 2,
                max_attempts=5,
                seed=42,
            )
        )
    assert any(issubclass(w.category, DeprecationWarning) and "explore_interleavings" in str(w.message) for w in caught)


def test_frontrun_explore_interleavings_getattr_warns():
    """frontrun.explore_interleavings via __getattr__ emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import frontrun

        _ = frontrun.explore_interleavings  # noqa: F841
    assert any(issubclass(w.category, DeprecationWarning) and "explore_interleavings" in str(w.message) for w in caught)


def test_frontrun_explore_async_interleavings_getattr_warns():
    """frontrun.explore_async_interleavings via __getattr__ emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import frontrun

        _ = frontrun.explore_async_interleavings  # noqa: F841
    assert any(
        issubclass(w.category, DeprecationWarning) and "explore_async_interleavings" in str(w.message) for w in caught
    )


def test_explore_random_works():
    """explore_random (canonical name) works without warning."""
    result = explore_random(
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
        explore_async_random(
            setup=AsyncCounter,
            tasks=[AsyncCounter.increment, AsyncCounter.increment],
            invariant=lambda c: c.value == 2,
            max_attempts=100,
            seed=42,
        )
    )
    assert isinstance(result, InterleavingResult)
    assert not result.property_holds


def test_explore_async_dpor_deprecated_warns():
    """explore_async_dpor emits DeprecationWarning (called directly)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from frontrun.async_dpor import explore_async_dpor

        asyncio.run(
            explore_async_dpor(
                setup=AsyncCounter,
                tasks=[AsyncCounter.increment, AsyncCounter.increment],
                invariant=lambda c: c.value == 2,
            )
        )
    assert any(issubclass(w.category, DeprecationWarning) and "explore_async_dpor" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# (f) detect_io in async DPOR covers Redis
# ---------------------------------------------------------------------------


def test_detect_io_in_async_dpor_deprecated_wrapper_implies_detect_redis():
    """explore_async_dpor(detect_io=True) does NOT separately require detect_redis=True."""
    # We test this at the API level: the deprecated wrapper should accept detect_io
    # and pass detect_redis=True to _explore_async_dpor without raising.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        from frontrun.async_dpor import explore_async_dpor

        # Just verify it runs without error; Redis isn't actually in scope in unit tests
        result = asyncio.run(
            explore_async_dpor(
                setup=AsyncCounter,
                tasks=[AsyncCounter.increment, AsyncCounter.increment],
                invariant=lambda c: c.value == 2,
                detect_io=True,  # should imply detect_redis=True internally
            )
        )
    assert isinstance(result, InterleavingResult)


def test_detect_redis_deprecated_warns():
    """explore_async_dpor(detect_redis=True) emits additional DeprecationWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from frontrun.async_dpor import explore_async_dpor

        asyncio.run(
            explore_async_dpor(
                setup=AsyncCounter,
                tasks=[AsyncCounter.increment, AsyncCounter.increment],
                invariant=lambda c: c.value == 2,
                detect_redis=True,
            )
        )
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("detect_redis" in m for m in messages)
    assert any("detect_io=True" in m for m in messages)


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
        explore(
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
        explore(
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
# (g) Every deprecation message pins a removal version
# ---------------------------------------------------------------------------


def test_every_deprecation_message_pins_a_removal_version():
    """Each entry in DEPRECATION_MESSAGES must announce when the API is removed.

    Guards against future entries being added without a planned removal version
    so users always have a concrete deadline for migrating off deprecated APIs.
    """
    import re

    from frontrun.common import DEPRECATION_MESSAGES

    # Match phrases like "removed in 0.6", "Will be removed in 1.0", etc.
    pattern = re.compile(r"removed in \d+\.\d+", re.IGNORECASE)
    missing = [name for name, msg in DEPRECATION_MESSAGES.items() if not pattern.search(msg)]
    assert not missing, (
        f"DEPRECATION_MESSAGES entries without a 'removed in X.Y' note: {missing}. "
        "Every deprecation must pin a concrete removal version."
    )


# ---------------------------------------------------------------------------
# (h) Straggler deprecations are in DEPRECATION_MESSAGES
# ---------------------------------------------------------------------------


def test_trace_executor_run_legacy_form_in_registry():
    """TraceExecutor.run() legacy single-call warning must come from DEPRECATION_MESSAGES.

    Ensures the inline warning in trace_markers.py is centralised so the
    removal-version enforcement in test_every_deprecation_message_pins_a_removal_version
    covers it automatically.
    """
    import warnings

    from frontrun import TraceExecutor
    from frontrun.common import DEPRECATION_MESSAGES, Schedule, Step

    assert "trace_executor_run_legacy_form" in DEPRECATION_MESSAGES, (
        "DEPRECATION_MESSAGES is missing 'trace_executor_run_legacy_form'. "
        "Add it so the removal-version enforcement test covers this deprecation."
    )

    schedule = Schedule([Step("t1", "m")])
    executor = TraceExecutor(schedule)

    def worker():
        x = 1  # frontrun: m

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        executor.run("t1", worker)
    executor.wait(timeout=5.0)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings, "Expected a DeprecationWarning from legacy run() form"
    # The warning text must match the registry entry exactly.
    assert dep_warnings[0].message.args[0] == DEPRECATION_MESSAGES["trace_executor_run_legacy_form"]


def test_detect_redis_param_in_registry():
    """detect_redis= deprecation warning must come from DEPRECATION_MESSAGES.

    Ensures the inline warning in async_dpor.py is centralised.
    """
    from frontrun.common import DEPRECATION_MESSAGES

    assert "detect_redis_param" in DEPRECATION_MESSAGES, (
        "DEPRECATION_MESSAGES is missing 'detect_redis_param'. "
        "Add it so the removal-version enforcement test covers this deprecation."
    )


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
            explore(
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
        coro = explore(
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
        explore_async_random(
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
