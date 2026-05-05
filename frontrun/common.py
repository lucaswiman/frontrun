"""Shared data structures for frontrun."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import permutations
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from frontrun._sql_anomaly import SqlAnomaly


_F = TypeVar("_F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Deprecation messages for the 0.5 → 0.6 shim layer.
# Centralised so the module-level __getattr__ in ``frontrun/__init__.py`` and
# the wrappers in ``bytecode.py`` / ``async_shuffler.py`` / ``async_dpor.py`` /
# ``_dpor_runtime/explore.py`` stay in sync.  Each message must include a
# concrete removal version (``removed in X.Y``); ``test_explore_unified``
# enforces this so future entries can't slip in without a deadline.
# ---------------------------------------------------------------------------

DEPRECATION_MESSAGES = {
    "explore_dpor": (
        "explore_dpor is deprecated; use frontrun.explore(strategy='dpor') "
        "(or frontrun.explore(...)) instead. The old API will be removed in 0.6."
    ),
    "explore_async_dpor": (
        "explore_async_dpor is deprecated; use frontrun.explore(strategy='dpor') "
        "with async workers instead. The old API will be removed in 0.6."
    ),
    "explore_interleavings": (
        "explore_interleavings is deprecated; use frontrun.explore(strategy='random') "
        "(or frontrun.explore_random(...)) instead. The old API will be removed in 0.6."
    ),
    "explore_async_interleavings": (
        "explore_async_interleavings (also accessed as async `explore_interleavings`) is deprecated; "
        "use frontrun.explore(strategy='random') (or frontrun.explore_async_random(...)) instead. "
        "The old API will be removed in 0.6."
    ),
    "trace_executor_run_legacy_form": (
        "Calling TraceExecutor.run() with individual thread names is deprecated; "
        "pass a dict {name: callable} instead, e.g. "
        "executor.run({'thread1': fn1, 'thread2': fn2}, timeout=5.0). "
        "Will be removed in 0.6."
    ),
    "detect_redis_param": (
        "detect_redis=True is deprecated; use detect_io=True instead "
        "(now covers Redis in both sync and async). "
        "The old parameter will be removed in 0.6."
    ),
}


def deprecate(fn: _F, message: str) -> _F:
    """Wrap *fn* in a :class:`DeprecationWarning`-emitting shim.

    The wrapper is :func:`functools.wraps`-preserving so ``help()`` and
    the inspectable signature still reflect *fn*.
    """

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return fn(*args, **kwargs)

    return _wrapper  # type: ignore[return-value]


def any_async(fns: Iterable[Any]) -> bool:
    """Return True if any element is a coroutine function.

    Non-callables are ignored so callers can pass dicts of ``name -> value``
    directly.
    """
    return any(inspect.iscoroutinefunction(fn) for fn in fns if callable(fn))


def check_invariant(invariant: Callable[[Any], Any], state: Any) -> tuple[bool, str | None]:
    """Evaluate *invariant* on *state*, tolerating ``AssertionError``.

    Returns ``(failed, assertion_message)``.  ``failed`` is True when the
    invariant returns a falsy value or raises ``AssertionError``.  When
    ``AssertionError`` was raised, its message is returned in the second
    slot so callers can fold it into their result's ``explanation``.
    """
    try:
        return (not invariant(state), None)
    except AssertionError as exc:
        return (True, str(exc))


class NondeterministicSQLError(Exception):
    """Raised when SQL INSERT statements are detected during exploration.

    Autoincrement/SERIAL/IDENTITY columns assign IDs based on execution
    order, making test results non-deterministic across interleavings.
    Pre-allocate rows with explicit IDs in your test setup instead.

    Pass ``warn_nondeterministic_sql=False`` to suppress this check if
    you understand the implications.
    """


@dataclass
class Step:
    """Represents a single step in the execution schedule.

    Attributes:
        execution_name: The name of the execution unit (thread/task) that should execute this step
        marker_name: The marker name that identifies this synchronization point
    """

    execution_name: str
    marker_name: str

    def __repr__(self):
        return f"Step({self.execution_name!r}, {self.marker_name!r})"


class Schedule:
    """Defines the execution order for tasks at synchronization points.

    A schedule is a linear sequence of steps that specify which task should
    execute which marker in order.
    """

    def __init__(self, steps: list[Step]):
        """Initialize a schedule with a list of steps.

        Args:
            steps: Ordered list of Step objects defining the execution sequence
        """
        self.steps = steps
        self._validate()

    def _validate(self):
        """Validate that the schedule is well-formed."""
        if not self.steps:
            raise ValueError("Schedule must contain at least one step")

    def __repr__(self):
        return f"Schedule({self.steps!r})"


@dataclass
class InterleavingResult:
    """Result of exploring interleavings.

    Returned by :func:`~frontrun.bytecode.explore_interleavings`,
    :func:`~frontrun.async_shuffler.explore_interleavings`, and
    :func:`~frontrun.dpor.explore_dpor`.

    Attributes:
        property_holds: True if the invariant held under all tested interleavings.
        counterexample: First schedule that violated the invariant (if any).
        num_explored: How many interleavings were tested.
        unique_interleavings: Number of distinct schedule orderings observed.
            Provides a lower bound on interleaving-space coverage.  Relevant
            for random bytecode exploration; DPOR always explores distinct
            interleavings so this equals ``num_explored``.
        failures: All failing (execution_number, schedule) pairs.  Only
            populated by DPOR when ``stop_on_first=False``.
        explanation: Human-readable explanation of the race condition, showing
            interleaved source lines and the conflict pattern. None if no
            race was found.
        reproduction_attempts: Number of times the counterexample schedule
            was re-run to test reproducibility.  0 if no counterexample.
        reproduction_successes: How many of those re-runs reproduced the
            invariant violation.
        sql_anomaly: Classified SQL isolation anomaly (if any SQL I/O events
            were recorded).  A :class:`~frontrun._sql_anomaly.SqlAnomaly`
            instance, or None if the failure did not involve SQL.
    """

    property_holds: bool
    counterexample: list[int] | Schedule | None = None
    num_explored: int = 0
    unique_interleavings: int = 0
    failures: list[tuple[int, list[int]]] = field(default_factory=list)
    explanation: str | None = None
    reproduction_attempts: int = 0
    reproduction_successes: int = 0
    sql_anomaly: SqlAnomaly | None = None
    races_detected: bool = False

    def assert_holds(self, msg_prefix: str = "") -> None:
        """Raise AssertionError with the race explanation if the invariant failed.

        Prefer this over ``assert result.property_holds, result.explanation``.

        Args:
            msg_prefix: Optional string prepended to the explanation in the
                AssertionError message.  Useful for identifying which assertion
                failed when multiple calls appear in one test.
        """
        if not self.property_holds:
            explanation = self.explanation or ""
            raise AssertionError(f"{msg_prefix}{explanation}" if msg_prefix else explanation)

    def __repr__(self) -> str:
        ce = self.counterexample
        if ce is not None and isinstance(ce, list) and len(ce) > 10:
            ce_repr = f"[{', '.join(map(str, ce[:5]))}, ...({len(ce)} steps)]"
        else:
            ce_repr = repr(ce)
        parts = [
            f"property_holds={self.property_holds}",
            f"counterexample={ce_repr}",
            f"num_explored={self.num_explored}",
        ]
        if self.races_detected:
            parts.append("races_detected=True")
        return f"InterleavingResult({', '.join(parts)})"


def compute_serializable_states(
    setup: Callable[[], Any],
    thread_funcs: list[Callable[[Any], None]],
    state_hash: Callable[[Any], Any] | None = None,
) -> set[Any]:
    """Compute the set of valid serializable states.

    Runs all N! sequential orderings of the thread functions and collects
    the hash of each resulting state.  An interleaved execution is
    *serializable* if its final state hash is in this set.

    Args:
        setup: Factory that creates fresh shared state.
        thread_funcs: Thread/task functions (each takes state as argument).
        state_hash: Hash function for state.  If None, uses ``repr()``.

    Returns:
        Set of valid state hashes.
    """
    if state_hash is None:
        state_hash = repr
    valid: set[Any] = set()
    for perm in permutations(range(len(thread_funcs))):
        s = setup()
        for i in perm:
            thread_funcs[i](s)
        valid.add(state_hash(s))
    return valid


async def compute_serializable_states_async(
    setup: Callable[[], Any],
    task_funcs: list[Callable[[Any], Any]],
    state_hash: Callable[[Any], Any] | None = None,
) -> set[Any]:
    """Async version of compute_serializable_states.

    Runs all N! sequential orderings of async task functions.
    """
    if state_hash is None:
        state_hash = repr
    valid: set[Any] = set()
    for perm in permutations(range(len(task_funcs))):
        s = setup()
        for i in perm:
            await task_funcs[i](s)
        valid.add(state_hash(s))
    return valid


def resolve_serializable_hash_fn(
    serializable_invariant: Callable[[Any], Any] | bool,
) -> Callable[[Any], Any] | None:
    """Extract the state-hash function from a ``serializable_invariant`` parameter.

    Returns the callable itself when it is a hash function, or ``None``
    when the caller passed ``True`` (meaning "use the default ``repr``").
    """
    return serializable_invariant if callable(serializable_invariant) else None


def check_serializability_violation(
    state: Any,
    serial_valid_states: set[Any],
    hash_fn: Callable[[Any], Any],
    execution_num: int,
) -> str | None:
    """Check whether *state* violates serializability.

    Returns an explanation string when the state hash is not in
    *serial_valid_states*, or ``None`` if it passes.

    *hash_fn* should be the resolved state-hash function (use
    :func:`resolve_serializable_hash_fn` to convert the raw
    ``serializable_invariant`` parameter, falling back to ``repr``
    when it returns ``None``).
    """
    state_h = hash_fn(state)
    if state_h not in serial_valid_states:
        return (
            f"Serializability violation in execution {execution_num}.\n"
            f"State {state_h!r} does not match any sequential ordering.\n"
            f"Valid sequential states: {serial_valid_states!r}"
        )
    return None
