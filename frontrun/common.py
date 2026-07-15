"""Shared data structures for frontrun."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import permutations
from typing import TYPE_CHECKING, Any

from frontrun._certificate import InconclusiveExploration

if TYPE_CHECKING:
    from frontrun._sql_anomaly import SqlAnomaly


def _is_async_callable(fn: Any) -> bool:
    """Return True if calling *fn* produces a coroutine.

    Handles plain ``async def`` functions as well as callable *objects* whose
    ``__call__`` is a coroutine function (e.g. an async worker passed as a
    class instance), which ``inspect.iscoroutinefunction`` does not detect
    when applied to the instance itself.
    """
    if inspect.iscoroutinefunction(fn):
        return True
    return inspect.iscoroutinefunction(getattr(fn, "__call__", None))


def any_async(fns: Iterable[Any]) -> bool:
    """Return True if any element is a coroutine function.

    Non-callables are ignored so callers can pass dicts of ``name -> value``
    directly.
    """
    return any(_is_async_callable(fn) for fn in fns if callable(fn))


def _reject_deferred_sync_result(  # pyright: ignore[reportUnusedFunction]  # imported by sync strategy runners
    result: Any, worker: Any, *, role: str = "sync worker"
) -> Any:
    """Fail closed when a synchronous callback returns unexecuted code."""
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        kind = "an awaitable"
    elif inspect.isasyncgen(result):
        kind = "an async generator"
    elif inspect.isgenerator(result):
        result.close()
        kind = "a generator"
    else:
        return result
    name = getattr(worker, "__qualname__", None) or repr(worker)
    raise TypeError(
        f"explore(): {role} {name} returned {kind}; its deferred body was not executed. "
        "Use an `async def` worker with execution='thread' for awaitables, and execute generator bodies inside the "
        "worker before returning."
    )


def _call_sync_setup(setup: Callable[[], Any]) -> Any:
    """Call a synchronous setup hook and reject a deferred body."""
    return _reject_deferred_sync_result(setup(), setup, role="setup")


def check_invariant(invariant: Callable[[Any], Any], state: Any) -> tuple[bool, str | None]:
    """Evaluate *invariant* on *state*, tolerating ``AssertionError``.

    Returns ``(failed, assertion_message)``.  ``failed`` is True when the
    invariant returns a falsy value or raises ``AssertionError``.  When
    ``AssertionError`` was raised, its message is returned in the second
    slot so callers can fold it into their result's ``explanation``.
    """
    try:
        value = _reject_deferred_sync_result(invariant(state), invariant, role="invariant")
        return (not value, None)
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

    Returned by :func:`frontrun.explore`.

    Attributes:
        property_holds: Tri-state verdict.  ``True`` is a pass *certificate*
            (at least one interleaving completed, every worker body ran, no
            coverage-degrading event) — it can only be produced by
            :func:`frontrun._certificate.certify_pass`.  ``False`` means a
            failure was found and implies a counterexample/failure record
            exists.  ``None`` means the exploration was inconclusive (no
            evidence either way — e.g. a budget expired before any
            interleaving completed); see ``inconclusive_reason``.  ``None``
            is falsy, so ``if result.property_holds:`` stays fail-closed.
        counterexample: First schedule that violated the invariant (if any).
        num_explored: How many interleavings were tested.
        unique_interleavings: Number of distinct schedule orderings observed.
            Provides a lower bound on interleaving-space coverage.  Relevant
            for random bytecode exploration; DPOR always explores distinct
            interleavings so this equals ``num_explored``.
        failures: All failing (execution_number, schedule) pairs.  Populated
            by DPOR (thread and process execution); holds every failing
            execution when ``stop_on_first=False``, otherwise at most the
            first.
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
        exhausted: Whether the search space was fully covered.  Populated by
            ``execution="process"`` (from
            :class:`~frontrun.CrossProcessResult` ``.exhausted``); ``None``
            means the mode that produced this result does not report it
            (thread/async execution currently leaves it unset).  A
            preemption-bounded DPOR search (the default,
            ``preemption_bound=2``) never claims ``True`` — full coverage
            requires ``preemption_bound=None``.
        failure_kind: Structured category of the failure for
            ``execution="process"`` — one of ``"invariant"``,
            ``"worker_error"``, ``"deadlock"``, ``"timeout"``,
            ``"nondeterministic"``, ``"step_limit"``, ``"branch_limit"``.
            ``None`` when the
            invariant held or for thread/async execution (which encodes the
            failure in ``explanation`` only).
        inconclusive_reason: Machine-readable cause (and remedy) when
            ``property_holds`` is ``None`` — e.g. "total_timeout=0.01s elapsed
            before any interleaving completed; increase total_timeout".
            ``None`` for pass/fail verdicts.
    """

    property_holds: bool | None
    counterexample: list[int] | Schedule | None = None
    num_explored: int = 0
    unique_interleavings: int = 0
    failures: list[tuple[int, list[int]]] = field(default_factory=list)
    explanation: str | None = None
    reproduction_attempts: int = 0
    reproduction_successes: int = 0
    sql_anomaly: SqlAnomaly | None = None
    races_detected: bool = False
    exhausted: bool | None = None
    failure_kind: str | None = None
    inconclusive_reason: str | None = None

    def assert_holds(self, msg_prefix: str = "", *, allow_inconclusive: bool = False) -> None:
        """Raise unless the exploration produced a pass certificate.

        Prefer this over ``assert result.property_holds, result.explanation``.

        Args:
            msg_prefix: Optional string prepended to the message.  Useful for
                identifying which assertion failed when multiple calls appear
                in one test.
            allow_inconclusive: Opt into the weaker claim "no failure found":
                do not raise when the result is inconclusive
                (``property_holds=None``).  A genuine failure still raises.

        Raises:
            AssertionError: A counterexample was found (``property_holds`` is
                ``False``); the message carries the explanation.
            InconclusiveExploration: The exploration was inconclusive
                (``property_holds`` is ``None``) and ``allow_inconclusive``
                was not set; the message names cause and remedy.
        """
        if self.property_holds is None:
            if allow_inconclusive:
                return
            reason = (
                self.inconclusive_reason
                or self.explanation
                or "exploration completed no interleavings and recorded no cause"
            )
            message = reason if "inconclusive" in reason.lower() else f"inconclusive: {reason}"
            raise InconclusiveExploration(f"{msg_prefix}{message}" if msg_prefix else message)
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
        if self.inconclusive_reason is not None:
            parts.append(f"inconclusive_reason={self.inconclusive_reason!r}")
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
        s = _call_sync_setup(setup)
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
        s = _call_sync_setup(setup)
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
