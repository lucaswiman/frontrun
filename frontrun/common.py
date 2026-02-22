"""Shared data structures for frontrun."""

from dataclasses import dataclass, field


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
    :func:`~frontrun.async_bytecode.explore_interleavings`, and
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
    """

    property_holds: bool
    counterexample: list[int] | None = None
    num_explored: int = 0
    unique_interleavings: int = 0
    failures: list[tuple[int, list[int]]] = field(default_factory=list)
    explanation: str | None = None
    reproduction_attempts: int = 0
    reproduction_successes: int = 0
