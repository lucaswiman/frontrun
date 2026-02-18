"""Shared data structures for interlace."""

from dataclasses import dataclass


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

    Attributes:
        property_holds: True if the invariant held under all tested interleavings.
        counterexample: A schedule that violated the invariant (if any).
        num_explored: How many interleavings were tested.
    """

    property_holds: bool
    counterexample: list[int] | None = None
    num_explored: int = 0
