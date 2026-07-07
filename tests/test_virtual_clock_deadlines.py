"""Unit tests for virtual-clock deadline ordering."""

from __future__ import annotations

from frontrun._virtual_clock import DeadlineCoordinator, VirtualClock


def test_deadline_coordinator_tracks_multiple_deadlines_per_actor() -> None:
    clock = VirtualClock()
    deadlines = DeadlineCoordinator()
    timeout_token = object()

    deadlines.add_sleep(0, clock.now() + 10.0, wake_id=100)
    deadlines.add_timeout(0, clock.now() + 1.0, timeout_token)

    assert deadlines.next_deadline() == clock.now() + 1.0

    due = deadlines.advance_to_next(clock)

    assert [(event.actor_id, event.token, event.deadline) for event in due] == [(0, timeout_token, clock.now())]
    assert deadlines.has_pending()
    assert deadlines.next_deadline() == clock.now() + 9.0


def test_deadline_coordinator_orders_equal_deadlines_by_actor_then_token() -> None:
    clock = VirtualClock()
    deadlines = DeadlineCoordinator()
    token_b = "b"
    token_a = "a"

    deadlines.add_timeout(2, clock.now() + 1.0, token_b)
    deadlines.add_timeout(1, clock.now() + 1.0, token_b)
    deadlines.add_timeout(1, clock.now() + 1.0, token_a)

    due = deadlines.advance_to_next(clock)

    assert [(event.actor_id, event.token) for event in due] == [(1, token_a), (1, token_b), (2, token_b)]
