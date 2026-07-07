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

    assert [(event.actor_id, event.token, event.deadline, event.kind, event.wake_id) for event in due] == [
        (0, timeout_token, clock.now(), "timeout", None)
    ]
    assert deadlines.has_pending()
    assert deadlines.next_deadline() == clock.now() + 9.0

    due = deadlines.advance_to_next(clock)

    assert [(event.actor_id, event.kind, event.wake_id) for event in due] == [(0, "sleep", 100)]
    assert not deadlines.has_pending()


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
    assert [event.kind for event in due] == ["timeout", "timeout", "timeout"]
    assert [event.wake_id for event in due] == [None, None, None]


def test_deadline_coordinator_cancels_sleep_timeout_and_actor_deadlines_independently() -> None:
    clock = VirtualClock()
    deadlines = DeadlineCoordinator()
    timeout_token = object()

    deadlines.add_sleep(1, clock.now() + 1.0, wake_id=10)
    deadlines.add_timeout(1, clock.now() + 2.0, timeout_token)
    deadlines.cancel(1, timeout_token)

    assert deadlines.next_deadline() == clock.now() + 1.0
    assert [(event.kind, event.wake_id) for event in deadlines.advance_to_next(clock)] == [("sleep", 10)]
    assert not deadlines.has_pending()

    deadlines.add_sleep(1, clock.now() + 1.0, wake_id=10)
    deadlines.add_timeout(1, clock.now() + 2.0, timeout_token)
    deadlines.cancel_sleep(1)

    assert deadlines.next_deadline() == clock.now() + 2.0
    assert [(event.kind, event.token) for event in deadlines.advance_to_next(clock)] == [("timeout", timeout_token)]
    assert not deadlines.has_pending()

    deadlines.add_sleep(1, clock.now() + 1.0, wake_id=10)
    deadlines.add_timeout(1, clock.now() + 2.0, timeout_token)
    deadlines.add_timeout(2, clock.now() + 3.0, "other")
    deadlines.cancel(1)

    assert deadlines.next_deadline() == clock.now() + 3.0
    assert [(event.actor_id, event.token) for event in deadlines.advance_to_next(clock)] == [(2, "other")]
    assert not deadlines.has_pending()
