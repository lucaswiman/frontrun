"""Defect #19: Redis IO-anchored replay desynchronised on run-specific keys.

The IO-anchored replay scheduler (defect #16) records anchors of the form
``redis <cmd> <key> <db_scope>`` during exploration and enforces them
verbatim during reproduction.  When the key embeds a run-specific random
value — the canonical case is an ORM that generates a fresh primary key in
``setup()`` on every run, e.g. redis-om's ULID pks — every replay's
``setup()`` produces a key that never appeared in the recorded schedule.
The very first ``before_io`` then failed the exact-string anchor comparison,
the scheduler recorded a "replay desynchronised" error, enforcement was
abandoned, and the threads free-ran: reproduction became a coin flip
(observed 1-7/10 for redis-om's ``User.get()`` + ``save()`` lost update,
while the same race on a *fixed* key reproduced 10/10).

The fix makes anchor matching structural: command and db scope must match
exactly, but the key field is rebound bijectively (recorded key -> replay
key, bound on first sight, consistent across all later commands).  See
``_IOAnchoredReplayScheduler._anchors_match``.

Running::

    REDIS_PORT=16399 frontrun python -m pytest tests/test_defect19_redis_replay_random_key.py -v
"""

from __future__ import annotations

import uuid

import pytest

try:
    import redis as redis_lib
except ImportError:
    pytest.skip("redis package not installed", allow_module_level=True)

import frontrun

pytestmark = pytest.mark.integration


class TestReplayWithRunSpecificKeys:
    def test_lost_update_on_random_key_reproduces(self, redis_port: int) -> None:
        """GET→modify→SET race on a key minted fresh by every setup() call.

        Models the redis-om ``User.get()`` + ``save()`` lost update: the
        object key contains a random pk, so replay anchors can only match
        via key rebinding, not string equality.
        """
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                # Fresh random key every run — like an ORM-generated ULID pk.
                self.key = f"defect19:user:{uuid.uuid4().hex}"
                r.hset(self.key, mapping={"email": "orig", "score": "0"})
                r.close()

        def update_email(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            doc = r.hgetall(state.key)
            doc["email"] = "new"
            r.hset(state.key, mapping=doc)
            r.close()

        def update_score(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            doc = r.hgetall(state.key)
            doc["score"] = "100"
            r.hset(state.key, mapping=doc)
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            doc = r.hgetall(state.key)
            r.close()
            return doc.get("email") == "new" and doc.get("score") == "100"

        result = frontrun.explore(
            setup=State,
            workers=[update_email, update_score],
            invariant=invariant,
            detect_io=True,
            reproduce_on_failure=10,
            deadlock_timeout=15.0,
        )

        assert not result.property_holds, "DPOR failed to detect the RMW lost update"
        assert result.reproduction_successes >= 8, (
            f"replay reproduced only {result.reproduction_successes}/"
            f"{result.reproduction_attempts} — IO anchors with run-specific "
            "keys are not being rebound (defect #19)"
        )

    def test_lost_update_on_fixed_key_still_reproduces(self, redis_port: int) -> None:
        """Control: the fixed-key variant (always worked) must stay 10/10."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("defect19:counter", "0")
                r.close()

        def increment(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            val = int(r.get("defect19:counter"))  # type: ignore[arg-type]
            r.set("defect19:counter", str(val + 1))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("defect19:counter"))  # type: ignore[arg-type]
            r.close()
            return result == 2

        result = frontrun.explore(
            setup=State,
            workers=[increment, increment],
            invariant=invariant,
            detect_io=True,
            reproduce_on_failure=10,
            deadlock_timeout=15.0,
        )

        assert not result.property_holds
        assert result.reproduction_successes >= 8
