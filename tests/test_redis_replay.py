"""Redis I/O-anchored replay across state-dependent paths and run-specific keys.

Regressions #16 and #19: CALL counts may vary between I/O boundaries, and
fresh setup keys must bind consistently to recorded keys during replay.
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


class TestRedisReplay:
    def test_toctou_with_early_return_reproduces(self, redis_port: int) -> None:
        """GET→CHECK→SET race with state-dependent early return reproduces reliably."""
        port = redis_port
        key = "defect16:status"

        class State:
            def __init__(self) -> None:
                import json

                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set(key, json.dumps({"status": "STARTED", "result": "started"}))
                r.close()

        def _encode(value: str) -> str:
            """Simulate celery's encode() — adds extra CALL scheduling points."""
            import json

            return json.dumps({"status": value, "result": value.lower()})

        def _decode(raw: str) -> dict:
            """Simulate celery's decode() — adds extra CALL scheduling points."""
            import json

            return json.loads(raw)

        def store_status(status: str):
            def worker(state: State) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                raw = r.get(key)
                meta = _decode(raw)  # type: ignore[arg-type]
                if meta["status"] == "SUCCESS":
                    r.close()
                    return  # State-dependent path omits encode and SET.
                encoded = _encode(status)
                r.set(key, encoded)
                r.close()

            return worker

        def invariant(state: State) -> bool:
            """Once SUCCESS is written, FAILURE must not overwrite it."""
            import json

            r = redis_lib.Redis(port=port, decode_responses=True)
            raw = r.get(key)
            r.close()
            meta = json.loads(raw)  # type: ignore[arg-type]
            return meta["status"] == "SUCCESS"

        result = frontrun.explore(
            setup=State,
            workers=[store_status("SUCCESS"), store_status("FAILURE")],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=10,
        )

        # DPOR should detect the race.
        assert not result.property_holds, "DPOR should detect the TOCTOU race"

        # With scheduler-owned IO anchors, replay should deterministically
        # reproduce the same Redis GET/CHECK/SET interleaving every time.
        assert result.reproduction_successes == 10, (
            f"Expected 10/10 reproduction but got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}. "
            f"Schedule has {len(result.counterexample)} steps."
        )

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

        def update_field(field: str, value: str):
            def worker(state: State) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                doc = r.hgetall(state.key)
                doc[field] = value
                r.hset(state.key, mapping=doc)
                r.close()

            return worker

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            doc = r.hgetall(state.key)
            r.close()
            return doc.get("email") == "new" and doc.get("score") == "100"

        result = frontrun.explore(
            setup=State,
            workers=[update_field("email", "new"), update_field("score", "100")],
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
        """Fixed scalar keys remain replayable alongside run-specific hash keys."""
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
