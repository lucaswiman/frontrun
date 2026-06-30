"""Defect #9: Redis DPOR counterexamples fail to reproduce (0/10).

DPOR correctly detects races in Redis GET-then-SET patterns, but
reproduction against real Redis consistently fails. All reference
Redis DPOR tests use ``reproduce_on_failure=0`` to work around this.

This test demonstrates the defect with the simplest possible Redis
race: two threads doing GET→increment→SET on the same counter key.
DPOR finds the lost-update interleaving, but reproduction fails 0/10.

Running::

    REDIS_PORT=16399 frontrun pytest tests/test_defect9_redis_reproduction.py -v --timeout=60
"""

from __future__ import annotations

import pytest

try:
    import redis as redis_lib
except ImportError:
    pytest.skip("redis package not installed", allow_module_level=True)

import frontrun


class TestRedisReproductionFailure:
    """Minimal reproduction of defect #9: Redis DPOR counterexamples
    fail to reproduce.

    The simplest Redis race: two threads do GET→increment→SET on the
    same counter. DPOR detects the lost-update interleaving, but when
    it replays the schedule against real Redis, both threads run too
    fast for the interleaving to manifest (0/10 reproduction).
    """

    def test_counter_race_detected_but_not_reproduced(self, redis_port: int) -> None:
        """DPOR finds the race but reproduction fails 0/10."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("defect9:counter", "0")
                r.close()

        def increment(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            val = int(r.get("defect9:counter"))  # type: ignore[arg-type]
            r.set("defect9:counter", str(val + 1))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("defect9:counter"))  # type: ignore[arg-type]
            r.close()
            return result == 2

        result = frontrun.explore(
            setup=State,
            workers=[increment, increment],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=10,
        )

        # DPOR should detect the race (property does NOT hold).
        assert not result.property_holds, "DPOR should detect lost-update on Redis counter"

        # Defect #9 fix: reproduction should now succeed with
        # patch_redis_for_replay fallback.
        assert result.reproduction_successes > 0, (
            f"Reproduction should succeed but got {result.reproduction_successes}/{result.reproduction_attempts} times."
        )
