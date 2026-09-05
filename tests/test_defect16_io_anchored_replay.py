"""Defect #16: detect_io=True replay fails when code has state-dependent paths.

When detect_io=True, every CALL opcode is a scheduling point. If the code
under test has a state-dependent early-return path (like celery's
``_store_result`` which returns early when it sees SUCCESS), the number of
opcode-level scheduling points changes between exploration and replay.
This desynchronises the replay schedule and prevents reproduction.

The fix: IO-anchored replay. During replay with detect_io=True, Redis commands
enter an explicit scheduler-owned ``before_io``/``after_io`` boundary.
Opcode-level scheduling points still block for single-threaded execution
between I/O boundaries, but the replay schedule is consumed only by those
explicit I/O anchors.

Running::

    REDIS_PORT=16399 frontrun pytest tests/test_defect16_io_anchored_replay.py -v --timeout=60
"""

from __future__ import annotations

import pytest

try:
    import redis as redis_lib
except ImportError:
    pytest.skip("redis package not installed", allow_module_level=True)

import frontrun

pytestmark = pytest.mark.integration


class TestStateDependentReplayReproduction:
    """Minimal reproduction of defect #16: state-dependent code paths
    cause replay schedule desynchronisation with detect_io=True.

    Models the celery _store_result pattern: GET→CHECK→conditional SET.
    Thread A writes SUCCESS, Thread B writes FAILURE. If B's GET sees
    STARTED (stale), it proceeds to SET(FAILURE), overwriting SUCCESS.

    The state-dependent path: if GET returns SUCCESS, the thread returns
    early with fewer Redis commands (and fewer opcode scheduling points).
    During replay, this path mismatch desynchronises the schedule.
    """

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

        def store_success(state: State) -> None:
            """Thread A: check-then-set SUCCESS (with early return if already SUCCESS).

            Models celery's _store_result: GET → decode → check → encode → SET.
            The early-return path skips encode and SET, producing fewer CALL
            opcodes and fewer scheduling points.
            """
            r = redis_lib.Redis(port=port, decode_responses=True)
            raw = r.get(key)
            meta = _decode(raw)  # type: ignore[arg-type]
            if meta["status"] == "SUCCESS":
                r.close()
                return  # Early return — fewer Redis ops AND fewer CALL opcodes
            encoded = _encode("SUCCESS")
            r.set(key, encoded)
            r.close()

        def store_failure(state: State) -> None:
            """Thread B: check-then-set FAILURE (with early return if already SUCCESS).

            Same pattern as store_success but writes FAILURE.
            """
            r = redis_lib.Redis(port=port, decode_responses=True)
            raw = r.get(key)
            meta = _decode(raw)  # type: ignore[arg-type]
            if meta["status"] == "SUCCESS":
                r.close()
                return  # Early return — fewer Redis ops AND fewer CALL opcodes
            encoded = _encode("FAILURE")
            r.set(key, encoded)
            r.close()

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
            workers=[store_success, store_failure],
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
