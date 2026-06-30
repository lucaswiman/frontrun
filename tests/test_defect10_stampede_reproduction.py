"""Defect #10: Stampede reproduction failure for pipeline-based caching patterns.

After defect #9 added ``patch_redis_for_replay``, simple GET→SET races
reproduce 10/10.  However, patterns that use Redis pipelines (e.g.
``get_many``/``set_many`` in Flask-Caching's ``@memoize``) show 0/10
reproduction.

Root cause: ``_intercept_pipeline_execute`` was missing the
``_redis_replay_mode`` check that ``_intercept_execute_command`` has.
During replay, the I/O reporter is ``None`` so ``_report_redis_access``
returns False for all pipeline commands → ``reported=False`` → no
scheduling point.  Single-command Redis operations were fine because
``_intercept_execute_command`` falls back to ``_redis_replay_mode``.

Fix: add ``_redis_replay_mode`` fallback to ``_intercept_pipeline_execute``.

Running::

    make test-integration-3.14 PYTEST_ARGS="-v -k test_defect10"
"""

from __future__ import annotations

import pytest

try:
    import redis as redis_lib
except ImportError:
    pytest.skip("redis package not installed", allow_module_level=True)

import frontrun


class TestStampedeReproduction:
    """Defect #10: pipeline-based stampede reproduction."""

    def test_simple_stampede_reproduces(self, redis_port: int) -> None:
        """Control: simple GET→compute→SET stampede reproduces (defect #9 fix)."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("defect10:cache", "defect10:compute_count")
                r.set("defect10:compute_count", "0")
                r.close()

        def get_or_compute(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            cached = r.get("defect10:cache")
            if cached is None:
                r.incr("defect10:compute_count")
                result = sum(i * 3 for i in range(50))
                r.set("defect10:cache", str(result))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            count = int(r.get("defect10:compute_count"))  # type: ignore[arg-type]
            r.close()
            return count == 1

        result = frontrun.explore(
            setup=State,
            workers=[get_or_compute, get_or_compute],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=10,
        )

        assert not result.property_holds, (
            f"DPOR should detect cache stampede race (explored {result.num_explored} interleavings)"
        )
        assert result.reproduction_successes > 0, (
            f"Simple stampede should reproduce (defect #9 fix) but got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_simple_counter_reproduces(self, redis_port: int) -> None:
        """Control: simple GET→increment→SET reproduces (defect #9 regression)."""
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.set("defect10:counter", "0")
                r.close()

        def increment(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)
            val = int(r.get("defect10:counter"))  # type: ignore[arg-type]
            r.set("defect10:counter", str(val + 1))
            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            result = int(r.get("defect10:counter"))  # type: ignore[arg-type]
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

        assert not result.property_holds, "DPOR should detect lost-update on Redis counter"
        assert result.reproduction_successes > 0, (
            f"Simple counter should reproduce (defect #9 fix) but got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_pipeline_stampede_reproduces(self, redis_port: int) -> None:
        """Defect #10 (minimal): pipeline-based cache stampede must reproduce.

        This is the minimal reproduction of defect #10.  The stampede uses
        Redis pipelines (``pipeline().get().set().execute()``) for the
        cache check, mimicking what cachelib's ``get_many``/``set_many``
        do under Flask-Caching's ``@memoize``.

        Before the fix, ``_intercept_pipeline_execute`` was missing the
        ``_redis_replay_mode`` check, so pipeline commands didn't create
        scheduling points during replay → 0/10 reproduction.
        """
        port = redis_port

        class State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.delete("defect10:pipe_version", "defect10:pipe_cache", "defect10:pipe_count")
                r.set("defect10:pipe_count", "0")
                r.close()

        def pipeline_stampede(state: State) -> None:
            r = redis_lib.Redis(port=port, decode_responses=True)

            # Version check via pipeline (like Flask-Caching's _memoize_version
            # which uses get_many/set_many via cachelib's MGET + pipeline SET)
            pipe = r.pipeline()
            pipe.get("defect10:pipe_version")
            pipe.get("defect10:pipe_cache")
            results = pipe.execute()

            version = results[0]
            cached = results[1]

            if version is None:
                # First call: set version via pipeline
                pipe = r.pipeline()
                pipe.set("defect10:pipe_version", "1")
                pipe.execute()

            if cached is None:
                # Cache miss: compute and store
                r.incr("defect10:pipe_count")
                r.set("defect10:pipe_cache", "computed")

            r.close()

        def invariant(state: State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            count = int(r.get("defect10:pipe_count"))  # type: ignore[arg-type]
            r.close()
            return count <= 1

        result = frontrun.explore(
            setup=State,
            workers=[pipeline_stampede, pipeline_stampede],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=10,
        )

        assert not result.property_holds, (
            f"DPOR should detect pipeline-based stampede (explored {result.num_explored} interleavings)"
        )
        assert result.reproduction_successes > 0, (
            f"Pipeline stampede should reproduce after defect #10 fix but got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )

    def test_memoize_stampede_reproduces(self, redis_port: int) -> None:
        """Defect #10 (original): Flask-Caching @memoize stampede reproduces.

        This tests the original pattern reported in defect #10.  Requires
        flask-caching; skipped if not installed.
        """
        port = redis_port

        try:
            from flask import Flask
            from flask_caching import Cache
        except ImportError:
            pytest.skip("flask-caching not installed")

        class _State:
            def __init__(self) -> None:
                r = redis_lib.Redis(port=port, decode_responses=True)
                r.flushdb()
                r.close()
                self.app = Flask(__name__)
                self.app.config["CACHE_TYPE"] = "RedisCache"
                self.app.config["CACHE_REDIS_PORT"] = port
                self.app.config["CACHE_DEFAULT_TIMEOUT"] = 60
                self.app.config["CACHE_KEY_PREFIX"] = "test_d10_memo_"
                self.cache = Cache(self.app)

                @self.cache.memoize(timeout=60)
                def _expensive_compute(x: int) -> int:
                    r = redis_lib.Redis(port=port, decode_responses=True)
                    r.incr("d10_memo_count")
                    r.close()
                    return x * 2

                self.expensive_compute = _expensive_compute

        def _make_fn(i: int):
            def fn(s: _State) -> None:
                with s.app.app_context():
                    s.expensive_compute(42)

            return fn

        def _inv(s: _State) -> bool:
            r = redis_lib.Redis(port=port, decode_responses=True)
            count = int(r.get("d10_memo_count"))  # type: ignore[arg-type]
            r.close()
            return count <= 1

        result = frontrun.explore(
            setup=_State,
            workers=[_make_fn(0), _make_fn(1)],
            invariant=_inv,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=10,
        )

        assert not result.property_holds, (
            f"DPOR should detect memoize stampede (explored {result.num_explored} interleavings)"
        )
        assert result.reproduction_successes > 0, (
            f"Memoize stampede should reproduce after defect #10 fix but got "
            f"{result.reproduction_successes}/{result.reproduction_attempts}"
        )
