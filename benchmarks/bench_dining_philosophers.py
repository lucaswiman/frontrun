"""Benchmark: 4-fold dining philosophers deadlock detection timing."""

from __future__ import annotations

import threading
import time

import frontrun


def bench_dining_philosophers_four() -> dict[str, object]:
    """Run 4-fold dining philosophers (pure lock deadlock) and return timing + execution count."""
    num_philosophers = 4

    class State:
        def __init__(self) -> None:
            self.forks = [threading.Lock() for _ in range(num_philosophers)]

    def make_philosopher(i: int):  # noqa: ANN202
        def philosopher(s: State) -> None:
            left = i
            right = (i + 1) % num_philosophers
            with s.forks[left]:
                with s.forks[right]:
                    pass

        return philosopher

    start = time.perf_counter()
    result = frontrun.explore(
        setup=State,
        workers=[make_philosopher(i) for i in range(num_philosophers)],
        invariant=lambda s: True,
        max_executions=50000,
        preemption_bound=2,
        detect_io=False,
        deadlock_timeout=2.0,
        stop_on_first=True,
    )
    elapsed = time.perf_counter() - start

    return {
        "elapsed_seconds": round(elapsed, 3),
        "num_explored": result.num_explored,
        "found_deadlock": not result.property_holds,
        "explanation": result.explanation,
    }


if __name__ == "__main__":
    import json
    import sys

    label = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(f"=== {label} ===")

    results = []
    for i in range(num_runs):
        r = bench_dining_philosophers_four()
        results.append(r)
        print(f"  Run {i + 1}: {r['elapsed_seconds']}s, {r['num_explored']} executions, deadlock={r['found_deadlock']}")

    times = sorted(r["elapsed_seconds"] for r in results)
    median_time = times[len(times) // 2]
    print(f"  Median time: {median_time}s")
    print(f"  Median executions: {results[0]['num_explored']}")

    # Save results
    output = {
        "label": label,
        "median_time": median_time,
        "runs": results,
    }
    outfile = f"benchmarks/{label.replace(' ', '_').lower()}_results.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {outfile}")
