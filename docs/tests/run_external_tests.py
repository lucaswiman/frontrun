#!/usr/bin/env python3
"""Run all external library tests and produce a unified summary."""

import sys
from dataclasses import dataclass

# Import all test modules
import test_cachetools_real
import test_pydis_real
import test_pydispatcher_real
import test_threadpoolctl_real
import test_tpool_real


@dataclass
class TestResults:
    """Results from a single library's tests."""

    name: str
    bug_description: str
    commit: str
    single_run_found: bool
    seeds_found: int
    avg_attempts: float
    reproduction_rate: int

    def summary_row(self) -> str:
        """Format as a summary table row."""
        found_rate = f"{self.seeds_found}/20" if self.seeds_found > 0 else "0/20"
        return f"| {self.name:<20} | {self.bug_description:<25} | {self.commit:<10} | {found_rate:<10} | {self.avg_attempts:<8} | {self.reproduction_rate}/10 |"


def run_cachetools():
    """Run cachetools tests."""
    print("\n" + "=" * 70)
    print("CACHETOOLS: Cache.__setitem__ lost update")
    print("=" * 70)

    print("\n--- Single run (seed=42) ---")
    result1 = test_cachetools_real.test_real_cachetools_lost_update()

    print("\n--- Seed sweep (20 seeds) ---")
    seeds1 = test_cachetools_real.test_real_cachetools_lost_update_sweep()

    print("\n--- Deterministic reproduction ---")
    repro1 = test_cachetools_real.test_real_cachetools_reproduce()

    seeds_found = len(seeds1)
    avg_attempts = sum(n for _, n in seeds1) / seeds_found if seeds_found > 0 else 0

    return TestResults(
        name="cachetools",
        bug_description="__setitem__ lost update",
        commit="e5f8f01",
        single_run_found=not result1.property_holds,
        seeds_found=seeds_found,
        avg_attempts=avg_attempts,
        reproduction_rate=repro1 if repro1 else 0,
    )


def run_threadpoolctl():
    """Run threadpoolctl tests."""
    print("\n" + "=" * 70)
    print("THREADPOOLCTL: _get_libc() TOCTOU")
    print("=" * 70)

    print("\n--- Single run (seed=42) ---")
    result1 = test_threadpoolctl_real.test_real_threadpoolctl_get_libc_toctou()

    print("\n--- Seed sweep (20 seeds) ---")
    seeds1 = test_threadpoolctl_real.test_real_threadpoolctl_get_libc_sweep()

    print("\n--- Deterministic reproduction ---")
    repro1 = test_threadpoolctl_real.test_real_threadpoolctl_reproduce()

    seeds_found = len(seeds1)
    avg_attempts = sum(n for _, n in seeds1) / seeds_found if seeds_found > 0 else 0

    return TestResults(
        name="threadpoolctl",
        bug_description="_get_libc TOCTOU",
        commit="cf38a18",
        single_run_found=not result1.property_holds,
        seeds_found=seeds_found,
        avg_attempts=avg_attempts,
        reproduction_rate=repro1 if repro1 else 0,
    )


def run_tpool():
    """Run TPool tests."""
    print("\n" + "=" * 70)
    print("TPOOL: _should_keep_going() TOCTOU")
    print("=" * 70)

    print("\n--- Single run (seed=42) ---")
    result1 = test_tpool_real.test_real_tpool_toctou_explore()

    print("\n--- Seed sweep (20 seeds) ---")
    seeds1 = test_tpool_real.test_real_tpool_toctou_sweep_seeds()

    print("\n--- Deterministic reproduction ---")
    repro1 = test_tpool_real.test_real_tpool_reproduce()

    seeds_found = len(seeds1)
    avg_attempts = sum(n for _, n in seeds1) / seeds_found if seeds_found > 0 else 0

    return TestResults(
        name="TPool",
        bug_description="_should_keep_going TOCTOU",
        commit="1bffaaf",
        single_run_found=not result1.property_holds,
        seeds_found=seeds_found,
        avg_attempts=avg_attempts,
        reproduction_rate=repro1 if repro1 else 0,
    )


def run_pydispatcher():
    """Run pydispatcher tests."""
    print("\n" + "=" * 70)
    print("PYDISPATCHER: connect() TOCTOU")
    print("=" * 70)

    print("\n--- Single run (seed=42) ---")
    result1 = test_pydispatcher_real.test_real_pydispatcher_connect_race()

    print("\n--- Seed sweep (20 seeds) ---")
    seeds1 = test_pydispatcher_real.test_real_pydispatcher_connect_race_sweep()

    print("\n--- Deterministic reproduction ---")
    repro1 = test_pydispatcher_real.test_real_pydispatcher_reproduce()

    seeds_found = len(seeds1)
    avg_attempts = sum(n for _, n in seeds1) / seeds_found if seeds_found > 0 else 0

    return TestResults(
        name="PyDispatcher",
        bug_description="connect() TOCTOU",
        commit="0c2768d",
        single_run_found=not result1.property_holds,
        seeds_found=seeds_found,
        avg_attempts=avg_attempts,
        reproduction_rate=repro1 if repro1 else 0,
    )


def run_pydis():
    """Run pydis tests."""
    print("\n" + "=" * 70)
    print("PYDIS: INCR lost update and SET NX race")
    print("=" * 70)

    print("\n--- Single run: INCR lost update (seed=42) ---")
    result1 = test_pydis_real.test_real_pydis_incr_lost_update()

    print("\n--- Seed sweep (20 seeds) ---")
    seeds1 = test_pydis_real.test_real_pydis_incr_sweep()

    print("\n--- Deterministic reproduction ---")
    repro1 = test_pydis_real.test_real_pydis_incr_reproduce()

    seeds_found = len(seeds1)
    avg_attempts = sum(n for _, n in seeds1) / seeds_found if seeds_found > 0 else 0

    return TestResults(
        name="pydis",
        bug_description="INCR lost update",
        commit="1b02b27",
        single_run_found=not result1.property_holds,
        seeds_found=seeds_found,
        avg_attempts=avg_attempts,
        reproduction_rate=repro1 if repro1 else 0,
    )


def main():
    """Run all tests and print summary."""
    results = []

    try:
        results.append(run_cachetools())
    except Exception as e:
        print(f"ERROR in cachetools: {e}", file=sys.stderr)

    try:
        results.append(run_threadpoolctl())
    except Exception as e:
        print(f"ERROR in threadpoolctl: {e}", file=sys.stderr)

    try:
        results.append(run_tpool())
    except Exception as e:
        print(f"ERROR in tpool: {e}", file=sys.stderr)

    try:
        results.append(run_pydispatcher())
    except Exception as e:
        print(f"ERROR in pydispatcher: {e}", file=sys.stderr)

    try:
        results.append(run_pydis())
    except Exception as e:
        print(f"ERROR in pydis: {e}", file=sys.stderr)

    # Print summary table
    print("\n" + "=" * 70)
    print("UNIFIED SUMMARY")
    print("=" * 70)
    print()
    print("| Library          | Bug                    | Commit | Seeds Found | Avg Attempts | Repro |")
    print("|------------------+------------------------+--------+-------------+--------------+-------|")
    for result in results:
        print(result.summary_row())

    print()
    total_seeds = sum(r.seeds_found for r in results)
    total_found = sum(1 for r in results if r.seeds_found == 20)
    avg_all = sum(r.avg_attempts for r in results) / len(results) if results else 0

    print(f"Total: {total_seeds} seeds found across {len(results)} libraries")
    print(f"Perfect detection: {total_found}/{len(results)} libraries (20/20 seeds)")
    print(f"Average attempts across all libraries: {avg_all:.2f}")
    print()


if __name__ == "__main__":
    main()
