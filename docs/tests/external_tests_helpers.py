"""Shared utilities for external library test runs."""


def print_exploration_result(result):
    """Print single exploration result."""
    print(f"\nExplored {result.num_explored} interleavings")
    print(f"Property holds: {result.property_holds}")
    if result.counterexample:
        print(f"Counterexample schedule length: {len(result.counterexample)}")


def print_seed_sweep_results(found_seeds, total_explored):
    """Print seed sweep results."""
    print(f"\nTotal interleavings explored: {total_explored}")
    print(f"Seeds that found the bug: {len(found_seeds)} / 20")
    for seed, n in found_seeds:
        print(f"  seed={seed}: found after {n} interleavings")
