"""Defect #20: bytecode-positional replay drifts when opcode counts vary.

The `_ReplayDporScheduler` replays a counterexample as a positional sequence
of thread IDs, one entry per traced-opcode scheduling point.  If the code
under test executes a run-varying number of traced opcodes between a racing
write and the corresponding read — python-gnupg's ``GPG.on_data`` race is the
real-world case: a real ``gpg`` subprocess (plus gpg-agent) sits between the
write and the read, and the surrounding loops' iteration counts depend on
pipe/agent timing — the positional schedule desynchronises.  A thread then
reaches its racing read hundreds of positions early, before the other
thread's racing write, and the recorded ordering is not re-enforced:
reproduction degrades to a coin flip (observed 2-10/10 for gnupg
``import_keys``, 0/10 for ``list_keys``).

The fix is access-anchored replay: the failing execution records its accesses
to the *racing* objects (as run-stable ``TypeName.attr`` labels), and the
replay scheduler gates those accesses on the recorded order, suspending
positional gating while a thread waits on an anchor.  On anchor-stream
mismatch the gate disables itself and replay degrades to the old positional
behavior instead of deadlocking.

The test models the gpg variability directly: a random-length traced loop
between the racing write and read, so every replay run drifts by a different
number of scheduling points.
"""

from __future__ import annotations

import random

import frontrun


class Holder:
    """Carrier for the racing attribute (modeled on gnupg's GPG.on_data)."""

    def __init__(self) -> None:
        self.cb: int | None = None


class State:
    def __init__(self) -> None:
        self.obj = Holder()
        self.results: list[int | None] = []


def _variable_work() -> int:
    # Run-varying traced-opcode count between the racing write and read —
    # the in-process model of "a real gpg subprocess with timing-dependent
    # loops sits between the write and the read".  random is seeded from OS
    # entropy, so exploration and every replay run drift differently.
    acc = 0
    for _ in range(random.randint(50, 800)):
        acc += 1
    return acc


def _make_worker(tid: int):
    def worker(s: State) -> None:
        s.obj.cb = tid  # racing WRITE (cf. `gpg.on_data = callback`)
        _variable_work()
        s.results.append(s.obj.cb)  # racing READ (cf. gnupg.py:1309)

    return worker


def _invariant(s: State) -> bool:
    # Violated when one thread's write serviced both reads (the recorded
    # counterexample: both writes before either read).
    return not (len(s.results) == 2 and len(set(s.results)) == 1)


def test_write_read_race_replays_despite_opcode_drift():
    result = frontrun.explore(
        setup=State,
        workers=[_make_worker(0), _make_worker(1)],
        invariant=_invariant,
        detect_io=False,
        reproduce_on_failure=10,
    )
    assert not result.property_holds, "DPOR failed to detect the write-read race"
    assert result.reproduction_successes >= 8, (
        f"replay reproduced only {result.reproduction_successes}/"
        f"{result.reproduction_attempts} — positional drift is not being "
        "compensated by access anchors (defect #20)"
    )
