"""Cross-process DPOR exploration.

Deterministic interleaving of separate OS processes contending on shared
external (SQL/Redis) state. See ``ideas/cross_process_exploration.md``.

The wire ``protocol`` and ``SchedulerProxy`` stand in for the in-process
scheduler inside spawned workers. The exhaustive and DPOR coordinators drive
those workers over the socket, with the latter reusing the Rust DPOR engine.
"""

from __future__ import annotations
