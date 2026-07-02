"""Cross-process DPOR exploration (Phase 1 plumbing).

Deterministic interleaving of separate OS processes contending on shared
external (SQL/Redis) state. See ``ideas/cross_process_exploration.md``.

Phase 1 covers the worker half — the wire ``protocol`` and the
``SchedulerProxy`` that stands in for the in-process scheduler inside a
spawned worker. The coordinator side (engine + row locks driven over the
socket) lands in a later slice.
"""

from __future__ import annotations
