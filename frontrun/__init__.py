"""
Frontrun: Deterministic concurrency testing for Python.

Trace markers (stable)::

    from frontrun.trace_markers import Schedule, Step, TraceExecutor

Async trace markers::

    from frontrun.async_trace_markers import AsyncTraceExecutor
    from frontrun.common import Schedule, Step

Bytecode exploration (experimental)::

    from frontrun.bytecode import explore_interleavings

Async bytecode exploration (experimental)::

    from frontrun.async_bytecode import explore_interleavings, await_point

DPOR (Dynamic Partial Order Reduction) systematic exploration::

    from frontrun.dpor import explore_dpor
"""

__version__ = "0.1.0"
