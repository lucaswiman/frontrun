"""
Interlace: Deterministic concurrency testing for Python.

Trace markers (stable)::

    from interlace.trace_markers import Schedule, Step, TraceExecutor

Async trace markers::

    from interlace.async_trace_markers import AsyncTraceExecutor
    from interlace.common import Schedule, Step

Bytecode exploration (experimental)::

    from interlace.bytecode import explore_interleavings
"""

__version__ = "0.0.1"
