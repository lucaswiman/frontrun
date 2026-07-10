API Reference
=============

Common Data Structures
-----------------------

.. automodule:: frontrun.common
   :members:
   :undoc-members:
   :show-inheritance:


Unified Exploration Entry Points
---------------------------------

.. autofunction:: frontrun.explore

.. autofunction:: frontrun.explore_random

.. autofunction:: frontrun.explore_async_random


Virtual Clock
-------------

See :doc:`virtual_clock` for a guide. ``frontrun.explore(..., clock="virtual")``
/ ``clock="explored"`` control time as a scheduled quantity; there is no
separate entry point.


Cross-Process Exploration
-------------------------

See :doc:`cross_process` for a guide. ``frontrun.explore(...,
execution="process")`` mirrors the thread/async interface; the functions below
are the lower-level entry point.

.. autofunction:: frontrun.explore_processes

.. autoclass:: frontrun.Subprocess
   :members:

.. autoclass:: frontrun.CrossProcessResult
   :members:


Trace Markers
--------------

.. automodule:: frontrun.trace_markers
   :members:
   :undoc-members:
   :show-inheritance:


Marker Schedule Exploration
-----------------------------

.. autofunction:: frontrun.trace_markers.marker_schedule_strategy

.. autofunction:: frontrun.trace_markers.all_marker_schedules

.. autofunction:: frontrun.trace_markers.explore_marker_interleavings


Async Trace Markers
--------------------

.. automodule:: frontrun.async_trace_markers
   :members:
   :undoc-members:
   :show-inheritance:


Async Bytecode Instrumentation
--------------------------------

.. automodule:: frontrun.async_shuffler
   :members:
   :undoc-members:
   :show-inheritance:


Trace Formatting
-----------------

.. automodule:: frontrun._trace_format
   :members:
   :undoc-members:
   :show-inheritance:


Async Scheduler Utilities
--------------------------

.. automodule:: frontrun.async_scheduler
   :members:
   :undoc-members:
   :show-inheritance:


ORM Helpers (contrib)
----------------------

.. automodule:: frontrun.contrib.django
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: frontrun.contrib.sqlalchemy
   :members:
   :undoc-members:
   :show-inheritance:
