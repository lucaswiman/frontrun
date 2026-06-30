Trace Filtering
===============

By default, frontrun only traces **user code** — files outside the Python
stdlib, ``site-packages``, and frontrun's own internals.  This keeps the
per-opcode overhead low and avoids producing thousands of spurious
scheduling points inside library code (e.g. SQLAlchemy's internal
``exec``/``compile`` calls, dataclass ``__init__`` methods, etc.).

Sometimes, however, the code under test lives *inside* an installed
package.  This is common with Django apps, plugin-based architectures,
and any project where business logic is distributed across separately
installable packages.  The ``trace_packages`` parameter lets you widen
the filter to include specific installed packages.


``trace_packages`` parameter
-----------------------------

All exploration entry points accept an optional ``trace_packages``
argument — a list of package-name patterns using :mod:`fnmatch` syntax
(``*`` matches any string, ``?`` matches one character).

.. code-block:: python

   import frontrun

   result = frontrun.explore(
       setup=make_state,
       workers=[thread_a, thread_b],
       invariant=check_invariant,
       trace_packages=["mylib.*", "django_filters.*"],
   )

Patterns are matched against the **dotted module name** derived from the
file's path inside ``site-packages``.  For example, a file at
``site-packages/django_filters/views.py`` has module name
``django_filters.views``.

.. note::

   In fnmatch syntax, ``*`` matches **any characters including dots**.
   This means ``django_*`` matches both ``django_filters`` (top-level)
   and ``django_filters.views`` (submodule).  If you need to match only
   the top-level package, there is no need — matching submodules is
   almost always the desired behaviour.

.. list-table:: Pattern examples
   :header-rows: 1

   * - Pattern
     - Matches
   * - ``django_*``
     - ``django_filters``, ``django_filters.views``, ``django_rest_framework``, etc. (top-level + all submodules)
   * - ``django_filters.*``
     - ``django_filters.views``, ``django_filters.filters``, etc. (submodules only, not ``django_filters`` itself)
   * - ``myapp.*``
     - ``myapp.models``, ``myapp.views``, ``myapp.utils.helpers``, etc.
   * - ``*``
     - Everything in site-packages (not recommended — very slow)

These entry points accept ``trace_packages``:

- :func:`frontrun.explore`
- :func:`frontrun.explore_random`
- :func:`frontrun.contrib.django.django_dpor` (has Django-specific defaults)


Django integration
-------------------

:func:`~frontrun.contrib.django.django_dpor` defaults ``trace_packages``
to :data:`~frontrun.contrib.django.DJANGO_TRACE_PACKAGES`:

.. code-block:: python

   DJANGO_TRACE_PACKAGES = ["django_*", "django.contrib.sites.*"]

This means Django third-party apps (``django_filters``,
``django_rest_framework``, etc.) and ``django.contrib.sites`` submodules
are traced automatically when using ``django_dpor``.

To add more packages, pass your own list:

.. code-block:: python

   from frontrun.contrib.django import django_dpor

   result = django_dpor(
       setup=make_state,
       threads=[thread_a, thread_b],
       invariant=check_invariant,
       trace_packages=["django_*", "django.contrib.sites.*", "myapp.*"],
   )

To disable extra tracing entirely (trace only user code), pass an empty
list:

.. code-block:: python

   result = django_dpor(
       setup=make_state,
       threads=[thread_a, thread_b],
       invariant=check_invariant,
       trace_packages=[],
   )


Advanced: ``TraceFilter`` class
--------------------------------

Under the hood, ``trace_packages`` creates a
:class:`~frontrun._tracing.TraceFilter` instance and installs it as the
active filter for the duration of the exploration.  You can also create
and inspect ``TraceFilter`` objects directly:

.. code-block:: python

   from frontrun._tracing import TraceFilter

   filt = TraceFilter(trace_packages=["django_*", "mylib.*"])

   # Test whether a specific file would be traced:
   filt.should_trace_file("/path/to/site-packages/django_filters/views.py")
   # -> True

   filt.should_trace_file("/path/to/site-packages/requests/api.py")
   # -> False

Files that are always excluded regardless of ``trace_packages``:

- ``threading.py`` (threading internals)
- ``<frozen ...>`` modules (frozen stdlib)
- Frontrun's own package directory

.. autodata:: frontrun.contrib.django.DJANGO_TRACE_PACKAGES

.. autoclass:: frontrun._tracing.TraceFilter
   :members: should_trace_file
