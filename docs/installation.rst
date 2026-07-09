Installation
=============

Frontrun requires Python 3.10 or later.

.. code-block:: bash

   pip install frontrun

Optional extras
---------------

Cross-process exploration (``frontrun.explore(execution="process")`` /
``frontrun.explore_processes()``) serialises workers with `dill
<https://pypi.org/project/dill/>`_, shipped as the ``process`` extra:

.. code-block:: bash

   pip install "frontrun[process]"

CLI Setup
----------

The ``frontrun`` CLI command is installed automatically with the package.
It wraps any command with I/O interception:

.. code-block:: bash

   frontrun pytest -vv tests/            # run pytest with I/O interception
   frontrun python examples/orm_race.py  # run a script with I/O interception

Building the I/O Library
~~~~~~~~~~~~~~~~~~~~~~~~~~

For C-level I/O interception (required for opaque database drivers, Redis
clients, etc.), build the native ``LD_PRELOAD`` library:

.. code-block:: bash

   make build-io    # requires Rust toolchain

This compiles ``libfrontrun_io.so`` (Linux) or ``libfrontrun_io.dylib``
(macOS) and copies it into the ``frontrun`` package directory where the
CLI can find it.

Pytest Plugin
--------------

Installing frontrun registers a pytest plugin via the ``pytest11`` entry point.
The plugin patches ``threading.Lock``, ``threading.RLock``, ``queue.Queue``,
and related primitives with cooperative versions **before test collection**.

Patching is **on by default when running under the ``frontrun`` CLI**.
When running plain ``pytest`` without the CLI, patching is off unless
explicitly requested:

.. code-block:: bash

   frontrun pytest                    # cooperative lock patching is active (auto)
   pytest --frontrun-patch-locks      # explicitly enable without CLI
   pytest --no-frontrun-patch-locks   # explicitly disable even under CLI

Tests that use ``frontrun.explore()`` or ``frontrun.explore_random()`` will be
automatically skipped when run without the frontrun CLI, preventing
confusing failures when the environment isn't properly set up.
