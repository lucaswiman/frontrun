Installation
=============

Frontrun requires Python 3.10 or later.

.. code-block:: bash

   pip install frontrun

Pytest Plugin (Automatic)
--------------------------

Installing frontrun registers a pytest plugin via the ``pytest11`` entry point.
The plugin automatically patches ``threading.Lock``, ``threading.RLock``,
``queue.Queue``, and related primitives with cooperative versions **before test
collection** — no configuration needed.

To disable automatic patching, pass ``--no-frontrun-patch-locks``:

.. code-block:: bash

   pytest --no-frontrun-patch-locks
