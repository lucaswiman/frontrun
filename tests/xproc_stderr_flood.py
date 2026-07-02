"""Child-only e2e helper: flood stderr at import, then exit nonzero.

Imported by a spawned ``worker_main`` child (target resolution imports the
module before connecting to the coordinator). Writes far more than a pipe
buffer (64 KiB on Linux) to stderr to prove the launcher's capture cannot
deadlock the child, then exits nonzero so ``diagnose()`` reads the capture.

Guarded on the worker env var so importing this module from the test process
itself (or via pytest collection) is a no-op.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("FRONTRUN_XPROC_WORKER_ID") is not None:
    sys.stderr.write("x" * 262_144 + "\n")
    sys.stderr.write("flooded stderr with 256 KiB\n")
    sys.stderr.flush()
    sys.exit(3)


def main() -> None:
    """Never reached: the module exits at import inside a worker child."""
