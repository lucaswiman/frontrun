"""frontrun CLI — launch subprocesses with I/O interception.

Usage::

    frontrun pytest -vv tests/test_concurrency.py
    frontrun python -c "print('hello')"
    frontrun uvicorn myapp:app

The CLI sets up the environment so that:

1. ``LD_PRELOAD`` (Linux) or ``DYLD_INSERT_LIBRARIES`` (macOS) points to
   ``libfrontrun_io.so`` / ``libfrontrun_io.dylib``, which intercepts libc
   I/O syscall wrappers (``connect``, ``send``, ``recv``, ``read``,
   ``write``, ``close``, etc.) and reports fd-to-resource mappings back
   to the DPOR scheduler.
2. ``FRONTRUN_ACTIVE=1`` is set so that the frontrun library knows it is
   running under the CLI and should apply monkey-patching automatically.
3. The subprocess inherits the rest of the parent environment.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

# Environment variable that signals frontrun is active
FRONTRUN_ACTIVE_ENV = "FRONTRUN_ACTIVE"

# Environment variable pointing to the preload library path
FRONTRUN_PRELOAD_LIB_ENV = "FRONTRUN_PRELOAD_LIB"


def _find_preload_library() -> Path | None:
    """Locate the compiled libfrontrun_io shared library.

    Search order:
    1. ``FRONTRUN_PRELOAD_LIB`` environment variable (explicit override)
    2. Next to this Python file (in-tree / editable install)
    3. In the crates/io build directory (development layout)
    """
    # 1. Explicit override
    env_path = os.environ.get(FRONTRUN_PRELOAD_LIB_ENV)
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    system = platform.system()
    if system == "Darwin":
        lib_name = "libfrontrun_io.dylib"
    else:
        lib_name = "libfrontrun_io.so"

    # 2. Next to this file (installed or copied by build)
    here = Path(__file__).resolve().parent
    candidate = here / lib_name
    if candidate.exists():
        return candidate

    # 3. In the crates/io build directory (development layout)
    project_root = here.parent
    for build_dir in [
        project_root / "crates" / "io" / "target" / "release",
        project_root / "crates" / "io" / "target" / "debug",
        project_root / "target" / "release",
        project_root / "target" / "debug",
    ]:
        candidate = build_dir / lib_name
        if candidate.exists():
            return candidate

    return None


def _build_env(preload_lib: Path | None) -> dict[str, str]:
    """Build the subprocess environment with frontrun activation."""
    env = os.environ.copy()
    env[FRONTRUN_ACTIVE_ENV] = "1"

    if preload_lib is not None:
        lib_str = str(preload_lib)
        env[FRONTRUN_PRELOAD_LIB_ENV] = lib_str

        system = platform.system()
        if system == "Darwin":
            existing = env.get("DYLD_INSERT_LIBRARIES", "")
            if existing:
                env["DYLD_INSERT_LIBRARIES"] = f"{lib_str}:{existing}"
            else:
                env["DYLD_INSERT_LIBRARIES"] = lib_str
        else:
            # Linux and other ELF-based systems
            existing = env.get("LD_PRELOAD", "")
            if existing:
                env["LD_PRELOAD"] = f"{lib_str}:{existing}"
            else:
                env["LD_PRELOAD"] = lib_str

    return env


def _warn_macos_sip(command: str) -> None:
    """Warn if the target command may be SIP-protected on macOS.

    macOS System Integrity Protection strips ``DYLD_INSERT_LIBRARIES``
    from system binaries (those under ``/usr/bin/``, ``/usr/sbin/``,
    etc.).  The preload library silently has no effect in that case.
    Homebrew, pyenv, conda, and venv Python installs are unaffected.
    """
    import shutil

    resolved = shutil.which(command) or command
    sip_prefixes = ("/usr/bin/", "/usr/sbin/", "/usr/libexec/", "/bin/", "/sbin/")
    if any(resolved.startswith(p) for p in sip_prefixes):
        print(
            f"frontrun: warning: {resolved} may be a macOS SIP-protected binary; "
            "DYLD_INSERT_LIBRARIES will be stripped by the kernel. "
            "Use a Homebrew, pyenv, or venv Python instead.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``frontrun`` CLI command."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: frontrun <command> [args...]", file=sys.stderr)
        print()
        print("Run a command with frontrun I/O interception enabled.")
        print()
        print("Examples:")
        print("  frontrun pytest -vv tests/")
        print("  frontrun python -c \"import requests; requests.get('http://example.com')\"")
        print("  frontrun uvicorn myapp:app")
        print()
        print("Environment variables:")
        print(f"  {FRONTRUN_ACTIVE_ENV}=1       Set automatically; signals frontrun is active")
        print(f"  {FRONTRUN_PRELOAD_LIB_ENV}  Override path to libfrontrun_io.so/.dylib")
        return 1

    preload_lib = _find_preload_library()
    if preload_lib is not None:
        print(f"frontrun: using preload library {preload_lib}", file=sys.stderr)
        if platform.system() == "Darwin":
            _warn_macos_sip(argv[0])
    else:
        print(
            "frontrun: preload library not found; running with monkey-patching only",
            file=sys.stderr,
        )

    env = _build_env(preload_lib)

    try:
        result = subprocess.run(argv, env=env)
        return result.returncode
    except FileNotFoundError:
        print(f"frontrun: command not found: {argv[0]}", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        return 130


def is_active() -> bool:
    """Return True if running under the ``frontrun`` CLI."""
    return os.environ.get(FRONTRUN_ACTIVE_ENV) == "1"


def require_active(caller: str) -> None:
    """Skip or raise if not running under the ``frontrun`` CLI.

    When called under **pytest** (detected via ``_pytest`` in
    ``sys.modules``), this raises ``pytest.skip`` so the test is
    reported as skipped with a helpful message.

    When called outside pytest, raises ``RuntimeError``.
    """
    if is_active():
        return

    # Also allow if cooperative patching is explicitly active
    # (e.g. --frontrun-patch-locks or direct patch_locks() call)
    from frontrun._cooperative import is_patched

    if is_patched():
        return

    msg = (
        f"{caller}() requires the frontrun CLI environment. "
        "Run your command with: frontrun pytest ... "
        "or use --frontrun-patch-locks"
    )

    if "_pytest" in sys.modules:
        import pytest

        pytest.skip(msg)
    else:
        raise RuntimeError(msg)


if __name__ == "__main__":
    sys.exit(main())
