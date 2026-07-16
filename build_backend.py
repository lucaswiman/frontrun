"""In-tree PEP 517 build backend that wraps maturin.

maturin's own backend builds only the ``frontrun._dpor`` extension (from
``crates/dpor``). It knows nothing about the sibling ``crates/io`` crate, which
compiles the ``LD_PRELOAD`` / ``DYLD_INSERT_LIBRARIES`` I/O interception library
(``libfrontrun_io.{so,dylib}``). The CI wheel jobs build that crate explicitly
and copy the artifact into the package, so binary wheels are complete — but a
source install builds the wheel through this PEP 517 backend, where nothing
otherwise compiles ``crates/io``. The result was a wheel missing the preload
library, silently degrading ``frontrun`` to Python monkey-patching only (see
issue #246).

This module re-exports every maturin PEP 517 hook unchanged and overrides the
wheel-producing hooks to first compile ``crates/io`` and drop the resulting
shared library into ``frontrun/``. maturin's ``include`` config
(``frontrun/libfrontrun_io.*``, wheel format) then packages it into the wheel,
exactly as it does for the CI-built wheels.

Degradation is deliberate and quiet where a preload library is not expected:

* On Windows there is no Unix preload library — the documented behavior is
  DPOR-only — so the compile step is skipped entirely.
* If a Rust toolchain is unavailable or the compile fails, we emit a warning and
  continue. maturin still needs Rust to build the ``_dpor`` extension, so a
  genuinely toolchain-less environment fails there with a clearer error; this
  fallback only matters for exotic setups and keeps us from turning a
  best-effort enhancement into a hard build failure.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Re-export the maturin hooks we do not override so this module is a drop-in
# PEP 517 backend. Frontends look these names up on the backend object.
from maturin import (  # noqa: F401
    build_editable as _maturin_build_editable,
)
from maturin import (
    build_sdist as build_sdist,
)
from maturin import (
    build_wheel as _maturin_build_wheel,
)
from maturin import (
    get_requires_for_build_editable as get_requires_for_build_editable,
)
from maturin import (
    get_requires_for_build_sdist as get_requires_for_build_sdist,
)
from maturin import (
    get_requires_for_build_wheel as get_requires_for_build_wheel,
)
from maturin import (
    prepare_metadata_for_build_editable as prepare_metadata_for_build_editable,
)
from maturin import (
    prepare_metadata_for_build_wheel as prepare_metadata_for_build_wheel,
)

_HERE = Path(__file__).resolve().parent
_IO_MANIFEST = _HERE / "crates" / "io" / "Cargo.toml"
_PACKAGE_DIR = _HERE / "frontrun"


def _preload_library_name() -> str | None:
    """Return the preload library filename for this platform, or ``None``.

    Windows is DPOR-only and has no Unix preload library, so it returns
    ``None`` and the compile step is skipped.
    """
    if sys.platform == "darwin":
        return "libfrontrun_io.dylib"
    if sys.platform == "win32":
        return None
    return "libfrontrun_io.so"


def _warn(message: str) -> None:
    print(f"build_backend: {message}", file=sys.stderr)


def _compile_preload_library() -> None:
    """Compile ``crates/io`` and copy the result into the ``frontrun`` package.

    Best-effort: any failure is reported but does not abort the build (see the
    module docstring). Skipped on platforms without a preload library and when
    the ``crates/io`` sources are absent (e.g. an unusual partial checkout).
    """
    lib_name = _preload_library_name()
    if lib_name is None:
        _warn("no preload library on this platform (DPOR-only); skipping crates/io build")
        return
    if not _IO_MANIFEST.exists():
        _warn(f"crates/io sources not found at {_IO_MANIFEST}; skipping preload library build")
        return

    cargo = shutil.which("cargo")
    if cargo is None:
        _warn("cargo not found on PATH; wheel will ship without the preload library")
        return

    # crates/io is a standalone workspace, so its artifact lands under
    # crates/io/target/release/ rather than the repo-root target dir.
    io_dir = _IO_MANIFEST.parent
    try:
        subprocess.run(
            [cargo, "build", "--release", "--manifest-path", str(_IO_MANIFEST)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _warn(f"failed to build crates/io ({exc}); wheel will ship without the preload library")
        return

    built = io_dir / "target" / "release" / lib_name
    if not built.exists():
        _warn(f"expected {built} after build but it is missing; wheel will ship without the preload library")
        return

    _PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, _PACKAGE_DIR / lib_name)
    _warn(f"bundled preload library {lib_name} into the wheel")


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _compile_preload_library()
    return _maturin_build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, object] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _compile_preload_library()
    return _maturin_build_editable(wheel_directory, config_settings, metadata_directory)
