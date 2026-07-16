"""Artifact smoke test: a wheel built from the published sdist must bundle the
native LD_PRELOAD I/O interception library.

Regression guard for https://github.com/lucaswiman/frontrun/issues/246. The
CI wheel jobs compile ``crates/io`` and copy ``libfrontrun_io.{so,dylib}`` into
the package explicitly, so binary wheels are complete. Source installs, however,
go through the PEP 517 build of the sdist, which historically only built the
``frontrun._dpor`` extension and silently shipped without the preload library —
``frontrun`` would then fall back to monkey-patching only, weakening C-level I/O
detection.

These tests assert that:

1. The sdist archive carries the ``crates/io`` sources (and workspace files).
2. A wheel built from that sdist through the real PEP 517 backend contains the
   compiled preload library, so ``frontrun.cli._find_preload_library()`` will
   succeed once the wheel is installed.

The build is genuinely expensive (it compiles two Rust crates), so the tests
are marked ``e2e`` and skip cleanly when the toolchain or a source checkout is
unavailable (e.g. when running against an installed package, or on Windows,
which is DPOR-only and has no Unix preload library).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

_IS_WINDOWS = platform.system() == "Windows"
_LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"
_LIB_NAME = f"libfrontrun_io{_LIB_SUFFIX}"


def _repo_root() -> Path | None:
    """Return the source checkout root, or ``None`` when not running from one."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.exists() and (parent / "crates" / "io" / "Cargo.toml").exists():
            return parent
    return None


def _require_tooling(root: Path | None) -> Path:
    if root is None:
        pytest.skip("not running from a source checkout (no crates/io)")
    if _IS_WINDOWS:
        pytest.skip("Windows is DPOR-only; no Unix preload library is built")
    if shutil.which("cargo") is None:
        pytest.skip("cargo (Rust toolchain) not available")
    try:
        import maturin  # noqa: F401
    except ImportError:
        pytest.skip("maturin not importable in the current environment")
    return root


def _build_sdist(root: Path, out_dir: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "maturin", "sdist", "--out", str(out_dir)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    sdists = list(out_dir.glob("*.tar.gz"))
    assert len(sdists) == 1, f"expected exactly one sdist, got {sdists}"
    return sdists[0]


def _read_build_system(pyproject: Path) -> tuple[str, list[str]]:
    """Return ``(build_backend, backend_path)`` from a pyproject's build-system.

    ``tomllib`` is stdlib on 3.11+; on 3.10 we fall back to the values this
    repository declares so the smoke test needs no third-party TOML parser.
    """
    try:
        import tomllib
    except ImportError:
        return "build_backend", ["."]
    data = tomllib.loads(pyproject.read_text())
    build_system = data.get("build-system", {})
    backend = build_system.get("build-backend", "")
    backend_path = build_system.get("backend-path", [])
    return backend, list(backend_path)


def _build_wheel_from_sdist(sdist: Path, work_dir: Path, out_dir: Path) -> Path:
    """Extract the sdist and build a wheel through its declared PEP 517 backend.

    We invoke the backend directly (as a build frontend would) rather than
    shelling out to pip, because the uv-created virtualenvs used for the test
    suite do not ship pip. This still exercises the real backend wiring — the
    ``build_backend`` wrapper that compiles ``crates/io`` before delegating to
    maturin.
    """
    with tarfile.open(sdist) as tar:
        try:
            tar.extractall(work_dir, filter="data")
        except TypeError:  # filter kwarg unavailable on older Pythons
            tar.extractall(work_dir)  # noqa: S202 - trusted, self-produced archive
    extracted = next(p for p in work_dir.iterdir() if p.is_dir())

    backend, backend_path = _read_build_system(extracted / "pyproject.toml")
    assert backend, "sdist pyproject.toml declares no build-backend"

    driver = (
        "import importlib, sys\n"
        f"sys.path[:0] = {backend_path!r}\n"
        f"backend = importlib.import_module({backend!r})\n"
        f"name = backend.build_wheel({str(out_dir)!r})\n"
        "print(name)\n"
    )
    # maturin's PEP 517 hook shells out to the ``maturin`` executable via PATH;
    # ensure the current interpreter's bin dir (where it lives in the venv) is
    # discoverable even when the test isn't run through the make PATH wrapper.
    env = {**os.environ, "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])}
    result = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=extracted,
        capture_output=True,
        text=True,
        timeout=1200,
        env=env,
    )
    assert result.returncode == 0, f"PEP 517 build_wheel failed:\n{result.stdout}\n{result.stderr}"

    wheels = list(out_dir.glob("frontrun-*.whl"))
    assert len(wheels) == 1, f"expected exactly one frontrun wheel, got {wheels}"
    return wheels[0]


@pytest.mark.timeout(300)
def test_sdist_contains_io_crate_sources(tmp_path: Path) -> None:
    """The sdist must ship the crates/io sources so the wheel can build them."""
    root = _require_tooling(_repo_root())
    sdist = _build_sdist(root, tmp_path)

    with tarfile.open(sdist) as tar:
        names = tar.getnames()

    def _has(suffix: str) -> bool:
        return any(name.endswith(suffix) for name in names)

    assert _has("crates/io/Cargo.toml"), f"crates/io/Cargo.toml missing from sdist: {names}"
    assert any("crates/io/src/" in name and name.endswith(".rs") for name in names), (
        f"crates/io Rust sources missing from sdist: {names}"
    )


@pytest.mark.timeout(900)
def test_wheel_from_sdist_bundles_preload_library(tmp_path: Path) -> None:
    """A wheel built from the sdist (real PEP 517 path) contains the preload lib."""
    root = _require_tooling(_repo_root())
    sdist = _build_sdist(root, tmp_path / "sdist")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = _build_wheel_from_sdist(sdist, work_dir, wheel_dir)

    with zipfile.ZipFile(wheel) as zf:
        members = zf.namelist()
    assert any(name.endswith(_LIB_NAME) for name in members), (
        f"{_LIB_NAME} missing from wheel built from sdist: {members}"
    )
