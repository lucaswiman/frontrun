"""Release-artifact safety checks."""

from pathlib import Path


def test_publish_workflow_does_not_ship_preload_less_sdist() -> None:
    """The current sdist omits crates/io and libfrontrun_io entirely.

    Publishing it silently downgrades source installs to Python-level monkey
    patching, so release automation must withhold the sdist until its build can
    provide the native preload library.
    """
    root = Path(__file__).resolve().parents[1]
    wheel_workflow = (root / ".github" / "workflows" / "build-wheels.yml").read_text()

    assert "command: sdist" not in wheel_workflow
    assert "dist-sdist" not in wheel_workflow
