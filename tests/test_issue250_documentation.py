"""Regression checks for the concrete documentation/packaging follow-ups in issue #250."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_maturin_tree_has_no_setuptools_manifest_or_packaged_dev_makefile() -> None:
    assert not (ROOT / "MANIFEST.in").exists()
    assert not (ROOT / "frontrun" / "Makefile").exists()


def test_installation_explains_when_the_native_io_library_needs_building() -> None:
    installation = _read("docs/installation.rst")

    assert "Binary wheels already bundle" in installation
    assert "When installing from a source checkout, build" in installation
    assert "source distribution, build" not in installation


def test_trace_filter_example_uses_the_running_interpreters_site_packages() -> None:
    trace_filtering = _read("docs/trace_filtering.rst")

    assert "/path/to/site-packages" not in trace_filtering
    assert "site.getsitepackages()" in trace_filtering


def test_dpor_report_commands_use_the_frontrun_wrapper() -> None:
    examples = _read("docs/examples.rst")

    assert not re.search(r"^    python examples/dpor_", examples, flags=re.MULTILINE)
    assert "frontrun python examples/dpor_bank_transfer.py" in examples


def test_readme_sample_output_and_async_redis_precondition_are_accurate() -> None:
    readme = _read("README.md")
    quickstart = readme.split("## Not just toy counters", 1)[0]

    assert "Lost update:" in quickstart
    assert "Write-write conflict:" not in quickstart
    assert "after 1 interleavings" not in readme
    assert "Seed `counter` to `0` before exploration" in readme


def test_issue250_fixes_are_recorded_under_unreleased() -> None:
    changelog = _read("CHANGELOG.rst")
    unreleased = changelog.split("0.7.0 (", 1)[0]

    assert "Unreleased\n----------" in unreleased
    assert "clock" in unreleased.lower()
    assert "cross-process" in unreleased.lower()
    assert "SQL" in unreleased
    assert "Redis" in unreleased
