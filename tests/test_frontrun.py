"""
Basic tests for frontrun library.
"""

from pathlib import Path

import frontrun


def test_import():
    """Test that frontrun module can be imported."""
    assert frontrun is not None


def test_version():
    """__init__.py, pyproject.toml, and docs/conf.py must agree on the version."""
    root = Path(__file__).resolve().parent.parent

    # pyproject.toml
    pyproject = root / "pyproject.toml"
    pyproject_version = None
    for line in pyproject.read_text().splitlines():
        if line.startswith("version"):
            pyproject_version = line.split("=", 1)[1].strip().strip('"')
            break
    assert pyproject_version is not None, "version not found in pyproject.toml"

    # docs/conf.py — parse version/release assignments directly
    docs_release = None
    docs_version = None
    for line in (root / "docs" / "conf.py").read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("release"):
            docs_release = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("version") and "info" not in stripped:
            docs_version = stripped.split("=", 1)[1].strip().strip('"')
    assert docs_release is not None, "release not found in docs/conf.py"
    assert docs_version is not None, "version not found in docs/conf.py"

    # All three sources must match
    assert frontrun.__version__ == pyproject_version, (
        f"__init__.py ({frontrun.__version__}) != pyproject.toml ({pyproject_version})"
    )
    assert frontrun.__version__ == docs_release, (
        f"__init__.py ({frontrun.__version__}) != docs/conf.py release ({docs_release})"
    )
    # docs short version (X.Y) should be a prefix of the full version
    assert docs_release.startswith(docs_version), (
        f"docs/conf.py version ({docs_version}) is not a prefix of release ({docs_release})"
    )
