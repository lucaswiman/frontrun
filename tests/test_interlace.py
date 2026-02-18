"""
Basic tests for interlace library.
"""

import interlace


def test_import():
    """Test that interlace module can be imported."""
    assert interlace is not None


def test_version():
    """Test that version is set."""
    assert hasattr(interlace, "__version__")
    assert interlace.__version__ == "0.0.1"
