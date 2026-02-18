"""
Basic tests for frontrun library.
"""

import frontrun


def test_import():
    """Test that frontrun module can be imported."""
    assert frontrun is not None


def test_version():
    """Test that version is set."""
    assert hasattr(frontrun, "__version__")
    assert frontrun.__version__ == "0.0.1"
