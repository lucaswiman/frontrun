"""
Basic tests for frontrun library.
"""

import re

import frontrun


def test_import():
    """Test that frontrun module can be imported."""
    assert frontrun is not None


def test_version():
    """__version__ is a valid version string."""
    assert isinstance(frontrun.__version__, str)
    assert re.match(r"^\d+\.\d+", frontrun.__version__), f"Invalid version format: {frontrun.__version__}"
