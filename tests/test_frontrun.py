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


def test_cooperative_lock_release_in_dpor_machinery_clears_owner():
    """Releasing a CooperativeLock inside DPOR machinery must still clear _owner_thread_id."""
    from frontrun._cooperative import CooperativeLock, _scheduler_tls

    lock = CooperativeLock()
    lock.acquire()
    lock._owner_thread_id = 42

    _scheduler_tls._in_dpor_machinery = True
    try:
        lock.release()
        assert lock._owner_thread_id is not None, (
            "BUG: _owner_thread_id should still be set (stale) because release() "
            "skips clearing it when _in_dpor_machinery() is True"
        )
    finally:
        _scheduler_tls._in_dpor_machinery = False
