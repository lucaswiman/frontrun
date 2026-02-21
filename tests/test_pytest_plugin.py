"""Tests for the frontrun pytest plugin and lock-patching infrastructure."""

import threading

from frontrun._cooperative import (
    CooperativeLock,
    patch_locks,
    unpatch_locks,
)

# ---------------------------------------------------------------------------
# patch_locks / unpatch_locks reference counting
# ---------------------------------------------------------------------------


class TestPatchLocksRefCounting:
    """Test that patch_locks/unpatch_locks reference counting is correct.

    The pytest plugin patches locks before collection, so threading.Lock is
    already CooperativeLock when these tests run.  Each test adds its own
    patch_locks/unpatch_locks on top of the plugin's baseline.
    """

    def test_plugin_patches_by_default(self):
        """The pytest plugin has already patched threading.Lock."""
        assert threading.Lock is CooperativeLock

    def test_extra_patch_unpatch(self):
        """An extra patch/unpatch pair on top of the plugin's baseline is a no-op."""
        patch_locks()
        try:
            assert threading.Lock is CooperativeLock
        finally:
            unpatch_locks()
        # Still patched — the plugin's patch_locks() call remains.
        assert threading.Lock is CooperativeLock

    def test_nested_patch_unpatch(self):
        """Multiple extra patch calls require matching unpatch calls."""
        patch_locks()
        patch_locks()
        try:
            assert threading.Lock is CooperativeLock
            unpatch_locks()
            # Still patched — one more extra + the plugin baseline
            assert threading.Lock is CooperativeLock
        finally:
            unpatch_locks()
        # Still patched — the plugin's baseline remains
        assert threading.Lock is CooperativeLock

    def test_over_unpatch_does_not_corrupt_count(self):
        """Extra unpatches beyond the plugin baseline don't break future patches."""
        # These are no-ops beyond the plugin's baseline (count won't go below 0)
        unpatch_locks()
        unpatch_locks()
        # Re-patch
        patch_locks()
        assert threading.Lock is CooperativeLock
        unpatch_locks()  # balance our patch_locks()


# ---------------------------------------------------------------------------
# _try_replace (generic dict/list/object replacement helper)
# ---------------------------------------------------------------------------


def _try_replace(container: object, old: object, new: object) -> None:
    """Best-effort replacement of *old* with *new* inside *container*."""
    if isinstance(container, dict):
        for key, value in list(container.items()):
            if value is old:
                try:
                    container[key] = new
                except TypeError:
                    pass
    elif isinstance(container, list):
        for i, value in enumerate(container):
            if value is old:
                container[i] = new
    elif hasattr(container, "__dict__"):
        d = container.__dict__
        if not isinstance(d, dict):
            return
        for key, value in list(d.items()):
            if value is old:
                try:
                    setattr(container, key, new)
                except (AttributeError, TypeError):
                    pass


class TestTryReplace:
    """Test the best-effort replacement helper."""

    def test_replace_in_dict(self):
        old = object()
        new = object()
        d = {"key": old, "other": 42}
        _try_replace(d, old, new)
        assert d["key"] is new
        assert d["other"] == 42

    def test_replace_in_list(self):
        old = object()
        new = object()
        lst = [1, old, 3]
        _try_replace(lst, old, new)
        assert lst[1] is new

    def test_replace_in_object_dict(self):
        old = object()
        new = object()

        class Holder:
            pass

        h = Holder()
        h.lock = old  # type: ignore[attr-defined]
        _try_replace(h, old, new)
        assert h.lock is new  # type: ignore[attr-defined]

    def test_no_crash_on_tuple(self):
        """Tuples are immutable — _try_replace should be a no-op."""
        old = object()
        new = object()
        t = (old,)
        _try_replace(t, old, new)
        assert t[0] is old  # unchanged


# ---------------------------------------------------------------------------
# _is_frontrun_internal
# ---------------------------------------------------------------------------


def _is_frontrun_internal(referrer: object) -> bool:
    """Return True if *referrer* is a frontrun-internal dict we shouldn't touch."""
    if not isinstance(referrer, dict):
        return False
    mod_name = referrer.get("__name__", "")
    if isinstance(mod_name, str) and mod_name.startswith("frontrun"):
        return True
    if referrer.get("__file__", ""):
        file = referrer["__file__"]
        if isinstance(file, str) and "_real_threading" in file:
            return True
    return False


class TestIsFrontrunInternal:
    def test_frontrun_module_dict(self):
        d = {"__name__": "frontrun._cooperative", "__file__": "whatever.py"}
        assert _is_frontrun_internal(d) is True

    def test_real_threading_dict(self):
        d = {"__name__": "something", "__file__": "/path/to/_real_threading.py"}
        assert _is_frontrun_internal(d) is True

    def test_normal_dict(self):
        d = {"__name__": "mymodule", "__file__": "/path/to/mymodule.py"}
        assert _is_frontrun_internal(d) is False

    def test_non_dict(self):
        assert _is_frontrun_internal([1, 2, 3]) is False


class TestIsFrontrunInternalLive:
    """Live test that _is_frontrun_internal identifies real frontrun module dicts."""

    def test_skips_frontrun_internals(self):
        import frontrun._cooperative as mod

        assert _is_frontrun_internal(mod.__dict__) is True
