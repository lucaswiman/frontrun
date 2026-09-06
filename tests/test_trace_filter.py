"""Tests for configurable trace filtering (TraceFilter, trace_packages)."""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import nullcontext

import pytest

from frontrun._tracing import (
    _SKIP_DIRS,
    TraceFilter,
    _filename_to_module,
    get_active_trace_filter,
    is_cmdline_user_code,
    is_dynamic_code,
    set_active_trace_filter,
    should_trace_file,
)

# ---------------------------------------------------------------------------
# Unit tests for _filename_to_module
# ---------------------------------------------------------------------------


def _get_site_packages() -> str:
    sp_dir = next((d for d in _SKIP_DIRS if "site-packages" in d), None)
    if sp_dir is None:
        pytest.skip("no site-packages in _SKIP_DIRS")
    return sp_dir


class TestFilenameToModule:
    def test_returns_none_for_user_code(self) -> None:
        assert _filename_to_module("/home/user/myproject/app.py") is None

    def test_converts_site_packages_path(self) -> None:
        sp_dir = _get_site_packages()
        fake = os.path.join(sp_dir, "django_filters", "views.py")
        assert _filename_to_module(fake) == "django_filters.views"

    def test_converts_init_py(self) -> None:
        sp_dir = _get_site_packages()
        fake = os.path.join(sp_dir, "django_filters", "__init__.py")
        assert _filename_to_module(fake) == "django_filters"

    def test_converts_nested_module(self) -> None:
        sp_dir = _get_site_packages()
        fake = os.path.join(sp_dir, "django", "contrib", "sites", "models.py")
        assert _filename_to_module(fake) == "django.contrib.sites.models"

    def test_converts_c_extension_with_abi_tag(self) -> None:
        sp_dir = _get_site_packages()
        fake = os.path.join(sp_dir, "_sqlite3.cpython-314-x86_64-linux-gnu.so")
        assert _filename_to_module(fake) == "_sqlite3"

    def test_converts_c_extension_simple_so(self) -> None:
        sp_dir = _get_site_packages()
        fake = os.path.join(sp_dir, "mymod.so")
        assert _filename_to_module(fake) == "mymod"

    def test_converts_nested_c_extension(self) -> None:
        sp_dir = _get_site_packages()
        fake = os.path.join(sp_dir, "pkg", "_speedups.cpython-314-x86_64-linux-gnu.so")
        assert _filename_to_module(fake) == "pkg._speedups"

    def test_prefers_longest_matching_skip_dir(self, monkeypatch) -> None:
        # On stock python.org layouts the stdlib dir and its site-packages
        # child are BOTH in _SKIP_DIRS, so a file matches two prefixes.  The
        # module name must be derived from the longest (most specific) prefix,
        # deterministically — not from whichever the set happens to yield first
        # (frozenset order depends on PYTHONHASHSEED).  Otherwise a user's
        # trace_packages=["mylib.*"] silently fails to match run-to-run.
        import frontrun._tracing as tracing

        nested = ("/opt/py/lib", "/opt/py/lib/site-packages")  # short prefix first
        monkeypatch.setattr(tracing, "_SKIP_DIRS", nested)
        result = _filename_to_module("/opt/py/lib/site-packages/mylib/views.py")
        assert result == "mylib.views"


# ---------------------------------------------------------------------------
# Unit tests for TraceFilter
# ---------------------------------------------------------------------------


class TestTraceFilter:
    def test_default_traces_user_code(self) -> None:
        filt = TraceFilter()
        assert filt.should_trace_file("/home/user/myproject/app.py") is True

    def test_default_skips_site_packages(self) -> None:
        sp_dir = _get_site_packages()
        filt = TraceFilter()
        fake = os.path.join(sp_dir, "django_filters", "views.py")
        assert filt.should_trace_file(fake) is False

    def test_trace_packages_allows_matching_site_packages(self) -> None:
        sp_dir = _get_site_packages()
        filt = TraceFilter(trace_packages=["django_*"])
        fake = os.path.join(sp_dir, "django_filters", "views.py")
        assert filt.should_trace_file(fake) is True

    def test_trace_packages_rejects_non_matching(self) -> None:
        sp_dir = _get_site_packages()
        filt = TraceFilter(trace_packages=["django_*"])
        fake = os.path.join(sp_dir, "requests", "api.py")
        assert filt.should_trace_file(fake) is False

    def test_trace_packages_dot_star_pattern(self) -> None:
        sp_dir = _get_site_packages()
        filt = TraceFilter(trace_packages=["django.contrib.sites.*"])
        fake = os.path.join(sp_dir, "django", "contrib", "sites", "models.py")
        assert filt.should_trace_file(fake) is True
        # But django.contrib.auth should NOT match
        fake2 = os.path.join(sp_dir, "django", "contrib", "auth", "models.py")
        assert filt.should_trace_file(fake2) is False

    def test_star_matches_nested_modules(self) -> None:
        """Verify that ``django_*`` matches submodules like ``django_filters.views``."""
        sp_dir = _get_site_packages()
        filt = TraceFilter(trace_packages=["django_*"])
        fake = os.path.join(sp_dir, "django_filters", "views.py")
        assert filt.should_trace_file(fake) is True

    def test_always_skips_threading(self) -> None:
        filt = TraceFilter(trace_packages=["*"])
        assert filt.should_trace_file(threading.__file__) is False

    def test_always_skips_frozen(self) -> None:
        filt = TraceFilter(trace_packages=["*"])
        assert filt.should_trace_file("<frozen importlib._bootstrap>") is False

    def test_always_skips_frontrun(self) -> None:
        from frontrun._tracing import _FRONTRUN_DIR

        filt = TraceFilter(trace_packages=["*"])
        assert filt.should_trace_file(os.path.join(_FRONTRUN_DIR, "dpor.py")) is False

    def test_multiple_patterns(self) -> None:
        sp_dir = _get_site_packages()
        filt = TraceFilter(trace_packages=["django_*", "requests.*"])
        assert filt.should_trace_file(os.path.join(sp_dir, "django_filters", "views.py")) is True
        assert filt.should_trace_file(os.path.join(sp_dir, "requests", "api.py")) is True
        assert filt.should_trace_file(os.path.join(sp_dir, "flask", "app.py")) is False

    def test_empty_list_traces_nothing_extra(self) -> None:
        """Passing ``trace_packages=[]`` should behave like the default (no extra tracing)."""
        sp_dir = _get_site_packages()
        filt = TraceFilter(trace_packages=[])
        fake = os.path.join(sp_dir, "django_filters", "views.py")
        assert filt.should_trace_file(fake) is False
        # But user code is still traced
        assert filt.should_trace_file("/home/user/myproject/app.py") is True


# ---------------------------------------------------------------------------
# Unit tests for active filter context management
# ---------------------------------------------------------------------------


class TestActiveFilter:
    def test_default_active_filter(self) -> None:
        filt = get_active_trace_filter()
        assert isinstance(filt, TraceFilter)

    def test_set_and_reset(self) -> None:
        custom = TraceFilter(trace_packages=["mylib.*"])
        set_active_trace_filter(custom)
        try:
            assert get_active_trace_filter() is custom
        finally:
            set_active_trace_filter(None)
        # After reset, should be the default again
        assert get_active_trace_filter() is not custom

    def test_should_trace_file_uses_active_filter(self) -> None:
        sp_dir = _get_site_packages()
        fake = os.path.join(sp_dir, "mylib", "core.py")

        # Default: should NOT trace
        assert should_trace_file(fake) is False

        # With custom filter: should trace
        set_active_trace_filter(TraceFilter(trace_packages=["mylib.*"]))
        try:
            assert should_trace_file(fake) is True
        finally:
            set_active_trace_filter(None)

        # After reset: should NOT trace again
        assert should_trace_file(fake) is False

    def test_filter_visible_from_worker_thread(self) -> None:
        """The active filter must be visible from worker threads (not thread-local)."""
        sp_dir = _get_site_packages()
        fake = os.path.join(sp_dir, "mylib", "core.py")
        results: list[bool] = []

        def worker() -> None:
            results.append(should_trace_file(fake))

        set_active_trace_filter(TraceFilter(trace_packages=["mylib.*"]))
        try:
            t = threading.Thread(target=worker)
            t.start()
            t.join()
        finally:
            set_active_trace_filter(None)

        assert results == [True], "Worker thread should see the active trace filter"


# ---------------------------------------------------------------------------
# Unit tests for is_cmdline_user_code (python -c tracing)
# ---------------------------------------------------------------------------


class TestCmdlineUserCode:
    """Test that python -c code is identified as user code for tracing."""

    def test_is_dynamic_code_for_string(self) -> None:
        assert is_dynamic_code("<string>") is True

    def test_cmdline_user_code_in_cmdline_mode(self) -> None:
        """When __main__ has no __file__ (python -c), <string> code with __main__ globals is user code."""
        import sys

        main = sys.modules["__main__"]
        had_file = hasattr(main, "__file__")
        saved_file = getattr(main, "__file__", None)
        # Simulate python -c mode by removing __file__
        if had_file:
            delattr(main, "__file__")
        try:
            # Code defined in the -c string has f_globals pointing to __main__.__dict__
            assert is_cmdline_user_code("<string>", main.__dict__) is True
            # But exec'd with custom globals should not match
            assert is_cmdline_user_code("<string>", {"__name__": "some_lib"}) is False
            # Non-<string> filenames should not match
            assert is_cmdline_user_code("myfile.py", main.__dict__) is False
            # <frozen ...> should not match
            assert is_cmdline_user_code("<frozen importlib>", main.__dict__) is False
        finally:
            if had_file:
                main.__file__ = saved_file  # type: ignore[attr-defined]

    def test_cmdline_user_code_in_script_mode(self) -> None:
        """When __main__ has __file__ (normal script), <string> is NOT cmdline user code."""
        import sys

        main = sys.modules["__main__"]
        had_file = hasattr(main, "__file__")
        saved_file = getattr(main, "__file__", None)
        # Ensure __main__ has __file__ (normal script mode)
        if not had_file:
            main.__file__ = "/some/test.py"  # type: ignore[attr-defined]
        try:
            assert is_cmdline_user_code("<string>", main.__dict__) is False
        finally:
            if had_file:
                main.__file__ = saved_file  # type: ignore[attr-defined]
            else:
                delattr(main, "__file__")

    def test_stdin_user_code_in_cmdline_mode(self) -> None:
        """Code piped via stdin (`frontrun python - < script.py`) has
        co_filename == "<stdin>" and __main__.__file__ == "<stdin>"; it must
        be treated as user code, or exploration certifies a pass having
        traced nothing.
        """
        import sys

        main = sys.modules["__main__"]
        had_file = hasattr(main, "__file__")
        saved_file = getattr(main, "__file__", None)
        # Real stdin mode: __main__.__file__ is the pseudo-filename "<stdin>".
        main.__file__ = "<stdin>"  # type: ignore[attr-defined]
        try:
            assert is_cmdline_user_code("<stdin>", main.__dict__) is True
            assert is_cmdline_user_code("<stdin>", {"__name__": "some_lib"}) is False
        finally:
            if had_file:
                main.__file__ = saved_file  # type: ignore[attr-defined]
            else:
                delattr(main, "__file__")


# ---------------------------------------------------------------------------
# Trace filter cleanup on exception
# ---------------------------------------------------------------------------


class TestTraceFilterCleanupOnException:
    """Trace filter must be cleaned up even if setup code raises."""

    def test_bytecode_explore_cleans_filter_on_setup_error(self) -> None:
        """If serializable_invariant setup raises, trace filter is cleaned up."""
        import frontrun

        def bad_setup():
            raise RuntimeError("setup failed!")

        old_filter = get_active_trace_filter()
        with pytest.raises(RuntimeError, match="setup failed"):
            frontrun.explore_random(
                setup=bad_setup,
                threads=[lambda s: None],
                invariant=lambda s: True,
                serializable_invariant=True,
                trace_packages=["my_package"],
                max_attempts=1,
                reproduce_on_failure=0,
            )
        # The trace filter must be restored, not leaked
        assert get_active_trace_filter() == old_filter, "Trace filter leaked after exception in explore_random setup"

    @pytest.mark.parametrize("strategy,asynchronous", [("dpor", False), ("dpor", True), ("random", False)])
    @pytest.mark.parametrize("setup_error", [False, True])
    @pytest.mark.parametrize("trace_packages", [None, ["inner.*"]])
    def test_exploration_restores_surrounding_filter(self, strategy, asynchronous, setup_error, trace_packages) -> None:
        import frontrun

        surrounding = TraceFilter(["surrounding.*"])

        def setup():
            if setup_error:
                raise RuntimeError("setup failed")
            return object()

        async def async_worker(_state):
            pass

        old_filter = get_active_trace_filter()
        set_active_trace_filter(surrounding)
        try:
            with pytest.raises(RuntimeError, match="setup failed") if setup_error else nullcontext():
                result = frontrun.explore(
                    strategy=strategy,
                    setup=setup,
                    workers=[async_worker] if asynchronous else [lambda _state: None],
                    invariant=lambda _state: True,
                    serializable_invariant=setup_error,
                    trace_packages=trace_packages,
                    reproduce_on_failure=0,
                    **({"max_executions": 1} if strategy == "dpor" else {"max_attempts": 1}),
                )
                if asynchronous:
                    result = asyncio.run(result)
                assert result.property_holds
            assert get_active_trace_filter() is surrounding
        finally:
            set_active_trace_filter(old_filter)
