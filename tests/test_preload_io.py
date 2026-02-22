"""Tests for the LD_PRELOAD I/O interception library and its Python bridge."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

import pytest

from frontrun._preload_io import (
    IOEventDispatcher,
    PreloadIOEvent,
    _parse_event_line,
    filter_user_io_events,
    read_io_events,
)
from frontrun.cli import _find_preload_library


@pytest.fixture
def preload_lib():
    """Find the preload library or skip the test."""
    lib = _find_preload_library()
    if lib is None:
        pytest.skip("libfrontrun_io.so not built (run `make build-io`)")
    return lib


class TestPreloadLibrary:
    """Test that the preload library intercepts I/O without crashing."""

    def test_echo_no_crash(self, preload_lib):
        """Basic smoke test: echo through preloaded process."""
        result = subprocess.run(
            ["/bin/echo", "hello"],
            env={**os.environ, "LD_PRELOAD": str(preload_lib)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_python_no_crash(self, preload_lib):
        """Python interpreter works under preload."""
        result = subprocess.run(
            [sys.executable, "-c", "print('ok')"],
            env={**os.environ, "LD_PRELOAD": str(preload_lib)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "ok" in result.stdout

    def test_file_io_logged(self, preload_lib):
        """File write/read operations are captured in the log."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as log_f:
            log_path = log_f.name

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as data_f:
            data_path = data_f.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"""
with open({data_path!r}, 'w') as f:
    f.write('test data')
with open({data_path!r}, 'r') as f:
    f.read()
""",
                ],
                env={
                    **os.environ,
                    "LD_PRELOAD": str(preload_lib),
                    "FRONTRUN_IO_LOG": log_path,
                },
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0

            events = read_io_events(log_path)
            # Should have file events for our data file
            data_events = [e for e in events if data_path in e.resource_id]
            assert len(data_events) > 0, f"No events for {data_path} in {events}"

            # Should have at least one write and one read
            writes = [e for e in data_events if e.kind == "write"]
            reads = [e for e in data_events if e.kind == "read"]
            assert len(writes) > 0, "No write events"
            assert len(reads) > 0, "No read events"
        finally:
            os.unlink(log_path)
            os.unlink(data_path)

    def test_frontrun_active_set(self, preload_lib):
        """The frontrun CLI sets FRONTRUN_ACTIVE=1."""
        from frontrun.cli import _build_env

        env = _build_env(preload_lib)
        assert env.get("FRONTRUN_ACTIVE") == "1"
        assert str(preload_lib) in env.get("LD_PRELOAD", "")


class TestEventParsing:
    """Test parsing of preload I/O events."""

    def test_parse_events(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "write\tfile:/tmp/data.txt\t3\t1234\t1234\n"
            "read\tsocket:127.0.0.1:5432\t7\t1234\t5678\n"
            "connect\tsocket:10.0.0.1:80\t8\t1234\t1234\n"
            "close\tfile:/tmp/data.txt\t3\t1234\t1234\n"
        )
        events = read_io_events(str(log_file))
        assert len(events) == 4
        assert events[0] == PreloadIOEvent("write", "file:/tmp/data.txt", 3, 1234, 1234)
        assert events[1] == PreloadIOEvent("read", "socket:127.0.0.1:5432", 7, 1234, 5678)
        assert events[2] == PreloadIOEvent("connect", "socket:10.0.0.1:80", 8, 1234, 1234)
        assert events[3] == PreloadIOEvent("close", "file:/tmp/data.txt", 3, 1234, 1234)

    def test_parse_empty_file(self, tmp_path):
        log_file = tmp_path / "empty.log"
        log_file.write_text("")
        assert read_io_events(str(log_file)) == []

    def test_parse_missing_file(self):
        assert read_io_events("/nonexistent/path") == []

    def test_parse_malformed_lines(self, tmp_path):
        log_file = tmp_path / "bad.log"
        log_file.write_text(
            "not\tenough\tfields\n"
            "write\tfile:/tmp/data.txt\tnot_a_number\t1234\t1234\n"
            "write\tfile:/tmp/good.txt\t3\t1234\t1234\n"
        )
        events = read_io_events(str(log_file))
        assert len(events) == 1
        assert events[0].resource_id == "file:/tmp/good.txt"


class TestEventFiltering:
    """Test filtering of I/O events to remove noise."""

    def test_keeps_socket_events(self):
        events = [
            PreloadIOEvent("connect", "socket:127.0.0.1:5432", 7, 1, 1),
            PreloadIOEvent("write", "socket:10.0.0.1:80", 8, 1, 1),
        ]
        filtered = filter_user_io_events(events)
        assert len(filtered) == 2

    def test_removes_stdlib_file_events(self):
        events = [
            PreloadIOEvent("read", "file:/usr/lib/python3.10/os.pyc", 3, 1, 1),
            PreloadIOEvent("read", "file:/usr/local/lib/python3.10/__pycache__/os.cpython-310.pyc", 3, 1, 1),
        ]
        filtered = filter_user_io_events(events)
        assert len(filtered) == 0

    def test_keeps_user_file_events(self):
        events = [
            PreloadIOEvent("write", "file:/tmp/data.db", 3, 1, 1),
            PreloadIOEvent("read", "file:/home/user/project/data.json", 4, 1, 1),
        ]
        filtered = filter_user_io_events(events)
        assert len(filtered) == 2

    def test_removes_site_packages(self):
        events = [
            PreloadIOEvent(
                "read", "file:/home/user/.venv/lib/python3.10/site-packages/sqlalchemy/__init__.pyc", 3, 1, 1
            ),
        ]
        filtered = filter_user_io_events(events)
        assert len(filtered) == 0


class TestParseEventLine:
    """Test the shared line parser used by both file and pipe transports."""

    def test_valid_line(self):
        ev = _parse_event_line("write\tfile:/tmp/data.txt\t3\t1234\t1234\n")
        assert ev == PreloadIOEvent("write", "file:/tmp/data.txt", 3, 1234, 1234)

    def test_empty_line(self):
        assert _parse_event_line("") is None
        assert _parse_event_line("\n") is None

    def test_too_few_fields(self):
        assert _parse_event_line("not\tenough\tfields") is None

    def test_non_numeric_fd(self):
        assert _parse_event_line("write\tfile:/tmp/x\tNaN\t1\t1") is None


class TestIOEventDispatcher:
    """Test the pipe-based event dispatcher."""

    def test_start_stop(self):
        """Dispatcher can start and stop without errors."""
        dispatcher = IOEventDispatcher()
        dispatcher.start()
        assert "FRONTRUN_IO_FD" in os.environ
        dispatcher.stop()
        assert "FRONTRUN_IO_FD" not in os.environ

    def test_context_manager(self):
        """Dispatcher works as a context manager."""
        with IOEventDispatcher() as d:
            assert "FRONTRUN_IO_FD" in os.environ
            assert d._write_fd is not None
        assert "FRONTRUN_IO_FD" not in os.environ

    def test_receives_events_from_pipe(self):
        """Events written to the pipe fd are received by listeners."""
        received: list[PreloadIOEvent] = []
        with IOEventDispatcher() as dispatcher:
            dispatcher.add_listener(received.append)

            # Simulate the Rust library writing an event to the pipe
            write_fd = dispatcher._write_fd
            assert write_fd is not None
            line = "write\tfile:/tmp/test.txt\t5\t100\t200\n"
            os.write(write_fd, line.encode())

            # Give the reader thread time to process
            for _ in range(50):
                time.sleep(0.01)
                if len(received) >= 1:
                    break

        assert len(received) == 1
        assert received[0] == PreloadIOEvent("write", "file:/tmp/test.txt", 5, 100, 200)

    def test_multiple_events(self):
        """Multiple events are received in order."""
        received: list[PreloadIOEvent] = []
        with IOEventDispatcher() as dispatcher:
            dispatcher.add_listener(received.append)

            write_fd = dispatcher._write_fd
            assert write_fd is not None
            lines = (
                "connect\tsocket:127.0.0.1:5432\t7\t1\t1\n"
                "write\tsocket:127.0.0.1:5432\t7\t1\t1\n"
                "read\tsocket:127.0.0.1:5432\t7\t1\t1\n"
                "close\tsocket:127.0.0.1:5432\t7\t1\t1\n"
            )
            os.write(write_fd, lines.encode())

            for _ in range(50):
                time.sleep(0.01)
                if len(received) >= 4:
                    break

        assert len(received) == 4
        assert [e.kind for e in received] == ["connect", "write", "read", "close"]

    def test_events_property(self):
        """The .events property returns all received events."""
        with IOEventDispatcher() as dispatcher:
            write_fd = dispatcher._write_fd
            assert write_fd is not None
            os.write(write_fd, b"write\tfile:/tmp/a\t3\t1\t1\n")
            os.write(write_fd, b"read\tfile:/tmp/b\t4\t1\t1\n")

            for _ in range(50):
                time.sleep(0.01)
                if len(dispatcher.events) >= 2:
                    break

            events = dispatcher.events
            assert len(events) == 2
            assert events[0].resource_id == "file:/tmp/a"
            assert events[1].resource_id == "file:/tmp/b"

    def test_remove_listener(self):
        """Removed listeners stop receiving events."""
        received: list[PreloadIOEvent] = []
        with IOEventDispatcher() as dispatcher:
            dispatcher.add_listener(received.append)

            write_fd = dispatcher._write_fd
            assert write_fd is not None
            os.write(write_fd, b"write\tfile:/tmp/a\t3\t1\t1\n")

            for _ in range(50):
                time.sleep(0.01)
                if len(received) >= 1:
                    break

            dispatcher.remove_listener(received.append)
            os.write(write_fd, b"write\tfile:/tmp/b\t4\t1\t1\n")
            time.sleep(0.1)

        assert len(received) == 1

    def test_malformed_lines_skipped(self):
        """Malformed lines don't crash the dispatcher."""
        received: list[PreloadIOEvent] = []
        with IOEventDispatcher() as dispatcher:
            dispatcher.add_listener(received.append)

            write_fd = dispatcher._write_fd
            assert write_fd is not None
            data = b"bad\tline\nwrite\tfile:/tmp/ok\t3\t1\t1\n\n"
            os.write(write_fd, data)

            for _ in range(50):
                time.sleep(0.01)
                if len(received) >= 1:
                    break

        assert len(received) == 1
        assert received[0].resource_id == "file:/tmp/ok"

    def test_pipe_with_preload_library(self, preload_lib):
        """Events from the actual preload library arrive via the pipe."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as data_f:
            data_path = data_f.name

        received: list[PreloadIOEvent] = []
        # We can't use IOEventDispatcher directly because it sets env vars
        # in the current process, but the preload lib runs in a subprocess.
        # Instead, create a pipe manually and pass it.
        r_fd, w_fd = os.pipe()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"""
with open({data_path!r}, 'w') as f:
    f.write('test data')
""",
                ],
                env={
                    **os.environ,
                    "LD_PRELOAD": str(preload_lib),
                    "FRONTRUN_IO_FD": str(w_fd),
                },
                capture_output=True,
                text=True,
                timeout=30,
                pass_fds=(w_fd,),
            )
            assert result.returncode == 0

            # Close write end so read sees EOF
            os.close(w_fd)
            w_fd = -1

            # Read all events from the pipe
            buf = b""
            while True:
                chunk = os.read(r_fd, 65536)
                if not chunk:
                    break
                buf += chunk

            for line_bytes in buf.split(b"\n"):
                line = line_bytes.decode("utf-8", errors="replace")
                ev = _parse_event_line(line)
                if ev is not None:
                    received.append(ev)

            data_events = [e for e in received if data_path in e.resource_id]
            assert len(data_events) > 0, f"No events for {data_path} in {received}"
            writes = [e for e in data_events if e.kind == "write"]
            assert len(writes) > 0, "No write events"
        finally:
            if w_fd >= 0:
                os.close(w_fd)
            os.close(r_fd)
            os.unlink(data_path)
