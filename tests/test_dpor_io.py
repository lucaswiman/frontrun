"""Tests for DPOR detection of file I/O race conditions.

These tests exercise DPOR's ability to detect races that involve
file system operations — TOCTOU bugs, lost updates on files, and
file-based synchronization anti-patterns.

Each test uses a lock to protect individual I/O operations (preventing
file truncation races that would crash the test) while leaving the
*compound* read-modify-write sequence unprotected — this is the
classic TOCTOU pattern that DPOR should detect and explore.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading

from frontrun.dpor import explore_dpor


class TestFileCounterRace:
    """Two threads incrementing a counter stored in a plain text file."""

    def test_unsynchronized_file_counter(self) -> None:
        """Lost update on a file-based counter.

        Individual read/write are atomic (locked), but the compound
        read-modify-write is not — DPOR should find the lost update.
        """
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "counter.txt")

        class FileCounter:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    f.write("0")

            def _read(self) -> int:
                with self.lock:
                    with open(path) as f:
                        return int(f.read())

            def _write(self, val: int) -> None:
                with self.lock:
                    with open(path, "w") as f:
                        f.write(str(val))

            def increment(self) -> None:
                val = self._read()
                self._write(val + 1)

            def value(self) -> int:
                return self._read()

        result = explore_dpor(
            setup=FileCounter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value() == 2,
        )
        assert not result.property_holds, "DPOR should detect the lost-update race on the file counter"
        assert result.explanation is not None
        assert "file:" in result.explanation

    def test_locked_file_counter_is_safe(self) -> None:
        """File counter protected by a single lock across read-modify-write."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "counter.txt")

        class LockedFileCounter:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    f.write("0")

            def increment(self) -> None:
                with self.lock:
                    with open(path) as f:
                        val = int(f.read())
                    with open(path, "w") as f:
                        f.write(str(val + 1))

            def value(self) -> int:
                with self.lock:
                    with open(path) as f:
                        return int(f.read())

        result = explore_dpor(
            setup=LockedFileCounter,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value() == 2,
        )
        assert result.property_holds, result.explanation


class TestFileFlagRace:
    """Race conditions on a file-based balance ledger."""

    def test_concurrent_balance_update(self) -> None:
        """Two threads both withdraw from a balance file — lost update.

        Individual read/write are atomic (locked), but the compound
        read-check-write is not.
        """
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "balance.txt")

        class Ledger:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    f.write("100")

            def _read_balance(self) -> int:
                with self.lock:
                    with open(path) as f:
                        return int(f.read())

            def _write_balance(self, val: int) -> None:
                with self.lock:
                    with open(path, "w") as f:
                        f.write(str(val))

            def withdraw(self, amount: int) -> None:
                balance = self._read_balance()
                self._write_balance(balance - amount)

            def balance(self) -> int:
                return self._read_balance()

        result = explore_dpor(
            setup=Ledger,
            threads=[
                lambda s: s.withdraw(30),
                lambda s: s.withdraw(20),
            ],
            invariant=lambda s: s.balance() == 50,
        )
        assert not result.property_holds, "DPOR should detect the lost-update race on the ledger"

    def test_locked_balance_update_is_safe(self) -> None:
        """Balance ledger protected by a single lock across read-check-write."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "balance.txt")

        class SafeLedger:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    f.write("100")

            def withdraw(self, amount: int) -> None:
                with self.lock:
                    with open(path) as f:
                        balance = int(f.read())
                    with open(path, "w") as f:
                        f.write(str(balance - amount))

            def balance(self) -> int:
                with self.lock:
                    with open(path) as f:
                        return int(f.read())

        result = explore_dpor(
            setup=SafeLedger,
            threads=[
                lambda s: s.withdraw(30),
                lambda s: s.withdraw(20),
            ],
            invariant=lambda s: s.balance() == 50,
        )
        assert result.property_holds, result.explanation


class TestReadModifyWrite:
    """Read-modify-write patterns on shared files."""

    def test_concurrent_json_update(self) -> None:
        """Two threads updating different keys in a JSON file lose each other's writes."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "data.json")

        class JsonStore:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    json.dump({}, f)

            def _load(self) -> dict[str, int]:
                with self.lock:
                    with open(path) as f:
                        return json.load(f)

            def _save(self, data: dict[str, int]) -> None:
                with self.lock:
                    with open(path, "w") as f:
                        json.dump(data, f)

            def update(self, key: str, value: int) -> None:
                data = self._load()
                data[key] = value
                self._save(data)

            def get_data(self) -> dict[str, int]:
                return self._load()

        result = explore_dpor(
            setup=JsonStore,
            threads=[
                lambda s: s.update("a", 1),
                lambda s: s.update("b", 2),
            ],
            invariant=lambda s: s.get_data() == {"a": 1, "b": 2},
        )
        assert not result.property_holds, "Concurrent updates to different keys lose writes without locking"

    def test_concurrent_json_update_with_lock(self) -> None:
        """Lock-protected JSON updates should preserve both keys."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "data.json")

        class LockedJsonStore:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    json.dump({}, f)

            def update(self, key: str, value: int) -> None:
                with self.lock:
                    with open(path) as f:
                        data = json.load(f)
                    data[key] = value
                    with open(path, "w") as f:
                        json.dump(data, f)

            def get_data(self) -> dict[str, int]:
                with self.lock:
                    with open(path) as f:
                        return json.load(f)

        result = explore_dpor(
            setup=LockedJsonStore,
            threads=[
                lambda s: s.update("a", 1),
                lambda s: s.update("b", 2),
            ],
            invariant=lambda s: s.get_data() == {"a": 1, "b": 2},
        )
        assert result.property_holds, result.explanation


class TestTraceCallChain:
    """Verify that DPOR I/O traces include call chain info."""

    def test_trace_shows_call_chain(self) -> None:
        """I/O traces should show which function the open() is called from."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "val.txt")

        class FileVal:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    f.write("0")

            def _read(self) -> int:
                with self.lock:
                    with open(path) as f:
                        return int(f.read())

            def _write(self, val: int) -> None:
                with self.lock:
                    with open(path, "w") as f:
                        f.write(str(val))

            def increment(self) -> None:
                val = self._read()
                self._write(val + 1)

            def value(self) -> int:
                return self._read()

        result = explore_dpor(
            setup=FileVal,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value() == 2,
        )
        assert not result.property_holds
        assert result.explanation is not None
        # Should show "Called from" with a call chain like "FileVal._read <- FileVal.increment"
        assert "Called from" in result.explanation
        assert "FileVal." in result.explanation


class TestTraceQuality:
    """Verify that DPOR I/O traces are concise and informative."""

    def test_trace_mentions_file_resource(self) -> None:
        """The explanation should reference the file path, not just attribute names."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "val.txt")

        class FileVal:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    f.write("0")

            def _read(self) -> int:
                with self.lock:
                    with open(path) as f:
                        return int(f.read())

            def _write(self, val: int) -> None:
                with self.lock:
                    with open(path, "w") as f:
                        f.write(str(val))

            def increment(self) -> None:
                val = self._read()
                self._write(val + 1)

            def value(self) -> int:
                return self._read()

        result = explore_dpor(
            setup=FileVal,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value() == 2,
        )
        assert not result.property_holds
        assert result.explanation is not None
        # Trace should show file I/O events, not just method lookups
        assert "file:" in result.explanation
        assert "val.txt" in result.explanation

    def test_trace_is_concise(self) -> None:
        """DPOR traces should be short — not dump every opcode."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "val.txt")

        class FileVal:
            def __init__(self) -> None:
                self.lock = threading.Lock()
                with open(path, "w") as f:
                    f.write("0")

            def _read(self) -> int:
                with self.lock:
                    with open(path) as f:
                        return int(f.read())

            def _write(self, val: int) -> None:
                with self.lock:
                    with open(path, "w") as f:
                        f.write(str(val))

            def increment(self) -> None:
                val = self._read()
                self._write(val + 1)

            def value(self) -> int:
                return self._read()

        result = explore_dpor(
            setup=FileVal,
            threads=[lambda c: c.increment(), lambda c: c.increment()],
            invariant=lambda c: c.value() == 2,
        )
        assert not result.property_holds
        assert result.explanation is not None
        # Count actual trace lines (between the header and footer)
        trace_lines = [line for line in result.explanation.split("\n") if line.strip().startswith("Thread ")]
        assert len(trace_lines) <= 15, f"Trace too verbose: {len(trace_lines)} lines\n{result.explanation}"
