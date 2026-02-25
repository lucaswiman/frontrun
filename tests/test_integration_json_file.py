"""Integration test: lost-update race on a JSON file-based key-value store.

Two threads concurrently increment counters stored in a JSON file.
Individual reads and writes are locked, but the compound read-modify-write
is not — DPOR and bytecode exploration should both find the lost update.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from functools import partial

from frontrun.bytecode import explore_interleavings
from frontrun.dpor import explore_dpor


class DB:
    def __init__(self, storage: str) -> None:
        self.storage = storage
        with open(self.storage, "w") as f:
            f.write(json.dumps({}))
        self.lock = threading.Lock()

    def set(self, k: str, v: object) -> None:
        cur = self.dict()
        cur[k] = v
        with self.lock:
            with open(self.storage, "w") as f:
                f.write(json.dumps(cur))

    def get(self, k: str, default: object = None) -> object:
        return self.dict().get(k, default)

    def dict(self) -> dict[str, object]:
        with self.lock:
            with open(self.storage) as f:
                contents = f.read()
                return json.loads(contents)


def _do_incrs(db: DB, items: list[str]) -> None:
    for item in items:
        count = db.get(item, 0)
        db.set(item, count + 1)  # type: ignore[operator]


_items1 = ["goat", "cat"]
_items2 = ["cat", "goat"]


def test_dpor_detects_lost_update() -> None:
    """DPOR should detect the lost update in the unsynchronized JSON DB."""
    path = os.path.join(tempfile.mkdtemp(), "db.json")
    result = explore_dpor(
        setup=lambda: DB(path),
        threads=[partial(_do_incrs, items=_items1), partial(_do_incrs, items=_items2)],
        invariant=lambda db: db.dict() == {"goat": 2, "cat": 2},
    )
    assert not result.property_holds, "DPOR should find the lost-update race on the JSON file DB"
    assert result.explanation is not None


def test_bytecode_detects_lost_update() -> None:
    """Bytecode exploration should detect the lost update."""
    path = os.path.join(tempfile.mkdtemp(), "db.json")
    result = explore_interleavings(
        setup=lambda: DB(path),
        threads=[partial(_do_incrs, items=_items1), partial(_do_incrs, items=_items2)],
        invariant=lambda db: db.dict() == {"goat": 2, "cat": 2},
    )
    assert not result.property_holds, "Bytecode exploration should find the lost-update race on the JSON file DB"
    assert result.explanation is not None
