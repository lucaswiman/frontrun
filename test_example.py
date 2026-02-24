import json
import os
import tempfile
import threading
from functools import partial

from frontrun.dpor import explore_dpor
from frontrun.bytecode import explore_interleavings


class DB:
    def __init__(self, storage: str):
        self.storage = storage
        with open(self.storage, 'w') as f:
            f.write(json.dumps({}))
        self.lock = threading.Lock()
    def set(self, k, v):
        cur = self.dict()
        cur[k] = v
        with self.lock:
            with open(self.storage, 'w') as f:
                f.write(json.dumps(cur))
    def get(self, k, default=None):
        return self.dict().get(k, default)
    def dict(self):
        with self.lock:
            with open(self.storage, 'r') as f:
                contents = f.read()
                return json.loads(contents)


def do_incrs(db: DB, items: list):
    for item in items:
        count = db.get(item, 0)
        db.set(item, count + 1)

items1 = [
    "goat",
    "cat",
]

items2 = [
    "cat",
    "goat"
]

def test_dpor():
    path = os.path.join(tempfile.mkdtemp(), 'db.json')
    result = explore_dpor(
        setup=lambda: DB(path),
        threads=[partial(do_incrs, items=items1), partial(do_incrs, items=items2)],
        invariant=lambda db: db.dict() == {"goat": 2, "cat": 2},
    )
    assert result.property_holds, result.explanation


def test_bytecode():
    path = os.path.join(tempfile.mkdtemp(), 'db.json')
    result = explore_interleavings(
        setup=lambda: DB(path),
        threads=[partial(do_incrs, items=items1), partial(do_incrs, items=items2)],
        invariant=lambda db: db.dict() == {"goat": 2, "cat": 2},
    )
    assert result.property_holds, result.explanation
