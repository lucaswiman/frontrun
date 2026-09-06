"""Shared event recorder for I/O interception tests."""

import threading


class IOLog:
    """Collect reporter events and query them under a lock."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, resource_id: str, kind: str) -> None:
        with self._lock:
            self.events.append((resource_id, kind))

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    @property
    def resource_ids(self) -> list[str]:
        with self._lock:
            return [r for r, _ in self.events]

    @property
    def kinds(self) -> list[str]:
        with self._lock:
            return [k for _, k in self.events]

    def events_for_table(self, table: str) -> list[tuple[str, str]]:
        prefix = f"sql:{table}"
        with self._lock:
            return [(r, k) for r, k in self.events if r == prefix or r.startswith(f"{prefix}:")]

    def has_write_for(self, table: str) -> bool:
        return any(k == "write" for _, k in self.events_for_table(table))

    def has_read_for(self, table: str) -> bool:
        return any(k == "read" for _, k in self.events_for_table(table))

    def tables_accessed(self) -> set[str]:
        with self._lock:
            return {r.split(":", 1)[1].split(":")[0] for r, _ in self.events if r.startswith("sql:")}
