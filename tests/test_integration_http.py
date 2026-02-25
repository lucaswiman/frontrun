"""Tests for DPOR detection of HTTP-based race conditions.

These tests start a local HTTP key-value server and exercise DPOR's
ability to detect application-level race conditions that manifest
through HTTP GET/POST requests over TCP sockets.

The race conditions are in the APPLICATION (client) code — the server
is a correctly-synchronized key-value store.  The client reads a value
via GET, modifies it in Python, and writes it back via POST.  Two
threads doing this concurrently can overwrite each other's updates.

DPOR detects these races through the LD_PRELOAD I/O interception
library, which intercepts libc-level send/recv calls on the HTTP
sockets and reports them as I/O accesses to the DPOR engine.

Running::

    make test-integration-3.10
    # or directly:
    frontrun .venv-3.10/bin/pytest tests/test_integration_http.py -v
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import pytest

from frontrun.dpor import explore_dpor

# ---------------------------------------------------------------------------
# Embedded HTTP key-value server
# ---------------------------------------------------------------------------
# The server is intentionally correct (thread-safe).  All race
# conditions tested below are in the client code that reads, modifies,
# and writes back values through HTTP requests.


class _KVHandler(BaseHTTPRequestHandler):
    """Thread-safe key-value store over HTTP.

    GET  /key        → 200 + JSON value, or 404
    POST /key  body  → 200, stores JSON body
    """

    # Shared across all handler instances (set per-server via closure).
    store: dict[str, str]
    store_lock: threading.Lock

    def do_GET(self) -> None:
        key = self.path.strip("/")
        with self.store_lock:
            value = self.store.get(key)
        if value is not None:
            body = value.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        key = self.path.strip("/")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        with self.store_lock:
            self.store[key] = body
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A002
        pass  # Suppress request logging


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _find_free_port() -> int:
    """Find an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_get(base_url: str, key: str) -> str:
    """GET a value from the KV server."""
    with urllib.request.urlopen(f"{base_url}/{key}") as resp:
        return resp.read().decode()


def _http_post(base_url: str, key: str, value: str) -> None:
    """POST a value to the KV server."""
    data = value.encode()
    req = urllib.request.Request(
        f"{base_url}/{key}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        resp.read()


@pytest.fixture(scope="module")
def http_server():
    """Start a local HTTP KV server and yield its base URL."""
    port = _find_free_port()
    store: dict[str, str] = {}
    lock = threading.Lock()

    # Create a handler class bound to this specific store
    handler_class = type(
        "_BoundKVHandler",
        (_KVHandler,),
        {"store": store, "store_lock": lock},
    )

    server = _ThreadedHTTPServer(("127.0.0.1", port), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    # Verify server is up
    for _ in range(10):
        try:
            _http_post(base_url, "_healthcheck", '"ok"')
            break
        except (ConnectionRefusedError, urllib.error.URLError):
            import time

            time.sleep(0.1)
    else:
        server.shutdown()
        pytest.skip("Could not start HTTP server")

    yield base_url, store, lock
    server.shutdown()
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# 1. Lost-update counter via HTTP GET/POST
# ---------------------------------------------------------------------------
# Two threads each GET a counter from the server, increment in Python,
# and POST the new value back.  Classic read-modify-write race.


class TestHttpCounterRace:
    """Lost-update race on an HTTP-backed counter."""

    def test_dpor_detects_lost_update(self, http_server: tuple[str, dict[str, str], threading.Lock]) -> None:
        """DPOR should detect the lost-update race via I/O interdiction."""
        base_url, store, lock = http_server

        class State:
            def __init__(self) -> None:
                with lock:
                    store.clear()
                _http_post(base_url, "counter", "0")

        def increment(state: State) -> None:
            val = int(_http_get(base_url, "counter"))
            _http_post(base_url, "counter", str(val + 1))

        def invariant(state: State) -> bool:
            return int(_http_get(base_url, "counter")) == 2

        result = explore_dpor(
            setup=State,
            threads=[increment, increment],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect lost-update on HTTP counter"
        assert result.explanation is not None

    def test_locked_counter_is_safe(self, http_server: tuple[str, dict[str, str], threading.Lock]) -> None:
        """Counter protected by a Python lock across the full read-modify-write."""
        base_url, store, lock = http_server

        class State:
            def __init__(self) -> None:
                self.app_lock = threading.Lock()
                with lock:
                    store.clear()
                _http_post(base_url, "counter", "0")

        def increment(state: State) -> None:
            with state.app_lock:
                val = int(_http_get(base_url, "counter"))
                _http_post(base_url, "counter", str(val + 1))

        def invariant(state: State) -> bool:
            return int(_http_get(base_url, "counter")) == 2

        result = explore_dpor(
            setup=State,
            threads=[increment, increment],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation

    def test_naive_threading_race_rate(self, http_server: tuple[str, dict[str, str], threading.Lock]) -> None:
        """Verify the race manifests intermittently with plain threads.

        Threads start with a random 0-15ms offset to model realistic
        request arrival timing.  HTTP round-trips are slow enough that
        simultaneous starts almost always race, so the offset brings the
        rate down to a realistic range.
        """
        import random
        import time

        base_url, store, lock = http_server
        trials = 500
        failures = 0
        rng = random.Random(42)

        for _ in range(trials):
            with lock:
                store.clear()
            _http_post(base_url, "counter", "0")

            def increment() -> None:
                val = int(_http_get(base_url, "counter"))
                _http_post(base_url, "counter", str(val + 1))

            t1 = threading.Thread(target=increment)
            t2 = threading.Thread(target=increment)
            t1.start()
            # Random offset models realistic request arrival timing.
            time.sleep(rng.uniform(0, 0.015))
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

            if int(_http_get(base_url, "counter")) != 2:
                failures += 1

        rate = failures / trials * 100
        assert failures > 0, f"Race never triggered in {trials} trials"


# ---------------------------------------------------------------------------
# 2. Stale-read configuration update
# ---------------------------------------------------------------------------
# Two threads both read a JSON config via GET, each update a different
# field, and POST the whole config back.  The second writer overwrites
# the first writer's field — a lost update on a compound document.


class TestHttpConfigRace:
    """Lost-update race on concurrent JSON config updates via HTTP."""

    def test_dpor_detects_config_overwrite(self, http_server: tuple[str, dict[str, str], threading.Lock]) -> None:
        """DPOR should detect that concurrent config updates lose a field."""
        base_url, store, lock = http_server

        class State:
            def __init__(self) -> None:
                with lock:
                    store.clear()
                _http_post(base_url, "config", json.dumps({}))

        def update_field_a(state: State) -> None:
            config = json.loads(_http_get(base_url, "config"))
            config["feature_a"] = True
            _http_post(base_url, "config", json.dumps(config))

        def update_field_b(state: State) -> None:
            config = json.loads(_http_get(base_url, "config"))
            config["feature_b"] = True
            _http_post(base_url, "config", json.dumps(config))

        def invariant(state: State) -> bool:
            config = json.loads(_http_get(base_url, "config"))
            return config.get("feature_a") is True and config.get("feature_b") is True

        result = explore_dpor(
            setup=State,
            threads=[update_field_a, update_field_b],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect config field overwrite"
        assert result.explanation is not None

    def test_locked_config_update_is_safe(self, http_server: tuple[str, dict[str, str], threading.Lock]) -> None:
        """Config update protected by a Python lock preserves both fields."""
        base_url, store, lock = http_server

        class State:
            def __init__(self) -> None:
                self.app_lock = threading.Lock()
                with lock:
                    store.clear()
                _http_post(base_url, "config", json.dumps({}))

        def update_field(state: State, field: str) -> None:
            with state.app_lock:
                config = json.loads(_http_get(base_url, "config"))
                config[field] = True
                _http_post(base_url, "config", json.dumps(config))

        def invariant(state: State) -> bool:
            config = json.loads(_http_get(base_url, "config"))
            return config.get("feature_a") is True and config.get("feature_b") is True

        result = explore_dpor(
            setup=State,
            threads=[
                lambda s: update_field(s, "feature_a"),
                lambda s: update_field(s, "feature_b"),
            ],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 3. Balance transfer via HTTP — conservation invariant
# ---------------------------------------------------------------------------
# Two threads transfer different amounts from account A to account B
# by reading both balances via GET, computing new values, and writing
# them back via POST.  The second writer overwrites the first's update,
# violating total balance conservation.


class TestHttpTransferRace:
    """Lost-update race on HTTP-backed balance transfers."""

    def test_dpor_detects_transfer_anomaly(self, http_server: tuple[str, dict[str, str], threading.Lock]) -> None:
        """DPOR should detect the lost update in concurrent transfers."""
        base_url, store, lock = http_server

        class State:
            def __init__(self) -> None:
                with lock:
                    store.clear()
                _http_post(base_url, "balance_a", "100")
                _http_post(base_url, "balance_b", "100")

        def transfer_10(state: State) -> None:
            bal_a = int(_http_get(base_url, "balance_a"))
            bal_b = int(_http_get(base_url, "balance_b"))
            _http_post(base_url, "balance_a", str(bal_a - 10))
            _http_post(base_url, "balance_b", str(bal_b + 10))

        def transfer_30(state: State) -> None:
            bal_a = int(_http_get(base_url, "balance_a"))
            bal_b = int(_http_get(base_url, "balance_b"))
            _http_post(base_url, "balance_a", str(bal_a - 30))
            _http_post(base_url, "balance_b", str(bal_b + 30))

        def invariant(state: State) -> bool:
            bal_a = int(_http_get(base_url, "balance_a"))
            bal_b = int(_http_get(base_url, "balance_b"))
            # Total must be conserved: 100 + 100 = 200
            return bal_a + bal_b == 200

        result = explore_dpor(
            setup=State,
            threads=[transfer_10, transfer_30],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect transfer lost-update"
        assert result.explanation is not None

    def test_locked_transfer_is_safe(self, http_server: tuple[str, dict[str, str], threading.Lock]) -> None:
        """Transfer protected by a Python lock preserves conservation."""
        base_url, store, lock = http_server

        class State:
            def __init__(self) -> None:
                self.app_lock = threading.Lock()
                with lock:
                    store.clear()
                _http_post(base_url, "balance_a", "100")
                _http_post(base_url, "balance_b", "100")

        def transfer(state: State, amount: int) -> None:
            with state.app_lock:
                bal_a = int(_http_get(base_url, "balance_a"))
                bal_b = int(_http_get(base_url, "balance_b"))
                _http_post(base_url, "balance_a", str(bal_a - amount))
                _http_post(base_url, "balance_b", str(bal_b + amount))

        def invariant(state: State) -> bool:
            bal_a = int(_http_get(base_url, "balance_a"))
            bal_b = int(_http_get(base_url, "balance_b"))
            return bal_a + bal_b == 200

        result = explore_dpor(
            setup=State,
            threads=[
                lambda s: transfer(s, 10),
                lambda s: transfer(s, 30),
            ],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation


# ---------------------------------------------------------------------------
# 4. Check-then-act: double purchase
# ---------------------------------------------------------------------------
# Two threads check available stock via GET, and if sufficient, POST
# a purchase and decrement stock.  Both can see stock available and
# both purchase, overselling.


class TestHttpInventoryRace:
    """Double-purchase race on HTTP-backed inventory."""

    def test_dpor_detects_oversell(self, http_server: tuple[str, dict[str, str], threading.Lock]) -> None:
        """DPOR should detect the oversell race."""
        base_url, store, lock = http_server

        class State:
            def __init__(self) -> None:
                with lock:
                    store.clear()
                _http_post(base_url, "stock", "1")
                _http_post(base_url, "sold", "0")

        def buy(state: State) -> None:
            stock = int(_http_get(base_url, "stock"))
            if stock > 0:
                _http_post(base_url, "stock", str(stock - 1))
                sold = int(_http_get(base_url, "sold"))
                _http_post(base_url, "sold", str(sold + 1))

        def invariant(state: State) -> bool:
            stock = int(_http_get(base_url, "stock"))
            sold = int(_http_get(base_url, "sold"))
            return stock >= 0 and sold <= 1

        result = explore_dpor(
            setup=State,
            threads=[buy, buy],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert not result.property_holds, "DPOR should detect oversell via HTTP"
        assert result.explanation is not None

    def test_locked_buy_is_safe(self, http_server: tuple[str, dict[str, str], threading.Lock]) -> None:
        """Purchase protected by a Python lock prevents oversell."""
        base_url, store, lock = http_server

        class State:
            def __init__(self) -> None:
                self.app_lock = threading.Lock()
                with lock:
                    store.clear()
                _http_post(base_url, "stock", "1")
                _http_post(base_url, "sold", "0")

        def buy(state: State) -> None:
            with state.app_lock:
                stock = int(_http_get(base_url, "stock"))
                if stock > 0:
                    _http_post(base_url, "stock", str(stock - 1))
                    sold = int(_http_get(base_url, "sold"))
                    _http_post(base_url, "sold", str(sold + 1))

        def invariant(state: State) -> bool:
            stock = int(_http_get(base_url, "stock"))
            sold = int(_http_get(base_url, "sold"))
            return stock >= 0 and sold <= 1

        result = explore_dpor(
            setup=State,
            threads=[buy, buy],
            invariant=invariant,
            detect_io=True,
            max_executions=50,
            deadlock_timeout=15.0,
            reproduce_on_failure=0,
        )
        assert result.property_holds, result.explanation
