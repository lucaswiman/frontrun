"""Length-prefixed JSON wire protocol for cross-process DPOR exploration.

One message is exchanged per external-access scheduling point between a worker
(:class:`~frontrun._dpor_runtime.xproc.proxy.SchedulerProxy`) and the
coordinator. JSON over the standard library keeps Phase 1 dependency-free; a
msgpack codec can drop in behind :func:`send_msg` / :func:`recv_msg` later if
framing overhead ever shows up (it is negligible next to the DB call each
message brackets).

Every frame is a ``{"t": <type>, ...}`` object prefixed by its big-endian
``uint32`` byte length. The ``"w"`` field, where present, is the worker's dense
logical id (``0..n-1``) — the same integer the DPOR engine uses as a thread id.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

# Worker -> coordinator
HELLO = "hello"  # first frame on connect; announces this worker's id
ACCESS = "access"  # an io-reporter (resource_id, kind) access report
REPORT_AND_WAIT = "report_and_wait"  # request a scheduling turn; expects GRANT/ABORT
ACQUIRE_LOCKS = "acquire_locks"  # block until row locks are held; expects GRANT/ABORT
RELEASE_LOCKS = "release_locks"  # drop all held row locks (fire-and-forget)
DONE = "done"  # worker finished (fire-and-forget)
ERROR = "error"  # worker raised (fire-and-forget)

# Coordinator -> worker
GRANT = "grant"  # turn granted; proceed
ABORT = "abort"  # exploration finished/errored; unwind

_LEN = struct.Struct(">I")


def send_msg(sock: socket.socket, msg: dict[str, Any]) -> None:
    """Serialise *msg* as a length-prefixed JSON frame and send it whole."""
    payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    sock.sendall(_LEN.pack(len(payload)) + payload)


def recv_msg(sock: socket.socket) -> dict[str, Any] | None:
    """Read one frame from *sock*. Return ``None`` if the peer closed cleanly."""
    header = _recv_exactly(sock, _LEN.size)
    if header is None:
        return None
    (length,) = _LEN.unpack(header)
    body = _recv_exactly(sock, length)
    if body is None:
        return None
    decoded: dict[str, Any] = json.loads(body.decode("utf-8"))
    return decoded


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly *n* bytes, reassembling across short reads. ``None`` on EOF."""
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
