#!/usr/bin/env python
"""A deliberately *unmodified* worker for the Phase 4 cross-process PoC.

Imports only the standard library — no frontrun. It connects to a TCP server and
sends a sequence of messages. When run under the frontrun_xproc_sched LD_PRELOAD
shim, each send() blocks until the coordinator grants the turn, so the
coordinator controls this worker's interleaving without the worker cooperating
in any way.

Usage: raw_socket_worker.py <host> <port> <worker_id> <num_messages>
"""

import socket
import sys


def main() -> None:
    host = sys.argv[1]
    port = int(sys.argv[2])
    worker_id = int(sys.argv[3])
    num_messages = int(sys.argv[4])

    conn = socket.create_connection((host, port))
    try:
        for i in range(num_messages):
            conn.send(f"w{worker_id}-{i}".encode())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
