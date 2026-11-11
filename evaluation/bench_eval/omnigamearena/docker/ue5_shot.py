#!/usr/bin/env python3
"""Talk to one UE5 RemoteInput endpoint: optionally open maps, then screenshot.

Used as the readiness / warm-up probe for a single UE5 instance. Exists as a file
rather than a heredoc because the parallel runner drives several instances at once
and each needs the same check against a different port.

env:
  PORT           RemoteInput port (required)
  HOST           default 127.0.0.1
  MAPS           comma-separated maps to `open` before the screenshot (optional)
  WARMUP_SLEEP   seconds to wait after each open for async PSO precompile (60)
  TIMEOUT        socket timeout in seconds (90)

exit: 0 = screenshot received, non-zero otherwise.
"""
import json
import os
import socket
import sys
import time

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ["PORT"])
TIMEOUT = float(os.environ.get("TIMEOUT", "90"))
MAPS = [m for m in os.environ.get("MAPS", "").split(",") if m.strip()]
WARMUP_SLEEP = int(os.environ.get("WARMUP_SLEEP", "60"))


def screenshot(timeout=TIMEOUT):
    """Request one JPEG and return (width, height, nbytes)."""
    buf = b""
    s = socket.socket()
    s.settimeout(timeout)
    s.connect((HOST, PORT))
    try:
        # UE5 may push a greeting; drain it so it is not mistaken for a reply.
        s.settimeout(1.0)
        try:
            buf = s.recv(4096)
        except socket.timeout:
            buf = b""
        s.settimeout(timeout)
        s.sendall(json.dumps({"type": "screenshot", "quality": 50}).encode() + b"\n")

        def read_line():
            nonlocal buf
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    raise ConnectionError("connection closed")
                buf += chunk
            i = buf.index(b"\n")
            raw, buf = buf[:i], buf[i + 1:]
            start = raw.find(b'{"')          # skip any log noise prefixing the JSON
            return raw[start:].decode("utf-8", "replace")

        for _ in range(40):
            head = json.loads(read_line())
            if head.get("type") != "screenshot":
                continue
            n = head["size"]
            while len(buf) < n:
                chunk = s.recv(max(4096, n - len(buf)))
                if not chunk:
                    raise ConnectionError("connection closed mid-image")
                buf += chunk
            return head.get("width"), head.get("height"), n
        raise RuntimeError("no screenshot header in 40 messages")
    finally:
        s.close()


def open_map(name):
    s = socket.socket()
    s.settimeout(10)
    s.connect((HOST, PORT))
    try:
        path = name if "/" in name else "/Game/Maps/" + name
        s.sendall(json.dumps({"type": "console", "command": "open " + path}).encode() + b"\n")
        time.sleep(1)
    finally:
        s.close()


def main():
    for m in MAPS:
        print(f"[shot:{PORT}] open {m}", flush=True)
        try:
            open_map(m)
        except Exception as exc:
            print(f"[shot:{PORT}] open {m} failed: {exc}", flush=True)
        time.sleep(WARMUP_SLEEP)

    last = None
    for attempt in range(3):
        try:
            w, h, n = screenshot()
            print(f"[shot:{PORT}] screenshot OK {w}x{h} bytes={n} (attempt {attempt})", flush=True)
            return 0
        except Exception as exc:
            last = exc
            print(f"[shot:{PORT}] attempt {attempt} failed: {exc}", flush=True)
            time.sleep(5)
    print(f"[shot:{PORT}] FAILED: {last}", flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
