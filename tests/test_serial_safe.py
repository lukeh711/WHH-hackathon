"""The serial write lock: the change that allows gesturing during speech."""

from __future__ import annotations

import threading
import time

from ohbot_kit import serial_safe


def test_lock_is_reentrant() -> None:
    """REGRESSION: read_sensor holds the lock across a whole write-then-read
    transaction, and readSensor's own inner _serwrite then re-acquires it.
    With a plain Lock that deadlocks and the program hangs forever."""
    with serial_safe.lock:
        acquired = serial_safe.lock.acquire(timeout=1)
        assert acquired, "lock is not reentrant -- read_sensor will deadlock"
        serial_safe.lock.release()


def test_install_is_idempotent() -> None:
    serial_safe.install_write_lock()
    first = __import__("ohbot", fromlist=["ohbot"]).ohbot._serwrite
    serial_safe.install_write_lock()
    second = __import__("ohbot", fromlist=["ohbot"]).ohbot._serwrite
    assert first is second, "double-wrapping would nest locks on every call"
    assert serial_safe.is_installed()


def test_writes_do_not_interleave() -> None:
    """Message-level atomicity is the whole point: a command cut in half by
    another thread's bytes is a corrupt motor instruction."""
    from ohbot import ohbot

    serial_safe.uninstall_write_lock()
    serial_safe.install_write_lock()

    seen: list[str] = []
    original = ohbot._serwrite

    def slow_write(s: str) -> None:
        seen.append("start:" + s)
        time.sleep(0.005)  # widen the window a real write would have
        seen.append("end:" + s)

    try:
        serial_safe.uninstall_write_lock()
        ohbot._serwrite = slow_write
        serial_safe.install_write_lock()

        threads = [
            threading.Thread(target=lambda i=i: ohbot._serwrite(f"m0{i}\n")) for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        serial_safe.uninstall_write_lock()
        ohbot._serwrite = original
        serial_safe.install_write_lock()

    # Every start must be followed by its own end, never another's start.
    for i in range(0, len(seen), 2):
        assert seen[i].startswith("start:")
        assert seen[i + 1] == "end:" + seen[i][len("start:") :]
