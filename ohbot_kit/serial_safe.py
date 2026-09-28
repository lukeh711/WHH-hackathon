"""Make Ohbot's serial writes thread-safe.

WHY THIS EXISTS:

ohbot._serwrite() serialises concurrent writes with a `writing` boolean, but
only on Windows and Linux -- on macOS there is no guard at all. Meanwhile
say() drives the lip motor from its own thread while audio plays. So any second
thread moving motors during speech can interleave with the lip-sync stream and
corrupt motor commands.

That is why idle motion used to be frozen during speech. But freezing motion is
exactly the wrong trade for an expressive robot: gesturing WHILE speaking is
what makes it read as present rather than as a speaker with a face.

The fix is small because every one of the library's 21 serial writes funnels
through _serwrite, and a standard move() emits exactly one newline-terminated
message per call:

    m0<motor>,<position>,<speed>\\n

So a lock around _serwrite gives message-level atomicity. Two threads may still
interleave whole commands -- that is fine, the firmware parses line by line --
but a command can no longer be cut in half by another thread's bytes.

Install this BEFORE starting any thread that moves motors.
"""

from __future__ import annotations

import threading

from ohbot import ohbot

# Reentrant on purpose. A caller may hold the lock across a whole
# write-then-read transaction (readSensor does exactly that), and the inner
# _serwrite would then deadlock on a plain Lock.
lock = threading.RLock()

_original_serwrite = ohbot._serwrite
_installed = False


def install_write_lock() -> threading.RLock:
    """Wrap ohbot._serwrite so concurrent writers can't interleave mid-message.

    Idempotent: calling it twice will not double-wrap.
    """
    global _installed
    if _installed:
        return lock

    def _locked_serwrite(s: str) -> None:
        with lock:
            _original_serwrite(s)

    ohbot._serwrite = _locked_serwrite
    _installed = True
    return lock


def uninstall_write_lock() -> None:
    """Restore the unguarded write. Mainly here to prove the lock matters."""
    global _installed
    ohbot._serwrite = _original_serwrite
    _installed = False


def is_installed() -> bool:
    return _installed
