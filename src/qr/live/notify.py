"""Optional desktop alerts via ``notify-send`` -- never required, never blocking.

The journal and the log are the record of truth (``qr.live.journal.record_alert``); this
is only a courtesy nudge on top of it, so a missing or failing notification must never
raise, and never stops a scheduled run.
"""

from __future__ import annotations

import shutil
import subprocess

_TIMEOUT_SECONDS = 5.0


def send(title: str, body: str = "") -> bool:
    """Best-effort desktop notification. Returns whether one was actually sent."""
    exe = shutil.which("notify-send")
    if exe is None:
        return False
    try:
        subprocess.run([exe, "--", title, body], check=False, timeout=_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return False
    return True
