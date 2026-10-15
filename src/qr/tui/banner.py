"""The logo at the top of the interactive menu, and the line of run information under it.

Shown only by the menu, never by normal commands. ``qr menu --no-banner`` or
``QR_NO_BANNER=1`` turns it off; ``NO_COLOR`` turns colours off (rich and the prompts both
honour it).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version

import pyfiglet
from rich.console import Console
from rich.text import Text

TITLE = "ZHUFU QUANT"
# "slant": 5 lines tall, 67 columns wide for the title, plain ASCII (fine on Windows too).
FONT = "slant"
MAX_HEIGHT = 8
MAX_WIDTH = 80


def banner_disabled(no_banner: bool = False, environ: Mapping[str, str] | None = None) -> bool:
    """Whether to skip the logo: ``--no-banner`` or a non-empty ``QR_NO_BANNER`` other than 0."""
    env = os.environ if environ is None else environ
    return no_banner or env.get("QR_NO_BANNER", "") not in ("", "0")


def colour_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """False when ``NO_COLOR`` is set to anything non-empty (https://no-color.org)."""
    env = os.environ if environ is None else environ
    return env.get("NO_COLOR", "") == ""


def logo_lines(width: int) -> list[str]:
    """The figlet logo, or the plain title when the terminal is narrower than the logo."""
    rendered = pyfiglet.figlet_format(TITLE, font=FONT, width=1000)
    lines = [line.rstrip() for line in rendered.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    widest = max((len(line) for line in lines), default=0)
    if widest > width or widest > MAX_WIDTH or len(lines) > MAX_HEIGHT:
        return [TITLE]
    return lines


def current_branch() -> str | None:
    """The git branch of the current folder, if it is a git checkout."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    branch = done.stdout.strip()
    return branch if done.returncode == 0 and branch else None


def active_engine_core(environ: Mapping[str, str] | None = None) -> str:
    """The event-engine core runs use by default, as a short description."""
    from qr.backtest.rust_core import CORE_ENV_VAR, rust_available

    env = os.environ if environ is None else environ
    core = env.get(CORE_ENV_VAR) or "python"
    note = f"from {CORE_ENV_VAR}" if env.get(CORE_ENV_VAR) else "default"
    if core == "rust" and not rust_available():
        note += "; not built, runs fall back to python"
    return f"{core} ({note})"


def info_line(environ: Mapping[str, str] | None = None) -> str:
    """Version, git branch, and active engine core, for the dim line under the logo."""
    try:
        qr_version = version("qr")
    except PackageNotFoundError:
        qr_version = "unknown"
    parts = [f"qr {qr_version}"]
    branch = current_branch()
    if branch:
        parts.append(f"branch {branch}")
    parts.append(f"engine_core {active_engine_core(environ)}")
    return "  ·  ".join(parts)


def print_banner(console: Console, environ: Mapping[str, str] | None = None) -> None:
    """Print the logo, sized to the console, and the information line under it."""
    style = "bold cyan" if colour_enabled(environ) else ""
    console.print(Text("\n".join(logo_lines(console.width)), style=style), highlight=False)
    console.print(Text(info_line(environ), style="dim"), highlight=False)
    console.print()
