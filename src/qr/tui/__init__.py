"""The interactive menu that ``qr`` opens when run on its own in a terminal.

A convenience layer over the normal commands: it asks with arrow keys instead of flags,
prints the equivalent ``qr ...`` command, and runs the command's own code.
"""

from __future__ import annotations

import sys

from rich.console import Console


def is_interactive() -> bool:
    """True only when both stdin and stdout are terminals (never in scripts or CI)."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def run(no_banner: bool = False, console: Console | None = None) -> None:
    """The main menu, with the logo on top unless turned off, until Quit or Ctrl+C.

    In a terminal the menu uses the alternate screen (like vim or less): each screen
    replaces the last one, and on exit the terminal shows exactly what it showed before.
    """
    from qr.tui.banner import banner_disabled
    from qr.tui.menu import Menu, QuestionaryPrompter

    console = console or Console()
    alternate = console.is_terminal and console.set_alt_screen(True)
    try:
        Menu(QuestionaryPrompter(), console, banner=not banner_disabled(no_banner)).run()
    finally:
        if alternate:
            console.set_alt_screen(False)
