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
    """Show the banner (unless turned off) and the main menu, until Quit or Ctrl+C."""
    from qr.tui.banner import banner_disabled, print_banner
    from qr.tui.menu import Menu, QuestionaryPrompter

    console = console or Console()
    if not banner_disabled(no_banner):
        print_banner(console)
    Menu(QuestionaryPrompter(), console).run()
