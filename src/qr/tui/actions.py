"""What the menu runs: the normal ``qr`` commands, with arguments the menu collected.

Every menu action becomes an :class:`Invocation`: the command line it is equivalent to
(printed before it runs, so you can copy it into a script) and the keyword arguments for
the command's own function in :mod:`qr.cli`. The menu calls that function directly, so it
does exactly what the command does; the tests check that each printed command line parses
to the same arguments.
"""

from __future__ import annotations

import shlex
import webbrowser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

# A choice of symbols, as the data commands take it.
ALL = "all"
GROUP = "universe"
PICK = "symbols"


@dataclass(frozen=True)
class Scope:
    """Which symbols a data command acts on: all of them, a named group, or a list."""

    kind: str = ALL
    group: str | None = None
    symbols: tuple[str, ...] = ()

    def argv(self) -> list[str]:
        if self.kind == GROUP:
            return ["--universe", str(self.group)]
        if self.kind == PICK:
            return ["--symbols", ",".join(self.symbols)]
        return []

    def kwargs(self) -> dict[str, Any]:
        return {
            "symbols": ",".join(self.symbols) if self.kind == PICK else None,
            "universe": self.group if self.kind == GROUP else None,
        }


@dataclass(frozen=True)
class Invocation:
    """A ``qr`` command: its arguments as typed, and the call that runs it."""

    argv: tuple[str, ...]
    command: str  # name of the command's function in qr.cli
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def command_line(self) -> str:
        return "qr " + shlex.join(self.argv)


def _common(**kwargs: Any) -> dict[str, Any]:
    """The options the menu never sets, at their command-line defaults."""
    return {"config": None, "data_dir": None, **kwargs}


# --------------------------------------------------------------------------- builders


def ingest(scope: Scope) -> Invocation:
    kwargs = _common(**scope.kwargs(), backfill=False)
    return Invocation(("ingest", *scope.argv()), "ingest", kwargs)


def validate(scope: Scope) -> Invocation:
    return Invocation(("validate", *scope.argv()), "validate", _common(**scope.kwargs()))


def status() -> Invocation:
    return Invocation(("status",), "status", _common(universe=None))


def crosscheck(scope: Scope, days: int) -> Invocation:
    argv = ("crosscheck", *scope.argv(), "--days", str(days))
    return Invocation(argv, "crosscheck", _common(**scope.kwargs(), days=days))


def export(symbol: str, adjustment: str, out: str) -> Invocation:
    argv = ("export", symbol, "--out", out, "--adjustment", adjustment)
    kwargs = {"symbol": symbol, "out": Path(out), "adjustment": adjustment, "data_dir": None}
    return Invocation(argv, "export", kwargs)


def backtest_run(
    config_file: str, engine: str | None = None, engine_core: str | None = None
) -> Invocation:
    argv = ["backtest", "run", config_file]
    if engine is not None:
        argv += ["--engine", engine]
    if engine_core is not None:
        argv += ["--engine-core", engine_core]
    kwargs = _common(
        config_file=Path(config_file),
        notes=None,
        reports_dir=None,
        engine=engine,
        engine_core=engine_core,
    )
    return Invocation(tuple(argv), "backtest_run", kwargs)


def backtest_list(strategy: str | None = None) -> Invocation:
    argv = ("backtest", "list", *(("--strategy", strategy) if strategy else ()))
    kwargs = {"strategy": strategy, "limit": 20, "data_dir": None}
    return Invocation(argv, "backtest_list", kwargs)


def backtest_compare(run_ids: Sequence[str]) -> Invocation:
    kwargs = {"run_ids": list(run_ids), "data_dir": None}
    return Invocation(("backtest", "compare", *run_ids), "backtest_compare", kwargs)


# --------------------------------------------------------------------------- running


def run(invocation: Invocation, console: Console) -> int:
    """Print the equivalent command, run it, and return its exit code.

    Errors are reported and the menu carries on; ``KeyboardInterrupt`` (Ctrl+C) is left to
    the menu, which goes back one level.
    """
    from qr import cli

    console.print(f"[dim]Running: {invocation.command_line}[/dim]", highlight=False)
    function = getattr(cli, invocation.command)
    try:
        function(**invocation.kwargs)
    except typer.Exit as exc:
        code = int(exc.exit_code)
    except Exception as exc:  # the menu must survive a failing command
        console.print(f"[red]error:[/red] {type(exc).__name__}: {exc}", highlight=False)
        return 1
    else:
        code = 0
    if code:
        console.print(f"[yellow]Finished with exit code {code}.[/yellow]")
    return code


def open_report(path: Path, console: Console) -> bool:
    """Open an HTML tearsheet in the default browser."""
    console.print(f"[dim]Opening: {path}[/dim]", highlight=False)
    if not path.is_file():
        console.print(f"[red]error:[/red] the report {path} no longer exists")
        return False
    return webbrowser.open(path.resolve().as_uri())


def show_help() -> None:
    """The output of ``qr --help``."""
    from typer.main import get_command

    from qr import cli

    command = get_command(cli.app)
    with command.make_context("qr", [], resilient_parsing=True) as context:
        command.get_help(context)  # Typer prints the help itself
