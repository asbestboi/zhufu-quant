"""The interactive menu: arrow-key navigation over the ``qr`` commands.

Every screen asks its questions through a :class:`Prompter` (questionary in a terminal,
a scripted stand-in in tests), turns the answers into an
:class:`~qr.tui.actions.Invocation`, and hands it to :func:`~qr.tui.actions.run`.

Navigation: every list has a Back entry, and Ctrl+C goes back one level (out of a question
or a running command back to its menu, out of a menu back to the main menu, and out of the
main menu to the shell, with exit code 0).

Every screen starts from a clean terminal: the menu clears it and redraws the logo, where
you are (``Main menu > Data``), and the list, so answered questions and old output do not
pile up. A command's output stays until you press a key.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text

from qr.backtest.config import load_backtest_config
from qr.backtest.registry import read_runs
from qr.backtest.rust_core import rust_available
from qr.backtest.strategies import STRATEGIES
from qr.config import ConfigError, UniverseConfig, config_path, data_dir, load_config
from qr.tui import actions
from qr.tui.actions import ALL, GROUP, PICK, Invocation, Scope

BACK = "← Back"
BREADCRUMB = " \u203a "  # a single right-pointing angle quote between the levels
QUIT = "Quit"
CONFIG_FOLDERS = (Path("configs/backtests"), Path("configs/private"))
TIINGO_HOURLY_LIMIT = 50  # free tier, requests an hour
RECENT_RUNS = 20


class Prompter(Protocol):
    """The questions the menu can ask. Each raises ``KeyboardInterrupt`` on Ctrl+C."""

    def select(self, message: str, choices: list[str], default: str | None = None) -> str: ...

    def checkbox(self, message: str, choices: list[str]) -> list[str]: ...

    def confirm(self, message: str, default: bool = True) -> bool: ...

    def text(
        self, message: str, default: str = "", validate: Callable[[str], bool | str] | None = None
    ) -> str: ...

    def pause(self, message: str) -> None: ...


class QuestionaryPrompter:
    """The real prompts, drawn by questionary (arrow keys, space to tick, Enter to accept)."""

    def select(self, message: str, choices: list[str], default: str | None = None) -> str:
        import questionary

        answer: str = questionary.select(message, choices, default=default).unsafe_ask()
        return answer

    def checkbox(self, message: str, choices: list[str]) -> list[str]:
        import questionary

        answer: list[str] = questionary.checkbox(message, choices).unsafe_ask()
        return answer

    def confirm(self, message: str, default: bool = True) -> bool:
        import questionary

        return bool(questionary.confirm(message, default=default).unsafe_ask())

    def text(
        self, message: str, default: str = "", validate: Callable[[str], bool | str] | None = None
    ) -> str:
        import questionary

        answer: str = questionary.text(message, default=default, validate=validate).unsafe_ask()
        return answer

    def pause(self, message: str) -> None:
        import questionary

        questionary.press_any_key_to_continue(message).unsafe_ask()


@dataclass
class Menu:
    """The menu's state: where to ask questions, where to print, and where we are."""

    prompter: Prompter
    console: Console
    banner: bool = False  # draw the logo at the top of every screen
    path: list[str] = field(default_factory=list)  # the breadcrumb, e.g. Main menu > Data
    notice: str = ""  # a one-off message for the top of the next screen
    _info: str | None = None

    # ------------------------------------------------------------------ screens

    def screen(self) -> None:
        """Clear the terminal and draw the header: logo, breadcrumb, and any notice."""
        from qr.tui.banner import colour_enabled, info_line, logo_lines

        self.console.clear()
        if self.banner:
            style = "bold cyan" if colour_enabled() else ""
            self.console.print(Text("\n".join(logo_lines(self.console.width)), style=style))
            if self._info is None:  # the git lookup is done once, not on every screen
                self._info = info_line()
            self.console.print(Text(self._info, style="dim"))
            self.console.print()
        if self.path:
            self.console.print(Text(BREADCRUMB.join(self.path), style="bold"))
        if self.notice:
            self.console.print(Text(self.notice, style="dim"))
            self.notice = ""
        self.console.print()

    # ------------------------------------------------------------------ navigation

    def run(self) -> None:
        """The main menu, until Quit or Ctrl+C."""
        sections: dict[str, Callable[[], None]] = {
            "Data": self.data_menu,
            "Backtests": self.backtest_menu,
            "Settings / info": self.settings_menu,
        }
        self.path = ["Main menu"]
        while True:
            self.screen()
            try:
                choice = self.prompter.select("Main menu", [*sections, QUIT])
            except KeyboardInterrupt:
                return
            if choice == QUIT:
                return
            try:
                sections[choice]()
            except KeyboardInterrupt:
                self.notice = "Back to the main menu."

    def submenu(self, title: str, items: dict[str, Callable[[], None]]) -> None:
        """A list of actions with a Back entry. Ctrl+C here goes back to the main menu."""
        self.path.append(title)
        try:
            while True:
                self.screen()
                choice = self.prompter.select(title, [*items, BACK])
                if choice == BACK:
                    return
                self.path.append(choice)
                self.screen()  # the action's questions and output start on a clean screen
                try:
                    items[choice]()
                except KeyboardInterrupt:
                    self.notice = f"Cancelled. Back to {title}."
                finally:
                    self.path.pop()
        finally:
            self.path.pop()

    def data_menu(self) -> None:
        self.submenu(
            "Data",
            {
                "Update data (ingest)": self.ingest,
                "Validate data": self.validate,
                "Coverage status": self.status,
                "Cross-check against Tiingo": self.crosscheck,
                "Export a symbol to CSV": self.export,
            },
        )

    def backtest_menu(self) -> None:
        self.submenu(
            "Backtests",
            {
                "Run a backtest": self.backtest_run,
                "List recent runs": self.backtest_list,
                "Compare runs": self.backtest_compare,
                "Open a report": self.open_report,
            },
        )

    def settings_menu(self) -> None:
        self.submenu(
            "Settings / info",
            {
                "Show config summary": self.config_summary,
                "Show available commands": self.show_commands,
            },
        )

    # ------------------------------------------------------------------ helpers

    def execute(self, invocation: Invocation) -> None:
        """Run a command, then wait for Enter so its output can be read."""
        actions.run(invocation, self.console)
        self.done()

    def done(self) -> None:
        self.prompter.pause("Done. Press any key to return to the menu.")

    def universe(self) -> UniverseConfig | None:
        try:
            return load_config()
        except ConfigError as exc:
            self.console.print(f"[red]error:[/red] {exc}")
            self.done()
            return None

    def ask_scope(self, cfg: UniverseConfig, verb: str) -> Scope | None:
        """All symbols, a universe group, or a hand-picked list; None for Back."""
        everything = f"All symbols ({len(cfg.universe)})"
        options = [everything]
        if cfg.groups:
            options.append("A universe group")
        options += ["Pick symbols", BACK]
        choice = self.prompter.select(f"{verb} which symbols?", options)
        if choice == BACK:
            return None
        if choice == everything:
            return Scope(ALL)
        if choice == "A universe group":
            labels = {f"{name} ({len(members)})": name for name, members in cfg.groups.items()}
            picked = self.prompter.select("Which group?", [*labels, BACK])
            return None if picked == BACK else Scope(GROUP, group=labels[picked])
        labels = {f"{i.symbol} ({i.asset_class})": i.symbol for i in cfg.universe}
        ticked = self.prompter.checkbox("Pick symbols (space to tick, Enter to accept)", [*labels])
        if not ticked:
            self.console.print("Nothing picked.")
            return None
        return Scope(PICK, symbols=tuple(labels[label] for label in ticked))

    def scope_size(self, cfg: UniverseConfig, scope: Scope) -> int:
        if scope.kind == GROUP and scope.group is not None:
            return len(cfg.group(scope.group))
        return len(scope.symbols) if scope.kind == PICK else len(cfg.universe)

    # ------------------------------------------------------------------ data

    def ingest(self) -> None:
        cfg = self.universe()
        if cfg is None or (scope := self.ask_scope(cfg, "Update")) is None:
            return
        if scope.kind == ALL:
            question = (
                f"Download new data for all {len(cfg.universe)} symbols? "
                "The first run can take several minutes."
            )
            if not self.prompter.confirm(question, default=True):
                return
        self.execute(actions.ingest(scope))

    def validate(self) -> None:
        cfg = self.universe()
        if cfg is None or (scope := self.ask_scope(cfg, "Validate")) is None:
            return
        self.execute(actions.validate(scope))

    def status(self) -> None:
        self.execute(actions.status())

    def crosscheck(self) -> None:
        cfg = self.universe()
        if cfg is None or (scope := self.ask_scope(cfg, "Cross-check")) is None:
            return
        days = int(
            self.prompter.text(
                "Compare how many calendar days?", default="365", validate=_positive_int
            )
        )
        if scope.kind == ALL:
            count = sum(1 for i in cfg.universe if i.source == "yahoo")
        else:
            count = self.scope_size(cfg, scope)
        requests = count + 1  # one per symbol, plus a check that Tiingo answers
        if not os.environ.get("TIINGO_API_KEY", "").strip():
            self.console.print(
                "[yellow]TIINGO_API_KEY is not set;[/yellow] the command will explain how to "
                "get a free key."
            )
        elif requests > 10:
            warning = (
                f"This uses about {requests} Tiingo requests; the free tier allows about "
                f"{TIINGO_HOURLY_LIMIT} an hour."
            )
            if requests > TIINGO_HOURLY_LIMIT:
                warning += " It will run out part-way: consider a smaller group."
            if not self.prompter.confirm(f"{warning} Continue?", default=requests <= 50):
                return
        self.execute(actions.crosscheck(scope, days))

    def export(self) -> None:
        cfg = self.universe()
        if cfg is None:
            return
        symbol = self.prompter.select(
            "Export which symbol?", [i.symbol for i in cfg.universe] + [BACK]
        )
        if symbol == BACK:
            return
        adjustment = self.prompter.select(
            "Which prices?", ["total_return", "split", "none", BACK], default="total_return"
        )
        if adjustment == BACK:
            return
        # Under the data folder by default, so exported market data is never committed.
        default = data_dir() / "exports" / f"{symbol.replace('/', '-')}_{adjustment}.csv"
        out = self.prompter.text("Write to which file?", default=default.as_posix())
        self.execute(actions.export(symbol, adjustment, out.strip()))

    # ------------------------------------------------------------------ backtests

    def backtest_run(self) -> None:
        files = discover_configs()
        if not files:
            self.console.print("No backtest configs found in configs/backtests/.")
            self.done()
            return
        labels = {describe_config(path): path for path in files}
        picked = self.prompter.select("Run which backtest?", [*labels, BACK])
        if picked == BACK:
            return
        path = labels[picked]
        config = load_backtest_config(path)

        entry = STRATEGIES.get(config.strategy)
        implemented = (
            {} if entry is None else {"vectorized": entry.vectorized, "event": entry.event}
        )
        engines = [name for name, impl in implemented.items() if impl is not None]
        keep = f"As in the config ({config.engine})"
        engine_choice = self.prompter.select(
            "Engine?", [keep, *(e for e in engines if e != config.engine), BACK]
        )
        if engine_choice == BACK:
            return
        engine = None if engine_choice == keep else engine_choice

        engine_core = None
        if (engine or config.engine) == "event":
            keep_core = f"As configured ({config.engine_core})"
            rust_label = "rust" if rust_available() else "rust (not built: falls back to python)"
            options = {keep_core: None, "python": "python", rust_label: "rust"}
            core_choice = self.prompter.select("Event-engine core?", [*options, BACK])
            if core_choice == BACK:
                return
            engine_core = options[core_choice]
        self.execute(actions.backtest_run(path.as_posix(), engine, engine_core))

    def recent_runs(self) -> pd.DataFrame:
        return read_runs(limit=RECENT_RUNS)

    def backtest_list(self) -> None:
        strategies = sorted(set(read_runs()["strategy"].dropna()))
        everything = "All strategies"
        choice = self.prompter.select("Which runs?", [everything, *strategies, BACK])
        if choice == BACK:
            return
        self.execute(actions.backtest_list(None if choice == everything else choice))

    def backtest_compare(self) -> None:
        runs = self.recent_runs()
        if len(runs) < 2:
            self.console.print("Fewer than two runs are logged; run some backtests first.")
            self.done()
            return
        labels = {describe_run(row): str(row["run_id"]) for row in runs.to_dict("records")}
        ticked = self.prompter.checkbox("Compare which runs? (space to tick)", [*labels])
        if len(ticked) < 2:
            self.console.print("Pick at least two runs to compare.")
            self.done()
            return
        self.execute(actions.backtest_compare([labels[label] for label in ticked]))

    def open_report(self) -> None:
        runs = self.recent_runs()
        runs = runs[runs["report_path"].notna()]
        if runs.empty:
            self.console.print("No reports yet; run a backtest first.")
            self.done()
            return
        labels = {describe_run(row): Path(row["report_path"]) for row in runs.to_dict("records")}
        picked = self.prompter.select("Open which report?", [*labels, BACK])
        if picked == BACK:
            return
        actions.open_report(labels[picked], self.console)
        self.done()

    # ------------------------------------------------------------------ settings

    def config_summary(self) -> None:
        self.console.print(summary_table())
        self.done()

    def show_commands(self) -> None:
        self.console.print("[dim]Running: qr --help[/dim]", highlight=False)
        actions.show_help()
        self.done()


# --------------------------------------------------------------------------- pure helpers


def _positive_int(text: str) -> bool | str:
    return (text.strip().isdigit() and int(text) > 0) or "Enter a whole number of days, 1 or more"


def discover_configs(folders: tuple[Path, ...] = CONFIG_FOLDERS) -> list[Path]:
    """Backtest configs in ``configs/backtests/`` and, if present, ``configs/private/``."""
    found: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            found += sorted(p for p in folder.rglob("*") if p.suffix in (".yaml", ".yml"))
    return found


def describe_config(path: Path) -> str:
    """A config's path with its strategy and engine, for a menu entry."""
    try:
        config = load_backtest_config(path)
    except ConfigError:
        return f"{path.as_posix()}  (invalid config)"
    return f"{path.as_posix()}  ({config.strategy}, {config.engine})"


def describe_run(row: Mapping[Hashable, Any]) -> str:
    """A registry row as one menu line: id, name, status, Sharpe."""
    sharpe = row.get("sharpe")
    finite = isinstance(sharpe, int | float) and math.isfinite(sharpe)
    shown = f"{sharpe:.2f}" if finite else "-"
    return f"{row['run_id']}  {row['name']}  {row['status']}  Sharpe {shown}"


def summary_table() -> Table:
    """The universe, folders, sources, and whether an API key is set (never the key)."""
    table = Table(title="Configuration", show_header=False)
    table.add_column("setting")
    table.add_column("value")
    table.add_row("universe config", str(config_path()))
    try:
        cfg = load_config()
    except ConfigError as exc:
        table.add_row("problem", str(exc))
        cfg = None
    if cfg is not None:
        table.add_row("symbols", str(len(cfg.universe)))
        for name, members in cfg.groups.items():
            table.add_row(f"group {name}", f"{len(members)}: {', '.join(members)}")
        table.add_row("source yahoo", "enabled")
        crypto = cfg.sources.crypto
        state = f"enabled ({crypto.exchange})" if crypto.enabled else "disabled"
        table.add_row("source crypto", state)
    folder = data_dir()
    table.add_row(
        "data directory", f"{folder} ({'exists' if folder.is_dir() else 'not created yet'})"
    )
    key_set = bool(os.environ.get("TIINGO_API_KEY", "").strip())
    table.add_row("TIINGO_API_KEY", "set" if key_set else "not set")
    from qr.tui.banner import active_engine_core

    table.add_row("engine_core", active_engine_core())
    table.add_row("Rust core module", "built" if rust_available() else "not built")
    return table
