"""The interactive menu: when it opens, the banner, every action, and Back / Ctrl+C.

Prompts are scripted (no terminal needed), and the command functions are replaced by
recorders, so the tests check *what* the menu would run and how it would print it.
"""

from __future__ import annotations

import inspect
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from rich.console import Console
from typer.testing import CliRunner

from qr import cli, tui
from qr.tui import actions, banner, menu
from qr.tui.actions import Invocation, Scope
from qr.tui.menu import BACK, QUIT, Menu

UNIVERSE = """
start_date: 2015-01-01
universe:
  - { symbol: SPY, source: yahoo, asset_class: etf }
  - { symbol: KO,  source: yahoo, asset_class: equity }
  - { symbol: PEP, source: yahoo, asset_class: equity }
  - { symbol: BTC/USD, source: crypto, asset_class: crypto }
groups:
  pairs: [KO, PEP]
"""


# --------------------------------------------------------------------------- helpers


class Scripted:
    """A Prompter that replays answers. ``KeyboardInterrupt`` as an answer raises it."""

    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def _next(self, message: str) -> Any:
        self.asked.append(message)
        if not self.answers:
            raise AssertionError(f"no scripted answer left for {message!r}")
        answer = self.answers.pop(0)
        if answer is KeyboardInterrupt:
            raise KeyboardInterrupt
        return answer

    def select(self, message: str, choices: list[str], default: str | None = None) -> str:
        answer = self._next(message)
        # A scripted answer may be a prefix of the menu entry, e.g. "All symbols".
        matches = [c for c in choices if c == answer] or [
            c for c in choices if c.startswith(answer)
        ]
        assert len(matches) == 1, f"{answer!r} is not one of {choices}"
        return matches[0]

    def checkbox(self, message: str, choices: list[str]) -> list[str]:
        wanted = self._next(message)
        return [c for c in choices if any(c.startswith(w) for w in wanted)]

    def confirm(self, message: str, default: bool = True) -> bool:
        return bool(self._next(message))

    def text(self, message: str, default: str = "", validate: Any = None) -> str:
        answer = self._next(message)
        return default if answer is None else str(answer)

    def pause(self, message: str) -> None:
        self.asked.append(message)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project folder with a universe config, two backtest configs, and a live config."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "universe.yaml").write_text(UNIVERSE, encoding="utf-8")
    repo = Path(__file__).parent.parent / "configs" / "backtests"
    for folder, name in (("backtests", "sma_spy_event.yaml"), ("private", "pairs_ko_pep.yaml")):
        (tmp_path / "configs" / folder).mkdir(parents=True)
        (tmp_path / "configs" / folder / name).write_text((repo / name).read_text("utf-8"), "utf-8")
    live_repo = Path(__file__).parent.parent / "configs" / "live" / "xs_momentum_demo.yaml"
    (tmp_path / "configs" / "live").mkdir(parents=True)
    (tmp_path / "configs" / "live" / "xs_momentum_demo.yaml").write_text(
        live_repo.read_text("utf-8"), "utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QR_CONFIG", raising=False)
    monkeypatch.delenv("QR_DATA_DIR", raising=False)
    monkeypatch.delenv("QR_ENGINE_CORE", raising=False)
    return tmp_path


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Replace every command function the menu can call with a recorder."""
    seen: list[tuple[str, dict[str, Any]]] = []
    for name in (
        "ingest",
        "validate",
        "status",
        "crosscheck",
        "export",
        "backtest_run",
        "backtest_list",
        "backtest_compare",
        "live_run",
        "live_approve",
        "live_status",
        "live_report_cmd",
        "live_dashboard_cmd",
    ):
        monkeypatch.setattr(cli, name, lambda _n=name, **kw: seen.append((_n, kw)))
    return seen


def play(answers: list[Any]) -> tuple[Scripted, str]:
    prompter = Scripted(answers)
    out = io.StringIO()
    Menu(prompter, Console(file=out, width=120, color_system=None)).run()
    return prompter, out.getvalue()


def cli_kwargs(argv: list[str]) -> tuple[str, dict[str, Any]]:
    """Run ``qr <argv>`` through the real CLI parser and capture what the command receives.

    Each command's registered function is swapped for a recorder with the same signature,
    so Typer parses the arguments exactly as it would for a user.
    """
    infos = [
        *cli.app.registered_commands,
        *cli.backtest_app.registered_commands,
        *cli.live_app.registered_commands,
    ]
    originals = [info.callback for info in infos]
    received: dict[str, Any] = {}

    def recorder_for(original: Callable[..., Any]) -> Callable[..., None]:
        def recorder(**kwargs: Any) -> None:
            received.update(kwargs, _command=original.__name__)

        recorder.__signature__ = inspect.signature(original, eval_str=True)  # type: ignore[attr-defined]
        recorder.__name__ = original.__name__
        return recorder

    for info, original in zip(infos, originals, strict=True):
        assert original is not None
        info.callback = recorder_for(original)
    try:
        result = CliRunner().invoke(cli.app, argv)
    finally:
        for info, original in zip(infos, originals, strict=True):
            info.callback = original
    assert result.exit_code == 0, result.output
    name = received.pop("_command")
    return name, received


# --------------------------------------------------------------------------- when it opens


def test_qr_alone_without_a_terminal_prints_help_and_exits_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_menu(**_: Any) -> None:
        raise AssertionError("the menu must not open without a terminal")

    monkeypatch.setattr(tui, "run", no_menu)
    for argv in ([], ["menu"]):
        result = CliRunner().invoke(cli.app, argv)
        assert result.exit_code == 0
        assert "Usage" in result.output and "ingest" in result.output
        assert "/____/" not in result.output  # no logo


def test_qr_alone_in_a_terminal_opens_the_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[bool] = []
    monkeypatch.setattr(tui, "is_interactive", lambda: True)
    monkeypatch.setattr(tui, "run", lambda no_banner=False: opened.append(no_banner))
    assert CliRunner().invoke(cli.app, []).exit_code == 0
    assert CliRunner().invoke(cli.app, ["menu", "--no-banner"]).exit_code == 0
    assert opened == [False, True]


def test_commands_never_open_the_menu(monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    monkeypatch.setattr(tui, "is_interactive", lambda: True)
    monkeypatch.setattr(tui, "run", lambda **_: pytest.fail("a command opened the menu"))
    result = CliRunner().invoke(cli.app, ["status"])
    assert "ZHUFU" not in result.output and "/____/" not in result.output


def test_ctrl_c_in_the_main_menu_exits_0_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tui, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "QuestionaryPrompter", lambda: Scripted([KeyboardInterrupt]))
    result = CliRunner().invoke(cli.app, ["menu", "--no-banner"])
    assert result.exit_code == 0
    assert "Traceback" not in result.output


# --------------------------------------------------------------------------- banner


def test_banner_fits_the_limits_and_falls_back_when_narrow() -> None:
    wide = banner.logo_lines(120)
    assert 1 < len(wide) <= banner.MAX_HEIGHT
    assert max(len(line) for line in wide) <= banner.MAX_WIDTH
    assert banner.logo_lines(40) == ["ZHUFU QUANT"]


def test_banner_can_be_turned_off() -> None:
    assert banner.banner_disabled(no_banner=True, environ={})
    assert banner.banner_disabled(environ={"QR_NO_BANNER": "1"})
    assert not banner.banner_disabled(environ={"QR_NO_BANNER": "0"})
    assert not banner.banner_disabled(environ={})


def test_tui_run_skips_the_banner_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(menu, "QuestionaryPrompter", lambda: Scripted([QUIT]))
    for kwargs, env in (({"no_banner": True}, None), ({}, "1")):
        if env:
            monkeypatch.setenv("QR_NO_BANNER", env)
        out = io.StringIO()
        tui.run(console=Console(file=out, width=100), **kwargs)
        assert "/____/" not in out.getvalue()
    monkeypatch.delenv("QR_NO_BANNER")
    out = io.StringIO()
    tui.run(console=Console(file=out, width=100))
    assert "/____/" in out.getvalue() and "engine_core" in out.getvalue()


def test_no_color_means_no_colour_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    def render(environ: dict[str, str]) -> str:
        out = io.StringIO()
        console = Console(file=out, width=100, force_terminal=True, no_color="NO_COLOR" in environ)
        banner.print_banner(console, environ)
        return out.getvalue()

    assert "\x1b[" in render({})
    plain = render({"NO_COLOR": "1"})
    assert "\x1b[3" not in plain and "/____/" in plain  # no colour codes, logo still there
    assert not banner.colour_enabled({"NO_COLOR": "1"})


def test_info_line_names_the_active_core() -> None:
    assert "engine_core python (default)" in banner.info_line({})
    assert "engine_core rust (from QR_ENGINE_CORE" in banner.info_line({"QR_ENGINE_CORE": "rust"})


# --------------------------------------------------------------------------- actions

# (menu answers, expected command line, expected command function)
ACTIONS: list[tuple[list[Any], str, str]] = [
    (["Data", "Update data", "All symbols", True], "qr ingest", "ingest"),
    (["Data", "Update data", "A universe group", "pairs"], "qr ingest --universe pairs", "ingest"),
    (
        ["Data", "Update data", "Pick symbols", ["KO", "SPY"]],
        "qr ingest --symbols SPY,KO",
        "ingest",
    ),
    (
        ["Data", "Validate data", "A universe group", "pairs"],
        "qr validate --universe pairs",
        "validate",
    ),
    (["Data", "Coverage status"], "qr status", "status"),
    (
        ["Data", "Cross-check", "Pick symbols", ["KO"], "30"],
        "qr crosscheck --symbols KO --days 30",
        "crosscheck",
    ),
    (
        ["Data", "Export", "KO", "split", "out/ko.csv"],
        "qr export KO --out out/ko.csv --adjustment split",
        "export",
    ),
    (
        [
            "Backtests",
            "Run a backtest",
            "configs/backtests/sma_spy_event.yaml",
            "As in the config",
            "rust",
        ],
        "qr backtest run configs/backtests/sma_spy_event.yaml --engine-core rust",
        "backtest_run",
    ),
    (
        ["Backtests", "Run a backtest", "configs/backtests/sma_spy_event.yaml", "vectorized"],
        "qr backtest run configs/backtests/sma_spy_event.yaml --engine vectorized",
        "backtest_run",
    ),
    (
        ["Backtests", "Run a backtest", "configs/private/pairs_ko_pep.yaml", "As in the config"],
        "qr backtest run configs/private/pairs_ko_pep.yaml",
        "backtest_run",
    ),
    (
        ["Backtests", "List recent runs", "sma_crossover"],
        "qr backtest list --strategy sma_crossover",
        "backtest_list",
    ),
    (["Backtests", "Compare runs", ["R2", "R1"]], "qr backtest compare R2 R1", "backtest_compare"),
    (
        ["Paper trading", "Run today's decision", "configs/live/xs_momentum_demo.yaml", True],
        "qr live run configs/live/xs_momentum_demo.yaml --dry-run",
        "live_run",
    ),
    (
        [
            "Paper trading",
            "Run today's decision",
            "configs/live/xs_momentum_demo.yaml",
            False,
            True,
        ],
        "qr live run configs/live/xs_momentum_demo.yaml --catch-up",
        "live_run",
    ),
    (
        ["Paper trading", "Approve a waiting plan", "configs/live/xs_momentum_demo.yaml"],
        "qr live approve configs/live/xs_momentum_demo.yaml",
        "live_approve",
    ),
    (
        ["Paper trading", "Status", "configs/live/xs_momentum_demo.yaml"],
        "qr live status configs/live/xs_momentum_demo.yaml",
        "live_status",
    ),
    (
        [
            "Paper trading",
            "Monthly report",
            "configs/live/xs_momentum_demo.yaml",
            "2023-01",
            False,
        ],
        "qr live report configs/live/xs_momentum_demo.yaml --month 2023-01",
        "live_report_cmd",
    ),
    (
        [
            "Paper trading",
            "Monthly report",
            "configs/live/xs_momentum_demo.yaml",
            "",
            True,
            "out/report",
        ],
        "qr live report configs/live/xs_momentum_demo.yaml --tracking --out out/report",
        "live_report_cmd",
    ),
    (
        ["Paper trading", "Dashboard", "configs/live/xs_momentum_demo.yaml", True],
        "qr live dashboard configs/live/xs_momentum_demo.yaml --open",
        "live_dashboard_cmd",
    ),
]


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    report = project / "reports" / "R1.html"
    report.parent.mkdir()
    report.write_text("<html></html>", encoding="utf-8")
    runs = pd.DataFrame(
        {
            "run_id": ["R2", "R1"],
            "name": ["b", "a"],
            "strategy": ["sma_crossover", "buy_and_hold"],
            "status": ["ok", "ok"],
            "sharpe": [0.5, float("nan")],
            "report_path": [None, str(report)],
        }
    )
    monkeypatch.setattr(menu, "read_runs", lambda *a, **k: runs)


@pytest.mark.parametrize(("answers", "command_line", "function"), ACTIONS)
def test_every_action_runs_its_command_and_prints_it(
    answers: list[Any],
    command_line: str,
    function: str,
    calls: list[tuple[str, dict[str, Any]]],
    registry: None,
) -> None:
    prompter, output = play([*answers, BACK, QUIT])
    assert f"Running: {command_line}" in output
    assert len(calls) == 1 and calls[0][0] == function
    assert any(m.startswith("Done.") for m in prompter.asked)


@pytest.mark.parametrize(
    "invocation",
    [
        actions.ingest(Scope()),
        actions.ingest(Scope("universe", group="pairs")),
        actions.validate(Scope("symbols", symbols=("KO", "PEP"))),
        actions.status(),
        actions.crosscheck(Scope("universe", group="pairs"), 90),
        actions.export("BTC/USD", "none", "data/exports/my file.csv"),
        actions.backtest_run("configs/backtests/sma_spy.yaml"),
        actions.backtest_run("configs/backtests/sma_spy.yaml", "event", "rust"),
        actions.backtest_list(),
        actions.backtest_list("xs_momentum"),
        actions.backtest_compare(["R1", "R2"]),
        actions.live_run("configs/live/xs_momentum_demo.yaml", True),
        actions.live_run("configs/live/xs_momentum_demo.yaml", False, True),
        actions.live_approve("configs/live/xs_momentum_demo.yaml"),
        actions.live_status("configs/live/xs_momentum_demo.yaml"),
        actions.live_report("configs/live/xs_momentum_demo.yaml", "2023-01"),
        actions.live_report("configs/live/xs_momentum_demo.yaml", None, True, "out/report"),
        actions.live_dashboard("configs/live/xs_momentum_demo.yaml", True),
    ],
    ids=lambda inv: inv.command_line,
)
def test_printed_command_parses_to_the_same_call(invocation: Invocation) -> None:
    """What `Running: qr ...` shows is exactly what the menu runs."""
    import shlex

    name, received = cli_kwargs(shlex.split(invocation.command_line)[1:])
    assert name == invocation.command
    assert received == dict(invocation.kwargs)


def test_ingesting_everything_needs_confirmation(
    calls: list[tuple[str, dict[str, Any]]], project: Path
) -> None:
    play(["Data", "Update data", "All symbols", False, BACK, QUIT])
    assert calls == []


def test_a_large_crosscheck_warns_about_the_tiingo_limit(
    calls: list[tuple[str, dict[str, Any]]], project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    many = "".join(
        f"  - {{ symbol: S{i:02d}, source: yahoo, asset_class: equity }}\n" for i in range(60)
    )
    universe = UNIVERSE.replace("groups:", many + "groups:")
    (project / "config" / "universe.yaml").write_text(universe, encoding="utf-8")
    monkeypatch.setenv("TIINGO_API_KEY", "secret")
    prompter, _ = play(["Data", "Cross-check", "All symbols", None, False, BACK, QUIT])
    warning = next(m for m in prompter.asked if "Tiingo requests" in m)
    assert "about 64 Tiingo requests" in warning and "50 an hour" in warning
    assert calls == []


def test_open_report_opens_the_tearsheet(
    registry: None, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(actions.webbrowser, "open", lambda url: opened.append(url) or True)
    _, output = play(["Backtests", "Open a report", "R1", BACK, QUIT])
    assert opened == [(project / "reports" / "R1.html").resolve().as_uri()]
    assert "Opening:" in output


def test_config_summary_never_shows_the_api_key(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "very-secret-token")
    _, output = play(["Settings / info", "Show config summary", BACK, QUIT])
    assert "very-secret-token" not in output
    assert "TIINGO_API_KEY" in output and "set" in output
    assert "group pairs" in output and "disabled" in output  # crypto source


def test_show_commands_prints_the_help(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, output = play(["Settings / info", "Show available commands", BACK, QUIT])
    assert "Running: qr --help" in output
    assert "backtest" in capsys.readouterr().out


def test_a_failing_command_does_not_end_the_menu(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(**_: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(cli, "status", broken)
    prompter, output = play(["Data", "Coverage status", BACK, QUIT])
    assert "RuntimeError: disk full" in output
    assert prompter.answers == []  # it carried on to Back and Quit


# --------------------------------------------------------------------------- navigation


def test_back_goes_up_one_level(calls: list[tuple[str, dict[str, Any]]], project: Path) -> None:
    prompter, _ = play(["Data", "Update data", BACK, BACK, "Backtests", BACK, QUIT])
    assert calls == []
    assert prompter.answers == []


def test_ctrl_c_in_a_question_returns_to_the_submenu(
    calls: list[tuple[str, dict[str, Any]]], project: Path
) -> None:
    _, output = play(["Data", "Update data", KeyboardInterrupt, "Coverage status", BACK, QUIT])
    assert "Back to Data" in output
    assert [name for name, _ in calls] == ["status"]


def test_ctrl_c_in_a_submenu_returns_to_the_main_menu(project: Path) -> None:
    prompter, output = play(["Backtests", KeyboardInterrupt, QUIT])
    assert "Back to the main menu" in output
    assert prompter.answers == []


def test_ctrl_c_while_a_command_runs_returns_to_the_menu(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(**_: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "status", interrupted)
    prompter, output = play(["Data", "Coverage status", BACK, QUIT])
    assert "Cancelled" in output
    assert prompter.answers == []


def test_ctrl_c_in_the_main_menu_leaves_quietly() -> None:
    prompter, _ = play([KeyboardInterrupt])
    assert prompter.asked == ["Main menu"]


# --------------------------------------------------------------------------- discovery


def test_configs_are_found_in_both_folders(project: Path) -> None:
    found = [p.as_posix() for p in menu.discover_configs()]
    assert found == ["configs/backtests/sma_spy_event.yaml", "configs/private/pairs_ko_pep.yaml"]
    assert "(sma_crossover, event)" in menu.describe_config(Path(found[0]))


def test_scope_choices_come_from_the_config(project: Path) -> None:
    prompter = Scripted(["A universe group", "pairs"])
    out = io.StringIO()
    m = Menu(prompter, Console(file=out))
    cfg = m.universe()
    assert cfg is not None
    assert m.ask_scope(cfg, "Update") == Scope("universe", group="pairs")


# --------------------------------------------------------------------------- screens

CLEAR = "\x1b[2J"
ALT_ON, ALT_OFF = "\x1b[?1049h", "\x1b[?1049l"


def terminal(legacy_windows: bool = False) -> tuple[Console, io.StringIO]:
    """A console that behaves like a modern terminal (VT codes, as on Linux, macOS, and
    Windows Terminal), or like the old Windows console with ``legacy_windows=True``."""
    out = io.StringIO()
    console = Console(
        file=out,
        width=100,
        force_terminal=True,
        color_system=None,
        legacy_windows=legacy_windows,
    )
    return console, out


def test_every_screen_starts_clean_with_a_breadcrumb(project: Path) -> None:
    console, out = terminal()
    Menu(Scripted(["Data", "Coverage status", BACK, QUIT]), console).run()
    screens = out.getvalue().split(CLEAR)[1:]
    # main menu, Data, the status action, Data again, main menu again
    assert len(screens) == 5
    assert menu.BREADCRUMB.join(["Main menu", "Data", "Coverage status"]) in screens[2]
    assert "Running: qr status" in screens[2]
    assert "Running: qr status" not in screens[3]  # the next screen no longer shows it


def test_the_logo_tops_every_screen(project: Path) -> None:
    console, out = terminal()
    Menu(Scripted(["Data", BACK, QUIT]), console, banner=True).run()
    screens = out.getvalue().split(CLEAR)[1:]
    assert len(screens) == 3 and all("/____/" in screen for screen in screens)


def test_a_notice_shows_once_on_the_next_screen(project: Path) -> None:
    console, out = terminal()
    Menu(Scripted(["Backtests", KeyboardInterrupt, "Data", BACK, QUIT]), console).run()
    screens = out.getvalue().split(CLEAR)[1:]
    with_notice = [s for s in screens if "Back to the main menu." in s]
    assert len(with_notice) == 1


def test_the_menu_uses_the_alternate_screen_and_always_leaves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console, out = terminal()
    monkeypatch.setattr(menu, "QuestionaryPrompter", lambda: Scripted([QUIT]))
    tui.run(no_banner=True, console=console)
    text = out.getvalue()
    assert text.startswith(ALT_ON) and text.rstrip().endswith(ALT_OFF)

    console, out = terminal()

    def broken() -> Scripted:
        raise RuntimeError("boom")

    monkeypatch.setattr(menu, "QuestionaryPrompter", broken)
    with pytest.raises(RuntimeError):
        tui.run(no_banner=True, console=console)
    assert ALT_OFF in out.getvalue()  # the terminal is restored even after an error


def test_no_screen_control_outside_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(menu, "QuestionaryPrompter", lambda: Scripted([QUIT]))
    out = io.StringIO()
    tui.run(no_banner=True, console=Console(file=out, width=100))
    assert "\x1b[" not in out.getvalue()


def test_an_old_windows_console_gets_the_menu_without_the_alternate_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console, out = terminal(legacy_windows=True)
    monkeypatch.setattr(menu, "QuestionaryPrompter", lambda: Scripted(["Data", BACK, QUIT]))
    tui.run(no_banner=True, console=console)
    assert ALT_ON not in out.getvalue() and ALT_OFF not in out.getvalue()
    assert out.getvalue().count(CLEAR) == 3  # screens are still redrawn
