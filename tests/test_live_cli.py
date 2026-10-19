"""``qr live``: a dry-run plan print, approval, status, report, and the kill switch."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import qr.live.cycle as cycle_module
import qr.live.schedule as schedule_module
import qr.live.sim_broker as sim_broker_module
from qr import calendars
from qr.cli import app

from .live.live_store import real_session_bars, seed_store

runner = CliRunner()


def _freeze_now(monkeypatch: pytest.MonkeyPatch, as_of: date) -> None:
    """Make the cycle, the sim broker, and the scheduler see ``as_of``'s close as "now".

    The CLI always acts on the real current month-end; these tests run it against a
    fixed historical fixture instead, so every module that reads its own clock has to be
    fixed to the same instant, or the scheduler's missed-decision maintenance pass (which
    reads real wall-clock time otherwise) would see a years-old gap and flag everything.
    """
    instant = datetime.combine(as_of, time(23, 59))

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz: object = None) -> _Frozen:
            return cls.fromtimestamp(instant.timestamp())

    monkeypatch.setattr(cycle_module, "datetime", _Frozen)
    monkeypatch.setattr(sim_broker_module, "datetime", _Frozen)
    monkeypatch.setattr(schedule_module, "datetime", _Frozen)


def _seeded_config(tmp_path: Path) -> tuple[Path, date]:
    symbols = ["AAA", "BBB"]
    bars = {
        s: real_session_bars(date(2022, 1, 1), date(2023, 2, 15), seed=i)
        for i, s in enumerate(symbols)
    }
    seed_store(tmp_path, bars)
    as_of = calendars.sessions("etf", date(2023, 1, 1), date(2023, 1, 31))[-1].date()

    config_path = tmp_path / "live.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "cli-test",
                "strategy": "buy_and_hold",
                "universe": symbols,
                "cash_buffer": 0.0,
                "min_trade_weight": 0.0,
                "working_dir": str(tmp_path),
            }
        )
    )
    return config_path, as_of


def test_run_dry_run_prints_a_plan_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, as_of = _seeded_config(tmp_path)
    _freeze_now(monkeypatch, as_of)
    result = runner.invoke(
        app,
        ["live", "run", str(config_path), "--dry-run", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "Plan" in result.output
    assert "AAA" in result.output and "BBB" in result.output
    assert not (tmp_path / "sim_cli-test.json").exists()


def test_run_approve_then_status_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, as_of = _seeded_config(tmp_path)
    _freeze_now(monkeypatch, as_of)
    approved = runner.invoke(
        app,
        ["live", "run", str(config_path), "--data-dir", str(tmp_path)],
        input="y\n",
    )
    assert approved.exit_code == 0, approved.output
    assert "completed" in approved.output
    assert (tmp_path / "sim_cli-test.json").exists()

    status = runner.invoke(app, ["live", "status", str(config_path), "--data-dir", str(tmp_path)])
    assert status.exit_code == 0, status.output
    assert "Latest decision" in status.output
    assert "Broker: sim" in status.output
    assert "Positions" in status.output
    assert "AAA" in status.output and "BBB" in status.output

    report = runner.invoke(
        app,
        [
            "live", "report", str(config_path), "--month", f"{as_of:%Y-%m}",
            "--data-dir", str(tmp_path),
        ],
    )  # fmt: skip
    assert report.exit_code == 0, report.output
    assert "buy_and_hold" in report.output


def test_run_refuses_a_real_send_against_ibkr_but_allows_dry_run(tmp_path: Path) -> None:
    config_path, _ = _seeded_config(tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "cli-test",
                "strategy": "buy_and_hold",
                "universe": ["AAA", "BBB"],
                "broker": "ibkr",
                "cash_buffer": 0.0,
                "min_trade_weight": 0.0,
                "working_dir": str(tmp_path),
            }
        )
    )
    # Refused before ever trying to connect to a (non-existent, in this test) Gateway --
    # this is Part 2b's own interim gate, not an IB connection failure.
    refused = runner.invoke(app, ["live", "run", str(config_path), "--data-dir", str(tmp_path)])
    assert refused.exit_code != 0
    assert "--dry-run" in refused.output

    dry_run = runner.invoke(
        app, ["live", "run", str(config_path), "--dry-run", "--data-dir", str(tmp_path)]
    )
    assert dry_run.exit_code != 0  # no IB Gateway running in this test
    assert "--dry-run" not in dry_run.output  # got past the gate; failed trying to connect


def test_paper_ledger_broker_is_not_gated_and_status_reports_it(tmp_path: Path) -> None:
    config_path, _ = _seeded_config(tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "cli-test",
                "strategy": "buy_and_hold",
                "universe": ["AAA", "BBB"],
                "broker": "paper_ledger",
                "capital_limit": 10_000.0,
                "cash_buffer": 0.0,
                "min_trade_weight": 0.0,
                "working_dir": str(tmp_path),
            }
        )
    )
    # Never gated behind --dry-run (unlike ibkr): it never leaves this machine.
    run = runner.invoke(app, ["live", "run", str(config_path), "--data-dir", str(tmp_path)])
    assert "--dry-run" not in run.output

    status = runner.invoke(app, ["live", "status", str(config_path), "--data-dir", str(tmp_path)])
    assert status.exit_code == 0, status.output
    assert "paper_ledger" in status.output
    assert "paper" in status.output
    assert "Cash: $10,000.00" in status.output


def test_run_with_no_input_leaves_the_plan_awaiting_qr_live_approve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, as_of = _seeded_config(tmp_path)
    _freeze_now(monkeypatch, as_of)
    result = runner.invoke(app, ["live", "run", str(config_path), "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "qr live approve" in result.output

    status = runner.invoke(app, ["live", "status", str(config_path), "--data-dir", str(tmp_path)])
    assert "No open positions" in status.output  # nothing was ever sent to the broker

    approved = runner.invoke(
        app, ["live", "approve", str(config_path), "--data-dir", str(tmp_path)], input="y\n"
    )
    assert approved.exit_code == 0, approved.output
    assert "completed" in approved.output
    assert (tmp_path / "sim_cli-test.json").exists()

    again = runner.invoke(app, ["live", "approve", str(config_path), "--data-dir", str(tmp_path)])
    assert again.exit_code == 0, again.output
    assert "No plan is awaiting approval" in again.output


def test_approve_with_yes_skips_the_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, as_of = _seeded_config(tmp_path)
    _freeze_now(monkeypatch, as_of)
    runner.invoke(app, ["live", "run", str(config_path), "--data-dir", str(tmp_path)])
    approved = runner.invoke(
        app, ["live", "approve", str(config_path), "--yes", "--data-dir", str(tmp_path)]
    )
    assert approved.exit_code == 0, approved.output
    assert "completed" in approved.output


def test_dashboard_command_writes_a_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, as_of = _seeded_config(tmp_path)
    _freeze_now(monkeypatch, as_of)
    runner.invoke(
        app,
        ["live", "run", str(config_path), "--data-dir", str(tmp_path)],
        input="y\n",
    )
    result = runner.invoke(
        app, ["live", "dashboard", str(config_path), "--data-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    path = tmp_path / "dashboard_cli-test.html"
    assert path.is_file()
    assert "<html>" in path.read_text(encoding="utf-8")


def test_report_tracking_writes_md_and_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, as_of = _seeded_config(tmp_path)
    _freeze_now(monkeypatch, as_of)
    runner.invoke(
        app,
        ["live", "run", str(config_path), "--data-dir", str(tmp_path)],
        input="y\n",
    )
    out = tmp_path / "reports" / f"{as_of:%Y-%m}"
    result = runner.invoke(
        app,
        [
            "live", "report", str(config_path), "--month", f"{as_of:%Y-%m}", "--tracking",
            "--out", str(out), "--data-dir", str(tmp_path),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert out.with_suffix(".md").is_file()
    assert out.with_suffix(".html").is_file()
    assert "Reliability" in out.with_suffix(".md").read_text(encoding="utf-8")


def test_kill_and_unkill(tmp_path: Path) -> None:
    killed = runner.invoke(app, ["live", "kill", "--data-dir", str(tmp_path)])
    assert killed.exit_code == 0
    assert (tmp_path / "KILL").exists()

    unkilled = runner.invoke(app, ["live", "unkill", "--data-dir", str(tmp_path)])
    assert unkilled.exit_code == 0
    assert not (tmp_path / "KILL").exists()
