"""``qr live``: a dry-run plan print, approval, status, report, and the kill switch."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import qr.live.cycle as cycle_module
import qr.live.sim_broker as sim_broker_module
from qr import calendars
from qr.cli import app

from .live.live_store import real_session_bars, seed_store

runner = CliRunner()


def _freeze_now(monkeypatch: pytest.MonkeyPatch, as_of: date) -> None:
    """Make both the cycle and the sim broker see ``as_of``'s close as "now".

    The CLI always acts on the real current month-end; these tests run it against a
    fixed historical fixture instead, so the clock the cycle reads has to be fixed too.
    """
    instant = datetime.combine(as_of, time(23, 59))

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz: object = None) -> _Frozen:
            return cls.fromtimestamp(instant.timestamp())

    monkeypatch.setattr(cycle_module, "datetime", _Frozen)
    monkeypatch.setattr(sim_broker_module, "datetime", _Frozen)


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

    report = runner.invoke(
        app,
        [
            "live", "report", str(config_path), "--month", f"{as_of:%Y-%m}",
            "--data-dir", str(tmp_path),
        ],
    )  # fmt: skip
    assert report.exit_code == 0, report.output
    assert "buy_and_hold" in report.output


def test_kill_and_unkill(tmp_path: Path) -> None:
    killed = runner.invoke(app, ["live", "kill", "--data-dir", str(tmp_path)])
    assert killed.exit_code == 0
    assert (tmp_path / "KILL").exists()

    unkilled = runner.invoke(app, ["live", "unkill", "--data-dir", str(tmp_path)])
    assert unkilled.exit_code == 0
    assert not (tmp_path / "KILL").exists()
