"""The CLI end to end, with the data source swapped for an in-memory one."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from qr import ingest as ingest_module
from qr import storage
from qr.cli import app
from qr.config import db_path

from .conftest import AFTER_JAN_16, FakeSource

runner = CliRunner()

CONFIG = """
start_date: 2024-01-01
sources: { yahoo: { request_delay_seconds: 0, max_retries: 0 } }
universe:
  - { symbol: AAA, source: yahoo, asset_class: equity }
  - { symbol: BBB, source: yahoo, asset_class: equity }
groups:
  first: [AAA]
"""


@pytest.fixture
def project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_bars: pd.DataFrame,
    split_dividend_bars: pd.DataFrame,
    split_dividend_actions: pd.DataFrame,
) -> FakeSource:
    """A temp config + data dir (via env vars), a fake source, and a frozen clock."""
    config = tmp_path / "universe.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("QR_CONFIG", str(config))
    monkeypatch.setenv("QR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(storage, "utcnow", lambda: AFTER_JAN_16)
    source = FakeSource({"AAA": clean_bars, "BBB": clean_bars})
    monkeypatch.setattr(ingest_module, "build_source", lambda name, cfg: source)
    return source


def test_init_creates_the_database(project: FakeSource, tmp_path: Path) -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert db_path(tmp_path / "data").is_file()
    assert runner.invoke(app, ["init"]).exit_code == 0  # repeatable


def test_status_before_init_explains_what_to_do(project: FakeSource) -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "qr init" in result.output


def test_ingest_status_validate_export(project: FakeSource, tmp_path: Path) -> None:
    assert runner.invoke(app, ["init"]).exit_code == 0

    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0, result.output
    assert "AAA" in result.output and "BBB" in result.output
    assert "No quality issues" in result.output

    again = runner.invoke(app, ["ingest", "--symbols", "AAA"])
    assert again.exit_code == 0, again.output
    with storage.connect(db_path(tmp_path / "data"), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM daily_bars").fetchall() == [(20,)]

    status = runner.invoke(app, ["status"])
    assert status.exit_code == 0, status.output
    assert "2024-01-02" in status.output and "2024-01-16" in status.output
    assert "10" in status.output

    validate = runner.invoke(app, ["validate"])
    assert validate.exit_code == 0, validate.output
    assert "No quality issues" in validate.output

    out = tmp_path / "exports" / "aaa.csv"
    export = runner.invoke(app, ["export", "AAA", "--out", str(out), "--adjustment", "none"])
    assert export.exit_code == 0, export.output
    exported = pd.read_csv(out)
    assert list(exported.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert exported["date"].iloc[0] == "2024-01-02"
    assert exported["close"].tolist() == project.bars["AAA"]["close"].tolist()


def test_ingest_exits_non_zero_when_a_symbol_fails(project: FakeSource) -> None:
    del project.bars["BBB"]
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "failed" in result.output
    # The healthy symbol was still ingested.
    status = runner.invoke(app, ["status"])
    assert "2024-01-16" in status.output


def test_validate_prints_a_summary_and_fails_on_errors(
    project: FakeSource, dirty_bars: pd.DataFrame
) -> None:
    project.bars["AAA"] = dirty_bars
    runner.invoke(app, ["ingest", "--symbols", "AAA"])
    result = runner.invoke(app, ["validate", "--symbols", "AAA"])
    assert result.exit_code == 1  # the fixture contains error-severity findings
    for check in ("ohlc_consistency", "non_positive_price", "negative_volume", "gap"):
        assert check in result.output
    assert "findings" in result.output

    unknown = runner.invoke(app, ["validate", "--symbols", "NOPE"])
    assert unknown.exit_code == 1
    assert "unknown symbols" in unknown.output


def test_export_errors(project: FakeSource, tmp_path: Path) -> None:
    runner.invoke(app, ["ingest"])
    out = str(tmp_path / "x.csv")
    unknown = runner.invoke(app, ["export", "NOPE", "--out", out])
    assert unknown.exit_code == 1 and "unknown symbols" in unknown.output
    bad_mode = runner.invoke(app, ["export", "AAA", "--out", out, "--adjustment", "magic"])
    assert bad_mode.exit_code == 1 and "unknown adjustment" in bad_mode.output


def test_bad_config_fails_fast(project: FakeSource, tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("start_date: 2024-01-01\nuniverse: []\n", encoding="utf-8")
    result = runner.invoke(app, ["ingest", "--config", str(broken)])
    assert result.exit_code == 1
    assert "invalid config" in result.output


def test_named_universe(project: FakeSource) -> None:
    result = runner.invoke(app, ["ingest", "--universe", "first"])
    assert result.exit_code == 0, result.output
    assert [call[0] for call in project.calls] == ["AAA"]

    runner.invoke(app, ["ingest"])
    status = runner.invoke(app, ["status", "--universe", "first"])
    assert status.exit_code == 0, status.output
    assert "AAA" in status.output and "BBB" not in status.output
    assert runner.invoke(app, ["validate", "--universe", "first"]).exit_code == 0

    both = runner.invoke(app, ["ingest", "--universe", "first", "--symbols", "AAA"])
    assert both.exit_code == 1 and "not both" in both.output
    for command in ("ingest", "validate", "status"):
        unknown = runner.invoke(app, [command, "--universe", "nope"])
        assert unknown.exit_code == 1, command
        assert "unknown universe 'nope' (available: first)" in unknown.output


def test_invalid_group_config_fails_with_clear_error(project: FakeSource, tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text(CONFIG.replace("first: [AAA]", "first: [AAA, ZZZ]"), encoding="utf-8")
    result = runner.invoke(app, ["ingest", "--config", str(broken), "--universe", "first"])
    assert result.exit_code == 1
    assert "group 'first' lists symbols that are not defined in universe: ZZZ" in result.output
