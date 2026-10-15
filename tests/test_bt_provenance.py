"""Provenance on every run: data hashes, environment, `qr backtest show` and `verify`."""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from qr import storage
from qr.backtest import provenance, registry
from qr.backtest.config import load_backtest_config
from qr.backtest.runner import execute
from qr.cli import app
from qr.config import db_path
from qr.snapshots import file_sha256

from .bt_helpers import synthetic_bars
from .conftest import AFTER_JAN_16

runner = CliRunner()

UNIVERSE = """
start_date: 2015-01-01
universe:
  - { symbol: AAA, source: yahoo, asset_class: equity }
  - { symbol: BBB, source: yahoo, asset_class: etf }
  - { symbol: CCC, source: yahoo, asset_class: equity }
  - { symbol: DDD, source: yahoo, asset_class: equity }
groups:
  duo: [AAA, CCC]
"""

CONFIG = """
name: provenance_test
engine: {engine}
strategy: buy_and_hold
universe: duo
start: 2015-04-01
end: 2016-02-24
initial_cash: 100000
benchmark: BBB
"""

# The registry as it was before provenance columns existed.
OLD_SCHEMA = """
CREATE TABLE backtest_runs (
  run_id TEXT PRIMARY KEY, created_at TIMESTAMP, strategy TEXT, engine TEXT,
  config_json TEXT, config_hash TEXT, git_commit TEXT, git_dirty BOOLEAN,
  data_snapshot TEXT, start_date DATE, end_date DATE,
  sharpe DOUBLE, cagr DOUBLE, max_drawdown DOUBLE, turnover DOUBLE,
  report_path TEXT, notes TEXT, name TEXT, status TEXT
);
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp folder with a universe file and a store of four synthetic symbols."""
    (tmp_path / "universe.yaml").write_text(UNIVERSE, encoding="utf-8")
    monkeypatch.setenv("QR_CONFIG", str(tmp_path / "universe.yaml"))
    monkeypatch.setenv("QR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.delenv("QR_ENGINE_CORE", raising=False)
    monkeypatch.chdir(tmp_path)
    with storage.connect(db_path(tmp_path / "data")) as con:
        storage.init_schema(con)
        for seed, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
            frame, actions = synthetic_bars(300, seed=seed, dividend=0.3 if seed == 0 else 0.0)
            storage.upsert_instrument(con, symbol, "yahoo", "equity", True, AFTER_JAN_16)
            storage.upsert_bars(con, symbol, "yahoo", frame, AFTER_JAN_16)
            storage.upsert_corporate_actions(con, symbol, actions)
    return tmp_path


def config_file(project: Path, engine: str = "vectorized") -> str:
    path = project / f"{engine}.yaml"
    path.write_text(CONFIG.format(engine=engine), encoding="utf-8")
    return str(path)


def market(project: Path, sql: str) -> None:
    with storage.connect(db_path(project / "data")) as con:
        con.execute(sql)


def test_every_run_records_data_and_environment(project: Path) -> None:
    config = load_backtest_config(config_file(project))
    outcome = execute(config)
    assert outcome.error is None
    run = registry.read_runs().iloc[0]

    expected = provenance.run_fingerprint(["AAA", "CCC", "BBB"], config.end)
    assert run["data_hash"] == expected.hash
    assert json.loads(run["data_per_symbol_json"]) == expected.per_symbol
    assert set(expected.per_symbol) == {"AAA", "BBB", "CCC"}  # DDD is not read
    assert run["data_rows"] == expected.rows
    assert expected.first_date < config.start  # warm-up history is data the run read
    assert expected.last_date == config.end
    assert pd.isna(run["data_as_of"])
    assert run["python_version"] == platform.python_version()
    assert run["uv_lock_sha256"] == file_sha256(provenance.PROJECT_ROOT / "uv.lock")
    assert pd.isna(run["engine_core"]) and pd.isna(run["rust_core_version"])
    assert run["os"] == platform.platform()
    assert run["git_commit"] == outcome.record.git_commit  # still recorded

    html = Path(run["report_path"]).read_text(encoding="utf-8")
    assert "<h2>Provenance</h2>" in html
    for text in (
        expected.hash,
        expected.per_symbol["CCC"],
        platform.python_version(),
        f"{expected.rows:,}",
    ):
        assert text in html


def test_the_core_that_actually_runs_is_recorded(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_backtest_config(config_file(project, "event"))
    assert provenance.core_used(config) == "python"
    monkeypatch.setenv("QR_ENGINE_CORE", "rust")
    rust = provenance.core_used(config)
    assert rust == ("rust" if provenance.rust_available() else "python")
    monkeypatch.delenv("QR_ENGINE_CORE")

    execute(config)
    run = registry.read_runs().iloc[0]
    assert run["engine_core"] == "python" and pd.isna(run["rust_core_version"])


def test_failed_runs_keep_their_provenance(project: Path) -> None:
    config = load_backtest_config(config_file(project))
    broken = config.model_copy(update={"strategy": "does_not_exist"})
    outcome = execute(broken)
    assert outcome.error is not None
    run = registry.read_runs().iloc[0]
    assert run["status"] == "failed"
    assert run["data_hash"] == provenance.run_fingerprint(["AAA", "BBB", "CCC"], config.end).hash
    assert run["python_version"] == platform.python_version()


def test_verify_matches_then_reports_a_revision(project: Path) -> None:
    assert runner.invoke(app, ["backtest", "run", config_file(project)]).exit_code == 0
    run_id = str(registry.read_runs().iloc[0]["run_id"])

    same = runner.invoke(app, ["backtest", "verify", run_id[:18]])
    assert same.exit_code == 0, same.output
    assert "match" in same.output and "changed" not in same.output

    # Data outside the run (another symbol, or bars after its end) is irrelevant.
    market(project, "UPDATE daily_bars SET close = close * 2 WHERE symbol = 'DDD'")
    market(project, "UPDATE daily_bars SET close = close * 2 WHERE date > DATE '2016-02-24'")
    assert runner.invoke(app, ["backtest", "verify", run_id]).exit_code == 0

    # A provider revision of one close inside the run's history.
    market(
        project,
        "UPDATE daily_bars SET close = close + 0.01 "
        "WHERE symbol = 'CCC' AND date = DATE '2015-06-01'",
    )
    changed = runner.invoke(app, ["backtest", "verify", run_id])
    assert changed.exit_code == 1
    assert "data changed" in changed.output
    assert "CCC: different bars or actions" in changed.output
    assert "AAA:" not in changed.output and "BBB:" not in changed.output


def test_verify_reports_a_symbol_that_disappeared(project: Path) -> None:
    assert execute(load_backtest_config(config_file(project))).error is None
    run = {str(k): v for k, v in registry.read_runs().iloc[0].to_dict().items()}
    market(project, "DELETE FROM daily_bars WHERE symbol = 'CCC'")
    market(project, "DELETE FROM instruments WHERE symbol = 'CCC'")
    outcome = provenance.verify_run(run)
    assert not outcome.matches
    assert outcome.changed == ["CCC"] and outcome.missing == ["CCC"]


def test_show_prints_config_provenance_and_metrics(project: Path) -> None:
    assert runner.invoke(app, ["backtest", "run", config_file(project)]).exit_code == 0
    run = registry.read_runs().iloc[0]
    shown = runner.invoke(app, ["backtest", "show", str(run["run_id"])])
    assert shown.exit_code == 0, shown.output
    for text in (
        str(run["run_id"]),
        "Provenance",
        str(run["data_hash"]),
        str(run["uv_lock_sha256"]),
        platform.python_version(),
        "none (vectorized engine)",
        "Data hash per symbol",
        "Sharpe",
        "CAGR",
        '"strategy": "buy_and_hold"',
        '"commission_model": "ibkr_tiered"',  # resolved defaults are shown too
    ):
        assert text in shown.output, text
    missing = runner.invoke(app, ["backtest", "show", "nope"])
    assert missing.exit_code == 1 and "no run with an id starting with 'nope'" in missing.output


def test_old_registry_rows_still_list_show_and_compare(project: Path) -> None:
    path = registry.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    old_config = load_backtest_config(config_file(project))
    with storage.connect(path) as con:
        con.execute(OLD_SCHEMA)
        con.execute(
            "INSERT INTO backtest_runs VALUES "
            "('20260101T000000-aaaaaa', ?, 'buy_and_hold', 'vectorized', ?, ?, 'abc123', "
            "false, '2024-01-17 00:00:00', DATE '2015-04-01', DATE '2016-02-24', "
            "1.1, 0.07, -0.2, 0.01, NULL, '', 'old_run', 'ok')",
            [datetime(2026, 1, 1), old_config.to_json(), old_config.config_hash()],
        )

    # Read before any migration: the new columns come back empty.
    old = registry.read_runs()
    assert list(old.columns) == registry.COLUMNS
    assert pd.isna(old.iloc[0]["data_hash"])
    shown = runner.invoke(app, ["backtest", "show", "20260101"])
    assert shown.exit_code == 0, shown.output
    assert "not recorded" in shown.output and "abc123" in shown.output
    nothing = runner.invoke(app, ["backtest", "verify", "20260101"])
    assert nothing.exit_code == 1 and "has no data hash" in nothing.output

    # A new run migrates the table; the old row is untouched.
    assert runner.invoke(app, ["backtest", "run", config_file(project)]).exit_code == 0
    runs = registry.read_runs()
    assert len(runs) == 2
    migrated = runs[runs["run_id"] == "20260101T000000-aaaaaa"].iloc[0]
    assert migrated["sharpe"] == 1.1 and pd.isna(migrated["data_hash"])
    listed = runner.invoke(app, ["backtest", "list"])
    assert listed.exit_code == 0 and "old_run" in listed.output
    compared = runner.invoke(app, ["backtest", "compare", *runs["run_id"].tolist()])
    assert compared.exit_code == 0, compared.output
    assert runner.invoke(app, ["backtest", "show", "20260101"]).exit_code == 0
