"""Configs, the runner, the registry, the tearsheet, and `qr backtest` end to end."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from qr import storage
from qr.backtest import registry
from qr.backtest.config import BacktestConfig, load_backtest_config
from qr.backtest.data import MarketData
from qr.backtest.report import format_metric, monthly_returns, render_report
from qr.backtest.runner import benchmark_curve, execute, run_backtest
from qr.backtest.strategies import STRATEGIES, build_strategy, register
from qr.backtest.strategy import EventStrategy
from qr.cli import app
from qr.config import ConfigError, db_path

from .bt_helpers import make_config, synthetic_bars, synthetic_market
from .conftest import AFTER_JAN_16

# Every test here runs on both event-engine cores (see the engine_core fixture).
pytestmark = pytest.mark.usefixtures("engine_core")

runner = CliRunner()

UNIVERSE = """
start_date: 2015-01-01
universe:
  - { symbol: AAA, source: yahoo, asset_class: equity }
  - { symbol: BBB, source: yahoo, asset_class: etf }
  - { symbol: CCC, source: yahoo, asset_class: equity }
groups:
  trio: [AAA, BBB, CCC]
"""

# Small windows so that 300 sessions are plenty.
PARAMS = {
    "buy_and_hold": "{}",
    "sma_crossover": "{ fast: 5, slow: 20 }",
    "xs_momentum": "{ lookback: 40, skip: 5, top_n: 2 }",
}


def config_text(strategy: str, engine: str, **extra: str) -> str:
    fields = {
        "name": f"{strategy}_{engine}",
        "engine": engine,
        "strategy": strategy,
        "params": PARAMS.get(strategy, "{}"),
        "universe": "trio",
        "start": "2015-04-01",
        "end": "2016-02-24",
        "initial_cash": "100000",
        "benchmark": "BBB",
        "rebalance": "weekly",
        **extra,
    }
    return "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp folder with a universe file and a store holding three synthetic symbols."""
    (tmp_path / "universe.yaml").write_text(UNIVERSE, encoding="utf-8")
    monkeypatch.setenv("QR_CONFIG", str(tmp_path / "universe.yaml"))
    monkeypatch.setenv("QR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.chdir(tmp_path)  # reports/ lands here, and no developer .env is read
    with storage.connect(db_path(tmp_path / "data")) as con:
        storage.init_schema(con)
        for seed, (symbol, asset_class) in enumerate(
            [("AAA", "equity"), ("BBB", "etf"), ("CCC", "equity")]
        ):
            frame, actions = synthetic_bars(300, seed=seed, dividend=0.3 if seed == 0 else 0.0)
            storage.upsert_instrument(con, symbol, "yahoo", asset_class, True, AFTER_JAN_16)
            storage.upsert_bars(con, symbol, "yahoo", frame, AFTER_JAN_16)
            storage.upsert_corporate_actions(con, symbol, actions)
    return tmp_path


def write(project: Path, name: str, text: str) -> str:
    path = project / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- config


def test_the_spec_example_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "sma_spy.yaml"
    path.write_text(
        """
name: sma_crossover_spy
engine: vectorized        # or: event
strategy: sma_crossover
params: { fast: 50, slow: 200 }
universe: [SPY]
start: 2012-01-01
end: 2024-12-31
initial_cash: 10000
benchmark: SPY
execution: next_open
rebalance: daily
costs:
  commission_model: ibkr_tiered
  spread_bps: 1
  slippage_bps: 2
constraints: { long_only: true, max_gross_leverage: 1.0 }
""",
        encoding="utf-8",
    )
    config = load_backtest_config(path)
    assert config.universe == ["SPY"] and config.start == date(2012, 1, 1)
    assert config.params == {"fast": 50, "slow": 200}
    assert config.costs.commission_model == "ibkr_tiered" and config.constraints.long_only
    # The hash identifies the experiment: stable, and different for a different config.
    assert config.config_hash() == load_backtest_config(path).config_hash()
    other = config.model_copy(update={"params": {"fast": 20, "slow": 100}})
    assert other.config_hash() != config.config_hash()
    assert BacktestConfig.model_validate_json(config.to_json()) == config


def test_shipped_example_configs_are_valid() -> None:
    folder = Path(__file__).parent.parent / "configs" / "backtests"
    names = sorted(p.name for p in folder.glob("*.yaml"))
    assert len(names) == 7
    for name in names:
        config = load_backtest_config(folder / name)
        assert config.strategy in STRATEGIES
        build_strategy(config.strategy, config.engine, config.params)


def test_invalid_configs_are_rejected_with_the_field_named(tmp_path: Path) -> None:
    def load(text: str) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(text, encoding="utf-8")
        load_backtest_config(path)

    base = config_text("buy_and_hold", "event")
    with pytest.raises(ConfigError, match="engine"):
        load(base.replace("engine: event", "engine: quantum"))
    with pytest.raises(ConfigError, match="is after end"):
        load(base.replace("start: 2015-04-01", "start: 2017-01-01"))
    with pytest.raises(ConfigError, match="slipage_bps"):
        load(base + "costs: { slipage_bps: 2 }\n")  # a typo must not pass silently
    with pytest.raises(ConfigError, match="repeats symbols"):
        load(base.replace("universe: trio", "universe: [AAA, AAA]"))
    with pytest.raises(ConfigError, match="not found"):
        load_backtest_config(tmp_path / "missing.yaml")


def test_strategy_lookup() -> None:
    with pytest.raises(ValueError, match="unknown strategy 'nope'"):
        build_strategy("nope", "event", {})
    with pytest.raises(ValueError, match="bad params"):
        build_strategy("sma_crossover", "event", {"speed": 3})
    with pytest.raises(ValueError, match="fast < slow"):
        build_strategy("sma_crossover", "vectorized", {"fast": 50, "slow": 20})

    class Mine(EventStrategy):
        pass

    register("mine", event=Mine)
    try:
        assert isinstance(build_strategy("mine", "event", {}), Mine)
        with pytest.raises(ValueError, match="no implementation for the vectorized engine"):
            build_strategy("mine", "vectorized", {})
    finally:
        del STRATEGIES["mine"]


# --------------------------------------------------------------------------- runner, report


def test_run_backtest_in_memory_and_render_the_report() -> None:
    data = synthetic_market(2, 400, seed=2)
    config = make_config(
        data, engine="vectorized", strategy="sma_crossover", params={"fast": 5, "slow": 20},
        costs={}, benchmark="S00", start=data.dates[30].date(),
    )  # fmt: skip
    closes = data.wide("close", adjusted=True)["S00"]
    result = run_backtest(config, data, closes)

    assert result.benchmark is not None
    assert result.benchmark.iloc[0] == config.initial_cash
    assert result.gross_equity.iloc[-1] > result.equity.iloc[-1]  # costs only ever cost
    assert result.gross.costs.to_numpy().sum() == 0.0
    assert result.metrics["total_costs"] > 0 and result.gross_metrics["total_costs"] == 0
    assert {"sharpe", "cagr", "max_drawdown", "turnover", "psr", "beta"} <= set(result.metrics)
    assert result.benchmark_metrics["total_return"] == pytest.approx(
        closes.loc[result.equity.index[-1]] / closes.loc[result.equity.index[0]] - 1
    )

    page = render_report(result, {"Run id": "test-run", "Notes": "<b>escaped</b>"})
    assert page.startswith("<!doctype html>")
    assert page.count("<svg") == 6  # equity, drawdown, heatmap, two rolling, exposure
    for text in (
        "Before and after costs",
        "Final value before costs",
        "Commission paid",
        "Monthly returns after costs",
        "Probabilistic Sharpe ratio",
        "does not account for multiple testing",
        "&quot;strategy&quot;: &quot;sma_crossover&quot;",  # the config used
        "test-run",
        "&lt;b&gt;escaped&lt;/b&gt;",
    ):
        assert text in page, text
    # Self-contained: nothing is fetched and nothing runs.
    for forbidden in ("<script", "<link", "<img", 'src="', 'href="http'):
        assert forbidden not in page, forbidden
    assert render_report(result, {"Run id": "test-run"}) == render_report(
        result, {"Run id": "test-run"}
    )


def test_benchmark_curve_and_report_helpers() -> None:
    closes = pd.Series([50.0, 55.0], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))
    sessions = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert benchmark_curve(closes, sessions, 1000.0).tolist() == [1000.0, 1000.0, 1100.0]  # type: ignore[union-attr]
    assert benchmark_curve(closes, sessions - pd.Timedelta(days=5), 1000.0) is None  # no price yet

    equity = pd.Series(
        [100.0, 110.0, 121.0, 108.9],
        index=pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-29", "2024-03-28"]),
    )
    table = monthly_returns(equity)
    assert table.loc[2024, 1] == pytest.approx(0.10)  # from the first day, not from nothing
    assert table.loc[2024, 3] == pytest.approx(-0.10)
    assert table.shape == (1, 12)
    assert format_metric(0.12345, "pct") == "12.35%"
    assert format_metric(1234.5, "money") == "1,234.50"
    assert format_metric(float("nan"), "num") == "-" and format_metric(None, "int") == "-"


# --------------------------------------------------------------------------- registry


def test_execute_logs_successes_and_failures(project: Path) -> None:
    config = load_backtest_config(write(project, "ok", config_text("buy_and_hold", "vectorized")))
    good = execute(config, notes="first try")
    assert good.error is None and good.result is not None
    assert good.result.data_snapshot == AFTER_JAN_16
    assert Path(good.record.report_path or "").is_file()

    broken = config.model_copy(update={"strategy": "does_not_exist"})
    bad = execute(broken)
    assert bad.result is None and isinstance(bad.error, ValueError)

    runs = registry.read_runs()
    assert list(runs.columns) == registry.COLUMNS
    assert runs["status"].tolist() == ["failed", "ok"]  # newest first; the failure is kept
    failed, ok = runs.iloc[0], runs.iloc[1]
    assert "unknown strategy 'does_not_exist'" in failed["notes"]
    assert pd.isna(failed["sharpe"]) and pd.isna(failed["report_path"])
    assert ok["notes"] == "first try" and ok["engine"] == "vectorized"
    assert ok["config_hash"] == config.config_hash() and ok["config_json"] == config.to_json()
    assert ok["data_snapshot"] == "2024-01-17 00:00:00"
    assert str(ok["start_date"])[:10] == "2015-04-01" and str(ok["end_date"])[:10] == "2016-02-24"
    assert ok["sharpe"] == pytest.approx(good.result.metrics["sharpe"])
    assert ok["turnover"] == pytest.approx(good.result.metrics["turnover"])
    # Ids are unique even within one second, and a prefix finds a run.
    assert good.record.run_id != bad.record.run_id
    assert registry.read_runs(run_ids=[good.record.run_id[:18]])["run_id"].tolist() == [
        good.record.run_id
    ]
    assert registry.read_runs(strategy="buy_and_hold")["status"].tolist() == ["ok"]
    assert len(registry.read_runs(limit=1)) == 1


def test_registry_reads_before_anything_is_logged(tmp_path: Path) -> None:
    assert registry.read_runs(tmp_path).empty
    assert list(registry.read_runs(tmp_path).columns) == registry.COLUMNS
    commit, dirty = registry.git_state(tmp_path)  # not a repository
    assert commit is None and dirty is None


# --------------------------------------------------------------------------- CLI


@pytest.mark.parametrize("engine", ["vectorized", "event"])
@pytest.mark.parametrize("strategy", ["buy_and_hold", "sma_crossover", "xs_momentum"])
def test_backtest_run_for_every_strategy_and_engine(
    project: Path, strategy: str, engine: str
) -> None:
    result = runner.invoke(
        app, ["backtest", "run", write(project, "c", config_text(strategy, engine))]
    )
    assert result.exit_code == 0, result.output
    for text in (
        "after costs",
        "before costs",
        "benchmark BBB",
        "Sharpe ratio",
        "Run id:",
        "Report:",
    ):
        assert text in result.output
    reports = list((project / "reports").glob("*.html"))
    assert len(reports) == 1 and f"{strategy}_{engine}" in reports[0].name
    assert "Before and after costs" in reports[0].read_text(encoding="utf-8")
    runs = registry.read_runs()
    assert runs["status"].tolist() == ["ok"] and runs["strategy"].tolist() == [strategy]
    assert Path(runs["report_path"].iloc[0]).resolve() == reports[0].resolve()
    assert runs["sharpe"].notna().all() and runs["max_drawdown"].iloc[0] <= 0


def test_backtest_list_and_compare(project: Path) -> None:
    empty = runner.invoke(app, ["backtest", "list"])
    assert empty.exit_code == 0 and "No backtest runs logged yet" in empty.output

    vector = write(project, "v", config_text("sma_crossover", "vectorized"))
    event = write(project, "e", config_text("sma_crossover", "event"))
    assert runner.invoke(app, ["backtest", "run", vector, "--notes", "screen"]).exit_code == 0
    assert (
        runner.invoke(app, ["backtest", "run", event, "--reports-dir", "elsewhere"]).exit_code == 0
    )
    assert len(list((project / "elsewhere").glob("*.html"))) == 1
    failing = write(
        project, "f", config_text("sma_crossover", "event", params="{ fast: 9, slow: 3 }")
    )
    failed = runner.invoke(app, ["backtest", "run", failing])
    assert failed.exit_code == 1
    assert "Logged as failed run" in failed.output and "fast < slow" in failed.output

    listed = runner.invoke(app, ["backtest", "list"])
    assert listed.exit_code == 0, listed.output
    assert listed.output.count("sma_crossover_vectorized") == 1
    assert "failed" in listed.output and "ok" in listed.output
    assert (
        "xs_momentum"
        not in runner.invoke(app, ["backtest", "list", "--strategy", "xs_momentum"]).output
    )

    runs = registry.read_runs()
    ids = runs[runs["status"] == "ok"]["run_id"].tolist()
    compared = runner.invoke(app, ["backtest", "compare", *ids])
    assert compared.exit_code == 0, compared.output
    for text in (
        *ids,
        "vectorized",
        "event",
        "Sharpe",
        "CAGR",
        "max drawdown",
        "turnover",
        "config hash",
    ):
        assert text in compared.output
    unknown = runner.invoke(app, ["backtest", "compare", "nope"])
    assert unknown.exit_code == 1 and "no run with an id starting with 'nope'" in unknown.output
    ambiguous = runner.invoke(app, ["backtest", "compare", "2"])
    assert ambiguous.exit_code == 1 and "3 runs with an id starting with '2'" in ambiguous.output


def test_backtest_run_reports_config_and_data_problems(project: Path) -> None:
    missing = runner.invoke(app, ["backtest", "run", str(project / "nowhere.yaml")])
    assert missing.exit_code == 1 and "backtest config not found" in missing.output
    assert registry.read_runs().empty  # a config that cannot be read is not an experiment

    text = config_text("buy_and_hold", "event").replace("universe: trio", "universe: [AAA, ZZZ]")
    unknown = runner.invoke(app, ["backtest", "run", write(project, "u", text)])
    assert unknown.exit_code == 1 and "unknown symbols: ZZZ" in unknown.output
    assert registry.read_runs()["status"].tolist() == ["failed"]


def test_market_data_from_the_store_matches_frames(project: Path) -> None:
    from qr.backtest.data import load_market_data

    stored = load_market_data(["AAA", "BBB"], date(2016, 2, 25))
    frames = {
        s: synthetic_bars(300, seed=i, dividend=0.3 if i == 0 else 0.0)
        for i, s in enumerate(["AAA", "BBB"])
    }
    built = MarketData.from_frames(
        {s: f for s, (f, _) in frames.items()}, {s: a for s, (_, a) in frames.items()}
    )
    pd.testing.assert_frame_equal(stored.raw, built.raw)
    pd.testing.assert_frame_equal(stored.adjusted, built.adjusted)
    pd.testing.assert_frame_equal(stored.actions, built.actions, check_dtype=False)
    assert stored.asset_classes == {"AAA": "equity", "BBB": "etf"}
    # The average-volume panel never includes the day it is used on.
    assert pd.isna(stored.adv_value(20)[0]).all()
    assert stored.adv_value(20)[1, 0] == pytest.approx(stored.panel("close")[0, 0] * 1_000_000)
