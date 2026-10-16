"""A symbol's daily total return as the risk-free rate (``risk_free: {symbol: ...}``)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from qr.backtest import registry
from qr.backtest.config import BacktestConfig, load_backtest_config
from qr.backtest.metrics import (
    excess_over,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from qr.backtest.runner import execute, load_risk_free_returns, run_backtest
from qr.cli import app
from qr.load import get_bars
from qr.research.deflated import return_moments
from qr.research.walkforward import ANCHORED, walk_forward

from .test_research_splits import config_text, study, write
from .test_research_walkforward import regimes, side_config

__all__ = ["regimes", "study"]  # the fixtures
runner = CliRunner()


def daily(values: list[float], start: str = "2024-01-02") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


# ------------------------------------------------------------------ metrics


def test_excess_over_a_series_and_a_constant() -> None:
    returns = daily([0.01, -0.02, 0.005, 0.0])
    rf = daily([0.0001, 0.0002, 0.0001, 0.0003])
    pd.testing.assert_series_equal(excess_over(returns, rf), returns - rf)
    # The constant path is the same arithmetic as before: annual rate over 252.
    pd.testing.assert_series_equal(excess_over(returns, 0.0252), returns - 0.0252 / 252)
    longer = daily([0.0001] * 10, start="2023-12-25")  # extra days are fine
    assert excess_over(returns, longer).notna().all()


def test_a_missing_risk_free_day_is_an_error() -> None:
    returns = daily([0.01, -0.02, 0.005])
    with pytest.raises(ValueError, match="no return on 1 of the 3 days, first 2024-01-04"):
        excess_over(returns, daily([0.0001, 0.0002]))


def test_sharpe_sortino_psr_against_a_series() -> None:
    rng = np.random.default_rng(2)
    returns = daily(list(0.0005 + 0.01 * rng.standard_normal(300)))
    rf = daily(list(0.0001 + 0.00005 * rng.standard_normal(300)))
    excess = returns - rf
    expected = excess.mean() / excess.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(returns, rf) == pytest.approx(expected, rel=1e-12)
    assert sharpe_ratio(returns, rf) != sharpe_ratio(returns, 0.0)
    downside = math.sqrt((np.minimum(excess.to_numpy(), 0) ** 2).mean())
    assert sortino_ratio(returns, rf) == pytest.approx(excess.mean() / downside * math.sqrt(252))
    zeros = daily([0.0] * 300)
    assert sharpe_ratio(returns, zeros) == sharpe_ratio(returns, 0.0)
    assert probabilistic_sharpe_ratio(excess) == pytest.approx(
        probabilistic_sharpe_ratio(returns - rf)
    )


# ------------------------------------------------------------------ config


def test_config_option_and_hash() -> None:
    raw = {"name": "t", "engine": "vectorized", "strategy": "buy_and_hold", "universe": ["A"],
           "start": "2020-01-01", "end": "2020-12-31"}  # fmt: skip
    plain = BacktestConfig.model_validate(raw)
    shy = BacktestConfig.model_validate(raw | {"risk_free": {"symbol": "SHY"}})
    assert plain.risk_free is None and shy.risk_free is not None
    assert shy.risk_free.symbol == "SHY"
    assert plain.config_hash() == BacktestConfig.model_validate(raw).config_hash()
    assert shy.config_hash() != plain.config_hash()
    with pytest.raises(ValueError, match="either risk_free_rate or risk_free"):
        BacktestConfig.model_validate(
            raw | {"risk_free": {"symbol": "SHY"}, "risk_free_rate": 0.02}
        )
    with pytest.raises(ValueError, match="symbol"):
        BacktestConfig.model_validate(raw | {"risk_free": {"symbol": ""}})


# ------------------------------------------------------------------ runs


def rf_config(study: Path, segment: str = "train", **extra: str) -> BacktestConfig:
    text = config_text(segment, risk_free="{ symbol: BBB }", **extra)
    return load_backtest_config(write(study, f"rf_{segment}", text))


def test_every_metric_and_the_stress_points_use_the_series(study: Path) -> None:
    config = rf_config(study)
    result = run_backtest(config)
    rf = get_bars("BBB", end=config.end)["close"].droplevel("symbol").pct_change()
    returns = result.equity.pct_change().iloc[1:]
    excess = returns - rf.reindex(returns.index)
    assert result.metrics["sharpe"] == pytest.approx(
        excess.mean() / excess.std(ddof=1) * math.sqrt(252), rel=1e-12
    )
    assert result.metrics["psr"] == pytest.approx(probabilistic_sharpe_ratio(excess), rel=1e-12)
    assert result.risk_free_returns is not None
    assert result.risk_free_returns.index.max() <= pd.Timestamp(config.end)  # no look-ahead
    # The stress points (and so the fragile flag) use it too.
    plain = run_backtest(config.model_copy(update={"risk_free": None}))
    assert result.metrics["sharpe"] != plain.metrics["sharpe"]
    assert result.metrics["cagr"] == plain.metrics["cagr"]  # not an excess-return number
    sensitivity = result.cost_sensitivity
    assert sensitivity is not None
    two = next(p for p in sensitivity.points if p.multiplier == 2.0)
    assert two.sharpe == result.stress_metrics[2.0]["sharpe"]
    assert two.sharpe != plain.stress_metrics[2.0]["sharpe"]
    # The benchmark is measured against the same rate.
    assert result.benchmark_metrics["sharpe"] != plain.benchmark_metrics["sharpe"]


def test_a_zero_series_matches_the_constant_zero(study: Path) -> None:
    config = rf_config(study)
    plain = run_backtest(config.model_copy(update={"risk_free": None}), cost_stress=False)
    days = plain.equity.index
    zeros = pd.Series(0.0, index=days)
    as_series = run_backtest(config, cost_stress=False, risk_free_returns=zeros)
    for key in ("sharpe", "sortino", "psr", "alpha"):
        assert as_series.metrics[key] == pytest.approx(plain.metrics[key], rel=1e-12, abs=1e-15)


def test_the_series_is_read_up_to_the_end_only(study: Path) -> None:
    config = rf_config(study)
    series = load_risk_free_returns(config)
    assert series.index.max() <= pd.Timestamp(config.end)
    later = load_risk_free_returns(config, end=pd.Timestamp("2015-12-31").date())
    assert later.index.max() > series.index.max()
    with pytest.raises(ValueError, match="no risk_free symbol"):
        load_risk_free_returns(config.model_copy(update={"risk_free": None}))


def test_reports_say_which_rate_was_used(study: Path) -> None:
    outcome = execute(rf_config(study, "validation", study_id="rf_study"))
    assert outcome.error is None and outcome.result is not None
    row = registry.read_runs().iloc[0]
    assert row["risk_free"] == "BBB daily total return"
    assumptions = outcome.result.assumptions
    assert assumptions is not None
    assert assumptions["sections"]["Data"]["Risk-free rate"].startswith("BBB daily total return")
    page = Path(str(outcome.record.report_path)).read_text(encoding="utf-8")
    assert "Risk-free rate for Sharpe" in page and "BBB daily total return" in page
    # The DSR's moments are of returns in excess of the series.
    excess = outcome.result.equity.pct_change().iloc[1:]
    excess = excess - outcome.result.risk_free_returns.reindex(excess.index)  # type: ignore[union-attr]
    assert row["return_skew"] == pytest.approx(return_moments(excess).skew)
    shown = runner.invoke(app, ["backtest", "show", row["run_id"]])
    assert "BBB daily total return" in shown.output
    path = write(study, "plain", config_text("validation"))
    ran = runner.invoke(app, ["backtest", "run", str(path), "--no-cost-stress"])
    assert "Risk-free rate: constant 0.00% a year" in ran.output
    assert registry.read_runs().iloc[0]["risk_free"] == "constant 0.00% a year"


def test_walk_forward_selects_on_excess_returns(regimes: Path) -> None:
    from qr.research.walkforward import _window_config

    config = load_backtest_config(side_config(regimes))
    with_rf = BacktestConfig.model_validate(config.model_dump() | {"risk_free": {"symbol": "TRD"}})
    grid = {"side": [1, -1]}
    plain = walk_forward(config, "regimes", ANCHORED, 2, 12, 12, grid=grid, log=False)
    result = walk_forward(with_rf, "regimes", ANCHORED, 2, 12, 12, grid=grid)
    for fold, base in zip(result.folds, plain.folds, strict=True):
        # The metric the choice was made on is the excess-return Sharpe of that window,
        # with the risk-free series read only up to the train window's end.
        window = _window_config(with_rf, fold.chosen, fold.fold.train_start, fold.fold.train_end)
        alone = run_backtest(window, cost_stress=False)
        assert alone.risk_free_returns is not None
        assert alone.risk_free_returns.index.max() <= pd.Timestamp(fold.fold.train_end)
        assert fold.train_metrics["sharpe"] == alone.metrics["sharpe"]
        if fold.chosen == base.chosen:
            assert fold.train_metrics["sharpe"] != base.train_metrics["sharpe"]
    assert result.metrics["sharpe"] != plain.metrics["sharpe"]  # the stitched curve too
    logged = registry.read_runs(include_walkforward=True)
    assert len(logged) and (logged["risk_free"] == "TRD daily total return").all()


def test_sweep_trials_and_dsr_use_the_series(study: Path) -> None:
    from qr.research.sweeps import load_sweep, run_sweep, summarize_sweep

    write(study, "base", config_text("train", risk_free="{ symbol: BBB }"))
    spec = write(
        study, "sweep", "name: rf_grid\nbase: base.yaml\nstudy_id: rf_v1\n"
        "grid:\n  lookback: [20, 40]\n",
    )  # fmt: skip
    result = run_sweep(load_sweep(spec))
    trials = registry.read_runs()
    trials = trials[trials["sweep_id"] == result.sweep_id].sort_values("trial_index")
    assert (trials["risk_free"] == "BBB daily total return").all()
    single = run_backtest(rf_config(study).model_copy(update={"params": {
        "lookback": 20, "skip": 5, "top_n": 1}}), cost_stress=False)  # fmt: skip
    assert trials.iloc[0]["sharpe"] == pytest.approx(single.metrics["sharpe"], rel=1e-12)
    summary = summarize_sweep(result.sweep_id)
    assert summary.deflated is not None
    shown = runner.invoke(app, ["sweep", "show", result.sweep_id])
    assert "Risk-free rate: BBB daily total return" in shown.output
