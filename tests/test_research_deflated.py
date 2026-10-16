"""The Deflated Sharpe Ratio (part 4, second step)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from typer.testing import CliRunner

from qr.backtest import registry
from qr.backtest.config import load_backtest_config
from qr.backtest.metrics import probabilistic_sharpe_ratio
from qr.backtest.runner import execute
from qr.cli import app
from qr.research.deflated import (
    EULER_GAMMA,
    deflated_sharpe,
    expected_max_sharpe,
    moments_from_row,
    return_moments,
)
from qr.research.sweeps import load_sweep, run_sweep, summarize_sweep

from .test_research_splits import config_text, study, write
from .test_research_sweeps import GRID, spec

__all__ = ["study"]  # the fixture
runner = CliRunner()


def returns(seed: int = 4, n: int = 750) -> pd.Series:
    """A fixed, fat-tailed, positively drifting daily series."""
    rng = np.random.default_rng(seed)
    return pd.Series(0.0004 + 0.01 * rng.standard_t(5, n))


TRIALS = [0.4, 1.1, -0.2, 0.7, 0.9, 0.3, 1.4, 0.0]  # annualized, as the registry stores them


def test_one_trial_is_the_psr() -> None:
    series = returns()
    moments = return_moments(series)
    sharpe = moments.sharpe * math.sqrt(252)
    found = deflated_sharpe(moments, [sharpe])
    assert found.benchmark == 0.0 and found.n_trials == 1
    assert found.dsr == pytest.approx(probabilistic_sharpe_ratio(series, 0.0), rel=1e-12)


def test_more_trials_or_more_spread_deflate_more() -> None:
    moments = return_moments(returns())
    by_count = [deflated_sharpe(moments, TRIALS, n_trials=n).dsr for n in (2, 5, 20, 100, 1000)]
    assert all(a > b for a, b in itertools.pairwise(by_count))
    narrow = [s * 0.5 for s in TRIALS]
    wide = [s * 2.0 for s in TRIALS]
    spreads = [deflated_sharpe(moments, t).dsr for t in (narrow, TRIALS, wide)]
    assert spreads[0] > spreads[1] > spreads[2]
    assert expected_max_sharpe(10, 0.0) == 0.0  # identical trials: nothing to deflate


def test_matches_an_independent_step_by_step_computation() -> None:
    series = returns(seed=11, n=1000)
    found = deflated_sharpe(return_moments(series), TRIALS)

    # Independently: scipy for moments and the normal distribution.
    values = series.to_numpy()
    T = len(values)
    sr = values.mean() / values.std(ddof=1)
    skew = stats.skew(values, bias=True)
    kurt = stats.kurtosis(values, fisher=False, bias=True)
    daily = np.array(TRIALS) / np.sqrt(252)
    V = daily.var(ddof=1)
    N = len(TRIALS)
    sr0 = np.sqrt(V) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / N)
                        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (N * np.e)))  # fmt: skip
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    expected = stats.norm.cdf(z)

    assert found.n_trials == N
    assert found.sharpe_variance == pytest.approx(V, rel=1e-12)
    assert found.benchmark == pytest.approx(sr0, rel=1e-12)
    assert found.dsr == pytest.approx(expected, rel=1e-10)
    assert 0 < found.dsr < probabilistic_sharpe_ratio(series, 0.0)


def test_failed_trials_count_but_have_no_sharpe() -> None:
    moments = return_moments(returns())
    with_nan = deflated_sharpe(moments, [*TRIALS, math.nan], n_trials=len(TRIALS) + 1)
    assert with_nan.n_trials == 9
    assert with_nan.dsr < deflated_sharpe(moments, TRIALS).dsr


def test_moments_from_a_registry_row() -> None:
    row = {"n_returns": 500, "sharpe": 1.0, "return_skew": -0.2, "return_kurtosis": 4.5}
    moments = moments_from_row(row)
    assert moments is not None and moments.sharpe == pytest.approx(1 / math.sqrt(252))
    assert moments_from_row({**row, "return_skew": None}) is None
    assert moments_from_row({**row, "n_returns": float("nan")}) is None


# ------------------------------------------------------------------ in reports


def test_dsr_in_sweep_reports_and_study_runs(study: Path) -> None:
    result = run_sweep(load_sweep(spec(study)))
    summary = summarize_sweep(result.sweep_id)
    assert summary.deflated is not None and summary.deflated.n_trials == 6
    assert summary.best_index == 3  # lookback 40, skip 5 (see test_research_sweeps)
    assert 0 <= summary.deflated.dsr <= summary.psr <= 1
    shown = runner.invoke(app, ["sweep", "show", result.sweep_id])
    assert "DSR" in shown.output and "in 6 trials" in shown.output
    ran = runner.invoke(app, ["sweep", "run", str(spec(study, GRID, "again"))])
    assert ran.exit_code == 0 and "in 12 trials" in ran.output
    page = sorted((study / "reports").glob("*_sweep.html"))[-1].read_text(encoding="utf-8")
    assert "Deflated Sharpe (DSR)" in page

    text = config_text("validation", study_id="mom_v1")
    config = load_backtest_config(write(study, "chosen", text))
    outcome = execute(config)
    assert outcome.result is not None and outcome.result.deflated is not None
    assert outcome.result.deflated.n_trials == 13
    html = Path(str(outcome.record.report_path)).read_text(encoding="utf-8")
    assert "Deflated Sharpe (DSR)" in html and "in 13 trials" in html
    run_id = registry.read_runs().iloc[0]["run_id"]
    shown = runner.invoke(app, ["backtest", "show", run_id])
    assert "Deflated Sharpe (DSR)" in shown.output and "Trials in study mom_v1" in shown.output
