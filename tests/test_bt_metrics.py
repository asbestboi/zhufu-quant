"""Each metric against a series whose answer is known in advance."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qr.backtest import metrics as m
from qr.backtest.result import ACCOUNT_COLUMNS, COST_COLUMNS, EngineOutput, empty_trades


def curve(values: list[float], start: str = "2023-01-02") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


def from_returns(returns: list[float]) -> pd.Series:
    return curve(list(100.0 * np.cumprod([1.0, *(1.0 + r for r in returns)])))


def test_constant_return_series() -> None:
    """Growing 0.1% every day: no drawdown, no volatility, and so no Sharpe ratio."""
    equity = from_returns([0.001] * 252)
    stats = m.equity_metrics(equity)
    assert stats["total_return"] == pytest.approx(1.001**252 - 1)
    assert stats["max_drawdown"] == 0.0
    assert stats["time_underwater"] == 0.0
    assert stats["drawdown_duration_days"] == 0
    assert stats["volatility"] == pytest.approx(0.0, abs=1e-12)
    # Zero risk gives no ratio rather than an infinite one.
    assert math.isnan(stats["sharpe"]) and math.isnan(stats["sortino"])
    assert math.isnan(stats["calmar"]) and math.isnan(stats["psr"])


def test_alternating_returns_have_known_ratios() -> None:
    """+2%, -1%, repeated: mean 0.5%, every return 1.5% away from it."""
    n = 200
    returns = m.daily_returns(from_returns([0.02, -0.01] * (n // 2)))
    deviation = 0.015 * math.sqrt(n / (n - 1))
    assert m.annualized_volatility(returns) == pytest.approx(deviation * math.sqrt(252))
    assert m.sharpe_ratio(returns) == pytest.approx(0.005 / deviation * math.sqrt(252))
    # Downside: half the days lose 1%, so the root mean square is 0.01 * sqrt(1/2).
    assert m.sortino_ratio(returns) == pytest.approx(
        0.005 / (0.01 * math.sqrt(0.5)) * math.sqrt(252)
    )
    # A risk-free rate of 252 * 0.5% a year eats the whole mean.
    assert m.sharpe_ratio(returns, risk_free_rate=0.005 * 252) == pytest.approx(0.0, abs=1e-9)
    assert m.max_drawdown(from_returns([0.02, -0.01] * (n // 2))) == pytest.approx(-0.01)


def test_cagr_uses_calendar_time() -> None:
    equity = pd.Series(
        [100.0, 150.0, 200.0], index=pd.to_datetime(["2020-01-01", "2020-07-01", "2022-01-01"])
    )
    years = 731 / 365.25
    assert m.cagr(equity) == pytest.approx(2.0 ** (1 / years) - 1)
    assert m.total_return(equity) == 1.0
    assert math.isnan(m.cagr(equity.iloc[:1]))
    assert math.isnan(m.cagr(pd.Series([100.0, -5.0], index=equity.index[:2])))  # wiped out


def test_drawdown_statistics() -> None:
    #        peak      trough         regained   new peak    still under
    equity = curve([100.0, 110.0, 105.0, 99.0, 110.0, 120.0, 115.0])
    assert m.drawdowns(equity).round(6).tolist() == [0, 0, -0.045455, -0.1, 0, 0, -0.041667]
    assert m.max_drawdown(equity) == pytest.approx(-0.1)
    # Peak on Tue Jan 3, regained on Fri Jan 6: three days. The open one at the end is one.
    assert m.drawdown_duration(equity) == 3
    assert m.time_underwater(equity) == pytest.approx(3 / 7)
    assert m.calmar_ratio(equity) == pytest.approx(m.cagr(equity) / 0.1)
    # A drawdown that never recovers is measured up to the last date.
    assert m.drawdown_duration(curve([100.0, 90.0, 80.0, 85.0])) == 3


def test_probabilistic_sharpe_ratio() -> None:
    symmetric = pd.Series([0.01, -0.01] * 50)
    assert m.probabilistic_sharpe_ratio(symmetric) == pytest.approx(0.5)  # no edge: a coin flip

    returns = pd.Series([0.02, -0.01, 0.015, 0.0, -0.005, 0.01, 0.005, -0.012, 0.018, 0.003])
    n = len(returns)
    mean, spread = returns.mean(), returns.std(ddof=1)
    sharpe = mean / spread
    centered = returns - mean
    sigma = math.sqrt((centered**2).mean())
    skew = (centered**3).mean() / sigma**3
    kurtosis = (centered**4).mean() / sigma**4
    z = sharpe * math.sqrt(n - 1) / math.sqrt(1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2)
    expected = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert m.probabilistic_sharpe_ratio(returns) == pytest.approx(expected)
    assert 0.5 < expected < 1.0
    # A higher bar is harder to clear, and more data makes the same Sharpe more certain.
    assert m.probabilistic_sharpe_ratio(returns, benchmark_sharpe=2.0) < expected
    assert m.probabilistic_sharpe_ratio(pd.concat([returns] * 5, ignore_index=True)) > expected


def test_beta_correlation_and_alpha() -> None:
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0.0005, 0.01, 400))
    doubled = 2.0 * market + 0.0001
    stats = m.benchmark_statistics(doubled, market)
    assert stats["beta"] == pytest.approx(2.0)
    assert stats["correlation"] == pytest.approx(1.0)
    assert stats["alpha"] == pytest.approx(0.0001 * 252)
    assert m.benchmark_statistics(-market, market)["correlation"] == pytest.approx(-1.0)
    flat = m.benchmark_statistics(market, pd.Series(0.0, index=market.index))
    assert all(math.isnan(v) for v in flat.values())


def test_trade_statistics() -> None:
    trades = empty_trades()
    assert m.trade_statistics(trades)["closed_trades"] == 0
    assert math.isnan(m.trade_statistics(trades)["hit_rate"])
    trades = pd.DataFrame({"realized_pnl": [0.0, 10.0, -5.0, 0.0, 20.0, np.nan]})
    stats = m.trade_statistics(trades)  # opening trades (0) and unknowns (NaN) do not count
    assert stats["closed_trades"] == 3
    assert stats["hit_rate"] == pytest.approx(2 / 3)
    assert stats["average_win"] == 15.0 and stats["average_loss"] == -5.0
    assert stats["profit_factor"] == 6.0


def output_with(equity: pd.Series, exposure: float, traded: dict[int, float]) -> EngineOutput:
    accounts = pd.DataFrame(np.nan, index=equity.index, columns=ACCOUNT_COLUMNS)
    accounts["equity"] = equity
    accounts["gross_exposure"] = accounts["net_exposure"] = equity * exposure
    trades = empty_trades()
    if traded:
        trades = pd.DataFrame(
            {"date": [equity.index[i] for i in traded], "value": list(traded.values())}
        ).reindex(columns=trades.columns)
    costs = pd.DataFrame(0.0, index=equity.index, columns=COST_COLUMNS)
    return EngineOutput(
        accounts=accounts, positions=pd.DataFrame(index=equity.index), trades=trades, costs=costs
    )


def test_turnover_holding_period_and_exposure() -> None:
    equity = curve([1000.0] * 252)
    # Half invested; 500 bought on day 0 and 500 sold on day 126: 1000 traded in a year.
    output = output_with(equity, 0.5, {0: 500.0, 126: 500.0})
    assert output.turnover.sum() == pytest.approx(1.0)
    assert m.annualized_turnover(output.turnover) == pytest.approx(1.0)
    # One-way turnover of 0.5 a year on an average exposure of 0.5: held for a year.
    assert m.average_holding_period(output) == pytest.approx(252.0)
    stats = m.compute_metrics(output, benchmark=equity * 2)
    assert stats["average_gross_exposure"] == pytest.approx(0.5)
    assert stats["trades"] == 2 and stats["total_costs"] == 0.0
    assert math.isnan(stats["beta"])  # a flat benchmark has no variance to regress on
    assert math.isnan(m.average_holding_period(output_with(equity, 0.5, {})))


def test_rolling_statistics_use_full_windows_only() -> None:
    returns = m.daily_returns(from_returns([0.02, -0.01] * 150))
    sharpe, volatility = m.rolling_sharpe(returns), m.rolling_volatility(returns)
    assert sharpe.iloc[:251].isna().all() and sharpe.iloc[251:].notna().all()
    window = returns.iloc[:252]
    assert sharpe.iloc[251] == pytest.approx(m.sharpe_ratio(window))
    assert volatility.iloc[251] == pytest.approx(m.annualized_volatility(window))
