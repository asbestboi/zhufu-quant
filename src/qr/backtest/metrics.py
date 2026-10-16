"""Performance and risk statistics, computed the same way for both engines.

Everything starts from the daily equity curve. Returns are simple daily returns; a year is
252 sessions for annualizing and 365.25 calendar days for CAGR. A statistic that cannot
be computed (too little data, zero volatility, no trades) is NaN, never a made-up number.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from qr.backtest.result import EngineOutput

PERIODS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25
ROLLING_WINDOW = 252  # sessions, for the rolling one-year statistics


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple return of each day against the previous one (the first day has none)."""
    return equity.pct_change().iloc[1:]


def total_return(equity: pd.Series) -> float:
    """Growth of the curve from its first to its last value."""
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else math.nan


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate over the calendar time the curve spans."""
    if len(equity) < 2:
        return math.nan
    years = (equity.index[-1] - equity.index[0]).days / DAYS_PER_YEAR
    growth = equity.iloc[-1] / equity.iloc[0]
    if years <= 0 or growth <= 0:
        return math.nan
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    """Sample standard deviation of daily returns, scaled to a year."""
    return (
        float(returns.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)) if len(returns) > 1 else math.nan
    )


# A risk-free rate: a constant annual rate, or a series of daily risk-free returns indexed
# by date (e.g. a short-term Treasury fund's total return; see BacktestConfig.risk_free).
RiskFree = float | pd.Series


def excess_over(returns: pd.Series, risk_free_rate: RiskFree) -> pd.Series:
    """``returns`` less the risk-free return of the same day.

    A series must cover every day of ``returns``: a missing day is an error, never a
    silent zero.
    """
    if isinstance(risk_free_rate, pd.Series):
        aligned = risk_free_rate.reindex(returns.index)
        missing = aligned.index[aligned.isna()]
        if len(missing):
            raise ValueError(
                f"the risk-free series has no return on {len(missing)} of the "
                f"{len(returns)} days, first {pd.Timestamp(missing[0]).date()}"
            )
        return returns - aligned
    return returns - risk_free_rate / PERIODS_PER_YEAR


_excess = excess_over


def sharpe_ratio(returns: pd.Series, risk_free_rate: RiskFree = 0.0) -> float:
    """Annualized mean excess return over its standard deviation. NaN when that is zero."""
    excess = _excess(returns, risk_free_rate)
    deviation = excess.std(ddof=1) if len(excess) > 1 else math.nan
    if not deviation > 1e-15:
        return math.nan
    return float(excess.mean() / deviation * math.sqrt(PERIODS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: RiskFree = 0.0) -> float:
    """Like Sharpe, but only returns below the risk-free rate count as risk.

    The downside deviation is the root mean square of the negative excess returns, taken
    over all days. NaN when there is no downside at all.
    """
    excess = _excess(returns, risk_free_rate)
    if len(excess) == 0:
        return math.nan
    downside = math.sqrt(float((np.minimum(excess.to_numpy(), 0.0) ** 2).mean()))
    if not downside > 1e-15:
        return math.nan
    return float(excess.mean() / downside * math.sqrt(PERIODS_PER_YEAR))


def drawdowns(equity: pd.Series) -> pd.Series:
    """How far the curve is below its running peak, as a fraction (0 at a new high)."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """The deepest drawdown, as a negative fraction (0.0 if the curve never falls)."""
    return float(drawdowns(equity).min()) if len(equity) else math.nan


def drawdown_duration(equity: pd.Series) -> int:
    """Calendar days of the longest stretch from a peak until that peak is regained.

    A drawdown still open at the end counts up to the last date.
    """
    if len(equity) == 0:
        return 0
    under = drawdowns(equity) < 0
    longest = 0
    peak_date = equity.index[0]
    was_under = False
    for date, is_under in under.items():
        if is_under or was_under:  # the day the peak is regained ends the stretch
            longest = max(longest, (date - peak_date).days)
        if not is_under:
            peak_date = date
        was_under = bool(is_under)
    return int(longest)


def time_underwater(equity: pd.Series) -> float:
    """Fraction of days spent below an earlier peak."""
    return float((drawdowns(equity) < 0).mean()) if len(equity) else math.nan


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR divided by the size of the maximum drawdown. NaN without a drawdown."""
    deepest = max_drawdown(equity)
    return float(cagr(equity) / abs(deepest)) if deepest < 0 else math.nan


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    """Probability that the true Sharpe ratio is above ``benchmark_sharpe`` (annualized).

    Bailey and López de Prado (2012): the observed Sharpe ratio is judged against its
    standard error, which grows with short samples, negative skew, and fat tails. It
    answers "is this one result distinguishable from luck?" and nothing more: it does
    **not** correct for having tried many strategies. That needs the Deflated Sharpe
    Ratio and the full list of runs in the registry (step 3).
    """
    n = len(returns)
    deviation = returns.std(ddof=1) if n > 2 else math.nan
    if not deviation > 1e-15:
        return math.nan
    observed = float(returns.mean() / deviation)  # per period
    target = benchmark_sharpe / math.sqrt(PERIODS_PER_YEAR)
    centered = returns - returns.mean()
    spread = float((centered**2).mean()) ** 0.5
    skew = float((centered**3).mean()) / spread**3
    kurtosis = float((centered**4).mean()) / spread**4
    variance = 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if variance <= 0:
        return math.nan
    z = (observed - target) * math.sqrt(n - 1) / math.sqrt(variance)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def benchmark_statistics(
    returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: RiskFree = 0.0
) -> dict[str, float]:
    """Beta, correlation, and annualized alpha against a benchmark's daily returns.

    ``alpha`` is the intercept of the regression of excess returns on the benchmark's
    excess returns, times 252.
    """
    joined = pd.concat([returns.rename("r"), benchmark_returns.rename("b")], axis=1).dropna()
    nan = {"beta": math.nan, "correlation": math.nan, "alpha": math.nan}
    if len(joined) < 3:
        return nan
    own = _excess(joined["r"], risk_free_rate)
    other = _excess(joined["b"], risk_free_rate)
    variance = float(other.var(ddof=1))
    if not variance > 1e-18:
        return nan
    beta = float(own.cov(other) / variance)
    spread = float(own.std(ddof=1))
    return {
        "beta": beta,
        "correlation": float(own.corr(other)) if spread > 1e-15 else math.nan,
        "alpha": float((own.mean() - beta * other.mean()) * PERIODS_PER_YEAR),
    }


def annualized_turnover(turnover: pd.Series) -> float:
    """Average daily traded value over equity, times 252.

    Buys and sells both count, so selling everything and buying something else is 2.0.
    """
    return float(turnover.mean() * PERIODS_PER_YEAR) if len(turnover) else math.nan


def average_holding_period(output: EngineOutput) -> float:
    """Roughly how many sessions a dollar stays invested: average exposure over turnover.

    One-way turnover (half of buys plus sells) replaces the whole book once per holding
    period. NaN when nothing was traded.
    """
    exposure = float((output.accounts["gross_exposure"] / output.equity).mean())
    one_way = annualized_turnover(output.turnover) / 2.0
    return float(exposure / one_way * PERIODS_PER_YEAR) if one_way > 0 else math.nan


def trade_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """Hit rate, average win and loss, and profit factor of the closing trades.

    Uses each trade's realized P&L, so it needs an engine that tracks cost basis (the
    event engine). P&L is before commissions; spread and slippage are in the prices.
    """
    nan = {
        "closed_trades": 0.0,
        "hit_rate": math.nan,
        "average_win": math.nan,
        "average_loss": math.nan,
        "profit_factor": math.nan,
    }
    if len(trades) == 0:
        return nan
    pnl = trades["realized_pnl"].dropna()
    pnl = pnl[pnl != 0]
    if len(pnl) == 0:
        return nan
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "closed_trades": float(len(pnl)),
        "hit_rate": float(len(wins) / len(pnl)),
        "average_win": float(wins.mean()) if len(wins) else math.nan,
        "average_loss": float(losses.mean()) if len(losses) else math.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else math.nan,
    }


def rolling_sharpe(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Sharpe ratio of each trailing ``window`` sessions (NaN until the window is full)."""
    rolling = returns.rolling(window, min_periods=window)
    deviation = rolling.std(ddof=1)
    ratio = rolling.mean() / deviation.where(deviation > 1e-15) * math.sqrt(PERIODS_PER_YEAR)
    return ratio.rename("rolling_sharpe")


def rolling_volatility(returns: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Annualized volatility of each trailing ``window`` sessions."""
    rolling = returns.rolling(window, min_periods=window).std(ddof=1)
    return (rolling * math.sqrt(PERIODS_PER_YEAR)).rename("rolling_volatility")


def equity_metrics(equity: pd.Series, risk_free_rate: RiskFree = 0.0) -> dict[str, float]:
    """The statistics that need nothing but an equity curve (also used for the benchmark)."""
    returns = daily_returns(equity)
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_rate),
        "sortino": sortino_ratio(returns, risk_free_rate),
        "calmar": calmar_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "drawdown_duration_days": float(drawdown_duration(equity)),
        "time_underwater": time_underwater(equity),
        "psr": probabilistic_sharpe_ratio(_excess(returns, risk_free_rate)),
    }


def compute_metrics(
    output: EngineOutput,
    benchmark: pd.Series | None = None,
    risk_free_rate: RiskFree = 0.0,
) -> dict[str, float]:
    """Every headline statistic of one simulation, as a flat name -> number dict."""
    equity = output.equity
    exposure = output.accounts[["gross_exposure", "net_exposure"]].div(equity, axis=0)
    metrics = equity_metrics(equity, risk_free_rate)
    metrics.update(
        {
            "turnover": annualized_turnover(output.turnover),
            "average_holding_days": average_holding_period(output),
            "average_gross_exposure": float(exposure["gross_exposure"].mean()),
            "average_net_exposure": float(exposure["net_exposure"].mean()),
            "trades": float(len(output.trades)),
            "total_costs": float(output.costs.to_numpy().sum()),
        }
    )
    metrics.update(trade_statistics(output.trades))
    if benchmark is not None:
        metrics.update(
            benchmark_statistics(daily_returns(equity), daily_returns(benchmark), risk_free_rate)
        )
    return metrics
