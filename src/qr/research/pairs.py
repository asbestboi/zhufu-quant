"""Pairs trading: hedge ratios, spreads, cointegration tests, signals, and leg weights.

Everything here works on *log prices*: ``log(price / first price)`` of a total-return
adjusted series (see :func:`log_prices`). Every rolling function returns, on each row, a
value computed from that row and earlier rows only, so it can be used in a backtest
without looking ahead. The notebooks in ``notebooks/pairs/`` explain the ideas; this
module holds the code so that it can be tested.

Conventions:

* ``y`` is the first leg, ``x`` the second, and the spread is ``y - beta * x``.
* Being *long the spread* (+1) means long ``y`` and short ``x``; short the spread (-1) is
  the reverse.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

# --------------------------------------------------------------------------- prices


def log_prices[T: (pd.DataFrame, pd.Series)](closes: T) -> T:
    """Log of each series divided by its first valid value (so each starts at 0).

    Use total-return adjusted closes. Back-adjusted price *levels* change when a later
    dividend is paid, but the ratio of two adjusted prices does not, so rebasing on the
    first bar makes the result independent of how far the data goes.
    """
    first = closes.bfill().iloc[0]
    return cast(T, np.log(closes / first))


def _aligned(y: pd.Series, x: pd.Series) -> tuple[pd.Series, pd.Series]:
    both = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    return both["y"], both["x"]


# --------------------------------------------------------------------------- hedge ratio


def hedge_ratio_ols(y: pd.Series, x: pd.Series) -> float:
    """Slope of an OLS regression of ``y`` on ``x`` (with an intercept), on all the data.

    It uses the whole sample, future included, so it is for explanation only: a backtest
    must use :func:`rolling_hedge_ratio`.
    """
    yy, xx = _aligned(y, x)
    if len(yy) < 2:
        raise ValueError("need at least two overlapping observations")
    variance = float(xx.var())
    if variance == 0:
        raise ValueError("x is constant: no hedge ratio")
    return float(yy.cov(xx) / variance)


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """OLS slope of ``y`` on ``x`` over the past ``window`` days, shifted by one day.

    The value on day ``t`` uses data up to ``t-1`` only. NaN until the window is full.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    rolling_x = x.rolling(window, min_periods=window)
    beta = rolling_x.cov(y) / rolling_x.var()
    return beta.shift(1).rename("beta")


# --------------------------------------------------------------------------- spread


def spread(y: pd.Series, x: pd.Series, beta: float | pd.Series) -> pd.Series:
    """``y - beta * x``: what is left of ``y`` after hedging it with ``beta`` units of ``x``."""
    return (y - beta * x).rename("spread")


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """How many standard deviations ``s`` is from its mean over the past ``window`` days.

    Today's value is included (it is known at today's close); nothing later is.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    rolling = s.rolling(window, min_periods=window)
    return ((s - rolling.mean()) / rolling.std()).rename("z")


def rolling_spread_zscore(
    y: pd.Series, x: pd.Series, beta: float | pd.Series, window: int
) -> pd.Series:
    """Z-score of today's spread, using today's ``beta`` for the whole window.

    ``rolling_zscore(spread(y, x, beta))`` mixes spreads built with different hedge
    ratios when ``beta`` changes over time, and a small change in ``beta`` then moves the
    spread by an amount that depends on the price level. Here the past ``window`` days
    are re-hedged with the current ``beta`` before the mean and standard deviation are
    taken, which removes both problems. With a constant ``beta`` the two are equal.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    ry, rx = y.rolling(window, min_periods=window), x.rolling(window, min_periods=window)
    deviation = (y - ry.mean()) - beta * (x - rx.mean())
    variance = ry.var() + beta**2 * rx.var() - 2 * beta * ry.cov(x)
    return (deviation / np.sqrt(variance.where(variance > 0))).rename("z")


def half_life(s: pd.Series) -> float:
    """Mean-reversion half-life of ``s`` in rows (days), from an AR(1) regression.

    Fits ``s[t] - s[t-1] = a + b * s[t-1]`` and returns ``-ln 2 / ln(1 + b)``: the time
    a deviation from the mean takes to shrink by half. ``inf`` when ``b >= 0`` (no pull
    back towards the mean).
    """
    values = s.dropna().to_numpy(dtype="float64")
    if len(values) < 3:
        raise ValueError("need at least three observations")
    lagged, delta = values[:-1], np.diff(values)
    design = np.column_stack([np.ones_like(lagged), lagged])
    (_, b), *_ = np.linalg.lstsq(design, delta, rcond=None)
    if b >= 0:
        return math.inf
    if b <= -1:
        return math.nan  # overshoots the mean every day: not a decaying process
    return float(-math.log(2) / math.log1p(b))


# --------------------------------------------------------------------------- cointegration


def engle_granger(y: pd.Series, x: pd.Series) -> dict[str, Any]:
    """Engle-Granger cointegration test of ``y`` on ``x`` (``statsmodels`` ``coint``).

    Returns the test ``statistic``, its ``pvalue`` (small means "the spread looks
    stationary"), the ``critical_values`` at 1%, 5%, and 10%, and the number of
    ``observations`` used.
    """
    yy, xx = _aligned(y, x)
    statistic, pvalue, critical = coint(yy, xx)
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "critical_values": dict(zip(("1%", "5%", "10%"), map(float, critical), strict=True)),
        "observations": len(yy),
    }


def rolling_cointegration_pvalue(y: pd.Series, x: pd.Series, window: int, step: int) -> pd.Series:
    """Engle-Granger p-value over the past ``window`` days, every ``step`` days.

    Indexed by the last day of each window, so each value uses only data up to its date.
    """
    if window < 20 or step < 1:
        raise ValueError("need window >= 20 and step >= 1")
    yy, xx = _aligned(y, x)
    ends = range(window - 1, len(yy), step)
    pvalues = [
        float(coint(yy.iloc[i - window + 1 : i + 1], xx.iloc[i - window + 1 : i + 1])[1])
        for i in ends
    ]
    return pd.Series(pvalues, index=yy.index[list(ends)], name="pvalue", dtype="float64")


# --------------------------------------------------------------------------- trading rules


def pair_signals(z: pd.Series, entry: float, exit: float, stop: float) -> pd.Series:
    """Position in the spread, in {-1, 0, +1}, decided at each day's close.

    * Flat: go short the spread when ``entry < z < stop`` (it is unusually high), long
      when ``-stop < z < -entry``.
    * Long: exit when ``z >= -exit`` (back near the mean, or past it); stop out when
      ``z <= -stop``. Short is the mirror image.
    * After a stop no new trade is opened until ``|z| < exit``: a spread that has just
      broken out is not bought again on the way back.
    * A missing ``z`` closes any position.
    """
    if not 0 <= exit < entry < stop:
        raise ValueError(f"need 0 <= exit < entry < stop, got {exit}, {entry}, {stop}")
    values = z.to_numpy(dtype="float64")
    out = np.zeros(len(values), dtype="int64")
    position, blocked = 0, False
    for i, value in enumerate(values):
        # How far the spread still is on the side we bet against (positive while it is).
        stretch = -position * value
        if not math.isfinite(value):
            position = 0
        elif position and stretch >= stop:
            position, blocked = 0, True
        elif position and stretch <= exit:
            position = 0
        if position == 0 and math.isfinite(value):
            if blocked and abs(value) < exit:
                blocked = False
            if not blocked and entry < abs(value) < stop:
                position = -1 if value > 0 else 1
        out[i] = position
    return pd.Series(out, index=z.index, name="signal")


def pair_weights(
    signal: pd.Series,
    beta: float | pd.Series,
    gross: float = 1.0,
    names: tuple[str, str] = ("y", "x"),
) -> pd.DataFrame:
    """Portfolio weights of the two legs for a spread position.

    Long the spread is ``+1/(1+|b|)`` in ``y`` and ``-b/(1+|b|)`` in ``x``, times
    ``gross``, so the legs' absolute weights add up to ``gross`` and each dollar of ``y``
    is hedged with ``b`` dollars of ``x``. ``b`` is the hedge ratio on the day the trade
    was opened and is kept until it closes: re-hedging every day would trade small
    amounts constantly. Days without a position, or without a hedge ratio, get 0.
    """
    if gross <= 0:
        raise ValueError("gross must be positive")
    held = signal.astype("float64")
    beta_series = (
        beta.reindex(signal.index)
        if isinstance(beta, pd.Series)
        else pd.Series(float(beta), index=signal.index)
    )
    opened = (held != 0) & (held != held.shift(1, fill_value=0.0))
    frozen = beta_series.where(opened).ffill().where(held != 0)
    scale = gross / (1.0 + frozen.abs())
    weights = pd.DataFrame({names[0]: held * scale, names[1]: -held * frozen * scale})
    return weights.fillna(0.0)


def round_trips(signal: pd.Series) -> pd.DataFrame:
    """One row per trade in ``signal``: side, the days it was opened and closed, and length.

    ``opened`` and ``closed`` are the decision days (the closes at which the signal
    changed); ``closed`` is NaT for a trade still open on the last day. ``days`` counts
    the sessions the position was held.
    """
    rows: list[dict[str, Any]] = []
    values, dates = signal.to_numpy(), signal.index
    start: int | None = None
    for i, value in enumerate(values):
        previous = values[i - 1] if i else 0
        if start is not None and value != previous:
            rows.append(
                {
                    "side": int(previous),
                    "opened": dates[start],
                    "closed": dates[i],
                    "days": i - start,
                }
            )
            start = None
        if value != 0 and value != previous:
            start = i
    if start is not None:
        rows.append(
            {
                "side": int(values[start]),
                "opened": dates[start],
                "closed": pd.NaT,
                "days": len(values) - start,
            }
        )
    return pd.DataFrame(rows, columns=["side", "opened", "closed", "days"])


def round_trip_returns(trips: pd.DataFrame, equity: pd.Series) -> pd.Series:
    """Account return of each trade, costs included, from a single-pair backtest's equity.

    A trade decided at the close of ``opened`` is filled on the next session and closed
    on the session after ``closed``, so its return is the equity at that later close over
    the equity at the close of ``opened``. A trade still open uses the last equity.
    """
    index = equity.index
    out = []
    for opened, closed in zip(trips["opened"], trips["closed"], strict=True):
        start = int(index.searchsorted(opened))
        end = len(index) - 1 if pd.isna(closed) else int(index.searchsorted(closed)) + 1
        end = min(end, len(index) - 1)
        if start >= len(index):
            out.append(math.nan)
            continue
        out.append(float(equity.iloc[end] / equity.iloc[start] - 1.0))
    return pd.Series(out, index=trips.index, name="return", dtype="float64")
