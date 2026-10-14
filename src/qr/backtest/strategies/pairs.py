"""Pairs trading on one pair: trade the spread back towards its mean.

Each day, at the close: the hedge ratio of ``y`` on ``x`` is estimated over the past
``beta_window`` days (up to yesterday), the spread's z-score over the past ``z_window``
days, and :func:`qr.research.pairs.pair_signals` turns the z-score into a position. The
position becomes weights on the two legs with the hedge ratio of the day it was opened.
See ``qr/research/pairs.py`` and the notebooks in ``notebooks/pairs/``.

Vectorized engine only.
"""

from __future__ import annotations

import pandas as pd

from qr.backtest.strategy import WeightStrategy, closes
from qr.research.pairs import (
    log_prices,
    pair_signals,
    pair_weights,
    rolling_hedge_ratio,
    rolling_spread_zscore,
)


class PairsWeights(WeightStrategy):
    """Long or short the spread ``y - beta * x``, with ``gross`` of equity in the two legs.

    Set ``beta`` to use a fixed hedge ratio instead of the rolling one. A fixed ratio
    fitted on the same data it is tested on uses hindsight: that is only for showing how
    much difference it makes.
    """

    def __init__(
        self,
        y: str,
        x: str,
        beta_window: int = 252,
        z_window: int = 60,
        entry: float = 2.0,
        exit: float = 0.5,
        stop: float = 4.0,
        gross: float = 1.0,
        beta: float | None = None,
    ) -> None:
        if y == x:
            raise ValueError("y and x must be different symbols")
        if not 0 <= exit < entry < stop:
            raise ValueError(f"need 0 <= exit < entry < stop, got {exit}, {entry}, {stop}")
        if beta_window < 2 or z_window < 2:
            raise ValueError("windows must be at least 2")
        self.y, self.x = y, x
        self.beta_window, self.z_window = beta_window, z_window
        self.entry, self.exit, self.stop = entry, exit, stop
        self.gross, self.beta = gross, beta

    def hedge_ratio(self, bars: pd.DataFrame) -> pd.Series:
        """The hedge ratio used on each day (known at that day's close)."""
        prices = log_prices(closes(bars)[[self.y, self.x]])
        if self.beta is not None:
            return pd.Series(self.beta, index=prices.index, name="beta")
        return rolling_hedge_ratio(prices[self.y], prices[self.x], self.beta_window)

    def zscore(self, bars: pd.DataFrame) -> pd.Series:
        """The spread's z-score on each day."""
        prices = log_prices(closes(bars)[[self.y, self.x]])
        beta = self.hedge_ratio(bars)
        return rolling_spread_zscore(prices[self.y], prices[self.x], beta, self.z_window)

    def signal(self, bars: pd.DataFrame) -> pd.Series:
        """The spread position decided at each day's close, in {-1, 0, +1}."""
        return pair_signals(self.zscore(bars), self.entry, self.exit, self.stop)

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pair_weights(
            self.signal(bars), self.hedge_ratio(bars), self.gross, names=(self.y, self.x)
        )
