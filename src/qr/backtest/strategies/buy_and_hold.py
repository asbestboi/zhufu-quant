"""Buy and hold: one symbol, or an equal-weight basket.

With one symbol this is the simplest possible strategy and the reference the engines are
checked against. With several, each symbol that has started trading gets the same weight,
restored on every rebalance date.
"""

from __future__ import annotations

import pandas as pd

from qr.backtest.events import Bar
from qr.backtest.strategy import Context, EventStrategy, WeightStrategy, closes


class BuyAndHoldWeights(WeightStrategy):
    """Equal weight in every symbol that has had a bar so far."""

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        listed = closes(bars).notna().cummax()
        return listed.div(listed.sum(axis=1), axis=0)


class BuyAndHoldEvents(EventStrategy):
    """Equal weight in every symbol that has had a bar so far, set on rebalance dates.

    Dividends arrive as cash, so each rebalance also reinvests them.
    """

    def __init__(self) -> None:
        self._listed: set[str] = set()

    def on_bar(self, ctx: Context, bars: dict[str, Bar]) -> None:
        self._listed.update(bars)
        if not ctx.is_rebalance:
            return
        listed = [s for s in ctx.symbols if s in self._listed]
        for symbol in listed:
            ctx.order_target_weight(symbol, 1.0 / len(listed))
