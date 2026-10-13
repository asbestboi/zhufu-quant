"""Cross-sectional momentum: hold the recent winners.

Symbols are ranked by their return from ``lookback`` sessions ago to ``skip`` sessions ago
(the usual "12-1 month" momentum is 252 and 21: the last month is skipped because very
recent returns tend to reverse). The ``top_n`` best get ``1 / top_n`` of the account each.
Ties go to the symbol listed first in the universe.
"""

from __future__ import annotations

import pandas as pd

from qr.backtest.events import Bar
from qr.backtest.strategy import Context, EventStrategy, WeightStrategy, closes, past_return


def _check(lookback: int, skip: int, top_n: int) -> None:
    if not 0 <= skip < lookback:
        raise ValueError(f"need 0 <= skip < lookback, got skip={skip}, lookback={lookback}")
    if top_n < 1:
        raise ValueError(f"top_n must be at least 1, got {top_n}")


class XsMomentumWeights(WeightStrategy):
    """``1 / top_n`` in each of the ``top_n`` symbols with the highest past return."""

    def __init__(self, lookback: int = 252, skip: int = 21, top_n: int = 3) -> None:
        _check(lookback, skip, top_n)
        self.lookback, self.skip, self.top_n = lookback, skip, top_n

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        momentum = past_return(closes(bars), self.lookback, self.skip)
        rank = momentum.rank(axis=1, ascending=False, method="first")
        return (rank <= self.top_n).astype("float64") / self.top_n


class XsMomentumEvents(EventStrategy):
    """The same ranking, recomputed on each rebalance date."""

    def __init__(self, lookback: int = 252, skip: int = 21, top_n: int = 3) -> None:
        _check(lookback, skip, top_n)
        self.lookback, self.skip, self.top_n = lookback, skip, top_n

    def on_bar(self, ctx: Context, bars: dict[str, Bar]) -> None:
        if not ctx.is_rebalance:
            return
        momentum: dict[str, float] = {}
        for symbol in ctx.symbols:
            history = ctx.history(symbol, self.lookback + 1)
            if len(history) == self.lookback + 1:
                momentum[symbol] = history.iloc[-1 - self.skip] / history.iloc[0] - 1.0
        # sorted() is stable, so equal returns keep the universe's order.
        winners = sorted(momentum, key=lambda s: -momentum[s])[: self.top_n]
        for symbol in ctx.symbols:
            if symbol in winners:
                ctx.order_target_weight(symbol, 1.0 / self.top_n)
            elif ctx.position(symbol):
                ctx.order_target_weight(symbol, 0.0)
