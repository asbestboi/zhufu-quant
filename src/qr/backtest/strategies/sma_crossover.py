"""Moving-average crossover: hold a symbol while its fast average is above its slow one.

Long or flat. Each symbol gets an equal slice of the account (``1 / number of symbols``)
that is either invested or left in cash, so one symbol (the classic 50/200 on SPY) is
simply all-in or all-out.
"""

from __future__ import annotations

import pandas as pd

from qr.backtest.events import Bar
from qr.backtest.strategy import Context, EventStrategy, WeightStrategy, closes, sma


def _check(fast: int, slow: int) -> None:
    if not 0 < fast < slow:
        raise ValueError(f"need 0 < fast < slow, got fast={fast}, slow={slow}")


class SmaCrossoverWeights(WeightStrategy):
    """``1 / n`` in each symbol whose ``fast``-day average close is above its ``slow``-day one."""

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        _check(fast, slow)
        self.fast, self.slow = fast, slow

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        prices = closes(bars)

        def is_long(column: pd.Series) -> pd.Series:
            # Averages run over the symbol's own bars; a day without a bar keeps the signal.
            own = column.dropna()
            signal = (sma(own, self.fast) > sma(own, self.slow)).astype("float64")
            return signal.reindex(column.index).ffill().fillna(0.0)

        return prices.apply(is_long) / prices.shape[1]


class SmaCrossoverEvents(EventStrategy):
    """The same rule, bar by bar."""

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        _check(fast, slow)
        self.fast, self.slow = fast, slow

    def on_bar(self, ctx: Context, bars: dict[str, Bar]) -> None:
        if not ctx.is_rebalance:
            return
        for symbol in ctx.symbols:
            prices = ctx.history(symbol, self.slow).to_numpy()
            if len(prices) < self.slow:
                continue  # still warming up: stay flat
            is_long = prices[-self.fast :].mean() > prices.mean()
            ctx.order_target_weight(symbol, 1.0 / len(ctx.symbols) if is_long else 0.0)
