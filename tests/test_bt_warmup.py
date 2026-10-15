"""Lookback warm-up: no symbol is traded before it has the strategy's ``warmup_bars``.

The engines already refuse weights on a symbol before its first bar. A strategy that
declares ``warmup_bars`` also gets nothing traded until a symbol has that many bars, so an
ETF that lists mid-backtest joins only once its signal is computable, on either engine.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

from qr.backtest.data import MarketData
from qr.backtest.events import Bar
from qr.backtest.strategy import Context, EventStrategy, WeightStrategy

from .bt_helpers import flat_bars, make_config, run

pytestmark = pytest.mark.usefixtures("engine_core")

# AAA trades Jan 2 to Jan 9; NEW lists on Jan 4.
DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]


def market() -> MarketData:
    late = flat_bars([50.0] * 4, start="2024-01-04")
    return MarketData.from_frames({"AAA": flat_bars([100.0] * 6), "NEW": late})


class HalfEachWeights(WeightStrategy):
    """Wants half the account in each symbol on every date, NaN-free."""

    def __init__(self, warmup: int) -> None:
        self.warmup_bars = warmup

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        dates = bars.index.get_level_values("date").unique()
        return pd.DataFrame(0.5, index=dates, columns=["AAA", "NEW"])


class HalfEachEvents(EventStrategy):
    def __init__(self, warmup: int) -> None:
        self.warmup_bars = warmup
        self.ready: list[tuple[str, str]] = []

    def on_bar(self, ctx: Context, bars: Mapping[str, Bar]) -> None:
        for symbol in ctx.symbols:
            if ctx.warmed_up(symbol):
                self.ready.append((f"{ctx.now:%Y-%m-%d}", symbol))
            ctx.order_target_weight(symbol, 0.5)


@pytest.mark.parametrize("engine", ["vectorized", "event"])
def test_without_warmup_a_symbol_trades_after_its_first_bar(engine: str) -> None:
    data = market()
    strategy = HalfEachWeights(0) if engine == "vectorized" else HalfEachEvents(0)
    output = run(data, make_config(data, engine=engine), strategy)
    # NEW first prints on Jan 4, is decided on that close, and is bought on Jan 5.
    assert output.positions["NEW"].tolist() == [0.0, 0.0, 0.0, 5000.0, 5000.0, 5000.0]


@pytest.mark.parametrize("engine", ["vectorized", "event"])
def test_warmup_delays_each_symbol_until_it_has_enough_bars(engine: str) -> None:
    data = market()
    strategy = HalfEachWeights(2) if engine == "vectorized" else HalfEachEvents(2)
    output = run(data, make_config(data, engine=engine), strategy)
    assert list(output.positions.index.strftime("%Y-%m-%d")) == DATES
    # AAA's second bar is Jan 3 (bought Jan 4); NEW's is Jan 5 (bought Jan 8).
    assert output.positions["AAA"].tolist() == [0.0, 0.0, 5000.0, 5000.0, 5000.0, 5000.0]
    assert output.positions["NEW"].tolist() == [0.0, 0.0, 0.0, 0.0, 5000.0, 5000.0]
    if isinstance(strategy, HalfEachEvents):
        first_ready = {s: d for d, s in reversed(strategy.ready)}
        assert first_ready == {"AAA": "2024-01-03", "NEW": "2024-01-05"}


@pytest.mark.parametrize("engine", ["vectorized", "event"])
def test_warmup_longer_than_the_history_never_trades(engine: str) -> None:
    data = market()
    strategy = HalfEachWeights(10) if engine == "vectorized" else HalfEachEvents(10)
    output = run(data, make_config(data, engine=engine), strategy)
    assert (output.positions == 0).all().all()
    assert output.trades.empty


class SellsDuringWarmup(EventStrategy):
    """Buys AAA on day one with no warm-up, then raises it: reducing must still work."""

    def __init__(self) -> None:
        self.calls = 0

    def on_bar(self, ctx: Context, bars: Mapping[str, Bar]) -> None:
        self.calls += 1
        if self.calls == 1:
            ctx.order_target_weight("AAA", 0.5)
        elif self.calls == 3:
            self.warmup_bars = 99
            ctx._warmup_bars = 99  # as if the strategy had declared it all along
            assert ctx.order_target_weight("AAA", 0.8) is None  # growing: dropped
            assert ctx.order_target_weight("AAA", 0.0) is not None  # closing: sent


def test_reducing_a_position_is_never_blocked_by_warmup() -> None:
    data = market()
    output = run(data, make_config(data, engine="event"), SellsDuringWarmup())
    assert output.positions["AAA"].tolist() == [0.0, 5000.0, 5000.0, 0.0, 0.0, 0.0]
