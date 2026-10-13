"""Small builders shared by the backtest tests: bars, configs, synthetic markets."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from qr.backtest.config import BacktestConfig
from qr.backtest.costs import CostModel
from qr.backtest.data import MarketData
from qr.backtest.event_engine import run_event
from qr.backtest.result import EngineOutput
from qr.backtest.strategies import build_strategy
from qr.backtest.strategy import EventStrategy, WeightStrategy
from qr.backtest.vectorized import run_vectorized

NO_COSTS: dict[str, Any] = {
    "commission_model": "none",
    "spread_bps": 0,
    "slippage_bps": 0,
    "borrow_rate": 0,
}

# Five sessions: (date, open, high, low, close). Used by the hand-calculated cases.
FIVE_DAYS = [
    ("2024-01-02", 100.0, 101.0, 99.0, 100.0),
    ("2024-01-03", 102.0, 104.0, 101.0, 103.0),
    ("2024-01-04", 103.0, 106.0, 102.0, 105.0),
    ("2024-01-05", 104.0, 105.0, 100.0, 101.0),
    ("2024-01-08", 100.0, 102.0, 98.0, 99.0),
]


def bars(
    rows: list[tuple[str, float, float, float, float]], volume: float = 1000.0
) -> pd.DataFrame:
    """A bars frame from ``(date, open, high, low, close)`` rows."""
    frame = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    return frame.assign(volume=volume)


def flat_bars(
    closes: list[float], start: str = "2024-01-02", volume: float = 1000.0
) -> pd.DataFrame:
    """Business-day bars where open, high, low, and close are all the given close."""
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {"date": dates, "open": closes, "high": closes, "low": closes, "close": closes}
    ).assign(volume=volume)


def make_config(data: MarketData, **overrides: Any) -> BacktestConfig:
    """A config covering all of ``data``; zero costs unless ``costs`` is given."""
    raw: dict[str, Any] = {
        "name": "test",
        "engine": "event",
        "strategy": "buy_and_hold",
        "universe": list(data.symbols),
        "start": data.dates[0].date(),
        "end": data.dates[-1].date(),
        "initial_cash": 10_000,
        "benchmark": None,
        "costs": NO_COSTS,
    }
    raw.update(overrides)
    return BacktestConfig.model_validate(raw)


def run(
    data: MarketData,
    config: BacktestConfig,
    strategy: WeightStrategy | EventStrategy | None = None,
) -> EngineOutput:
    """Run ``config`` on ``data`` with its configured costs."""
    costs = CostModel(config.costs, data.symbols, data.asset_classes)
    strategy = strategy or build_strategy(config.strategy, config.engine, config.params)
    if isinstance(strategy, WeightStrategy):
        return run_vectorized(strategy, data, config, costs)
    return run_event(strategy, data, config, costs)


def synthetic_bars(
    days: int,
    seed: int = 0,
    start: str = "2015-01-02",
    drift: float = 0.0004,
    volatility: float = 0.012,
    dividend: float = 0.0,
    dividend_every: int = 63,
    split_on: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A random-walk symbol as unadjusted bars plus its corporate actions.

    With ``dividend`` set the price drops by that amount every ``dividend_every`` sessions
    (the ex-dates); with ``split_on`` there is a 2:1 split on that session.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    close = np.empty(days)
    open_ = np.empty(days)
    actions: list[tuple[pd.Timestamp, str, float]] = []
    price = 100.0
    for i in range(days):
        if split_on is not None and i == split_on:
            price /= 2.0
            actions.append((dates[i], "split", 2.0))
        if dividend and i and i % dividend_every == 0:
            price -= dividend
            actions.append((dates[i], "dividend", dividend))
        open_[i] = price * np.exp(rng.normal(0.0, 0.004))
        price = price * np.exp(rng.normal(drift, volatility))
        close[i] = price
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": 1_000_000.0,
        }
    )
    return frame, pd.DataFrame(actions, columns=["date", "action", "value"])


def synthetic_market(symbols: int, days: int, seed: int = 0, **kwargs: Any) -> MarketData:
    """Several independent random-walk symbols named S00, S01, ..."""
    frames, actions = {}, {}
    for i in range(symbols):
        frames[f"S{i:02d}"], actions[f"S{i:02d}"] = synthetic_bars(days, seed + i, **kwargs)
    return MarketData.from_frames(frames, actions)


def day(text: str) -> date:
    return date.fromisoformat(text)
