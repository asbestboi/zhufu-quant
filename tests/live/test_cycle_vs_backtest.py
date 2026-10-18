"""The big end-to-end check: three monthly live cycles land on the same positions as the
backtester, run once over the same three months.

Flat (constant) prices throughout remove any question of *which* price a fill uses --
the decision price, the planner's sizing price, and the backtester's next-session fill
price are all the same number, so a mismatch anywhere in qr.live's rounding or sizing
math would show up as a difference in whole shares, not get lost in price noise.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from qr import calendars
from qr.backtest.config import BacktestConfig, CostConfig
from qr.backtest.data import load_market_data
from qr.live.cycle import COMPLETED, run_cycle
from qr.live.sim_broker import SimBroker

from ..bt_helpers import NO_COSTS, run
from .live_config import make_live_config
from .live_store import seed_store

SYMBOLS = ["AAA", "BBB", "CCC"]
PRICES = {"AAA": 100.0, "BBB": 90.0, "CCC": 110.0}
LOOKBACK, SKIP, TOP_N = 10, 2, 2


def _flat_bars(start: date, end: date, price: float) -> pd.DataFrame:
    dates = calendars.sessions("etf", start, end)
    return pd.DataFrame(
        {
            "date": dates,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1_000_000.0,
        }
    )


def _month_ends(data_dir: Path) -> list[date]:
    sessions = calendars.sessions("etf", date(2023, 1, 1), date(2023, 3, 31))
    by_month = sessions.to_series().dt.to_period("M")
    return [sessions[by_month == m][-1].date() for m in by_month.unique()]


def test_three_monthly_cycles_match_the_backtest_within_whole_shares(tmp_path: Path) -> None:
    start, end = date(2022, 11, 1), date(2023, 4, 15)
    bars = {symbol: _flat_bars(start, end, price) for symbol, price in PRICES.items()}
    seed_store(tmp_path, bars)
    decision_dates = _month_ends(tmp_path)
    assert len(decision_dates) == 3

    params = {"lookback": LOOKBACK, "skip": SKIP, "top_n": TOP_N}
    config = make_live_config(
        SYMBOLS, tmp_path, strategy="xs_momentum", params=params, initial_cash=100_000.0
    )
    broker = SimBroker(initial_cash=config.initial_cash)
    for as_of in decision_dates:
        broker.set_prices(PRICES)
        result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
        assert result.status == COMPLETED, result.message

    live_positions = broker.account().positions
    assert live_positions  # not a vacuous all-zero run

    data = load_market_data(SYMBOLS, decision_dates[-1], data_dir=tmp_path)
    backtest_config = BacktestConfig(
        name="reference",
        engine="event",
        strategy="xs_momentum",
        params=params,
        universe=SYMBOLS,
        start=start,
        end=decision_dates[-1],
        initial_cash=config.initial_cash,
        benchmark=None,
        rebalance="monthly",
        costs=CostConfig(**NO_COSTS),
        cost_stress=[],
    )
    reference = run(data, backtest_config)
    assert reference.final_state is not None
    reference_positions = reference.final_state.positions

    for symbol in SYMBOLS:
        live_qty = live_positions[symbol].quantity if symbol in live_positions else 0.0
        ref_qty = reference_positions[symbol].quantity if symbol in reference_positions else 0.0
        assert live_qty == ref_qty, f"{symbol}: live={live_qty} backtest={ref_qty}"
