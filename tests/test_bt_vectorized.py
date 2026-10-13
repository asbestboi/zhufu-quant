"""The vectorized engine: hand-calculated cases, constraints, schedules, and speed."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qr import storage
from qr.backtest.calendar import rebalance_dates, rebalance_mask
from qr.backtest.config import Constraints
from qr.backtest.costs import CostModel
from qr.backtest.data import MarketData, load_market_data
from qr.backtest.strategies import build_strategy
from qr.backtest.strategy import WeightStrategy, closes
from qr.backtest.vectorized import LookAheadError, apply_constraints, run_vectorized
from qr.config import db_path
from qr.load import get_bars

from .bt_helpers import FIVE_DAYS, bars, flat_bars, make_config, run, synthetic_market
from .conftest import AFTER_JAN_16


class Fixed(WeightStrategy):
    """The same weights every day."""

    def __init__(self, **weights: float) -> None:
        self.weights = weights

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        prices = closes(bars)
        return pd.DataFrame(self.weights, index=prices.index)


class Table(WeightStrategy):
    """Weights given explicitly per date."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.table


def five_days() -> MarketData:
    return MarketData.from_frames({"AAA": bars(FIVE_DAYS)})


def vector(data: MarketData, **overrides: object):
    return make_config(data, engine="vectorized", **overrides)


# --------------------------------------------------------------------------- hand-calculated


def test_fully_invested_without_costs() -> None:
    """Decide on the Jan 2 close (100): 10000 / 100 = 100 shares. The Jan 3 open is 102,
    so 100 shares would cost 10200; with no leverage the order is cut to the 10000 there
    is. From then on equity follows the price: 10000 * close / 102.
    """
    data = five_days()
    output = run(data, vector(data), Fixed(AAA=1.0))
    expected = [10000.0, *(10000.0 * close / 102.0 for close in (103.0, 105.0, 101.0, 99.0))]
    np.testing.assert_allclose(output.accounts["equity"], expected, rtol=1e-12)
    np.testing.assert_allclose(output.accounts["cash"], [10000.0, 0, 0, 0, 0], atol=1e-8)
    assert len(output.trades) == 1  # later days want what is already held: no trade
    assert output.trades["value"].iloc[0] == pytest.approx(10000.0)
    assert output.trades["quantity"].iloc[0] == pytest.approx(10000.0 / 102.0)
    assert output.accounts["realized_pnl"].isna().all()  # no cost basis in this engine


def test_a_gap_down_leaves_cash_until_the_next_rebalance() -> None:
    """Decide at 100: 50 shares for half the account. They fill at the 98 open for 4900
    and close at 99: equity 5100 + 50 * 99 = 10050. Weights were targets, not promises.
    """
    data = MarketData.from_frames(
        {"AAA": bars([("2024-01-02", 100, 100, 100, 100), ("2024-01-03", 98, 99, 98, 99)])}
    )
    output = run(data, vector(data), Fixed(AAA=0.5))
    assert output.accounts["cash"].tolist() == [10000.0, 5100.0]
    assert output.accounts["equity"].tolist() == [10000.0, 10050.0]


def test_costs_come_out_of_the_account() -> None:
    """With a 1% commission and no leverage, x invested costs 0.01 x and both must fit in
    10000: x = 10000 / 1.01 = 9900.990099, commission 99.009901.
    """
    data = five_days()
    config = vector(data, costs={"commission_model": "percent", "commission_pct": 0.01,
                                 "spread_bps": 0, "slippage_bps": 0})  # fmt: skip
    output = run(data, config, Fixed(AAA=1.0))
    invested = 10000.0 / 1.01
    assert output.trades["value"].iloc[0] == pytest.approx(invested, rel=1e-12)
    assert output.trades["commission"].iloc[0] == pytest.approx(invested * 0.01, rel=1e-12)
    assert output.accounts["fees_paid"].iloc[-1] == pytest.approx(invested * 0.01, rel=1e-12)
    np.testing.assert_allclose(
        output.accounts["equity"].iloc[1:],
        [invested * close / 102.0 for close in (103.0, 105.0, 101.0, 99.0)],
        rtol=1e-12,
    )
    assert abs(output.accounts["cash"].iloc[-1]) < 1e-8
    assert len(output.trades) == 1


def test_spread_slippage_and_borrow_on_a_short() -> None:
    """Short half the account at a flat 100: 5000 sold, 3 bp of spread plus slippage is
    1.50; the 36% borrow rate costs 5000 * 0.001 a calendar day, three on the Monday.
    """
    data = MarketData.from_frames({"AAA": flat_bars([100.0] * 5)})
    costs = {"commission_model": "none", "spread_bps": 1, "slippage_bps": 2, "borrow_rate": 0.36}
    config = vector(data, costs=costs, rebalance=[date(2024, 1, 2)])  # one decision
    output = run(data, config, Fixed(AAA=-0.5))
    assert output.costs["spread"].round(9).tolist() == [0.0, 0.5, 0.0, 0.0, 0.0]
    assert output.costs["slippage"].round(9).tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert output.costs["borrow"].round(9).tolist() == [0.0, 5.0, 5.0, 5.0, 15.0]
    np.testing.assert_allclose(
        output.accounts["equity"], [10000.0, 9993.5, 9988.5, 9983.5, 9968.5], rtol=1e-12
    )
    assert output.accounts["net_exposure"].iloc[-1] == pytest.approx(-5000.0)
    assert output.trades["side"].tolist() == ["sell"]


def test_a_strategy_that_never_trades_keeps_exactly_its_cash() -> None:
    data = five_days()
    output = run(data, vector(data, costs={}), Fixed(AAA=0.0))
    assert output.accounts["equity"].tolist() == [10_000.0] * 5
    assert len(output.trades) == 0 and output.costs.to_numpy().sum() == 0.0


def test_next_close_execution_and_weight_changes() -> None:
    """Weights 1, then 0 from the Jan 4 decision: in at the Jan 3 close, out at Jan 5's."""
    data = five_days()
    table = pd.DataFrame({"AAA": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=data.dates)
    output = run(data, vector(data, execution="next_close"), Table(table))
    # 100 shares decided at 100 would cost 10300 at the Jan 3 close: cut to 10000.
    expected = [10000.0, 10000.0, 10000 * 105 / 103, 10000 * 101 / 103, 10000 * 101 / 103]
    np.testing.assert_allclose(output.accounts["equity"], expected, rtol=1e-12)
    assert output.trades["date"].dt.day.tolist() == [3, 5]
    assert output.trades["price"].tolist() == [103.0, 101.0]
    assert output.accounts["gross_exposure"].iloc[-1] == 0.0


# --------------------------------------------------------------------------- real store


@pytest.fixture
def store(
    data_dir: Path, split_dividend_bars: pd.DataFrame, split_dividend_actions: pd.DataFrame
) -> Path:
    with storage.connect(db_path(data_dir)) as con:
        storage.init_schema(con)
        storage.upsert_instrument(con, "AAA", "yahoo", "equity", True, AFTER_JAN_16)
        storage.upsert_bars(con, "AAA", "yahoo", split_dividend_bars, AFTER_JAN_16)
        storage.upsert_corporate_actions(con, "AAA", split_dividend_actions)
    return data_dir


def test_buy_and_hold_matches_the_total_return_series(store: Path) -> None:
    """Zero costs, bought at the Jan 3 close: equity is the total-return close, rescaled."""
    data = load_market_data(["AAA"], date(2024, 1, 8), data_dir=store)
    assert data.asset_classes == {"AAA": "equity"}
    config = vector(data, strategy="buy_and_hold", execution="next_close")
    output = run(data, config)
    total_return = get_bars("AAA", data_dir=store)["close"].droplevel("symbol")
    held = output.equity.iloc[1:]  # in the market from the Jan 3 close
    np.testing.assert_allclose(
        held / held.iloc[0], total_return.iloc[1:] / total_return.iloc[1], rtol=1e-12
    )
    # Through a 2:1 split and a dividend, without either showing up as a jump.
    assert held.iloc[-1] / held.iloc[0] == pytest.approx(50.0 / (51.0 * 51.0 / 52.0))


# --------------------------------------------------------------------------- constraints


def test_apply_constraints() -> None:
    weights = np.array([[0.8, -0.4, 0.2], [0.3, 0.3, 0.1], [2.0, 2.0, 0.0]])
    long_only = apply_constraints(weights, Constraints(long_only=True))
    np.testing.assert_allclose(long_only[0], [0.8, 0.0, 0.2])
    capped = apply_constraints(weights, Constraints(max_weight=0.25, max_gross_leverage=2.0))
    np.testing.assert_allclose(capped[0], [0.25, -0.25, 0.2])
    scaled = apply_constraints(weights, Constraints())  # gross leverage at most 1
    np.testing.assert_allclose(np.abs(scaled).sum(axis=1), [1.0, 0.7, 1.0])
    np.testing.assert_allclose(scaled[0], weights[0] / 1.4)
    np.testing.assert_allclose(scaled[1], weights[1])  # already within the limit


def test_no_weight_before_a_symbol_has_data() -> None:
    late = flat_bars([50.0, 50.0, 50.0], start="2024-01-04")
    data = MarketData.from_frames({"AAA": flat_bars([100.0] * 5), "NEW": late})
    output = run(data, vector(data), Fixed(AAA=0.5, NEW=0.5))
    # NEW first prints on Jan 4, is decided on that close, and is bought on Jan 5.
    assert output.positions["NEW"].tolist() == [0.0, 0.0, 0.0, 5000.0, 5000.0]
    assert output.positions["AAA"].tolist() == [0.0, 5000.0, 5000.0, 5000.0, 5000.0]
    assert output.weights is not None
    assert output.weights["NEW"].tolist() == [0.0, 0.0, 0.5, 0.5, 0.5]


def test_bad_weights_are_refused() -> None:
    data = five_days()
    with pytest.raises(ValueError, match="outside the universe: ZZZ"):
        run(data, vector(data), Fixed(ZZZ=1.0))
    with pytest.raises(ValueError, match="must be finite"):
        run(data, vector(data), Fixed(AAA=float("inf")))
    with pytest.raises(ValueError, match="no sessions between"):
        run(data, vector(data, start=date(2025, 1, 1), end=date(2025, 2, 1)), Fixed(AAA=1.0))


# --------------------------------------------------------------------------- schedules


def test_rebalance_schedules() -> None:
    sessions = pd.bdate_range("2024-01-02", "2024-03-15")
    assert rebalance_mask(sessions, "daily").all()
    monthly = rebalance_dates(sessions, "monthly")
    # The first session, then the last session of each finished month.
    assert [str(d.date()) for d in monthly] == ["2024-01-02", "2024-01-31", "2024-02-29"]
    weekly = rebalance_dates(sessions[:11], "weekly")
    assert [str(d.date()) for d in weekly] == ["2024-01-02", "2024-01-05", "2024-01-12"]
    # A Saturday moves to the next session; a date after the data is dropped.
    custom = rebalance_dates(sessions, [date(2024, 1, 13), date(2024, 2, 1), date(2030, 1, 1)])
    assert [str(d.date()) for d in custom] == ["2024-01-15", "2024-02-01"]
    with pytest.raises(ValueError, match="unknown rebalance schedule"):
        rebalance_mask(sessions, "hourly")


def test_weights_drift_between_rebalances() -> None:
    rising = flat_bars([100.0, 100.0, 110.0, 121.0, 133.1])
    data = MarketData.from_frames({"UP": rising, "FLAT": flat_bars([100.0] * 5)})
    config = vector(data, rebalance=[date(2024, 1, 2)])  # one decision, then hold
    output = run(data, config, Fixed(UP=0.5, FLAT=0.5))
    assert len(output.trades) == 2
    shares = output.positions.iloc[-1] / output.accounts["equity"].iloc[-1]
    assert shares["UP"] == pytest.approx(6655 / 11655)  # drifted up from one half
    daily = run(data, vector(data), Fixed(UP=0.5, FLAT=0.5))
    assert len(daily.trades) > 2  # rebalanced back towards one half every day


# --------------------------------------------------------------------------- look-ahead guard


class Peeker(WeightStrategy):
    """Cheats: holds the symbol only on days before it rises."""

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        prices = closes(bars)
        return (prices.shift(-1) > prices).astype("float64")


def test_a_strategy_that_uses_tomorrow_is_caught() -> None:
    data = synthetic_market(1, 60)
    with pytest.raises(LookAheadError, match="Peeker looks ahead"):
        run(data, vector(data), Peeker())
    # With the check switched off the cheat runs, and looks wonderful.
    cheated = run(data, vector(data, lookahead_checks=0), Peeker())
    honest = run(data, vector(data, strategy="buy_and_hold"))
    assert cheated.equity.iloc[-1] > honest.equity.iloc[-1]


# --------------------------------------------------------------------------- speed


def test_ten_years_of_fifty_symbols_in_under_two_seconds() -> None:
    data = synthetic_market(50, 2520, start="2012-01-03")
    config = vector(
        data,
        strategy="xs_momentum",
        params={"top_n": 10},
        rebalance="monthly",
        costs={},  # the default cost model
        initial_cash=1_000_000,
    )
    data.panel("close", adjusted=True)  # loading and pivoting the data is not the engine
    started = time.perf_counter()
    for free in (False, True):  # a full run is one pass with costs and one without
        strategy = build_strategy("xs_momentum", "vectorized", config.params)
        assert isinstance(strategy, WeightStrategy)
        output = run_vectorized(
            strategy, data, config, CostModel(config.costs, data.symbols, free=free)
        )
    elapsed = time.perf_counter() - started
    assert len(output.equity) == 2520
    assert elapsed < 2.0, f"took {elapsed:.2f}s"
