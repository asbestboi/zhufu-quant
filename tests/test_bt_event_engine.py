"""The event-driven engine and its broker against hand-calculated cases."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from qr.backtest.account import AccountState, Trade
from qr.backtest.data import MarketData
from qr.backtest.events import Bar
from qr.backtest.result import EngineOutput
from qr.backtest.strategy import Context, EventStrategy

from .bt_helpers import FIVE_DAYS, bars, flat_bars, make_config, run

Action = Callable[[Context], Any]


class Script(EventStrategy):
    """Runs the given actions on the given session dates (at that session's close)."""

    def __init__(self, plan: dict[str, Action]) -> None:
        self.plan = {pd.Timestamp(day): action for day, action in plan.items()}
        self.fills: list[Trade] = []
        self.seen: list[tuple[pd.Timestamp, int]] = []

    def on_bar(self, ctx: Context, bars: dict[str, Bar]) -> None:
        self.seen.append((pd.Timestamp(ctx.now), len(ctx.history("AAA", 1000))))
        action = self.plan.get(pd.Timestamp(ctx.now))
        if action:
            action(ctx)

    def on_fill(self, ctx: Context, trade: Trade) -> None:
        self.fills.append(trade)


def five_days(**frames: pd.DataFrame) -> MarketData:
    return MarketData.from_frames({"AAA": bars(FIVE_DAYS), **frames})


def play(data: MarketData, plan: dict[str, Action], **config: Any) -> tuple[EngineOutput, Script]:
    script = Script(plan)
    return run(data, make_config(data, **config), script), script


def column(output: EngineOutput, name: str) -> list[float]:
    return [round(float(v), 6) for v in output.accounts[name]]


# --------------------------------------------------------------------------- hand-calculated


def test_five_day_series_with_known_trades() -> None:
    """Buy 50 at market, sell 20 at market, buy 10 on a limit; every number by hand.

    Costs: 0.01 a share with a 1.00 minimum, and 10 bp of half-spread on market orders.

    Jan 3  buy 50 at the 102 open: pays 102.102 a share (spread 5.10) plus 1.00.
           cash 10000 - 5105.10 - 1 = 4893.90; 50 shares at the 103 close = 5150.
    Jan 4  nothing fills; 50 shares at 105 = 5250.
    Jan 5  sell 20 at the 104 open: receives 103.896 a share (spread 2.08) minus 1.00.
           cash 4893.90 + 2077.92 - 1 = 6970.82; 30 shares at 101 = 3030.
           realized 20 * (103.896 - 102.102) = 35.88.
    Jan 8  limit buy 10 at 99: the bar opens at 100 and trades down to 98, so it fills at
           99.00 with no spread, plus 1.00. cash 6970.82 - 990 - 1 = 5979.82;
           40 shares at 99 = 3960, average cost (30 * 102.102 + 990) / 40 = 101.3265.
    """
    costs = {
        "commission_model": "per_share",
        "commission_per_share": 0.01,
        "commission_min_per_order": 1.0,
        "spread_bps": 10,
        "slippage_bps": 0,
    }
    output, script = play(
        five_days(),
        {
            "2024-01-02": lambda ctx: ctx.order("AAA", 50),
            "2024-01-04": lambda ctx: ctx.order("AAA", -20),
            "2024-01-05": lambda ctx: ctx.order_limit("AAA", 10, 99.0),
        },
        costs=costs,
    )
    assert column(output, "cash") == [10000.0, 4893.9, 4893.9, 6970.82, 5979.82]
    assert column(output, "equity") == [10000.0, 10043.9, 10143.9, 10000.82, 9939.82]
    assert column(output, "fees_paid") == [0.0, 1.0, 1.0, 2.0, 3.0]
    assert column(output, "realized_pnl") == [0.0, 0.0, 0.0, 35.88, 35.88]
    assert column(output, "unrealized_pnl") == [0.0, 44.9, 144.9, -33.06, -93.06]
    assert column(output, "gross_exposure") == [0.0, 5150.0, 5250.0, 3030.0, 3960.0]
    assert output.positions["AAA"].tolist() == [0.0, 5150.0, 5250.0, 3030.0, 3960.0]

    trades = output.trades
    assert trades["date"].dt.strftime("%m-%d").tolist() == ["01-03", "01-05", "01-08"]
    assert trades["side"].tolist() == ["buy", "sell", "buy"]
    assert trades["quantity"].tolist() == [50.0, 20.0, 10.0]
    assert trades["price"].round(6).tolist() == [102.102, 103.896, 99.0]
    assert trades["commission"].tolist() == [1.0, 1.0, 1.0]
    assert trades["spread"].round(6).tolist() == [5.1, 2.08, 0.0]
    assert output.costs["commission"].tolist() == [0.0, 1.0, 0.0, 1.0, 1.0]

    final = output.final_state
    assert final is not None
    assert final.positions["AAA"].quantity == 40
    assert final.positions["AAA"].avg_cost == pytest.approx(101.3265)
    assert AccountState.from_json(final.to_json()) == final
    assert [t.id for t in script.fills] == ["T0000001", "T0000002", "T0000003"]
    assert output.orders is not None and output.orders["status"].tolist() == ["filled"] * 3


def test_split_and_dividend_are_applied_like_a_broker_does(
    split_dividend_bars: pd.DataFrame, split_dividend_actions: pd.DataFrame
) -> None:
    """Opens 99, 101, 50.5, 51.5, 50.5; closes 100, 102, 51, 52, 50.

    Jan 3  buy 10 at the 101 open: cash 8990, equity 8990 + 10 * 102 = 10010.
    Jan 4  2:1 split: 20 shares with a cost of 50.50, value unchanged at 20 * 51.
           The resting sell limit of 5 at 110 becomes 10 at 55.
    Jan 5  20 * 52 = 1040, equity 10030.
    Jan 8  dividend 1.00 a share on 20 shares: cash 9010; 20 * 50 = 1000, equity 10010.
    """
    data = MarketData.from_frames({"AAA": split_dividend_bars}, {"AAA": split_dividend_actions})
    output, _ = play(
        data,
        {
            "2024-01-02": lambda ctx: ctx.order("AAA", 10),
            "2024-01-03": lambda ctx: ctx.order_limit("AAA", -5, 110.0),
        },
    )
    assert column(output, "cash") == [10000.0, 8990.0, 8990.0, 8990.0, 9010.0]
    assert column(output, "equity") == [10000.0, 10010.0, 10010.0, 10030.0, 10010.0]
    assert column(output, "realized_pnl") == [0.0, 0.0, 0.0, 0.0, 20.0]
    final = output.final_state
    assert final is not None
    assert final.positions["AAA"].quantity == 20
    assert final.positions["AAA"].avg_cost == 50.5
    (resting,) = final.open_orders.values()
    assert resting.quantity == 10 and resting.limit_price == 55.0


def test_a_strategy_that_never_trades_keeps_exactly_its_cash() -> None:
    output, _ = play(five_days(), {}, costs={})  # default costs, nothing to charge them on
    assert output.accounts["equity"].tolist() == [10_000.0] * 5
    assert output.accounts["cash"].tolist() == [10_000.0] * 5
    assert len(output.trades) == 0
    assert output.costs.to_numpy().sum() == 0.0


def test_identical_runs_give_identical_output() -> None:
    plan: dict[str, Action] = {
        "2024-01-02": lambda ctx: ctx.order_target_weight("AAA", 0.7),
        "2024-01-04": lambda ctx: ctx.order_target_weight("AAA", 0.2),
    }
    first, _ = play(five_days(), plan, costs={})
    second, _ = play(five_days(), plan, costs={})
    for name in ("accounts", "positions", "trades", "costs", "orders"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name), check_exact=True)
    assert first.final_state is not None and second.final_state is not None
    assert first.final_state.to_json() == second.final_state.to_json()


# --------------------------------------------------------------------------- no look-ahead


def test_orders_never_fill_on_the_bar_that_triggered_them() -> None:
    output, script = play(
        five_days(),
        {"2024-01-03": lambda ctx: ctx.order("AAA", 10)},
        execution="next_close",
    )
    # Decided on the Jan 3 close, filled at the Jan 4 close (105), never at Jan 3's.
    assert output.trades["date"].tolist() == [pd.Timestamp("2024-01-04")]
    assert output.trades["price"].tolist() == [105.0]
    # history() grows by one bar a day: the strategy never holds more than the past.
    assert [length for _, length in script.seen] == [1, 2, 3, 4, 5]


def test_history_is_adjusted_with_past_actions_only(
    split_dividend_bars: pd.DataFrame, split_dividend_actions: pd.DataFrame
) -> None:
    data = MarketData.from_frames({"AAA": split_dividend_bars}, {"AAA": split_dividend_actions})
    seen: dict[str, list[float]] = {}

    def look(ctx: Context) -> None:
        seen[f"{ctx.now:%m-%d}"] = ctx.history("AAA", 10).round(9).tolist()
        seen[f"{ctx.now:%m-%d} raw"] = ctx.history("AAA", 10, adjusted=False).tolist()

    play(data, {day: look for day in ("2024-01-03", "2024-01-05", "2024-01-08")})
    assert seen["01-03"] == [100.0, 102.0]  # the split has not happened yet
    assert seen["01-05"] == [50.0, 51.0, 51.0, 52.0]  # split known, dividend not yet
    factor = 51.0 / 52.0
    expected = [50 * factor, 51 * factor, 51 * factor, 52 * factor, 50.0]
    assert seen["01-08"] == [round(v, 9) for v in expected]
    assert seen["01-08 raw"] == [100.0, 102.0, 51.0, 52.0, 50.0]


# --------------------------------------------------------------------------- order rules


def test_limit_orders_need_the_price_to_trade_through() -> None:
    output, _ = play(
        five_days(),
        {
            "2024-01-02": lambda ctx: (
                ctx.order_limit("AAA", 10, 101.0),  # Jan 3 low is exactly 101: no fill
                ctx.order_limit("AAA", 10, 101.0, tif="day"),  # the same, but expires
                ctx.order_limit("AAA", 10, 103.5),  # Jan 3 opens below it: fills at the open
            ),
        },
    )
    assert output.orders is not None
    orders = output.orders.set_index("id")
    # The good-till-cancelled order fills on Jan 5, when the bar trades down to 100.
    assert orders["status"].tolist() == ["filled", "expired", "filled"]
    fills = output.trades.set_index("order_id")
    assert fills.loc["O0000003", "price"] == 102.0  # better than the 103.5 asked for
    assert fills.loc["O0000003", "date"] == pd.Timestamp("2024-01-03")
    assert fills.loc["O0000001", "price"] == 101.0
    assert fills.loc["O0000001", "date"] == pd.Timestamp("2024-01-05")


def test_sell_limit_and_cancel() -> None:
    def later(ctx: Context) -> None:
        ctx.order_limit("AAA", -5, 105.5)  # Jan 4 high is 106: fills at 105.5
        doomed = ctx.order_limit("AAA", -5, 200.0)
        assert doomed is not None
        ctx.cancel(doomed.id)

    output, _ = play(five_days(), {"2024-01-02": lambda ctx: ctx.order("AAA", 10),
                                   "2024-01-03": later})  # fmt: skip
    assert output.orders is not None
    assert output.orders["status"].tolist() == ["filled", "filled", "cancelled"]
    assert output.trades["price"].tolist() == [102.0, 105.5]


def test_long_only_and_leverage_limits_cut_orders_down() -> None:
    output, _ = play(
        five_days(),
        {
            # 10000 of cash buys 98 shares at the 102 open, not the 150 asked for.
            "2024-01-02": lambda ctx: ctx.order("AAA", 150),
            # Long-only: selling 500 only closes the 98 held.
            "2024-01-04": lambda ctx: ctx.order("AAA", -500),
            # Flat and long-only: a short sale is refused outright.
            "2024-01-05": lambda ctx: ctx.order("AAA", -10),
        },
        constraints={"long_only": True},
    )
    assert output.trades["quantity"].tolist() == [98.0, 98.0]
    assert output.orders is not None
    assert output.orders["status"].tolist() == ["partially_filled", "partially_filled", "rejected"]
    assert output.orders["reason"].tolist() == [
        "leverage limit",
        "long-only: cannot sell short",
        "long-only: cannot sell short",
    ]
    assert (output.accounts["cash"] >= 0).all()


def test_leverage_above_one_and_short_borrow_fee() -> None:
    data = MarketData.from_frames({"AAA": flat_bars([100.0] * 5)})
    output, _ = play(
        data,
        {"2024-01-02": lambda ctx: ctx.order("AAA", -50)},
        costs={"commission_model": "none", "spread_bps": 0, "slippage_bps": 0, "borrow_rate": 0.36},
    )
    # Short 5000 from Wednesday: 0.36 / 360 = 0.1% a calendar day, and Monday pays for 3.
    assert output.costs["borrow"].round(6).tolist() == [0.0, 5.0, 5.0, 5.0, 15.0]
    assert column(output, "equity") == [10000.0, 9995.0, 9990.0, 9985.0, 9970.0]
    assert column(output, "net_exposure") == [0.0, -5000.0, -5000.0, -5000.0, -5000.0]

    levered, _ = play(
        data,
        {"2024-01-02": lambda ctx: ctx.order_target_weight("AAA", 1.5)},
        constraints={"max_gross_leverage": 2.0},
    )
    assert levered.trades["quantity"].tolist() == [150.0]
    assert column(levered, "cash")[-1] == -5000.0


def test_participation_cap_partial_and_reject() -> None:
    data = five_days()  # 1000 shares a day at about 100: roughly 100,000 of dollar volume
    plan: dict[str, Action] = {"2024-01-03": lambda ctx: ctx.order("AAA", 50)}
    partial, _ = play(data, plan, costs={"max_participation": 0.02})
    # Average dollar volume before Jan 4 is (100 + 103) / 2 * 1000 = 101,500; 2% of it is
    # 2,030, which buys 19 whole shares at the 103 open.
    assert partial.trades["quantity"].tolist() == [19.0]
    assert partial.orders is not None
    assert partial.orders["reason"].tolist() == ["volume participation cap"]
    rejected, _ = play(
        data, plan, costs={"max_participation": 0.02, "participation_action": "reject"}
    )
    assert len(rejected.trades) == 0
    assert rejected.orders is not None and rejected.orders["status"].tolist() == ["rejected"]


def test_sells_are_matched_before_buys() -> None:
    data = MarketData.from_frames({"AAA": flat_bars([100.0] * 4), "BBB": flat_bars([50.0] * 4)})

    def switch(ctx: Context) -> None:
        ctx.order_target_weight("BBB", 1.0)  # submitted first, but needs the sale's cash
        ctx.order_target_weight("AAA", 0.0)

    output, _ = play(
        data,
        {"2024-01-02": lambda ctx: ctx.order_target_weight("AAA", 1.0), "2024-01-03": switch},
    )
    assert output.trades["symbol"].tolist() == ["AAA", "AAA", "BBB"]
    assert output.trades["side"].tolist() == ["buy", "sell", "buy"]
    assert output.trades["quantity"].tolist() == [100.0, 100.0, 200.0]


def test_order_validation_and_target_helpers() -> None:
    def check(ctx: Context) -> None:
        assert ctx.order("AAA", 0.4) is None  # rounds to zero whole shares
        with pytest.raises(ValueError, match="positive limit_price"):
            ctx.order("AAA", 5, "limit")
        with pytest.raises(KeyError, match="not in the backtest's universe"):
            ctx.order("ZZZ", 5)
        first = ctx.order_target_quantity("AAA", 30)
        second = ctx.order_target_quantity("AAA", 20)  # replaces the first, does not add
        assert first is not None and second is not None
        assert [o.id for o in ctx.open_orders("AAA")] == [second.id]
        assert ctx.equity == 10_000.0 and ctx.cash == 10_000.0 and ctx.position("AAA") == 0

    output, _ = play(five_days(), {"2024-01-02": check})
    assert output.trades["quantity"].tolist() == [20.0]


def test_fractional_shares_and_min_trade_weight() -> None:
    data = MarketData.from_frames({"AAA": flat_bars([300.0] * 4)})
    plan: dict[str, Action] = {"2024-01-02": lambda ctx: ctx.order_target_weight("AAA", 0.5)}
    whole, _ = play(data, plan)
    assert whole.trades["quantity"].tolist() == [16.0]  # 5000 / 300, rounded down
    fractional, _ = play(data, plan, whole_shares=False)
    assert fractional.trades["quantity"].tolist() == [pytest.approx(5000 / 300)]

    # A daily re-target of 2% more is inside a 5% no-trade band; an exit never is.
    nudges: dict[str, Action] = {
        "2024-01-02": lambda ctx: ctx.order_target_weight("AAA", 0.50),
        "2024-01-03": lambda ctx: ctx.order_target_weight("AAA", 0.52),
        "2024-01-04": lambda ctx: ctx.order_target_weight("AAA", 0.0),
    }
    banded, _ = play(data, nudges, whole_shares=False, constraints={"min_trade_weight": 0.05})
    assert banded.trades["side"].tolist() == ["buy", "sell"]
