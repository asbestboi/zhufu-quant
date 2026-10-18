"""Planner: rounding, the no-trade band, the cash buffer, and sells-before-buys."""

from __future__ import annotations

import pytest

from qr.live.planner import plan_orders

from .live_helpers import make_account


def test_whole_share_rounding_truncates_towards_zero() -> None:
    account = make_account(cash=10_000.0)
    plan = plan_orders({"AAA": 1.0}, account, {"AAA": 333.0}, cash_buffer=0.0, min_trade_weight=0.0)
    # 10_000 / 333 = 30.03...: truncated down, not rounded to 30 by chance alone.
    assert plan.orders == [plan.orders[0]]
    order = plan.orders[0]
    assert order.symbol == "AAA" and order.side == "buy" and order.quantity == 30.0


def test_cash_buffer_shrinks_the_investable_amount() -> None:
    account = make_account(cash=10_000.0)
    plan = plan_orders(
        {"AAA": 1.0}, account, {"AAA": 100.0}, cash_buffer=0.10, min_trade_weight=0.0
    )
    # Only 90% of equity is investable: 9_000 / 100 = 90 shares, not 100.
    assert plan.orders[0].quantity == 90.0


def test_cash_buffer_must_be_a_valid_fraction() -> None:
    account = make_account(cash=10_000.0)
    with pytest.raises(ValueError, match="cash_buffer"):
        plan_orders({"AAA": 1.0}, account, {"AAA": 100.0}, cash_buffer=1.0)


def test_trade_below_min_trade_weight_is_skipped() -> None:
    # Already close to target: the small top-up trade is below the no-trade band.
    account = make_account(cash=100.0, positions={"AAA": (99.0, 100.0)})
    plan = plan_orders(
        {"AAA": 1.0}, account, {"AAA": 100.0}, cash_buffer=0.0, min_trade_weight=0.02
    )
    assert plan.orders == []
    assert "AAA" in plan.skipped


def test_closing_a_position_to_flat_always_trades_even_if_small() -> None:
    # A tiny residual position: target weight 0 must still close it out.
    account = make_account(cash=9_999.0, positions={"AAA": (1.0, 100.0)})
    plan = plan_orders({}, account, {"AAA": 100.0}, cash_buffer=0.0, min_trade_weight=0.5)
    assert len(plan.orders) == 1
    order = plan.orders[0]
    assert order.symbol == "AAA" and order.side == "sell" and order.quantity == 1.0


def test_no_price_for_a_wanted_symbol_is_skipped_not_an_error() -> None:
    account = make_account(cash=10_000.0)
    plan = plan_orders({"AAA": 0.5, "BBB": 0.5}, account, {"AAA": 100.0}, cash_buffer=0.0)
    assert [o.symbol for o in plan.orders] == ["AAA"]
    assert "BBB" in plan.skipped


def test_sells_come_before_buys() -> None:
    account = make_account(cash=0.0, positions={"AAA": (50.0, 100.0)})
    plan = plan_orders(
        {"AAA": 0.0, "BBB": 1.0},
        account,
        {"AAA": 100.0, "BBB": 50.0},
        cash_buffer=0.0,
        min_trade_weight=0.0,
    )
    assert [o.side for o in plan.orders] == ["sell", "buy"]
    assert [o.symbol for o in plan.orders] == ["AAA", "BBB"]


def test_no_change_needed_produces_no_order() -> None:
    account = make_account(cash=0.0, positions={"AAA": (100.0, 100.0)})
    plan = plan_orders({"AAA": 1.0}, account, {"AAA": 100.0}, cash_buffer=0.0, min_trade_weight=0.0)
    assert plan.orders == []
    assert plan.skipped == {}
