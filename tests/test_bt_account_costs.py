"""The account ledger and the cost model, each against hand-calculated values."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from qr.backtest.account import Account, AccountState
from qr.backtest.config import CostConfig
from qr.backtest.costs import CostModel
from qr.backtest.events import Fill, Order

T0 = datetime(2024, 1, 2)
T1 = datetime(2024, 1, 3)


def fill(side: str, quantity: float, price: float, commission: float = 0.0) -> Fill:
    return Fill("O1", "AAA", side, quantity, price, commission, T1)  # type: ignore[arg-type]


def model(
    names: list[str] | None = None, classes: dict[str, str] | None = None, /, **costs: object
) -> CostModel:
    return CostModel(CostConfig.model_validate(costs), names or ["AAA"], classes)


# --------------------------------------------------------------------------- account


def test_buy_add_and_partial_sell_use_average_cost() -> None:
    account = Account(cash=10_000.0, timestamp=T0)
    account.apply_fill(fill("buy", 10, 100.0, commission=1.0))
    account.apply_fill(fill("buy", 10, 110.0, commission=1.0))
    position = account.positions["AAA"]
    assert position.quantity == 20
    assert position.avg_cost == 105.0
    assert account.cash == 10_000 - 1000 - 1100 - 2

    trade = account.apply_fill(fill("sell", 5, 120.0, commission=1.0))
    assert trade.realized_pnl == 5 * (120.0 - 105.0)
    assert trade.id == "T0000003" and trade.fees == 1.0 and trade.order_id == "O1"
    assert account.positions["AAA"].quantity == 15
    assert account.positions["AAA"].avg_cost == 105.0  # selling does not change the cost
    assert account.realized_pnl == 75.0
    assert account.fees_paid == 3.0
    assert account.unrealized_pnl == 15 * (120.0 - 105.0)
    assert account.equity == 10_000 + 75 + 225 - 3


def test_selling_through_zero_opens_a_short_at_the_fill_price() -> None:
    account = Account(cash=10_000.0, timestamp=T0)
    account.apply_fill(fill("buy", 10, 100.0))
    trade = account.apply_fill(fill("sell", 25, 90.0))
    assert trade.realized_pnl == 10 * (90.0 - 100.0)  # only the 10 closed shares
    position = account.positions["AAA"]
    assert position.quantity == -15 and position.avg_cost == 90.0
    assert position.market_value == -1350.0
    assert account.gross_exposure == 1350.0
    assert account.cash == 10_000 - 1000 + 2250

    account.mark({"AAA": 80.0}, T1)
    assert account.unrealized_pnl == 150.0  # short 15 shares, price fell by 10
    account.apply_fill(fill("buy", 15, 80.0))
    assert "AAA" not in account.positions
    assert account.realized_pnl == -100.0 + 150.0
    assert account.equity == account.cash == 10_050.0


def test_split_dividend_and_fee() -> None:
    account = Account(cash=1_000.0, timestamp=T0)
    account.apply_fill(fill("buy", 10, 50.0))
    account.apply_split("AAA", 2.0)
    assert account.positions["AAA"].quantity == 20
    assert account.positions["AAA"].avg_cost == 25.0
    assert account.positions["AAA"].market_value == 500.0  # a split changes no value
    assert account.apply_dividend("AAA", 0.5) == 10.0
    assert account.cash == 510.0 and account.realized_pnl == 10.0
    account.charge_fee(2.5)
    assert account.cash == 507.5 and account.fees_paid == 2.5
    assert account.apply_dividend("ZZZ", 1.0) == 0.0  # nothing held, nothing paid
    with pytest.raises(ValueError, match="invalid split ratio"):
        account.apply_split("AAA", 0.0)


def test_a_short_position_pays_the_dividend() -> None:
    account = Account(cash=1_000.0, timestamp=T0)
    account.apply_fill(fill("sell", 10, 50.0))
    assert account.apply_dividend("AAA", 0.5) == -5.0
    assert account.cash == 1_000 + 500 - 5


def test_equity_identity_holds_after_every_step() -> None:
    account = Account(cash=5_000.0, timestamp=T0)
    steps = [("buy", 7, 101.3, 0.35), ("sell", 3, 99.1, 0.35), ("sell", 9, 104.7, 1.0)]
    for side, quantity, price, commission in steps:
        account.apply_fill(fill(side, quantity, price, commission))
        account.mark({"AAA": price + 0.7}, T1)
        expected = 5_000 + account.realized_pnl + account.unrealized_pnl - account.fees_paid
        assert account.equity == pytest.approx(expected, abs=1e-9)


def test_state_round_trips_through_json_exactly() -> None:
    account = Account(cash=0.1 + 0.2, timestamp=T0)  # 0.30000000000000004
    account.apply_fill(fill("buy", 3, 1 / 3, commission=1e-9))
    account.open_orders["O7"] = Order("O7", "AAA", "sell", 2.0, "limit", T0, limit_price=1.1)
    state = account.state()

    restored = AccountState.from_json(state.to_json())
    assert restored == state
    assert restored.to_json() == state.to_json()
    assert restored.cash == account.cash  # bit for bit, not approximately

    rebuilt = Account.from_state(restored, trade_count=account.trade_count)
    assert rebuilt.state() == state
    # A snapshot is independent of the account it came from.
    account.apply_fill(fill("buy", 1, 2.0))
    assert state.positions["AAA"].quantity == 3


# --------------------------------------------------------------------------- commissions


def test_ibkr_tiered_commission() -> None:
    costs = model()  # the default preset
    assert costs.order_costs("AAA", 100, 50.0).commission == pytest.approx(0.35)  # at the minimum
    assert costs.order_costs("AAA", 1000, 50.0).commission == pytest.approx(3.5)  # 0.0035 a share
    # 20,000 shares at 0.10: per-share is 70.00, capped at 1% of the 2,000 value.
    assert costs.order_costs("AAA", 20_000, 0.10).commission == pytest.approx(20.0)
    # When the minimum exceeds the 1% cap, the minimum wins (the pessimistic reading).
    assert costs.order_costs("AAA", 10, 0.20).commission == pytest.approx(0.35)
    assert costs.order_costs("AAA", 0, 50.0).total == 0.0


def test_other_commission_models_and_overridden_numbers() -> None:
    assert model(commission_model="ibkr_fixed").order_costs("AAA", 100, 50.0).commission == 1.0
    assert model(commission_model="ibkr_fixed").order_costs("AAA", 1000, 50.0).commission == 5.0
    assert model(commission_model="per_share").order_costs("AAA", 10, 50.0).commission == 0.05
    percent = model(commission_model="percent", commission_pct=0.002)
    assert percent.order_costs("AAA", 100, 50.0).commission == pytest.approx(10.0)
    assert model(commission_model="none").order_costs("AAA", 100, 50.0).commission == 0.0
    custom = model(commission_per_share=0.01, commission_min_per_order=2.0)
    assert custom.order_costs("AAA", 100, 50.0).commission == 2.0
    assert custom.order_costs("AAA", 1000, 50.0).commission == 10.0


def test_spread_slippage_and_volume_impact() -> None:
    costs = model(spread_bps=1, slippage_bps=2)
    order = costs.order_costs("AAA", 100, 50.0)  # 5,000 traded
    assert order.spread == pytest.approx(0.5)
    assert order.slippage == pytest.approx(1.0)
    assert order.total == pytest.approx(0.35 + 0.5 + 1.0)

    impact = model(slippage_bps=2, impact_bps=100)
    # 5,000 is 10% of the 50,000 average daily volume: 10 extra basis points.
    assert impact.order_costs("AAA", 100, 50.0, 50_000.0).slippage == pytest.approx(6.0)
    root = model(slippage_bps=0, impact_bps=100, impact_exponent=0.5)
    assert root.order_costs("AAA", 100, 50.0, 500_000.0).slippage == pytest.approx(5.0)


def test_borrow_fee() -> None:
    costs = model(borrow_rate=0.036)
    assert costs.borrow_fee("AAA", 10_000.0, 10) == pytest.approx(10.0)  # 3.6% * 10/360
    assert costs.borrow_fee("AAA", -10_000.0, 1) == pytest.approx(1.0)
    np.testing.assert_allclose(costs.borrow_fees(np.array([10_000.0]), 3), [3.0])


def test_participation_cap() -> None:
    partial = model(max_participation=0.1)
    assert partial.fillable_shares("AAA", 160, 50.0, 50_000.0) == pytest.approx(100.0)
    assert partial.fillable_shares("AAA", 80, 50.0, 50_000.0) == 80
    assert partial.fillable_shares("AAA", 80, 50.0) == 0.0  # no volume history: no trade
    reject = model(max_participation=0.1, participation_action="reject")
    assert reject.fillable_shares("AAA", 160, 50.0, 50_000.0) == 0.0
    assert model().fillable_shares("AAA", 1e9, 50.0) == 1e9  # no cap configured

    values = partial.fillable_value(np.array([-8000.0]), np.array([50_000.0]))
    np.testing.assert_allclose(values, [-5000.0])


def test_asset_class_and_symbol_overrides() -> None:
    costs = model(
        ["SPY", "BTC", "TLT"],
        {"SPY": "etf", "BTC": "crypto", "TLT": "etf"},
        spread_bps=1,
        asset_classes={"crypto": {"commission_model": "percent", "commission_pct": 0.0026}},
        symbols={"TLT": {"spread_bps": 0.5, "commission_min_per_order": 1.0}},
    )
    assert costs.order_costs("SPY", 100, 50.0).commission == pytest.approx(0.35)
    assert costs.order_costs("BTC", 0.5, 60_000.0).commission == pytest.approx(78.0)
    assert costs.order_costs("BTC", 0.5, 60_000.0).spread == pytest.approx(3.0)  # inherited
    assert costs.order_costs("TLT", 100, 100.0).spread == pytest.approx(0.5)
    assert costs.order_costs("TLT", 100, 100.0).commission == 1.0  # tiered, minimum raised


def test_array_and_single_order_costs_agree() -> None:
    symbols = [f"S{i}" for i in range(6)]
    costs = model(
        symbols,
        slippage_bps=3,
        impact_bps=40,
        max_participation=0.05,
        symbols={"S1": {"commission_model": "percent"}, "S2": {"participation_action": "reject"}},
    )
    rng = np.random.default_rng(1)
    shares = rng.uniform(-5000, 5000, 6)
    shares[3] = 0.0
    price = rng.uniform(1, 300, 6)
    adv = rng.uniform(1e5, 1e6, 6)
    adv[4] = np.nan
    together = costs.trade_costs(shares, price, adv)
    for i, symbol in enumerate(symbols):
        alone = costs.order_costs(symbol, shares[i], price[i], adv[i])
        assert alone.commission == together.commission[i]
        assert alone.spread == together.spread[i]
        assert alone.slippage == together.slippage[i]
    allowed = costs.fillable_value(shares * price, adv)
    for i, symbol in enumerate(symbols):
        alone_shares = costs.fillable_shares(symbol, abs(shares[i]), price[i], adv[i])
        assert alone_shares * price[i] == pytest.approx(abs(allowed[i]))


def test_free_model_costs_nothing_but_keeps_the_cap() -> None:
    config = CostConfig.model_validate({"max_participation": 0.1, "borrow_rate": 0.05})
    free = CostModel(config, ["AAA"], free=True)
    assert free.order_costs("AAA", 100, 50.0, 50_000.0).total == 0.0
    assert free.borrow_fee("AAA", 10_000.0, 30) == 0.0
    assert free.fillable_shares("AAA", 160, 50.0, 50_000.0) == pytest.approx(100.0)
