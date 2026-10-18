"""IBKRBroker, entirely offline: a FakeIB stands in for ib_async's IB client, and
``contract_factory``/``order_factory`` stand in for its Stock/LimitOrder classes -- so
none of this needs ib_async installed.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from qr.live.broker import BrokerError
from qr.live.ibkr import IBKRBroker, IBKRError
from tests.live.fake_ib import (
    FakeIB,
    FakePosition,
    FakeTicker,
    fake_contract,
    fake_order,
)

OPEN_DAY = datetime(2024, 1, 16)  # a real NYSE session; its open is 14:30 UTC
BEFORE_OPEN = datetime(2024, 1, 16, 14, 0, 0)
WITHIN_GRACE = datetime(2024, 1, 16, 14, 40, 0)
AFTER_GRACE = datetime(2024, 1, 16, 15, 0, 0)


class _FixedNow:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now


def make_broker(
    fake: FakeIB | None = None,
    *,
    port: int = 4002,
    limit_buffer: float = 0.003,
) -> tuple[IBKRBroker, FakeIB]:
    fake = fake if fake is not None else FakeIB()
    broker = IBKRBroker(
        port=port,
        limit_buffer=limit_buffer,
        ib=fake,
        contract_factory=fake_contract,
        order_factory=fake_order,
    )
    return broker, fake


def freeze(monkeypatch: pytest.MonkeyPatch, now: datetime) -> None:
    monkeypatch.setattr("qr.live.ibkr.datetime", _FixedNow(now))


# ------------------------------------------------------------------------- is_paper


def test_is_paper_true_for_a_paper_account_on_a_paper_port() -> None:
    broker, _ = make_broker(FakeIB(account_id="DU1234567"), port=4002)
    assert broker.is_paper is True
    assert broker.account_id == "DU1234567"


def test_is_paper_false_for_a_live_account_id_even_on_a_paper_port() -> None:
    broker, _ = make_broker(FakeIB(account_id="U1234567"), port=4002)
    assert broker.is_paper is False


def test_is_paper_false_for_a_live_port_even_with_a_du_account() -> None:
    broker, _ = make_broker(FakeIB(account_id="DU1234567"), port=4001)
    assert broker.is_paper is False


def test_no_managed_accounts_raises() -> None:
    fake = FakeIB()
    fake.managedAccounts = lambda: []  # type: ignore[method-assign]
    with pytest.raises(IBKRError, match="no managed accounts"):
        make_broker(fake)


# ------------------------------------------------------------------------- account


def test_account_maps_cash_equity_and_positions() -> None:
    fake = FakeIB(
        cash=5_000.0,
        net_liquidation=15_000.0,
        positions=[FakePosition(fake_contract("AAA"), 100.0, 90.0, 10_000.0)],
    )
    broker, _ = make_broker(fake)
    account = broker.account()
    assert account.cash == 5_000.0
    assert account.equity == 15_000.0
    assert account.positions["AAA"].quantity == 100.0
    assert account.positions["AAA"].market_value == 10_000.0
    assert account.gross_exposure == 10_000.0
    assert account.net_exposure == 10_000.0
    assert account.unrealized_pnl == pytest.approx(10_000.0 - 100.0 * 90.0)


def test_account_handles_an_empty_portfolio() -> None:
    broker, _ = make_broker(FakeIB(cash=1_000.0, net_liquidation=1_000.0))
    account = broker.account()
    assert account.positions == {}
    assert account.gross_exposure == 0.0
    assert account.unrealized_pnl == 0.0


def test_account_skips_zero_quantity_portfolio_rows() -> None:
    fake = FakeIB(positions=[FakePosition(fake_contract("AAA"), 0.0, 0.0, 0.0)])
    broker, _ = make_broker(fake)
    assert broker.account().positions == {}


# ------------------------------------------------------------------------- prices


def test_prices_filters_out_symbols_with_no_quote() -> None:
    fake = FakeIB(quotes={"AAA": FakeTicker(fake_contract("AAA"), last=150.0, time=OPEN_DAY)})
    broker, _ = make_broker(fake)
    quotes = broker.prices(["AAA", "BBB"])
    assert set(quotes) == {"AAA"}
    assert quotes["AAA"].price == 150.0
    assert quotes["AAA"].as_of == OPEN_DAY


def test_prices_uses_now_when_ticker_has_no_time(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_now = datetime(2024, 1, 16, 16, 0, 0)
    freeze(monkeypatch, fixed_now)
    fake = FakeIB(quotes={"AAA": FakeTicker(fake_contract("AAA"), last=150.0, time=None)})
    broker, _ = make_broker(fake)
    assert broker.prices(["AAA"])["AAA"].as_of == fixed_now


# ------------------------------------------------------------------------- place_order


def test_place_order_buy_prices_from_the_mid_plus_the_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze(monkeypatch, AFTER_GRACE)
    fake = FakeIB(quotes={"AAA": FakeTicker(fake_contract("AAA"), bid=99.9, ask=100.1)})
    broker, _ = make_broker(fake, limit_buffer=0.003)
    order = broker.place_order("AAA", "buy", 10.0)
    assert order.limit_price == pytest.approx(100.3)
    assert order.status == "filled"  # FakeIB fills immediately by default
    assert order.filled_quantity == 10.0


def test_place_order_sell_prices_from_the_mid_minus_the_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze(monkeypatch, AFTER_GRACE)
    fake = FakeIB(quotes={"AAA": FakeTicker(fake_contract("AAA"), bid=99.9, ask=100.1)})
    broker, _ = make_broker(fake, limit_buffer=0.003)
    order = broker.place_order("AAA", "sell", 10.0)
    assert order.limit_price == pytest.approx(99.7)


def test_place_order_rejects_before_the_market_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze(monkeypatch, BEFORE_OPEN)
    broker, _ = make_broker()
    with pytest.raises(IBKRError, match="has not opened"):
        broker.place_order("AAA", "buy", 10.0, limit_price=100.0)


def test_place_order_rejects_within_the_grace_period_after_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze(monkeypatch, WITHIN_GRACE)
    broker, _ = make_broker()
    with pytest.raises(IBKRError, match="waiting for"):
        broker.place_order("AAA", "buy", 10.0, limit_price=100.0)


def test_place_order_succeeds_once_past_the_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze(monkeypatch, AFTER_GRACE)
    broker, _ = make_broker()
    order = broker.place_order("AAA", "buy", 10.0, limit_price=100.0)
    assert order.symbol == "AAA"
    assert order.limit_price == 100.0


def test_place_order_rejects_non_positive_quantity(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze(monkeypatch, AFTER_GRACE)
    broker, _ = make_broker()
    with pytest.raises(BrokerError, match="positive"):
        broker.place_order("AAA", "buy", 0.0, limit_price=100.0)


def test_place_order_rejects_a_non_limit_order_type(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze(monkeypatch, AFTER_GRACE)
    broker, _ = make_broker()
    with pytest.raises(IBKRError, match="only places day limit orders"):
        broker.place_order("AAA", "buy", 10.0, type="market", limit_price=100.0)


def test_place_order_raises_when_no_quote_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze(monkeypatch, AFTER_GRACE)
    broker, _ = make_broker()  # no quotes configured -> FakeTicker with everything None
    with pytest.raises(IBKRError, match="no quote available"):
        broker.place_order("AAA", "buy", 10.0)


# ------------------------------------------------------------------------- order lifecycle


def test_cancel_order_cancels_a_non_terminal_order(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze(monkeypatch, AFTER_GRACE)
    broker, _fake = make_broker(FakeIB(fill_immediately=False))
    order = broker.place_order("AAA", "buy", 10.0, limit_price=100.0)
    broker.cancel_order(order.id)
    assert broker.order_status(order.id).status == "cancelled"


def test_cancel_order_ignores_an_already_terminal_order(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze(monkeypatch, AFTER_GRACE)
    broker, _fake = make_broker()  # fills immediately -> already terminal
    order = broker.place_order("AAA", "buy", 10.0, limit_price=100.0)
    broker.cancel_order(order.id)  # must not raise, must not call ib.cancelOrder
    assert broker.order_status(order.id).status == "filled"


def test_order_status_unknown_id_raises() -> None:
    broker, _ = make_broker()
    with pytest.raises(BrokerError, match="unknown order id"):
        broker.order_status("nope")


def test_open_orders_excludes_terminal_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze(monkeypatch, AFTER_GRACE)
    broker, _ = make_broker(FakeIB(fill_immediately=False))
    open_order = broker.place_order("AAA", "buy", 10.0, limit_price=100.0)
    filled = broker.place_order("BBB", "sell", 5.0, limit_price=50.0)
    broker.cancel_order(filled.id)  # BBB now cancelled -> terminal
    remaining = broker.open_orders()
    assert [o.id for o in remaining] == [open_order.id]
