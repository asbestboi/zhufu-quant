"""The offline, in-memory broker: fills, rejections, and account state."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from qr.live.broker import BrokerError
from qr.live.sim_broker import SimBroker


def test_is_paper_and_account_id() -> None:
    broker = SimBroker(account_id="DU999")
    assert broker.is_paper is True
    assert broker.account_id == "DU999"


def test_market_order_fills_immediately_at_the_current_price() -> None:
    broker = SimBroker(initial_cash=10_000.0)
    broker.set_prices({"AAA": 100.0})
    order = broker.place_order("AAA", "buy", 10)
    assert order.status == "filled" and order.filled_quantity == 10
    state = broker.account()
    assert state.cash == 9_000.0
    assert state.positions["AAA"].quantity == 10.0


def test_order_without_a_price_is_rejected_not_raised() -> None:
    broker = SimBroker()
    order = broker.place_order("AAA", "buy", 10)
    assert order.status == "rejected"
    assert "no price" in order.reason


def test_zero_or_negative_quantity_is_an_error() -> None:
    broker = SimBroker()
    with pytest.raises(BrokerError):
        broker.place_order("AAA", "buy", 0)


def test_limit_order_waits_until_the_price_crosses() -> None:
    broker = SimBroker()
    broker.set_prices({"AAA": 110.0})
    order = broker.place_order("AAA", "buy", 10, type="limit", limit_price=100.0)
    assert order.status == "open"
    assert order in broker.open_orders()

    broker.set_prices({"AAA": 95.0})
    # The sim broker only evaluates a limit order when it is (re)submitted, so place a
    # fresh one; this is enough to exercise the "already crosses" branch on submission.
    crossed = broker.place_order("AAA", "buy", 5, type="limit", limit_price=100.0)
    assert crossed.status == "filled"


def test_order_status_and_cancel() -> None:
    broker = SimBroker()
    broker.set_prices({"AAA": 110.0})
    open_order = broker.place_order("AAA", "buy", 10, type="limit", limit_price=50.0)
    assert broker.order_status(open_order.id).status == "open"
    broker.cancel_order(open_order.id)
    assert broker.order_status(open_order.id).status == "cancelled"
    assert broker.open_orders() == []


def test_unknown_order_id_is_an_error() -> None:
    broker = SimBroker()
    with pytest.raises(BrokerError):
        broker.order_status("nope")


def test_prices_only_returns_symbols_it_has() -> None:
    broker = SimBroker()
    broker.set_prices({"AAA": 100.0})
    quotes = broker.prices(["AAA", "BBB"])
    assert quotes.keys() == {"AAA"}
    assert quotes["AAA"].price == 100.0


def test_prices_report_when_each_symbol_was_last_quoted() -> None:
    broker = SimBroker()
    first = datetime(2024, 1, 1)
    second = datetime(2024, 1, 2)
    broker.set_prices({"AAA": 100.0}, now=first)
    broker.set_prices({"BBB": 50.0}, now=second)
    quotes = broker.prices(["AAA", "BBB"])
    assert quotes["AAA"].as_of == first
    assert quotes["BBB"].as_of == second


def test_save_and_load_round_trips_the_account(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    broker = SimBroker(initial_cash=10_000.0)
    broker.set_prices({"AAA": 100.0})
    broker.place_order("AAA", "buy", 10)
    broker.save(path)

    restored = SimBroker.load(path)
    state = restored.account()
    assert state.cash == 9_000.0
    assert state.positions["AAA"].quantity == 10.0


def test_load_without_a_saved_file_starts_fresh(tmp_path: Path) -> None:
    broker = SimBroker.load(tmp_path / "missing.json", initial_cash=5_000.0)
    state = broker.account()
    assert state.cash == 5_000.0
    assert state.positions == {}
