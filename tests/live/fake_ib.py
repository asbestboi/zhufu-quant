"""A fake ``ib_async.IB`` client: just enough surface for IBKRBroker, no network, no
``ib_async`` import needed. Mirrors ``tests/test_sources_crypto.py``'s ``FakeExchange``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FakeAccountValue:
    tag: str
    currency: str
    value: str


@dataclass
class FakeContract:
    symbol: str
    secType: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"


@dataclass
class FakePosition:
    contract: FakeContract
    position: float
    averageCost: float
    marketValue: float


@dataclass
class FakeTicker:
    contract: FakeContract
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    close: float | None = None
    time: datetime | None = None

    def marketPrice(self) -> float:
        for candidate in (self.last, self.close):
            if candidate is not None:
                return candidate
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return float("nan")


@dataclass
class FakeOrderStatus:
    status: str = "Submitted"
    filled: float = 0.0


@dataclass
class FakeOrder:
    action: str
    totalQuantity: float
    lmtPrice: float
    tif: str = "DAY"
    orderId: int = 0


@dataclass
class FakeTrade:
    contract: FakeContract
    order: FakeOrder
    orderStatus: FakeOrderStatus = field(default_factory=FakeOrderStatus)


def fake_contract(symbol: str) -> FakeContract:
    return FakeContract(symbol)


def fake_order(action: str, quantity: float, price: float, tif: str) -> FakeOrder:
    return FakeOrder(action, quantity, price, tif)


class FakeIB:
    """A fake IB Gateway connection: one account, preset positions and quotes."""

    def __init__(
        self,
        account_id: str = "DU1234567",
        cash: float = 100_000.0,
        net_liquidation: float | None = None,
        positions: list[FakePosition] | None = None,
        quotes: dict[str, FakeTicker] | None = None,
        fill_immediately: bool = True,
    ) -> None:
        self.account_id = account_id
        self.cash = cash
        self.net_liquidation = net_liquidation if net_liquidation is not None else cash
        self.positions = positions or []
        self.quotes = quotes or {}
        self.fill_immediately = fill_immediately
        self.trades: list[FakeTrade] = []
        self._order_ids = itertools.count(1)
        self.cancelled_contracts: list[Any] = []

    def managedAccounts(self) -> list[str]:
        return [self.account_id]

    def accountSummary(self, account: str) -> list[FakeAccountValue]:
        return [
            FakeAccountValue("TotalCashValue", "BASE", str(self.cash)),
            FakeAccountValue("NetLiquidation", "BASE", str(self.net_liquidation)),
        ]

    def portfolio(self, account: str) -> list[FakePosition]:
        return self.positions

    def qualifyContracts(self, contract: FakeContract) -> list[FakeContract]:
        return [contract]

    def reqMktData(self, contract: FakeContract, generic: str, snapshot: bool) -> FakeTicker:
        return self.quotes.get(contract.symbol, FakeTicker(contract))

    def cancelMktData(self, contract: FakeContract) -> None:
        self.cancelled_contracts.append(contract)

    def sleep(self, seconds: float) -> None:
        pass

    def placeOrder(self, contract: FakeContract, order: FakeOrder) -> FakeTrade:
        order.orderId = next(self._order_ids)
        status = FakeOrderStatus(
            status="Filled" if self.fill_immediately else "Submitted",
            filled=order.totalQuantity if self.fill_immediately else 0.0,
        )
        trade = FakeTrade(contract=contract, order=order, orderStatus=status)
        self.trades.append(trade)
        return trade

    def cancelOrder(self, order: FakeOrder) -> None:
        for trade in self.trades:
            if trade.order is order:
                trade.orderStatus.status = "Cancelled"
