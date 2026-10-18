"""An offline, in-memory broker: no network, immediate fills, zero costs.

For development and the test suite only: it stands in for a real broker so the rest of
``qr.live`` (planner, safety, journal, the cycle) can be built and tested without IB
Gateway. Market orders fill immediately at the caller's current price feed (set with
:meth:`SimBroker.set_prices`); there is no partial fill, slippage, or commission.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from qr.backtest.account import Account, AccountState
from qr.backtest.events import BUY, Fill, Order, OrderType, Side, TimeInForce
from qr.live.broker import Broker, BrokerError, PriceQuote


class SimBroker(Broker):
    """A simulated paper account, kept entirely in memory."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        account_id: str = "DU_SIM",
        now: datetime | None = None,
    ) -> None:
        self._account = Account(cash=initial_cash, timestamp=now or datetime.now())
        self._account_id = account_id
        self._prices: dict[str, float] = {}
        self._price_as_of: dict[str, datetime] = {}
        self._orders: dict[str, Order] = {}
        self._ids = itertools.count(1)

    @property
    def is_paper(self) -> bool:
        return True

    @property
    def account_id(self) -> str:
        return self._account_id

    def set_prices(self, prices: dict[str, float], now: datetime | None = None) -> None:
        """Replace the current price feed used to fill orders and mark positions.

        ``now`` is when these prices were quoted (default: this call's wall-clock time),
        recorded per symbol so :meth:`prices` can report a genuine, per-symbol age.
        """
        stamp = now or datetime.now()
        self._prices.update(prices)
        self._price_as_of.update({symbol: stamp for symbol in prices})
        if now is not None:
            self._account.mark(self._prices, now)

    def account(self) -> AccountState:
        self._account.mark(self._prices, self._account.timestamp)
        return self._account.state()

    def prices(self, symbols: Sequence[str]) -> dict[str, PriceQuote]:
        return {
            s: PriceQuote(price=self._prices[s], as_of=self._price_as_of[s])
            for s in symbols
            if s in self._prices
        }

    def place_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        type: OrderType = "market",
        limit_price: float | None = None,
        tif: TimeInForce = "day",
    ) -> Order:
        if quantity <= 0:
            raise BrokerError(f"order quantity must be positive, got {quantity}")
        order_id = f"SIM{next(self._ids):06d}"
        now = self._account.timestamp
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            type=type,
            submitted_at=now,
            limit_price=limit_price,
            tif=tif,
        )
        price = self._prices.get(symbol)
        if price is None:
            order.status = "rejected"
            order.reason = f"no price available for {symbol}"
            self._orders[order_id] = order
            return order
        if type == "limit" and limit_price is not None:
            crosses = price <= limit_price if side == BUY else price >= limit_price
            if not crosses:
                order.status = "open"
                self._orders[order_id] = order
                return order
        fill = Fill(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            commission=0.0,
            timestamp=now,
        )
        self._account.apply_fill(fill)
        order.status = "filled"
        order.filled_quantity = quantity
        self._orders[order_id] = order
        return order

    def cancel_order(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order is not None and order.status == "open":
            order.status = "cancelled"

    def order_status(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise BrokerError(f"unknown order id: {order_id}")
        return order

    def open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status == "open"]

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        """Write the account to ``path`` as JSON, so the next CLI invocation can resume it.

        Pending orders are not persisted: the planner only ever places market orders,
        which this broker always resolves (filled or rejected) before returning.
        """
        path.write_text(self.account().to_json())

    @classmethod
    def load(
        cls, path: Path, *, initial_cash: float = 100_000.0, account_id: str = "DU_SIM"
    ) -> SimBroker:
        """A fresh broker, or one restored from a file written by :meth:`save`."""
        broker = cls(initial_cash=initial_cash, account_id=account_id)
        if path.exists():
            state = AccountState.from_json(path.read_text())
            broker._account = Account(
                cash=state.cash,
                timestamp=state.timestamp,
                positions=state.positions,
                realized_pnl=state.realized_pnl,
                fees_paid=state.fees_paid,
            )
        return broker
