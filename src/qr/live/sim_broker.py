"""An offline, in-memory broker: no network, zero costs, and (by default) immediate fills.

For development and the test suite only: it stands in for a real broker so the rest of
``qr.live`` (planner, safety, journal, the cycle) can be built and tested without IB
Gateway. By default, orders fill immediately and fully at the caller's current price
feed (set with :meth:`SimBroker.set_prices`). ``fill_behavior`` and ``fill_delay_ticks``
switch that off, so the order-lifecycle poller (:mod:`qr.live.cycle`) can be exercised
against a broker that fills slowly, partially, or never, exactly like a real one can:

* ``"full"`` (default), ``fill_delay_ticks=0``: instant full fill, as before.
* ``"full"``, ``fill_delay_ticks=N``: stays ``open`` for ``N`` calls to :meth:`tick`,
  then fills fully.
* ``"partial"``: stays ``open`` for ``fill_delay_ticks`` ticks, then fills
  ``partial_fraction`` of the quantity and settles at ``partially_filled`` forever
  (never completes on its own -- only :meth:`cancel_order` ends it).
* ``"never"``: stays ``open`` forever; only :meth:`cancel_order` ends it.

:meth:`tick` is the only thing that advances a delayed or partial fill -- polling
:meth:`order_status` never does, so a test controls exactly when a fill "happens" by
calling it (typically from a fake ``sleep``).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from qr.backtest.account import Account, AccountState
from qr.backtest.events import BUY, Fill, Order, OrderType, Side, TimeInForce
from qr.live.broker import Broker, BrokerError, PriceQuote

FillBehavior = Literal["full", "partial", "never"]

_OPEN_STATUSES = ("open", "partially_filled")


class SimBroker(Broker):
    """A simulated paper account, kept entirely in memory."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        account_id: str = "DU_SIM",
        now: datetime | None = None,
        fill_behavior: FillBehavior = "full",
        fill_delay_ticks: int = 0,
        partial_fraction: float = 0.5,
    ) -> None:
        self._account = Account(cash=initial_cash, timestamp=now or datetime.now())
        self._account_id = account_id
        self._prices: dict[str, float] = {}
        self._price_as_of: dict[str, datetime] = {}
        self._orders: dict[str, Order] = {}
        self._ids = itertools.count(1)
        self._fill_behavior = fill_behavior
        self._fill_delay_ticks = fill_delay_ticks
        self._partial_fraction = partial_fraction
        self._pending: dict[str, int] = {}  # order_id -> ticks remaining

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
        if self._fill_behavior == "never":
            order.status = "open"
            self._orders[order_id] = order
            return order
        if self._fill_behavior == "partial" or self._fill_delay_ticks > 0:
            order.status = "open"
            self._orders[order_id] = order
            self._pending[order_id] = self._fill_delay_ticks
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

    def tick(self) -> None:
        """Advance simulated time by one step, resolving any delayed pending fills.

        Never called by :meth:`order_status` or :meth:`place_order` -- only this
        resolves a delayed or partial fill, so a test (or a fake ``sleep``) controls
        exactly when it happens.
        """
        for order_id in list(self._pending):
            remaining = self._pending[order_id] - 1
            if remaining > 0:
                self._pending[order_id] = remaining
                continue
            del self._pending[order_id]
            order = self._orders[order_id]
            price = self._prices.get(order.symbol, order.limit_price)
            if price is None:
                continue
            partial = self._fill_behavior == "partial"
            quantity = order.quantity * self._partial_fraction if partial else order.quantity
            fill = Fill(
                order_id=order_id, symbol=order.symbol, side=order.side, quantity=quantity,
                price=price, commission=0.0, timestamp=self._account.timestamp,
            )  # fmt: skip
            self._account.apply_fill(fill)
            order.filled_quantity = quantity
            order.status = "partially_filled" if partial else "filled"

    def cancel_order(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order is not None and order.status in _OPEN_STATUSES:
            order.status = "cancelled"
            self._pending.pop(order_id, None)

    def order_status(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise BrokerError(f"unknown order id: {order_id}")
        return order

    def open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status in _OPEN_STATUSES]

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        """Write the account to ``path`` as JSON, so the next CLI invocation can resume it.

        Open orders are not persisted -- only cash, positions, and realized P&L. With the
        default ``fill_behavior="full"``, ``fill_delay_ticks=0``, that's moot: every order
        resolves (filled or rejected) before this is ever reached. But with a delayed,
        partial, or ``"never"`` fill behavior, an order still open when the process exits
        is gone on the next load; unlike a real broker, there is no live IB Gateway for
        :mod:`qr.live.cycle`'s resumption logic to re-query across a restart. That is a
        real limitation of this in-memory broker, not something :meth:`load` works around.
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
