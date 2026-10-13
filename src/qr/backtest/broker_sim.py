"""The simulated broker: keeps the order book and turns orders into fills.

One session is matched against one daily bar, in this order:

1. **Market orders** at the open, sells before buys so that sales pay for purchases.
2. **Limit orders**. A buy at limit ``L`` fills at the open when the open is at or below
   ``L`` (a better price than asked for), otherwise at ``L`` when the low is *below*
   ``L``. A bar that only touches the limit does not fill it: with one bar a day there is
   no telling whether the order was reached in the queue. Sells mirror this.
3. **Market-on-close orders** at the close, sells before buys.

Market and market-on-close fills pay spread and slippage through a worse price. A limit
order resting at its price pays neither. Every fill pays commission.

Before filling, an order is cut down by, in turn: the long-only rule, the volume
participation cap, and the leverage limit (so a cash account cannot spend money it does
not have). Whatever is cut is cancelled; an order fills once or not at all.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime

from qr.backtest.account import Account, Trade
from qr.backtest.config import Constraints
from qr.backtest.costs import CostModel, TradeCosts
from qr.backtest.events import BUY, SELL, Bar, Fill, Order, OrderType, Side, TimeInForce

BISECTION_STEPS = 60


class SimBroker:
    """Order book and matching for one account."""

    def __init__(
        self,
        account: Account,
        costs: CostModel,
        constraints: Constraints,
        whole_shares: bool = True,
    ) -> None:
        self.account = account
        self.costs = costs
        self.constraints = constraints
        self.whole_shares = whole_shares
        self.orders: list[Order] = []  # every order ever submitted, in order

    # ------------------------------------------------------------------ order book

    def _round(self, quantity: float) -> float:
        # A hair of tolerance so 99.99999999999999 shares still counts as 100.
        return float(math.floor(quantity + 1e-9)) if self.whole_shares else quantity

    def submit(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        type: OrderType,
        now: datetime,
        limit_price: float | None = None,
        tif: TimeInForce = "gtc",
    ) -> Order | None:
        """Queue an order for the next session. None when it rounds to zero shares."""
        if type == "limit":
            if limit_price is None or not limit_price > 0:
                raise ValueError("a limit order needs a positive limit_price")
        elif limit_price is not None:
            raise ValueError(f"a {type} order takes no limit_price")
        if not math.isfinite(quantity):
            raise ValueError(f"order quantity must be finite, got {quantity}")
        quantity = self._round(quantity)
        if quantity <= 0:
            return None
        order = Order(
            id=f"O{len(self.orders) + 1:07d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            type=type,
            submitted_at=now,
            limit_price=limit_price,
            tif=tif,
        )
        self.orders.append(order)
        self.account.open_orders[order.id] = order
        return order

    def _close(self, order: Order, status: str, reason: str = "") -> None:
        order.status = status  # type: ignore[assignment]
        order.reason = reason
        self.account.open_orders.pop(order.id, None)

    def cancel(self, order_id: str) -> None:
        """Cancel an open order; anything else is ignored."""
        order = self.account.open_orders.get(order_id)
        if order is not None:
            self._close(order, "cancelled")

    def apply_split(self, symbol: str, ratio: float) -> None:
        """Restate open orders for a split, as a real broker does."""
        for order in self.account.open_orders.values():
            if order.symbol == symbol:
                order.quantity *= ratio
                if order.limit_price is not None:
                    order.limit_price /= ratio

    # ------------------------------------------------------------------ matching

    def run_session(
        self, bars: Mapping[str, Bar], adv_value: Mapping[str, float], now: datetime
    ) -> list[tuple[Fill, Trade]]:
        """Match the open orders against one session's bars and book the fills.

        Leaves the account marked at the session's close.
        """
        done: list[tuple[Fill, Trade]] = []
        self.account.mark({s: b.open for s, b in bars.items()}, now)
        waiting = [o for o in self.account.open_orders.values() if o.symbol in bars]

        def sells_first(type: OrderType) -> list[Order]:
            return sorted((o for o in waiting if o.type == type), key=lambda o: o.side != SELL)

        for order in sells_first("market"):
            self._execute(order, bars[order.symbol].open, False, adv_value, now, done)
        for order in (o for o in waiting if o.type == "limit"):
            bar = bars[order.symbol]
            limit = order.limit_price
            assert limit is not None
            if order.side == BUY:
                at_open, through = bar.open <= limit, bar.low < limit
            else:
                at_open, through = bar.open >= limit, bar.high > limit
            if at_open:
                self._execute(order, bar.open, False, adv_value, now, done)
            elif through:
                self._execute(order, limit, True, adv_value, now, done)
            elif order.tif == "day":
                self._close(order, "expired")
        self.account.mark({s: b.close for s, b in bars.items()}, now)
        for order in sells_first("moc"):
            self._execute(order, bars[order.symbol].close, False, adv_value, now, done)
        return done

    def _order_costs(
        self, order: Order, quantity: float, price: float, adv: float, passive: bool
    ) -> TradeCosts:
        costs = self.costs.order_costs(order.symbol, quantity, price, adv)
        return TradeCosts(costs.commission, 0.0, 0.0) if passive else costs

    def _execute(
        self,
        order: Order,
        price: float,
        passive: bool,
        adv_value: Mapping[str, float],
        now: datetime,
        done: list[tuple[Fill, Trade]],
    ) -> None:
        """Fill ``order`` at reference ``price`` as far as the limits allow."""
        adv = adv_value.get(order.symbol, math.nan)
        held = self.account.quantity(order.symbol)
        quantity, reason = order.quantity, ""

        if self.constraints.long_only and order.side == SELL and quantity > max(held, 0.0):
            quantity, reason = max(held, 0.0), "long-only: cannot sell short"
        allowed = self.costs.fillable_shares(order.symbol, quantity, price, adv)
        if allowed < quantity:
            quantity, reason = self._round(allowed), "volume participation cap"
        affordable = self._leverage_limit(order, quantity, held, price, adv, passive)
        if affordable < quantity:
            quantity, reason = affordable, "leverage limit"
        if quantity <= 0:
            self._close(order, "rejected", reason)
            return

        costs = self._order_costs(order, quantity, price, adv, passive)
        direction = 1.0 if order.side == BUY else -1.0
        fill_price = price + direction * (costs.spread + costs.slippage) / quantity
        if order.limit_price is not None:  # never worse than the limit that was asked for
            fill_price = (
                min(fill_price, order.limit_price)
                if order.side == BUY
                else max(fill_price, order.limit_price)
            )
        paid = abs(fill_price - price) * quantity
        modelled = costs.spread + costs.slippage
        share = paid / modelled if modelled > 0 else 0.0
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=fill_price,
            commission=costs.commission,
            timestamp=now,
            spread_cost=costs.spread * share,
            slippage_cost=costs.slippage * share,
        )
        order.filled_quantity = quantity
        self._close(order, "filled" if quantity >= order.quantity else "partially_filled", reason)
        done.append((fill, self.account.apply_fill(fill)))

    def _leverage_limit(
        self, order: Order, quantity: float, held: float, price: float, adv: float, passive: bool
    ) -> float:
        """The largest part of ``quantity`` that keeps gross exposure within the limit.

        Shares that reduce the position are always allowed, also when the account is
        already over the limit because prices moved.
        """
        direction = 1.0 if order.side == BUY else -1.0
        reducing = min(quantity, abs(held)) if held * direction < 0 else 0.0
        if quantity <= reducing:
            return quantity
        position = self.account.positions.get(order.symbol)
        own_value = position.market_value if position else 0.0
        other_value = self.account.market_value - own_value
        other_gross = self.account.gross_exposure - abs(own_value)
        cash = self.account.cash
        limit = self.constraints.max_gross_leverage

        def within(size: float) -> bool:
            costs = self._order_costs(order, size, price, adv, passive)
            after = held + direction * size
            equity = cash - direction * size * price - costs.total + other_value + after * price
            return bool(other_gross + abs(after) * price <= limit * equity * (1 + 1e-12))

        if within(quantity):
            return quantity
        low, high = reducing, quantity  # low is always allowed, high is not
        if self.whole_shares:
            low_n, high_n = math.floor(low), math.floor(high)
            while high_n - low_n > 1:
                middle = (low_n + high_n) // 2
                if within(float(middle)):
                    low_n = middle
                else:
                    high_n = middle
            return max(float(low_n), reducing) if within(float(low_n)) else reducing
        for _ in range(BISECTION_STEPS):
            half = (low + high) / 2
            if within(half):
                low = half
            else:
                high = half
        return low
