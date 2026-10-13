"""The unified account format, shared by both engines now and by paper/live trading later.

:class:`AccountState` is a plain, serializable snapshot (inspired by QUANTAXIS's QIFI).
:class:`Account` is the ledger that produces it: it applies fills, splits, dividends, and
fees, and keeps this identity true at all times::

    equity = initial cash + realized_pnl + unrealized_pnl - fees_paid

Conventions: positions use the average-cost method; ``realized_pnl`` includes dividends
and excludes fees; ``fees_paid`` is commissions plus short borrow fees. Spread and
slippage are not fees: they are part of each trade's price.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any

from qr.backtest.events import BUY, Fill, Order, Side


@dataclass(slots=True)
class Position:
    """A holding in one symbol. ``quantity`` is negative for a short."""

    quantity: float
    avg_cost: float
    market_value: float


@dataclass(frozen=True, slots=True)
class Trade:
    """The immutable record of one fill."""

    id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    fees: float
    timestamp: datetime
    order_id: str
    realized_pnl: float = 0.0  # profit this trade locked in by closing shares, before fees


@dataclass
class AccountState:
    """A snapshot of an account. Round-trips through JSON without losing a bit."""

    timestamp: datetime
    cash: float
    positions: dict[str, Position]  # symbol -> Position(quantity, avg_cost, market_value)
    open_orders: dict[str, Order]
    equity: float  # cash + market value of positions
    gross_exposure: float
    net_exposure: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cash": self.cash,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "open_orders": {i: o.to_dict() for i, o in self.open_orders.items()},
            "equity": self.equity,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees_paid": self.fees_paid,
        }

    def to_json(self) -> str:
        """Serialize to JSON. Floats are written with full precision."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AccountState:
        """Inverse of :meth:`to_dict`."""
        return cls(
            timestamp=datetime.fromisoformat(raw["timestamp"]),
            cash=raw["cash"],
            positions={s: Position(**p) for s, p in raw["positions"].items()},
            open_orders={i: Order.from_dict(o) for i, o in raw["open_orders"].items()},
            equity=raw["equity"],
            gross_exposure=raw["gross_exposure"],
            net_exposure=raw["net_exposure"],
            realized_pnl=raw["realized_pnl"],
            unrealized_pnl=raw["unrealized_pnl"],
            fees_paid=raw["fees_paid"],
        )

    @classmethod
    def from_json(cls, text: str) -> AccountState:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))


@dataclass
class Account:
    """A cash account with positions, kept up to date fill by fill."""

    cash: float
    timestamp: datetime
    positions: dict[str, Position] = field(default_factory=dict)
    open_orders: dict[str, Order] = field(default_factory=dict)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    trade_count: int = 0

    # ------------------------------------------------------------------ derived values

    @property
    def market_value(self) -> float:
        """Signed market value of all positions."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def equity(self) -> float:
        """Cash plus the market value of positions."""
        return self.cash + self.market_value

    @property
    def gross_exposure(self) -> float:
        """Long plus short market value, both counted positive."""
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Market value minus cost of the open positions."""
        return sum(p.market_value - p.quantity * p.avg_cost for p in self.positions.values())

    def quantity(self, symbol: str) -> float:
        """Signed number of shares held in ``symbol`` (0.0 when flat)."""
        position = self.positions.get(symbol)
        return position.quantity if position else 0.0

    # ------------------------------------------------------------------ updates

    def apply_fill(self, fill: Fill) -> Trade:
        """Book a fill: move cash, update the position, and return the trade record."""
        signed = fill.quantity if fill.side == BUY else -fill.quantity
        position = self.positions.get(fill.symbol) or Position(0.0, 0.0, 0.0)
        held = position.quantity
        realized = 0.0
        if held == 0 or (held > 0) == (signed > 0):
            # Opening or adding: blend the new shares into the average cost.
            cost = abs(held) * position.avg_cost + fill.quantity * fill.price
            position.avg_cost = cost / (abs(held) + fill.quantity)
        else:
            closed = min(fill.quantity, abs(held))
            direction = 1.0 if held > 0 else -1.0
            realized = closed * (fill.price - position.avg_cost) * direction
            if fill.quantity > abs(held):  # went through zero: the rest is a new position
                position.avg_cost = fill.price
        position.quantity = held + signed
        position.market_value = position.quantity * fill.price
        if position.quantity == 0:
            self.positions.pop(fill.symbol, None)
        else:
            self.positions[fill.symbol] = position

        self.cash -= signed * fill.price + fill.commission
        self.fees_paid += fill.commission
        self.realized_pnl += realized
        self.timestamp = fill.timestamp
        self.trade_count += 1
        return Trade(
            id=f"T{self.trade_count:07d}",
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            fees=fill.commission,
            timestamp=fill.timestamp,
            order_id=fill.order_id,
            realized_pnl=realized,
        )

    def apply_split(self, symbol: str, ratio: float) -> None:
        """A ``ratio``-for-one split: more shares at a lower cost, same value.

        Fractional shares are kept as they are (a real broker pays cash in lieu).
        """
        if not ratio > 0:
            raise ValueError(f"invalid split ratio {ratio} for {symbol}")
        position = self.positions.get(symbol)
        if position:
            position.quantity *= ratio
            position.avg_cost /= ratio

    def apply_dividend(self, symbol: str, per_share: float) -> float:
        """Pay a cash dividend on the shares held; shorts pay it. Returns the cash moved."""
        amount = self.quantity(symbol) * per_share
        self.cash += amount
        self.realized_pnl += amount
        return amount

    def charge_fee(self, amount: float) -> None:
        """Take a fee (for example short borrow) out of cash."""
        self.cash -= amount
        self.fees_paid += amount

    def mark(self, prices: Mapping[str, float], timestamp: datetime) -> None:
        """Revalue positions at ``prices``; symbols without a price keep their last value."""
        for symbol, position in self.positions.items():
            price = prices.get(symbol)
            if price is not None:
                position.market_value = position.quantity * price
        self.timestamp = timestamp

    # ------------------------------------------------------------------ snapshots

    def state(self) -> AccountState:
        """An independent snapshot of the account as it is now."""
        return AccountState(
            timestamp=self.timestamp,
            cash=self.cash,
            positions={
                s: Position(p.quantity, p.avg_cost, p.market_value)
                for s, p in self.positions.items()
            },
            open_orders={i: replace(o) for i, o in self.open_orders.items()},
            equity=self.equity,
            gross_exposure=self.gross_exposure,
            net_exposure=self.market_value,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            fees_paid=self.fees_paid,
        )

    @classmethod
    def from_state(cls, state: AccountState, trade_count: int = 0) -> Account:
        """Rebuild a working account from a snapshot (for example one loaded from JSON)."""
        return cls(
            cash=state.cash,
            timestamp=state.timestamp,
            positions={
                s: Position(p.quantity, p.avg_cost, p.market_value)
                for s, p in state.positions.items()
            },
            open_orders={i: replace(o) for i, o in state.open_orders.items()},
            realized_pnl=state.realized_pnl,
            fees_paid=state.fees_paid,
            trade_count=trade_count,
        )
