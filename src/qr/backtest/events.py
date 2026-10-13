"""Event types of the event-driven engine: bars, orders, fills, corporate actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "moc"]
TimeInForce = Literal["gtc", "day"]
OrderStatus = Literal["open", "filled", "partially_filled", "cancelled", "rejected", "expired"]

BUY: Side = "buy"
SELL: Side = "sell"


@dataclass(frozen=True, slots=True)
class Bar:
    """One symbol's daily bar, in unadjusted (as-traded) prices."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class Order:
    """An instruction to trade ``quantity`` (always positive) shares.

    ``market`` fills at the next bar's open, ``moc`` (market-on-close) at the next bar's
    close, and ``limit`` when the next bar trades through ``limit_price``. An order fills
    at most once: whatever a volume or leverage limit cuts off is cancelled, not queued.
    """

    id: str
    symbol: str
    side: Side
    quantity: float
    type: OrderType
    submitted_at: datetime
    limit_price: float | None = None
    tif: TimeInForce = "gtc"
    status: OrderStatus = "open"
    filled_quantity: float = 0.0
    reason: str = ""

    @property
    def signed_quantity(self) -> float:
        """Quantity with the side's sign: positive buys, negative sells."""
        return self.quantity if self.side == BUY else -self.quantity

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "type": self.type,
            "submitted_at": self.submitted_at.isoformat(),
            "limit_price": self.limit_price,
            "tif": self.tif,
            "status": self.status,
            "filled_quantity": self.filled_quantity,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Order:
        """Inverse of :meth:`to_dict`."""
        return cls(**{**raw, "submitted_at": datetime.fromisoformat(raw["submitted_at"])})


@dataclass(frozen=True, slots=True)
class Fill:
    """What the broker reports when an order executes.

    ``price`` already includes spread and slippage (a buy pays more, a sell receives
    less); ``spread_cost`` and ``slippage_cost`` say how many dollars of the trade's value
    that was. ``commission`` is charged on top.
    """

    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    spread_cost: float = 0.0
    slippage_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """A split (``value`` is the ratio) or cash dividend (``value`` is dollars per share)."""

    symbol: str
    timestamp: datetime
    action: Literal["split", "dividend"]
    value: float
