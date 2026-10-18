"""The broker interface: everything ``qr.live`` needs from an account-holding broker.

:class:`SimBroker` (``sim_broker.py``) implements this offline, for development and
tests. ``ibkr.py`` (paper only) implements it against Interactive Brokers.

``is_paper`` is how :mod:`qr.live.safety` enforces the paper-only guard: it reads this
property on the real broker object, which each adapter must determine correctly from
the account itself (for IBKR: the account id and the port it connected on), not from
anything a config file can set. There is deliberately no config flag that could stand
in for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from qr.backtest.account import AccountState
from qr.backtest.events import Order, OrderType, Side, TimeInForce


class BrokerError(Exception):
    """Raised when a broker operation cannot be carried out."""


@dataclass(frozen=True)
class PriceQuote:
    """A price and when the broker last updated it, for the staleness check in safety.py.

    A broker that cannot report this honestly (most can) should set ``as_of`` to the
    earliest time it could possibly be wrong, never to "now" as a default -- that would
    make the staleness check pass vacuously.
    """

    price: float
    as_of: datetime


class Broker(ABC):
    """What ``qr.live`` needs from a broker: account, prices, and orders."""

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Whether this broker connection is to a paper (simulated-money) account."""

    @property
    @abstractmethod
    def account_id(self) -> str:
        """The broker's identifier for this account, for the journal and reports."""

    @abstractmethod
    def account(self) -> AccountState:
        """A snapshot of cash, positions, and open orders."""

    @abstractmethod
    def prices(self, symbols: Sequence[str]) -> dict[str, PriceQuote]:
        """The latest known price of each symbol that has one, and when it was quoted.

        A symbol with no available price is left out, never given as 0 or NaN.
        """

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        type: OrderType = "market",
        limit_price: float | None = None,
        tif: TimeInForce = "day",
    ) -> Order:
        """Submit an order for ``quantity`` shares (always positive; ``side`` sets sign)."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order. Unknown or already-finished orders are ignored."""

    @abstractmethod
    def order_status(self, order_id: str) -> Order:
        """The current state of a previously placed order."""

    @abstractmethod
    def open_orders(self) -> list[Order]:
        """Every order not yet filled, cancelled, rejected, or expired."""
