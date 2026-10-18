"""Interactive Brokers adapter, via ``ib_async`` (optional: ``uv sync --extra live``).

Connects to IB Gateway or TWS in **paper mode only**. ``is_paper`` is computed from the
connected account's id (Interactive Brokers always starts a paper account's id with
``DU``) and the port this adapter connected on (the well-known paper ports, never a
live one) -- never from a config flag, matching the paper-only guard's whole point
(:mod:`qr.live.safety`). If either check fails, every order is refused.

No credentials are ever read, held, or logged here: IB Gateway handles its own login,
and this adapter only ever receives a host, a port, and a client id (see the README for
how to set those up on Arch Linux).

The ``ib`` constructor argument exists for tests: a real connection is made only when it
is left as ``None``, exactly as :class:`qr.sources.crypto.CryptoSource` defers importing
``ccxt`` until a client is actually needed. ``contract_factory`` and ``order_factory``
exist for the same reason, one level down: building a real request needs ``ib_async``'s
``Stock``/``LimitOrder`` classes even when ``ib`` itself is a test double, so those too
are swapped out instead, rather than imported unconditionally. Nothing here requires
``ib_async`` to be installed except an actual connection with its own contracts and
orders.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from qr import calendars
from qr.backtest.account import AccountState, Position
from qr.backtest.events import BUY, SELL, Order, OrderType, Side, TimeInForce
from qr.live.broker import Broker, BrokerError, PriceQuote

PAPER_PORTS = {4002, 7497}  # IB Gateway paper, TWS paper
LIVE_PORTS = {4001, 7496}  # IB Gateway live, TWS live -- never used to size or send orders

DEFAULT_LIMIT_BUFFER = 0.003  # 0.3% through the mid, in the order's own direction
MARKET_OPEN_GRACE_MINUTES = 15
PRICE_WAIT_SECONDS = 2.0

_TERMINAL_IB_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


class IBKRError(BrokerError):
    """Raised when the IB connection or an order cannot be carried out as asked."""


def _status(ib_status: str) -> str:
    """IB's order status vocabulary -> ours (``qr.backtest.events.OrderStatus``)."""
    if ib_status == "Filled":
        return "filled"
    if ib_status in ("Cancelled", "ApiCancelled", "Inactive"):
        return "cancelled"
    return "open"  # PendingSubmit, PreSubmitted, Submitted, PendingCancel, ...


def _default_contract(symbol: str) -> Any:
    from ib_async import Stock

    return Stock(symbol, "SMART", "USD")


def _default_order(action: str, quantity: float, price: float, tif: str) -> Any:
    from ib_async import LimitOrder

    return LimitOrder(action, quantity, price, tif=tif)


class IBKRBroker(Broker):
    """A paper-trading account at Interactive Brokers, through IB Gateway or TWS."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        limit_buffer: float = DEFAULT_LIMIT_BUFFER,
        ib: Any = None,
        contract_factory: Callable[[str], Any] = _default_contract,
        order_factory: Callable[[str, float, float, str], Any] = _default_order,
    ) -> None:
        self._port = port
        self._limit_buffer = limit_buffer
        self._contract_factory = contract_factory
        self._order_factory = order_factory
        self._ids = itertools.count(1)
        self._orders: dict[str, Any] = {}  # our id -> ib_async Trade
        self._submitted_at: dict[str, datetime] = {}
        if ib is not None:
            self._ib = ib
        else:
            try:
                from ib_async import IB
            except ImportError as exc:
                raise IBKRError(
                    "qr.live.ibkr needs ib_async; install it with `uv sync --extra live`"
                ) from exc
            self._ib = IB()
            self._ib.connect(host, port, clientId=client_id)
            self._ib.reqMarketDataType(3)  # delayed data: paper accounts rarely have more
        accounts = self._ib.managedAccounts()
        if not accounts:
            raise IBKRError("connected, but IB reports no managed accounts")
        self._account_id: str = str(accounts[0])

    # ------------------------------------------------------------------ account

    @property
    def is_paper(self) -> bool:
        return self._port in PAPER_PORTS and self._account_id.startswith("DU")

    @property
    def account_id(self) -> str:
        return self._account_id

    def account(self) -> AccountState:
        summary = {
            (row.tag, row.currency): row.value for row in self._ib.accountSummary(self._account_id)
        }
        cash = float(summary.get(("TotalCashValue", "BASE"), 0.0))
        equity = float(summary.get(("NetLiquidation", "BASE"), cash))
        positions = {
            item.contract.symbol: Position(
                quantity=item.position,
                avg_cost=item.averageCost,
                market_value=item.marketValue,
            )
            for item in self._ib.portfolio(self._account_id)
            if item.position
        }
        gross = sum(abs(p.market_value) for p in positions.values())
        net = sum(p.market_value for p in positions.values())
        unrealized = sum(p.market_value - p.quantity * p.avg_cost for p in positions.values())
        return AccountState(
            timestamp=datetime.now(),
            cash=cash,
            positions=positions,
            open_orders={},
            equity=equity,
            gross_exposure=gross,
            net_exposure=net,
            realized_pnl=0.0,
            unrealized_pnl=unrealized,
            fees_paid=0.0,
        )

    # ------------------------------------------------------------------ prices

    def _contract(self, symbol: str) -> Any:
        (qualified,) = self._ib.qualifyContracts(self._contract_factory(symbol))
        return qualified

    def _ticker(self, symbol: str) -> Any:
        ticker = self._ib.reqMktData(self._contract(symbol), "", snapshot=True)
        self._ib.sleep(PRICE_WAIT_SECONDS)
        self._ib.cancelMktData(ticker.contract)
        return ticker

    def prices(self, symbols: Sequence[str]) -> dict[str, PriceQuote]:
        quotes: dict[str, PriceQuote] = {}
        for symbol in symbols:
            ticker = self._ticker(symbol)
            price = ticker.marketPrice()
            if price is None or price != price:  # NaN: ib_async's "no price" sentinel
                continue
            as_of = ticker.time or datetime.now()
            quotes[symbol] = PriceQuote(price=float(price), as_of=as_of.replace(tzinfo=None))
        return quotes

    def _mid(self, ticker: Any) -> float | None:
        bid, ask = ticker.bid, ticker.ask
        if bid and ask and bid > 0 and ask > 0:
            return float(bid + ask) / 2.0
        price = ticker.marketPrice()
        return float(price) if price == price else None  # NaN check

    # ------------------------------------------------------------------ orders

    def place_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        type: OrderType = "limit",
        limit_price: float | None = None,
        tif: TimeInForce = "day",
    ) -> Order:
        if quantity <= 0:
            raise BrokerError(f"order quantity must be positive, got {quantity}")
        if type != "limit":
            raise IBKRError(f"qr.live.ibkr only places day limit orders, not {type!r}")
        self._check_market_open()

        contract = self._contract(symbol)
        if limit_price is None:
            ticker = self._ticker(symbol)
            mid = self._mid(ticker)
            if mid is None:
                raise IBKRError(f"no quote available for {symbol}; cannot price a limit order")
            buffer_ = self._limit_buffer if side == BUY else -self._limit_buffer
            limit_price = round(mid * (1.0 + buffer_), 2)

        now = datetime.now()
        order_id = f"IB{next(self._ids):06d}"
        action = "BUY" if side == BUY else "SELL"
        ib_order = self._order_factory(action, quantity, limit_price, "DAY")
        trade = self._ib.placeOrder(contract, ib_order)
        self._orders[order_id] = trade
        self._submitted_at[order_id] = now
        status = trade.orderStatus
        return Order(
            id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            type="limit",
            submitted_at=now,
            limit_price=limit_price,
            tif=tif,
            status=_status(status.status),  # type: ignore[arg-type]
            filled_quantity=status.filled or 0.0,
        )

    def _check_market_open(self) -> None:
        now = datetime.now()
        try:
            opened = calendars.session_open("etf", now.date())
        except ValueError as exc:
            raise IBKRError(f"the exchange is not open today: {exc}") from exc
        if now < opened:
            raise IBKRError(f"the exchange has not opened yet (opens {opened:%H:%M} UTC)")
        elapsed = (now - opened).total_seconds() / 60.0
        if elapsed < MARKET_OPEN_GRACE_MINUTES:
            raise IBKRError(
                f"only {elapsed:.0f} minutes since the open; waiting for "
                f"{MARKET_OPEN_GRACE_MINUTES} before placing orders"
            )

    def cancel_order(self, order_id: str) -> None:
        trade = self._orders.get(order_id)
        if trade is not None and trade.orderStatus.status not in _TERMINAL_IB_STATUSES:
            self._ib.cancelOrder(trade.order)

    def order_status(self, order_id: str) -> Order:
        trade = self._orders.get(order_id)
        if trade is None:
            raise BrokerError(f"unknown order id: {order_id}")
        status = trade.orderStatus
        side: Side = BUY if trade.order.action == "BUY" else SELL
        return Order(
            id=order_id,
            symbol=trade.contract.symbol,
            side=side,
            quantity=trade.order.totalQuantity,
            type="limit",
            submitted_at=self._submitted_at.get(order_id, datetime.now()),
            limit_price=trade.order.lmtPrice,
            tif="day",
            status=_status(status.status),  # type: ignore[arg-type]
            filled_quantity=status.filled or 0.0,
        )

    def open_orders(self) -> list[Order]:
        return [
            self.order_status(order_id)
            for order_id, trade in self._orders.items()
            if trade.orderStatus.status not in _TERMINAL_IB_STATUSES
        ]
