"""A fully local, journal-backed paper broker: fills day limit orders against real
downloaded market data, never a live feed and never anything off this machine.

Used for the forward test. Unlike :class:`qr.live.sim_broker.SimBroker` (an in-memory
stand-in for development) or :class:`qr.live.ibkr.IBKRBroker` (a real paper account at a
real broker), this one keeps no state of its own at all between calls: ``account()``,
``open_orders()``, and ``order_status()`` are all rebuilt, every time, from the journal
(:mod:`qr.live.journal`) and the data store (:mod:`qr.load`). A fresh instance therefore
behaves identically to one that has been running the whole time, which is exactly what
makes a crash or a plain ``qr live run`` restart safe to resume -- there is no separate
account file to lose or to fall out of sync with the journal.

An order fills at the next session's open if the limit allows it (a "better than asked"
price), or at the limit itself if a later session's bar trades through it within
``window_sessions`` sessions of the decision, exactly like the backtester's own limit
matching (:mod:`qr.backtest.broker_sim`) -- this is what the ``test_cycle_vs_backtest``
style parity test checks. If a session's bar the resolution needs has not been
downloaded yet, the order simply stays open; the next call (not necessarily the same
process) tries again. Costs (commission, spread, slippage) come from the same
:class:`qr.backtest.costs.CostModel` the backtester uses. Every fill records the SHA-256
of exactly the bar it was decided from (:func:`qr.load.fingerprint`), so the same orders
replayed against the same data always produce the same fills.

Selling more than is currently held is refused outright (no shorting), and nothing here
ever reaches past the local data store and journal -- there is no network client in this
module at all. :mod:`qr.live.safety` enforces the kill switch, ``capital_limit``, and the
gross-leverage limit before an order is ever placed; this broker does not re-derive a
leverage limit of its own at fill time (the backtester's bisection search for the largest
affordable size is not replicated here).

Trades the 12 US-listed research ETFs directly (``config/universe.yaml``'s ``etf``
group) -- there is no UCITS mapping here, by design (see the README): a real-money
account based in the EU would need the UCITS-domiciled equivalent of each fund.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from qr import calendars
from qr.backtest.account import Account, AccountState
from qr.backtest.config import CostConfig
from qr.backtest.costs import CostModel, TradeCosts
from qr.backtest.events import BUY, SELL, Fill, Order, OrderType, Side, TimeInForce
from qr.live import journal
from qr.live.broker import Broker, BrokerError, PriceQuote
from qr.load import fingerprint, get_bars

DEFAULT_LIMIT_BUFFER = 0.003  # 0.3% through the latest close, in the order's direction
DEFAULT_WINDOW_SESSIONS = 3

_TERMINAL = ("filled", "cancelled", "rejected", "expired")


class PaperLedgerError(BrokerError):
    """Raised when an order cannot be carried out as asked."""


def _decision_date(did: str) -> date:
    """The date encoded in a decision id (``f"{strategy}:{decision_date}"``)."""
    return datetime.strptime(did.rsplit(":", 1)[-1], "%Y-%m-%d").date()


class PaperLedgerBroker(Broker):
    """A local paper account whose fills come from the real data store, not a feed."""

    def __init__(
        self,
        strategy: str,
        universe: Sequence[str],
        capital_limit: float,
        *,
        asset_class: str = "etf",
        window_sessions: int = DEFAULT_WINDOW_SESSIONS,
        limit_buffer: float = DEFAULT_LIMIT_BUFFER,
        costs: CostConfig | None = None,
        data_dir: Path | str | None = None,
        today: Callable[[], date] = date.today,
        account_id: str = "PAPER_LEDGER",
    ) -> None:
        self._strategy = strategy
        self._universe = list(universe)
        self._capital_limit = capital_limit
        self._asset_class = asset_class
        self._window_sessions = window_sessions
        self._limit_buffer = limit_buffer
        self._costs = CostModel(
            costs or CostConfig(), self._universe, {s: asset_class for s in self._universe}
        )
        self._data_dir = data_dir
        self._today = today
        self._account_id = account_id

    # ------------------------------------------------------------------ account

    @property
    def is_paper(self) -> bool:
        return True

    @property
    def account_id(self) -> str:
        return self._account_id

    def account(self) -> AccountState:
        """Cash and positions, replayed from every fill ever journaled for this strategy."""
        account = Account(cash=self._capital_limit, timestamp=datetime.now())
        fills = journal.read_fills_for_strategy(self._strategy, data_directory=self._data_dir)
        for _, row in fills.iterrows():
            side: Side = BUY if row["side"] == "buy" else SELL
            fill = Fill(
                order_id=row["order_id"], symbol=row["symbol"], side=side,
                quantity=float(row["quantity"]), price=float(row["price"]),
                commission=float(row["commission"]),
                timestamp=pd.Timestamp(row["filled_at"]).to_pydatetime(),
            )  # fmt: skip
            account.apply_fill(fill)
        quotes = self.prices(list(account.positions))
        if quotes:
            account.mark({s: q.price for s, q in quotes.items()}, datetime.now())
        return account.state()

    # ------------------------------------------------------------------ prices

    def prices(self, symbols: Sequence[str]) -> dict[str, PriceQuote]:
        quotes: dict[str, PriceQuote] = {}
        if not symbols:
            return quotes
        bars = get_bars(
            list(symbols), end=self._today(), adjustment="none", data_dir=self._data_dir
        )
        if bars.empty:
            return quotes
        for symbol in symbols:
            if symbol not in bars.index.get_level_values("symbol"):
                continue
            sub = bars.xs(symbol, level="symbol")
            last_date = pd.Timestamp(sub.index[-1]).date()
            quotes[symbol] = PriceQuote(
                price=float(sub["close"].iloc[-1]),
                as_of=datetime.combine(last_date, datetime.min.time()),
            )
        return quotes

    def _latest_price(self, symbol: str) -> float | None:
        quote = self.prices([symbol]).get(symbol)
        return quote.price if quote else None

    def _limit_price(self, reference_price: float, side: Side) -> float:
        buffer_ = self._limit_buffer if side == BUY else -self._limit_buffer
        return round(reference_price * (1.0 + buffer_), 2)

    # ------------------------------------------------------------------ orders

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
        if type != "limit":
            raise PaperLedgerError(
                f"qr.live.paper_ledger only places day limit orders, not {type!r}"
            )
        now = datetime.now()
        order_id = f"PL{uuid4().hex[:16]}"
        if limit_price is None:
            reference = self._latest_price(symbol)
            if reference is None:
                return Order(
                    id=order_id, symbol=symbol, side=side, quantity=quantity, type="limit",
                    submitted_at=now, tif=tif, status="rejected",
                    reason=f"no price available for {symbol}",
                )  # fmt: skip
            limit_price = self._limit_price(reference, side)
        return Order(
            id=order_id, symbol=symbol, side=side, quantity=quantity, type="limit",
            submitted_at=now, limit_price=limit_price, tif=tif, status="open",
        )  # fmt: skip

    def cancel_order(self, order_id: str) -> None:
        row = journal.find_order(order_id, data_directory=self._data_dir)
        if row is None or row["status"] in _TERMINAL:
            return
        journal.record_order(
            order_id, row["decision_id"], row["symbol"], row["side"], float(row["quantity"]),
            float(row["planned_price"]), "cancelled",
            submitted_at=pd.Timestamp(row["submitted_at"]).to_pydatetime(),
            data_directory=self._data_dir,
        )  # fmt: skip

    def _candidate_sessions(self, decision_date: date) -> list[date]:
        buffer_days = self._window_sessions * 3 + 10  # generous enough for weekends/holidays
        start = decision_date + timedelta(days=1)
        end = decision_date + timedelta(days=buffer_days)
        window = calendars.sessions(self._asset_class, start, end)
        return [pd.Timestamp(ts).date() for ts in window[: self._window_sessions]]

    def _reconstruct(
        self, order_id: str, symbol: str, side: Side, quantity: float,
        submitted_at: datetime, limit_price: float, status: str,
    ) -> Order:  # fmt: skip
        fill_row = journal.find_fill_by_order(order_id, data_directory=self._data_dir)
        filled_quantity = float(fill_row["quantity"]) if fill_row is not None else 0.0
        return Order(
            id=order_id, symbol=symbol, side=side, quantity=quantity, type="limit",
            submitted_at=submitted_at, limit_price=limit_price, tif="day",
            status=status, filled_quantity=filled_quantity,  # type: ignore[arg-type]
        )  # fmt: skip

    def order_status(self, order_id: str) -> Order:
        row = journal.find_order(order_id, data_directory=self._data_dir)
        if row is None:
            raise BrokerError(f"unknown order id: {order_id}")
        symbol = str(row["symbol"])
        side: Side = BUY if row["side"] == "buy" else SELL
        quantity = float(row["quantity"])
        planned_price = float(row["planned_price"])
        submitted_at = pd.Timestamp(row["submitted_at"]).to_pydatetime()
        did = str(row["decision_id"])
        limit_price = self._limit_price(planned_price, side)

        if row["status"] in _TERMINAL:
            return self._reconstruct(
                order_id, symbol, side, quantity, submitted_at, limit_price, str(row["status"])
            )

        for session in self._candidate_sessions(_decision_date(did)):
            bars = get_bars(
                [symbol], start=session, end=session, adjustment="none", data_dir=self._data_dir
            )
            if bars.empty:
                return Order(
                    id=order_id, symbol=symbol, side=side, quantity=quantity, type="limit",
                    submitted_at=submitted_at, limit_price=limit_price, tif="day", status="open",
                )  # fmt: skip
            bar = bars.iloc[0]
            if side == BUY:
                at_open, through = bar["open"] <= limit_price, bar["low"] < limit_price
            else:
                at_open, through = bar["open"] >= limit_price, bar["high"] > limit_price
            if not (at_open or through):
                continue

            fill_price_raw = float(bar["open"]) if at_open else limit_price
            passive = through and not at_open
            held = self.account().positions.get(symbol)
            held_quantity = held.quantity if held else 0.0
            fill_quantity = min(quantity, max(held_quantity, 0.0)) if side == SELL else quantity
            data_hash = fingerprint(
                [symbol], start=session, end=session, data_dir=self._data_dir
            ).hash
            if fill_quantity <= 0:
                journal.record_alert(
                    f"{symbol}: sell of {quantity:g} shares cancelled, no long position to sell",
                    severity="warning", did=did, data_directory=self._data_dir,
                )  # fmt: skip
                return Order(
                    id=order_id, symbol=symbol, side=side, quantity=quantity, type="limit",
                    submitted_at=submitted_at, limit_price=limit_price, tif="day",
                    status="cancelled", filled_quantity=0.0,
                )  # fmt: skip

            costs = self._costs.order_costs(symbol, fill_quantity, fill_price_raw)
            if passive:
                costs = TradeCosts(costs.commission, 0.0, 0.0)
            direction = 1.0 if side == BUY else -1.0
            fill_price = (
                fill_price_raw + direction * (costs.spread + costs.slippage) / fill_quantity
            )
            fill_price = (
                min(fill_price, limit_price) if side == BUY else max(fill_price, limit_price)
            )
            journal.record_fill(
                order_id, did, symbol, side, fill_quantity, fill_price, costs.commission,
                data_hash=data_hash, filled_at=datetime.combine(session, datetime.min.time()),
                data_directory=self._data_dir,
            )  # fmt: skip
            return Order(
                id=order_id, symbol=symbol, side=side, quantity=quantity, type="limit",
                submitted_at=submitted_at, limit_price=limit_price, tif="day",
                status="filled", filled_quantity=fill_quantity,
            )  # fmt: skip

        journal.record_alert(
            f"{symbol}: {quantity:g} shares unfilled after {self._window_sessions} sessions; "
            "cancelled, not chased",
            severity="warning", did=did, data_directory=self._data_dir,
        )  # fmt: skip
        return Order(
            id=order_id, symbol=symbol, side=side, quantity=quantity, type="limit",
            submitted_at=submitted_at, limit_price=limit_price, tif="day",
            status="cancelled", filled_quantity=0.0,
        )  # fmt: skip

    def open_orders(self) -> list[Order]:
        rows = journal.read_orders_for_strategy(self._strategy, data_directory=self._data_dir)
        if rows.empty:
            return []
        open_rows = rows[~rows["status"].isin(_TERMINAL)]
        return [self.order_status(str(row["order_id"])) for _, row in open_rows.iterrows()]
