"""The event-driven engine: a strategy reacts to one session at a time and submits orders.

A session ``t`` runs in this order, all in unadjusted prices:

1. Corporate actions with ex-date ``t``: splits restate positions and open orders, then
   dividends are paid in cash on the shares held (shorts pay them).
2. The broker matches the orders submitted on earlier sessions against bar ``t``
   (see ``broker_sim.py``).
3. Positions are marked at the close of ``t`` and shorts pay their borrow fee.
4. The strategy gets ``on_fill`` for each trade, then ``on_bar`` with the bars of ``t``.
   Whatever it submits now waits for session ``t+1``, so a decision made on the close of
   ``t`` can never be executed at a price from ``t``.

:class:`EngineCore` is everything a strategy's :class:`~qr.backtest.strategy.Context` and
the driver :func:`run_event` need: time, the account, the order book, and price history.
:class:`PythonCore` is the reference implementation; a faster core (the spec's optional
Rust one) only has to implement the same interface, and strategies need not change.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from qr.backtest.account import Account, AccountState, Trade
from qr.backtest.broker_sim import SimBroker
from qr.backtest.calendar import rebalance_mask
from qr.backtest.config import BacktestConfig
from qr.backtest.costs import CostModel
from qr.backtest.data import MarketData
from qr.backtest.events import Bar, CorporateAction, Fill, Order, OrderType, Side, TimeInForce
from qr.backtest.result import (
    ACCOUNT_COLUMNS,
    COST_COLUMNS,
    TRADE_COLUMNS,
    EngineOutput,
    empty_trades,
)
from qr.backtest.strategy import Context, EventStrategy
from qr.frames import PRICE_COLUMNS, SPLIT


@dataclass(frozen=True)
class Step:
    """What happened in one session."""

    timestamp: datetime
    bars: dict[str, Bar]
    fills: list[Fill]
    trades: list[Trade]
    actions: list[CorporateAction]
    borrow_fee: float


class EngineCore(ABC):
    """The event loop, order matching, and account, behind one small interface."""

    symbols: list[str]

    @abstractmethod
    def step(self) -> Step | None:
        """Advance one session and return what happened, or None after the last one."""

    @property
    @abstractmethod
    def now(self) -> datetime:
        """The current session (before the first step: the first session's date)."""

    @property
    @abstractmethod
    def cash(self) -> float:
        """Cash in the account."""

    @property
    @abstractmethod
    def equity(self) -> float:
        """Cash plus positions at their latest marks."""

    @property
    @abstractmethod
    def is_rebalance(self) -> bool:
        """Whether the current session is a rebalance date."""

    @abstractmethod
    def account_state(self) -> AccountState:
        """A snapshot of the account."""

    @abstractmethod
    def position(self, symbol: str) -> float:
        """Signed shares held in ``symbol``."""

    @abstractmethod
    def history(self, symbol: str, n: int, field: str, adjusted: bool) -> pd.Series:
        """The last ``n`` values of a bar field up to the current session, never beyond."""

    @abstractmethod
    def target_quantity(self, symbol: str, weight: float) -> float | None:
        """Shares that make ``symbol`` worth ``weight`` of equity at its latest close."""

    @abstractmethod
    def submit(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        type: OrderType | None,
        limit_price: float | None,
        tif: TimeInForce,
    ) -> Order | None:
        """Queue an order for the next session."""

    @abstractmethod
    def cancel(self, order_id: str) -> None:
        """Cancel an open order."""

    @abstractmethod
    def open_orders(self, symbol: str | None = None) -> list[Order]:
        """Orders waiting to be executed."""

    @abstractmethod
    def all_orders(self) -> list[Order]:
        """Every order submitted so far, with its current status."""


class PythonCore(EngineCore):
    """The reference core, in plain Python."""

    def __init__(self, data: MarketData, config: BacktestConfig, costs: CostModel) -> None:
        self.symbols = list(data.symbols)
        self._config = config
        self._costs = costs
        self._dates = data.dates
        self._sessions = data.sessions(config.start, config.end)
        if len(self._sessions) == 0:
            raise ValueError(f"no sessions between {config.start} and {config.end}")
        self._first = int(data.dates.searchsorted(self._sessions[0]))
        self._rebalance = rebalance_mask(self._sessions, config.rebalance)
        self._index = {s: j for j, s in enumerate(self.symbols)}

        self._raw = {c: data.panel(c) for c in (*PRICE_COLUMNS, "volume")}
        self._adjusted = {c: data.panel(c, adjusted=True) for c in PRICE_COLUMNS}
        self._adv = data.adv_value(costs.adv_window)
        has_bar = np.isfinite(np.stack([self._raw[c] for c in PRICE_COLUMNS])).all(axis=0)
        self._has_bar = has_bar
        # Row numbers of each symbol's own bars, to look back over them quickly.
        self._rows = [np.flatnonzero(has_bar[:, j]) for j in range(len(self.symbols))]
        # Without gaps a look-back is a plain slice, which is much cheaper to take.
        self._gapless = [len(r) == 0 or r[-1] - r[0] + 1 == len(r) for r in self._rows]
        stamps = data.dates.to_numpy()
        self._days = np.diff(stamps, prepend=stamps[:1]) / np.timedelta64(1, "D")
        self._actions = self._group_actions(data)

        self._k = -1  # index of the current session; -1 before the first step
        self._t = self._first - 1  # the same position in data.dates
        start = self._sessions[0].to_pydatetime()
        self._account = Account(cash=float(config.initial_cash), timestamp=start)
        self._equity = self._account.equity  # refreshed once per session, read many times
        self._broker = SimBroker(self._account, costs, config.constraints, config.whole_shares)

    def _group_actions(self, data: MarketData) -> dict[int, list[CorporateAction]]:
        """Corporate actions by the row they take effect on: splits first, then dividends."""
        grouped: dict[int, list[CorporateAction]] = {}
        actions = data.actions[data.actions["symbol"].isin(self.symbols)]
        # An ex-date without a session (it should not happen) takes effect on the next one.
        rows = self._dates.searchsorted(pd.DatetimeIndex(actions["date"]), side="left")
        for row, record in zip(rows, actions.to_dict("records"), strict=True):
            if row >= len(self._dates):
                continue
            action = CorporateAction(
                symbol=str(record["symbol"]),
                timestamp=self._dates[int(row)].to_pydatetime(),
                action=record["action"],
                value=float(record["value"]),
            )
            grouped.setdefault(int(row), []).append(action)
        for todays in grouped.values():
            todays.sort(key=lambda a: (a.action != SPLIT, self._index[a.symbol]))
        return grouped

    # ------------------------------------------------------------------ the loop

    def step(self) -> Step | None:
        if self._k + 1 >= len(self._sessions):
            return None
        self._k += 1
        self._t = t = self._first + self._k
        now = self._sessions[self._k].to_pydatetime()

        actions = self._actions.get(t, [])
        for action in actions:
            if action.action == SPLIT:
                self._account.apply_split(action.symbol, action.value)
                self._broker.apply_split(action.symbol, action.value)
            else:
                self._account.apply_dividend(action.symbol, action.value)

        bars = {
            symbol: Bar(
                symbol=symbol,
                timestamp=now,
                open=float(self._raw["open"][t, j]),
                high=float(self._raw["high"][t, j]),
                low=float(self._raw["low"][t, j]),
                close=float(self._raw["close"][t, j]),
                volume=float(self._raw["volume"][t, j]),
            )
            for j, symbol in enumerate(self.symbols)
            if self._has_bar[t, j]
        }
        adv = {o.symbol: float(self._adv[t, self._index[o.symbol]]) for o in self.open_orders()}
        done = self._broker.run_session(bars, adv, now)

        borrow = 0.0
        for symbol, position in self._account.positions.items():
            if position.quantity < 0:
                fee = self._costs.borrow_fee(symbol, position.market_value, float(self._days[t]))
                borrow += fee
        if borrow:
            self._account.charge_fee(borrow)
        self._equity = self._account.equity
        return Step(
            timestamp=now,
            bars=bars,
            fills=[fill for fill, _ in done],
            trades=[trade for _, trade in done],
            actions=actions,
            borrow_fee=borrow,
        )

    # ------------------------------------------------------------------ reads

    @property
    def now(self) -> datetime:
        return self._sessions[max(self._k, 0)].to_pydatetime()

    @property
    def cash(self) -> float:
        return self._account.cash

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def is_rebalance(self) -> bool:
        return self._k >= 0 and bool(self._rebalance[self._k])

    def account_state(self) -> AccountState:
        return self._account.state()

    def position(self, symbol: str) -> float:
        return self._account.quantity(symbol)

    def _column(self, symbol: str) -> int:
        if symbol not in self._index:
            raise KeyError(f"{symbol} is not in the backtest's universe")
        return self._index[symbol]

    def history(
        self, symbol: str, n: int, field: str = "close", adjusted: bool = True
    ) -> pd.Series:
        if field not in self._raw:
            raise ValueError(f"unknown bar field: {field!r}")
        j = self._column(symbol)
        rows = self._rows[j]
        # Only the symbol's bars at or before the current session are ever touched.
        seen = int(np.searchsorted(rows, self._t, side="right"))
        own = rows[max(0, seen - max(n, 0)) : seen]
        if len(own) == 0:
            return pd.Series([], index=self._dates[:0], name=symbol, dtype="float64")
        pick: slice | np.ndarray = slice(own[0], own[-1] + 1) if self._gapless[j] else own
        if field == "volume" or not adjusted:
            values = self._raw[field][pick, j].copy()
        else:
            # The stored adjusted series is anchored at the end of the data. Rescaling it
            # so the latest visible close is the traded one removes every action after
            # the current session: this is the series as it was known today.
            last = own[-1]
            anchor = self._raw["close"][last, j] / self._adjusted["close"][last, j]
            values = self._adjusted[field][pick, j] * anchor
        return pd.Series(values, index=self._dates[pick], name=symbol)

    def _last_close(self, symbol: str) -> float | None:
        j = self._column(symbol)
        seen = int(np.searchsorted(self._rows[j], self._t, side="right"))
        return float(self._raw["close"][self._rows[j][seen - 1], j]) if seen else None

    def target_quantity(self, symbol: str, weight: float) -> float | None:
        price = self._last_close(symbol)
        if price is None or not math.isfinite(weight):
            return None
        constraints = self._config.constraints
        if constraints.long_only:
            weight = max(weight, 0.0)
        if constraints.max_weight is not None:
            weight = min(max(weight, -constraints.max_weight), constraints.max_weight)
        quantity = weight * max(self._equity, 0.0) / price
        held = self._account.quantity(symbol)
        too_small = abs(quantity - held) * price < constraints.min_trade_weight * self._equity
        if weight != 0 and too_small:
            return held  # inside the no-trade band: leave the position as it is
        return (
            float(math.trunc(quantity + math.copysign(1e-9, quantity)))
            if (self._config.whole_shares)
            else quantity
        )

    # ------------------------------------------------------------------ orders

    def submit(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        type: OrderType | None = None,
        limit_price: float | None = None,
        tif: TimeInForce = "gtc",
    ) -> Order | None:
        self._column(symbol)
        if type is None:
            type = "market" if self._config.execution == "next_open" else "moc"
        return self._broker.submit(symbol, side, quantity, type, self.now, limit_price, tif)

    def cancel(self, order_id: str) -> None:
        self._broker.cancel(order_id)

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        orders = self._account.open_orders.values()
        return [o for o in orders if symbol is None or o.symbol == symbol]

    def all_orders(self) -> list[Order]:
        return list(self._broker.orders)


CoreFactory = Callable[[MarketData, BacktestConfig, CostModel], EngineCore]


def run_event(
    strategy: EventStrategy,
    data: MarketData,
    config: BacktestConfig,
    costs: CostModel,
    core_factory: CoreFactory = PythonCore,
) -> EngineOutput:
    """Backtest an event strategy and collect the same tables the vectorized engine gives.

    The run is deterministic: the same data, config, and strategy give identical output.
    """
    core = core_factory(data, config, costs)
    ctx = Context(core)
    dates: list[datetime] = []
    account_rows: list[list[float]] = []
    position_rows: list[dict[str, float]] = []
    cost_rows: list[list[float]] = []
    trade_rows: list[dict[str, object]] = []
    state: AccountState | None = None

    strategy.on_start(ctx)
    while (step := core.step()) is not None:
        for trade in step.trades:
            strategy.on_fill(ctx, trade)
        state = core.account_state()
        dates.append(step.timestamp)
        account_rows.append([getattr(state, column) for column in ACCOUNT_COLUMNS])
        position_rows.append({s: p.market_value for s, p in state.positions.items()})
        cost_rows.append(
            [
                sum(f.commission for f in step.fills),
                sum(f.spread_cost for f in step.fills),
                sum(f.slippage_cost for f in step.fills),
                step.borrow_fee,
            ]
        )
        for fill, trade in zip(step.fills, step.trades, strict=True):
            trade_rows.append(
                {
                    "date": step.timestamp,
                    "symbol": fill.symbol,
                    "side": fill.side,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "value": fill.quantity * fill.price,
                    "commission": fill.commission,
                    "spread": fill.spread_cost,
                    "slippage": fill.slippage_cost,
                    "realized_pnl": trade.realized_pnl,
                    "order_id": fill.order_id,
                }
            )
        strategy.on_bar(ctx, step.bars)
    strategy.on_end(ctx)

    index = pd.DatetimeIndex(dates, name="date").astype("datetime64[ns]")
    positions = pd.DataFrame(position_rows, index=index).reindex(columns=core.symbols)
    trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS) if trade_rows else empty_trades()
    trades["date"] = trades["date"].astype("datetime64[ns]")
    return EngineOutput(
        accounts=pd.DataFrame(account_rows, index=index, columns=ACCOUNT_COLUMNS, dtype="float64"),
        positions=positions.fillna(0.0).astype("float64"),
        trades=trades,
        costs=pd.DataFrame(cost_rows, index=index, columns=COST_COLUMNS, dtype="float64"),
        orders=pd.DataFrame([o.to_dict() for o in core.all_orders()]),
        final_state=state,
    )
