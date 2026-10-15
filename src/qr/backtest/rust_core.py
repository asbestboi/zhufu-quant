"""The Rust core: the event engine's bar store, loop, matching, and account in Rust.

:class:`RustCore` implements the same :class:`~qr.backtest.event_engine.EngineCore`
interface as :class:`~qr.backtest.event_engine.PythonCore` on top of the compiled module
``qr_core`` (``rust/qr_core/``), so strategies, ``run_event``, metrics, and reports do not
change. It gives identical results: same trades, same account, bit for bit.

What lives where:

* **Rust** owns the bars (copied in once), the event loop, corporate actions, order
  matching, fills, costs, and the account. ``history()`` returns read-only NumPy views of
  the Rust bar store where it can, without copying.
* **Python** (this module) turns what Rust reports into the usual ``Fill``, ``Trade``,
  ``Order``, and ``AccountState`` objects, and keeps the ``Order`` objects strategies hold
  up to date. Strategies themselves stay in Python.

Pick the core with ``engine_core: python | rust`` in a backtest config, or override it for
every run with the environment variable ``QR_ENGINE_CORE``. Without the compiled module the
Python core is used and a warning is logged, unless ``QR_REQUIRE_RUST_CORE=1`` (used in CI
to make sure the Rust core really runs) turns that into an error.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import replace
from datetime import datetime
from types import ModuleType

import numpy as np
import pandas as pd

from qr.backtest.account import AccountState, Position, Trade
from qr.backtest.calendar import rebalance_mask
from qr.backtest.config import BacktestConfig
from qr.backtest.costs import CostModel
from qr.backtest.data import MarketData
from qr.backtest.event_engine import (
    CoreFactory,
    EngineCore,
    FloatArray,
    PythonCore,
    SessionBars,
    Step,
    group_actions,
)
from qr.backtest.events import BUY, SELL, Fill, Order, OrderType, Side, TimeInForce
from qr.frames import PRICE_COLUMNS, SPLIT, VALUE_COLUMNS

logger = logging.getLogger(__name__)

CORE_ENV_VAR = "QR_ENGINE_CORE"
REQUIRE_ENV_VAR = "QR_REQUIRE_RUST_CORE"
BUILD_COMMAND = "uv run maturin develop --release -m rust/qr_core/Cargo.toml"

# Bar fields by the number the Rust module uses for them.
FIELDS = {name: number for number, name in enumerate(VALUE_COLUMNS)}
ORDER_ID = re.compile(r"O(\d+)")


def _load() -> ModuleType | None:
    try:
        import qr_core  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return None
    module: ModuleType = qr_core
    return module


_qr_core = _load()


def rust_available() -> bool:
    """Whether the compiled ``qr_core`` module is installed."""
    return _qr_core is not None


def select_core(config: BacktestConfig) -> CoreFactory:
    """The core a run should use: ``$QR_ENGINE_CORE`` if set, else ``config.engine_core``.

    Falls back to the Python core, with a warning, when the Rust core is wanted but not
    installed; with ``QR_REQUIRE_RUST_CORE=1`` that is an error instead.
    """
    wanted = os.environ.get(CORE_ENV_VAR) or config.engine_core
    if wanted not in ("python", "rust"):
        raise ValueError(f"{CORE_ENV_VAR} must be 'python' or 'rust', not {wanted!r}")
    if wanted == "python":
        return PythonCore
    if rust_available():
        return RustCore
    message = f"the Rust core (qr_core) is not installed; build it with `{BUILD_COMMAND}`"
    if os.environ.get(REQUIRE_ENV_VAR) == "1":
        raise RuntimeError(message)
    logger.warning("%s. Using the Python core instead.", message)
    return PythonCore


class RustCore(EngineCore):
    """The event engine's core in Rust, behind the same interface as :class:`PythonCore`."""

    def __init__(self, data: MarketData, config: BacktestConfig, costs: CostModel) -> None:
        if _qr_core is None:
            raise RuntimeError(f"the Rust core is not installed; build it with `{BUILD_COMMAND}`")
        self.symbols = list(data.symbols)
        self._config = config
        self._index = {s: j for j, s in enumerate(self.symbols)}
        self._dates = data.dates
        sessions = data.sessions(config.start, config.end)
        if len(sessions) == 0:
            raise ValueError(f"no sessions between {config.start} and {config.end}")
        self._first = int(data.dates.searchsorted(sessions[0]))
        self._timestamps: list[datetime] = list(sessions.to_pydatetime())
        self._actions = group_actions(data, self.symbols)

        stamps = data.dates.to_numpy()
        days = np.diff(stamps, prepend=stamps[:1]) / np.timedelta64(1, "D")
        cost_params = [
            (
                p.per_share,
                p.min_per_order,
                p.max_pct,
                p.pct,
                p.spread,
                p.slippage,
                p.impact,
                p.impact_exponent,
                p.borrow_rate,
                p.max_participation,
                p.reject_over_cap,
            )
            for p in (costs.params[s] for s in self.symbols)
        ]
        actions = [
            (row, self._index[a.symbol], a.action == SPLIT, a.value)
            for row, todays in sorted(self._actions.items())
            for a in todays
        ]
        constraints = config.constraints
        self._engine = _qr_core.Engine(
            self.symbols,
            [data.panel(field) for field in VALUE_COLUMNS],
            [data.panel(field, adjusted=True) for field in PRICE_COLUMNS],
            data.adv_value(costs.adv_window),
            cost_params,
            actions,
            rebalance_mask(sessions, config.rebalance).tolist(),
            days.tolist(),
            self._first,
            float(config.initial_cash),
            constraints.long_only,
            constraints.max_gross_leverage,
            constraints.max_weight,
            constraints.min_trade_weight,
            config.whole_shares,
        )
        # Read-only views of the Rust bar store, for the session bars handed to strategies.
        self._panels = {field: self._engine.panel(n) for field, n in FIELDS.items()}
        self._has_bar = self._engine.has_bar()
        self._orders: list[Order] = []  # every order, live; order number n is _orders[n - 1]

    # ------------------------------------------------------------------ the loop

    def step(self) -> Step | None:
        result = self._engine.step()
        if result is None:
            return None
        raw_fills, borrow_fee, changed = result
        k = self._engine.session
        t = self._first + k
        now = self._timestamps[k]
        fills: list[Fill] = []
        trades: list[Trade] = []
        for number, j, is_buy, quantity, price, fees, spread, slip, trade, pnl in raw_fills:
            side: Side = BUY if is_buy else SELL
            symbol, order_id = self.symbols[j], f"O{number:07d}"
            fills.append(Fill(order_id, symbol, side, quantity, price, fees, now, spread, slip))
            trades.append(
                Trade(f"T{trade:07d}", symbol, side, quantity, price, fees, now, order_id, pnl)
            )
        for number in changed:
            self._sync(number)
        return Step(
            timestamp=now,
            bars=SessionBars(self._panels, self.symbols, self._index, self._has_bar, t, now),
            fills=fills,
            trades=trades,
            actions=self._actions.get(t, []),
            borrow_fee=borrow_fee,
        )

    def _sync(self, number: int) -> None:
        """Copy an order's changing fields from Rust onto its Python object."""
        quantity, limit_price, status, filled, reason = self._engine.order(number)
        order = self._orders[number - 1]
        if quantity != order.quantity:
            order.quantity = quantity
        if limit_price != order.limit_price:
            order.limit_price = limit_price
        order.status = status
        order.filled_quantity = filled
        order.reason = reason

    # ------------------------------------------------------------------ reads

    @property
    def now(self) -> datetime:
        k = self._engine.session
        return self._timestamps[0 if k is None else k]

    @property
    def cash(self) -> float:
        return float(self._engine.cash)

    @property
    def equity(self) -> float:
        return float(self._engine.equity)

    @property
    def is_rebalance(self) -> bool:
        return bool(self._engine.is_rebalance)

    def account_state(self) -> AccountState:
        cash, held, equity, gross, net, realized, unrealized, fees = self._engine.account_state()
        return AccountState(
            timestamp=self.now,
            cash=cash,
            positions={self.symbols[j]: Position(q, cost, value) for j, q, cost, value in held},
            open_orders={o.id: replace(o) for o in self.open_orders()},
            equity=equity,
            gross_exposure=gross,
            net_exposure=net,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            fees_paid=fees,
        )

    def position(self, symbol: str) -> float:
        j = self._index.get(symbol)
        return 0.0 if j is None else float(self._engine.position(j))

    def _column(self, symbol: str) -> int:
        if symbol not in self._index:
            raise KeyError(f"{symbol} is not in the backtest's universe")
        return self._index[symbol]

    def history(
        self, symbol: str, n: int, field: str = "close", adjusted: bool = True
    ) -> FloatArray:
        if field not in FIELDS:
            raise ValueError(f"unknown bar field: {field!r}")
        values: FloatArray = self._engine.history(self._column(symbol), n, FIELDS[field], adjusted)
        return values

    def history_index(self, symbol: str, n: int) -> pd.DatetimeIndex:
        rows = self._engine.window(self._column(symbol), n)
        index: pd.DatetimeIndex = self._dates[rows]
        return index

    def target_quantity(self, symbol: str, weight: float) -> float | None:
        j = self._column(symbol)
        if not math.isfinite(weight):
            return None
        quantity: float | None = self._engine.target_quantity(j, float(weight))
        return quantity

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
        j = self._column(symbol)
        if type is None:
            type = "market" if self._config.execution == "next_open" else "moc"
        number = self._engine.submit(j, side == BUY, quantity, type, limit_price, tif == "day")
        if number is None:
            return None
        # The quantity exactly as the Python broker stores it (rounded down to whole shares).
        if self._config.whole_shares:
            quantity = float(math.floor(quantity + 1e-9))
        order = Order(
            id=f"O{number:07d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            type=type,
            submitted_at=self.now,
            limit_price=limit_price,
            tif=tif,
        )
        self._orders.append(order)
        return order

    def cancel(self, order_id: str) -> None:
        match = ORDER_ID.fullmatch(order_id)
        if match and self._engine.cancel(int(match.group(1))):
            self._sync(int(match.group(1)))

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        if symbol is None:
            numbers = self._engine.open_orders()
        elif symbol in self._index:
            numbers = self._engine.open_orders(self._index[symbol])
        else:
            return []
        return [self._orders[n - 1] for n in numbers]

    def all_orders(self) -> list[Order]:
        return list(self._orders)
