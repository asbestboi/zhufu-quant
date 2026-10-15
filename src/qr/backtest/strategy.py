"""Strategy base classes for both engines, and indicator helpers that cannot look ahead.

A strategy never receives the full dataset. The vectorized engine hands a
:class:`WeightStrategy` the bars up to the end of the backtest and then delays its weights
by one bar; the event engine hands an :class:`EventStrategy` one day of bars at a time and
a :class:`Context` whose ``history`` stops at the current bar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from qr.backtest.account import AccountState, Trade
from qr.backtest.events import BUY, SELL, Bar, Order, OrderType, TimeInForce

if TYPE_CHECKING:
    from qr.backtest.event_engine import EngineCore

# --------------------------------------------------------------------------- indicators
#
# Every helper returns, on each row, a value computed from that row and earlier rows only.
# The window must be full: until then the result is NaN rather than a short-window guess.


def sma[T: (pd.DataFrame, pd.Series)](prices: T, window: int) -> T:
    """Simple moving average of the last ``window`` rows, the current one included."""
    return prices.rolling(window, min_periods=window).mean()


def past_return[T: (pd.DataFrame, pd.Series)](prices: T, lookback: int, skip: int = 0) -> T:
    """Return from ``lookback`` rows ago to ``skip`` rows ago (``skip=0`` means "to now")."""
    if not 0 <= skip < lookback:
        raise ValueError("need 0 <= skip < lookback")
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def rolling_volatility[T: (pd.DataFrame, pd.Series)](returns: T, window: int) -> T:
    """Standard deviation of the last ``window`` returns (not annualized)."""
    return returns.rolling(window, min_periods=window).std()


def zscore[T: (pd.DataFrame, pd.Series)](values: T, window: int) -> T:
    """How many rolling standard deviations the current value is from its rolling mean."""
    rolling = values.rolling(window, min_periods=window)
    return (values - rolling.mean()) / rolling.std()


def closes(bars: pd.DataFrame) -> pd.DataFrame:
    """The (date x symbol) table of closes from a ``get_bars``-shaped frame."""
    return bars["close"].unstack("symbol")


# --------------------------------------------------------------------------- vectorized


class WeightStrategy(ABC):
    """A strategy for the vectorized engine: a table of target weights.

    Constructor keyword arguments are the config's ``params``.

    ``warmup_bars`` is how many bars of a symbol the signal needs, the current one
    included (a 200-day average needs 200). Until a symbol has that many, the engine
    drops its weight, whatever the strategy returned; 0 means from its first bar.
    """

    warmup_bars: int = 0

    @abstractmethod
    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Target portfolio weights as a (date x symbol) DataFrame.

        ``bars`` is total-return adjusted and shaped like ``qr.load.get_bars`` output. The
        row for date ``t`` must be computed from bars dated ``t`` or earlier: it is the
        decision made at the close of ``t``, and the engine trades it on the next
        session. Weights are fractions of equity (0.5 is half the account, negative is
        short); NaN means no position. Adjusted price *levels* shift when a later
        dividend is paid, so use ratios of prices, never the level itself.
        """


# --------------------------------------------------------------------------- event-driven


class Context:
    """What an :class:`EventStrategy` can see and do at the current bar.

    Reads are limited to the past; orders are executed on the next bar at the earliest.
    """

    def __init__(self, core: EngineCore, warmup_bars: int = 0) -> None:
        self._core = core
        self._warmup_bars = warmup_bars

    # reads

    @property
    def now(self) -> datetime:
        """The session whose close has just printed."""
        return self._core.now

    @property
    def symbols(self) -> list[str]:
        """The configured universe, including symbols that have no bars yet."""
        return self._core.symbols

    @property
    def account(self) -> AccountState:
        """A snapshot of the account, marked at the latest close."""
        return self._core.account_state()

    @property
    def cash(self) -> float:
        """Cash in the account."""
        return self._core.cash

    @property
    def equity(self) -> float:
        """Cash plus positions at the latest close."""
        return self._core.equity

    @property
    def is_rebalance(self) -> bool:
        """Whether today is a rebalance date under the config's ``rebalance`` schedule."""
        return self._core.is_rebalance

    def position(self, symbol: str) -> float:
        """Signed number of shares held in ``symbol``."""
        return self._core.position(symbol)

    def history(
        self, symbol: str, n: int, field: str = "close", adjusted: bool = True
    ) -> npt.NDArray[np.float64]:
        """The last ``n`` values of ``field`` for ``symbol``, up to and including today.

        A read-only NumPy array, oldest first; shorter when the symbol has fewer bars.
        With ``adjusted=True`` (default) prices are split- and dividend-adjusted using
        only actions that have already happened, so today's value is the traded price and
        a moving average is not broken by a split. Volume is always as traded.

        Unadjusted fields and volume are views of the engine's data, adjusted prices a
        small new array; neither may be written to. Use :meth:`history_df` for a pandas
        Series with dates.
        """
        return self._core.history(symbol, n, field, adjusted)

    def history_df(
        self, symbol: str, n: int, field: str = "close", adjusted: bool = True
    ) -> pd.Series:
        """Like :meth:`history`, as a pandas Series indexed by date (a copy).

        Building the Series costs far more than the lookup itself (about 40 microseconds
        a call against 6 for :meth:`history`), so prefer :meth:`history` in code that runs
        on every bar.
        """
        values = self._core.history(symbol, n, field, adjusted)
        index = self._core.history_index(symbol, n)
        return pd.Series(np.array(values), index=index, name=symbol)

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        """Orders waiting to be executed, optionally for one symbol."""
        return self._core.open_orders(symbol)

    # orders

    def order(
        self,
        symbol: str,
        quantity: float,
        type: OrderType | None = None,
        limit_price: float | None = None,
        tif: TimeInForce = "gtc",
    ) -> Order | None:
        """Submit an order for ``quantity`` shares: positive buys, negative sells.

        Without ``type`` the order follows the config's ``execution``: a market order
        (next open) or a market-on-close order (next close). Returns None when the
        quantity rounds to zero shares.
        """
        if not self.warmed_up(symbol) and self._opens(symbol, quantity):
            return None
        side = BUY if quantity > 0 else SELL
        return self._core.submit(symbol, side, abs(quantity), type, limit_price, tif)

    def warmed_up(self, symbol: str) -> bool:
        """Whether ``symbol`` has the strategy's ``warmup_bars`` bars, today's included."""
        n = self._warmup_bars
        return n <= 0 or len(self._core.history(symbol, n, "close", False)) >= n

    def _opens(self, symbol: str, quantity: float) -> bool:
        """Whether an order of ``quantity`` would grow the position or open a new one."""
        held = self._core.position(symbol)
        return held == 0 or (held > 0) == (quantity > 0)

    def order_limit(
        self, symbol: str, quantity: float, limit_price: float, tif: TimeInForce = "gtc"
    ) -> Order | None:
        """Submit a limit order: fills when a later bar trades through ``limit_price``."""
        return self.order(symbol, quantity, "limit", limit_price, tif)

    def order_target_quantity(self, symbol: str, quantity: float) -> Order | None:
        """Trade whatever takes the position in ``symbol`` to ``quantity`` shares.

        Unfilled market orders for the symbol are cancelled first, so repeated calls do
        not pile up.
        """
        for pending in self._core.open_orders(symbol):
            if pending.type != "limit":
                self._core.cancel(pending.id)
        delta = quantity - self._core.position(symbol)
        return self.order(symbol, delta) if delta else None

    def order_target_weight(self, symbol: str, weight: float) -> Order | None:
        """Trade towards ``weight`` of current equity in ``symbol``, sized at today's close.

        The config's ``max_weight``, ``long_only``, and ``min_trade_weight`` are applied.
        Returns None when no trade is needed or the symbol has no price yet.
        """
        quantity = self._core.target_quantity(symbol, weight)
        return None if quantity is None else self.order_target_quantity(symbol, quantity)

    def cancel(self, order_id: str) -> None:
        """Cancel an open order. Unknown or finished orders are ignored."""
        self._core.cancel(order_id)


class EventStrategy:
    """A strategy for the event-driven engine. Override the hooks you need.

    Constructor keyword arguments are the config's ``params``. ``warmup_bars`` works as
    for :class:`WeightStrategy`: orders that would open or grow a position in a symbol
    with fewer bars are dropped (``Context.order`` returns None); reducing one is not.
    """

    warmup_bars: int = 0

    def on_start(self, ctx: Context) -> None:
        """Called once before the first bar."""

    def on_bar(self, ctx: Context, bars: Mapping[str, Bar]) -> None:
        """Called after each session's close with that session's bars (unadjusted).

        ``bars`` is a read-only mapping from symbol to :class:`Bar` holding the symbols
        that traded that day; a ``Bar`` is only built when a symbol is looked up.

        Orders submitted here are executed on the next session.
        """

    def on_fill(self, ctx: Context, trade: Trade) -> None:
        """Called for each trade of the session, before ``on_bar``."""

    def on_end(self, ctx: Context) -> None:
        """Called once after the last bar."""
