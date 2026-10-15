"""Type stubs for the compiled module ``qr_core`` (see ``src/lib.rs``).

Symbols are column numbers, orders are 1-based sequence numbers, and bar fields are
numbers: 0 open, 1 high, 2 low, 3 close, 4 volume. ``qr.backtest.rust_core.RustCore``
turns all of this into the usual Python objects.
"""

import numpy as np
import numpy.typing as npt

# (per_share, min_per_order, max_pct, pct, spread, slippage, impact, impact_exponent,
#  borrow_rate, max_participation, reject_over_cap)
CostTuple = tuple[float, float, float, float, float, float, float, float, float, float, bool]
# (order, symbol, is_buy, quantity, price, commission, spread, slippage, trade, realized_pnl)
FillTuple = tuple[int, int, bool, float, float, float, float, float, int, float]
# (quantity, limit_price, status, filled_quantity, reason)
OrderTuple = tuple[float, float | None, str, float, str]
# (cash, [(symbol, quantity, avg_cost, market_value)], equity, gross, net,
#  realized_pnl, unrealized_pnl, fees_paid)
StateTuple = tuple[
    float, list[tuple[int, float, float, float]], float, float, float, float, float, float
]

class Engine:
    def __init__(
        self,
        symbols: list[str],
        raw: list[npt.NDArray[np.float64]],
        adjusted: list[npt.NDArray[np.float64]],
        adv: npt.NDArray[np.float64],
        costs: list[CostTuple],
        actions: list[tuple[int, int, bool, float]],
        rebalance: list[bool],
        days: list[float],
        first: int,
        initial_cash: float,
        long_only: bool,
        max_gross_leverage: float,
        max_weight: float | None,
        min_trade_weight: float,
        whole_shares: bool,
    ) -> None: ...
    def step(self) -> tuple[list[FillTuple], float, list[int]] | None: ...
    @property
    def session(self) -> int | None: ...
    @property
    def cash(self) -> float: ...
    @property
    def equity(self) -> float: ...
    @property
    def is_rebalance(self) -> bool: ...
    def position(self, symbol: int) -> float: ...
    def account_state(self) -> StateTuple: ...
    def history(
        self, symbol: int, n: int, field: int, adjusted: bool
    ) -> npt.NDArray[np.float64]: ...
    def window(self, symbol: int, n: int) -> list[int]: ...
    def panel(self, field: int) -> npt.NDArray[np.float64]: ...
    def has_bar(self) -> npt.NDArray[np.bool_]: ...
    def target_quantity(self, symbol: int, weight: float) -> float | None: ...
    def submit(
        self,
        symbol: int,
        is_buy: bool,
        quantity: float,
        order_type: str,
        limit_price: float | None,
        day: bool,
    ) -> int | None: ...
    def cancel(self, number: int) -> bool: ...
    def open_orders(self, symbol: int | None = None) -> list[int]: ...
    def order(self, number: int) -> OrderTuple: ...
