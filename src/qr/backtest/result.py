"""What a backtest returns: the same tables whichever engine produced them."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from qr.backtest.account import AccountState, Position
from qr.backtest.config import BacktestConfig
from qr.backtest.stress import CostSensitivity

ACCOUNT_COLUMNS = [
    "cash",
    "equity",
    "gross_exposure",
    "net_exposure",
    "realized_pnl",
    "unrealized_pnl",
    "fees_paid",
]
COST_COLUMNS = ["commission", "spread", "slippage", "borrow"]
TRADE_COLUMNS = [
    "date",
    "symbol",
    "side",
    "quantity",
    "price",
    "value",
    "commission",
    "spread",
    "slippage",
    "realized_pnl",
    "order_id",
]


def empty_trades() -> pd.DataFrame:
    """A trades table with the right columns and no rows."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in TRADE_COLUMNS}).astype(
        {c: "float64" for c in TRADE_COLUMNS if c not in ("date", "symbol", "side", "order_id")}
        | {"date": "datetime64[ns]"}
    )


@dataclass
class EngineOutput:
    """One simulation. Every table is indexed by session date unless noted.

    * ``accounts``: daily account snapshots at the close (``ACCOUNT_COLUMNS``). The
      vectorized engine does not track cost basis, so its two P&L columns are NaN.
    * ``positions``: market value per symbol.
    * ``trades``: one row per fill (``TRADE_COLUMNS``), in execution order. ``price`` is
      the price before spread and slippage in the vectorized engine and the fill price,
      which includes them, in the event engine; ``value`` is ``quantity * price``.
    * ``costs``: dollars paid per day (``COST_COLUMNS``).
    * ``weights``: the target weights decided on each rebalance date (vectorized only).
    * ``orders``: every order with its final status (event engine only).
    * ``final_state``: the account after the last session (event engine only).
    """

    accounts: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame
    weights: pd.DataFrame | None = None
    orders: pd.DataFrame | None = None
    final_state: AccountState | None = None

    @property
    def equity(self) -> pd.Series:
        """Account equity at each close."""
        return self.accounts["equity"]

    @property
    def turnover(self) -> pd.Series:
        """Value traded each day (buys plus sells) as a fraction of that day's equity."""
        traded = self.trades.groupby("date")["value"].sum() if len(self.trades) else None
        daily = pd.Series(0.0, index=self.accounts.index)
        if traded is not None:
            daily = daily.add(traded, fill_value=0.0).reindex(self.accounts.index)
        return (daily / self.equity).rename("turnover")

    def snapshot(self, day: Any) -> AccountState:
        """The account at the close of ``day`` in the unified format.

        Open orders are not kept per day, and the vectorized engine has no share counts
        or cost basis: its positions carry market value only.
        """
        when = pd.Timestamp(day)
        numbers = np.asarray(self.accounts.loc[when], dtype="float64").tolist()
        row = dict(zip(self.accounts.columns, numbers, strict=True))
        values = np.asarray(self.positions.loc[when], dtype="float64").tolist()
        return AccountState(
            timestamp=when.to_pydatetime(),
            cash=float(row["cash"]),
            positions={
                str(symbol): Position(quantity=np.nan, avg_cost=np.nan, market_value=float(value))
                for symbol, value in zip(self.positions.columns, values, strict=True)
                if value != 0
            },
            open_orders={},
            equity=float(row["equity"]),
            gross_exposure=float(row["gross_exposure"]),
            net_exposure=float(row["net_exposure"]),
            realized_pnl=float(row["realized_pnl"]),
            unrealized_pnl=float(row["unrealized_pnl"]),
            fees_paid=float(row["fees_paid"]),
        )


@dataclass
class BacktestResult:
    """A finished backtest: the run with costs, the same run without, and the benchmark."""

    config: BacktestConfig
    net: EngineOutput
    gross: EngineOutput
    benchmark: pd.Series | None = None  # total-return value of initial_cash in the benchmark
    metrics: dict[str, float] = field(default_factory=dict)
    gross_metrics: dict[str, float] = field(default_factory=dict)
    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    data_snapshot: datetime | None = None
    # Filled by run_backtest: see qr.backtest.assumptions and qr.backtest.stress.
    assumptions: dict[str, Any] | None = None
    cost_sensitivity: CostSensitivity | None = None
    # Metrics of each stress run other than 1x, by multiplier.
    stress_metrics: dict[float, dict[str, float]] = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        """Equity after costs."""
        return self.net.equity

    @property
    def gross_equity(self) -> pd.Series:
        """Equity of the same strategy with every cost set to zero."""
        return self.gross.equity
