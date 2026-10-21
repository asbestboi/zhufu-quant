"""A shadow backtest: the same strategy and costs, run as a backtest for comparison.

Never trades -- it exists only to check the paper ledger's own fills against what the
backtester would have done over the same months (the pre-registration's consistency
check), and to put the candidate's live performance next to a benchmark's, for the
monthly report and dashboard. Runs the production event engine directly
(:func:`qr.backtest.event_engine.run_event`), the same one a real backtest uses, with
none of :func:`qr.backtest.runner.execute`'s side effects (no registry row, no report
file) -- this is called every day, not once per experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from qr.backtest.account import AccountState
from qr.backtest.config import BacktestConfig, CostConfig
from qr.backtest.costs import CostModel
from qr.backtest.data import load_market_data
from qr.backtest.event_engine import run_event
from qr.backtest.strategies import build_strategy
from qr.backtest.strategy import EventStrategy
from qr.live import journal
from qr.load import get_bars


@dataclass(frozen=True)
class ShadowResult:
    """One strategy's shadow run over a period: its equity curve and the end state."""

    equity: pd.Series  # equity at each session's close, indexed by date
    final_state: AccountState | None


def run_shadow(
    strategy: str,
    params: dict[str, object],
    universe: list[str],
    start: date,
    end: date,
    *,
    initial_cash: float,
    costs: CostConfig | None = None,
    data_dir: Path | str | None = None,
) -> ShadowResult:
    """Run ``strategy`` as an event-engine backtest from ``start`` to ``end``.

    Always the event engine, never the vectorized one: it is what ``qr.live.decide``
    uses through ``target_weights``, and what a live decision is actually compared
    against, so the two never differ over something as incidental as which engine ran.
    """
    data = load_market_data(universe, end, data_dir=data_dir)
    cost_config = costs or CostConfig()
    cost_model = CostModel(cost_config, data.symbols, data.asset_classes)
    impl = build_strategy(strategy, "event", params)
    if not isinstance(impl, EventStrategy):
        raise ValueError(f"'{strategy}' is not an event strategy; qr.live needs one")
    config = BacktestConfig(
        name="shadow", engine="event", strategy=strategy, params=params, universe=universe,
        start=start, end=end, initial_cash=initial_cash, benchmark=None, rebalance="monthly",
        costs=cost_config, cost_stress=[],
    )  # fmt: skip
    output = run_event(impl, data, config, cost_model)
    return ShadowResult(equity=output.equity, final_state=output.final_state)


def ledger_state(
    strategy: str,
    capital_limit: float,
    as_of: date | None = None,
    data_dir: Path | str | None = None,
) -> tuple[float, dict[str, float]]:
    """``(cash, quantities)`` from every fill through ``as_of`` (all of them if ``None``).

    The one place this strategy's cash and positions are derived from its fills --
    ``qr.live.report`` and the dashboard both build on this, rather than each replaying
    the journal their own way.
    """
    fills = journal.read_fills_for_strategy(strategy, data_directory=data_dir)
    cash = capital_limit
    quantities: dict[str, float] = {}
    if not fills.empty:
        relevant = fills
        if as_of is not None:
            cutoff = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)
            relevant = fills[pd.to_datetime(fills["filled_at"]) <= cutoff]
        for _, fill in relevant.iterrows():
            signed = fill["quantity"] if fill["side"] == "buy" else -fill["quantity"]
            quantities[fill["symbol"]] = quantities.get(fill["symbol"], 0.0) + signed
            cash -= signed * fill["price"] + fill["commission"]
    return cash, {symbol: qty for symbol, qty in quantities.items() if qty}


def mark_to_market(
    strategy: str, capital_limit: float, as_of: date, data_dir: Path | str | None = None
) -> float:
    """Cash plus the market value of every fill through ``as_of``, at its real close."""
    cash, held = ledger_state(strategy, capital_limit, as_of, data_dir)
    if held:
        bars = get_bars(list(held), end=as_of, adjustment="none", data_dir=data_dir)
        if not bars.empty:
            closes = bars["close"].groupby(level="symbol").last()
            for symbol, quantity in held.items():
                if symbol in closes.index:
                    cash += quantity * float(closes[symbol])
    return cash


def ledger_equity_curve(
    strategy: str, capital_limit: float, data_dir: Path | str | None = None
) -> pd.Series:
    """The paper ledger's own equity at each of its decision dates, marked at that
    date's real close.

    Monthly resolution, one point per decision -- the journal is the source of truth for
    fills; this only marks them to market, using exactly the same prices
    `qr.live.paper_ledger` itself reads through `qr.load.get_bars`.
    """
    decisions = journal.read_decisions(strategy, data_directory=data_dir)
    if decisions.empty:
        return pd.Series(dtype=float)
    points = {
        pd.Timestamp(as_of): mark_to_market(strategy, capital_limit, as_of, data_dir)
        for as_of in (pd.Timestamp(d).date() for d in sorted(decisions["decision_date"]))
    }
    return pd.Series(points).sort_index()


def period_return(equity: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """The simple return of ``equity`` between the sessions at or before ``start``/``end``.

    ``None`` if ``equity`` has no observation at or before ``start`` to measure from.
    """
    before_start = equity.loc[:start]
    before_end = equity.loc[:end]
    if before_start.empty or before_end.empty:
        return None
    return float(before_end.iloc[-1] / before_start.iloc[-1] - 1.0)
