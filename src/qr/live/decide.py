"""Computing a registered strategy's target weights as of a date, on the real engine.

:func:`target_weights` runs the exact code path the backtester uses for an event
strategy: the real ``on_start``/``on_bar`` calls, on real historical data loaded the same
way a backtest loads it (:func:`qr.backtest.data.load_market_data`). The only thing
replaced is the terminal step of turning a decision into a simulated fill: a strategy
expresses its decision to ``qr.live`` by calling ``Context.order_target_weight``, exactly
as it would in a backtest, and that call is captured here instead of being executed.
Sizing the resulting trade (rounding, the minimum-trade band, the cash buffer) is
``qr.live.planner``'s job, not this module's.

Whether today actually *is* a decision date is decided by the caller (the monthly cycle),
using :func:`qr.backtest.calendar.rebalance_dates` on the strategy's rebalance schedule;
this module trusts ``as_of`` and only checks that data exists for it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from qr.backtest.config import BacktestConfig
from qr.backtest.costs import CostModel
from qr.backtest.data import load_market_data
from qr.backtest.event_engine import EngineCore, PythonCore
from qr.backtest.events import Order
from qr.backtest.strategies import build_strategy
from qr.backtest.strategy import Context, EventStrategy


class DecisionError(Exception):
    """Raised when a strategy's target weights cannot be computed as of a date."""


class _WeightCapturingContext(Context):
    """A :class:`Context` that records ``order_target_weight`` calls instead of trading.

    Every read (history, is_rebalance, symbols, account, ...) is the real engine's, so
    the strategy sees exactly what it would in a backtest; only the terminal action is
    replaced. A strategy that trades through ``order``/``order_target_quantity`` directly
    instead of ``order_target_weight`` is not supported for live decisions.
    """

    def __init__(self, core: EngineCore, warmup_bars: int) -> None:
        super().__init__(core, warmup_bars)
        self.weights: dict[str, float] = {}

    def order_target_weight(self, symbol: str, weight: float) -> Order | None:
        self.weights[symbol] = weight
        return None


def target_weights(
    strategy_name: str,
    params: dict[str, Any],
    symbols: list[str],
    as_of: date,
    data_dir: Path | str | None = None,
) -> dict[str, float]:
    """The weights ``strategy_name`` targets as of the close of ``as_of``.

    A symbol the strategy did not ask for a position in is left out of the returned
    mapping (0), matching ``Context.order_target_weight`` only being called for a
    symbol the strategy holds or wants to hold.

    Raises :class:`DecisionError` if ``strategy_name`` is not a registered event
    strategy, or there is no stored session on ``as_of``.
    """
    try:
        strategy = build_strategy(strategy_name, "event", params)
    except ValueError as exc:
        raise DecisionError(str(exc)) from exc
    if not isinstance(strategy, EventStrategy):
        raise DecisionError(f"'{strategy_name}' is not an event strategy; qr.live needs one")

    try:
        data = load_market_data(symbols, as_of, data_dir=data_dir)
    except ValueError as exc:
        raise DecisionError(str(exc)) from exc

    config = BacktestConfig(
        name="qr-live-decision",
        engine="event",
        strategy=strategy_name,
        params=params,
        universe=symbols,
        start=as_of,
        end=as_of,
        # A single-session window: that session is always its own decision date, so the
        # schedule itself does not need to re-derive "is as_of a month end".
        rebalance="daily",
        cost_stress=[],
    )
    costs = CostModel(config.costs, data.symbols, data.asset_classes, free=True)
    try:
        core = PythonCore(data, config, costs)
    except ValueError as exc:
        raise DecisionError(f"no session on {as_of}: {exc}") from exc
    ctx = _WeightCapturingContext(core, strategy.warmup_bars)

    strategy.on_start(ctx)
    step = core.step()
    if step is None:
        raise DecisionError(f"no session on {as_of}")
    strategy.on_bar(ctx, step.bars)
    return dict(ctx.weights)
