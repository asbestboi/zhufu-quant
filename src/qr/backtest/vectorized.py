"""The vectorized engine: target weights in, daily account values out.

How a day works (``t`` is a session, prices are total-return adjusted):

1. Holdings move from the previous close to the open of ``t``.
2. If a decision was made at the close of ``t-1`` and ``execution`` is ``next_open``, it
   is traded at the open of ``t``, and costs are paid.
3. Holdings move from the open to the close of ``t``.
4. With ``execution: next_close`` the same trade happens here, at the close of ``t``.
5. Short positions pay their borrow fee and the account is recorded.
6. If ``t`` is a rebalance date, the strategy's weights for ``t`` become the pending
   decision. They can only be traded on the next session: that one-bar delay is applied
   here, by the engine, so a strategy cannot trade on the bar it has just seen.

A decision is turned into a number of shares per symbol right away, from the equity and
the closes of the decision date: ``shares = weight * equity / close``. That is the order
a trader could actually send. Because the fill comes at a later price, the weights reached
are close to the targets but not equal to them, and a purchase that the gap made
unaffordable is scaled down to the leverage limit. Sizing at the fill price instead would
quietly use a price the decision could not have known.

Between rebalances nothing is traded and weights drift with prices. Positions are kept in
dollars, so shares are fractional; commissions per share use the unadjusted price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from qr.backtest.calendar import rebalance_mask
from qr.backtest.config import BacktestConfig, Constraints
from qr.backtest.costs import CostModel
from qr.backtest.data import MarketData
from qr.backtest.result import ACCOUNT_COLUMNS, COST_COLUMNS, EngineOutput, empty_trades
from qr.backtest.strategy import WeightStrategy

FloatArray = npt.NDArray[np.float64]

MAX_COST_ITERATIONS = 25
DUST = 1e-9  # trades below this fraction of equity are float noise, not orders


class LookAheadError(Exception):
    """A strategy's weights for a date changed when later data was taken away."""


def clean_weights(weights: pd.DataFrame, data: MarketData) -> pd.DataFrame:
    """Align a strategy's output to ``data``: every date, every symbol, NaN as 0."""
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("target_weights must return a (date x symbol) DataFrame")
    unknown = [str(c) for c in weights.columns if c not in data.symbols]
    if unknown:
        raise ValueError(f"weights for symbols outside the universe: {', '.join(unknown)}")
    aligned = weights.reindex(index=data.dates, columns=data.symbols).astype("float64")
    if np.isinf(aligned.to_numpy()).any():
        raise ValueError("target weights must be finite")
    return aligned.fillna(0.0)


def apply_constraints(weights: FloatArray, constraints: Constraints) -> FloatArray:
    """Clip weights to the config's limits; an over-levered row is scaled down as a whole."""
    out = weights.copy()
    if constraints.long_only:
        out = np.maximum(out, 0.0)
    if constraints.max_weight is not None:
        out = np.clip(out, -constraints.max_weight, constraints.max_weight)
    gross = np.abs(out).sum(axis=1, keepdims=True)
    limit = constraints.max_gross_leverage
    return out * np.where(gross > limit, limit / np.where(gross > 0, gross, 1.0), 1.0)


def verify_no_lookahead(
    strategy: WeightStrategy,
    data: MarketData,
    weights: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    checks: int,
) -> None:
    """Recompute the weights on truncated data and compare them with the full run.

    For ``checks`` dates spread over the backtest, the strategy is run again on the bars
    up to that date only. The weights it gives for that date must be the ones it gave
    with all the data available; if not, the strategy used the future (or is not
    deterministic) and :class:`LookAheadError` is raised.
    """
    if checks <= 0 or len(sessions) < 3:
        return
    picks = np.unique(np.linspace(0, len(sessions) - 1, checks + 2)[1:-1].astype(int))
    bar_dates = data.adjusted.index.get_level_values("date")
    for day in sessions[picks]:
        truncated = clean_weights(strategy.target_weights(data.adjusted[bar_dates <= day]), data)
        then, now = truncated.loc[day].to_numpy(), weights.loc[day].to_numpy()
        if not np.allclose(then, now, rtol=1e-9, atol=1e-9):
            worst = data.symbols[int(np.argmax(np.abs(then - now)))]
            raise LookAheadError(
                f"{type(strategy).__name__} looks ahead: its weights for {day.date()} change "
                f"when data after that date is removed (largest change: {worst})"
            )


@dataclass
class _Rebalance:
    holdings: FloatArray
    cash: float
    trade: FloatArray
    commission: FloatArray
    spread: FloatArray
    slippage: FloatArray


def _rebalance(
    holdings: FloatArray,
    cash: float,
    target: FloatArray,
    price: FloatArray,
    adv: FloatArray,
    costs: CostModel,
    max_leverage: float,
    min_trade_weight: float = 0.0,
) -> _Rebalance | None:
    """Trade ``holdings`` towards the dollar positions ``target`` at ``price``.

    ``target`` is NaN for symbols that cannot trade today; they are left alone. Trades
    smaller than ``min_trade_weight`` of equity are skipped unless they close a position.
    Trades that add exposure are scaled down together when, after paying costs from cash, gross
    exposure would exceed ``max_leverage`` times equity. Costs depend on the trades and
    the scaling depends on the costs, so the two are iterated until they stop moving.
    Trades that reduce exposure are never scaled.
    """
    equity = cash + float(holdings.sum())
    if equity <= 0:  # wiped out: nothing left to trade with
        return None
    tradable = np.isfinite(target) & np.isfinite(price) & (price > 0)
    safe_price = np.where(tradable, price, 1.0)
    trade = np.where(tradable, target - holdings, 0.0)
    small = np.abs(trade) < max(DUST, min_trade_weight) * equity
    exits = (target == 0) & (np.abs(trade) >= DUST * equity)
    trade[small & ~exits] = 0.0
    trade = costs.fillable_value(trade, adv)
    if not trade.any():
        return None

    # Split each trade into the part that closes exposure and the part that adds it.
    after = holdings + trade
    flips = holdings * after < 0
    shrinks = np.abs(after) <= np.abs(holdings)
    closing = np.where(flips, -holdings, np.where(shrinks, trade, 0.0))
    adding = trade - closing
    base_gross = float(np.abs(holdings + closing).sum())
    added_gross = float(np.abs(adding).sum())

    scale = 1.0
    for _ in range(MAX_COST_ITERATIONS):
        trade = closing + scale * adding
        parts = costs.trade_costs(trade / safe_price, safe_price, adv)
        cost = float(parts.total.sum())
        room = max_leverage * (equity - cost) - base_gross
        wanted = 1.0 if added_gross <= max(room, 0.0) else max(room, 0.0) / added_gross
        if abs(wanted - scale) <= 1e-13:
            break
        scale = wanted
    new_holdings = holdings + trade
    return _Rebalance(
        holdings=new_holdings,
        cash=equity - float(new_holdings.sum()) - cost,
        trade=trade,
        commission=parts.commission,
        spread=parts.spread,
        slippage=parts.slippage,
    )


def simulate(
    weights: FloatArray,
    data: MarketData,
    config: BacktestConfig,
    costs: CostModel,
) -> EngineOutput:
    """Run the day loop for decision ``weights`` (one row per date in ``data.dates``)."""
    sessions = data.sessions(config.start, config.end)
    if len(sessions) == 0:
        raise ValueError(f"no sessions between {config.start} and {config.end}")
    first = int(data.dates.searchsorted(sessions[0]))
    count, width = len(sessions), len(data.symbols)
    decides = rebalance_mask(sessions, config.rebalance)
    at_open = config.execution == "next_open"
    max_leverage = config.constraints.max_gross_leverage
    min_trade = config.constraints.min_trade_weight

    adj_open, adj_close = data.panel("open", adjusted=True), data.panel("close", adjusted=True)
    raw_open, raw_close = data.panel("open"), data.panel("close")
    adv = data.adv_value(costs.adv_window)
    # Price moves as growth factors; a day without a bar is a factor of 1 (no move).
    last_close = pd.DataFrame(adj_close).ffill().to_numpy()
    previous = np.vstack([np.full((1, width), np.nan), last_close[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        overnight = adj_open / previous
        intraday = adj_close / np.where(np.isfinite(adj_open), adj_open, previous)
    overnight = np.where(np.isfinite(overnight), overnight, 1.0)
    intraday = np.where(np.isfinite(intraday), intraday, 1.0)
    stamps = data.dates.to_numpy()
    days = np.diff(stamps, prepend=stamps[:1]) / np.timedelta64(1, "D")

    holdings = np.zeros(width)
    cash = float(config.initial_cash)
    fees = 0.0
    pending: FloatArray | None = None  # shares to hold, in adjusted-price units
    accounts = np.full((count, len(ACCOUNT_COLUMNS)), np.nan)
    positions = np.zeros((count, width))
    cost_rows = np.zeros((count, len(COST_COLUMNS)))
    fills: list[tuple[int, FloatArray, FloatArray, _Rebalance]] = []

    def execute(k: int, t: int, adjusted_price: FloatArray, price: FloatArray) -> None:
        nonlocal holdings, cash, fees, pending
        assert pending is not None
        target = pending * adjusted_price
        pending = None
        done = _rebalance(holdings, cash, target, price, adv[t], costs, max_leverage, min_trade)
        if done is None:
            return
        holdings, cash = done.holdings, done.cash
        fees += float(done.commission.sum())
        cost_rows[k, 0] += done.commission.sum()
        cost_rows[k, 1] += done.spread.sum()
        cost_rows[k, 2] += done.slippage.sum()
        fills.append((k, price, done.trade, done))

    for k in range(count):
        t = first + k
        holdings = holdings * overnight[t]
        if pending is not None and at_open:
            execute(k, t, adj_open[t], raw_open[t])
        holdings = holdings * intraday[t]
        if pending is not None and not at_open:
            execute(k, t, adj_close[t], raw_close[t])

        if (holdings < 0).any():
            borrow = float(costs.borrow_fees(np.maximum(-holdings, 0.0), days[t]).sum())
            cash -= borrow
            fees += borrow
            cost_rows[k, 3] = borrow

        value = float(holdings.sum())
        equity = cash + value
        accounts[k, 0] = cash
        accounts[k, 1] = equity
        accounts[k, 2] = float(np.abs(holdings).sum())
        accounts[k, 3] = value
        accounts[k, 6] = fees
        positions[k] = holdings

        if decides[k]:
            # The decision, as an order: shares sized with today's equity and closes.
            with np.errstate(divide="ignore", invalid="ignore"):
                shares = weights[t] * max(equity, 0.0) / last_close[t]
            pending = np.where(weights[t] == 0, 0.0, shares)

    symbols = pd.Index(data.symbols)
    return EngineOutput(
        accounts=pd.DataFrame(accounts, index=sessions, columns=ACCOUNT_COLUMNS),
        positions=pd.DataFrame(positions, index=sessions, columns=symbols),
        trades=_trades_frame(fills, sessions, data.symbols),
        costs=pd.DataFrame(cost_rows, index=sessions, columns=COST_COLUMNS),
        weights=pd.DataFrame(
            weights[first : first + count][decides], index=sessions[decides], columns=symbols
        ),
    )


def _trades_frame(
    fills: list[tuple[int, FloatArray, FloatArray, _Rebalance]],
    sessions: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    if not fills:
        return empty_trades()
    # One (rebalances x symbols) array per column, then keep the cells that traded.
    signed = np.stack([trade for _, _, trade, _ in fills])
    row, column = np.nonzero(signed)
    price = np.stack([price for _, price, _, _ in fills])[row, column]
    value = np.abs(signed[row, column])
    return pd.DataFrame(
        {
            "date": sessions[np.array([k for k, _, _, _ in fills])[row]],
            "symbol": np.array(symbols, dtype=object)[column],
            "side": np.where(signed[row, column] > 0, "buy", "sell"),
            "quantity": value / price,
            "price": price,
            "value": value,
            "commission": np.stack([done.commission for *_, done in fills])[row, column],
            "spread": np.stack([done.spread for *_, done in fills])[row, column],
            "slippage": np.stack([done.slippage for *_, done in fills])[row, column],
            "realized_pnl": np.nan,
            "order_id": "",
        }
    )


def run_vectorized(
    strategy: WeightStrategy,
    data: MarketData,
    config: BacktestConfig,
    costs: CostModel,
) -> EngineOutput:
    """Backtest a weight strategy.

    The strategy sees the adjusted bars up to the config's end date and returns a weight
    for every date; the engine then checks those weights for look-ahead
    (:func:`verify_no_lookahead`), applies the constraints, removes weights on symbols
    that have not had a bar yet (or fewer than the strategy's ``warmup_bars``), and
    trades each decision one session later.
    """
    sessions = data.sessions(config.start, config.end)
    weights = clean_weights(strategy.target_weights(data.adjusted), data)
    verify_no_lookahead(strategy, data, weights, sessions, config.lookahead_checks)
    limited = apply_constraints(weights.to_numpy(), config.constraints)
    # A symbol is not part of the universe until its first bar has printed, and not
    # tradable until it has the strategy's warm-up bars (see WeightStrategy.warmup_bars).
    bars_seen = np.cumsum(np.isfinite(data.panel("close")), axis=0)
    listed = bars_seen >= max(1, strategy.warmup_bars)
    return simulate(limited * listed, data, config, costs)
