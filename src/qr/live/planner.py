"""Target weights + current positions + prices -> an order plan.

Mirrors the backtester's own no-trade band (``qr.backtest.event_engine.PythonCore.
target_quantity``: a trade smaller than ``min_trade_weight`` of equity is skipped unless
it would close the position to flat), so a live run and the shadow backtest size trades
the same way. The only addition is the cash buffer: target weights are scaled down so the
plan never intends to invest the last ``cash_buffer`` fraction of equity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from qr.backtest.account import AccountState
from qr.backtest.events import BUY, SELL, Side

DEFAULT_CASH_BUFFER = 0.01
DEFAULT_MIN_TRADE_WEIGHT = 0.02


@dataclass(frozen=True)
class PlannedOrder:
    """One order the plan wants to send."""

    symbol: str
    side: Side
    quantity: float  # always positive; ``side`` carries the sign
    price: float
    estimated_value: float
    current_weight: float
    target_weight: float


@dataclass(frozen=True)
class Plan:
    """A full plan for one decision: every symbol considered, and the orders it implies."""

    orders: list[PlannedOrder]
    skipped: dict[str, str]  # symbol -> why no order was planned
    equity: float
    cash_buffer: float


def plan_orders(
    target_weights: Mapping[str, float],
    account: AccountState,
    prices: Mapping[str, float],
    *,
    cash_buffer: float = DEFAULT_CASH_BUFFER,
    min_trade_weight: float = DEFAULT_MIN_TRADE_WEIGHT,
    whole_shares: bool = True,
) -> Plan:
    """Plan the orders that move ``account`` towards ``target_weights``.

    A symbol with a target weight but no price is skipped (not an error here: callers
    run their own price-sanity checks first and decide whether to proceed at all).
    Orders are returned sells before buys, so a sell can free cash a buy needs.
    """
    if not 0 <= cash_buffer < 1:
        raise ValueError(f"cash_buffer must be in [0, 1), got {cash_buffer}")
    equity = account.equity
    investable = equity * (1.0 - cash_buffer)
    symbols = sorted(set(target_weights) | {s for s, p in account.positions.items() if p.quantity})

    orders: list[PlannedOrder] = []
    skipped: dict[str, str] = {}
    for symbol in symbols:
        target = target_weights.get(symbol, 0.0)
        held = account.positions.get(symbol)
        held_quantity = held.quantity if held else 0.0
        current_weight = (held.market_value / equity) if held and equity else 0.0
        price = prices.get(symbol)
        if price is None or price <= 0:
            if target or held_quantity:
                skipped[symbol] = "no price available"
            continue
        target_value = target * investable
        target_quantity = target_value / price
        if whole_shares:
            target_quantity = math.trunc(target_quantity + math.copysign(1e-9, target_quantity))
        delta = target_quantity - held_quantity
        if delta == 0:
            continue
        trade_value = abs(delta) * price
        # The no-trade band never blocks closing a position to flat (target 0).
        if target != 0 and trade_value < min_trade_weight * equity:
            skipped[symbol] = (
                f"trade value ${trade_value:,.2f} is below the {min_trade_weight:.1%} "
                "minimum-trade band"
            )
            continue
        side: Side = BUY if delta > 0 else SELL
        orders.append(
            PlannedOrder(
                symbol=symbol,
                side=side,
                quantity=abs(delta),
                price=price,
                estimated_value=trade_value,
                current_weight=current_weight,
                target_weight=target,
            )
        )
    orders.sort(key=lambda o: (o.side != SELL, o.symbol))
    return Plan(orders=orders, skipped=skipped, equity=equity, cash_buffer=cash_buffer)
