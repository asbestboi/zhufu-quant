"""Small builders shared by the qr.live tests."""

from __future__ import annotations

from datetime import datetime

from qr.backtest.account import AccountState, Position

NOW = datetime(2024, 1, 31, 21, 0, 0)


def make_account(
    cash: float, positions: dict[str, tuple[float, float]] | None = None, now: datetime = NOW
) -> AccountState:
    """An :class:`AccountState` from cash and ``{symbol: (quantity, price)}``."""
    built = {
        symbol: Position(quantity=qty, avg_cost=price, market_value=qty * price)
        for symbol, (qty, price) in (positions or {}).items()
    }
    market_value = sum(p.market_value for p in built.values())
    gross = sum(abs(p.market_value) for p in built.values())
    return AccountState(
        timestamp=now,
        cash=cash,
        positions=built,
        open_orders={},
        equity=cash + market_value,
        gross_exposure=gross,
        net_exposure=market_value,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        fees_paid=0.0,
    )
