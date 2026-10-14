"""Strategies, and the lookup from a config's ``strategy`` name to a class.

The reference strategies (buy and hold, SMA crossover, momentum) exist to validate the
engines, not to make money; each has a version for both engines. ``pairs`` comes from
the step 3 research and runs on the vectorized engine only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qr.backtest.strategies.buy_and_hold import BuyAndHoldEvents, BuyAndHoldWeights
from qr.backtest.strategies.pairs import PairsWeights
from qr.backtest.strategies.sma_crossover import SmaCrossoverEvents, SmaCrossoverWeights
from qr.backtest.strategies.xs_momentum import XsMomentumEvents, XsMomentumWeights
from qr.backtest.strategy import EventStrategy, WeightStrategy


@dataclass(frozen=True)
class StrategyEntry:
    """The implementations of one named strategy; None where an engine has none."""

    vectorized: type[WeightStrategy] | None = None
    event: type[EventStrategy] | None = None


STRATEGIES: dict[str, StrategyEntry] = {
    "buy_and_hold": StrategyEntry(BuyAndHoldWeights, BuyAndHoldEvents),
    "sma_crossover": StrategyEntry(SmaCrossoverWeights, SmaCrossoverEvents),
    "xs_momentum": StrategyEntry(XsMomentumWeights, XsMomentumEvents),
    "pairs": StrategyEntry(PairsWeights, None),
}


def register(
    name: str,
    vectorized: type[WeightStrategy] | None = None,
    event: type[EventStrategy] | None = None,
) -> None:
    """Make your own strategy available to configs under ``name``."""
    STRATEGIES[name] = StrategyEntry(vectorized, event)


def build_strategy(
    name: str, engine: str, params: dict[str, Any]
) -> WeightStrategy | EventStrategy:
    """A fresh instance of strategy ``name`` for ``engine``, built with ``params``."""
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy '{name}' (available: {', '.join(sorted(STRATEGIES))})")
    cls = STRATEGIES[name].vectorized if engine == "vectorized" else STRATEGIES[name].event
    if cls is None:
        raise ValueError(f"strategy '{name}' has no implementation for the {engine} engine")
    try:
        return cls(**params)
    except TypeError as exc:
        raise ValueError(f"strategy '{name}': bad params {params}: {exc}") from exc
