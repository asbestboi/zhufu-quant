"""Strategies, and the lookup from a config's ``strategy`` name to a class.

The reference strategies (buy and hold, SMA crossover, momentum) exist to validate the
engines, not to make money; each has a version for both engines. ``pairs`` comes from
the step 3 research and runs on the vectorized engine only.

Other packages add strategies through the ``qr.strategies`` entry-point group. Each entry
point names a module or a callable; it is loaded (and a callable called) the first time a
name is looked up, and is expected to call :func:`register`. In the plugin's
``pyproject.toml``::

    [project.entry-points."qr.strategies"]
    mine = "my_package.strategies:register_all"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from qr.backtest.strategies.buy_and_hold import BuyAndHoldEvents, BuyAndHoldWeights
from qr.backtest.strategies.pairs import PairsWeights
from qr.backtest.strategies.sma_crossover import SmaCrossoverEvents, SmaCrossoverWeights
from qr.backtest.strategies.xs_momentum import XsMomentumEvents, XsMomentumWeights
from qr.backtest.strategy import EventStrategy, WeightStrategy

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "qr.strategies"


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


_plugins_loaded = False


def load_plugins() -> None:
    """Load every installed ``qr.strategies`` entry point, once per process.

    A plugin that fails to load is an error: a config naming one of its strategies
    would otherwise fail later with a confusing "unknown strategy".
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            loaded = entry.load()
            if callable(loaded):
                loaded()
        except Exception as exc:
            raise ValueError(
                f"strategy plugin '{entry.name}' ({entry.value}) failed to load: {exc}"
            ) from exc
        logger.debug("loaded strategy plugin %s (%s)", entry.name, entry.value)


def build_strategy(
    name: str, engine: str, params: dict[str, Any]
) -> WeightStrategy | EventStrategy:
    """A fresh instance of strategy ``name`` for ``engine``, built with ``params``."""
    if name not in STRATEGIES:
        load_plugins()
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy '{name}' (available: {', '.join(sorted(STRATEGIES))})")
    cls = STRATEGIES[name].vectorized if engine == "vectorized" else STRATEGIES[name].event
    if cls is None:
        raise ValueError(f"strategy '{name}' has no implementation for the {engine} engine")
    try:
        return cls(**params)
    except TypeError as exc:
        raise ValueError(f"strategy '{name}': bad params {params}: {exc}") from exc
