"""Trading costs shared by both engines: commission, spread, slippage, borrow, volume caps.

All amounts are in dollars and never negative. One :class:`CostModel` is built per run
for a fixed list of symbols; the vectorized engine calls it with arrays (one entry per
symbol), the event engine with single orders, and both do the same arithmetic.

* **Commission** = ``per_share * shares + pct * value``, capped at ``max_pct * value``,
  then raised to ``min_per_order``. When the minimum is larger than the cap the minimum
  wins, which is the more expensive reading of a broker's price list.
* **Spread** = ``value * spread_bps / 1e4``: every trade crosses half the spread.
* **Slippage** = ``value * (slippage_bps + impact_bps * participation ** exponent) / 1e4``
  where ``participation`` is the order's value divided by the average daily dollar volume.
* **Borrow fee** = ``short value * borrow_rate * days / 360``, charged for every calendar
  day a short position is held.
* **Cost multiplier**: ``multiplier`` scales every rate above (the commission cap and
  minimum included), so each cost is exactly ``multiplier`` times its 1x amount on the
  same trade. The participation cap is about liquidity, not cost, and is not scaled.
* **Participation cap**: an order worth more than ``max_participation`` times the average
  daily dollar volume is cut down to that size, or refused, depending on the config.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from qr.backtest.config import CostConfig, CostOverride

FloatArray = npt.NDArray[np.float64]

BORROW_DAY_COUNT = 360.0
DEFAULT_ASSET_CLASS = "equity"
# "No cap" as a huge finite number, so that multiplying it by a zero value gives zero.
NO_CAP = 1e30

# Starting points for `commission_model`; every number can be overridden in the config.
# IBKR figures are its US stock price list (exchange and regulatory fees not included).
COMMISSION_PRESETS: dict[str, dict[str, float]] = {
    "ibkr_tiered": {"per_share": 0.0035, "min_per_order": 0.35, "max_pct": 0.01, "pct": 0.0},
    "ibkr_fixed": {"per_share": 0.005, "min_per_order": 1.0, "max_pct": 0.01, "pct": 0.0},
    "per_share": {"per_share": 0.005, "min_per_order": 0.0, "max_pct": NO_CAP, "pct": 0.0},
    "percent": {"per_share": 0.0, "min_per_order": 0.0, "max_pct": NO_CAP, "pct": 0.001},
    "none": {"per_share": 0.0, "min_per_order": 0.0, "max_pct": NO_CAP, "pct": 0.0},
}

_OVERRIDABLE = tuple(CostOverride.model_fields)


@dataclass(frozen=True)
class CostParams:
    """The resolved cost numbers for one symbol. Rates are fractions, not basis points."""

    per_share: float
    min_per_order: float
    max_pct: float  # NO_CAP when there is no cap
    pct: float
    spread: float
    slippage: float
    impact: float
    impact_exponent: float
    borrow_rate: float
    max_participation: float  # inf when there is no cap
    reject_over_cap: bool

    @property
    def needs_volume(self) -> bool:
        """Whether a rule of this symbol depends on its trading volume."""
        return math.isfinite(self.max_participation) or self.impact > 0


@dataclass(frozen=True)
class TradeCosts:
    """Cost of one order (floats) or of one order per symbol (arrays)."""

    commission: Any
    spread: Any
    slippage: Any

    @property
    def total(self) -> Any:
        """Commission, spread, and slippage together."""
        return self.commission + self.spread + self.slippage


def resolve_params(config: CostConfig, symbol: str, asset_class: str) -> CostParams:
    """Cost numbers for ``symbol``: base config, then its asset class, then the symbol."""
    settings: dict[str, Any] = {name: getattr(config, name) for name in _OVERRIDABLE}
    for override in (config.asset_classes.get(asset_class), config.symbols.get(symbol)):
        if override is None:
            continue
        given = override.model_dump(exclude_none=True)
        if "commission_model" in given:
            # A different preset replaces the inherited commission numbers as a whole.
            for name in _OVERRIDABLE:
                if name.startswith("commission_"):
                    settings[name] = None
        settings.update(given)
    preset = dict(COMMISSION_PRESETS[settings["commission_model"]])
    for key in ("per_share", "min_per_order", "max_pct", "pct"):
        if settings[f"commission_{key}"] is not None:
            preset[key] = float(settings[f"commission_{key}"])
    cap = settings["max_participation"]
    return CostParams(
        per_share=preset["per_share"],
        min_per_order=preset["min_per_order"],
        max_pct=preset["max_pct"],
        pct=preset["pct"],
        spread=float(settings["spread_bps"]) / 1e4,
        slippage=float(settings["slippage_bps"]) / 1e4,
        impact=float(settings["impact_bps"]) / 1e4,
        impact_exponent=float(settings["impact_exponent"]),
        borrow_rate=float(settings["borrow_rate"]),
        max_participation=math.inf if cap is None else float(cap),
        reject_over_cap=settings["participation_action"] == "reject",
    )


def scale_params(params: CostParams, multiplier: float) -> CostParams:
    """``params`` with every cost rate multiplied by ``multiplier``.

    The commission cap is scaled with the rest. Leaving it alone would let the cap stop
    the commission from growing, so costs would no longer scale linearly. "No cap" stays
    no cap.
    """
    return replace(
        params,
        per_share=params.per_share * multiplier,
        min_per_order=params.min_per_order * multiplier,
        max_pct=params.max_pct if params.max_pct >= NO_CAP else params.max_pct * multiplier,
        pct=params.pct * multiplier,
        spread=params.spread * multiplier,
        slippage=params.slippage * multiplier,
        impact=params.impact * multiplier,
        borrow_rate=params.borrow_rate * multiplier,
    )


class CostModel:
    """Costs for a fixed, ordered list of symbols.

    Array methods take and return one entry per symbol, in the order of ``symbols``; the
    ``order_*`` methods do the same arithmetic for a single order. With ``free=True``
    every cost is zero but the participation cap still applies: that is the "before
    costs" run, which should differ from the real one by costs alone. ``multiplier``
    scales every cost (see the module docstring); ``free`` wins over it.
    """

    def __init__(
        self,
        config: CostConfig,
        symbols: Sequence[str],
        asset_classes: Mapping[str, str] | None = None,
        *,
        free: bool = False,
        multiplier: float = 1.0,
    ) -> None:
        if multiplier < 0:
            raise ValueError(f"cost multiplier must be zero or more, got {multiplier}")
        classes = asset_classes or {}
        self.symbols = list(symbols)
        self.adv_window = config.adv_window
        self.free = free
        self.multiplier = multiplier
        self.params: dict[str, CostParams] = {}
        for symbol in self.symbols:
            params = resolve_params(config, symbol, classes.get(symbol, DEFAULT_ASSET_CLASS))
            if multiplier != 1.0:  # skipped at 1x, so the usual run is bit-for-bit unchanged
                params = scale_params(params, multiplier)
            if free:
                params = replace(
                    params,
                    per_share=0.0,
                    min_per_order=0.0,
                    pct=0.0,
                    spread=0.0,
                    slippage=0.0,
                    impact=0.0,
                    borrow_rate=0.0,
                )
            self.params[symbol] = params

        def column(name: str) -> FloatArray:
            return np.array([getattr(self.params[s], name) for s in self.symbols], dtype="float64")

        self._per_share = column("per_share")
        self._min_per_order = column("min_per_order")
        self._max_pct = column("max_pct")
        self._pct = column("pct")
        self._spread = column("spread")
        self._slippage = column("slippage")
        self._impact = column("impact")
        self._impact_exponent = column("impact_exponent")
        self._borrow_rate = column("borrow_rate")
        self._max_participation = column("max_participation")
        self._reject = np.array([self.params[s].reject_over_cap for s in self.symbols], dtype=bool)
        # A volume-dependent rule cannot be evaluated without a volume history.
        self._needs_volume = np.array([self.params[s].needs_volume for s in self.symbols])
        self._has_impact = bool((self._impact > 0).any())
        self._any_volume_rule = bool(self._needs_volume.any())

    # ------------------------------------------------------------------ arrays

    def trade_costs(
        self, shares: FloatArray, price: FloatArray, adv_value: FloatArray | None = None
    ) -> TradeCosts:
        """Costs of trading ``shares`` (sign ignored) of every symbol at ``price``.

        ``adv_value`` is each symbol's average daily dollar volume; it only matters when
        ``impact_bps`` is set.
        """
        shares = np.abs(shares)
        value = shares * price
        commission = self._per_share * shares + self._pct * value
        commission = np.maximum(np.minimum(commission, self._max_pct * value), self._min_per_order)
        commission = np.where(value > 0, commission, 0.0)
        slippage = value * self._slippage
        if self._has_impact and adv_value is not None:
            participation = value / np.where(adv_value > 0, adv_value, np.inf)
            slippage = slippage + value * (self._impact * participation**self._impact_exponent)
        return TradeCosts(commission=commission, spread=value * self._spread, slippage=slippage)

    def fillable_value(self, value: FloatArray, adv_value: FloatArray | None = None) -> FloatArray:
        """Signed dollar ``value`` of each order after the participation cap.

        A symbol with a volume-dependent rule and no volume history cannot trade.
        """
        if not self._any_volume_rule:
            return value
        adv = np.full(len(self.symbols), np.nan) if adv_value is None else adv_value
        has_volume = adv > 0  # False for NaN: no volume history yet
        cap = self._max_participation * np.where(has_volume, adv, 1.0)
        limit = np.where(self._needs_volume, np.where(has_volume, cap, 0.0), np.inf)
        size = np.abs(value)
        capped = np.where(self._reject, 0.0, np.minimum(size, limit))
        return np.sign(value) * np.where(size > limit, capped, size)

    def borrow_fees(self, short_value: FloatArray, days: float) -> FloatArray:
        """Fee for holding ``short_value`` (positive dollars per symbol) for ``days`` days."""
        return np.abs(short_value) * self._borrow_rate * days / BORROW_DAY_COUNT

    # ------------------------------------------------------------------ single orders

    def order_costs(
        self, symbol: str, shares: float, price: float, adv_value: float = math.nan
    ) -> TradeCosts:
        """Costs of one order of ``shares`` (sign ignored) of ``symbol`` at ``price``."""
        p = self.params[symbol]
        shares = abs(shares)
        value = shares * price
        if not value > 0:
            return TradeCosts(0.0, 0.0, 0.0)
        commission = p.per_share * shares + p.pct * value
        commission = max(min(commission, p.max_pct * value), p.min_per_order)
        slippage = value * p.slippage
        if p.impact > 0 and adv_value > 0:
            slippage = slippage + value * (p.impact * (value / adv_value) ** p.impact_exponent)
        return TradeCosts(commission, value * p.spread, slippage)

    def fillable_shares(
        self, symbol: str, shares: float, price: float, adv_value: float = math.nan
    ) -> float:
        """How many of ``shares`` (positive) the participation cap lets through."""
        p = self.params[symbol]
        if not p.needs_volume:
            return shares
        if not adv_value > 0:
            return 0.0
        limit = p.max_participation * adv_value
        if shares * price <= limit:
            return shares
        return 0.0 if p.reject_over_cap else limit / price

    def borrow_fee(self, symbol: str, short_value: float, days: float) -> float:
        """Fee for holding a short worth ``short_value`` dollars for ``days`` calendar days."""
        return abs(short_value) * self.params[symbol].borrow_rate * days / BORROW_DAY_COUNT
