"""Cost stress test: the same run at other cost levels, and what that says about it.

The pairs study showed that costs decide everything. So every run is also evaluated with
costs scaled by each multiplier in the config's ``cost_stress`` (by default 0x, 1x, and
2x). 0x is the run before costs; 1x is the run itself; 2x asks whether the result would
survive a broker, spread, or slippage twice as bad as assumed.

From those points come two numbers:

* **Break-even cost multiplier**: where after-cost CAGR reaches zero, by linear
  interpolation between neighbouring stress points. If CAGR stays positive up to the
  largest multiplier tested, it is "above N x"; if it is not positive even at the
  smallest one, the strategy loses money before costs (there is no break-even).
* **Fragile**: Sharpe at 2x is below zero, or below half the Sharpe at 1x. It needs both
  the 1x and the 2x point. A Sharpe that cannot be computed counts as fragile, the more
  conservative reading.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

FRAGILE_MULTIPLIER = 2.0
FRAGILE_SHARE = 0.5  # Sharpe at 2x must keep at least this share of the 1x Sharpe

# Which side of the tested range the break-even lies on, when it is not inside it.
INSIDE = "inside"
ABOVE = "above"  # still profitable at the largest multiplier tested
NONE = "none"  # not profitable even at the smallest multiplier tested


@dataclass(frozen=True)
class StressPoint:
    """The run's headline numbers with costs scaled by ``multiplier``."""

    multiplier: float
    cagr: float
    sharpe: float
    max_drawdown: float
    turnover: float
    costs_paid: float

    @classmethod
    def from_metrics(cls, multiplier: float, metrics: Mapping[str, float]) -> StressPoint:
        """A point from a ``compute_metrics`` dictionary."""
        return cls(
            multiplier=float(multiplier),
            cagr=float(metrics.get("cagr", math.nan)),
            sharpe=float(metrics.get("sharpe", math.nan)),
            max_drawdown=float(metrics.get("max_drawdown", math.nan)),
            turnover=float(metrics.get("turnover", math.nan)),
            costs_paid=float(metrics.get("total_costs", math.nan)),
        )


@dataclass(frozen=True)
class BreakEven:
    """The break-even cost multiplier, or which side of the tested range it lies on."""

    where: str  # INSIDE | ABOVE | NONE
    multiplier: float | None  # the interpolated value when INSIDE, else None
    tested_up_to: float

    def describe(self) -> str:
        """For reports: e.g. ``3.4x``, ``above 2x``, or a note that there is none."""
        if self.where == INSIDE and self.multiplier is not None:
            return f"{self.multiplier:.2f}x"
        if self.where == ABOVE:
            return f"above {self.tested_up_to:g}x"
        return "none: CAGR is not positive even at the lowest cost tested"


def break_even(points: Sequence[StressPoint]) -> BreakEven | None:
    """Where after-cost CAGR reaches zero; None with no usable point."""
    usable = sorted((p for p in points if math.isfinite(p.cagr)), key=lambda p: p.multiplier)
    if not usable:
        return None
    top = usable[-1].multiplier
    if usable[0].cagr <= 0:
        return BreakEven(NONE, None, top)
    for low, high in pairwise(usable):
        if high.cagr <= 0:
            share = low.cagr / (low.cagr - high.cagr)
            return BreakEven(
                INSIDE, low.multiplier + share * (high.multiplier - low.multiplier), top
            )
    return BreakEven(ABOVE, None, top)


def is_fragile(points: Sequence[StressPoint]) -> bool | None:
    """Sharpe at 2x below 0 or below half the 1x Sharpe; None without both points."""
    by_multiplier = {p.multiplier: p for p in points}
    base, stressed = by_multiplier.get(1.0), by_multiplier.get(FRAGILE_MULTIPLIER)
    if base is None or stressed is None:
        return None
    if not (math.isfinite(base.sharpe) and math.isfinite(stressed.sharpe)):
        return True
    return stressed.sharpe < 0 or stressed.sharpe < FRAGILE_SHARE * base.sharpe


@dataclass(frozen=True)
class CostSensitivity:
    """All stress points of a run, in multiplier order, and the two verdicts."""

    points: tuple[StressPoint, ...]
    break_even: BreakEven | None
    fragile: bool | None

    @classmethod
    def from_points(cls, points: Sequence[StressPoint]) -> CostSensitivity:
        ordered = tuple(sorted(points, key=lambda p: p.multiplier))
        return cls(ordered, break_even(ordered), is_fragile(ordered))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dictionary (NaN becomes None)."""

        def clean(value: Any) -> Any:
            return None if isinstance(value, float) and not math.isfinite(value) else value

        return {
            "points": [{k: clean(v) for k, v in asdict(p).items()} for p in self.points],
            "break_even": None if self.break_even is None else asdict(self.break_even),
            "fragile": self.fragile,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CostSensitivity:
        """The inverse of :meth:`to_dict`."""

        def number(value: Any) -> float:
            return math.nan if value is None else float(value)

        points = tuple(
            StressPoint(**{k: number(v) for k, v in point.items()}) for point in raw["points"]
        )
        found = raw.get("break_even")
        return cls(
            points,
            None if found is None else BreakEven(**found),
            raw.get("fragile"),
        )
