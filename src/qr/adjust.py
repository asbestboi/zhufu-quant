"""Split and total-return adjustment using the standard backward method.

Backward adjustment keeps the most recent bar unchanged and rescales earlier bars so
that the series is continuous across corporate actions:

* **Split** with ratio ``r`` on ex-date ``t``: every price before ``t`` is divided by
  ``r`` and every volume before ``t`` is multiplied by ``r``.
* **Cash dividend** ``D`` with ex-date ``t``: every price before ``t`` is multiplied by
  ``1 - D / C`` where ``C`` is the close of the last session before ``t``. This is the
  factor that makes the adjusted close-to-close return over the ex-date equal the
  total return ``(close_t + D) / C - 1``.

Volumes are adjusted for splits only (the usual convention).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from qr.frames import DIVIDEND, PRICE_COLUMNS, SPLIT

logger = logging.getLogger(__name__)

Adjustment = Literal["none", "split", "total_return"]


def split_factors(dates: pd.Series, actions: pd.DataFrame) -> np.ndarray:
    """Price multiplier per date that removes splits: ``1 / (product of later ratios)``.

    A split counts as "later" when its ex-date is strictly after the bar's date.
    """
    factors = np.ones(len(dates), dtype="float64")
    day_values = dates.to_numpy()
    splits = actions[actions["action"] == SPLIT]
    for split_date, ratio in zip(
        splits["date"].to_numpy(), splits["value"].to_numpy(), strict=True
    ):
        if not ratio > 0:
            raise ValueError(f"invalid split ratio {ratio} on {pd.Timestamp(split_date).date()}")
        factors[day_values < split_date] /= ratio
    return factors


def dividend_factors(bars: pd.DataFrame, actions: pd.DataFrame) -> np.ndarray:
    """Price multiplier per date that reinvests dividends (on top of the split factor).

    The dividend and the previous close are compared on the same split basis, so a
    dividend and a split sharing an ex-date are handled correctly. Dividends with no
    earlier bar are ignored (there is nothing before them to adjust), and a dividend
    at least as large as the previous close is skipped with a warning because it can
    only be bad data.
    """
    dates = bars["date"]
    day_values = dates.to_numpy()
    factors = np.ones(len(bars), dtype="float64")
    dividends = actions[actions["action"] == DIVIDEND]
    if dividends.empty or bars.empty:
        return factors
    split_adjusted_close = bars["close"].to_numpy(dtype="float64") * split_factors(dates, actions)
    dividend_split_factor = split_factors(dividends["date"], actions)
    for ex_date, cash, split_factor in zip(
        dividends["date"].to_numpy(),
        dividends["value"].to_numpy(dtype="float64"),
        dividend_split_factor,
        strict=True,
    ):
        before = day_values < ex_date
        if not before.any():
            continue
        previous_close = split_adjusted_close[before][-1]
        adjusted_cash = cash * split_factor
        if not 0 <= adjusted_cash < previous_close:
            logger.warning(
                "ignoring dividend %.6f on %s: not smaller than the previous close %.6f",
                adjusted_cash,
                pd.Timestamp(ex_date).date(),
                previous_close,
            )
            continue
        factors[before] *= 1.0 - adjusted_cash / previous_close
    return factors


def adjust_bars(
    bars: pd.DataFrame, actions: pd.DataFrame, adjustment: Adjustment = "total_return"
) -> pd.DataFrame:
    """Adjust one symbol's unadjusted bars.

    ``bars`` needs ``date, open, high, low, close, volume`` sorted by date and ``actions``
    needs ``date, action, value``; other columns pass through untouched. The input frames
    are not modified. Only actions the caller passes in are applied, so filter ``actions``
    first to control what information the adjustment may use.
    """
    if adjustment not in ("none", "split", "total_return"):
        raise ValueError(f"unknown adjustment: {adjustment!r}")
    out = bars.copy()
    if adjustment == "none" or out.empty or actions.empty:
        return out
    if not out["date"].is_monotonic_increasing:
        raise ValueError("bars must be sorted by date before adjusting")

    split = split_factors(out["date"], actions)
    price = split.copy()
    if adjustment == "total_return":
        price = price * dividend_factors(out, actions)
    for column in PRICE_COLUMNS:
        out[column] = out[column].to_numpy(dtype="float64") * price
    out["volume"] = out["volume"].to_numpy(dtype="float64") / split
    return out
