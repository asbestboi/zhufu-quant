"""Canonical DataFrame shapes shared by sources, storage, and loading.

Bars are always ``date, open, high, low, close, volume`` and corporate actions are
``date, action, value``. Dates are timezone-naive ``datetime64[ns]`` at midnight and mean
"the trading session on that calendar day".
"""

from __future__ import annotations

from typing import Any

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]
VALUE_COLUMNS = [*PRICE_COLUMNS, "volume"]
BAR_COLUMNS = ["date", *VALUE_COLUMNS]
ACTION_COLUMNS = ["date", "action", "value"]

SPLIT = "split"
DIVIDEND = "dividend"


def to_dates(values: Any) -> pd.Series:
    """Convert anything date-like to a tz-naive midnight ``datetime64[ns]`` Series."""
    series = pd.Series(pd.to_datetime(values))
    if series.dt.tz is not None:
        series = series.dt.tz_localize(None)
    return series.dt.normalize().astype("datetime64[ns]")


def to_timestamp(day: Any) -> pd.Timestamp:
    """Convert a single date-like value to a tz-naive midnight Timestamp."""
    return pd.Timestamp(to_dates([day]).iloc[0])


def empty_bars() -> pd.DataFrame:
    """An empty frame with the canonical bar columns and dtypes."""
    return normalize_bars(pd.DataFrame(columns=BAR_COLUMNS))


def empty_actions() -> pd.DataFrame:
    """An empty frame with the canonical corporate-action columns and dtypes."""
    return normalize_actions(pd.DataFrame(columns=ACTION_COLUMNS))


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with canonical bar dtypes, sorted by date.

    Extra columns are kept (sources use them for provider-original values in raw
    snapshots). Duplicate dates are *not* removed here; that is a validation finding.
    """
    missing = [c for c in BAR_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"bars frame is missing columns: {missing}")
    out = frame.reset_index(drop=True).copy()
    out["date"] = to_dates(out["date"]).to_numpy()
    for column in VALUE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    extra = [c for c in out.columns if c not in BAR_COLUMNS]
    out = out[[*BAR_COLUMNS, *extra]]
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def normalize_actions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with canonical corporate-action dtypes, sorted by date and action."""
    missing = [c for c in ACTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"corporate actions frame is missing columns: {missing}")
    out = frame.reset_index(drop=True)[ACTION_COLUMNS].copy()
    out["date"] = to_dates(out["date"]).to_numpy()
    out["action"] = out["action"].astype("object")
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype("float64")
    unknown = sorted(set(out["action"]) - {SPLIT, DIVIDEND})
    if unknown:
        raise ValueError(f"unknown corporate action types: {unknown}")
    return out.sort_values(["date", "action"], kind="stable").reset_index(drop=True)
