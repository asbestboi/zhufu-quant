"""Dataset fingerprints: one SHA-256 that names the exact data a result came from.

The fingerprint covers the *unadjusted* bars and the corporate actions, which is all the
stored information a backtest reads (adjusted prices are a deterministic function of
them). It is computed on a canonical byte form, so it does not depend on query row order,
on the DataFrame index, or on the platform:

* symbols are hashed one at a time and combined in sorted order;
* within a symbol, bars are sorted by date and actions by date then type;
* each symbol's bytes start with a header naming the format version, the symbol, and its
  row counts, so the boundary between bars and actions is unambiguous;
* dates are int64 nanoseconds since 1970-01-01, little-endian;
* each value column (``open, high, low, close, volume``, and an action's ``value``) is
  encoded as one byte per row (1 = NaN, else 0) followed by float64 little-endian bytes,
  with NaN written as 0.0 and -0.0 as 0.0, so NaN payloads and signed zeros cannot make
  equal data hash differently;
* an action's type is one byte (1 = split, 2 = dividend).

The combined hash also includes the adjustment mode the data was loaded with, since the
same rows adjusted differently are a different input to a strategy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from qr.frames import DIVIDEND, SPLIT, VALUE_COLUMNS, to_dates

FORMAT_VERSION = "qr-fingerprint-v1"
ACTION_CODES = {SPLIT: 1, DIVIDEND: 2}


@dataclass(frozen=True)
class DataFingerprint:
    """What :func:`qr.load.fingerprint` returns.

    ``rows`` counts bars (corporate actions are hashed but not counted); ``first_date``
    and ``last_date`` are the first and last bar over all symbols; ``latest_ingested_at``
    is when the newest of those bars was written (naive UTC). Dates are None when there
    are no bars.
    """

    hash: str
    per_symbol: dict[str, str] = field(default_factory=dict)
    rows: int = 0
    first_date: date | None = None
    last_date: date | None = None
    latest_ingested_at: datetime | None = None

    def changed_symbols(self, other: DataFingerprint) -> list[str]:
        """Symbols whose hash differs between the two, or that only one of them has."""
        names = sorted(set(self.per_symbol) | set(other.per_symbol))
        return [s for s in names if self.per_symbol.get(s) != other.per_symbol.get(s)]


def _dates(values: pd.Series) -> bytes:
    days = to_dates(values).to_numpy(dtype="datetime64[ns]")
    return days.astype("<i8").tobytes()


def _floats(values: pd.Series) -> bytes:
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", copy=True)
    missing = np.isnan(numbers)
    numbers[missing] = 0.0
    numbers += 0.0  # -0.0 becomes 0.0
    return missing.astype("u1").tobytes() + numbers.astype("<f8").tobytes()


def hash_symbol(symbol: str, bars: pd.DataFrame, actions: pd.DataFrame) -> str:
    """SHA-256 of one symbol's unadjusted bars and corporate actions, in canonical form.

    ``bars`` needs ``date`` and the value columns; ``actions`` needs ``date, action,
    value``. Other columns and the index are ignored.
    """
    bars = bars.assign(_day=to_dates(bars["date"]).to_numpy())
    bars = bars.sort_values("_day", kind="stable").reset_index(drop=True)
    actions = actions.assign(
        _day=to_dates(actions["date"]).to_numpy(),
        _code=actions["action"].map(ACTION_CODES),
    )
    if actions["_code"].isna().any():
        unknown = sorted(set(actions.loc[actions["_code"].isna(), "action"].astype(str)))
        raise ValueError(f"unknown corporate action types: {unknown}")
    actions = actions.sort_values(["_day", "_code"], kind="stable").reset_index(drop=True)

    digest = hashlib.sha256()
    header = f"{FORMAT_VERSION}|{symbol}|bars={len(bars)}|actions={len(actions)}\n"
    digest.update(header.encode("utf-8"))
    digest.update(_dates(bars["_day"]))
    for column in VALUE_COLUMNS:
        digest.update(_floats(bars[column]))
    digest.update(_dates(actions["_day"]))
    digest.update(actions["_code"].to_numpy(dtype="u1").tobytes())
    digest.update(_floats(actions["value"]))
    return digest.hexdigest()


def combine(per_symbol: dict[str, str], adjustment: str) -> str:
    """The dataset hash: the per-symbol hashes in symbol order, plus the adjustment."""
    digest = hashlib.sha256(f"{FORMAT_VERSION}|adjustment={adjustment}\n".encode())
    for symbol in sorted(per_symbol):
        digest.update(f"{symbol}\0{per_symbol[symbol]}\n".encode())
    return digest.hexdigest()
