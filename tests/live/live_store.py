"""Seeding a real DuckDB store for qr.live tests that read through ``qr.load``."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qr import storage
from qr.calendars import sessions as exchange_sessions
from qr.config import db_path

NOW = datetime(2026, 1, 1)


def seed_store(data_dir: Path, bars: dict[str, pd.DataFrame]) -> Path:
    """Write per-symbol unadjusted bars (as :func:`..bt_helpers.synthetic_bars` makes
    them) into a fresh DuckDB store at ``data_dir``, with no corporate actions."""
    with storage.connect(db_path(data_dir)) as con:
        storage.init_schema(con)
        for symbol, frame in bars.items():
            storage.upsert_instrument(con, symbol, "yahoo", "etf", True, NOW)
            storage.upsert_bars(con, symbol, "yahoo", frame, NOW)
    return data_dir


def real_session_bars(
    start: date, end: date, seed: int = 0, drift: float = 0.0004, volatility: float = 0.012
) -> pd.DataFrame:
    """A random-walk symbol's unadjusted bars on the *real* NYSE calendar.

    Unlike :func:`..bt_helpers.synthetic_bars` (plain business days), this uses
    :func:`qr.calendars.sessions`, so its last bar's date can be checked against
    :func:`qr.live.cycle.is_month_end` consistently -- both read the same calendar.
    """
    dates = exchange_sessions("etf", start, end)
    rng = np.random.default_rng(seed)
    close = np.empty(len(dates))
    open_ = np.empty(len(dates))
    price = 100.0
    for i in range(len(dates)):
        open_[i] = price * np.exp(rng.normal(0.0, 0.004))
        price = price * np.exp(rng.normal(drift, volatility))
        close[i] = price
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": 1_000_000.0,
        }
    )
