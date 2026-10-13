"""Trading-session calendars: which days we expect a bar for.

Equities and ETFs follow the NYSE calendar (via ``pandas_market_calendars``); crypto trades
every day and its daily candles are cut at 00:00 UTC.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal

from qr.frames import to_dates

CRYPTO = "crypto"
EXCHANGE_CALENDAR = "NYSE"


@lru_cache(maxsize=1)
def _exchange() -> Any:
    return mcal.get_calendar(EXCHANGE_CALENDAR)


def _schedule(start: date, end: date) -> pd.DataFrame:
    schedule: pd.DataFrame = _exchange().schedule(start_date=start, end_date=end)
    return schedule


def sessions(asset_class: str, start: date, end: date) -> pd.DatetimeIndex:
    """All expected sessions in ``[start, end]`` (inclusive) as tz-naive midnights."""
    if start > end:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    if asset_class == CRYPTO:
        return pd.DatetimeIndex(to_dates(pd.date_range(start, end, freq="D")))
    return pd.DatetimeIndex(to_dates(_schedule(start, end).index))


def last_completed_session(asset_class: str, now: datetime) -> date:
    """The most recent session whose bar is final at ``now`` (naive UTC).

    For exchanges this is the latest session whose closing bell is at or before ``now``.
    For crypto it is yesterday (UTC), because today's candle is still forming.
    """
    if asset_class == CRYPTO:
        return (now - timedelta(days=1)).date()
    schedule = _schedule(now.date() - timedelta(days=14), now.date())
    closes = schedule["market_close"].dt.tz_convert("UTC").dt.tz_localize(None)
    done = schedule.index[(closes <= pd.Timestamp(now)).to_numpy()]
    if len(done) == 0:  # cannot happen with a 14-day lookback, but fail loudly if it does
        raise RuntimeError(f"no completed {EXCHANGE_CALENDAR} session before {now}")
    last: date = done[-1].date()
    return last


def sessions_back(asset_class: str, day: date, n: int) -> date:
    """The session ``n`` sessions before ``day`` (``day`` itself counts as zero if it is one)."""
    if asset_class == CRYPTO:
        return day - timedelta(days=n)
    # 2 calendar days per session plus slack always covers n sessions.
    window = sessions(asset_class, day - timedelta(days=2 * n + 14), day)
    if len(window) <= n:
        raise RuntimeError(f"not enough sessions before {day}")
    back: date = window[-(n + 1)].date()
    return back


def sessions_between(asset_class: str, after: date, until: date) -> int:
    """Number of sessions strictly after ``after`` and up to and including ``until``."""
    return len(sessions(asset_class, after + timedelta(days=1), until))
