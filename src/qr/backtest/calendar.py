"""Rebalance schedules: on which sessions a strategy may change its target portfolio.

A rebalance date is a *decision* date. The decision uses data up to that session's close
and is executed on the following session, so the schedule itself never needs the future.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd

from qr.frames import to_dates

BoolArray = npt.NDArray[np.bool_]


def rebalance_mask(sessions: pd.DatetimeIndex, schedule: str | Sequence[date]) -> BoolArray:
    """True for every session in ``sessions`` that is a rebalance decision date.

    * ``"daily"``: every session.
    * ``"weekly"`` / ``"monthly"``: the last session of each week / month, so the trade
      happens on the first session of the next one. The first session is always included
      (the initial allocation), and a trailing week or month that is still running at the
      end of ``sessions`` has no decision date yet.
    * a list of dates: exactly those; a date that is not a session moves to the next
      session, never to an earlier one.
    """
    count = len(sessions)
    if isinstance(schedule, str):
        if schedule == "daily":
            return np.ones(count, dtype=bool)
        if schedule not in ("weekly", "monthly"):
            raise ValueError(f"unknown rebalance schedule: {schedule!r}")
        periods = sessions.to_period("W" if schedule == "weekly" else "M")
        mask = np.zeros(count, dtype=bool)
        mask[:-1] = periods[:-1] != periods[1:]
        if count:
            mask[0] = True
        return mask
    wanted = pd.DatetimeIndex(to_dates(list(schedule)))
    positions = sessions.searchsorted(wanted, side="left")
    mask = np.zeros(count, dtype=bool)
    mask[positions[positions < count]] = True
    return mask


def rebalance_dates(sessions: pd.DatetimeIndex, schedule: str | Sequence[date]) -> pd.DatetimeIndex:
    """The rebalance decision dates among ``sessions`` (see :func:`rebalance_mask`)."""
    return sessions[rebalance_mask(sessions, schedule)]
