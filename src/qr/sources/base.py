"""Abstract data source interface plus the retry helper every source shares."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 60.0


class SourceError(Exception):
    """A provider failed, or returned nothing usable, after all retries."""


class DataSource(ABC):
    """A provider of daily bars and corporate actions.

    Implementations must return **unadjusted** (as-traded) prices and raise
    :class:`SourceError` rather than return an empty frame from ``fetch_daily``.
    """

    name: str

    @abstractmethod
    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Unadjusted daily bars for ``[start, end]`` (both inclusive).

        Columns: ``date, open, high, low, close, volume`` (see ``qr.frames``). Extra
        columns are allowed and end up only in the raw snapshot.
        """

    @abstractmethod
    def fetch_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """Full known history of splits and dividends.

        Columns: ``date, action, value`` where ``action`` is ``'split'`` (value = new
        shares per old share, e.g. 2.0 for a 2:1 split) or ``'dividend'`` (value = cash
        per share as actually paid, i.e. not split-adjusted). May be empty.
        """


def with_retries[T](
    call: Callable[[], T],
    *,
    max_retries: int,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    what: str = "request",
) -> T:
    """Run ``call``, retrying with exponential backoff; raise SourceError when exhausted.

    ``max_retries`` is the number of retries after the first attempt, and the wait before
    retry ``k`` (1-based) is ``backoff_seconds * 2**(k-1)``, capped at one minute.
    """
    attempts = max_retries + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(backoff_seconds * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                what,
                attempt,
                attempts,
                exc,
                delay,
            )
            sleep(delay)
    raise SourceError(f"{what} failed after {attempts} attempts: {last_error}") from last_error
