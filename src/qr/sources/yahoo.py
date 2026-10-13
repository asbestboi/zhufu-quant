"""Yahoo Finance source (via the unofficial ``yfinance`` package).

Yahoo never serves truly unadjusted prices: even with ``auto_adjust=False`` its OHLC,
volume, and dividend amounts are already adjusted for every split that happened since.
We undo that here using Yahoo's own split list so that what we store is what actually
traded on the day, and all adjustment logic lives in one place (``qr.adjust``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from qr.frames import (
    ACTION_COLUMNS,
    DIVIDEND,
    PRICE_COLUMNS,
    SPLIT,
    empty_actions,
    normalize_actions,
    normalize_bars,
    to_dates,
)
from qr.sources.base import DataSource, SourceError, with_retries

PROBE_SYMBOL = "SPY"  # a ticker that will always exist, for the connectivity probe
PRICE_DECIMALS = 6  # enough for any quoted price; removes float noise from un-adjusting
_YAHOO_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def _default_ticker(symbol: str) -> Any:
    import yfinance as yf

    return yf.Ticker(symbol)


def _numeric_column(history: pd.DataFrame, name: str) -> pd.Series:
    """A provider column as floats, with zeros where it is absent or not a number."""
    if name not in history.columns:
        return pd.Series(0.0, index=history.index)
    return pd.to_numeric(history[name], errors="coerce").fillna(0.0).astype("float64")


def split_factor_after(dates: pd.Series, splits: pd.DataFrame) -> np.ndarray:
    """For each date, the product of all split ratios with an ex-date strictly after it.

    Multiplying a Yahoo (split-adjusted) price by this factor gives the as-traded price.
    """
    factors = np.ones(len(dates), dtype="float64")
    day_values = dates.to_numpy()
    for split_date, ratio in zip(
        splits["date"].to_numpy(), splits["value"].to_numpy(), strict=True
    ):
        factors[day_values < split_date] *= ratio
    return factors


class YahooSource(DataSource):
    """Daily bars, splits, and dividends from Yahoo Finance."""

    name = "yahoo"

    def __init__(
        self,
        max_retries: int = 5,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        ticker_factory: Callable[[str], Any] = _default_ticker,
    ) -> None:
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._ticker_factory = ticker_factory
        # Yahoo-basis actions per symbol, so one ingest makes one full-history request.
        self._actions_cache: dict[str, pd.DataFrame] = {}

    def _history(self, symbol: str, what: str, **kwargs: Any) -> pd.DataFrame:
        """One ``Ticker.history`` call with retries; an empty answer counts as a failure."""

        def call() -> pd.DataFrame:
            frame: pd.DataFrame = self._ticker_factory(symbol).history(
                interval="1d", auto_adjust=False, actions=True, raise_errors=True, **kwargs
            )
            if frame is None or frame.empty:
                raise SourceError(f"Yahoo returned no rows for {symbol}")
            return frame

        return with_retries(
            call,
            max_retries=self._max_retries,
            backoff_seconds=self._backoff_seconds,
            sleep=self._sleep,
            what=f"yahoo {what} {symbol}",
        )

    def probe(self) -> None:
        """Ask for a few recent SPY bars, once, without retries."""
        try:
            frame = self._ticker_factory(PROBE_SYMBOL).history(
                period="5d", interval="1d", auto_adjust=False, actions=False, raise_errors=True
            )
        except Exception as exc:
            raise SourceError(f"Yahoo did not answer a test request: {exc}") from exc
        if frame is None or frame.empty:
            raise SourceError("Yahoo answered a test request with no data")

    def _yahoo_actions(self, symbol: str) -> pd.DataFrame:
        """Splits and dividends as Yahoo reports them (dividends are split-adjusted)."""
        if symbol in self._actions_cache:
            return self._actions_cache[symbol]
        history = self._history(symbol, "actions", period="max")
        dates = to_dates(history.index)
        parts: list[pd.DataFrame] = []
        splits = _numeric_column(history, "Stock Splits")
        split_rows = (splits > 0).to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "date": dates[split_rows].to_numpy(),
                    "action": SPLIT,
                    "value": splits.to_numpy()[split_rows],
                }
            )
        )
        # Capital-gain distributions (funds/ETFs) are cash paid per share: same as a dividend.
        cash = _numeric_column(history, "Dividends")
        if "Capital Gains" in history.columns:
            cash = cash + _numeric_column(history, "Capital Gains")
        cash_rows = (cash > 0).to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "date": dates[cash_rows].to_numpy(),
                    "action": DIVIDEND,
                    "value": cash.to_numpy()[cash_rows],
                }
            )
        )
        actions = normalize_actions(pd.concat(parts, ignore_index=True)[ACTION_COLUMNS])
        self._actions_cache[symbol] = actions
        return actions

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Unadjusted daily bars for ``[start, end]``.

        The provider's own values are kept in ``yahoo_*`` columns for the raw snapshot.
        """
        yahoo_actions = self._yahoo_actions(symbol)
        splits = yahoo_actions[yahoo_actions["action"] == SPLIT]
        # yfinance treats `end` as exclusive.
        history = self._history(symbol, "bars", start=start, end=end + timedelta(days=1))
        missing = [c for c in _YAHOO_COLUMNS if c not in history.columns]
        if missing:
            raise SourceError(f"Yahoo response for {symbol} is missing columns: {missing}")

        frame = history[list(_YAHOO_COLUMNS)].rename(columns=_YAHOO_COLUMNS).reset_index(drop=True)
        frame.insert(0, "date", to_dates(history.index).to_numpy())
        for column in [*PRICE_COLUMNS, "volume"]:
            frame[f"yahoo_{column}"] = frame[column].astype("float64")
        if "Adj Close" in history.columns:
            frame["yahoo_adj_close"] = history["Adj Close"].to_numpy(dtype="float64")

        factor = split_factor_after(frame["date"], splits)
        for column in PRICE_COLUMNS:
            frame[column] = (frame[column].astype("float64") * factor).round(PRICE_DECIMALS)
        frame["volume"] = (frame["volume"].astype("float64") / factor).round(0)

        bars = normalize_bars(frame)
        in_range = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
        bars = bars[in_range].reset_index(drop=True)
        if bars.empty:
            raise SourceError(f"Yahoo returned no rows for {symbol} between {start} and {end}")
        return bars

    def fetch_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """Splits and as-paid dividends (Yahoo's split adjustment of dividends is undone)."""
        yahoo_actions = self._yahoo_actions(symbol)
        if yahoo_actions.empty:
            return empty_actions()
        actions = yahoo_actions.copy()
        splits = actions[actions["action"] == SPLIT]
        is_dividend = (actions["action"] == DIVIDEND).to_numpy()
        factor = split_factor_after(actions["date"], splits)
        values = actions["value"].to_numpy(dtype="float64")
        actions["value"] = np.where(is_dividend, (values * factor).round(PRICE_DECIMALS), values)
        return normalize_actions(actions)
