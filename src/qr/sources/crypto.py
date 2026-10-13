"""Crypto daily candles through ``ccxt`` (optional: install with ``uv sync --extra crypto``).

Candles are cut at 00:00 UTC and markets trade every day. There are no corporate actions.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from qr.frames import BAR_COLUMNS, empty_actions, normalize_bars
from qr.sources.base import DataSource, SourceError, with_retries

DAY_MS = 24 * 60 * 60 * 1000
MAX_PAGES = 10_000  # safety net against an exchange that never stops returning data


def _to_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def _default_client(exchange: str) -> Any:
    try:
        import ccxt
    except ImportError as exc:
        raise SourceError(
            "the crypto source needs ccxt; install it with `uv sync --extra crypto`"
        ) from exc
    if not hasattr(ccxt, exchange):
        raise SourceError(f"ccxt has no exchange named '{exchange}'")
    return getattr(ccxt, exchange)({"enableRateLimit": True})


class CryptoSource(DataSource):
    """Daily OHLCV candles from one ccxt exchange."""

    name = "crypto"

    def __init__(
        self,
        exchange: str = "kraken",
        max_retries: int = 5,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        client: Any = None,
        page_limit: int | None = None,
    ) -> None:
        self._exchange = exchange
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._client = client
        self._page_limit = page_limit

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _default_client(self._exchange)
        return self._client

    def _page(self, symbol: str, since_ms: int) -> list[list[float]]:
        def call() -> list[list[float]]:
            page: list[list[float]] = self._get_client().fetch_ohlcv(
                symbol, timeframe="1d", since=since_ms, limit=self._page_limit
            )
            return page

        return with_retries(
            call,
            max_retries=self._max_retries,
            backoff_seconds=self._backoff_seconds,
            sleep=self._sleep,
            what=f"{self._exchange} candles {symbol}",
        )

    def probe(self) -> None:
        """Ask the exchange for its market list, once, without retries."""
        try:
            self._get_client().load_markets()
        except SourceError:
            raise
        except Exception as exc:
            raise SourceError(f"{self._exchange} did not answer a test request: {exc}") from exc

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Daily candles for ``[start, end]``, following pagination until ``end`` is reached.

        Some exchanges only serve a recent window regardless of ``since`` (Kraken: the
        last 720 candles), so the result may begin later than ``start``.
        """
        start_ms, end_ms = _to_ms(start), _to_ms(end)
        candles: dict[int, list[float]] = {}
        since = start_ms
        for _ in range(MAX_PAGES):
            page = self._page(symbol, since)
            fresh = [c for c in page if int(c[0]) >= since and int(c[0]) not in candles]
            if not fresh:
                break  # nothing new: either the end of history or an exchange ignoring `since`
            for candle in fresh:
                candles[int(candle[0])] = candle
            newest = max(int(c[0]) for c in fresh)
            if newest >= end_ms:
                break
            since = newest + DAY_MS

        rows = [c for ts, c in sorted(candles.items()) if start_ms <= ts <= end_ms]
        if not rows:
            raise SourceError(
                f"{self._exchange} returned no candles for {symbol} between {start} and {end}"
            )
        frame = pd.DataFrame([r[:6] for r in rows], columns=BAR_COLUMNS)
        epoch = datetime(1970, 1, 1)
        frame["date"] = [epoch + timedelta(milliseconds=int(ts)) for ts in frame["date"]]
        return normalize_bars(frame)

    def fetch_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """Crypto has no splits or dividends."""
        return empty_actions()
