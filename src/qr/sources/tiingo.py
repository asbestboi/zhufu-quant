"""Tiingo end-of-day prices (https://www.tiingo.com), a second opinion on Yahoo's data.

This source is only used by ``qr crosscheck``; it is never the primary source and its bars
are never stored in ``daily_bars``. It needs a (free) API token in ``TIINGO_API_KEY``. The
free tier allows about 50 requests an hour and 1,000 a day, one request per symbol.

Tiingo's ``open/high/low/close/volume`` are as traded, and each row also carries the cash
dividend (``divCash``) and split ratio (``splitFactor``) that took effect that day.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd

from qr.frames import DIVIDEND, SPLIT, normalize_actions, normalize_bars
from qr.sources.base import (
    DataSource,
    PermanentSourceError,
    SourceError,
    SourceUnavailableError,
    with_retries,
)

API_URL = "https://api.tiingo.com"
KEY_VARIABLE = "TIINGO_API_KEY"
TIMEOUT_SECONDS = 20
EARLIEST = date(1962, 1, 1)  # Tiingo's history does not go back further than this

_TIINGO_COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "adjClose": "tiingo_adj_close",
    "divCash": "div_cash",
    "splitFactor": "split_factor",
}

FetchJson = Callable[[str, str], Any]


def _http_get_json(url: str, token: str) -> Any:
    """GET ``url`` with the token in a header (so it never appears in a URL or a log)."""
    request = urllib.request.Request(
        url, headers={"Authorization": f"Token {token}", "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise SourceUnavailableError(
                f"Tiingo rejected the API key (HTTP {exc.code}); check {KEY_VARIABLE}"
            ) from exc
        if exc.code == 404:
            raise PermanentSourceError("Tiingo does not know this ticker (HTTP 404)") from exc
        if exc.code == 429:
            raise SourceUnavailableError(
                "Tiingo's rate limit was reached (free tier: about 50 requests an hour)"
            ) from exc
        raise
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        # Tiingo reports some problems (e.g. an exhausted allowance) as plain text.
        raise SourceUnavailableError(f"Tiingo answered with: {body[:200].strip()}") from exc


def _numeric_column(rows: pd.DataFrame, name: str, default: float) -> pd.Series:
    """A response column as floats, with ``default`` where it is absent or not a number."""
    if name not in rows.columns:
        return pd.Series(default, index=rows.index)
    return pd.to_numeric(rows[name], errors="coerce").fillna(default).astype("float64")


class TiingoSource(DataSource):
    """Daily bars, splits, and dividends from Tiingo's end-of-day API."""

    name = "tiingo"

    def __init__(
        self,
        api_key: str,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        fetch_json: FetchJson = _http_get_json,
    ) -> None:
        if not api_key:
            raise SourceError(f"the Tiingo source needs an API key ({KEY_VARIABLE})")
        self._api_key = api_key
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._fetch_json = fetch_json
        # The latest price rows per symbol; corporate actions are read from them.
        self._rows: dict[str, pd.DataFrame] = {}

    def probe(self) -> None:
        """Call Tiingo's test endpoint once. It answers even with a bad key, so this
        only shows that Tiingo is reachable; a rejected key surfaces on the first fetch."""
        try:
            self._fetch_json(f"{API_URL}/api/test", self._api_key)
        except SourceError:
            raise
        except Exception as exc:
            raise SourceError(f"Tiingo did not answer a test request: {exc}") from exc

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """As-traded daily bars for ``[start, end]``.

        Extra columns for the raw snapshot: ``tiingo_adj_close``, ``div_cash``,
        ``split_factor``.
        """
        ticker = urllib.parse.quote(symbol, safe="")
        query = urllib.parse.urlencode({"startDate": start, "endDate": end, "format": "json"})
        url = f"{API_URL}/tiingo/daily/{ticker}/prices?{query}"
        rows = with_retries(
            lambda: self._fetch_json(url, self._api_key),
            max_retries=self._max_retries,
            backoff_seconds=self._backoff_seconds,
            sleep=self._sleep,
            what=f"tiingo bars {symbol}",
        )
        if not isinstance(rows, list):
            detail = rows.get("detail", rows) if isinstance(rows, dict) else rows
            raise SourceError(f"unexpected Tiingo response for {symbol}: {detail}")
        if not rows:
            raise SourceError(f"Tiingo returned no rows for {symbol} between {start} and {end}")

        raw = pd.DataFrame(rows)
        missing = [c for c in ("date", "open", "high", "low", "close", "volume") if c not in raw]
        if missing:
            raise SourceError(f"Tiingo response for {symbol} is missing columns: {missing}")
        present = {old: new for old, new in _TIINGO_COLUMNS.items() if old in raw.columns}
        frame = raw[["date", *present]].rename(columns=present)
        bars = normalize_bars(frame)
        self._rows[symbol] = bars
        return bars

    def fetch_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """Splits and as-paid dividends found in the price rows.

        Tiingo has no separate corporate-actions endpoint, so this covers the window of
        the latest ``fetch_daily`` call for the symbol (or, if there was none, its whole
        history, at the cost of one more request).
        """
        if symbol not in self._rows:
            self.fetch_daily(symbol, EARLIEST, date.today())
        rows = self._rows[symbol]
        splits = _numeric_column(rows, "split_factor", default=1.0)
        cash = _numeric_column(rows, "div_cash", default=0.0)
        is_split = ((splits > 0) & (splits != 1.0)).to_numpy()
        is_dividend = (cash > 0).to_numpy()
        dates = rows["date"].to_numpy()
        parts = [
            pd.DataFrame(
                {"date": dates[is_split], "action": SPLIT, "value": splits.to_numpy()[is_split]}
            ),
            pd.DataFrame(
                {
                    "date": dates[is_dividend],
                    "action": DIVIDEND,
                    "value": cash.to_numpy()[is_dividend],
                }
            ),
        ]
        return normalize_actions(pd.concat(parts, ignore_index=True))
