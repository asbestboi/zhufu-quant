"""CryptoSource against a fake ccxt exchange: no network."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from qr.sources.base import SourceError
from qr.sources.crypto import DAY_MS, CryptoSource, _to_ms

FIRST = date(2024, 1, 1)


def candles(count: int) -> list[list[float]]:
    """``count`` daily candles starting 2024-01-01, close = 100 + day index."""
    start = _to_ms(FIRST)
    return [
        [start + i * DAY_MS, 99.0 + i, 101.0 + i, 98.0 + i, 100.0 + i, 10.0 + i]
        for i in range(count)
    ]


class FakeExchange:
    """Serves at most ``page`` candles per call, starting at ``since`` like most exchanges."""

    def __init__(self, data: list[list[float]], page: int, honours_since: bool = True) -> None:
        self.data = data
        self.page = page
        self.honours_since = honours_since
        self.calls: list[int] = []

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int | None
    ) -> list[list[float]]:
        assert timeframe == "1d"
        self.calls.append(since)
        if not self.honours_since:
            return self.data[-self.page :]  # e.g. Kraken: always the most recent window
        return [c for c in self.data if c[0] >= since][: self.page]


def source_for(exchange: Any) -> CryptoSource:
    return CryptoSource(client=exchange, max_retries=0, sleep=lambda _: None)


def test_paginates_until_the_end_date() -> None:
    exchange = FakeExchange(candles(10), page=4)
    bars = source_for(exchange).fetch_daily("BTC/USD", FIRST, date(2024, 1, 10))
    assert len(bars) == 10
    assert bars["date"].dt.day.tolist() == list(range(1, 11))  # weekends included
    assert bars["close"].tolist() == [100.0 + i for i in range(10)]
    assert bars["volume"].iloc[0] == 10.0
    # Three pages: each request starts the day after the newest candle seen so far.
    assert exchange.calls == [_to_ms(FIRST), _to_ms(date(2024, 1, 5)), _to_ms(date(2024, 1, 9))]


def test_result_is_limited_to_the_requested_range() -> None:
    exchange = FakeExchange(candles(10), page=100)
    bars = source_for(exchange).fetch_daily("BTC/USD", date(2024, 1, 3), date(2024, 1, 6))
    assert bars["date"].dt.day.tolist() == [3, 4, 5, 6]
    assert len(exchange.calls) == 1


def test_stops_when_history_runs_out() -> None:
    exchange = FakeExchange(candles(5), page=4)
    bars = source_for(exchange).fetch_daily("BTC/USD", FIRST, date(2024, 3, 1))
    assert len(bars) == 5
    assert len(exchange.calls) == 3  # the third page is empty


def test_exchange_that_ignores_since_terminates() -> None:
    exchange = FakeExchange(candles(10), page=4, honours_since=False)
    bars = source_for(exchange).fetch_daily("BTC/USD", FIRST, date(2024, 1, 10))
    # Only the recent window is available; the caller sees history starting later.
    assert bars["date"].dt.day.tolist() == [7, 8, 9, 10]


def test_no_candles_is_an_error() -> None:
    with pytest.raises(SourceError, match="no candles"):
        source_for(FakeExchange([], page=4)).fetch_daily("BTC/USD", FIRST, date(2024, 1, 10))


def test_no_corporate_actions() -> None:
    actions = source_for(FakeExchange([], page=4)).fetch_corporate_actions("BTC/USD")
    assert actions.empty
    assert list(actions.columns) == ["date", "action", "value"]
