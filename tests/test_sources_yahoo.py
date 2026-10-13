"""YahooSource against a fake ``yfinance.Ticker``: no network.

Yahoo serves prices, volumes, and dividends already adjusted for later splits, so the
fake returns the split-adjusted view of bars_split_dividend.csv and the source must give
back the as-traded values in that fixture.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from qr.adjust import adjust_bars
from qr.frames import BAR_COLUMNS
from qr.sources.base import SourceError, with_retries
from qr.sources.yahoo import YahooSource


def yahoo_frame(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """What Yahoo would return for these as-traded bars and as-paid actions."""
    adjusted = adjust_bars(bars, actions, "split")
    frame = pd.DataFrame(
        {
            "Open": adjusted["open"].to_numpy(),
            "High": adjusted["high"].to_numpy(),
            "Low": adjusted["low"].to_numpy(),
            "Close": adjusted["close"].to_numpy(),
            "Adj Close": adjusted["close"].to_numpy(),
            "Volume": adjusted["volume"].to_numpy(),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=pd.DatetimeIndex(bars["date"]).tz_localize("America/New_York"),
    )
    frame.loc["2024-01-03", "Dividends"] = 0.4  # 0.80 paid before the 2:1 split
    frame.loc["2024-01-04", "Stock Splits"] = 2.0
    frame.loc["2024-01-08", "Dividends"] = 1.0
    return frame


class FakeTicker:
    """Stands in for ``yfinance.Ticker``; fails ``failures`` times before answering."""

    def __init__(self, frame: pd.DataFrame, failures: int = 0) -> None:
        self.frame = frame
        self.failures = failures
        self.requests: list[dict[str, Any]] = []

    def history(self, **kwargs: Any) -> pd.DataFrame:
        self.requests.append(kwargs)
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("rate limited")
        if kwargs.get("period") == "max":
            return self.frame
        days = self.frame.index.tz_localize(None)
        start, end = pd.Timestamp(kwargs["start"]), pd.Timestamp(kwargs["end"])
        return self.frame[(days >= start) & (days < end)]  # yfinance: end is exclusive


@pytest.fixture
def ticker(split_dividend_bars: pd.DataFrame, split_dividend_actions: pd.DataFrame) -> FakeTicker:
    return FakeTicker(yahoo_frame(split_dividend_bars, split_dividend_actions))


def source_for(ticker: FakeTicker, sleeps: list[float] | None = None, **kwargs: Any) -> YahooSource:
    recorded = sleeps if sleeps is not None else []
    return YahooSource(sleep=recorded.append, ticker_factory=lambda symbol: ticker, **kwargs)


def test_fetch_daily_undoes_yahoos_split_adjustment(
    ticker: FakeTicker, split_dividend_bars: pd.DataFrame
) -> None:
    bars = source_for(ticker).fetch_daily("AAA", date(2024, 1, 1), date(2024, 1, 31))
    pd.testing.assert_frame_equal(bars[BAR_COLUMNS], split_dividend_bars)
    # The provider's own numbers are kept alongside for the raw snapshot.
    assert bars["yahoo_close"].tolist() == [50.0, 51.0, 51.0, 52.0, 50.0]
    assert bars["yahoo_volume"].tolist() == [2000.0] * 5
    assert "yahoo_adj_close" in bars.columns


def test_fetch_daily_requests_unadjusted_data_with_inclusive_end(ticker: FakeTicker) -> None:
    bars = source_for(ticker).fetch_daily("AAA", date(2024, 1, 3), date(2024, 1, 5))
    assert [d.day for d in bars["date"]] == [3, 4, 5]
    request = ticker.requests[-1]
    assert request["auto_adjust"] is False
    assert request["start"] == date(2024, 1, 3)
    assert request["end"] == date(2024, 1, 6)


def test_corporate_actions_are_as_paid(ticker: FakeTicker) -> None:
    actions = source_for(ticker).fetch_corporate_actions("AAA")
    assert [(d.day, a, v) for d, a, v in actions.itertuples(index=False)] == [
        (3, "dividend", 0.8),  # Yahoo's 0.40 times the later 2:1 split
        (4, "split", 2.0),
        (8, "dividend", 1.0),
    ]


def test_capital_gains_count_as_dividends(ticker: FakeTicker) -> None:
    ticker.frame["Capital Gains"] = 0.0
    ticker.frame.loc["2024-01-08", "Capital Gains"] = 0.25
    actions = source_for(ticker).fetch_corporate_actions("AAA")
    assert actions[actions["date"] == "2024-01-08"]["value"].tolist() == [1.25]


def test_full_history_is_requested_once_per_symbol(ticker: FakeTicker) -> None:
    source = source_for(ticker)
    source.fetch_daily("AAA", date(2024, 1, 1), date(2024, 1, 31))
    source.fetch_corporate_actions("AAA")
    assert [r.get("period") for r in ticker.requests].count("max") == 1


def test_retries_with_exponential_backoff(ticker: FakeTicker) -> None:
    ticker.failures = 3
    sleeps: list[float] = []
    source = source_for(ticker, sleeps, max_retries=5, backoff_seconds=1.0)
    assert not source.fetch_corporate_actions("AAA").empty
    assert sleeps == [1.0, 2.0, 4.0]


def test_gives_up_after_max_retries(ticker: FakeTicker) -> None:
    ticker.failures = 99
    sleeps: list[float] = []
    source = source_for(ticker, sleeps, max_retries=2)
    with pytest.raises(SourceError, match="failed after 3 attempts: rate limited"):
        source.fetch_daily("AAA", date(2024, 1, 1), date(2024, 1, 31))
    assert len(sleeps) == 2


def test_empty_response_is_an_error(ticker: FakeTicker) -> None:
    source = source_for(ticker, max_retries=1)
    with pytest.raises(SourceError, match="no rows"):
        source.fetch_daily("AAA", date(2023, 1, 1), date(2023, 1, 31))
    empty = FakeTicker(ticker.frame.iloc[:0])
    with pytest.raises(SourceError, match="no rows"):
        source_for(empty, max_retries=0).fetch_corporate_actions("AAA")


def test_backoff_is_capped() -> None:
    sleeps: list[float] = []

    def always_fails() -> None:
        raise ValueError("nope")

    with pytest.raises(SourceError):
        with_retries(always_fails, max_retries=8, backoff_seconds=1.0, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
