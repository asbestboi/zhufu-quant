"""TiingoSource against canned JSON: no network."""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from datetime import date
from typing import Any

import pandas as pd
import pytest

from qr.frames import BAR_COLUMNS
from qr.sources.base import PermanentSourceError, SourceError, SourceUnavailableError
from qr.sources.tiingo import TiingoSource, _http_get_json


def row(day: str, close: float, split: float = 1.0, dividend: float = 0.0) -> dict[str, Any]:
    """One row of Tiingo's ``/tiingo/daily/{ticker}/prices`` response."""
    return {
        "date": f"{day}T00:00:00.000Z",
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000,
        "adjClose": close / 2,
        "adjOpen": 0.0,
        "adjHigh": 0.0,
        "adjLow": 0.0,
        "adjVolume": 2000,
        "divCash": dividend,
        "splitFactor": split,
    }


ROWS = [
    row("2024-01-02", 100.0),
    row("2024-01-03", 102.0, dividend=0.8),
    row("2024-01-04", 51.0, split=2.0),
    row("2024-01-05", 52.0),
]


class FakeApi:
    """Stands in for the HTTP call; raises the queued errors before answering."""

    def __init__(self, answer: Any, errors: list[Exception] | None = None) -> None:
        self.answer = answer
        self.errors = errors or []
        self.requests: list[tuple[str, str]] = []

    def __call__(self, url: str, token: str) -> Any:
        self.requests.append((url, token))
        if self.errors:
            raise self.errors.pop(0)
        return self.answer


def source_for(api: FakeApi, sleeps: list[float] | None = None) -> TiingoSource:
    recorded = sleeps if sleeps is not None else []
    return TiingoSource("secret-token", sleep=recorded.append, fetch_json=api)


def test_fetch_daily_parses_the_response() -> None:
    api = FakeApi(ROWS)
    bars = source_for(api).fetch_daily("BRK-B", date(2024, 1, 1), date(2024, 1, 31))

    assert bars[BAR_COLUMNS].iloc[0].tolist() == [
        pd.Timestamp("2024-01-02"), 99.5, 101.0, 99.0, 100.0, 1000.0,
    ]  # fmt: skip
    assert bars["date"].dtype == "datetime64[ns]"
    assert bars["close"].tolist() == [100.0, 102.0, 51.0, 52.0]
    assert bars["tiingo_adj_close"].tolist() == [50.0, 51.0, 25.5, 26.0]
    url, token = api.requests[0]
    assert url == (
        "https://api.tiingo.com/tiingo/daily/BRK-B/prices"
        "?startDate=2024-01-01&endDate=2024-01-31&format=json"
    )
    assert token == "secret-token" and "secret-token" not in url


def test_corporate_actions_come_from_the_price_rows() -> None:
    api = FakeApi(ROWS)
    source = source_for(api)
    source.fetch_daily("AAA", date(2024, 1, 1), date(2024, 1, 31))
    actions = source.fetch_corporate_actions("AAA")

    assert [(d.day, a, v) for d, a, v in actions.itertuples(index=False)] == [
        (3, "dividend", 0.8),
        (4, "split", 2.0),
    ]
    assert len(api.requests) == 1  # no second request

    # Without an earlier fetch it asks for the whole history.
    fresh = source_for(api)
    assert len(fresh.fetch_corporate_actions("AAA")) == 2
    assert "startDate=1962-01-01" in api.requests[-1][0]


def test_transient_errors_are_retried_and_permanent_ones_are_not() -> None:
    sleeps: list[float] = []
    api = FakeApi(ROWS, errors=[TimeoutError("slow"), ConnectionError("reset")])
    assert len(source_for(api, sleeps).fetch_daily("AAA", date(2024, 1, 1), date(2024, 1, 31))) == 4
    assert sleeps == [1.0, 2.0]

    sleeps.clear()
    api = FakeApi(ROWS, errors=[PermanentSourceError("Tiingo rejected the API key (HTTP 401)")])
    with pytest.raises(PermanentSourceError, match="rejected the API key"):
        source_for(api, sleeps).fetch_daily("AAA", date(2024, 1, 1), date(2024, 1, 31))
    assert sleeps == [] and len(api.requests) == 1

    api = FakeApi(ROWS, errors=[OSError("down")] * 3)
    with pytest.raises(SourceError, match="failed after 3 attempts"):
        source_for(api, sleeps).fetch_daily("AAA", date(2024, 1, 1), date(2024, 1, 31))


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ([], "no rows"),
        ({"detail": "Not found."}, "unexpected Tiingo response for AAA: Not found."),
        ([{"date": "2024-01-02T00:00:00.000Z", "close": 1.0}], "missing columns"),
    ],
)
def test_unusable_responses_raise(answer: Any, expected: str) -> None:
    with pytest.raises(SourceError, match=expected):
        source_for(FakeApi(answer)).fetch_daily("AAA", date(2024, 1, 1), date(2024, 1, 31))


def test_probe_and_missing_key() -> None:
    api = FakeApi({"message": "You successfully sent a request"})
    source_for(api).probe()
    assert api.requests == [("https://api.tiingo.com/api/test", "secret-token")]

    with pytest.raises(SourceError, match="did not answer a test request: unplugged"):
        source_for(FakeApi(None, errors=[OSError("unplugged")])).probe()
    with pytest.raises(PermanentSourceError):
        source_for(FakeApi(None, errors=[PermanentSourceError("bad key")])).probe()
    with pytest.raises(SourceError, match="TIINGO_API_KEY"):
        TiingoSource("")


@pytest.mark.parametrize(
    ("code", "expected"),
    [(401, "rejected the API key"), (404, "does not know this ticker"), (429, "rate limit")],
)
def test_http_errors_that_retrying_cannot_fix(
    monkeypatch: pytest.MonkeyPatch, code: int, expected: str
) -> None:
    # Only an unknown ticker (404) is about one symbol; the others condemn the whole source.
    kind = PermanentSourceError if code == 404 else SourceUnavailableError

    def refuse(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.HTTPError(request.full_url, code, "nope", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(kind, match=expected) as caught:
        _http_get_json("https://api.tiingo.com/api/test", "secret-token")
    assert type(caught.value) is kind


def test_http_layer_sends_the_token_in_a_header(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[urllib.request.Request] = []
    bodies = [b'{"message": "ok"}', b"Error: over your hourly request allocation"]

    def answer(request: urllib.request.Request, timeout: float) -> Any:
        seen.append(request)
        return io.BytesIO(bodies.pop(0))

    monkeypatch.setattr(urllib.request, "urlopen", answer)
    assert _http_get_json("https://api.tiingo.com/api/test", "secret-token") == {"message": "ok"}
    assert seen[0].get_header("Authorization") == "Token secret-token"
    with pytest.raises(SourceUnavailableError, match="hourly request allocation"):
        _http_get_json("https://api.tiingo.com/api/test", "secret-token")

    def server_error(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", server_error)
    with pytest.raises(urllib.error.HTTPError):  # transient: left for with_retries
        _http_get_json("https://api.tiingo.com/api/test", "secret-token")
