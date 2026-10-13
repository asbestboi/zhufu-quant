"""Optional integration test against the real Yahoo API. Run with ``pytest --run-network``."""

from __future__ import annotations

from datetime import date

import pytest

from qr import validate
from qr.config import ValidationSettings
from qr.sources.yahoo import YahooSource
from qr.storage import utcnow


@pytest.mark.network
def test_yahoo_returns_unadjusted_aapl_around_its_2014_split() -> None:
    source = YahooSource(max_retries=2)
    bars = source.fetch_daily("AAPL", date(2014, 6, 2), date(2014, 6, 13))
    actions = source.fetch_corporate_actions("AAPL")

    assert len(bars) == 10
    closes = bars.set_index("date")["close"]
    # As traded: about 645 the day before the 7:1 split and about 94 the day of it.
    assert closes["2014-06-06"] == pytest.approx(645.57, abs=0.01)
    assert closes["2014-06-09"] == pytest.approx(93.70, abs=0.01)

    splits = actions[actions["action"] == "split"].set_index("date")["value"]
    assert splits["2014-06-09"] == 7.0
    dividends = actions[actions["action"] == "dividend"].set_index("date")["value"]
    assert dividends["2014-05-08"] == pytest.approx(3.29, abs=0.001)  # as paid, pre-split

    issues = validate.run_checks(
        "AAPL", bars, actions, "equity", ValidationSettings(), utcnow(), check_staleness=False
    )
    assert issues == []
