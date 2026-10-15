"""The event engine's data path: read-only history arrays, history_df, and session bars."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qr.backtest.data import MarketData
from qr.backtest.events import Bar
from qr.backtest.strategy import Context, EventStrategy

from .bt_helpers import FIVE_DAYS, bars, make_config, run

# Every test here runs on both event-engine cores (see the engine_core fixture).
pytestmark = pytest.mark.usefixtures("engine_core")

# BBB has no bar on Jan 4: its history has a gap, and that session has no BBB bar.
GAPPY = [row for row in FIVE_DAYS if row[0] != "2024-01-04"]


class Recorder(EventStrategy):
    """Calls ``look(ctx, bars)`` on every session and keeps what it returns."""

    def __init__(self, look: Any) -> None:
        self.look = look
        self.seen: list[Any] = []

    def on_bar(self, ctx: Context, bars: Mapping[str, Bar]) -> None:
        self.seen.append(self.look(ctx, bars))


def record(look: Any, data: MarketData | None = None) -> list[Any]:
    data = data or MarketData.from_frames({"AAA": bars(FIVE_DAYS), "BBB": bars(GAPPY)})
    strategy = Recorder(look)
    run(data, make_config(data), strategy)
    return strategy.seen


def test_history_is_a_read_only_array() -> None:
    def look(ctx: Context, _: Mapping[str, Bar]) -> None:
        for adjusted in (True, False):
            values = ctx.history("AAA", 3, adjusted=adjusted)
            assert isinstance(values, np.ndarray) and values.dtype == np.float64
            assert not values.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                values[0] = 0.0
        assert not ctx.history("AAA", 0).flags.writeable  # also when empty

    record(look)


def test_unadjusted_history_without_gaps_is_a_view_not_a_copy() -> None:
    def look(ctx: Context, _: Mapping[str, Bar]) -> tuple[bool, bool]:
        return (
            ctx.history("AAA", 3, adjusted=False).flags.owndata,
            ctx.history("AAA", 3, field="volume").flags.owndata,
        )

    assert record(look) == [(False, False)] * 5


def test_history_ends_at_the_current_session() -> None:
    def look(ctx: Context, bars: Mapping[str, Bar]) -> tuple[int, bool, bool]:
        raw = ctx.history("AAA", 1000, adjusted=False)
        dates = ctx.history_df("AAA", 1000).index
        return len(raw), raw[-1] == bars["AAA"].close, dates[-1] == pd.Timestamp(ctx.now)

    # One more bar each day, the last one always today's: nothing from later sessions.
    assert record(look) == [(n, True, True) for n in range(1, 6)]


@pytest.mark.parametrize("symbol", ["AAA", "BBB"])
@pytest.mark.parametrize("adjusted", [True, False])
def test_history_df_matches_history(symbol: str, adjusted: bool) -> None:
    def look(ctx: Context, _: Mapping[str, Bar]) -> None:
        for field in ("open", "high", "low", "close", "volume"):
            values = ctx.history(symbol, 3, field=field, adjusted=adjusted)
            frame = ctx.history_df(symbol, 3, field=field, adjusted=adjusted)
            assert isinstance(frame, pd.Series) and frame.name == symbol
            assert frame.to_numpy().tolist() == values.tolist()
            frame.iloc[0] = -1.0  # a copy: writing to it leaves the engine's data alone
            assert ctx.history(symbol, 3, field=field, adjusted=adjusted)[0] != -1.0

    record(look)


def test_history_skips_the_missing_session() -> None:
    def look(ctx: Context, _: Mapping[str, Bar]) -> list[str]:
        return [f"{d:%m-%d}" for d in ctx.history_df("BBB", 10).index]

    assert record(look)[-1] == ["01-02", "01-03", "01-05", "01-08"]


def test_session_bars_behave_like_a_dict_of_todays_bars() -> None:
    def look(ctx: Context, bars: Mapping[str, Bar]) -> dict[str, Any]:
        return {
            "keys": list(bars),
            "len": len(bars),
            "has_bbb": "BBB" in bars,
            "unknown": "ZZZ" in bars,
            "bars": dict(bars),
        }

    seen = record(look)
    assert [s["keys"] for s in seen] == [["AAA", "BBB"]] * 2 + [["AAA"]] + [["AAA", "BBB"]] * 2
    assert [s["len"] for s in seen] == [2, 2, 1, 2, 2]
    assert [s["has_bbb"] for s in seen] == [True, True, False, True, True]
    assert not any(s["unknown"] for s in seen)
    day, open_, high, low, close = FIVE_DAYS[1]
    assert seen[1]["bars"]["AAA"] == Bar(
        "AAA", pd.Timestamp(day).to_pydatetime(), open_, high, low, close, 1000.0
    )


def test_session_bars_raise_key_error_for_a_symbol_without_a_bar() -> None:
    def look(ctx: Context, bars: Mapping[str, Bar]) -> None:
        if "BBB" not in bars:
            with pytest.raises(KeyError):
                bars["BBB"]
        with pytest.raises(KeyError):
            bars["ZZZ"]

    record(look)
