"""Adjustment math against values worked out by hand.

Fixture: closes 100, 102, 51, 52, 50 on Jan 2, 3, 4, 5, 8 (2024), with a 2:1 split on
Jan 4 and a 1.00 dividend going ex on Jan 8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qr.adjust import adjust_bars, dividend_factors, split_factors
from qr.frames import empty_actions, normalize_actions


def actions_of(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return normalize_actions(pd.DataFrame(rows, columns=["date", "action", "value"]))


def test_none_returns_bars_unchanged(
    split_dividend_bars: pd.DataFrame, split_dividend_actions: pd.DataFrame
) -> None:
    out = adjust_bars(split_dividend_bars, split_dividend_actions, "none")
    pd.testing.assert_frame_equal(out, split_dividend_bars)
    assert out is not split_dividend_bars


def test_split_adjustment(
    split_dividend_bars: pd.DataFrame, split_dividend_actions: pd.DataFrame
) -> None:
    out = adjust_bars(split_dividend_bars, split_dividend_actions, "split")
    # Bars before the Jan 4 split are halved; volume doubles.
    assert out["close"].tolist() == [50.0, 51.0, 51.0, 52.0, 50.0]
    assert out["open"].tolist() == [49.5, 50.5, 50.5, 51.5, 50.5]
    assert out["high"].tolist() == [50.5, 51.5, 52.0, 53.0, 51.0]
    assert out["low"].tolist() == [49.0, 50.0, 50.0, 51.0, 49.0]
    assert out["volume"].tolist() == [2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
    # The input is never modified.
    assert split_dividend_bars["close"].tolist() == [100.0, 102.0, 51.0, 52.0, 50.0]


def test_total_return_adjustment(
    split_dividend_bars: pd.DataFrame, split_dividend_actions: pd.DataFrame
) -> None:
    out = adjust_bars(split_dividend_bars, split_dividend_actions, "total_return")
    # Dividend factor = 1 - 1.00 / 52 (the close before the ex-date) = 51/52, applied to
    # every bar before Jan 8 on top of the split factor.
    f = 51.0 / 52.0
    np.testing.assert_allclose(out["close"], [50 * f, 51 * f, 51 * f, 52 * f, 50.0], rtol=1e-12)
    np.testing.assert_allclose(out["open"], [49.5 * f, 50.5 * f, 50.5 * f, 51.5 * f, 50.5])
    # Volume is adjusted for the split only.
    assert out["volume"].tolist() == [2000.0] * 5
    # Over the ex-date the adjusted return is close / (previous close - dividend) - 1.
    assert out["close"].iloc[4] / out["close"].iloc[3] - 1 == pytest.approx(50 / (52 - 1) - 1)


def test_dividend_only(split_dividend_bars: pd.DataFrame) -> None:
    bars = split_dividend_bars.iloc[2:].reset_index(drop=True)  # 51, 52, 50
    out = adjust_bars(bars, actions_of(("2024-01-08", "dividend", 1.0)), "total_return")
    np.testing.assert_allclose(out["close"], [51 * 51 / 52, 51.0, 50.0], rtol=1e-12)
    # A split-only adjustment ignores dividends.
    split_only = adjust_bars(bars, actions_of(("2024-01-08", "dividend", 1.0)), "split")
    assert split_only["close"].tolist() == [51.0, 52.0, 50.0]


def test_split_and_dividend_on_the_same_day(split_dividend_bars: pd.DataFrame) -> None:
    # 2:1 split and a 0.51 dividend (in post-split dollars) both on Jan 4. The previous
    # close of 102 is 51 post-split, so the dividend factor is 1 - 0.51/51 = 0.99.
    actions = actions_of(("2024-01-04", "split", 2.0), ("2024-01-04", "dividend", 0.51))
    out = adjust_bars(split_dividend_bars, actions, "total_return")
    np.testing.assert_allclose(out["close"], [50 * 0.99, 51 * 0.99, 51.0, 52.0, 50.0], rtol=1e-12)


def test_multiple_splits_compound(split_dividend_bars: pd.DataFrame) -> None:
    actions = actions_of(("2024-01-03", "split", 3.0), ("2024-01-05", "split", 2.0))
    factors = split_factors(split_dividend_bars["date"], actions)
    np.testing.assert_allclose(factors, [1 / 6, 1 / 2, 1 / 2, 1.0, 1.0])


def test_actions_outside_the_bars(split_dividend_bars: pd.DataFrame) -> None:
    # A dividend before the first bar has nothing to adjust; one after the last bar
    # adjusts everything.
    early = actions_of(("2023-12-01", "dividend", 1.0))
    np.testing.assert_array_equal(dividend_factors(split_dividend_bars, early), np.ones(5))
    late = actions_of(("2024-02-01", "dividend", 0.5))
    np.testing.assert_allclose(dividend_factors(split_dividend_bars, late), [0.99] * 5)


def test_impossible_dividend_is_ignored(split_dividend_bars: pd.DataFrame) -> None:
    actions = actions_of(("2024-01-08", "dividend", 60.0))  # more than the 52 close
    np.testing.assert_array_equal(dividend_factors(split_dividend_bars, actions), np.ones(5))


def test_no_actions_and_bad_input(split_dividend_bars: pd.DataFrame) -> None:
    out = adjust_bars(split_dividend_bars, empty_actions(), "total_return")
    pd.testing.assert_frame_equal(out, split_dividend_bars)
    with pytest.raises(ValueError, match="unknown adjustment"):
        adjust_bars(split_dividend_bars, empty_actions(), "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sorted"):
        adjust_bars(
            split_dividend_bars.iloc[::-1], actions_of(("2024-01-04", "split", 2.0)), "split"
        )
    with pytest.raises(ValueError, match="invalid split ratio"):
        adjust_bars(split_dividend_bars, actions_of(("2024-01-04", "split", 0.0)), "split")
