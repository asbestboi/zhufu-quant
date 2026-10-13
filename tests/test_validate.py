"""Every check against bars_dirty.csv, which contains one of each problem, and against
bars_clean.csv, which must produce nothing."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime

import pandas as pd
import pytest

from qr import storage, validate
from qr.config import UniverseConfig, ValidationSettings
from qr.frames import empty_actions, normalize_actions
from qr.storage import QualityIssue

from .conftest import AFTER_JAN_16

SETTINGS = ValidationSettings()
AFTER_JAN_31 = datetime(2024, 2, 1, 0, 0, 0)


def dates(issues: list[QualityIssue]) -> list[date | None]:
    return [i.date for i in issues]


def test_clean_data_has_no_findings(clean_bars: pd.DataFrame) -> None:
    issues = validate.run_checks(
        "AAA", clean_bars, empty_actions(), "equity", SETTINGS, AFTER_JAN_16
    )
    assert issues == []


def test_ohlc_consistency(dirty_bars: pd.DataFrame) -> None:
    issues = validate.check_ohlc_consistency("AAA", dirty_bars)
    assert dates(issues) == [date(2024, 1, 3)]
    assert issues[0].severity == "error"
    assert "high=99.5" in issues[0].details


def test_ohlc_consistency_low_above_open_and_float_noise(clean_bars: pd.DataFrame) -> None:
    bars = clean_bars.copy()
    bars.loc[0, "low"] = bars.loc[0, "open"] + 0.25
    bars.loc[1, "high"] = bars.loc[1, "close"] * (1 - 1e-8)  # float32 noise is tolerated
    assert dates(validate.check_ohlc_consistency("AAA", bars)) == [date(2024, 1, 2)]


def test_non_positive_prices(dirty_bars: pd.DataFrame) -> None:
    assert dates(validate.check_non_positive_prices("AAA", dirty_bars)) == [date(2024, 1, 4)]


def test_negative_volume(dirty_bars: pd.DataFrame) -> None:
    issues = validate.check_negative_volume("AAA", dirty_bars)
    assert dates(issues) == [date(2024, 1, 5)]
    assert issues[0].details == "volume=-5.0"


def test_duplicates(dirty_bars: pd.DataFrame) -> None:
    issues = validate.check_duplicates("AAA", dirty_bars)
    assert dates(issues) == [date(2024, 1, 8)]
    assert issues[0].details == "2 bars for the same date"


def test_missing_values(dirty_bars: pd.DataFrame) -> None:
    issues = validate.check_missing_values("AAA", dirty_bars)
    assert dates(issues) == [date(2024, 1, 26)]
    assert issues[0].details == "missing: close"


def test_gaps_are_grouped_into_runs(dirty_bars: pd.DataFrame) -> None:
    issues = validate.check_gaps("AAA", dirty_bars, "equity")
    assert dates(issues) == [date(2024, 1, 9)]
    assert issues[0].severity == "warning"
    assert issues[0].details == "2 expected sessions missing through 2024-01-10"


def test_gaps_for_crypto_expect_every_day(clean_bars: pd.DataFrame) -> None:
    # The clean equity fixture skips two weekends and a holiday: gaps for a 24/7 market.
    issues = validate.check_gaps("BTC/USD", clean_bars, "crypto")
    assert dates(issues) == [date(2024, 1, 6), date(2024, 1, 13)]
    assert issues[0].details == "2 expected sessions missing through 2024-01-07"
    assert issues[1].details == "3 expected sessions missing through 2024-01-15"
    assert validate.check_unexpected_sessions("BTC/USD", clean_bars, "crypto") == []


def test_unexpected_sessions(dirty_bars: pd.DataFrame) -> None:
    issues = validate.check_unexpected_sessions("AAA", dirty_bars, "equity")
    assert dates(issues) == [date(2024, 1, 13)]  # a Saturday


def test_outlier_returns(dirty_bars: pd.DataFrame) -> None:
    issues = validate.check_outlier_returns("AAA", dirty_bars, empty_actions(), 0.25)
    assert dates(issues) == [date(2024, 1, 17)]
    assert "+50.00%" in issues[0].details
    # A higher threshold lets it through.
    assert validate.check_outlier_returns("AAA", dirty_bars, empty_actions(), 0.60) == []


def test_outlier_returns_ignore_known_splits(
    split_dividend_bars: pd.DataFrame, split_dividend_actions: pd.DataFrame
) -> None:
    # Raw close goes 102 -> 51 on the split date: -50%, but fully explained by the split.
    assert (
        validate.check_outlier_returns("AAA", split_dividend_bars, split_dividend_actions, 0.25)
        == []
    )
    # Without the split on record the same move is flagged.
    unknown = validate.check_outlier_returns("AAA", split_dividend_bars, empty_actions(), 0.25)
    assert dates(unknown) == [date(2024, 1, 4)]
    # A split on record that the prices do not reflect is flagged too.
    phantom = normalize_actions(
        pd.DataFrame([("2024-01-05", "split", 2.0)], columns=["date", "action", "value"])
    )
    flagged = validate.check_outlier_returns("AAA", split_dividend_bars, phantom, 0.25)
    assert date(2024, 1, 5) in dates(flagged)


def test_stale_data(clean_bars: pd.DataFrame) -> None:
    # Last bar Jan 16. On Jan 17 (before the close) nothing is missing yet.
    assert validate.check_stale("AAA", clean_bars, "equity", 5, AFTER_JAN_16) == []
    # Five sessions later (Jan 17, 18, 19, 22, 23) is still within the limit...
    assert validate.check_stale("AAA", clean_bars, "equity", 5, datetime(2024, 1, 24, 0, 0)) == []
    # ...the sixth is not.
    issues = validate.check_stale("AAA", clean_bars, "equity", 5, datetime(2024, 1, 25, 0, 0))
    assert dates(issues) == [date(2024, 1, 16)]
    assert issues[0].details == "last bar is 6 sessions behind 2024-01-24"
    empty = validate.check_stale("AAA", clean_bars.iloc[:0], "equity", 5, AFTER_JAN_16)
    assert empty[0].details == "no bars stored"


def test_zero_volume_streaks(dirty_bars: pd.DataFrame) -> None:
    issues = validate.check_zero_volume_streaks("AAA", dirty_bars, max_streak=3)
    assert dates(issues) == [date(2024, 1, 22)]
    assert issues[0].details == "4 consecutive zero-volume bars through 2024-01-25"
    # A streak exactly at the limit is allowed.
    assert validate.check_zero_volume_streaks("AAA", dirty_bars, max_streak=4) == []


def test_run_checks_reports_every_problem_in_the_dirty_fixture(dirty_bars: pd.DataFrame) -> None:
    issues = validate.run_checks(
        "AAA", dirty_bars, empty_actions(), "equity", SETTINGS, AFTER_JAN_31
    )
    expected = (set(validate.STATE_CHECKS) | {validate.DUPLICATE_CHECK}) - {"stale_data"}
    assert {i.check_name for i in issues} == expected
    late = validate.run_checks(
        "AAA", dirty_bars, empty_actions(), "equity", SETTINGS, datetime(2024, 3, 1)
    )
    assert "stale_data" in {i.check_name for i in late}


def test_validate_symbols_replaces_findings_and_keeps_revisions(
    con: storage.Connection, make_config: Callable[..., UniverseConfig], dirty_bars: pd.DataFrame
) -> None:
    config = make_config("AAA", "OLD")
    stored = dirty_bars.dropna().drop_duplicates("date")
    for symbol, active in (("AAA", True), ("OLD", False)):
        storage.upsert_instrument(con, symbol, "yahoo", "equity", active, AFTER_JAN_31)
        storage.upsert_bars(con, symbol, "yahoo", stored, AFTER_JAN_31)
    revision = QualityIssue("AAA", date(2024, 1, 5), validate.REVISION_CHECK, "warning", "x")
    storage.insert_quality_issues(con, [revision], AFTER_JAN_31)

    late = datetime(2024, 3, 1)
    first = validate.validate_symbols(con, config, now=late)
    second = validate.validate_symbols(con, config, now=late)
    assert first == second

    issues = storage.read_quality_issues(con)
    # Running twice does not duplicate findings, and the revision history survives.
    assert len(issues) == len(first) + 1
    assert (issues["check_name"] == validate.REVISION_CHECK).sum() == 1
    by_symbol = {s: set(g["check_name"]) for s, g in issues.groupby("symbol")}
    assert "stale_data" in by_symbol["AAA"]
    assert "stale_data" not in by_symbol["OLD"]  # inactive symbols are allowed to stop

    summary = validate.summarize(storage.read_quality_issues(con, ["AAA"]))
    gap = summary[summary["check_name"] == "gap"].iloc[0]
    # Two gaps: Jan 9-10, and Jan 26 whose incomplete row was not stored.
    assert (gap["symbol"], gap["severity"], gap["count"]) == ("AAA", "warning", 2)
    assert (gap["first_date"].date(), gap["last_date"].date()) == (
        date(2024, 1, 9),
        date(2024, 1, 26),
    )
    assert validate.summarize(issues.iloc[:0]).empty


def test_validate_symbols_rejects_unknown_symbol(
    con: storage.Connection, make_config: Callable[..., UniverseConfig]
) -> None:
    with pytest.raises(ValueError, match="unknown symbols: NOPE"):
        validate.validate_symbols(con, make_config("AAA"), ["NOPE"])
