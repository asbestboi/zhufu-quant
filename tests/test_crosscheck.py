"""Cross-checking stored data against a second in-memory source.

AAA is bars_split_dividend.csv (five sessions, 2:1 split on Jan 4) and BBB is
bars_clean.csv (ten sessions, no actions), both ingested from a fake "yahoo".
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from qr import snapshots, storage
from qr.config import UniverseConfig, db_path, raw_dir
from qr.crosscheck import compare_closes, compare_splits, run_crosscheck
from qr.frames import empty_actions
from qr.ingest import run_ingest
from qr.sources.base import SourceError, SourceUnavailableError
from qr.validate import validate_symbols

from .conftest import AFTER_JAN_16, Clock, FakeSource

MakeConfig = Callable[..., UniverseConfig]


@pytest.fixture
def config(make_config: MakeConfig) -> UniverseConfig:
    return make_config("AAA", "BBB")


@pytest.fixture
def store(
    config: UniverseConfig,
    data_dir: Path,
    split_dividend_bars: pd.DataFrame,
    split_dividend_actions: pd.DataFrame,
    clean_bars: pd.DataFrame,
) -> Path:
    yahoo = FakeSource(
        {"AAA": split_dividend_bars, "BBB": clean_bars}, {"AAA": split_dividend_actions}
    )
    run_ingest(
        config, data_dir, sources={"yahoo": yahoo}, clock=Clock(AFTER_JAN_16), sleep=lambda _: None
    )
    return data_dir


@pytest.fixture
def tiingo(
    split_dividend_bars: pd.DataFrame,
    split_dividend_actions: pd.DataFrame,
    clean_bars: pd.DataFrame,
) -> FakeSource:
    """A second source that agrees with the first on everything."""
    return FakeSource(
        {"AAA": split_dividend_bars.copy(), "BBB": clean_bars.copy()},
        {"AAA": split_dividend_actions.copy()},
        name="tiingo",
    )


def check(config: UniverseConfig, data_dir: Path, source: FakeSource, **kwargs: object):  # type: ignore[no-untyped-def]
    return run_crosscheck(
        config,
        data_dir,
        source=source,
        days=30,
        clock=Clock(AFTER_JAN_16),
        sleep=lambda _: None,
        **kwargs,  # type: ignore[arg-type]
    )


def crosscheck_issues(data_dir: Path) -> pd.DataFrame:
    with storage.connect(db_path(data_dir), read_only=True) as con:
        issues = storage.read_quality_issues(con)
    return issues[issues["check_name"] == "crosscheck"].reset_index(drop=True)


def test_agreeing_sources_leave_no_findings(
    config: UniverseConfig, store: Path, tiingo: FakeSource
) -> None:
    report = check(config, store, tiingo)

    assert (report.source, report.start, report.end) == (
        "tiingo",
        date(2023, 12, 17),
        date(2024, 1, 16),
    )
    assert [(r.symbol, r.status, r.days_compared, r.mismatches) for r in report.results] == [
        ("AAA", "ok", 5, 0),
        ("BBB", "ok", 10, 0),
    ]
    assert not report.failed
    assert crosscheck_issues(store).empty
    assert tiingo.calls == [
        ("AAA", date(2023, 12, 17), date(2024, 1, 16)),
        ("BBB", date(2023, 12, 17), date(2024, 1, 16)),
    ]


def test_a_deliberate_close_mismatch_is_recorded(
    config: UniverseConfig, store: Path, tiingo: FakeSource
) -> None:
    bars = tiingo.bars["BBB"]
    bars.loc[bars["date"] == "2024-01-09", "close"] *= 1.02  # 2% off: beyond the 0.5% tolerance
    bars.loc[bars["date"] == "2024-01-10", "close"] *= 1.004  # 0.4% off: tolerated
    report = check(config, store, tiingo)

    result = report.results[1]
    assert (result.symbol, result.status, result.days_compared, result.mismatches) == (
        "BBB",
        "mismatch",
        10,
        1,
    )
    assert result.worst_difference == pytest.approx(0.02)
    assert result.worst_date == date(2024, 1, 9)
    assert not report.failed  # a disagreement is a finding, not a failed run

    found = crosscheck_issues(store)
    assert len(found) == 1
    row = found.iloc[0]
    assert (row["symbol"], row["severity"], row["date"].date()) == (
        "BBB",
        "warning",
        date(2024, 1, 9),
    )
    assert row["details"] == "close yahoo 102.5 vs tiingo 104.55 (+2.00%)"

    # Running again replaces the finding instead of adding a second copy.
    check(config, store, tiingo)
    assert len(crosscheck_issues(store)) == 1
    # Once the sources agree again, the finding goes away.
    bars.loc[bars["date"] == "2024-01-09", "close"] /= 1.02
    check(config, store, tiingo)
    assert crosscheck_issues(store).empty


def test_mismatched_splits_are_recorded(
    config: UniverseConfig, store: Path, tiingo: FakeSource
) -> None:
    tiingo.actions["AAA"] = empty_actions()  # the second source knows of no split
    report = check(config, store, tiingo, symbols=["AAA"])

    assert [(r.status, r.mismatches) for r in report.results] == [("mismatch", 1)]
    found = crosscheck_issues(store)
    assert found["details"].tolist() == ["split yahoo 2 vs tiingo none"]
    assert found["date"].iloc[0].date() == date(2024, 1, 4)


def test_validate_keeps_crosscheck_findings(
    config: UniverseConfig, store: Path, tiingo: FakeSource
) -> None:
    tiingo.actions["AAA"] = empty_actions()
    check(config, store, tiingo)
    with storage.connect(db_path(store)) as con:
        validate_symbols(con, config, now=AFTER_JAN_16)
    assert len(crosscheck_issues(store)) == 1


def test_responses_are_snapshotted_without_touching_stored_bars(
    config: UniverseConfig, store: Path, tiingo: FakeSource
) -> None:
    tiingo.bars["BBB"].loc[0, "close"] = 1.0
    with storage.connect(db_path(store), read_only=True) as con:
        before = storage.read_bars(con, ["AAA", "BBB"])
    check(config, store, tiingo)
    with storage.connect(db_path(store), read_only=True) as con:
        after = storage.read_bars(con, ["AAA", "BBB"])
    pd.testing.assert_frame_equal(before, after)

    taken = snapshots.list_snapshots(raw_dir(store), "tiingo", "BBB")
    assert len(taken) == 1
    assert pd.read_parquet(taken[0].bars_path)["close"].iloc[0] == 1.0
    assert len(snapshots.list_snapshots(raw_dir(store), "yahoo", "BBB")) == 1  # untouched


def test_failures_and_skips(
    make_config: MakeConfig, store: Path, tiingo: FakeSource, clean_bars: pd.DataFrame
) -> None:
    config = make_config(
        universe=[
            {"symbol": "AAA", "source": "yahoo", "asset_class": "equity"},
            {"symbol": "BBB", "source": "yahoo", "asset_class": "equity"},
            {"symbol": "NEW", "source": "yahoo", "asset_class": "equity"},
            {"symbol": "BTC/USD", "source": "crypto", "asset_class": "crypto"},
        ]
    )
    del tiingo.bars["AAA"]  # the second source has never heard of AAA
    report = check(config, store, tiingo, symbols=["AAA", "BBB", "NEW", "BTC/USD", "ZZZ"])

    assert {r.symbol: r.status for r in report.results} == {
        "AAA": "failed",
        "BBB": "ok",
        "NEW": "skipped",  # in the config but never ingested
        "BTC/USD": "skipped",
        "ZZZ": "failed",  # not in the config at all
    }
    assert "no data for AAA" in report.results[0].message
    assert report.failed

    # By default only the stock and ETF symbols are checked.
    assert [r.symbol for r in check(config, store, tiingo).results] == ["AAA", "BBB", "NEW"]


def test_unreachable_source_stops_the_run(
    config: UniverseConfig, store: Path, tiingo: FakeSource
) -> None:
    tiingo.dies_after = 0
    with pytest.raises(SourceError, match="connection refused"):
        check(config, store, tiingo)
    assert tiingo.calls == []


def test_compare_closes_counts_unmatched_days(clean_bars: pd.DataFrame) -> None:
    result = compare_closes("AAA", clean_bars, clean_bars.iloc[2:], tolerance=0.005)
    assert (result.days_compared, result.unmatched_days, result.issues) == (8, 2, [])
    nothing = compare_closes("AAA", clean_bars.iloc[:2], clean_bars.iloc[5:], tolerance=0.005)
    assert (nothing.days_compared, nothing.unmatched_days, nothing.worst_date) == (0, 7, None)


def test_compare_splits_only_looks_inside_the_window(
    split_dividend_actions: pd.DataFrame,
) -> None:
    theirs = split_dividend_actions.copy()
    theirs.loc[theirs["action"] == "split", "value"] = 3.0
    found = compare_splits(
        "AAA", split_dividend_actions, theirs, date(2024, 1, 1), date(2024, 1, 31)
    )
    assert [(i.date, i.details) for i in found] == [(date(2024, 1, 4), "split yahoo 2 vs tiingo 3")]
    assert (
        compare_splits("AAA", split_dividend_actions, theirs, date(2024, 1, 5), date(2024, 1, 31))
        == []
    )
    only_theirs = compare_splits(
        "AAA", empty_actions(), theirs, date(2024, 1, 1), date(2024, 1, 31)
    )
    assert [i.details for i in only_theirs] == ["split tiingo 3 vs yahoo none"]


def test_a_rejected_key_stops_the_run_at_the_first_symbol(
    config: UniverseConfig, store: Path, tiingo: FakeSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(symbol: str, start: date, end: date) -> pd.DataFrame:
        tiingo.calls.append((symbol, start, end))
        raise SourceUnavailableError("Tiingo rejected the API key (HTTP 403)")

    monkeypatch.setattr(tiingo, "fetch_daily", reject)
    with pytest.raises(SourceUnavailableError, match="rejected the API key"):
        check(config, store, tiingo)
    assert [c[0] for c in tiingo.calls] == ["AAA"]  # BBB was never asked for
