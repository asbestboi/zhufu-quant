"""Ingestion against an in-memory source: windows, idempotency, revisions, fault isolation.

bars_clean.csv covers the ten NYSE sessions from 2024-01-02 to 2024-01-16.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qr import snapshots, storage
from qr.config import UniverseConfig, db_path, raw_dir
from qr.ingest import detect_revisions, fetch_window, run_ingest
from qr.sources.base import DataSource

from .conftest import AFTER_JAN_16, Clock, FakeSource

MakeConfig = Callable[..., UniverseConfig]


def ingest(
    config: UniverseConfig,
    data_dir: Path,
    source: DataSource,
    now: datetime = AFTER_JAN_16,
    **kwargs: object,
):  # type: ignore[no-untyped-def]
    return run_ingest(
        config,
        data_dir,
        sources={"yahoo": source},
        clock=Clock(now),
        sleep=lambda _: None,
        **kwargs,
    )  # type: ignore[arg-type]


def query(data_dir: Path, sql: str) -> list[tuple[object, ...]]:
    with storage.connect(db_path(data_dir), read_only=True) as con:
        return con.execute(sql).fetchall()


def stored_bars(data_dir: Path, symbol: str) -> pd.DataFrame:
    with storage.connect(db_path(data_dir), read_only=True) as con:
        return storage.read_bars(con, [symbol])


def issues(data_dir: Path) -> pd.DataFrame:
    with storage.connect(db_path(data_dir), read_only=True) as con:
        return storage.read_quality_issues(con)


# ------------------------------------------------------------------ date window logic


def test_first_run_fetches_from_start_date() -> None:
    window = fetch_window(None, "equity", date(2010, 1, 1), AFTER_JAN_16)
    assert window == (date(2010, 1, 1), date(2024, 1, 16))


def test_incremental_run_overlaps_five_trading_days() -> None:
    # Last stored bar Tue Jan 16; five sessions back is Mon Jan 8 (Jan 15 is a holiday).
    window = fetch_window(date(2024, 1, 16), "equity", date(2010, 1, 1), datetime(2024, 1, 19))
    assert window == (date(2024, 1, 8), date(2024, 1, 18))


def test_window_never_starts_before_start_date_or_includes_a_forming_bar() -> None:
    during_session = datetime(2024, 1, 16, 15, 0)  # 10:00 in New York
    window = fetch_window(date(2024, 1, 12), "equity", date(2024, 1, 10), during_session)
    assert window == (date(2024, 1, 10), date(2024, 1, 12))


def test_crypto_window_uses_calendar_days() -> None:
    window = fetch_window(date(2024, 1, 16), "crypto", date(2010, 1, 1), datetime(2024, 1, 19, 8))
    assert window == (date(2024, 1, 11), date(2024, 1, 18))


# ------------------------------------------------------------------ end-to-end runs


def test_first_run_populates_everything(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    source = FakeSource({"AAA": clean_bars})
    report = ingest(make_config("AAA"), data_dir, source)

    assert not report.failed
    assert [(r.symbol, r.status, r.rows_written) for r in report.results] == [("AAA", "ok", 10)]
    assert source.calls == [("AAA", date(2024, 1, 1), date(2024, 1, 16))]
    pd.testing.assert_frame_equal(stored_bars(data_dir, "AAA")[clean_bars.columns], clean_bars)

    assert query(data_dir, "SELECT symbol, first_date, last_date, is_active FROM instruments") == [
        ("AAA", date(2024, 1, 2), date(2024, 1, 16), True)
    ]
    log = query(data_dir, "SELECT run_id, symbol, rows_written, status FROM ingestion_log")
    assert log == [(report.run_id, "AAA", 10, "ok")]
    assert issues(data_dir).empty


def test_running_twice_adds_no_duplicate_rows(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    source = FakeSource({"AAA": clean_bars})
    config = make_config("AAA")
    first = ingest(config, data_dir, source)
    second = ingest(config, data_dir, source)

    assert second.results[0].rows_written == 0
    assert first.run_id != second.run_id
    assert query(data_dir, "SELECT count(*) FROM daily_bars") == [(10,)]
    assert query(data_dir, "SELECT count(*) FROM instruments") == [(1,)]
    assert query(data_dir, "SELECT count(DISTINCT run_id), count(*) FROM ingestion_log") == [(2, 2)]
    # The second request only covers the five-session overlap.
    assert source.calls[1] == ("AAA", date(2024, 1, 8), date(2024, 1, 16))
    assert issues(data_dir).empty


def test_incremental_run_adds_only_new_bars(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    config = make_config("AAA")
    source = FakeSource({"AAA": clean_bars.iloc[:8]})  # through Thu Jan 11
    ingest(config, data_dir, source, now=datetime(2024, 1, 12, 0, 0))
    assert query(data_dir, "SELECT count(*) FROM daily_bars") == [(8,)]

    source.bars["AAA"] = clean_bars
    report = ingest(config, data_dir, source)
    assert report.results[0].rows_written == 2
    assert source.calls[1] == ("AAA", date(2024, 1, 4), date(2024, 1, 16))
    stored = stored_bars(data_dir, "AAA")
    assert len(stored) == 10
    # Older rows keep their original ingestion time; the new ones carry the later one.
    assert (
        stored["ingested_at"].tolist()
        == [pd.Timestamp("2024-01-12")] * 8 + [pd.Timestamp(AFTER_JAN_16)] * 2
    )


def test_every_download_leaves_a_raw_snapshot(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    source = FakeSource({"AAA": clean_bars})
    config = make_config("AAA")
    ingest(config, data_dir, source)
    ingest(config, data_dir, source)  # same second on the clock: must not overwrite

    taken = snapshots.list_snapshots(raw_dir(data_dir), "yahoo", "AAA")
    assert [s.taken_at for s in taken] == [
        datetime(2024, 1, 17, 0, 0, 0),
        datetime(2024, 1, 17, 0, 0, 1),
    ]
    assert taken[0].bars_path == raw_dir(data_dir) / "yahoo" / "AAA" / "20240117_000000.parquet"
    assert all(s.bars_path.is_file() and s.actions_path.is_file() for s in taken)
    assert len(pd.read_parquet(taken[0].bars_path)) == 10  # full first download
    assert len(pd.read_parquet(taken[1].bars_path)) == 6  # the overlap window


# ------------------------------------------------------------------ revisions


def test_revised_history_is_logged_then_overwritten(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    config = make_config("AAA")
    source = FakeSource({"AAA": clean_bars})
    ingest(config, data_dir, source)

    revised = clean_bars.copy()
    revised.loc[revised["date"] == "2024-01-12", ["close", "volume"]] = [104.75, 5000.0]
    revised.loc[revised["date"] == "2024-01-11", "close"] *= 1 + 1e-7  # below the tolerance
    source.bars["AAA"] = revised
    report = ingest(config, data_dir, source, now=datetime(2024, 1, 17, 6, 0))

    assert report.results[0].status == "ok"
    found = issues(data_dir)
    assert len(found) == 1
    row = found.iloc[0]
    assert (row["symbol"], row["check_name"], row["severity"]) == ("AAA", "revision", "warning")
    assert row["date"].date() == date(2024, 1, 12)
    assert row["details"] == "close 104.5 -> 104.75; volume 1800 -> 5000"
    stored = stored_bars(data_dir, "AAA").set_index("date")
    assert stored.loc["2024-01-12", "close"] == 104.75

    # A third run with the same data reports nothing new.
    ingest(config, data_dir, source, now=datetime(2024, 1, 17, 7, 0))
    assert len(issues(data_dir)) == 1


def test_detect_revisions_in_corporate_actions(
    clean_bars: pd.DataFrame, split_dividend_actions: pd.DataFrame
) -> None:
    changed = split_dividend_actions.copy()
    changed.loc[changed["action"] == "dividend", "value"] = 1.1
    found = detect_revisions(
        "AAA", clean_bars, clean_bars, split_dividend_actions, changed, tolerance=1e-4
    )
    assert [(i.date, i.check_name, i.details) for i in found] == [
        (date(2024, 1, 8), "revision", "dividend 1 -> 1.1")
    ]
    assert detect_revisions("AAA", clean_bars.iloc[:0], clean_bars, changed, changed, 1e-4) == []


# ------------------------------------------------------------------ failures


def test_one_failing_symbol_does_not_stop_the_run(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    source = FakeSource({"AAA": clean_bars, "CCC": clean_bars})  # nothing for BAD
    sleeps: list[float] = []
    config = make_config(
        "AAA", "BAD", "CCC", sources={"yahoo": {"request_delay_seconds": 0.5, "max_retries": 0}}
    )
    report = run_ingest(
        config, data_dir, sources={"yahoo": source}, clock=Clock(AFTER_JAN_16), sleep=sleeps.append
    )

    assert [(r.symbol, r.status) for r in report.results] == [
        ("AAA", "ok"),
        ("BAD", "failed"),
        ("CCC", "ok"),
    ]
    assert report.failed
    assert "no data for BAD" in report.results[1].message
    assert query(data_dir, "SELECT symbol, count(*) FROM daily_bars GROUP BY 1 ORDER BY 1") == [
        ("AAA", 10),
        ("CCC", 10),
    ]
    assert query(data_dir, "SELECT status FROM ingestion_log WHERE symbol = 'BAD'") == [("failed",)]
    assert sleeps == [0.5, 0.5]  # the configured delay between symbols


def test_partial_response_is_stored_but_reported(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    holey = clean_bars[clean_bars["date"] != "2024-01-09"].reset_index(drop=True)
    report = ingest(make_config("AAA"), data_dir, FakeSource({"AAA": holey}))

    result = report.results[0]
    assert result.status == "partial"
    assert report.failed
    assert result.rows_written == 9
    assert "1 of 10 expected sessions missing (first 2024-01-09" in result.message
    assert issues(data_dir)["check_name"].tolist() == ["gap"]


def test_response_ending_early_is_partial(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    report = ingest(make_config("AAA"), data_dir, FakeSource({"AAA": clean_bars.iloc[:8]}))
    assert report.results[0].status == "partial"
    assert "2 of 10 expected sessions missing (first 2024-01-12, last 2024-01-16)" in (
        report.results[0].message
    )


def test_duplicate_and_incomplete_rows_are_dropped_and_reported(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    messy = pd.concat([clean_bars, clean_bars.iloc[[4]]], ignore_index=True)
    messy.loc[2, "close"] = np.nan
    report = ingest(make_config("AAA"), data_dir, FakeSource({"AAA": messy}))

    result = report.results[0]
    assert result.status == "partial"
    assert "1 duplicated dates" in result.message
    assert "1 rows with missing values" in result.message
    assert result.rows_written == 9
    assert (
        stored_bars(data_dir, "AAA")[["open", "high", "low", "close", "volume"]]
        .notna()
        .all(axis=None)
    )
    assert set(issues(data_dir)["check_name"]) == {"duplicate_bar", "gap"}


def test_unknown_symbol_is_a_failure(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    report = ingest(
        make_config("AAA"), data_dir, FakeSource({"AAA": clean_bars}), symbols=["AAA", "ZZZ"]
    )
    assert {r.symbol: r.status for r in report.results} == {"ZZZ": "failed", "AAA": "ok"}
    assert report.failed


def test_symbols_filter(make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame) -> None:
    source = FakeSource({"AAA": clean_bars, "BBB": clean_bars})
    report = ingest(make_config("AAA", "BBB"), data_dir, source, symbols=["BBB"])
    assert [r.symbol for r in report.results] == ["BBB"]
    assert [c[0] for c in source.calls] == ["BBB"]
    assert query(data_dir, "SELECT DISTINCT symbol FROM daily_bars") == [("BBB",)]


# ------------------------------------------------------------------ universe changes


def test_disabled_source_is_skipped_not_failed(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    config = make_config(
        universe=[
            {"symbol": "AAA", "source": "yahoo", "asset_class": "etf"},
            {"symbol": "BTC/USD", "source": "crypto", "asset_class": "crypto"},
        ]
    )
    report = ingest(config, data_dir, FakeSource({"AAA": clean_bars}))
    assert [(r.symbol, r.status) for r in report.results] == [("AAA", "ok"), ("BTC/USD", "skipped")]
    assert not report.failed
    assert query(data_dir, "SELECT symbol, is_active FROM instruments ORDER BY 1") == [
        ("AAA", True),
        ("BTC/USD", False),
    ]


def test_symbol_removed_from_config_is_kept_but_inactive(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    source = FakeSource({"AAA": clean_bars, "BBB": clean_bars})
    ingest(make_config("AAA", "BBB"), data_dir, source)
    ingest(make_config("AAA"), data_dir, source)
    assert query(data_dir, "SELECT symbol, is_active FROM instruments ORDER BY 1") == [
        ("AAA", True),
        ("BBB", False),
    ]
    assert query(data_dir, "SELECT count(*) FROM daily_bars WHERE symbol = 'BBB'") == [(10,)]
