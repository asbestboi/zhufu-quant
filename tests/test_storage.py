from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from qr import storage
from qr.storage import QualityIssue

T1 = datetime(2024, 1, 17, 1, 0, 0)
T2 = datetime(2024, 1, 18, 1, 0, 0)


def count(con: storage.Connection, table: str) -> int:
    row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def test_init_schema_is_repeatable(con: storage.Connection) -> None:
    storage.init_schema(con)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert tables == {
        "instruments",
        "daily_bars",
        "corporate_actions",
        "ingestion_log",
        "quality_issues",
        "snapshot_manifest",
    }


def test_read_only_connect_needs_existing_db(tmp_path: Path) -> None:
    with (
        pytest.raises(storage.StorageError, match="qr init"),
        storage.connect(tmp_path / "missing.duckdb", read_only=True),
    ):
        pass


def test_upsert_bars_is_idempotent(con: storage.Connection, clean_bars: pd.DataFrame) -> None:
    assert storage.upsert_bars(con, "AAA", "yahoo", clean_bars, T1) == 10
    assert storage.upsert_bars(con, "AAA", "yahoo", clean_bars, T2) == 0
    assert count(con, "daily_bars") == 10

    stored = storage.read_bars(con, ["AAA"])
    pd.testing.assert_frame_equal(stored[clean_bars.columns], clean_bars)
    # Unchanged rows keep the time they were first written.
    assert set(stored["ingested_at"]) == {pd.Timestamp(T1)}


def test_upsert_bars_overwrites_only_changed_rows(
    con: storage.Connection, clean_bars: pd.DataFrame
) -> None:
    storage.upsert_bars(con, "AAA", "yahoo", clean_bars, T1)
    revised = clean_bars.copy()
    revised.loc[3, "close"] = 999.0
    assert storage.upsert_bars(con, "AAA", "yahoo", revised, T2) == 1

    stored = storage.read_bars(con, ["AAA"])
    assert count(con, "daily_bars") == 10
    assert stored.loc[3, "close"] == 999.0
    assert stored.loc[3, "ingested_at"] == pd.Timestamp(T2)
    assert (stored.drop(index=3)["ingested_at"] == pd.Timestamp(T1)).all()


def test_upsert_bars_rejects_duplicate_dates(
    con: storage.Connection, clean_bars: pd.DataFrame
) -> None:
    doubled = pd.concat([clean_bars, clean_bars.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate dates"):
        storage.upsert_bars(con, "AAA", "yahoo", doubled, T1)


def test_read_bars_filters_by_symbol_and_date(
    con: storage.Connection, clean_bars: pd.DataFrame
) -> None:
    storage.upsert_bars(con, "AAA", "yahoo", clean_bars, T1)
    storage.upsert_bars(con, "BBB", "yahoo", clean_bars, T1)
    window = storage.read_bars(con, ["AAA"], date(2024, 1, 4), date(2024, 1, 9))
    assert set(window["symbol"]) == {"AAA"}
    assert [d.day for d in window["date"]] == [4, 5, 8, 9]


def test_upsert_corporate_actions_is_idempotent(
    con: storage.Connection, split_dividend_actions: pd.DataFrame
) -> None:
    assert storage.upsert_corporate_actions(con, "AAA", split_dividend_actions) == 2
    assert storage.upsert_corporate_actions(con, "AAA", split_dividend_actions) == 0
    assert count(con, "corporate_actions") == 2

    changed = split_dividend_actions.copy()
    changed.loc[changed["action"] == "dividend", "value"] = 1.25
    assert storage.upsert_corporate_actions(con, "AAA", changed) == 1
    stored = storage.read_corporate_actions(con, ["AAA"])
    assert count(con, "corporate_actions") == 2
    assert stored.set_index("action")["value"].to_dict() == {"split": 2.0, "dividend": 1.25}


def test_instruments_are_deactivated_never_deleted(
    con: storage.Connection, clean_bars: pd.DataFrame
) -> None:
    for symbol in ("AAA", "GONE"):
        storage.upsert_instrument(con, symbol, "yahoo", "equity", True, T1)
    storage.upsert_bars(con, "GONE", "yahoo", clean_bars, T1)
    storage.refresh_coverage(con, "GONE", T1)

    storage.deactivate_instruments_not_in(con, ["AAA"], T2)
    instruments = storage.read_instruments(con).set_index("symbol")
    assert bool(instruments.loc["AAA", "is_active"]) is True
    assert bool(instruments.loc["GONE", "is_active"]) is False
    assert instruments.loc["GONE", "first_date"].date() == date(2024, 1, 2)
    assert instruments.loc["GONE", "last_date"].date() == date(2024, 1, 16)
    assert storage.last_bar_date(con, "GONE") == date(2024, 1, 16)
    assert storage.last_bar_date(con, "AAA") is None


def test_coverage_counts_rows_and_issues(con: storage.Connection, clean_bars: pd.DataFrame) -> None:
    storage.upsert_instrument(con, "AAA", "yahoo", "equity", True, T1)
    storage.upsert_instrument(con, "EMPTY", "yahoo", "equity", True, T1)
    storage.upsert_bars(con, "AAA", "yahoo", clean_bars, T1)
    storage.insert_quality_issues(
        con,
        [
            QualityIssue("AAA", date(2024, 1, 3), "gap", "warning", "x"),
            QualityIssue("AAA", date(2024, 1, 4), "gap", "warning", "x"),
            QualityIssue("AAA", date(2024, 1, 5), "ohlc_consistency", "error", "x"),
        ],
        T1,
    )
    coverage = storage.read_coverage(con).set_index("symbol")
    assert coverage.loc["AAA", ["n_rows", "errors", "warnings"]].tolist() == [10, 1, 2]
    assert coverage.loc["EMPTY", ["n_rows", "errors", "warnings"]].tolist() == [0, 0, 0]

    storage.delete_quality_issues(con, "AAA", ["gap"])
    assert storage.read_quality_issues(con, ["AAA"])["check_name"].tolist() == ["ohlc_consistency"]


def test_transaction_rolls_back_on_error(con: storage.Connection, clean_bars: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError), storage.transaction(con):
        storage.upsert_bars(con, "AAA", "yahoo", clean_bars, T1)
        raise RuntimeError("boom")
    assert count(con, "daily_bars") == 0
