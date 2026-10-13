"""The public loading API on a small store.

AAA is bars_split_dividend.csv (closes 100, 102, 51, 52, 50 with a 2:1 split on Jan 4
and a 1.00 dividend ex Jan 8). BBB is bars_clean.csv (ten sessions, no actions).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qr import storage
from qr.config import UniverseConfig, db_path
from qr.ingest import run_ingest
from qr.load import get_bars, get_returns, list_symbols

from .conftest import AFTER_JAN_16, Clock, FakeSource

F = 51.0 / 52.0  # dividend factor: 1 - 1.00 / 52


@pytest.fixture
def store(
    data_dir: Path,
    split_dividend_bars: pd.DataFrame,
    split_dividend_actions: pd.DataFrame,
    clean_bars: pd.DataFrame,
) -> Path:
    with storage.connect(db_path(data_dir)) as con:
        storage.init_schema(con)
        storage.upsert_instrument(con, "AAA", "yahoo", "equity", True, AFTER_JAN_16)
        storage.upsert_instrument(con, "BBB", "yahoo", "equity", False, AFTER_JAN_16)
        storage.upsert_bars(con, "AAA", "yahoo", split_dividend_bars, AFTER_JAN_16)
        storage.upsert_corporate_actions(con, "AAA", split_dividend_actions)
        storage.upsert_bars(con, "BBB", "yahoo", clean_bars, AFTER_JAN_16)
    return data_dir


def closes(frame: pd.DataFrame) -> list[float]:
    return [float(v) for v in frame["close"]]


def test_shape_of_the_result(store: Path) -> None:
    bars = get_bars(["BBB", "AAA"], data_dir=store)
    assert bars.index.names == ["date", "symbol"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert all(str(t) == "float64" for t in bars.dtypes)
    assert bars.index.get_level_values("date").dtype == "datetime64[ns]"
    assert bars.index.is_unique
    assert len(bars) == 15
    # Sorted by date, then symbol.
    assert bars.index[:3].tolist() == [
        (pd.Timestamp("2024-01-02"), "AAA"),
        (pd.Timestamp("2024-01-02"), "BBB"),
        (pd.Timestamp("2024-01-03"), "AAA"),
    ]
    wide = bars["close"].unstack("symbol")
    assert wide.shape == (10, 2)


def test_adjustment_none(store: Path, split_dividend_bars: pd.DataFrame) -> None:
    bars = get_bars("AAA", adjustment="none", data_dir=store)
    expected = split_dividend_bars.assign(symbol="AAA").set_index(["date", "symbol"])
    pd.testing.assert_frame_equal(bars, expected)


def test_adjustment_split(store: Path) -> None:
    bars = get_bars("AAA", adjustment="split", data_dir=store)
    assert closes(bars) == [50.0, 51.0, 51.0, 52.0, 50.0]
    assert bars["volume"].tolist() == [2000.0] * 5


def test_adjustment_total_return_is_the_default(store: Path) -> None:
    bars = get_bars("AAA", data_dir=store)
    np.testing.assert_allclose(closes(bars), [50 * F, 51 * F, 51 * F, 52 * F, 50.0], rtol=1e-12)
    np.testing.assert_allclose(bars["open"], [49.5 * F, 50.5 * F, 50.5 * F, 51.5 * F, 50.5])
    assert bars["volume"].tolist() == [2000.0] * 5
    pd.testing.assert_frame_equal(bars, get_bars("AAA", adjustment="total_return", data_dir=store))


def test_symbol_without_actions_is_unchanged_by_adjustment(
    store: Path, clean_bars: pd.DataFrame
) -> None:
    assert closes(get_bars("BBB", data_dir=store)) == clean_bars["close"].tolist()


def test_start_and_end_are_inclusive(store: Path) -> None:
    bars = get_bars("BBB", start=date(2024, 1, 4), end=date(2024, 1, 9), data_dir=store)
    assert [d.day for d in bars.index.get_level_values("date")] == [4, 5, 8, 9]


def test_start_does_not_change_adjusted_values(store: Path) -> None:
    bars = get_bars("AAA", start=date(2024, 1, 5), data_dir=store)
    np.testing.assert_allclose(closes(bars), [52 * F, 50.0], rtol=1e-12)


def test_adjustment_only_uses_actions_up_to_end(store: Path) -> None:
    # Ending on Jan 5, the Jan 8 dividend has not happened yet: split adjustment only,
    # and the last bar equals its unadjusted value.
    assert closes(get_bars("AAA", end=date(2024, 1, 5), data_dir=store)) == [50.0, 51.0, 51.0, 52.0]
    # Ending on Jan 3, the split has not happened either.
    assert closes(get_bars("AAA", end=date(2024, 1, 3), data_dir=store)) == [100.0, 102.0]


def test_unknown_symbol_and_empty_request(store: Path) -> None:
    with pytest.raises(KeyError, match="unknown symbols: NOPE"):
        get_bars(["AAA", "NOPE"], data_dir=store)
    with pytest.raises(ValueError, match="no symbols"):
        get_bars([], data_dir=store)


def test_missing_database_has_a_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(storage.StorageError, match="qr init"):
        get_bars("AAA", data_dir=tmp_path / "nowhere")


def test_get_returns_simple_and_log(store: Path) -> None:
    simple = get_returns("AAA", kind="simple", data_dir=store)
    assert simple.columns.tolist() == ["AAA"]
    assert [d.day for d in simple.index] == [3, 4, 5, 8]  # the first bar has no return
    # Total returns: the split day is flat, and the ex-date return includes the dividend.
    np.testing.assert_allclose(
        simple["AAA"], [51 / 50 - 1, 0.0, 52 / 51 - 1, 50 / 51 - 1], atol=1e-12
    )
    log = get_returns("AAA", data_dir=store)
    np.testing.assert_allclose(log["AAA"], np.log1p(simple["AAA"]), atol=1e-12)
    with pytest.raises(ValueError, match="unknown return kind"):
        get_returns("AAA", kind="weird", data_dir=store)  # type: ignore[arg-type]


def test_get_returns_wide_with_start(store: Path) -> None:
    returns = get_returns(["BBB", "AAA"], start=date(2024, 1, 8), kind="simple", data_dir=store)
    assert returns.columns.tolist() == ["BBB", "AAA"]
    assert returns.index[0] == pd.Timestamp("2024-01-08")  # uses the Jan 5 close before it
    assert returns.loc["2024-01-08", "BBB"] == pytest.approx(103.0 / 102.0 - 1)
    assert returns.loc["2024-01-08", "AAA"] == pytest.approx(50 / 51 - 1)
    assert returns["AAA"].iloc[1:].isna().all()  # AAA has no bars after Jan 8
    assert len(returns) == 6


def test_list_symbols(store: Path) -> None:
    assert list_symbols(data_dir=store) == ["AAA", "BBB"]
    assert list_symbols(active_only=True, data_dir=store) == ["AAA"]


# ------------------------------------------------------------------ point-in-time


@pytest.fixture
def revised_store(
    data_dir: Path, make_config: Callable[..., UniverseConfig], split_dividend_bars: pd.DataFrame
) -> Path:
    """Three downloads of AAA at T1 < T2 < T3:

    T1 (Jan 4, 00:00): bars through Jan 3, no corporate actions known yet.
    T2 (Jan 9, 00:00): all five bars plus the split and the dividend.
    T3 (Jan 9, 06:00): the provider revises the Jan 5 close from 52 to 52.5.
    """
    config = make_config("AAA")
    actions = pd.read_csv(Path(__file__).parent / "fixtures" / "actions_split_dividend.csv")
    source = FakeSource({"AAA": split_dividend_bars.iloc[:2]})

    def run(now: datetime) -> None:
        run_ingest(
            config, data_dir, sources={"yahoo": source}, clock=Clock(now), sleep=lambda _: None
        )

    run(datetime(2024, 1, 4, 0, 0))
    source.bars["AAA"] = split_dividend_bars
    source.actions["AAA"] = actions
    run(datetime(2024, 1, 9, 0, 0))
    revised = split_dividend_bars.copy()
    revised.loc[3, "close"] = 52.5
    source.bars["AAA"] = revised
    run(datetime(2024, 1, 9, 6, 0))
    return data_dir


def test_as_of_returns_what_was_known_then(revised_store: Path) -> None:
    latest = get_bars("AAA", adjustment="none", data_dir=revised_store)
    assert closes(latest) == [100.0, 102.0, 51.0, 52.5, 50.0]

    before_any = get_bars("AAA", as_of=datetime(2024, 1, 3, 23, 59), data_dir=revised_store)
    assert before_any.empty
    assert before_any.index.names == ["date", "symbol"]

    at_t1 = get_bars("AAA", as_of=datetime(2024, 1, 5), data_dir=revised_store)
    assert closes(at_t1) == [100.0, 102.0]  # no split known yet, so nothing to adjust

    at_t2 = get_bars(
        "AAA", adjustment="none", as_of=datetime(2024, 1, 9, 3), data_dir=revised_store
    )
    assert closes(at_t2) == [100.0, 102.0, 51.0, 52.0, 50.0]  # before the revision

    at_t3 = get_bars(
        "AAA", adjustment="none", as_of=datetime(2024, 1, 9, 6), data_dir=revised_store
    )
    assert closes(at_t3) == closes(latest)


def test_as_of_adjusts_with_the_actions_known_then(revised_store: Path) -> None:
    at_t2 = get_bars("AAA", as_of=datetime(2024, 1, 9, 3), data_dir=revised_store)
    np.testing.assert_allclose(closes(at_t2), [50 * F, 51 * F, 51 * F, 52 * F, 50.0], rtol=1e-12)
    # After the revision the dividend factor uses the revised previous close.
    at_t3 = get_bars("AAA", as_of=datetime(2024, 1, 9, 7), data_dir=revised_store)
    g = 1 - 1 / 52.5
    np.testing.assert_allclose(closes(at_t3), [50 * g, 51 * g, 51 * g, 52.5 * g, 50.0], rtol=1e-12)
    pd.testing.assert_frame_equal(at_t3, get_bars("AAA", data_dir=revised_store))


def test_as_of_accepts_timezone_aware_datetimes(revised_store: Path) -> None:
    prague = timezone(timedelta(hours=1))
    # 04:00 in UTC+1 is 03:00 UTC: after T2, before T3.
    aware = get_bars(
        "AAA",
        adjustment="none",
        as_of=datetime(2024, 1, 9, 4, tzinfo=prague),
        data_dir=revised_store,
    )
    assert closes(aware) == [100.0, 102.0, 51.0, 52.0, 50.0]
    utc = get_bars(
        "AAA", adjustment="none", as_of=datetime(2024, 1, 9, 6, tzinfo=UTC), data_dir=revised_store
    )
    assert closes(utc)[3] == 52.5
