"""Dataset fingerprints: deterministic, and sensitive to exactly the data that was used."""

from __future__ import annotations

import hashlib
import struct
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qr import snapshots, storage
from qr.config import db_path, raw_dir
from qr.fingerprint import combine, hash_symbol
from qr.load import fingerprint

from .conftest import AFTER_JAN_16

PINNED = "3085b3a385e922bbb32ae3e2362b9ba1cc3adb1c770dd891bdea450985b27d81"

ACTIONS = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-05", "2024-01-10"]),
        "action": ["dividend", "split"],
        "value": [0.25, 2.0],
    }
)


def store(
    data_dir: Path,
    symbol: str,
    bars: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    when: datetime = AFTER_JAN_16,
) -> None:
    """Write bars and actions for ``symbol`` straight into the database."""
    with storage.connect(db_path(data_dir)) as con:
        storage.init_schema(con)
        storage.upsert_instrument(con, symbol, "yahoo", "equity", True, when)
        storage.upsert_bars(con, symbol, "yahoo", bars, when)
        if actions is not None:
            storage.upsert_corporate_actions(con, symbol, actions)


@pytest.fixture
def two_symbols(data_dir: Path, clean_bars: pd.DataFrame) -> Path:
    store(data_dir, "AAA", clean_bars, ACTIONS)
    store(data_dir, "BBB", clean_bars.assign(close=clean_bars["close"] * 2))
    return data_dir


def test_the_canonical_form_matches_its_description() -> None:
    bars = pd.DataFrame(
        {"date": ["2024-01-03", "2024-01-02"], "open": [2.0, 1.0], "high": [2.5, 1.5]}
    ).assign(low=[np.nan, 0.5], close=[2.25, 1.25], volume=[200.0, -0.0])
    actions = pd.DataFrame({"date": ["2024-01-03"], "action": ["dividend"], "value": [0.1]})

    # Built by hand from the format in qr.fingerprint's docstring.
    day = 86_400 * 10**9
    jan2, jan3 = 19_724 * day, 19_725 * day
    columns = [(1.0, 2.0), (1.5, 2.5), (0.5, None), (1.25, 2.25), (0.0, 200.0)]
    expected = b"qr-fingerprint-v1|X|bars=2|actions=1\n" + struct.pack("<qq", jan2, jan3)
    for values in columns:
        expected += bytes(1 if v is None else 0 for v in values)
        expected += struct.pack("<dd", *(0.0 if v is None else v for v in values))
    expected += struct.pack("<q", jan3) + bytes([2]) + bytes([0]) + struct.pack("<d", 0.1)

    assert hash_symbol("X", bars, actions) == hashlib.sha256(expected).hexdigest()
    # Pinned as well: if this changes, every stored hash stops matching.
    assert hashlib.sha256(expected).hexdigest() == PINNED


def test_row_order_index_and_nan_encoding_do_not_matter(clean_bars: pd.DataFrame) -> None:
    reference = hash_symbol("AAA", clean_bars, ACTIONS)
    shuffled = clean_bars.sample(frac=1.0, random_state=7)
    assert hash_symbol("AAA", shuffled, ACTIONS.iloc[::-1]) == reference
    by_date = clean_bars.set_index("date", drop=False)
    assert hash_symbol("AAA", by_date, ACTIONS) == reference
    as_text = clean_bars.assign(date=clean_bars["date"].dt.strftime("%Y-%m-%d"))
    assert hash_symbol("AAA", as_text, ACTIONS) == reference

    # Two NaNs with different bit patterns, and -0.0 against 0.0, hash the same...
    quiet = clean_bars.assign(volume=np.nan)
    payload = np.full(len(clean_bars), np.nan).view("u8") | np.uint64(1)
    other_nan = clean_bars.assign(volume=payload.view("f8"))
    assert hash_symbol("AAA", quiet, ACTIONS) == hash_symbol("AAA", other_nan, ACTIONS)
    zero = clean_bars.assign(volume=0.0)
    assert hash_symbol("AAA", zero, ACTIONS) == hash_symbol(
        "AAA", zero.assign(volume=-0.0), ACTIONS
    )
    # ...but a missing value is not the same as zero.
    assert hash_symbol("AAA", quiet, ACTIONS) != hash_symbol("AAA", zero, ACTIONS)


def test_same_data_gives_the_same_fingerprint(
    two_symbols: Path, tmp_path: Path, clean_bars: pd.DataFrame
) -> None:
    first = fingerprint(["AAA", "BBB"], data_dir=two_symbols)
    assert fingerprint(["BBB", "AAA", "BBB"], data_dir=two_symbols) == first
    assert first.rows == 20
    assert (first.first_date, first.last_date) == (date(2024, 1, 2), date(2024, 1, 16))
    assert first.latest_ingested_at == AFTER_JAN_16
    assert set(first.per_symbol) == {"AAA", "BBB"}

    # The same rows written into another store in reverse order.
    other = tmp_path / "other"
    backwards = clean_bars.sort_values("date", ascending=False, ignore_index=True)
    store(other, "BBB", backwards.assign(close=backwards["close"] * 2))
    store(other, "AAA", backwards, ACTIONS.sort_values("date", ascending=False, ignore_index=True))
    assert fingerprint(["AAA", "BBB"], data_dir=other).hash == first.hash


@pytest.mark.parametrize(
    ("table", "change"),
    [
        ("daily_bars", "SET close = close + 0.01 WHERE date = DATE '2024-01-08'"),
        ("daily_bars", "SET volume = volume + 1 WHERE date = DATE '2024-01-03'"),
        ("corporate_actions", "SET value = 0.26 WHERE action = 'dividend'"),
    ],
)
def test_one_changed_value_changes_only_its_symbol(
    two_symbols: Path, table: str, change: str
) -> None:
    before = fingerprint(["AAA", "BBB"], data_dir=two_symbols)
    with storage.connect(db_path(two_symbols)) as con:
        con.execute(f"UPDATE {table} {change} AND symbol = 'AAA'")
    after = fingerprint(["AAA", "BBB"], data_dir=two_symbols)
    assert after.hash != before.hash
    assert after.per_symbol["BBB"] == before.per_symbol["BBB"]
    assert after.changed_symbols(before) == ["AAA"]


def test_symbols_outside_the_request_do_not_matter(
    two_symbols: Path, clean_bars: pd.DataFrame
) -> None:
    before = fingerprint(["AAA", "BBB"], data_dir=two_symbols)
    store(two_symbols, "ZZZ", clean_bars)
    assert fingerprint(["AAA", "BBB"], data_dir=two_symbols) == before


def test_dates_and_adjustment(two_symbols: Path) -> None:
    full = fingerprint("AAA", data_dir=two_symbols)
    window = fingerprint("AAA", date(2024, 1, 8), date(2024, 1, 12), data_dir=two_symbols)
    assert window.hash != full.hash
    assert (window.first_date, window.last_date, window.rows) == (
        date(2024, 1, 8),
        date(2024, 1, 12),
        5,
    )
    # Only the split (2024-01-10) is in the window: removing the earlier dividend changes
    # the full-history fingerprint but not the window's.
    with storage.connect(db_path(two_symbols)) as con:
        con.execute("DELETE FROM corporate_actions WHERE action = 'dividend'")
    assert fingerprint("AAA", date(2024, 1, 8), date(2024, 1, 12), data_dir=two_symbols) == window
    assert fingerprint("AAA", data_dir=two_symbols) != full

    none = fingerprint("AAA", adjustment="none", data_dir=two_symbols)
    assert none.per_symbol == fingerprint("AAA", data_dir=two_symbols).per_symbol
    assert none.hash != fingerprint("AAA", data_dir=two_symbols).hash
    assert none.hash == combine(none.per_symbol, "none")


def test_unknown_symbols_are_refused(two_symbols: Path) -> None:
    with pytest.raises(KeyError, match="NOPE"):
        fingerprint(["AAA", "NOPE"], data_dir=two_symbols)


def test_as_of_hashes_the_snapshots_known_then(data_dir: Path, clean_bars: pd.DataFrame) -> None:
    first = AFTER_JAN_16
    revised = clean_bars.assign(close=clean_bars["close"] + 1.0)
    for when, frame in ((first, clean_bars), (first + timedelta(days=1), revised)):
        snapshots.write_snapshot(raw_dir(data_dir), "yahoo", "AAA", frame, ACTIONS, when)
        store(data_dir, "AAA", frame, ACTIONS, when)

    then = fingerprint("AAA", as_of=first, data_dir=data_dir)
    now = fingerprint("AAA", data_dir=data_dir)
    assert then.hash != now.hash
    assert then.latest_ingested_at == first
    assert now.latest_ingested_at == first + timedelta(days=1)
    assert fingerprint("AAA", as_of=first + timedelta(days=2), data_dir=data_dir).hash == now.hash
