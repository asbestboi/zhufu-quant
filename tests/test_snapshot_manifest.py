"""The snapshot manifest: a SHA-256 row for every raw snapshot file."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from qr import snapshots, storage
from qr.cli import app
from qr.config import UniverseConfig, db_path, raw_dir
from qr.ingest import run_ingest

from .conftest import AFTER_JAN_16, Clock, FakeSource

MakeConfig = Callable[..., UniverseConfig]
runner = CliRunner()


def manifest(data_dir: Path) -> pd.DataFrame:
    with storage.connect(db_path(data_dir), read_only=True) as con:
        return storage.read_snapshot_manifest(con)


def ingest(config: UniverseConfig, data_dir: Path, source: FakeSource, now: datetime) -> None:
    run_ingest(
        config,
        data_dir,
        sources={"yahoo": source},
        clock=Clock(now),
        sleep=lambda _: None,
    )


def test_every_new_snapshot_gets_manifest_rows(
    make_config: MakeConfig,
    data_dir: Path,
    clean_bars: pd.DataFrame,
    split_dividend_actions: pd.DataFrame,
) -> None:
    source = FakeSource({"AAA": clean_bars}, {"AAA": split_dividend_actions})
    ingest(make_config("AAA"), data_dir, source, AFTER_JAN_16)

    rows = manifest(data_dir)
    assert rows["kind"].tolist() == ["actions", "bars"]
    assert set(rows["path"]) == {
        "yahoo/AAA/20240117_000000.parquet",
        "yahoo/AAA/20240117_000000.actions.parquet",
    }
    assert (rows["source"] == "yahoo").all() and (rows["symbol"] == "AAA").all()
    assert (rows["fetched_at"] == pd.Timestamp(AFTER_JAN_16)).all()
    by_kind = rows.set_index("kind")
    assert by_kind.loc["bars", "rows"] == 10
    assert by_kind.loc["actions", "rows"] == len(split_dividend_actions)
    for row in rows.to_dict("records"):
        content = (raw_dir(data_dir) / str(row["path"])).read_bytes()
        assert row["sha256"] == hashlib.sha256(content).hexdigest()


def test_backfill_hashes_old_files_once(
    make_config: MakeConfig, data_dir: Path, clean_bars: pd.DataFrame
) -> None:
    ingest(make_config("AAA"), data_dir, FakeSource({"AAA": clean_bars}), AFTER_JAN_16)
    with storage.connect(db_path(data_dir)) as con:
        recorded = storage.read_snapshot_manifest(con)
        con.execute("DELETE FROM snapshot_manifest")  # as if written before the manifest
        added, already = snapshots.backfill_manifest(con, raw_dir(data_dir))
        assert (added, already) == (2, 0)
        assert snapshots.backfill_manifest(con, raw_dir(data_dir)) == (0, 2)
        backfilled = storage.read_snapshot_manifest(con)
    pd.testing.assert_frame_equal(backfilled, recorded)


def test_backfill_maps_crypto_directories_back_to_symbols(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    bars = pd.DataFrame(
        {"date": [pd.Timestamp("2024-01-02")], "open": [1.0], "high": [1.0], "low": [1.0]}
    ).assign(close=1.0, volume=5.0)
    snapshots.write_snapshot(
        raw_root, "crypto", "BTC/USD", bars, bars.iloc[:0][["date"]], AFTER_JAN_16
    )
    (raw_root / "crypto" / "BTC-USD" / "notes.parquet").write_bytes(b"")  # not a snapshot
    with storage.connect(tmp_path / "market.duckdb") as con:
        storage.init_schema(con)
        storage.upsert_instrument(con, "BTC/USD", "crypto", "crypto", True, AFTER_JAN_16)
        added, _ = snapshots.backfill_manifest(con, raw_root)
        rows = storage.read_snapshot_manifest(con)
    assert added == 2
    assert set(rows["symbol"]) == {"BTC/USD"}


def test_existing_rows_are_never_rewritten(con: storage.Connection) -> None:
    entry = storage.ManifestEntry("yahoo/A/x.parquet", "yahoo", "A", "bars", AFTER_JAN_16, 1, "a")
    assert storage.record_snapshots(con, [entry]) == 1
    changed = storage.ManifestEntry("yahoo/A/x.parquet", "yahoo", "A", "bars", AFTER_JAN_16, 2, "b")
    assert storage.record_snapshots(con, [changed]) == 0
    assert storage.read_snapshot_manifest(con)["sha256"].tolist() == ["a"]


def test_backfill_command(
    make_config: MakeConfig,
    data_dir: Path,
    clean_bars: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest(make_config("AAA"), data_dir, FakeSource({"AAA": clean_bars}), AFTER_JAN_16)
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["snapshots", "backfill", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    assert "Added 0 snapshot files" in result.output
    assert "2 were already listed" in result.output
