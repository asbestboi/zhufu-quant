"""Immutable raw Parquet snapshots of every download.

Layout: ``{raw}/{source}/{symbol}/{YYYYMMDD_HHMMSS}.parquet`` for bars and
``{YYYYMMDD_HHMMSS}.actions.parquet`` next to it for corporate actions. Timestamps are UTC.
Files are written once and never modified, which is what makes point-in-time loading
(``get_bars(as_of=...)``) and provider-revision forensics possible.

Each file's SHA-256 goes into the ``snapshot_manifest`` table when it is written (and
``qr snapshots backfill`` adds files from before the manifest existed), so a changed or
damaged snapshot can be detected later.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from qr import storage
from qr.frames import empty_actions, empty_bars, normalize_actions, normalize_bars
from qr.storage import Connection, ManifestEntry

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
ACTIONS_SUFFIX = ".actions.parquet"


@dataclass(frozen=True)
class Snapshot:
    """One download: when it was taken and where its two files live."""

    taken_at: datetime
    bars_path: Path
    actions_path: Path


def symbol_dir(raw_root: Path, source: str, symbol: str) -> Path:
    """Snapshot directory for a symbol (``/`` in crypto pairs is not a valid file name)."""
    return raw_root / source / symbol.replace("/", "-")


def _paths(directory: Path, taken_at: datetime) -> tuple[Path, Path]:
    stem = taken_at.strftime(TIMESTAMP_FORMAT)
    return directory / f"{stem}.parquet", directory / f"{stem}{ACTIONS_SUFFIX}"


def write_snapshot(
    raw_root: Path,
    source: str,
    symbol: str,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    taken_at: datetime,
) -> Snapshot:
    """Persist one download exactly as the source returned it; never overwrites.

    If a snapshot already exists for the same second, the timestamp is moved forward
    one second at a time until the name is free.
    """
    directory = symbol_dir(raw_root, source, symbol)
    directory.mkdir(parents=True, exist_ok=True)
    taken_at = taken_at.replace(microsecond=0)
    bars_path, actions_path = _paths(directory, taken_at)
    while bars_path.exists() or actions_path.exists():
        taken_at += timedelta(seconds=1)
        bars_path, actions_path = _paths(directory, taken_at)
    bars.to_parquet(bars_path, index=False)
    actions.to_parquet(actions_path, index=False)
    return Snapshot(taken_at, bars_path, actions_path)


def list_snapshots(raw_root: Path, source: str, symbol: str) -> list[Snapshot]:
    """All snapshots for a symbol, oldest first."""
    directory = symbol_dir(raw_root, source, symbol)
    if not directory.is_dir():
        return []
    found: list[Snapshot] = []
    for bars_path in directory.glob("*.parquet"):
        if bars_path.name.endswith(ACTIONS_SUFFIX):
            continue
        taken_at = datetime.strptime(bars_path.stem, TIMESTAMP_FORMAT)
        found.append(Snapshot(taken_at, bars_path, _paths(directory, taken_at)[1]))
    return sorted(found, key=lambda s: s.taken_at)


def bars_as_of(snapshots: list[Snapshot], as_of: datetime) -> pd.DataFrame:
    """Bars as they were known at ``as_of``: for each date, the value from the latest
    snapshot taken at or before ``as_of``."""
    known = [s for s in snapshots if s.taken_at <= as_of]
    if not known:
        return empty_bars()
    frames = [normalize_bars(pd.read_parquet(s.bars_path)) for s in known]  # oldest first
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates("date", keep="last")
    return normalize_bars(merged)


def actions_as_of(snapshots: list[Snapshot], as_of: datetime) -> pd.DataFrame:
    """Corporate actions from the latest snapshot taken at or before ``as_of``.

    Each actions snapshot is the provider's full history, so the latest one is complete.
    """
    known = [s for s in snapshots if s.taken_at <= as_of and s.actions_path.exists()]
    if not known:
        return empty_actions()
    return normalize_actions(pd.read_parquet(known[-1].actions_path))


# --------------------------------------------------------------------------- manifest


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, as hex."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_entries(
    raw_root: Path,
    source: str,
    symbol: str,
    snapshot: Snapshot,
    skip: set[str] | frozenset[str] = frozenset(),
) -> list[ManifestEntry]:
    """Manifest rows for a snapshot's files that exist (bars, then actions).

    Files whose relative path is in ``skip`` are left out without being read.
    """
    entries: list[ManifestEntry] = []
    for kind, path in (("bars", snapshot.bars_path), ("actions", snapshot.actions_path)):
        relative = path.relative_to(raw_root).as_posix()
        if relative in skip or not path.is_file():
            continue
        entries.append(
            ManifestEntry(
                path=relative,
                source=source,
                symbol=symbol,
                kind=kind,
                fetched_at=snapshot.taken_at,
                rows=len(pd.read_parquet(path)),
                sha256=file_sha256(path),
            )
        )
    return entries


def all_snapshots(
    raw_root: Path, known_symbols: Mapping[str, str] | None = None
) -> Iterator[tuple[str, str, Snapshot]]:
    """Every snapshot under ``raw_root`` as ``(source, symbol, snapshot)``.

    Directory names are mapped back to symbols through ``known_symbols`` (directory name
    to symbol, needed for crypto pairs whose ``/`` became ``-``); unknown directories keep
    their name. Files whose name is not a snapshot timestamp are skipped.
    """
    known = dict(known_symbols or {})
    if not raw_root.is_dir():
        return
    for source_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for directory in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            symbol = known.get(directory.name, directory.name)
            for bars_path in sorted(directory.glob("*.parquet")):
                if bars_path.name.endswith(ACTIONS_SUFFIX):
                    continue
                try:
                    taken_at = datetime.strptime(bars_path.stem, TIMESTAMP_FORMAT)
                except ValueError:
                    continue
                snapshot = Snapshot(taken_at, bars_path, _paths(directory, taken_at)[1])
                yield source_dir.name, symbol, snapshot


def backfill_manifest(con: Connection, raw_root: Path) -> tuple[int, int]:
    """Hash every snapshot file not in the manifest yet and add it.

    Returns ``(added, already_listed)``. Files already listed are not re-hashed, so this
    is cheap to repeat. Directory names are matched to symbols through the instruments
    table.
    """
    listed = {str(path) for path in con.execute("SELECT path FROM snapshot_manifest").df()["path"]}
    instruments = storage.read_instruments(con)
    known = {
        symbol_dir(raw_root, str(row["source"]), str(row["symbol"])).name: str(row["symbol"])
        for row in instruments.to_dict("records")
    }
    added = already = 0
    for source, symbol, snapshot in all_snapshots(raw_root, known):
        for path in (snapshot.bars_path, snapshot.actions_path):
            if path.is_file() and path.relative_to(raw_root).as_posix() in listed:
                already += 1
        fresh = manifest_entries(raw_root, source, symbol, snapshot, skip=listed)
        added += storage.record_snapshots(con, fresh)
    return added, already
