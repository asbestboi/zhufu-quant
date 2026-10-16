"""Sealed holdouts: a study's test segment is run once, and that result stands.

The first run of a study on a sealed segment stores its result in ``holdout_results``
(in the registry database). After that, for the same study and segment:

* the **same** config (same config hash) shows the stored result and runs nothing;
* a **different** config is refused, with the reason. The refusal can be overridden
  with an explicit reason (``--unseal-reason``), which is recorded permanently in
  ``holdout_overrides``, and the run's report carries a red "holdout reused" warning.
  The stored result is never replaced: the first look is the one that counts.

A sealed segment can only be run as part of a study (``study_id``), because "the first
run" has no meaning otherwise.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from qr import storage
from qr.backtest.registry import registry_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS holdout_results (
  study_id TEXT, split_name TEXT, segment TEXT, config_hash TEXT, data_hash TEXT,
  run_id TEXT, sealed_at TIMESTAMP, metrics_json TEXT,
  PRIMARY KEY (study_id, split_name, segment)
);
CREATE TABLE IF NOT EXISTS holdout_overrides (
  study_id TEXT, split_name TEXT, segment TEXT, run_id TEXT, config_hash TEXT,
  sealed_config_hash TEXT, reason TEXT, created_at TIMESTAMP
);
"""


class HoldoutError(Exception):
    """A sealed segment cannot be run as asked; the message says why."""


@dataclass(frozen=True)
class SealedResult:
    """The stored first result of a study on a sealed segment."""

    study_id: str
    split_name: str
    segment: str
    config_hash: str
    data_hash: str | None
    run_id: str
    sealed_at: datetime
    metrics: dict[str, float]


def _connect(data_directory: Path | str | None, read_only: bool = False):  # type: ignore[no-untyped-def]
    return storage.connect(registry_path(data_directory), read_only=read_only)


def find(
    study_id: str, split_name: str, segment: str, data_directory: Path | str | None = None
) -> SealedResult | None:
    """The sealed result of ``study_id`` on ``split_name``/``segment``, if there is one."""
    if not registry_path(data_directory).exists():
        return None
    with _connect(data_directory, read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if "holdout_results" not in tables:
            return None
        row = con.execute(
            "SELECT study_id, split_name, segment, config_hash, data_hash, run_id, sealed_at, "
            "metrics_json FROM holdout_results "
            "WHERE study_id = ? AND split_name = ? AND segment = ?",
            [study_id, split_name, segment],
        ).fetchone()
    if row is None:
        return None
    return SealedResult(
        study_id=row[0],
        split_name=row[1],
        segment=row[2],
        config_hash=row[3],
        data_hash=row[4],
        run_id=row[5],
        sealed_at=row[6],
        metrics=json.loads(row[7]),
    )


def check(
    study_id: str | None,
    split_name: str,
    segment: str,
    config_hash: str,
    unseal_reason: str | None,
    data_directory: Path | str | None = None,
) -> SealedResult | None:
    """Decide whether a run on a sealed segment may go ahead.

    Returns the stored result when this exact config already ran (show it, run nothing);
    None when the run may go ahead (the first run, or an override with a reason). Raises
    :class:`HoldoutError` when the run is refused.
    """
    if not study_id:
        raise HoldoutError(
            f"segment '{segment}' of split '{split_name}' is sealed: set study_id in the "
            "config, so the first run can be stored and later ones checked against it"
        )
    sealed = find(study_id, split_name, segment, data_directory)
    if sealed is None or sealed.config_hash == config_hash:
        return sealed
    if unseal_reason and unseal_reason.strip():
        return None
    raise HoldoutError(
        f"study '{study_id}' already used sealed segment '{segment}' of split "
        f"'{split_name}' on {sealed.sealed_at:%Y-%m-%d %H:%M} (run {sealed.run_id}, config "
        f"{sealed.config_hash}). This config ({config_hash}) is different: running it "
        "would turn the test into a second validation set, so the stored result stands. "
        'To run it anyway, give --unseal-reason "..."; the reason is recorded permanently '
        "and the report is marked as reusing the holdout."
    )


def _clean(metrics: Mapping[str, float]) -> dict[str, float | None]:
    return {k: (float(v) if math.isfinite(float(v)) else None) for k, v in metrics.items()}


def store(
    study_id: str,
    split_name: str,
    segment: str,
    config_hash: str,
    data_hash: str | None,
    run_id: str,
    metrics: Mapping[str, float],
    sealed_at: datetime,
    data_directory: Path | str | None = None,
) -> None:
    """Store the first result of a study on a sealed segment. Never overwrites."""
    with _connect(data_directory) as con:
        con.execute(SCHEMA)
        con.execute(
            "INSERT INTO holdout_results VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [study_id, split_name, segment, config_hash, data_hash, run_id, sealed_at,
             json.dumps(_clean(metrics), sort_keys=True)],
        )  # fmt: skip


def record_override(
    study_id: str,
    split_name: str,
    segment: str,
    run_id: str,
    config_hash: str,
    sealed_config_hash: str,
    reason: str,
    created_at: datetime,
    data_directory: Path | str | None = None,
) -> None:
    """Permanently record that a sealed segment was run again, and why."""
    with _connect(data_directory) as con:
        con.execute(SCHEMA)
        con.execute(
            "INSERT INTO holdout_overrides VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [study_id, split_name, segment, run_id, config_hash, sealed_config_hash,
             reason, created_at],
        )  # fmt: skip


def read_results(data_directory: Path | str | None = None) -> pd.DataFrame:
    """Every sealed result, oldest first."""
    return _read("holdout_results", "sealed_at", data_directory)


def read_overrides(data_directory: Path | str | None = None) -> pd.DataFrame:
    """Every recorded override, oldest first."""
    return _read("holdout_overrides", "created_at", data_directory)


def _read(table: str, order: str, data_directory: Path | str | None) -> pd.DataFrame:
    if not registry_path(data_directory).exists():
        return pd.DataFrame()
    with _connect(data_directory, read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if table not in tables:
            return pd.DataFrame()
        frame: pd.DataFrame = con.execute(f"SELECT * FROM {table} ORDER BY {order}").df()
        return frame
