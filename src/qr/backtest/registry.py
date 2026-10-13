"""The experiment log: every backtest run, kept forever, including the failures.

Why it matters: after trying twenty ideas, the best one looks good by luck alone. To
correct for that (the Deflated Sharpe Ratio, step 3) the number of ideas tried has to be
known, and it is only known if nothing is ever left out. So runs are logged automatically
and never deleted, whether they worked, lost money, or crashed.

Runs live in their own DuckDB file, ``backtests.duckdb`` in the data directory, so
research code never opens the market-data database for writing.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from qr import storage
from qr.config import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id TEXT PRIMARY KEY, created_at TIMESTAMP, strategy TEXT, engine TEXT,
  config_json TEXT, config_hash TEXT, git_commit TEXT, git_dirty BOOLEAN,
  data_snapshot TEXT,      -- latest ingested_at of data used
  start_date DATE, end_date DATE,
  sharpe DOUBLE, cagr DOUBLE, max_drawdown DOUBLE, turnover DOUBLE,
  report_path TEXT, notes TEXT,
  name TEXT, status TEXT   -- 'ok' | 'failed' (notes then holds the error)
);
"""

COLUMNS = [
    "run_id",
    "created_at",
    "strategy",
    "engine",
    "config_json",
    "config_hash",
    "git_commit",
    "git_dirty",
    "data_snapshot",
    "start_date",
    "end_date",
    "sharpe",
    "cagr",
    "max_drawdown",
    "turnover",
    "report_path",
    "notes",
    "name",
    "status",
]

OK = "ok"
FAILED = "failed"


@dataclass(frozen=True)
class RunRecord:
    """One row of ``backtest_runs``."""

    run_id: str
    created_at: datetime
    strategy: str
    engine: str
    config_json: str
    config_hash: str
    git_commit: str | None
    git_dirty: bool | None
    data_snapshot: str | None
    start_date: date
    end_date: date
    sharpe: float | None
    cagr: float | None
    max_drawdown: float | None
    turnover: float | None
    report_path: str | None
    notes: str
    name: str
    status: str


def registry_path(data_directory: Path | str | None = None) -> Path:
    """Path of the registry database inside the data directory."""
    return data_dir(data_directory) / "backtests.duckdb"


def new_run_id(created_at: datetime) -> str:
    """A unique id that sorts by time, e.g. ``20261013T142233-3f9a1c``."""
    return f"{created_at:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


def git_state(cwd: Path | str | None = None) -> tuple[str | None, bool | None]:
    """The current commit hash and whether the working tree has uncommitted changes.

    ``(None, None)`` outside a git repository or without git installed.
    """

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None, None
    status = git("status", "--porcelain")
    return commit, None if status is None else bool(status)


def log_run(record: RunRecord, data_directory: Path | str | None = None) -> None:
    """Append ``record`` to the registry, creating the database if needed."""
    with storage.connect(registry_path(data_directory)) as con:
        con.execute(SCHEMA)
        placeholders = ", ".join("?" for _ in COLUMNS)
        con.execute(
            f"INSERT INTO backtest_runs ({', '.join(COLUMNS)}) VALUES ({placeholders})",
            [getattr(record, column) for column in COLUMNS],
        )


def read_runs(
    data_directory: Path | str | None = None,
    strategy: str | None = None,
    run_ids: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Logged runs, newest first; empty (with the right columns) when nothing is logged.

    ``run_ids`` may be abbreviated: any unique prefix of a run id matches.
    """
    path = registry_path(data_directory)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    query = "SELECT * FROM backtest_runs WHERE TRUE"
    params: list[object] = []
    if strategy is not None:
        query += " AND strategy = ?"
        params.append(strategy)
    if run_ids is not None:
        query += " AND (" + " OR ".join("starts_with(run_id, ?)" for _ in run_ids) + ")"
        params.extend(run_ids)
    query += " ORDER BY created_at DESC, run_id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with storage.connect(path, read_only=True) as con:
        return con.execute(query, params).df()[COLUMNS]
