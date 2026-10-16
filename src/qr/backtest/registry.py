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

# Columns added after the first release, applied to old registries on the next write.
# Additive only: old rows keep NULL in them.
MIGRATIONS = [
    # Provenance (research-discipline spec, part 1). See provenance.py.
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS data_hash TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS data_per_symbol_json TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS data_rows BIGINT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS data_as_of TIMESTAMP",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS python_version TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS uv_lock_sha256 TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS engine_core TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS rust_core_version TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS os TEXT",
    # Assumptions and cost stress (part 2). Stress runs are rows of their own, linked to
    # the run they stress by parent_run_id; see stress.py.
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS parent_run_id TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS cost_multiplier DOUBLE",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS assumptions_json TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS cost_sensitivity_json TEXT",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS break_even_multiplier DOUBLE",
    "ALTER TABLE backtest_runs ADD COLUMN IF NOT EXISTS fragile BOOLEAN",
]

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
    "data_hash",
    "data_per_symbol_json",
    "data_rows",
    "data_as_of",
    "python_version",
    "uv_lock_sha256",
    "engine_core",
    "rust_core_version",
    "os",
    "parent_run_id",
    "cost_multiplier",
    "assumptions_json",
    "cost_sensitivity_json",
    "break_even_multiplier",
    "fragile",
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
    data_hash: str | None = None
    data_per_symbol_json: str | None = None  # {"SYMBOL": sha256, ...}
    data_rows: int | None = None
    data_as_of: datetime | None = None  # None: the store as it was when the run started
    python_version: str | None = None
    uv_lock_sha256: str | None = None
    engine_core: str | None = None  # the core that ran; None for the vectorized engine
    rust_core_version: str | None = None
    os: str | None = None
    parent_run_id: str | None = None  # set on cost-stress runs: the run they stress
    cost_multiplier: float | None = None  # the multiplier the costs ran at
    assumptions_json: str | None = None
    cost_sensitivity_json: str | None = None
    break_even_multiplier: float | None = None  # None when outside the tested range
    fragile: bool | None = None


def registry_path(data_directory: Path | str | None = None) -> Path:
    """Path of the registry database inside the data directory."""
    return data_dir(data_directory) / "backtests.duckdb"


def new_run_id(created_at: datetime) -> str:
    """A unique id that sorts by time, e.g. ``20261013T142233-3f9a1c``."""
    return f"{created_at:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


def stress_run_id(parent_run_id: str, multiplier: float) -> str:
    """The id of ``parent_run_id``'s stress run at ``multiplier``, e.g. ``...-3f9a1c-x2``."""
    return f"{parent_run_id}-x{multiplier:g}"


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
        for statement in MIGRATIONS:
            con.execute(statement)
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
    *,
    include_stress: bool = False,
) -> pd.DataFrame:
    """Logged runs, newest first; empty (with the right columns) when nothing is logged.

    ``run_ids`` may be abbreviated: any unique prefix of a run id matches. A registry
    written before a column existed reads that column as missing values. Cost-stress
    runs are left out unless ``include_stress`` (see :func:`read_stress_runs`).
    """
    path = registry_path(data_directory)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    query = "SELECT * FROM backtest_runs WHERE TRUE"
    params: list[object] = []
    if not include_stress and _has_column(path, "parent_run_id"):
        query += " AND parent_run_id IS NULL"
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
        return con.execute(query, params).df().reindex(columns=COLUMNS)


def read_stress_runs(run_id: str, data_directory: Path | str | None = None) -> pd.DataFrame:
    """The cost-stress runs of ``run_id`` (a full id), by multiplier."""
    path = registry_path(data_directory)
    if not path.exists() or not _has_column(path, "parent_run_id"):
        return pd.DataFrame(columns=COLUMNS)
    query = "SELECT * FROM backtest_runs WHERE parent_run_id = ? ORDER BY cost_multiplier"
    with storage.connect(path, read_only=True) as con:
        return con.execute(query, [run_id]).df().reindex(columns=COLUMNS)


def _has_column(path: Path, column: str) -> bool:
    """Whether ``backtest_runs`` in the registry at ``path`` has ``column`` yet."""
    with storage.connect(path, read_only=True) as con:
        found = con.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'backtest_runs' AND column_name = ?",
            [column],
        ).fetchone()
    return bool(found and found[0])
