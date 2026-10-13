"""DuckDB schema, idempotent upserts, and reads.

Everything here takes an open connection so callers control transactions. Prices in
``daily_bars`` are always UNADJUSTED; adjustment happens at load time (see ``adjust.py``).
Timestamps are naive UTC.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from qr.frames import ACTION_COLUMNS, BAR_COLUMNS, VALUE_COLUMNS, to_dates

Connection = duckdb.DuckDBPyConnection

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
  symbol        TEXT PRIMARY KEY,
  source        TEXT NOT NULL,
  asset_class   TEXT NOT NULL,
  first_date    DATE,
  last_date     DATE,
  is_active     BOOLEAN DEFAULT TRUE,   -- keep delisted symbols, never delete
  updated_at    TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_bars (
  symbol     TEXT NOT NULL,
  date       DATE NOT NULL,
  open       DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
  volume     DOUBLE,
  source     TEXT NOT NULL,
  ingested_at TIMESTAMP NOT NULL,
  PRIMARY KEY (symbol, date)
);  -- UNADJUSTED prices

CREATE TABLE IF NOT EXISTS corporate_actions (
  symbol  TEXT NOT NULL,
  date    DATE NOT NULL,
  action  TEXT NOT NULL,   -- 'split' | 'dividend'
  value   DOUBLE NOT NULL, -- split ratio or cash dividend per share
  PRIMARY KEY (symbol, date, action)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
  run_id      TEXT, symbol TEXT, started_at TIMESTAMP, finished_at TIMESTAMP,
  rows_written INTEGER, status TEXT, message TEXT
);

CREATE TABLE IF NOT EXISTS quality_issues (
  symbol TEXT, date DATE, check_name TEXT, severity TEXT, details TEXT, detected_at TIMESTAMP
);
"""


class StorageError(Exception):
    """Raised when the database is missing or not initialised."""


@dataclass(frozen=True)
class QualityIssue:
    """One row of ``quality_issues``. ``date`` is None for symbol-level findings."""

    symbol: str
    date: date | None
    check_name: str
    severity: str  # 'error' | 'warning'
    details: str


def utcnow() -> datetime:
    """Current time as naive UTC, the convention for every stored timestamp."""
    return datetime.now(UTC).replace(tzinfo=None)


@contextmanager
def connect(path: Path | str, read_only: bool = False) -> Iterator[Connection]:
    """Open the DuckDB file, creating its directory when writing."""
    path = Path(path)
    if read_only and not path.exists():
        raise StorageError(f"database not found: {path} (run `qr init` first)")
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def init_schema(con: Connection) -> None:
    """Create all tables if they do not exist. Safe to call repeatedly."""
    con.execute(SCHEMA)


@contextmanager
def transaction(con: Connection) -> Iterator[None]:
    """Run a block atomically: commit on success, roll back on any exception."""
    con.execute("BEGIN")
    try:
        yield
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")


# --------------------------------------------------------------------------- instruments


def upsert_instrument(
    con: Connection, symbol: str, source: str, asset_class: str, is_active: bool, now: datetime
) -> None:
    """Insert or update an instrument's metadata without touching its coverage dates."""
    con.execute(
        """
        INSERT INTO instruments (symbol, source, asset_class, is_active, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (symbol) DO UPDATE SET
          source = excluded.source, asset_class = excluded.asset_class,
          is_active = excluded.is_active, updated_at = excluded.updated_at
        """,
        [symbol, source, asset_class, is_active, now],
    )


def deactivate_instruments_not_in(con: Connection, symbols: Sequence[str], now: datetime) -> None:
    """Mark instruments that left the configured universe as inactive (rows are kept)."""
    con.execute(
        """
        UPDATE instruments SET is_active = FALSE, updated_at = ?
        WHERE is_active AND NOT list_contains(?, symbol)
        """,
        [now, list(symbols)],
    )


def refresh_coverage(con: Connection, symbol: str, now: datetime) -> None:
    """Recompute ``first_date`` / ``last_date`` for a symbol from its stored bars."""
    con.execute(
        """
        UPDATE instruments SET
          first_date = (SELECT min(date) FROM daily_bars WHERE symbol = ?),
          last_date  = (SELECT max(date) FROM daily_bars WHERE symbol = ?),
          updated_at = ?
        WHERE symbol = ?
        """,
        [symbol, symbol, now, symbol],
    )


def read_instruments(con: Connection) -> pd.DataFrame:
    """All instruments, ordered by symbol."""
    return con.execute("SELECT * FROM instruments ORDER BY symbol").df()


def last_bar_date(con: Connection, symbol: str) -> date | None:
    """Date of the newest stored bar for ``symbol``, or None if there are none."""
    row = con.execute("SELECT max(date) FROM daily_bars WHERE symbol = ?", [symbol]).fetchone()
    return row[0] if row else None


# --------------------------------------------------------------------------- bars


def upsert_bars(
    con: Connection, symbol: str, source: str, bars: pd.DataFrame, ingested_at: datetime
) -> int:
    """Insert new bars and overwrite changed ones; return how many rows were written.

    Rows identical to what is stored are left alone, including their ``ingested_at``,
    so re-running an ingest is a no-op and ``ingested_at`` means "last time this value
    changed". ``bars`` must have unique dates.
    """
    if bars.empty:
        return 0
    if bars["date"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate dates passed to upsert_bars")
    con.register("incoming", bars[BAR_COLUMNS])
    distinct = " OR ".join(f"daily_bars.{c} IS DISTINCT FROM excluded.{c}" for c in VALUE_COLUMNS)
    changed = " OR ".join(f"b.{c} IS DISTINCT FROM i.{c}" for c in VALUE_COLUMNS)
    row = con.execute(
        f"""
        SELECT count(*) FROM incoming i
        LEFT JOIN daily_bars b ON b.symbol = ? AND b.date = CAST(i.date AS DATE)
        WHERE b.symbol IS NULL OR {changed}
        """,
        [symbol],
    ).fetchone()
    written = int(row[0]) if row else 0
    if written:
        con.execute(
            f"""
            INSERT INTO daily_bars (symbol, date, open, high, low, close, volume,
                                    source, ingested_at)
            SELECT ?, CAST(date AS DATE), open, high, low, close, volume, ?, ? FROM incoming
            ON CONFLICT (symbol, date) DO UPDATE SET
              open = excluded.open, high = excluded.high, low = excluded.low,
              close = excluded.close, volume = excluded.volume,
              source = excluded.source, ingested_at = excluded.ingested_at
            WHERE {distinct}
            """,
            [symbol, source, ingested_at],
        )
    con.unregister("incoming")
    return written


def read_bars(
    con: Connection,
    symbols: Sequence[str],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Unadjusted bars for ``symbols`` in ``[start, end]``, ordered by symbol then date.

    Columns: symbol, date, open, high, low, close, volume, source, ingested_at.
    """
    query = "SELECT * FROM daily_bars WHERE list_contains(?, symbol)"
    params: list[object] = [list(symbols)]
    if start is not None:
        query += " AND date >= ?"
        params.append(start)
    if end is not None:
        query += " AND date <= ?"
        params.append(end)
    frame = con.execute(query + " ORDER BY symbol, date", params).df()
    frame["date"] = to_dates(frame["date"]).to_numpy()
    return frame


# --------------------------------------------------------------------------- corporate actions


def upsert_corporate_actions(con: Connection, symbol: str, actions: pd.DataFrame) -> int:
    """Insert new corporate actions and overwrite changed values; return rows written.

    Actions that disappear from the provider are kept: we never delete history.
    """
    if actions.empty:
        return 0
    con.register("incoming", actions[ACTION_COLUMNS])
    row = con.execute(
        """
        SELECT count(*) FROM incoming i
        LEFT JOIN corporate_actions c
          ON c.symbol = ? AND c.date = CAST(i.date AS DATE) AND c.action = i.action
        WHERE c.symbol IS NULL OR c.value IS DISTINCT FROM i.value
        """,
        [symbol],
    ).fetchone()
    written = int(row[0]) if row else 0
    if written:
        con.execute(
            """
            INSERT INTO corporate_actions (symbol, date, action, value)
            SELECT ?, CAST(date AS DATE), action, value FROM incoming
            ON CONFLICT (symbol, date, action) DO UPDATE SET value = excluded.value
            WHERE corporate_actions.value IS DISTINCT FROM excluded.value
            """,
            [symbol],
        )
    con.unregister("incoming")
    return written


def read_corporate_actions(con: Connection, symbols: Sequence[str]) -> pd.DataFrame:
    """Corporate actions for ``symbols``. Columns: symbol, date, action, value."""
    frame = con.execute(
        """
        SELECT * FROM corporate_actions WHERE list_contains(?, symbol)
        ORDER BY symbol, date, action
        """,
        [list(symbols)],
    ).df()
    frame["date"] = to_dates(frame["date"]).to_numpy()
    return frame


# --------------------------------------------------------------------------- logs and issues


def log_ingestion(
    con: Connection,
    run_id: str,
    symbol: str,
    started_at: datetime,
    finished_at: datetime,
    rows_written: int,
    status: str,
    message: str,
) -> None:
    """Append one row to ``ingestion_log``."""
    con.execute(
        "INSERT INTO ingestion_log VALUES (?, ?, ?, ?, ?, ?, ?)",
        [run_id, symbol, started_at, finished_at, rows_written, status, message],
    )


def insert_quality_issues(
    con: Connection, issues: Sequence[QualityIssue], detected_at: datetime
) -> None:
    """Append findings to ``quality_issues``."""
    if not issues:
        return
    con.executemany(
        "INSERT INTO quality_issues VALUES (?, ?, ?, ?, ?, ?)",
        [[i.symbol, i.date, i.check_name, i.severity, i.details, detected_at] for i in issues],
    )


def delete_quality_issues(con: Connection, symbol: str, check_names: Sequence[str]) -> None:
    """Remove a symbol's findings for the given checks (used before re-running them)."""
    con.execute(
        "DELETE FROM quality_issues WHERE symbol = ? AND list_contains(?, check_name)",
        [symbol, list(check_names)],
    )


def read_quality_issues(con: Connection, symbols: Sequence[str] | None = None) -> pd.DataFrame:
    """Stored findings, optionally limited to ``symbols``."""
    if symbols is None:
        return con.execute("SELECT * FROM quality_issues ORDER BY symbol, date, check_name").df()
    return con.execute(
        """
        SELECT * FROM quality_issues WHERE list_contains(?, symbol)
        ORDER BY symbol, date, check_name
        """,
        [list(symbols)],
    ).df()


def read_coverage(con: Connection) -> pd.DataFrame:
    """Per-symbol coverage for ``qr status``: dates, row count, and open issue counts."""
    return con.execute(
        """
        SELECT i.symbol, i.source, i.asset_class, i.is_active,
               b.first_date, b.last_date, coalesce(b.n_rows, 0) AS n_rows,
               coalesce(q.errors, 0) AS errors, coalesce(q.warnings, 0) AS warnings
        FROM instruments i
        LEFT JOIN (
          SELECT symbol, min(date) AS first_date, max(date) AS last_date, count(*) AS n_rows
          FROM daily_bars GROUP BY symbol
        ) b USING (symbol)
        LEFT JOIN (
          SELECT symbol,
                 count(*) FILTER (WHERE severity = 'error') AS errors,
                 count(*) FILTER (WHERE severity = 'warning') AS warnings
          FROM quality_issues GROUP BY symbol
        ) q USING (symbol)
        ORDER BY i.symbol
        """
    ).df()
