"""The paper-trading journal: every run, decision, order, fill, and alert, kept forever.

Lives in its own DuckDB file, ``live.duckdb`` in the data directory, next to (but
independent of) the market-data and backtest-registry databases. A decision's id is
``f"{strategy}:{decision_date:%Y-%m-%d}"``: looking it up before planning is what makes a
run idempotent (see :mod:`qr.live.cycle`) -- re-running the same decision finds it already
``completed`` and sends nothing, and resuming after a crash finds it ``submitted`` and
only reconciles fills, never re-plans or re-sends.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from qr import storage
from qr.config import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_runs (
  run_id TEXT PRIMARY KEY,
  config_name TEXT NOT NULL,
  started_at TIMESTAMP NOT NULL,
  finished_at TIMESTAMP,
  status TEXT NOT NULL,      -- 'running' | 'ok' | 'failed'
  dry_run BOOLEAN NOT NULL,
  catch_up BOOLEAN NOT NULL,
  decision_id TEXT,          -- the decision this run acted on, if any
  message TEXT
);

CREATE TABLE IF NOT EXISTS live_decisions (
  decision_id TEXT PRIMARY KEY,     -- f"{strategy}:{decision_date}"
  strategy TEXT NOT NULL,
  decision_date DATE NOT NULL,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL,
  status TEXT NOT NULL,             -- 'planned' | 'submitted' | 'completed' | 'failed'
  target_weights_json TEXT NOT NULL,
  equity DOUBLE,
  cash_buffer DOUBLE,
  data_hash TEXT,
  qr_commit TEXT,
  research_commit TEXT
);

CREATE TABLE IF NOT EXISTS live_orders (
  order_id TEXT PRIMARY KEY,        -- the broker's order id
  decision_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity DOUBLE NOT NULL,
  planned_price DOUBLE NOT NULL,
  submitted_at TIMESTAMP NOT NULL,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_fills (
  fill_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL,
  decision_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity DOUBLE NOT NULL,
  price DOUBLE NOT NULL,
  commission DOUBLE NOT NULL,
  filled_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS live_positions (
  run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  as_of TIMESTAMP NOT NULL,
  quantity DOUBLE NOT NULL,
  market_value DOUBLE NOT NULL,
  PRIMARY KEY (run_id, symbol)
);

CREATE TABLE IF NOT EXISTS live_alerts (
  alert_id TEXT PRIMARY KEY,
  run_id TEXT,
  decision_id TEXT,
  created_at TIMESTAMP NOT NULL,
  severity TEXT NOT NULL,     -- 'warning' | 'error'
  message TEXT NOT NULL
);
"""

PLANNED = "planned"
SUBMITTED = "submitted"
COMPLETED = "completed"
FAILED = "failed"


def decision_id(strategy: str, decision_date: date) -> str:
    """The deterministic id of a decision: the same strategy and date always give one."""
    return f"{strategy}:{decision_date:%Y-%m-%d}"


def journal_path(data_directory: Path | str | None = None) -> Path:
    """Path of the journal database inside the data directory."""
    return data_dir(data_directory) / "live.duckdb"


def init_schema(data_directory: Path | str | None = None) -> None:
    """Create the journal database and its tables, if they do not exist yet."""
    with storage.connect(journal_path(data_directory)) as con:
        con.execute(SCHEMA)


@dataclass(frozen=True)
class DecisionRecord:
    """One row of ``live_decisions``."""

    decision_id: str
    strategy: str
    decision_date: date
    created_at: datetime
    updated_at: datetime
    status: str
    target_weights: dict[str, float]
    equity: float | None
    cash_buffer: float | None
    data_hash: str | None
    qr_commit: str | None
    research_commit: str | None


def _row_to_decision(row: pd.Series) -> DecisionRecord:
    return DecisionRecord(
        decision_id=row["decision_id"],
        strategy=row["strategy"],
        decision_date=pd.Timestamp(row["decision_date"]).date(),
        created_at=pd.Timestamp(row["created_at"]).to_pydatetime(),
        updated_at=pd.Timestamp(row["updated_at"]).to_pydatetime(),
        status=row["status"],
        target_weights=json.loads(row["target_weights_json"]),
        equity=None if pd.isna(row["equity"]) else float(row["equity"]),
        cash_buffer=None if pd.isna(row["cash_buffer"]) else float(row["cash_buffer"]),
        data_hash=row["data_hash"] if not pd.isna(row["data_hash"]) else None,
        qr_commit=row["qr_commit"] if not pd.isna(row["qr_commit"]) else None,
        research_commit=row["research_commit"] if not pd.isna(row["research_commit"]) else None,
    )


def find_decision(
    strategy: str, decision_date: date, data_directory: Path | str | None = None
) -> DecisionRecord | None:
    """The journaled decision for ``strategy`` and ``decision_date``, if one exists."""
    path = journal_path(data_directory)
    if not path.exists():
        return None
    with storage.connect(path, read_only=True) as con:
        frame = con.execute(
            "SELECT * FROM live_decisions WHERE decision_id = ?",
            [decision_id(strategy, decision_date)],
        ).df()
    return None if frame.empty else _row_to_decision(frame.iloc[0])


def read_decisions(strategy: str, data_directory: Path | str | None = None) -> pd.DataFrame:
    """Every decision journaled for ``strategy``, oldest first."""
    path = journal_path(data_directory)
    columns = [
        "decision_id",
        "strategy",
        "decision_date",
        "created_at",
        "updated_at",
        "status",
        "target_weights_json",
        "equity",
        "cash_buffer",
        "data_hash",
        "qr_commit",
        "research_commit",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    with storage.connect(path, read_only=True) as con:
        return con.execute(
            "SELECT * FROM live_decisions WHERE strategy = ? ORDER BY decision_date", [strategy]
        ).df()


def record_decision(
    strategy: str,
    decision_date: date,
    target_weights: dict[str, float],
    *,
    equity: float | None = None,
    cash_buffer: float | None = None,
    data_hash: str | None = None,
    qr_commit: str | None = None,
    research_commit: str | None = None,
    now: datetime | None = None,
    data_directory: Path | str | None = None,
) -> str:
    """Create a decision row with status ``planned``. Returns its id.

    Raises :class:`ValueError` if this decision already exists: the caller must check
    :func:`find_decision` first, which is exactly what makes a run idempotent.
    """
    did = decision_id(strategy, decision_date)
    now = now or storage.utcnow()
    init_schema(data_directory)
    with storage.connect(journal_path(data_directory)) as con:
        existing = con.execute(
            "SELECT count(*) FROM live_decisions WHERE decision_id = ?", [did]
        ).fetchone()
        if existing and existing[0]:
            raise ValueError(f"decision {did} already exists; check find_decision() first")
        con.execute(
            """
            INSERT INTO live_decisions
              (decision_id, strategy, decision_date, created_at, updated_at, status,
               target_weights_json, equity, cash_buffer, data_hash, qr_commit, research_commit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                did,
                strategy,
                decision_date,
                now,
                now,
                PLANNED,
                json.dumps(target_weights, sort_keys=True),
                equity,
                cash_buffer,
                data_hash,
                qr_commit,
                research_commit,
            ],
        )
    return did


def set_decision_status(
    did: str, status: str, *, now: datetime | None = None, data_directory: Path | str | None = None
) -> None:
    """Update a decision's status (``planned`` -> ``submitted`` -> ``completed``/``failed``)."""
    init_schema(data_directory)
    with storage.connect(journal_path(data_directory)) as con:
        con.execute(
            "UPDATE live_decisions SET status = ?, updated_at = ? WHERE decision_id = ?",
            [status, now or storage.utcnow(), did],
        )


def record_order(
    order_id: str,
    did: str,
    symbol: str,
    side: str,
    quantity: float,
    planned_price: float,
    status: str,
    *,
    submitted_at: datetime | None = None,
    data_directory: Path | str | None = None,
) -> None:
    """Append one row to ``live_orders``. ``order_id`` is the broker's own id."""
    init_schema(data_directory)
    with storage.connect(journal_path(data_directory)) as con:
        con.execute(
            """
            INSERT INTO live_orders
              (order_id, decision_id, symbol, side, quantity, planned_price, submitted_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (order_id) DO UPDATE SET status = excluded.status
            """,
            [
                order_id,
                did,
                symbol,
                side,
                quantity,
                planned_price,
                submitted_at or storage.utcnow(),
                status,
            ],
        )


def read_orders(did: str, data_directory: Path | str | None = None) -> pd.DataFrame:
    """Every order journaled for a decision, in the order they were submitted."""
    path = journal_path(data_directory)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "order_id",
                "decision_id",
                "symbol",
                "side",
                "quantity",
                "planned_price",
                "submitted_at",
                "status",
            ]
        )
    with storage.connect(path, read_only=True) as con:
        return con.execute(
            "SELECT * FROM live_orders WHERE decision_id = ? ORDER BY submitted_at", [did]
        ).df()


def record_fill(
    order_id: str,
    did: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    commission: float,
    *,
    filled_at: datetime | None = None,
    data_directory: Path | str | None = None,
) -> None:
    """Append one row to ``live_fills``, idempotently keyed by the order (one fill each)."""
    init_schema(data_directory)
    with storage.connect(journal_path(data_directory)) as con:
        con.execute(
            """
            INSERT INTO live_fills
              (fill_id, order_id, decision_id, symbol, side, quantity, price, commission, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (fill_id) DO NOTHING
            """,
            [
                f"F-{order_id}",
                order_id,
                did,
                symbol,
                side,
                quantity,
                price,
                commission,
                filled_at or storage.utcnow(),
            ],
        )


def read_fills(did: str, data_directory: Path | str | None = None) -> pd.DataFrame:
    """Every fill journaled for a decision."""
    path = journal_path(data_directory)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "fill_id",
                "order_id",
                "decision_id",
                "symbol",
                "side",
                "quantity",
                "price",
                "commission",
                "filled_at",
            ]
        )
    with storage.connect(path, read_only=True) as con:
        return con.execute(
            "SELECT * FROM live_fills WHERE decision_id = ? ORDER BY filled_at", [did]
        ).df()


def record_positions(
    run_id: str,
    positions: dict[str, tuple[float, float]],  # symbol -> (quantity, market_value)
    *,
    as_of: datetime | None = None,
    data_directory: Path | str | None = None,
) -> None:
    """Snapshot every held position after a run, for reconciliation and reporting."""
    as_of = as_of or storage.utcnow()
    init_schema(data_directory)
    with storage.connect(journal_path(data_directory)) as con:
        con.executemany(
            """
            INSERT INTO live_positions (run_id, symbol, as_of, quantity, market_value)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (run_id, symbol) DO UPDATE SET
              as_of = excluded.as_of, quantity = excluded.quantity,
              market_value = excluded.market_value
            """,
            [[run_id, symbol, as_of, qty, value] for symbol, (qty, value) in positions.items()],
        )


def read_positions(run_id: str, data_directory: Path | str | None = None) -> pd.DataFrame:
    """A run's position snapshot."""
    path = journal_path(data_directory)
    if not path.exists():
        return pd.DataFrame(columns=["run_id", "symbol", "as_of", "quantity", "market_value"])
    with storage.connect(path, read_only=True) as con:
        return con.execute(
            "SELECT * FROM live_positions WHERE run_id = ? ORDER BY symbol", [run_id]
        ).df()


def record_alert(
    message: str,
    *,
    severity: str = "error",
    run_id: str | None = None,
    did: str | None = None,
    now: datetime | None = None,
    data_directory: Path | str | None = None,
) -> None:
    """Append one row to ``live_alerts``."""
    init_schema(data_directory)
    with storage.connect(journal_path(data_directory)) as con:
        con.execute(
            "INSERT INTO live_alerts VALUES (?, ?, ?, ?, ?, ?)",
            [uuid.uuid4().hex, run_id, did, now or storage.utcnow(), severity, message],
        )


def read_alerts(data_directory: Path | str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Every alert, newest first."""
    path = journal_path(data_directory)
    if not path.exists():
        return pd.DataFrame(
            columns=["alert_id", "run_id", "decision_id", "created_at", "severity", "message"]
        )
    query = "SELECT * FROM live_alerts ORDER BY created_at DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with storage.connect(path, read_only=True) as con:
        return con.execute(query).df()


def record_run(
    run_id: str,
    config_name: str,
    *,
    dry_run: bool,
    catch_up: bool,
    started_at: datetime | None = None,
    did: str | None = None,
    data_directory: Path | str | None = None,
) -> None:
    """Start a run row, with status ``running``."""
    init_schema(data_directory)
    with storage.connect(journal_path(data_directory)) as con:
        con.execute(
            """
            INSERT INTO live_runs
              (run_id, config_name, started_at, status, dry_run, catch_up, decision_id)
            VALUES (?, ?, ?, 'running', ?, ?, ?)
            """,
            [run_id, config_name, started_at or storage.utcnow(), dry_run, catch_up, did],
        )


def finish_run(
    run_id: str,
    status: str,
    *,
    message: str = "",
    finished_at: datetime | None = None,
    data_directory: Path | str | None = None,
) -> None:
    """Close a run row with its final status (``ok`` or ``failed``)."""
    with storage.connect(journal_path(data_directory)) as con:
        con.execute(
            "UPDATE live_runs SET status = ?, message = ?, finished_at = ? WHERE run_id = ?",
            [status, message, finished_at or storage.utcnow(), run_id],
        )


def read_runs(data_directory: Path | str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Every run, newest first."""
    path = journal_path(data_directory)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "run_id",
                "config_name",
                "started_at",
                "finished_at",
                "status",
                "dry_run",
                "catch_up",
                "decision_id",
                "message",
            ]
        )
    query = "SELECT * FROM live_runs ORDER BY started_at DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with storage.connect(path, read_only=True) as con:
        return con.execute(query).df()
