"""Data quality checks.

Each ``check_*`` function is pure: it takes one symbol's unadjusted bars and returns a
list of findings. ``validate_symbols`` runs them all against the database and replaces
the stored findings, so the ``quality_issues`` table always shows the current state.
Two kinds of finding are written by ingestion instead and are never cleared, because
they record events in a provider response rather than the state of stored data:
``revision`` (the provider changed history) and ``duplicate_bar`` (a response repeated a
date; only one row can be stored).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from qr import calendars, storage
from qr.adjust import split_factors
from qr.config import UniverseConfig, ValidationSettings
from qr.frames import PRICE_COLUMNS, VALUE_COLUMNS
from qr.storage import Connection, QualityIssue

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"

# Yahoo prices are float32 underneath, so OHLC can disagree by a few parts in 1e8.
OHLC_RELATIVE_TOLERANCE = 1e-6

# Checks owned by this module; their findings are recomputed on every validation run.
STATE_CHECKS = (
    "missing_values",
    "ohlc_consistency",
    "non_positive_price",
    "negative_volume",
    "gap",
    "unexpected_session",
    "outlier_return",
    "stale_data",
    "zero_volume_streak",
)
REVISION_CHECK = "revision"
DUPLICATE_CHECK = "duplicate_bar"


def _day(value: Any) -> date:
    day: date = pd.Timestamp(value).date()
    return day


def check_missing_values(symbol: str, bars: pd.DataFrame) -> list[QualityIssue]:
    """Any NaN in open/high/low/close/volume."""
    bad = bars[bars[VALUE_COLUMNS].isna().any(axis=1)]
    return [
        QualityIssue(
            symbol,
            _day(row.date),
            "missing_values",
            ERROR,
            "missing: " + ", ".join(c for c in VALUE_COLUMNS if pd.isna(getattr(row, c))),
        )
        for row in bad.itertuples()
    ]


def check_ohlc_consistency(symbol: str, bars: pd.DataFrame) -> list[QualityIssue]:
    """``high >= max(open, close)`` and ``low <= min(open, close)`` (and ``high >= low``)."""
    top = bars[["open", "close"]].max(axis=1)
    bottom = bars[["open", "close"]].min(axis=1)
    slack = OHLC_RELATIVE_TOLERANCE * bars["close"].abs()
    bad = bars[(bars["high"] < top - slack) | (bars["low"] > bottom + slack)]
    return [
        QualityIssue(
            symbol,
            _day(row.date),
            "ohlc_consistency",
            ERROR,
            f"open={row.open} high={row.high} low={row.low} close={row.close}",
        )
        for row in bad.itertuples()
    ]


def check_non_positive_prices(symbol: str, bars: pd.DataFrame) -> list[QualityIssue]:
    """Any open/high/low/close that is zero or negative."""
    bad = bars[(bars[PRICE_COLUMNS] <= 0).any(axis=1)]
    return [
        QualityIssue(
            symbol,
            _day(row.date),
            "non_positive_price",
            ERROR,
            f"open={row.open} high={row.high} low={row.low} close={row.close}",
        )
        for row in bad.itertuples()
    ]


def check_negative_volume(symbol: str, bars: pd.DataFrame) -> list[QualityIssue]:
    """Any volume below zero."""
    bad = bars[bars["volume"] < 0]
    return [
        QualityIssue(symbol, _day(row.date), "negative_volume", ERROR, f"volume={row.volume}")
        for row in bad.itertuples()
    ]


def check_duplicates(symbol: str, bars: pd.DataFrame) -> list[QualityIssue]:
    """More than one bar for the same date.

    The database's primary key makes this impossible for stored data, so the check
    matters for provider responses, which ingestion runs it on before upserting.
    """
    counts = bars.groupby("date").size()
    return [
        QualityIssue(symbol, _day(day), DUPLICATE_CHECK, ERROR, f"{count} bars for the same date")
        for day, count in counts[counts > 1].items()
    ]


def check_gaps(symbol: str, bars: pd.DataFrame, asset_class: str) -> list[QualityIssue]:
    """Expected sessions between the first and last bar that have no bar.

    Consecutive missing sessions are reported as one finding dated at the first of them.
    """
    if bars.empty:
        return []
    expected = calendars.sessions(asset_class, _day(bars["date"].min()), _day(bars["date"].max()))
    missing = ~expected.isin(bars["date"])
    issues: list[QualityIssue] = []
    run_start: int | None = None
    for position, is_missing in enumerate([*missing, False]):
        if is_missing and run_start is None:
            run_start = position
        elif not is_missing and run_start is not None:
            first, last = expected[run_start], expected[position - 1]
            count = position - run_start
            details = (
                "1 expected session missing"
                if count == 1
                else f"{count} expected sessions missing through {last.date()}"
            )
            issues.append(QualityIssue(symbol, first.date(), "gap", WARNING, details))
            run_start = None
    return issues


def check_unexpected_sessions(
    symbol: str, bars: pd.DataFrame, asset_class: str
) -> list[QualityIssue]:
    """Bars dated on days the calendar says the market was closed."""
    if bars.empty:
        return []
    expected = calendars.sessions(asset_class, _day(bars["date"].min()), _day(bars["date"].max()))
    extra = bars[~bars["date"].isin(expected)]
    return [
        QualityIssue(symbol, _day(day), "unexpected_session", WARNING, "bar on a non-trading day")
        for day in extra["date"]
    ]


def check_outlier_returns(
    symbol: str, bars: pd.DataFrame, actions: pd.DataFrame, threshold: float
) -> list[QualityIssue]:
    """Daily close-to-close moves larger than ``threshold`` that no known split explains.

    Returns are measured on split-adjusted closes, so a recorded split produces no
    finding, while a split missing from ``corporate_actions`` (or prices that did not
    actually split) does.
    """
    if len(bars) < 2:
        return []
    adjusted = bars["close"].to_numpy(dtype="float64") * split_factors(bars["date"], actions)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = adjusted[1:] / adjusted[:-1] - 1.0
    issues: list[QualityIssue] = []
    for position in np.flatnonzero(np.abs(returns) > threshold):
        row = bars.iloc[position + 1]
        issues.append(
            QualityIssue(
                symbol,
                _day(row["date"]),
                "outlier_return",
                WARNING,
                f"split-adjusted return {returns[position]:+.2%} exceeds {threshold:.0%}",
            )
        )
    return issues


def check_stale(
    symbol: str, bars: pd.DataFrame, asset_class: str, max_age_sessions: int, now: datetime
) -> list[QualityIssue]:
    """The newest bar is more than ``max_age_sessions`` sessions behind the calendar."""
    if bars.empty:
        return [QualityIssue(symbol, None, "stale_data", WARNING, "no bars stored")]
    last = _day(bars["date"].max())
    latest = calendars.last_completed_session(asset_class, now)
    behind = calendars.sessions_between(asset_class, last, latest)
    if behind <= max_age_sessions:
        return []
    details = f"last bar is {behind} sessions behind {latest}"
    return [QualityIssue(symbol, last, "stale_data", WARNING, details)]


def check_zero_volume_streaks(
    symbol: str, bars: pd.DataFrame, max_streak: int
) -> list[QualityIssue]:
    """Runs of consecutive zero-volume bars longer than ``max_streak``."""
    issues: list[QualityIssue] = []
    is_zero = (bars["volume"] == 0).to_numpy()
    run_start: int | None = None
    for position, zero in enumerate([*is_zero, False]):
        if zero and run_start is None:
            run_start = position
        elif not zero and run_start is not None:
            length = position - run_start
            if length > max_streak:
                first = _day(bars["date"].iloc[run_start])
                last = _day(bars["date"].iloc[position - 1])
                issues.append(
                    QualityIssue(
                        symbol,
                        first,
                        "zero_volume_streak",
                        WARNING,
                        f"{length} consecutive zero-volume bars through {last}",
                    )
                )
            run_start = None
    return issues


def run_checks(
    symbol: str,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    asset_class: str,
    settings: ValidationSettings,
    now: datetime,
    check_staleness: bool = True,
) -> list[QualityIssue]:
    """Run every check on one symbol's unadjusted bars (sorted by date)."""
    issues = [
        *check_missing_values(symbol, bars),
        *check_ohlc_consistency(symbol, bars),
        *check_non_positive_prices(symbol, bars),
        *check_negative_volume(symbol, bars),
        *check_duplicates(symbol, bars),
        *check_gaps(symbol, bars, asset_class),
        *check_unexpected_sessions(symbol, bars, asset_class),
        *check_outlier_returns(symbol, bars, actions, settings.outlier_return_threshold),
        *check_zero_volume_streaks(symbol, bars, settings.max_zero_volume_streak),
    ]
    if check_staleness:
        issues += check_stale(symbol, bars, asset_class, settings.stale_after_trading_days, now)
    return issues


def validate_symbols(
    con: Connection,
    config: UniverseConfig,
    symbols: Sequence[str] | None = None,
    now: datetime | None = None,
) -> list[QualityIssue]:
    """Run all checks for stored instruments and replace their findings in the database.

    ``symbols`` defaults to every instrument in the database. Inactive instruments are
    checked too, except for staleness (a delisted symbol is expected to stop updating).
    """
    now = now or storage.utcnow()
    instruments = storage.read_instruments(con)
    if symbols is not None:
        unknown = sorted(set(symbols) - set(instruments["symbol"]))
        if unknown:
            raise ValueError(f"unknown symbols: {', '.join(unknown)}")
        instruments = instruments[instruments["symbol"].isin(list(symbols))]

    found: list[QualityIssue] = []
    for instrument in instruments.itertuples():
        symbol = str(instrument.symbol)
        bars = storage.read_bars(con, [symbol])
        actions = storage.read_corporate_actions(con, [symbol])
        issues = run_checks(
            symbol,
            bars,
            actions,
            str(instrument.asset_class),
            config.validation,
            now,
            check_staleness=bool(instrument.is_active),
        )
        with storage.transaction(con):
            storage.delete_quality_issues(con, symbol, STATE_CHECKS)
            storage.insert_quality_issues(con, issues, now)
        logger.info("validated %s: %d findings", symbol, len(issues))
        found += issues
    return found


def summarize(issues: pd.DataFrame) -> pd.DataFrame:
    """Group stored findings (as read from ``quality_issues``) for a readable summary.

    Columns: symbol, check_name, severity, count, first_date, last_date.
    """
    columns = ["symbol", "check_name", "severity", "count", "first_date", "last_date"]
    if issues.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        issues.groupby(["symbol", "check_name", "severity"], sort=True)
        .agg(count=("check_name", "size"), first_date=("date", "min"), last_date=("date", "max"))
        .reset_index()
    )
    return summary[columns]
