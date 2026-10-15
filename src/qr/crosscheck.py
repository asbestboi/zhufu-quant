"""Compare stored data with a second, independent source.

Nothing here changes ``daily_bars``: the second source is a witness, not a replacement.
For each symbol the unadjusted closes and the split events of a recent window are
compared, and every disagreement becomes a ``crosscheck`` warning in ``quality_issues``.
A disagreement does not say which source is wrong, only that one of them is.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from qr import calendars, snapshots, storage
from qr.config import UniverseConfig, db_path, raw_dir
from qr.frames import SPLIT, normalize_actions, normalize_bars
from qr.sources.base import DataSource, SourceError, SourceUnavailableError
from qr.storage import QualityIssue
from qr.validate import WARNING

logger = logging.getLogger(__name__)

CROSSCHECK_CHECK = "crosscheck"
SPLIT_TOLERANCE = 1e-6  # relative; split ratios should agree exactly

OK = "ok"
MISMATCH = "mismatch"
FAILED = "failed"
SKIPPED = "skipped"


@dataclass(frozen=True)
class SymbolCrosscheck:
    """Outcome of cross-checking one symbol."""

    symbol: str
    status: str  # ok | mismatch | failed | skipped
    days_compared: int = 0
    mismatches: int = 0  # closes beyond the tolerance plus mismatched splits
    worst_difference: float = 0.0  # largest relative difference between two closes
    worst_date: date | None = None
    unmatched_days: int = 0  # days only one of the two sources has a bar for
    message: str = ""


@dataclass
class CrosscheckReport:
    """Outcome of a whole cross-check run."""

    source: str
    start: date
    end: date
    results: list[SymbolCrosscheck] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """True if the comparison source could not be read for some symbol."""
        return any(r.status == FAILED for r in self.results)


@dataclass(frozen=True)
class CloseComparison:
    """Result of :func:`compare_closes`."""

    issues: list[QualityIssue]
    days_compared: int
    worst_difference: float
    worst_date: date | None
    unmatched_days: int


def build_source(config: UniverseConfig, api_key: str) -> DataSource:
    """Construct the comparison source (Tiingo)."""
    from qr.sources.tiingo import TiingoSource

    return TiingoSource(api_key, max_retries=config.crosscheck.max_retries)


def compare_closes(
    symbol: str,
    stored: pd.DataFrame,
    other: pd.DataFrame,
    tolerance: float,
    stored_name: str = "yahoo",
    other_name: str = "tiingo",
) -> CloseComparison:
    """Compare unadjusted closes on the dates both frames have.

    A day is a mismatch when the closes differ by more than ``tolerance`` relative to the
    stored one.
    """
    both = stored[["date", "close"]].merge(
        other[["date", "close"]], on="date", how="outer", suffixes=("_stored", "_other")
    )
    matched = both.dropna(subset=["close_stored", "close_other"])
    unmatched = len(both) - len(matched)
    if matched.empty:
        return CloseComparison([], 0, 0.0, None, unmatched)

    difference = (matched["close_other"] - matched["close_stored"]) / matched["close_stored"]
    sizes = difference.abs().to_numpy()
    worst = int(sizes.argmax())
    issues = [
        QualityIssue(
            symbol,
            pd.Timestamp(row["date"]).date(),
            CROSSCHECK_CHECK,
            WARNING,
            f"close {stored_name} {row['close_stored']:g} vs {other_name} "
            f"{row['close_other']:g} ({diff:+.2%})",
        )
        for row, diff in zip(matched.to_dict("records"), difference, strict=True)
        if abs(diff) > tolerance
    ]
    return CloseComparison(
        issues,
        len(matched),
        float(sizes[worst]),
        pd.Timestamp(matched["date"].iloc[worst]).date(),
        unmatched,
    )


def compare_splits(
    symbol: str,
    stored_actions: pd.DataFrame,
    other_actions: pd.DataFrame,
    start: date,
    end: date,
    stored_name: str = "yahoo",
    other_name: str = "tiingo",
) -> list[QualityIssue]:
    """Findings for splits in ``[start, end]`` that only one source has, or that differ."""

    def splits(actions: pd.DataFrame) -> pd.DataFrame:
        in_window = (actions["date"] >= pd.Timestamp(start)) & (
            actions["date"] <= pd.Timestamp(end)
        )
        return actions.loc[in_window & (actions["action"] == SPLIT), ["date", "value"]]

    both = splits(stored_actions).merge(
        splits(other_actions), on="date", how="outer", suffixes=("_stored", "_other")
    )
    issues: list[QualityIssue] = []
    for row in both.sort_values("date").to_dict("records"):
        ours, theirs = float(row["value_stored"]), float(row["value_other"])
        if pd.isna(ours):
            details = f"split {other_name} {theirs:g} vs {stored_name} none"
        elif pd.isna(theirs):
            details = f"split {stored_name} {ours:g} vs {other_name} none"
        elif abs(theirs - ours) > SPLIT_TOLERANCE * abs(ours):
            details = f"split {stored_name} {ours:g} vs {other_name} {theirs:g}"
        else:
            continue
        day = pd.Timestamp(row["date"]).date()
        issues.append(QualityIssue(symbol, day, CROSSCHECK_CHECK, WARNING, details))
    return issues


def run_crosscheck(
    config: UniverseConfig,
    data_directory: Path | str | None = None,
    symbols: Sequence[str] | None = None,
    days: int = 365,
    source: DataSource | None = None,
    api_key: str = "",
    clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> CrosscheckReport:
    """Cross-check the last ``days`` calendar days of ``symbols`` (default: every stock and
    ETF in the config) against the comparison source.

    A :class:`~qr.sources.base.SourceError` is raised if the source is unreachable (it is
    probed first), rejects the key, or runs out of request allowance mid-run. Each
    response is saved as a raw snapshot under the comparison source's name. Findings for
    the window replace earlier ``crosscheck`` findings for the same window, so repeating
    a run does not pile up duplicates.
    """
    if days < 1:
        raise ValueError("days must be at least 1")
    now = (clock or storage.utcnow)()
    end = calendars.last_completed_session("equity", now)
    start = end - timedelta(days=days)
    other = source if source is not None else build_source(config, api_key)
    report = CrosscheckReport(other.name, start, end)
    raw_root = raw_dir(data_directory)

    wanted = (
        list(symbols)
        if symbols is not None
        else [i.symbol for i in config.universe if i.source == "yahoo"]
    )
    other.probe()

    with storage.connect(db_path(data_directory)) as con:
        storage.init_schema(con)
        fetched = False
        for symbol in wanted:
            instrument = config.instrument(symbol)
            if instrument is None:
                result = SymbolCrosscheck(symbol, FAILED, message="not in the configured universe")
            elif instrument.asset_class == "crypto":
                result = SymbolCrosscheck(
                    symbol, SKIPPED, message="only stocks and ETFs are checked"
                )
            else:
                stored = storage.read_bars(con, [symbol], start, end)
                if stored.empty:
                    message = "no stored bars in the window (run `qr ingest` first)"
                    result = SymbolCrosscheck(symbol, SKIPPED, message=message)
                else:
                    if fetched:
                        sleep(config.crosscheck.request_delay_seconds)
                    fetched = True
                    try:
                        bars = normalize_bars(other.fetch_daily(symbol, start, end))
                        actions = normalize_actions(other.fetch_corporate_actions(symbol))
                    except SourceUnavailableError:
                        raise  # bad key or used-up allowance: the other symbols would fail too
                    except Exception as exc:  # fault isolation: report it and move on
                        known = isinstance(exc, SourceError)
                        message = str(exc) if known else f"{type(exc).__name__}: {exc}"
                        result = SymbolCrosscheck(symbol, FAILED, message=message)
                    else:
                        snapshot = snapshots.write_snapshot(
                            raw_root, other.name, symbol, bars, actions, now
                        )
                        storage.record_snapshots(
                            con, snapshots.manifest_entries(raw_root, other.name, symbol, snapshot)
                        )
                        closes = compare_closes(
                            symbol,
                            stored,
                            bars.drop_duplicates("date", keep="last"),
                            config.crosscheck.close_tolerance,
                            instrument.source,
                            other.name,
                        )
                        issues = closes.issues + compare_splits(
                            symbol,
                            storage.read_corporate_actions(con, [symbol]),
                            actions,
                            start,
                            end,
                            instrument.source,
                            other.name,
                        )
                        with storage.transaction(con):
                            storage.delete_quality_issues(
                                con, symbol, [CROSSCHECK_CHECK], start, end
                            )
                            storage.insert_quality_issues(con, issues, now)
                        result = SymbolCrosscheck(
                            symbol,
                            MISMATCH if issues else OK,
                            closes.days_compared,
                            len(issues),
                            closes.worst_difference,
                            closes.worst_date,
                            closes.unmatched_days,
                        )
            if result.status == FAILED:
                logger.error("%s: cross-check failed: %s", symbol, result.message)
            else:
                logger.info(
                    "%s: %s (%d days compared, %d mismatches)",
                    symbol,
                    result.status,
                    result.days_compared,
                    result.mismatches,
                )
            report.results.append(result)
    return report
