"""Incremental download orchestration.

For every instrument: work out the date window, fetch, snapshot the raw response, record
provider revisions, upsert, and log the outcome. One symbol failing never stops the run.

A provider that is down is a different matter: retrying every symbol against it would
take half a minute each. So each source gets a connectivity probe before the run and a
circuit breaker during it (see ``run_ingest``).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from qr import calendars, snapshots, storage
from qr.config import Instrument, UniverseConfig, db_path, raw_dir
from qr.frames import VALUE_COLUMNS, normalize_actions, normalize_bars
from qr.sources.base import DataSource, SourceError
from qr.storage import Connection, QualityIssue
from qr.validate import REVISION_CHECK, WARNING, check_duplicates, validate_symbols

logger = logging.getLogger(__name__)

OVERLAP_SESSIONS = 5  # re-fetch this many sessions before the last stored bar

OK = "ok"
PARTIAL = "partial"
FAILED = "failed"
SKIPPED = "skipped"


@dataclass(frozen=True)
class SymbolResult:
    """Outcome of ingesting one symbol."""

    symbol: str
    status: str  # ok | partial | failed | skipped
    rows_written: int = 0
    message: str = ""


@dataclass
class IngestReport:
    """Outcome of a whole run."""

    run_id: str
    results: list[SymbolResult] = field(default_factory=list)
    # Sources found unreachable, with the reason; their symbols were skipped.
    sources_down: dict[str, str] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """True if any symbol failed or came back incomplete, or a source was down."""
        return bool(self.sources_down) or any(r.status in (FAILED, PARTIAL) for r in self.results)


def build_source(name: str, config: UniverseConfig) -> DataSource:
    """Construct the data source called ``name`` from the config."""
    if name == "yahoo":
        from qr.sources.yahoo import YahooSource

        return YahooSource(max_retries=config.sources.yahoo.max_retries)
    if name == "crypto":
        from qr.sources.crypto import CryptoSource

        return CryptoSource(exchange=config.sources.crypto.exchange)
    raise ValueError(f"unknown source: {name}")


def fetch_window(
    last_stored: date | None, asset_class: str, start_date: date, now: datetime
) -> tuple[date, date]:
    """The inclusive ``(start, end)`` date range to request for a symbol.

    First run: from the configured ``start_date``. Later runs: from five sessions before
    the last stored bar, to catch late corrections. The end is always the last *completed*
    session, so a bar that is still forming is never stored.
    """
    end = calendars.last_completed_session(asset_class, now)
    if last_stored is None:
        return start_date, end
    return max(start_date, calendars.sessions_back(asset_class, last_stored, OVERLAP_SESSIONS)), end


def detect_revisions(
    symbol: str,
    stored_bars: pd.DataFrame,
    new_bars: pd.DataFrame,
    stored_actions: pd.DataFrame,
    new_actions: pd.DataFrame,
    tolerance: float,
) -> list[QualityIssue]:
    """Findings for values the provider changed since we last stored them.

    A value counts as revised when it moved by more than ``tolerance`` relative to the
    stored value.
    """
    issues: list[QualityIssue] = []
    both = stored_bars.merge(new_bars, on="date", suffixes=("_old", "_new"))
    for row in both.to_dict("records"):
        changes = []
        for column in VALUE_COLUMNS:
            old, new = float(row[f"{column}_old"]), float(row[f"{column}_new"])
            if pd.isna(old) and pd.isna(new):
                continue
            if pd.isna(old) or pd.isna(new) or abs(new - old) > tolerance * abs(old):
                changes.append(f"{column} {old:g} -> {new:g}")
        if changes:
            day = pd.Timestamp(row["date"]).date()
            issues.append(QualityIssue(symbol, day, REVISION_CHECK, WARNING, "; ".join(changes)))

    actions = stored_actions.merge(new_actions, on=["date", "action"], suffixes=("_old", "_new"))
    for row in actions.to_dict("records"):
        old, new = float(row["value_old"]), float(row["value_new"])
        if abs(new - old) > tolerance * abs(old):
            day = pd.Timestamp(row["date"]).date()
            details = f"{row['action']} {old:g} -> {new:g}"
            issues.append(QualityIssue(symbol, day, REVISION_CHECK, WARNING, details))
    return issues


def _ingest_symbol(
    con: Connection,
    instrument: Instrument,
    source: DataSource,
    config: UniverseConfig,
    raw_root: Path,
    now: datetime,
) -> SymbolResult:
    """Fetch, snapshot, and store one symbol. Raises on failure."""
    symbol = instrument.symbol
    last_stored = storage.last_bar_date(con, symbol)
    start, end = fetch_window(last_stored, instrument.asset_class, config.start_date, now)
    if start > end:
        return SymbolResult(symbol, OK, 0, f"nothing to fetch ({start} is after {end})")

    logger.info("%s: fetching %s to %s from %s", symbol, start, end, source.name)
    try:
        bars = normalize_bars(source.fetch_daily(symbol, start, end))
        actions = normalize_actions(source.fetch_corporate_actions(symbol))
    except SourceError:
        raise
    except Exception as exc:  # anything a provider throws is a download failure
        raise SourceError(f"{type(exc).__name__}: {exc}") from exc
    if bars.empty:
        raise SourceError(f"{source.name} returned no rows for {symbol}")

    # Raw response goes to disk before anything can alter or reject it.
    snapshots.write_snapshot(raw_root, source.name, symbol, bars, actions, now)

    problems: list[str] = []
    issues: list[QualityIssue] = check_duplicates(symbol, bars)
    if issues:
        problems.append(f"{len(issues)} duplicated dates (kept the last of each)")
        bars = bars.drop_duplicates("date", keep="last").reset_index(drop=True)
    incomplete = bars[VALUE_COLUMNS].isna().any(axis=1)
    if incomplete.any():
        problems.append(f"{int(incomplete.sum())} rows with missing values (dropped)")
        bars = bars[~incomplete].reset_index(drop=True)
    if bars.empty:
        raise SourceError(f"{source.name} returned no usable rows for {symbol}")

    # On the first run we cannot know when the symbol started trading, so coverage is
    # only checked from the first bar the provider actually returned.
    covered_from = start if last_stored is not None else pd.Timestamp(bars["date"].min()).date()
    expected = calendars.sessions(instrument.asset_class, covered_from, end)
    missing = expected[~expected.isin(bars["date"])]
    if len(missing):
        problems.append(
            f"{len(missing)} of {len(expected)} expected sessions missing "
            f"(first {missing[0].date()}, last {missing[-1].date()})"
        )

    issues += detect_revisions(
        symbol,
        storage.read_bars(con, [symbol], start, end),
        bars,
        storage.read_corporate_actions(con, [symbol]),
        actions,
        config.validation.revision_tolerance,
    )
    with storage.transaction(con):
        written = storage.upsert_bars(con, symbol, source.name, bars, now)
        storage.upsert_corporate_actions(con, symbol, actions)
        storage.refresh_coverage(con, symbol, now)
        storage.insert_quality_issues(con, issues, now)

    if problems:
        return SymbolResult(symbol, PARTIAL, written, "; ".join(problems))
    return SymbolResult(symbol, OK, written, f"{start} to {end}")


def run_ingest(
    config: UniverseConfig,
    data_directory: Path | str | None = None,
    symbols: Sequence[str] | None = None,
    sources: Mapping[str, DataSource] | None = None,
    clock: Callable[[], datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> IngestReport:
    """Incrementally download the universe (or just ``symbols``) into the local store.

    ``sources`` overrides the providers built from the config (used by tests), ``clock``
    supplies naive-UTC "now" (default: ``storage.utcnow``). Quality checks run afterwards
    for every symbol that was written. Check ``report.failed`` to decide the exit code.

    Failing fast when a provider is down:

    - Before anything is downloaded, every source that is about to be used is probed with
      one cheap request. If the probe fails, all of that source's symbols are skipped.
    - During the run, a download failure cannot be told apart reliably from a bad ticker
      (Yahoo answers both with "no timezone found"). So failures are counted per source,
      and a success resets the count. When ``ingest.max_consecutive_failures`` symbols in
      a row have failed, the source is probed again: if the probe fails too the source
      is down and its remaining symbols are skipped; if it succeeds the failures were
      about those symbols and the run carries on.

    Either way the source ends up in ``report.sources_down`` and the run counts as failed.
    """
    report = IngestReport(run_id=uuid.uuid4().hex[:12])
    clock = clock or storage.utcnow
    built: dict[str, DataSource] = dict(sources or {})
    raw_root = raw_dir(data_directory)

    with storage.connect(db_path(data_directory)) as con:
        storage.init_schema(con)
        now = clock()
        for instrument in config.universe:
            storage.upsert_instrument(
                con,
                instrument.symbol,
                instrument.source,
                instrument.asset_class,
                config.source_enabled(instrument.source),
                now,
            )
        storage.deactivate_instruments_not_in(con, [i.symbol for i in config.universe], now)

        selected: list[Instrument] = []
        for symbol in symbols if symbols is not None else [i.symbol for i in config.universe]:
            found = config.instrument(symbol)
            if found is None:
                message = "not in the configured universe"
                logger.error("%s: %s", symbol, message)
                storage.log_ingestion(con, report.run_id, symbol, now, now, 0, FAILED, message)
                report.results.append(SymbolResult(symbol, FAILED, 0, message))
            else:
                selected.append(found)

        def probe_failure(name: str) -> str | None:
            """Why ``name`` is unreachable, or None if it answered (or cannot be built)."""
            try:
                if name not in built:
                    built[name] = build_source(name, config)
            except Exception:
                return None  # a set-up problem, reported per symbol below
            try:
                built[name].probe()
            except Exception as exc:
                return str(exc)
            return None

        down = report.sources_down
        for name in dict.fromkeys(i.source for i in selected if config.source_enabled(i.source)):
            reason = probe_failure(name)
            if reason is not None:
                down[name] = reason
                logger.error("source '%s' is unreachable, skipping its symbols: %s", name, reason)

        fetched_from: set[str] = set()
        failures_in_a_row: dict[str, int] = {}
        for instrument in selected:
            symbol = instrument.symbol
            name = instrument.source
            started = clock()
            if not config.source_enabled(name):
                result = SymbolResult(symbol, SKIPPED, 0, f"source '{name}' disabled")
                logger.info("%s: skipped (%s)", symbol, result.message)
            elif name in down:
                result = SymbolResult(symbol, SKIPPED, 0, f"source '{name}' is down")
                logger.warning("%s: skipped (%s)", symbol, result.message)
            else:
                if name == "yahoo" and "yahoo" in fetched_from:
                    sleep(config.sources.yahoo.request_delay_seconds)
                fetched_from.add(name)
                download_failed = False
                try:
                    if name not in built:
                        built[name] = build_source(name, config)
                    result = _ingest_symbol(con, instrument, built[name], config, raw_root, started)
                except SourceError as exc:
                    download_failed = True
                    result = SymbolResult(symbol, FAILED, 0, f"{type(exc).__name__}: {exc}")
                except Exception as exc:  # fault isolation: log it and move on
                    result = SymbolResult(symbol, FAILED, 0, f"{type(exc).__name__}: {exc}")

                failures_in_a_row[name] = (
                    failures_in_a_row.get(name, 0) + 1 if download_failed else 0
                )
                if failures_in_a_row[name] >= config.ingest.max_consecutive_failures:
                    reason = probe_failure(name)
                    if reason is None:
                        logger.info("source '%s' still answers; the failures are per symbol", name)
                        failures_in_a_row[name] = 0
                    else:
                        down[name] = (
                            f"{failures_in_a_row[name]} symbols in a row failed, then: {reason}"
                        )
                        logger.error("source '%s' is down, skipping the rest: %s", name, reason)
                if result.status == FAILED:
                    logger.error("%s: failed: %s", symbol, result.message)
                elif result.status == PARTIAL:
                    logger.error("%s: partial response: %s", symbol, result.message)
                else:
                    logger.info("%s: wrote %d rows", symbol, result.rows_written)
            storage.log_ingestion(
                con,
                report.run_id,
                symbol,
                started,
                clock(),
                result.rows_written,
                result.status,
                result.message,
            )
            report.results.append(result)

        written = [r.symbol for r in report.results if r.status in (OK, PARTIAL)]
        if written:
            validate_symbols(con, config, written, clock())
    return report
