"""Public loading API: the only module later steps (backtesting, research) should import.

All functions read from the local store, by default ``./data`` or ``$QR_DATA_DIR``;
pass ``data_dir=`` to point somewhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from qr import snapshots, storage
from qr.adjust import Adjustment, adjust_bars
from qr.config import db_path, raw_dir
from qr.fingerprint import DataFingerprint, combine, hash_symbol
from qr.frames import ACTION_COLUMNS, BAR_COLUMNS, VALUE_COLUMNS, to_timestamp


def _symbol_list(symbols: list[str] | str) -> list[str]:
    requested = [symbols] if isinstance(symbols, str) else list(symbols)
    if not requested:
        raise ValueError("no symbols given")
    return list(dict.fromkeys(requested))  # de-duplicate, keep order


def _naive_utc(moment: datetime) -> datetime:
    return moment.astimezone(UTC).replace(tzinfo=None) if moment.tzinfo else moment


@dataclass(frozen=True)
class _Stored:
    """One symbol's unadjusted bars up to ``end``, the corporate actions dated on or
    before its last bar, and when each bar was written (aligned with ``bars``)."""

    bars: pd.DataFrame
    actions: pd.DataFrame
    ingested_at: pd.Series


def _unadjusted(
    requested: list[str],
    end: date | None,
    as_of: datetime | None,
    data_dir: Path | str | None,
) -> dict[str, _Stored]:
    """What :func:`get_bars` adjusts, per symbol in the order requested.

    From the database, or with ``as_of`` from the raw snapshots taken by then (each bar
    then counts as written when the newest such snapshot was taken). Raises ``KeyError``
    for symbols that are not in the instruments table.
    """
    end_ts = to_timestamp(end) if end is not None else None
    with storage.connect(db_path(data_dir), read_only=True) as con:
        instruments = storage.read_instruments(con).set_index("symbol")
        unknown = [s for s in requested if s not in instruments.index]
        if unknown:
            raise KeyError(f"unknown symbols: {', '.join(unknown)}")
        if as_of is None:
            # Full history up to `end`: adjustment needs the bars before `start`.
            all_bars = storage.read_bars(con, requested, None, end)
            all_actions = storage.read_corporate_actions(con, requested)

    found: dict[str, _Stored] = {}
    for symbol in requested:
        if as_of is None:
            own = all_bars[all_bars["symbol"] == symbol].reset_index(drop=True)
            bars = own[BAR_COLUMNS]
            ingested = pd.Series(pd.to_datetime(own["ingested_at"]), dtype="datetime64[ns]")
            actions = all_actions[all_actions["symbol"] == symbol]
        else:
            source = str(instruments.loc[symbol, "source"])
            taken = snapshots.list_snapshots(raw_dir(data_dir), source, symbol)
            moment = _naive_utc(as_of)
            bars = snapshots.bars_as_of(taken, moment)[BAR_COLUMNS]
            bars = bars.dropna(subset=VALUE_COLUMNS).reset_index(drop=True)
            actions = snapshots.actions_as_of(taken, moment)
            known = [s.taken_at for s in taken if s.taken_at <= moment]
            latest = pd.Timestamp(known[-1]) if known else pd.NaT
            ingested = pd.Series([latest] * len(bars), dtype="datetime64[ns]")
        if end_ts is not None:
            keep = (bars["date"] <= end_ts).to_numpy()
            bars = bars[keep].reset_index(drop=True)
            ingested = ingested[keep].reset_index(drop=True)
        if not bars.empty:
            actions = actions[actions["date"] <= bars["date"].iloc[-1]]
        found[symbol] = _Stored(bars, actions, ingested)
    return found


def get_bars(
    symbols: list[str] | str,
    start: date | None = None,
    end: date | None = None,
    adjustment: Adjustment = "total_return",
    as_of: datetime | None = None,
    *,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Daily bars for one or more symbols.

    Returns a DataFrame indexed by a ``(date, symbol)`` MultiIndex, sorted by date then
    symbol, with float64 columns ``open, high, low, close, volume``. ``date`` is a
    tz-naive ``datetime64[ns]`` at midnight meaning the trading session of that day.
    ``start`` and ``end`` are both inclusive. Use ``bars["close"].unstack("symbol")``
    for a wide (date x symbol) table.

    ``adjustment``:

    * ``"none"``: prices and volumes exactly as traded.
    * ``"split"``: backward-adjusted for splits (prices and volumes).
    * ``"total_return"`` (default): also adjusted for cash dividends, so close-to-close
      changes are total returns. Volumes are split-adjusted only.

    Adjustment is anchored at the last returned bar and uses only corporate actions
    dated on or before ``end``: with ``end`` set, the newest bar equals its unadjusted
    value and nothing that happened after ``end`` affects the result. Adjusted price
    *levels* therefore depend on ``end``; returns do not.

    ``as_of`` (naive UTC, or tz-aware) gives a point-in-time view: only data from raw
    snapshots downloaded at or before that moment is used, with each bar taking the value
    of the latest such snapshot. Limits: it can only go back to your first download
    (history fetched then is whatever the provider believed at that time), it needs the
    files under ``data/raw`` to still exist, and its resolution is the download time.

    Raises ``KeyError`` for symbols that are not in the instruments table.
    """
    requested = _symbol_list(symbols)
    start_ts = to_timestamp(start) if start is not None else None
    frames: list[pd.DataFrame] = []
    for symbol, stored in _unadjusted(requested, end, as_of, data_dir).items():
        adjusted = adjust_bars(stored.bars, stored.actions, adjustment)
        if start_ts is not None:
            adjusted = adjusted[adjusted["date"] >= start_ts]
        adjusted = adjusted.assign(symbol=symbol)
        frames.append(adjusted[["date", "symbol", *VALUE_COLUMNS]])

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "symbol"], kind="stable")
    return result.set_index(["date", "symbol"])


def get_returns(
    symbols: list[str] | str,
    start: date | None = None,
    end: date | None = None,
    kind: Literal["simple", "log"] = "log",
    *,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Daily close-to-close total returns as a wide (date x symbol) DataFrame.

    Computed from total-return adjusted closes, so dividends are included. The return
    on date ``t`` is from the close of the previous bar to the close of ``t``; the first
    row is ``start`` when a bar exists before it. A symbol's return is NaN on days it
    has no bar and on its first bar. Rows where every symbol is NaN are dropped.
    ``kind`` is ``"log"`` (default, ``ln(P_t / P_{t-1})``) or ``"simple"``.
    """
    if kind not in ("simple", "log"):
        raise ValueError(f"unknown return kind: {kind!r}")
    requested = _symbol_list(symbols)
    bars = get_bars(requested, None, end, adjustment="total_return", data_dir=data_dir)
    closes = bars["close"].unstack("symbol").reindex(columns=requested)
    returns = pd.DataFrame(index=closes.index, columns=requested, dtype="float64")
    for symbol in requested:
        # Per symbol on its own bars, so a day missing for one symbol never turns a
        # two-day move into what looks like a one-day return for another.
        own = closes[symbol].dropna()
        ratio = own / own.shift(1)
        own_returns = (
            pd.Series(np.log(ratio.to_numpy()), index=own.index) if kind == "log" else ratio - 1.0
        )
        returns[symbol] = own_returns.reindex(closes.index)
    if start is not None:
        returns = returns[returns.index >= to_timestamp(start)]
    returns.columns.name = "symbol"
    return returns.dropna(how="all")


def list_symbols(active_only: bool = False, *, data_dir: Path | str | None = None) -> list[str]:
    """Symbols known to the store, sorted. ``active_only`` drops delisted/disabled ones."""
    with storage.connect(db_path(data_dir), read_only=True) as con:
        instruments = storage.read_instruments(con)
    if active_only:
        instruments = instruments[instruments["is_active"].astype(bool)]
    return sorted(str(s) for s in instruments["symbol"])


def _known(symbols: list[str], instruments: pd.DataFrame) -> None:
    unknown = [s for s in symbols if s not in set(instruments["symbol"])]
    if unknown:
        raise KeyError(f"unknown symbols: {', '.join(unknown)}")


def get_corporate_actions(
    symbols: list[str] | str,
    end: date | None = None,
    *,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Splits and cash dividends for one or more symbols, up to and including ``end``.

    Columns ``symbol, date, action, value`` sorted by date, symbol, then action. ``action``
    is ``"split"`` (``value`` is the ratio: 2.0 for a 2:1 split) or ``"dividend"`` (``value``
    is cash per share on the share basis of its ex-date, i.e. after a split sharing that
    date). ``date`` is the ex-date. Raises ``KeyError`` for unknown symbols.
    """
    requested = _symbol_list(symbols)
    with storage.connect(db_path(data_dir), read_only=True) as con:
        _known(requested, storage.read_instruments(con))
        actions = storage.read_corporate_actions(con, requested)
    if end is not None:
        actions = actions[actions["date"] <= to_timestamp(end)]
    actions = actions.sort_values(["date", "symbol", "action"], kind="stable")
    return actions[["symbol", *ACTION_COLUMNS]].reset_index(drop=True)


def get_asset_classes(
    symbols: list[str] | str, *, data_dir: Path | str | None = None
) -> dict[str, str]:
    """Asset class (``etf``, ``equity``, ``crypto``) of each symbol, in the order given."""
    requested = _symbol_list(symbols)
    with storage.connect(db_path(data_dir), read_only=True) as con:
        instruments = storage.read_instruments(con)
    _known(requested, instruments)
    classes = instruments.set_index("symbol")["asset_class"]
    return {symbol: str(classes[symbol]) for symbol in requested}


def data_version(
    symbols: list[str] | str,
    end: date | None = None,
    *,
    data_dir: Path | str | None = None,
) -> datetime | None:
    """When the newest of these symbols' bars (up to ``end``) was last written (naive UTC).

    Two runs that report the same value read the same bars, which is what makes a
    backtest reproducible. None when no bars are stored.
    """
    requested = _symbol_list(symbols)
    query = "SELECT max(ingested_at) FROM daily_bars WHERE list_contains(?, symbol)"
    params: list[object] = [requested]
    if end is not None:
        query += " AND date <= ?"
        params.append(end)
    with storage.connect(db_path(data_dir), read_only=True) as con:
        row = con.execute(query, params).fetchone()
    latest = row[0] if row else None
    return None if latest is None else pd.Timestamp(latest).to_pydatetime()


def fingerprint(
    symbols: list[str] | str,
    start: date | None = None,
    end: date | None = None,
    adjustment: Adjustment = "total_return",
    as_of: datetime | None = None,
    *,
    data_dir: Path | str | None = None,
) -> DataFingerprint:
    """A SHA-256 of exactly the data ``get_bars`` reads for these arguments.

    It covers each symbol's unadjusted bars in ``[start, end]`` and its corporate actions
    dated from ``start`` to its last bar (earlier actions cannot change those bars), read
    from the database or, with ``as_of``, from the snapshots taken by then. The format is
    described in ``qr.fingerprint``. Symbol order and repeats do not matter. Raises
    ``KeyError`` for unknown symbols.
    """
    requested = sorted(_symbol_list(symbols))
    start_ts = to_timestamp(start) if start is not None else None
    per_symbol: dict[str, str] = {}
    days: list[pd.Timestamp] = []
    written: list[pd.Timestamp] = []
    rows = 0
    for symbol, stored in _unadjusted(requested, end, as_of, data_dir).items():
        bars, actions, ingested = stored.bars, stored.actions, stored.ingested_at
        if start_ts is not None:
            keep = (bars["date"] >= start_ts).to_numpy()
            bars, ingested = bars[keep], ingested[keep]
            actions = actions[actions["date"] >= start_ts]
        per_symbol[symbol] = hash_symbol(symbol, bars, actions)
        rows += len(bars)
        if not bars.empty:
            days += [bars["date"].min(), bars["date"].max()]
        newest = ingested.max()
        if not pd.isna(newest):
            written.append(pd.Timestamp(newest))
    return DataFingerprint(
        hash=combine(per_symbol, adjustment),
        per_symbol=per_symbol,
        rows=rows,
        first_date=min(days).date() if days else None,
        last_date=max(days).date() if days else None,
        latest_ingested_at=max(written).to_pydatetime() if written else None,
    )
