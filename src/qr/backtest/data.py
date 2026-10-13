"""The data layer both engines read from: one bundle of bars and corporate actions.

Loaded through ``qr.load`` only. The vectorized engine uses the total-return adjusted
bars, the event engine the unadjusted bars plus the corporate actions, and both see the
same sessions, so their results can be compared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from qr.adjust import adjust_bars
from qr.frames import ACTION_COLUMNS, VALUE_COLUMNS, normalize_actions, normalize_bars, to_dates
from qr.load import get_asset_classes, get_bars, get_corporate_actions

FloatArray = npt.NDArray[np.float64]


@dataclass
class MarketData:
    """Bars and corporate actions for a fixed list of symbols, up to one end date.

    ``raw`` and ``adjusted`` have the shape ``qr.load.get_bars`` returns: a ``(date,
    symbol)`` MultiIndex with ``open, high, low, close, volume``. ``raw`` is as traded;
    ``adjusted`` is total-return adjusted and anchored at the last bar. ``actions`` has
    ``symbol, date, action, value``.
    """

    symbols: list[str]
    raw: pd.DataFrame
    adjusted: pd.DataFrame
    actions: pd.DataFrame
    asset_classes: dict[str, str] = field(default_factory=dict)
    _panels: dict[tuple[str, str], FloatArray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.raw.index.equals(self.adjusted.index):
            raise ValueError("raw and adjusted bars must cover the same (date, symbol) rows")
        dates = self.raw.index.get_level_values("date").unique().sort_values()
        self.dates = pd.DatetimeIndex(dates, name="date")

    @classmethod
    def from_frames(
        cls,
        bars: Mapping[str, pd.DataFrame],
        actions: Mapping[str, pd.DataFrame] | None = None,
        asset_classes: Mapping[str, str] | None = None,
    ) -> MarketData:
        """Build from per-symbol unadjusted bars (``date, open, ..., volume``) in memory.

        ``actions`` maps a symbol to its ``date, action, value`` frame. Adjusted prices
        are computed exactly as ``get_bars`` does.
        """
        symbols = list(bars)
        raw_parts: list[pd.DataFrame] = []
        adjusted_parts: list[pd.DataFrame] = []
        action_parts: list[pd.DataFrame] = []
        for symbol in symbols:
            own = normalize_bars(bars[symbol])[["date", *VALUE_COLUMNS]]
            own_actions = normalize_actions(
                (actions or {}).get(symbol, pd.DataFrame(columns=ACTION_COLUMNS))
            )
            if not own.empty:
                own_actions = own_actions[own_actions["date"] <= own["date"].iloc[-1]]
            raw_parts.append(own.assign(symbol=symbol))
            adjusted_parts.append(adjust_bars(own, own_actions).assign(symbol=symbol))
            action_parts.append(own_actions.assign(symbol=symbol))

        def long(parts: list[pd.DataFrame]) -> pd.DataFrame:
            frame = pd.concat(parts, ignore_index=True).sort_values(
                ["date", "symbol"], kind="stable"
            )
            return frame.set_index(["date", "symbol"])[VALUE_COLUMNS]

        all_actions = pd.concat(action_parts, ignore_index=True)
        all_actions = all_actions.sort_values(["date", "symbol", "action"], kind="stable")
        return cls(
            symbols=symbols,
            raw=long(raw_parts),
            adjusted=long(adjusted_parts),
            actions=all_actions[["symbol", *ACTION_COLUMNS]].reset_index(drop=True),
            asset_classes=dict(asset_classes or {}),
        )

    def panel(self, column: str, adjusted: bool = False) -> FloatArray:
        """A ``(dates x symbols)`` array of one bar column; NaN where a symbol has no bar."""
        key = (column, "adjusted" if adjusted else "raw")
        if key not in self._panels:
            source = self.adjusted if adjusted else self.raw
            wide = source[column].unstack("symbol").reindex(index=self.dates, columns=self.symbols)
            self._panels[key] = wide.to_numpy(dtype="float64")
        return self._panels[key]

    def wide(self, column: str, adjusted: bool = False) -> pd.DataFrame:
        """Like :meth:`panel`, as a DataFrame indexed by date with one column per symbol."""
        return pd.DataFrame(
            self.panel(column, adjusted), index=self.dates, columns=pd.Index(self.symbols)
        )

    def adv_value(self, window: int) -> FloatArray:
        """Average daily dollar volume over the ``window`` sessions *before* each date.

        Dollar volume (close times volume) is unchanged by splits. Shifted by one row, so
        the value on a date uses nothing from that date. NaN until a symbol has a bar.
        """
        key = (f"adv{window}", "raw")
        if key not in self._panels:
            dollars = self.wide("close") * self.wide("volume")
            mean = dollars.rolling(window, min_periods=1).mean().shift(1)
            self._panels[key] = mean.to_numpy(dtype="float64")
        return self._panels[key]

    def sessions(self, start: date, end: date) -> pd.DatetimeIndex:
        """The dates with at least one bar in ``[start, end]``."""
        first, last = to_dates([start, end])
        return pd.DatetimeIndex(self.dates[(self.dates >= first) & (self.dates <= last)])


def load_market_data(
    symbols: Sequence[str], end: date, *, data_dir: Path | str | None = None
) -> MarketData:
    """Load everything a backtest of ``symbols`` ending on ``end`` may use.

    History before the backtest's start is included on purpose: strategies need it to
    warm up their indicators. Nothing dated after ``end`` is read.
    """
    requested = list(symbols)
    raw = get_bars(requested, end=end, adjustment="none", data_dir=data_dir)
    adjusted = get_bars(requested, end=end, adjustment="total_return", data_dir=data_dir)
    if raw.empty:
        raise ValueError(f"no bars stored for {', '.join(requested)} up to {end}")
    return MarketData(
        symbols=requested,
        raw=raw,
        adjusted=adjusted,
        actions=get_corporate_actions(requested, end=end, data_dir=data_dir),
        asset_classes=get_asset_classes(requested, data_dir=data_dir),
    )
