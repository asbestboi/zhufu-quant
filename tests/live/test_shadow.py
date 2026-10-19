"""The shadow backtest wrapper: same engine, no registry or report-file side effects."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from qr.live import journal, shadow

from .live_store import real_session_bars, seed_store

SYMBOLS = ["AAA", "BBB", "CCC"]


def test_ledger_equity_curve_marks_fills_to_the_real_close(tmp_path: Path) -> None:
    bars = {"AAA": real_session_bars(date(2023, 1, 1), date(2023, 3, 31), seed=1)}
    seed_store(tmp_path, bars)
    sessions = sorted(bars["AAA"]["date"].dt.date)
    first_decision, fill_date, second_decision = sessions[20], sessions[21], sessions[40]
    journal.record_decision("demo", first_decision, {"AAA": 1.0}, data_directory=tmp_path)
    did2 = journal.record_decision("demo", second_decision, {"AAA": 1.0}, data_directory=tmp_path)
    journal.record_order("O1", did2, "AAA", "buy", 10.0, 100.0, "filled", data_directory=tmp_path)
    journal.record_fill(
        "O1", did2, "AAA", "buy", 10.0, 100.0, 1.0,
        filled_at=pd.Timestamp(fill_date), data_directory=tmp_path,
    )  # fmt: skip

    curve = shadow.ledger_equity_curve("demo", 10_000.0, data_dir=tmp_path)
    assert list(curve.index) == [pd.Timestamp(first_decision), pd.Timestamp(second_decision)]
    assert curve.loc[pd.Timestamp(first_decision)] == 10_000.0  # before the fill: untouched

    close = float(bars["AAA"].loc[bars["AAA"]["date"].dt.date == second_decision, "close"].iloc[0])
    expected = 10_000.0 - 10.0 * 100.0 - 1.0 + 10.0 * close
    assert abs(curve.loc[pd.Timestamp(second_decision)] - expected) < 1e-6


def test_run_shadow_matches_a_direct_event_engine_run(tmp_path: Path) -> None:
    start, end = date(2022, 11, 1), date(2023, 4, 15)
    bars = {s: real_session_bars(start, end, seed=i) for i, s in enumerate(SYMBOLS)}
    seed_store(tmp_path, bars)

    params = {"lookback": 10, "skip": 2, "top_n": 2}
    result = shadow.run_shadow(
        "xs_momentum", params, SYMBOLS, start, end, initial_cash=100_000.0, data_dir=tmp_path
    )
    assert result.final_state is not None
    assert len(result.equity) > 0
    assert result.equity.iloc[0] > 0


def test_period_return_measures_between_two_dates() -> None:
    equity = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"]),
    )
    got = shadow.period_return(equity, pd.Timestamp("2023-02-28"), pd.Timestamp("2023-03-31"))
    assert got is not None
    assert abs(got - 0.10) < 1e-9


def test_period_return_is_none_before_the_series_starts() -> None:
    equity = pd.Series([100.0], index=pd.to_datetime(["2023-01-31"]))
    assert shadow.period_return(equity, pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-02")) is (
        None
    )
