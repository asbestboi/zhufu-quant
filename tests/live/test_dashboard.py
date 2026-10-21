"""The self-contained HTML dashboard: a smoke test that it builds valid, informative
output, not a pixel-level check of the chart."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from qr import calendars
from qr.live import dashboard, journal
from qr.live.config import BenchmarkConfig

from .live_config import make_live_config
from .live_store import real_session_bars, seed_store

SYMBOLS = ["AAA", "BBB"]


def test_dashboard_with_no_history_still_renders(tmp_path: Path) -> None:
    config = make_live_config(SYMBOLS, tmp_path, strategy="buy_and_hold")
    page = dashboard.build_dashboard_html(config, data_dir=tmp_path)
    assert "<html>" in page
    assert "No equity history yet" in page
    assert "None yet" in page


def test_dashboard_after_a_decision_shows_positions_orders_and_a_chart(tmp_path: Path) -> None:
    start, end = date(2022, 11, 1), date(2023, 2, 15)
    bars = {s: real_session_bars(start, end, seed=i) for i, s in enumerate(SYMBOLS)}
    seed_store(tmp_path, bars)
    decision_date = sorted(bars["AAA"]["date"].dt.date)[-1]

    config = make_live_config(
        SYMBOLS, tmp_path, strategy="buy_and_hold", capital_limit=10_000.0,
        benchmark=BenchmarkConfig(strategy="buy_and_hold", params={}),
    )  # fmt: skip
    did = journal.record_decision(
        config.strategy, decision_date, {"AAA": 0.5, "BBB": 0.5}, data_directory=tmp_path
    )
    close_a = float(bars["AAA"].loc[bars["AAA"]["date"].dt.date == decision_date, "close"].iloc[0])
    journal.record_order("O1", did, "AAA", "buy", 10.0, close_a, "filled", data_directory=tmp_path)
    journal.record_fill("O1", did, "AAA", "buy", 10.0, close_a, 0.0, data_directory=tmp_path)

    page = dashboard.build_dashboard_html(config, data_dir=tmp_path)
    assert "<svg" in page
    assert "AAA" in page
    assert "buy_and_hold (ledger)" in page
    assert "buy_and_hold (shadow)" in page
    assert "O1" not in page  # order ids aren't shown, just their decision/symbol/status
    assert "filled" in page


def test_live_preview_shows_todays_target_weights_before_any_real_decision(
    tmp_path: Path,
) -> None:
    """The preview is computable, and clearly labelled, with no decision ever journaled --
    this is what makes it useful day to day instead of waiting for the next month-end."""
    start, end = date(2022, 11, 1), date(2023, 2, 15)  # a Wednesday, not a month-end
    bars = {s: real_session_bars(start, end, seed=i) for i, s in enumerate(SYMBOLS)}
    seed_store(tmp_path, bars)
    config = make_live_config(SYMBOLS, tmp_path, strategy="buy_and_hold", capital_limit=10_000.0)

    page = dashboard.build_dashboard_html(config, data_dir=tmp_path)
    assert "Live preview" in page
    assert "not a real decision" in page
    assert "Next scheduled decision" in page
    assert "target weight" in page
    assert "AAA" in page and "BBB" in page


def test_live_preview_labels_a_real_month_end_session_as_the_real_decision(
    tmp_path: Path,
) -> None:
    end = calendars.sessions("etf", date(2023, 1, 1), date(2023, 1, 31))[-1].date()
    bars = {s: real_session_bars(date(2022, 11, 1), end, seed=i) for i, s in enumerate(SYMBOLS)}
    seed_store(tmp_path, bars)
    config = make_live_config(SYMBOLS, tmp_path, strategy="buy_and_hold", capital_limit=10_000.0)

    page = dashboard.build_dashboard_html(config, data_dir=tmp_path)
    assert "this month&#x27;s real decision" in page


def test_write_dashboard_creates_the_file(tmp_path: Path) -> None:
    config = make_live_config(SYMBOLS, tmp_path, strategy="buy_and_hold", name="my-test")
    path = dashboard.write_dashboard(config, data_dir=tmp_path)
    assert path.name == "dashboard_my-test.html"
    assert path.is_file()
    assert "<html>" in path.read_text(encoding="utf-8")
