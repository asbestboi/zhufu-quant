"""Section 8's monthly tracking report: reliability tally, consistency check, shadow."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from qr.live import journal, report, schedule
from qr.live.config import BenchmarkConfig

from .live_config import make_live_config
from .live_store import real_session_bars, seed_store

SYMBOLS = ["AAA", "BBB"]


def test_reliability_tally_counts_missed_decisions_only(tmp_path: Path) -> None:
    journal.record_decision("demo", date(2026, 8, 31), {"AAA": 1.0}, data_directory=tmp_path)
    did = journal.record_decision("demo", date(2026, 9, 30), {"AAA": 1.0}, data_directory=tmp_path)
    journal.set_decision_status(did, journal.COMPLETED, data_directory=tmp_path)
    journal.record_alert(
        f"{schedule.MISSED_TAG} demo:2026-08-31: plan expired unapproved after 3 trading days",
        severity="warning", did="demo:2026-08-31", data_directory=tmp_path,
    )  # fmt: skip

    tally = report.reliability_tally("demo", data_dir=tmp_path)
    assert tally.due == 2
    assert tally.missed == 1
    assert tally.missed_decision_ids == ["demo:2026-08-31"]


def test_reliability_tally_counts_a_never_run_month_end_with_no_decision_row(
    tmp_path: Path,
) -> None:
    journal.record_alert(
        f"{schedule.MISSED_TAG} demo:2026-07-31: no run executed within 3 trading days "
        "of the month-end close",
        severity="warning", data_directory=tmp_path,
    )  # fmt: skip
    tally = report.reliability_tally("demo", data_dir=tmp_path)
    assert tally.due == 1
    assert tally.missed == 1


def test_consistency_check_passes_with_no_fills_and_no_shadow_trades(tmp_path: Path) -> None:
    start, end = date(2022, 11, 1), date(2022, 12, 15)
    bars = {s: real_session_bars(start, end, seed=i) for i, s in enumerate(SYMBOLS)}
    seed_store(tmp_path, bars)
    decision_date = sorted(bars["AAA"]["date"].dt.date)[-1]
    # xs_momentum needs a lookback longer than the data on hand, so it trades nothing --
    # both sides are flat, which is itself a real (if trivial) match to check.
    config = make_live_config(
        SYMBOLS, tmp_path, strategy="xs_momentum",
        params={"lookback": 252, "skip": 2, "top_n": 1}, capital_limit=10_000.0,
    )  # fmt: skip
    journal.record_decision(config.strategy, decision_date, {}, data_directory=tmp_path)

    result = report.consistency_check(config, decision_date, decision_date, data_dir=tmp_path)
    assert result.ok
    assert result.mismatches == {}


def test_consistency_check_flags_a_ledger_that_diverges_from_the_shadow(tmp_path: Path) -> None:
    start, end = date(2022, 11, 1), date(2023, 4, 15)
    bars = {s: real_session_bars(start, end, seed=i) for i, s in enumerate(SYMBOLS)}
    seed_store(tmp_path, bars)
    decision_date = sorted(bars["AAA"]["date"].dt.date)[-1]
    config = make_live_config(SYMBOLS, tmp_path, strategy="buy_and_hold", capital_limit=10_000.0)
    journal.record_decision(config.strategy, decision_date, {"AAA": 0.5, "BBB": 0.5},
                             data_directory=tmp_path)  # fmt: skip
    # No fills ever journaled for it -- the shadow buy_and_hold backtest, which always
    # invests, is guaranteed to disagree with this empty ledger.

    result = report.consistency_check(
        config, decision_date.replace(day=1), decision_date, data_dir=tmp_path
    )
    assert not result.ok
    assert result.mismatches  # at least one symbol disagrees


def test_build_tracking_report_is_empty_shaped_with_no_decisions(tmp_path: Path) -> None:
    config = make_live_config(SYMBOLS, tmp_path, strategy="buy_and_hold")
    tracking = report.build_tracking_report(config, "2026-09", data_dir=tmp_path)
    assert tracking.consistency is None
    assert tracking.candidate_return is None
    text = report.render_tracking_markdown(tracking)
    assert "Reliability: 0 missed of 0" in text


def test_tracking_report_includes_a_configured_benchmark(tmp_path: Path) -> None:
    start, end = date(2022, 11, 1), date(2023, 2, 15)
    bars = {s: real_session_bars(start, end, seed=i) for i, s in enumerate(SYMBOLS)}
    seed_store(tmp_path, bars)
    sessions = sorted(bars["AAA"]["date"].dt.date)
    decision_date = sessions[-1]

    config = make_live_config(
        SYMBOLS, tmp_path, strategy="buy_and_hold", capital_limit=10_000.0,
        benchmark=BenchmarkConfig(strategy="buy_and_hold", params={}),
    )  # fmt: skip
    did = journal.record_decision(
        config.strategy, decision_date, {"AAA": 0.5, "BBB": 0.5}, data_directory=tmp_path
    )
    journal.set_decision_status(did, journal.COMPLETED, data_directory=tmp_path)

    month = f"{decision_date:%Y-%m}"
    tracking = report.build_tracking_report(config, month, data_dir=tmp_path)
    assert tracking.cumulative_benchmark_return is not None
    html = report.render_tracking_html(tracking)
    assert "<html>" in html
