"""Daily maintenance: expiring stale plans and flagging missed decisions."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from qr.live import journal, schedule
from qr.live.cycle import ASSET_CLASS

from .live_config import make_live_config

SYMBOLS = ["AAA", "BBB"]


def _config(tmp_path: Path) -> object:
    return make_live_config(SYMBOLS, tmp_path)


def test_a_fresh_plan_within_the_window_is_left_alone(tmp_path: Path) -> None:
    config = _config(tmp_path)
    journal.record_decision(
        config.strategy, date(2026, 9, 30), {"AAA": 1.0}, data_directory=tmp_path
    )
    events = schedule.run_maintenance(config, tmp_path, now=datetime(2026, 10, 1))
    assert events == []
    decision = journal.find_decision(config.strategy, date(2026, 9, 30), data_directory=tmp_path)
    assert decision is not None and decision.status == journal.PLANNED


def test_a_plan_older_than_three_trading_days_expires_and_is_alerted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    journal.record_decision(
        config.strategy, date(2026, 9, 30), {"AAA": 1.0}, data_directory=tmp_path
    )
    # 2026-09-30 is a Wednesday; four sessions later is well past the 3-trading-day window.
    now = datetime(2026, 10, 7)
    events = schedule.run_maintenance(config, tmp_path, now=now)
    assert len(events) == 1
    assert "plan expired unapproved" in events[0].message
    decision = journal.find_decision(config.strategy, date(2026, 9, 30), data_directory=tmp_path)
    assert decision is not None and decision.status == journal.EXPIRED


def test_expiring_the_same_plan_twice_does_not_duplicate_the_alert(tmp_path: Path) -> None:
    config = _config(tmp_path)
    journal.record_decision(
        config.strategy, date(2026, 9, 30), {"AAA": 1.0}, data_directory=tmp_path
    )
    now = datetime(2026, 10, 7)
    schedule.run_maintenance(config, tmp_path, now=now)
    events = schedule.run_maintenance(config, tmp_path, now=datetime(2026, 10, 8))
    assert events == []  # already expired -- no longer PLANNED, nothing left to flag


def test_a_month_end_nobody_ever_ran_is_flagged_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    # The system was already live well before the close -- a run on record a month
    # earlier -- so this close really was missed, not just never watched.
    journal.record_run(
        "R0", config.name, dry_run=False, catch_up=False,
        started_at=datetime(2026, 8, 15), data_directory=tmp_path,
    )  # fmt: skip
    # No decision row at all for the 2026-09-30 close.
    events = schedule.run_maintenance(config, tmp_path, now=datetime(2026, 10, 7))
    assert len(events) == 1
    assert "no run executed" in events[0].message

    again = schedule.run_maintenance(config, tmp_path, now=datetime(2026, 10, 8))
    assert again == []  # deduped against the alert already written


def test_a_month_end_before_the_systems_first_ever_run_is_not_flagged(tmp_path: Path) -> None:
    """Regression: a real production incident. The system's very first invocation,
    run any time after a real month-end that predates the deployment entirely, must
    not retroactively call that month-end a miss -- there was no system yet to miss it."""
    config = _config(tmp_path)
    # No run recorded at all -- this *is* the system's first ever invocation, and
    # 2026-09-30 already closed weeks before it.
    events = schedule.run_maintenance(config, tmp_path, now=datetime(2026, 10, 21))
    assert events == []


def test_a_submitted_decision_running_late_is_flagged_but_not_touched(tmp_path: Path) -> None:
    config = _config(tmp_path)
    did = journal.record_decision(
        config.strategy, date(2026, 9, 30), {"AAA": 1.0}, data_directory=tmp_path
    )
    journal.set_decision_status(did, journal.SUBMITTED, data_directory=tmp_path)
    events = schedule.run_maintenance(config, tmp_path, now=datetime(2026, 10, 7))
    assert len(events) == 1
    assert "still not completed" in events[0].message
    decision = journal.find_decision(config.strategy, date(2026, 9, 30), data_directory=tmp_path)
    assert decision is not None and decision.status == journal.SUBMITTED  # untouched


def test_a_completed_decision_is_never_flagged(tmp_path: Path) -> None:
    config = _config(tmp_path)
    did = journal.record_decision(
        config.strategy, date(2026, 9, 30), {"AAA": 1.0}, data_directory=tmp_path
    )
    journal.set_decision_status(did, journal.COMPLETED, data_directory=tmp_path)
    events = schedule.run_maintenance(config, tmp_path, now=datetime(2026, 10, 10))
    assert events == []


def test_sessions_since_matches_the_calendar(tmp_path: Path) -> None:
    from qr import calendars

    assert schedule._sessions_since(date(2026, 9, 30), date(2026, 9, 30)) == 0
    expected = calendars.sessions_between(ASSET_CLASS, date(2026, 9, 30), date(2026, 10, 7))
    assert schedule._sessions_since(date(2026, 9, 30), date(2026, 10, 7)) == expected
