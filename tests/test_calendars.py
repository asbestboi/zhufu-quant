from __future__ import annotations

from datetime import date, datetime

from qr import calendars


def test_exchange_sessions_skip_weekends_and_holidays() -> None:
    sessions = calendars.sessions("equity", date(2024, 1, 11), date(2024, 1, 17))
    # Jan 13-14 is a weekend and Jan 15 is Martin Luther King Jr. Day.
    assert [s.day for s in sessions] == [11, 12, 16, 17]


def test_crypto_sessions_are_every_day() -> None:
    sessions = calendars.sessions("crypto", date(2024, 1, 11), date(2024, 1, 17))
    assert [s.day for s in sessions] == [11, 12, 13, 14, 15, 16, 17]


def test_empty_range() -> None:
    assert len(calendars.sessions("equity", date(2024, 1, 5), date(2024, 1, 4))) == 0


def test_last_completed_session_waits_for_the_close() -> None:
    # NYSE closes at 16:00 New York = 21:00 UTC in January.
    assert calendars.last_completed_session("etf", datetime(2024, 1, 16, 20, 59)) == date(
        2024, 1, 12
    )
    assert calendars.last_completed_session("etf", datetime(2024, 1, 16, 21, 0)) == date(
        2024, 1, 16
    )
    # Weekend: still Friday's session.
    assert calendars.last_completed_session("etf", datetime(2024, 1, 14, 12, 0)) == date(
        2024, 1, 12
    )


def test_last_completed_session_for_crypto_is_yesterday_utc() -> None:
    assert calendars.last_completed_session("crypto", datetime(2024, 1, 14, 0, 5)) == date(
        2024, 1, 13
    )


def test_sessions_back() -> None:
    # Five sessions before Tue Jan 16: 12, 11, 10, 9, 8 (the 15th is a holiday).
    assert calendars.sessions_back("equity", date(2024, 1, 16), 5) == date(2024, 1, 8)
    assert calendars.sessions_back("crypto", date(2024, 1, 16), 5) == date(2024, 1, 11)


def test_sessions_between() -> None:
    assert calendars.sessions_between("equity", date(2024, 1, 12), date(2024, 1, 17)) == 2
    assert calendars.sessions_between("equity", date(2024, 1, 16), date(2024, 1, 16)) == 0
    assert calendars.sessions_between("crypto", date(2024, 1, 12), date(2024, 1, 17)) == 5
