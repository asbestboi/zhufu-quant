"""Daily scheduling maintenance: expire stale plans, flag missed decisions.

Run at the start of every ``qr live run`` and ``qr live approve`` call -- whether fired
by a systemd timer or typed by hand -- so the missed-decision bookkeeping is automatic
and never depends on the timer's own script getting it right. Each lapse is recorded as
a journal alert tagged with :data:`MISSED_TAG`, exactly once (checked against the
alerts already written before adding another), which is what makes calling this on
every invocation safe rather than a growing pile of duplicate warnings.

A decision counts as missed in exactly two ways (never more, see the pre-registration
this scheduler was built for): its plan sat unapproved for more than
:data:`PLAN_EXPIRY_SESSIONS` trading days (handled here by expiring it outright, so
nothing can later approve and send it), or no run ever got to it at all within
:data:`MISSED_RUN_SESSIONS` trading days of the month-end close. A decision that is
merely running late -- submitted, still resolving fills -- is flagged for visibility but
left alone: the order lifecycle's own window-end cancellation is what eventually
resolves it, never this pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from qr import calendars
from qr.live import journal
from qr.live.config import LiveConfig
from qr.live.cycle import ASSET_CLASS, is_month_end

PLAN_EXPIRY_SESSIONS = 3
MISSED_RUN_SESSIONS = 3
STILL_RUNNING_SESSIONS = 3
MISSED_TAG = "[missed]"


@dataclass(frozen=True)
class MaintenanceEvent:
    """One thing this pass found and recorded, for the caller to print or notify."""

    decision_id: str
    message: str


def _sessions_since(day: date, latest: date) -> int:
    """Trading sessions strictly after ``day`` and up to ``latest`` (0 if ``day >= latest``)."""
    if day >= latest:
        return 0
    return calendars.sessions_between(ASSET_CLASS, day, latest)


def _already_flagged(tag: str, did: str, data_dir: object) -> bool:
    alerts = journal.read_alerts(data_directory=data_dir)
    if alerts.empty:
        return False
    prefix = f"{tag} {did}:"
    return bool(alerts["message"].str.startswith(prefix).any())


def _latest_month_end(latest: date) -> date | None:
    """The most recent month-end session at or before ``latest``, if one is nearby."""
    start = latest.replace(day=1) - timedelta(days=5)
    sessions = [d.date() for d in calendars.sessions(ASSET_CLASS, start, latest)]
    for day in reversed(sessions):
        if is_month_end(day):
            return day
    return None


def run_maintenance(
    config: LiveConfig, data_dir: object = None, *, now: datetime | None = None
) -> list[MaintenanceEvent]:
    """Expire stale unapproved plans, and flag month-ends nobody ever ran, as missed.

    Idempotent (see module docstring): safe, and intended, to call on every invocation
    of ``qr live run`` or ``qr live approve``, not only from the timer's own script.
    """
    events: list[MaintenanceEvent] = []
    latest = calendars.last_completed_session(ASSET_CLASS, now or datetime.now())

    decisions = journal.read_decisions(config.strategy, data_directory=data_dir)
    seen_dates: set[date] = set()
    for _, row in decisions.iterrows():
        decision_date = pd.Timestamp(row["decision_date"]).date()
        seen_dates.add(decision_date)
        did = str(row["decision_id"])
        age = _sessions_since(decision_date, latest)

        if row["status"] == journal.PLANNED and age > PLAN_EXPIRY_SESSIONS:
            journal.set_decision_status(did, journal.EXPIRED, data_directory=data_dir)
            message = (
                f"{MISSED_TAG} {did}: plan expired unapproved after "
                f"{PLAN_EXPIRY_SESSIONS} trading days"
            )
            journal.record_alert(message, severity="warning", did=did, data_directory=data_dir)
            events.append(MaintenanceEvent(did, message))
        elif row["status"] == journal.SUBMITTED and age > STILL_RUNNING_SESSIONS:
            # Not executed (completed) within the window, whatever the reason -- a miss
            # under the pre-registration's own 4.1(a), same as a never-run month-end.
            # The decision itself is left alone: the order lifecycle's own window-end
            # cancellation, not this pass, is what eventually resolves it.
            if _already_flagged(MISSED_TAG, did, data_dir):
                continue
            message = (
                f"{MISSED_TAG} {did}: still not completed {age} trading days after close "
                "-- awaiting broker resolution"
            )
            journal.record_alert(message, severity="warning", did=did, data_directory=data_dir)
            events.append(MaintenanceEvent(did, message))

    month_end = _latest_month_end(latest)
    if (
        month_end is not None
        and month_end not in seen_dates
        and _sessions_since(month_end, latest) > MISSED_RUN_SESSIONS
    ):
        did = journal.decision_id(config.strategy, month_end)
        if not _already_flagged(MISSED_TAG, did, data_dir):
            message = (
                f"{MISSED_TAG} {did}: no run executed within {MISSED_RUN_SESSIONS} trading "
                "days of the month-end close"
            )
            journal.record_alert(message, severity="warning", data_directory=data_dir)
            events.append(MaintenanceEvent(did, message))

    return events
