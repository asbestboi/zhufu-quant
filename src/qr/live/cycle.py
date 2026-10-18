"""The monthly cycle: decide, plan, check, approve, submit, reconcile, journal.

Idempotent and resumable by construction, not by special-casing a crash: every call
looks up the decision by its deterministic id (:mod:`qr.live.journal`); an already
``completed`` decision is a no-op, and the order plan for a ``planned`` or ``submitted``
decision is always *recomputed* from the frozen target weights and the broker's current
account and prices. Orders already filled (by an earlier, interrupted attempt) therefore
already show up in the account, so the recomputed plan asks for only what is still
missing -- a crash between "order submitted" and "fill journaled" is recovered from by
simply running the cycle again, never by replaying a stored order list.

The target weights themselves are computed once, at the moment the decision is created,
and then frozen in the journal: approving or resuming a decision never recomputes them,
so what gets approved is exactly what gets (eventually) submitted, even if the data store
has moved on by the time that happens.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from qr import calendars
from qr.live import journal, safety
from qr.live.broker import Broker, PriceQuote
from qr.live.config import LiveConfig
from qr.live.decide import DecisionError, target_weights
from qr.live.journal import DecisionRecord
from qr.live.planner import Plan, plan_orders
from qr.live.safety import SafetyViolation

ASSET_CLASS = "etf"
CATCH_UP_LOOKBACK_SESSIONS = 5
RECONCILE_TOLERANCE_SHARES = 1e-6

NO_DECISION = "no_decision"
AWAITING_APPROVAL = "awaiting_approval"
SUBMITTED = "submitted"
COMPLETED = "completed"
ALREADY_COMPLETED = "already_completed"
FAILED = "failed"


@dataclass
class CycleResult:
    """What one call to :func:`run_cycle` did."""

    run_id: str
    status: str
    decision_id: str | None = None
    plan: Plan | None = None
    message: str = ""


def _month_end_sessions(day: date) -> list[date]:
    start = day.replace(day=1)
    end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    return [d.date() for d in calendars.sessions(ASSET_CLASS, start, end)]


def is_month_end(day: date) -> bool:
    """Whether ``day`` is the last expected trading session of its calendar month.

    Uses the real exchange calendar (:mod:`qr.calendars`), not just the data on hand: a
    live decision must know this before the next session exists to prove it.
    """
    sessions = _month_end_sessions(day)
    return bool(sessions) and sessions[-1] == day


def _pending_month_end(
    strategy: str, latest_session: date, data_dir: Path | str | None
) -> date | None:
    """A recent month-end session with no completed decision yet, newest first.

    Used only when ``catch_up`` is set: the full expiry policy (a decision older than
    three trading days is skipped and recorded as missed) is a scheduling concern, built
    with the timer in Part 4; this just finds the most recent unresolved one.
    """
    start = calendars.sessions_back(ASSET_CLASS, latest_session, CATCH_UP_LOOKBACK_SESSIONS)
    candidates = [d.date() for d in calendars.sessions(ASSET_CLASS, start, latest_session)]
    for day in reversed(candidates):
        if is_month_end(day):
            existing = journal.find_decision(strategy, day, data_directory=data_dir)
            if existing is None or existing.status != journal.COMPLETED:
                return day
    return None


def _plan_for(
    config: LiveConfig, decision: DecisionRecord, broker: Broker
) -> tuple[Plan, dict[str, PriceQuote], datetime]:
    account = broker.account()
    quotes = broker.prices(config.universe)
    checked_at = datetime.now()
    prices = {symbol: quote.price for symbol, quote in quotes.items()}
    cash_buffer = decision.cash_buffer if decision.cash_buffer is not None else config.cash_buffer
    plan = plan_orders(
        decision.target_weights,
        account,
        prices,
        cash_buffer=cash_buffer,
        min_trade_weight=config.min_trade_weight,
    )
    return plan, quotes, checked_at


def _submit(did: str, plan: Plan, broker: Broker, *, data_dir: Path | str | None) -> None:
    for order in plan.orders:
        result = broker.place_order(order.symbol, order.side, order.quantity)
        journal.record_order(
            result.id, did, order.symbol, order.side, order.quantity, order.price,
            result.status, data_directory=data_dir,
        )  # fmt: skip
        if result.filled_quantity:
            journal.record_fill(
                result.id, did, order.symbol, order.side, result.filled_quantity,
                order.price, 0.0, data_directory=data_dir,
            )  # fmt: skip
        if result.status == "filled" and abs(result.filled_quantity - order.quantity) > (
            RECONCILE_TOLERANCE_SHARES
        ):
            journal.record_alert(
                f"{order.symbol}: planned {order.quantity:g} shares, broker filled "
                f"{result.filled_quantity:g}",
                severity="warning",
                did=did,
                data_directory=data_dir,
            )


TERMINAL_ORDER_STATUSES = ["filled", "cancelled", "rejected", "expired"]


def _is_resolved(did: str, nothing_left_to_plan: bool, data_dir: Path | str | None) -> bool:
    """Whether a decision has nothing further to wait on.

    Either every order ever journaled for it has reached a terminal state, or none was
    ever needed in the first place (``nothing_left_to_plan`` and none journaled yet).
    """
    orders = journal.read_orders(did, data_directory=data_dir)
    if orders.empty:
        return nothing_left_to_plan
    return bool(orders["status"].isin(TERMINAL_ORDER_STATUSES).all())


def run_cycle(
    config: LiveConfig,
    broker: Broker,
    *,
    dry_run: bool = False,
    approve: bool = False,
    catch_up: bool = False,
    as_of: date | None = None,
    now: datetime | None = None,
    data_dir: Path | str | None = None,
) -> CycleResult:
    """Run (or resume, or approve) one decision of ``config``'s monthly cycle.

    ``as_of`` overrides the decision date entirely (for tests); otherwise it is derived
    from the latest completed session and the calendar. ``approve`` sends the orders of
    an already-planned decision; without it (or with ``dry_run``), the cycle stops after
    planning and safety checks, and returns the plan for review.
    """
    run_id = f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    journal.record_run(
        run_id, config.name, dry_run=dry_run, catch_up=catch_up, data_directory=data_dir
    )
    decision: DecisionRecord | None = None

    try:
        decision_date = as_of
        if decision_date is None:
            latest = calendars.last_completed_session(ASSET_CLASS, now or datetime.now())
            if is_month_end(latest):
                decision_date = latest
            elif catch_up:
                decision_date = _pending_month_end(config.strategy, latest, data_dir)
            if decision_date is None:
                journal.finish_run(run_id, "ok", message="no decision due", data_directory=data_dir)
                return CycleResult(run_id, NO_DECISION, message=f"no decision due as of {latest}")

        decision = journal.find_decision(config.strategy, decision_date, data_directory=data_dir)
        if decision is not None and decision.status == journal.COMPLETED:
            journal.finish_run(run_id, "ok", message="already completed", data_directory=data_dir)
            return CycleResult(run_id, ALREADY_COMPLETED, decision.decision_id)

        if decision is None:
            weights = target_weights(
                config.strategy, config.params, config.universe, decision_date, data_dir=data_dir
            )
            journal.record_decision(
                config.strategy, decision_date, weights, equity=broker.account().equity,
                cash_buffer=config.cash_buffer, data_directory=data_dir,
            )  # fmt: skip
            decision = journal.find_decision(
                config.strategy, decision_date, data_directory=data_dir
            )
            assert decision is not None

        plan, quotes, checked_at = _plan_for(config, decision, broker)
        account = broker.account()
        safety.run_safety_checks(
            broker=broker,
            working_dir=config.working_dir,
            target_weights=decision.target_weights,
            allowed_symbols=config.universe,
            orders=plan.orders,
            equity=account.equity,
            quotes=quotes,
            now=checked_at,
            max_gross_leverage=config.max_gross_leverage,
            max_order_value=config.max_single_order_value,
            max_total_turnover=config.max_total_turnover,
            max_price_age_days=config.max_price_age_days,
            max_price_jump=config.max_price_jump,
        )

        if decision.status == journal.PLANNED and (dry_run or not approve):
            journal.finish_run(run_id, "ok", message="awaiting approval", data_directory=data_dir)
            return CycleResult(run_id, AWAITING_APPROVAL, decision.decision_id, plan)

        if decision.status == journal.PLANNED:
            journal.set_decision_status(
                decision.decision_id, journal.SUBMITTED, data_directory=data_dir
            )
        _submit(decision.decision_id, plan, broker, data_dir=data_dir)

        if _is_resolved(decision.decision_id, not plan.orders, data_dir):
            journal.set_decision_status(
                decision.decision_id, journal.COMPLETED, data_directory=data_dir
            )
            positions = {
                symbol: (p.quantity, p.market_value)
                for symbol, p in broker.account().positions.items()
            }
            journal.record_positions(run_id, positions, data_directory=data_dir)
            journal.finish_run(run_id, "ok", message="completed", data_directory=data_dir)
            return CycleResult(run_id, COMPLETED, decision.decision_id, plan)

        journal.finish_run(run_id, "ok", message="orders pending", data_directory=data_dir)
        return CycleResult(run_id, SUBMITTED, decision.decision_id, plan)

    except (DecisionError, SafetyViolation) as exc:
        did = decision.decision_id if decision is not None else None
        if did is not None:
            journal.set_decision_status(did, journal.FAILED, data_directory=data_dir)
        journal.record_alert(
            str(exc), severity="error", run_id=run_id, did=did, data_directory=data_dir
        )
        journal.finish_run(run_id, "failed", message=str(exc), data_directory=data_dir)
        return CycleResult(run_id, FAILED, did, message=str(exc))
