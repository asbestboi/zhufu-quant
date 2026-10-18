"""Every rule that can keep an order from being sent, each independently testable.

The paper-only guard (:func:`check_paper_account`) reads ``broker.is_paper``, a property
each broker adapter must compute from the account itself; there is no config flag that
can stand in for it, so nothing in a live config can switch it off. Every check raises
:class:`SafetyViolation` with a human-readable reason; the caller (the monthly cycle)
is responsible for sending no orders, journaling the reason as an alert, and stopping.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from qr.backtest.account import AccountState
from qr.live.broker import Broker, PriceQuote
from qr.live.planner import PlannedOrder

KILL_FILE_NAME = "KILL"
KILL_ENV_VAR = "QR_LIVE_KILL"

DEFAULT_MAX_GROSS_LEVERAGE = 1.0
DEFAULT_MAX_ORDER_VALUE = 0.5
DEFAULT_MAX_TOTAL_TURNOVER = 1.2
DEFAULT_MAX_PRICE_AGE_DAYS = 4.0
DEFAULT_MAX_PRICE_JUMP = 0.20


class SafetyViolation(Exception):
    """Raised by a safety check that failed. No order may be sent after one is raised."""


def check_paper_account(broker: Broker) -> None:
    """Refuse unless the broker itself reports a paper account. Cannot be disabled."""
    if not broker.is_paper:
        raise SafetyViolation(
            f"broker account '{broker.account_id}' does not report itself as a paper "
            "account; refusing to send any order (this check cannot be disabled)"
        )


def check_kill_switch(working_dir: Path | str) -> None:
    """Refuse if the ``KILL`` file exists, or ``QR_LIVE_KILL=1`` is set."""
    kill_file = Path(working_dir) / KILL_FILE_NAME
    if kill_file.exists():
        raise SafetyViolation(f"kill switch: {kill_file} exists")
    if os.environ.get(KILL_ENV_VAR) == "1":
        raise SafetyViolation(f"kill switch: {KILL_ENV_VAR}=1 is set")


def check_universe(target_weights: Mapping[str, float], allowed_symbols: Sequence[str]) -> None:
    """Refuse if a target weight names an instrument outside the configured universe."""
    unknown = sorted(set(target_weights) - set(allowed_symbols))
    if unknown:
        raise SafetyViolation(
            f"target weights name instruments outside the configured universe: {', '.join(unknown)}"
        )


def check_no_short_no_leverage(
    target_weights: Mapping[str, float], *, max_gross_leverage: float = DEFAULT_MAX_GROSS_LEVERAGE
) -> None:
    """Refuse a short target weight, or gross weight over ``max_gross_leverage``."""
    negative = sorted(s for s, w in target_weights.items() if w < 0)
    if negative:
        raise SafetyViolation(
            f"target weights are short (long-only is required): {', '.join(negative)}"
        )
    gross = sum(abs(w) for w in target_weights.values())
    if gross > max_gross_leverage + 1e-9:
        raise SafetyViolation(
            f"target weights sum to {gross:.3f}, over the {max_gross_leverage:.0%} leverage limit"
        )


def check_order_limits(
    orders: Sequence[PlannedOrder],
    capital: float,
    *,
    max_order_value: float = DEFAULT_MAX_ORDER_VALUE,
    max_total_turnover: float = DEFAULT_MAX_TOTAL_TURNOVER,
) -> None:
    """Refuse a single order, or the plan's total turnover, over its limit of ``capital``.

    ``capital`` is the account's *managed* capital (``LiveConfig.capital_limit``), not
    its total equity -- the same base the plan itself was sized against.
    """
    if capital <= 0:
        return
    too_big = [o.symbol for o in orders if o.estimated_value > max_order_value * capital]
    if too_big:
        raise SafetyViolation(
            f"orders exceed the {max_order_value:.0%} max-single-order-value limit: "
            f"{', '.join(too_big)}"
        )
    turnover = sum(o.estimated_value for o in orders)
    if turnover > max_total_turnover * capital:
        raise SafetyViolation(
            f"total turnover ${turnover:,.2f} exceeds the {max_total_turnover:.0%} of "
            "managed-capital limit"
        )


def check_capital_limit(
    orders: Sequence[PlannedOrder], account: AccountState, capital_limit: float
) -> None:
    """Refuse a plan whose resulting exposure would exceed the managed capital.

    Computed from this plan's own order sizing (already capital-relative, see
    ``planner.py``) plus every other held position's current market value -- an
    absolute dollar ceiling, independent of whatever weight-leverage limit is
    configured, and a check against the planner having been sized off the wrong number
    in the first place (it should never be, but this does not take that on faith).
    """
    if capital_limit <= 0:
        return
    traded = {o.symbol for o in orders}
    resulting = sum(abs(o.target_weight) * capital_limit for o in orders)
    resulting += sum(abs(p.market_value) for s, p in account.positions.items() if s not in traded)
    if resulting > capital_limit + 1e-6:
        raise SafetyViolation(
            f"plan would expose ${resulting:,.2f}, over the ${capital_limit:,.2f} "
            "managed-capital limit"
        )


def check_price_sanity(
    quotes: Mapping[str, PriceQuote],
    reference: Mapping[str, float],
    *,
    now: datetime,
    max_age_days: float = DEFAULT_MAX_PRICE_AGE_DAYS,
    max_jump: float = DEFAULT_MAX_PRICE_JUMP,
) -> None:
    """Refuse a price older than ``max_age_days``, or more than ``max_jump`` from last time."""
    stale = sorted(
        s for s, q in quotes.items() if (now - q.as_of).total_seconds() / 86400 > max_age_days
    )
    if stale:
        raise SafetyViolation(
            f"stale prices (older than {max_age_days:g} days): {', '.join(stale)}"
        )
    jumped = []
    for symbol, quote in quotes.items():
        prior = reference.get(symbol)
        if prior and prior > 0:
            change = abs(quote.price - prior) / prior
            if change > max_jump:
                jumped.append(f"{symbol} ({change:+.0%})")
    if jumped:
        raise SafetyViolation(
            f"price moved more than {max_jump:.0%} since the last known price: "
            f"{', '.join(sorted(jumped))}"
        )


def run_safety_checks(
    *,
    broker: Broker,
    working_dir: Path | str,
    target_weights: Mapping[str, float],
    allowed_symbols: Sequence[str],
    orders: Sequence[PlannedOrder],
    account: AccountState,
    capital: float,
    quotes: Mapping[str, PriceQuote] | None = None,
    reference_prices: Mapping[str, float] | None = None,
    now: datetime | None = None,
    max_gross_leverage: float = DEFAULT_MAX_GROSS_LEVERAGE,
    max_order_value: float = DEFAULT_MAX_ORDER_VALUE,
    max_total_turnover: float = DEFAULT_MAX_TOTAL_TURNOVER,
    max_price_age_days: float = DEFAULT_MAX_PRICE_AGE_DAYS,
    max_price_jump: float = DEFAULT_MAX_PRICE_JUMP,
) -> None:
    """Run every safety rule, in order, raising on the first that fails.

    The paper-only guard and the kill switch run before anything else looks at the
    decision itself, so a misconfigured or killed run never gets as far as judging it.
    ``capital`` is the managed capital (``LiveConfig.capital_limit``), not the account's
    total equity -- every limit here is a fraction (or, for ``check_capital_limit``, the
    whole) of that number, never of the broader account.
    """
    check_paper_account(broker)
    check_kill_switch(working_dir)
    check_universe(target_weights, allowed_symbols)
    check_no_short_no_leverage(target_weights, max_gross_leverage=max_gross_leverage)
    check_order_limits(
        orders, capital, max_order_value=max_order_value, max_total_turnover=max_total_turnover
    )
    check_capital_limit(orders, account, capital)
    if quotes is not None:
        check_price_sanity(
            quotes,
            reference_prices or {},
            now=now or datetime.now(),
            max_age_days=max_price_age_days,
            max_jump=max_price_jump,
        )
