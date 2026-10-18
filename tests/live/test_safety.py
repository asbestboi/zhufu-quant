"""Every safety rule, each checked on its own and the paper-only guard specifically."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from qr.live.broker import PriceQuote
from qr.live.planner import PlannedOrder
from qr.live.safety import (
    SafetyViolation,
    check_kill_switch,
    check_no_short_no_leverage,
    check_order_limits,
    check_paper_account,
    check_price_sanity,
    check_universe,
    run_safety_checks,
)
from qr.live.sim_broker import SimBroker


class _LiveBroker(SimBroker):
    """A stand-in for a real (non-paper) broker, for the one test that needs it."""

    @property
    def is_paper(self) -> bool:
        return False


def order(
    symbol: str = "AAA", side: str = "buy", quantity: float = 1.0, price: float = 100.0
) -> PlannedOrder:
    return PlannedOrder(
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        quantity=quantity,
        price=price,
        estimated_value=quantity * price,
        current_weight=0.0,
        target_weight=0.1,
    )


def test_paper_only_guard_passes_a_paper_broker() -> None:
    check_paper_account(SimBroker())


def test_paper_only_guard_rejects_a_non_paper_broker() -> None:
    with pytest.raises(SafetyViolation, match="paper account"):
        check_paper_account(_LiveBroker())


def test_kill_switch_file(tmp_path: Path) -> None:
    check_kill_switch(tmp_path)  # no KILL file: fine
    (tmp_path / "KILL").touch()
    with pytest.raises(SafetyViolation, match="kill switch"):
        check_kill_switch(tmp_path)


def test_kill_switch_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QR_LIVE_KILL", "1")
    with pytest.raises(SafetyViolation, match="QR_LIVE_KILL"):
        check_kill_switch(tmp_path)


def test_universe_check_rejects_an_unlisted_symbol() -> None:
    check_universe({"AAA": 0.5}, ["AAA", "BBB"])
    with pytest.raises(SafetyViolation, match="CCC"):
        check_universe({"AAA": 0.5, "CCC": 0.5}, ["AAA", "BBB"])


def test_long_only_rejects_a_short_weight() -> None:
    with pytest.raises(SafetyViolation, match="short"):
        check_no_short_no_leverage({"AAA": -0.1})


def test_leverage_limit() -> None:
    check_no_short_no_leverage({"AAA": 0.5, "BBB": 0.5})
    with pytest.raises(SafetyViolation, match="leverage"):
        check_no_short_no_leverage({"AAA": 0.6, "BBB": 0.6})


def test_max_single_order_value() -> None:
    orders = [order(quantity=10, price=100.0)]  # $1,000
    check_order_limits(orders, equity=10_000, max_order_value=0.5)
    with pytest.raises(SafetyViolation, match="max-single-order-value"):
        check_order_limits(orders, equity=10_000, max_order_value=0.05)


def test_max_total_turnover() -> None:
    orders = [order("AAA", quantity=5, price=100.0), order("BBB", quantity=5, price=100.0)]
    check_order_limits(orders, equity=10_000, max_order_value=1.0, max_total_turnover=1.2)
    with pytest.raises(SafetyViolation, match="turnover"):
        check_order_limits(orders, equity=500, max_order_value=1.0, max_total_turnover=1.2)


def test_stale_price_is_rejected() -> None:
    now = datetime(2024, 1, 10)
    quotes = {"AAA": PriceQuote(price=100.0, as_of=datetime(2024, 1, 1))}
    with pytest.raises(SafetyViolation, match="stale"):
        check_price_sanity(quotes, {}, now=now, max_age_days=4.0)


def test_price_jump_is_rejected() -> None:
    now = datetime(2024, 1, 10)
    quotes = {"AAA": PriceQuote(price=150.0, as_of=now)}
    check_price_sanity(quotes, {"AAA": 149.0}, now=now, max_jump=0.20)
    with pytest.raises(SafetyViolation, match="moved more than"):
        check_price_sanity(quotes, {"AAA": 100.0}, now=now, max_jump=0.20)


def test_run_safety_checks_runs_the_paper_guard_before_anything_else(tmp_path: Path) -> None:
    # A non-paper broker fails even with an otherwise-fine (empty) decision.
    with pytest.raises(SafetyViolation, match="paper account"):
        run_safety_checks(
            broker=_LiveBroker(),
            working_dir=tmp_path,
            target_weights={},
            allowed_symbols=[],
            orders=[],
            equity=0.0,
        )


def test_run_safety_checks_passes_a_clean_plan(tmp_path: Path) -> None:
    run_safety_checks(
        broker=SimBroker(),
        working_dir=tmp_path,
        target_weights={"AAA": 0.5},
        allowed_symbols=["AAA", "BBB"],
        orders=[order(quantity=5, price=100.0)],
        equity=10_000.0,
    )
