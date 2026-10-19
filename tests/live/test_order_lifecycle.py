"""qr.live.cycle._submit_and_track: submit, poll, cancel stragglers at the window end,
resumable and idempotent across restarts -- against a SimBroker that fills slowly,
partially, or never.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from qr.backtest.events import BUY, Side
from qr.live import journal
from qr.live.cycle import _submit_and_track
from qr.live.planner import Plan, PlannedOrder
from qr.live.sim_broker import SimBroker

DID = "test:2024-01-31"
START = datetime(2024, 1, 31, 15, 0, 0)


class FakeClock:
    """A controllable stand-in for wall-clock time, advanced only by ``sleep``."""

    def __init__(self, start: datetime) -> None:
        self.current = start

    def now(self) -> datetime:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def _tracking_clock(
    broker: SimBroker, start: datetime = START
) -> tuple[FakeClock, Callable[[float], None]]:
    """A :class:`FakeClock` whose ``sleep`` also advances ``broker``'s simulated fills."""
    clock = FakeClock(start)

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        broker.tick()

    return clock, sleep


def _plan_for(symbol: str, side: Side = BUY, quantity: float = 10.0, price: float = 100.0) -> Plan:
    order = PlannedOrder(
        symbol=symbol, side=side, quantity=quantity, price=price,
        estimated_value=quantity * price, current_weight=0.0, target_weight=0.5,
    )  # fmt: skip
    return Plan(orders=[order], skipped={}, capital=10_000.0, cash_buffer=0.0)


def test_an_order_that_fills_immediately_needs_no_polling(tmp_path: Path) -> None:
    broker = SimBroker()
    broker.set_prices({"AAA": 100.0})
    _submit_and_track(
        DID, _plan_for("AAA"), broker,
        order_window_minutes=30, poll_interval_seconds=5, poll_timeout_seconds=60,
        data_dir=tmp_path,
    )  # fmt: skip
    orders = journal.read_orders(DID, data_directory=tmp_path)
    assert len(orders) == 1
    assert orders.iloc[0]["status"] == "filled"
    fills = journal.read_fills(DID, data_directory=tmp_path)
    assert fills.iloc[0]["quantity"] == 10.0


def test_a_slow_fill_resolves_within_the_window_without_being_cancelled(tmp_path: Path) -> None:
    broker = SimBroker(fill_delay_ticks=2)
    broker.set_prices({"AAA": 100.0})
    clock, sleep = _tracking_clock(broker)
    _submit_and_track(
        DID, _plan_for("AAA"), broker,
        order_window_minutes=30, poll_interval_seconds=5, poll_timeout_seconds=60,
        data_dir=tmp_path, now=clock.now, sleep=sleep,
    )  # fmt: skip
    orders = journal.read_orders(DID, data_directory=tmp_path)
    assert orders.iloc[0]["status"] == "filled"
    assert journal.read_fills(DID, data_directory=tmp_path).iloc[0]["quantity"] == 10.0
    assert broker.account().positions["AAA"].quantity == 10.0


def test_a_partial_fill_is_cancelled_and_recorded_unfilled_at_the_window_end(
    tmp_path: Path,
) -> None:
    broker = SimBroker(fill_behavior="partial", fill_delay_ticks=1, partial_fraction=0.4)
    broker.set_prices({"AAA": 100.0})
    clock, sleep = _tracking_clock(broker)
    _submit_and_track(
        DID, _plan_for("AAA", quantity=10.0), broker,
        order_window_minutes=1, poll_interval_seconds=10, poll_timeout_seconds=600,
        data_dir=tmp_path, now=clock.now, sleep=sleep,
    )  # fmt: skip

    orders = journal.read_orders(DID, data_directory=tmp_path)
    assert orders.iloc[0]["status"] == "cancelled"
    fills = journal.read_fills(DID, data_directory=tmp_path)
    assert fills.iloc[0]["quantity"] == 4.0  # 40% of 10
    assert broker.account().positions["AAA"].quantity == 4.0

    alerts = journal.read_alerts(data_directory=tmp_path)
    message = alerts.loc[alerts["decision_id"] == DID, "message"].iloc[0]
    assert "6" in message and "unfilled" in message and "not chased" in message


def test_an_order_that_never_fills_is_cancelled_at_the_window_end(tmp_path: Path) -> None:
    broker = SimBroker(fill_behavior="never")
    broker.set_prices({"AAA": 100.0})
    clock, sleep = _tracking_clock(broker)
    _submit_and_track(
        DID, _plan_for("AAA"), broker,
        order_window_minutes=1, poll_interval_seconds=10, poll_timeout_seconds=600,
        data_dir=tmp_path, now=clock.now, sleep=sleep,
    )  # fmt: skip

    orders = journal.read_orders(DID, data_directory=tmp_path)
    assert orders.iloc[0]["status"] == "cancelled"
    assert journal.read_fills(DID, data_directory=tmp_path).empty
    assert broker.account().positions == {}
    assert broker.open_orders() == []


def test_a_poll_timeout_before_the_window_ends_leaves_the_order_open_for_later(
    tmp_path: Path,
) -> None:
    broker = SimBroker(fill_behavior="never")
    broker.set_prices({"AAA": 100.0})
    clock, sleep = _tracking_clock(broker)
    _submit_and_track(
        DID, _plan_for("AAA"), broker,
        order_window_minutes=30, poll_interval_seconds=5, poll_timeout_seconds=12,
        data_dir=tmp_path, now=clock.now, sleep=sleep,
    )  # fmt: skip

    orders = journal.read_orders(DID, data_directory=tmp_path)
    assert orders.iloc[0]["status"] == "open"  # neither filled nor cancelled yet
    assert broker.open_orders() != []  # never cancelled -- the window hasn't ended


def test_resuming_after_a_poll_timeout_never_resubmits_and_keeps_tracking(tmp_path: Path) -> None:
    broker = SimBroker(fill_behavior="never")
    broker.set_prices({"AAA": 100.0})
    clock, sleep = _tracking_clock(broker)
    plan = _plan_for("AAA")

    _submit_and_track(
        DID, plan, broker,
        order_window_minutes=30, poll_interval_seconds=5, poll_timeout_seconds=12,
        data_dir=tmp_path, now=clock.now, sleep=sleep,
    )  # fmt: skip
    first_order_id = journal.read_orders(DID, data_directory=tmp_path).iloc[0]["order_id"]

    # Simulate the broker now reporting this order filled, then resume tracking it --
    # a second call to _submit_and_track must not place a second order for AAA.
    broker._orders[first_order_id].status = "filled"
    broker._orders[first_order_id].filled_quantity = 10.0
    _submit_and_track(
        DID, plan, broker,
        order_window_minutes=30, poll_interval_seconds=5, poll_timeout_seconds=60,
        data_dir=tmp_path, now=clock.now, sleep=sleep,
    )  # fmt: skip

    orders = journal.read_orders(DID, data_directory=tmp_path)
    assert len(orders) == 1  # never resubmitted
    assert orders.iloc[0]["order_id"] == first_order_id
    assert orders.iloc[0]["status"] == "filled"


def test_the_order_window_is_anchored_to_the_first_submission_not_each_resume(
    tmp_path: Path,
) -> None:
    broker = SimBroker(fill_behavior="never")
    broker.set_prices({"AAA": 100.0})
    clock, sleep = _tracking_clock(broker)
    plan = _plan_for("AAA")

    # First call: submits, polls briefly, times out with the order still open.
    _submit_and_track(
        DID, plan, broker,
        order_window_minutes=1, poll_interval_seconds=20, poll_timeout_seconds=20,
        data_dir=tmp_path, now=clock.now, sleep=sleep,
    )  # fmt: skip
    assert journal.read_orders(DID, data_directory=tmp_path).iloc[0]["status"] == "open"

    # Second call, resumed: even though *this* call's own poll_timeout is generous, the
    # order window (anchored to the first submission) has already elapsed, so the
    # straggler is cancelled here rather than waiting out a fresh window.
    _submit_and_track(
        DID, plan, broker,
        order_window_minutes=1, poll_interval_seconds=5, poll_timeout_seconds=600,
        data_dir=tmp_path, now=clock.now, sleep=sleep,
    )  # fmt: skip
    orders = journal.read_orders(DID, data_directory=tmp_path)
    assert len(orders) == 1
    assert orders.iloc[0]["status"] == "cancelled"


def test_two_symbols_track_independently(tmp_path: Path) -> None:
    broker = SimBroker(fill_behavior="full", fill_delay_ticks=1)
    broker.set_prices({"AAA": 100.0, "BBB": 50.0})
    clock, sleep = _tracking_clock(broker)
    plan = Plan(
        orders=[
            PlannedOrder("AAA", "buy", 10.0, 100.0, 1_000.0, 0.0, 0.5),
            PlannedOrder("BBB", "buy", 20.0, 50.0, 1_000.0, 0.0, 0.5),
        ],
        skipped={},
        capital=10_000.0,
        cash_buffer=0.0,
    )
    _submit_and_track(
        DID, plan, broker,
        order_window_minutes=30, poll_interval_seconds=5, poll_timeout_seconds=60,
        data_dir=tmp_path, now=clock.now, sleep=sleep,
    )  # fmt: skip
    orders = journal.read_orders(DID, data_directory=tmp_path)
    assert len(orders) == 2
    assert set(orders["status"]) == {"filled"}
    account = broker.account()
    assert account.positions["AAA"].quantity == 10.0
    assert account.positions["BBB"].quantity == 20.0
