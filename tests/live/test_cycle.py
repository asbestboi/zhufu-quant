"""The monthly cycle: idempotency, resume after a crash, kill switch, price sanity, and
the big end-to-end check that a live run and the shadow backtest agree."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from qr import calendars
from qr.live import journal
from qr.live.config import LiveConfig
from qr.live.cycle import (
    ALREADY_COMPLETED,
    ALREADY_EXPIRED,
    COMPLETED,
    FAILED,
    NO_DECISION,
    SUBMITTED,
    is_month_end,
    run_cycle,
)
from qr.live.sim_broker import SimBroker

from .live_config import make_live_config
from .live_store import real_session_bars, seed_store

SYMBOLS = ["AAA", "BBB"]


def _jan_2023_end() -> date:
    sessions = calendars.sessions("etf", date(2023, 1, 1), date(2023, 1, 31))
    return sessions[-1].date()


def _seeded_two_symbol_store(data_dir: Path) -> tuple[list[str], date, dict[str, float]]:
    as_of = _jan_2023_end()
    bars = {
        symbol: real_session_bars(date(2022, 1, 1), date(2023, 2, 15), seed=i)
        for i, symbol in enumerate(SYMBOLS)
    }
    seed_store(data_dir, bars)
    closes = {
        symbol: float(frame.loc[frame["date"].dt.date == as_of, "close"].iloc[0])
        for symbol, frame in bars.items()
    }
    return SYMBOLS, as_of, closes


def test_is_month_end() -> None:
    as_of = _jan_2023_end()
    assert is_month_end(as_of)
    assert not is_month_end(date(2023, 1, 3))


def test_no_decision_when_today_is_not_a_month_end(tmp_path: Path) -> None:
    config = make_live_config(SYMBOLS, tmp_path)
    broker = SimBroker()
    result = run_cycle(config, broker, now=datetime(2023, 1, 18, 20, 0), data_dir=tmp_path)
    assert result.status == NO_DECISION


def test_dry_run_never_touches_the_broker(tmp_path: Path) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = make_live_config(symbols, tmp_path)
    broker = SimBroker()
    broker.set_prices(closes)
    result = run_cycle(config, broker, dry_run=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == "awaiting_approval"
    assert result.plan is not None and len(result.plan.orders) == 2
    assert broker.account().positions == {}


def test_approve_submits_and_completes(tmp_path: Path) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = make_live_config(symbols, tmp_path)
    broker = SimBroker()
    broker.set_prices(closes)
    result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == COMPLETED
    account = broker.account()
    assert set(account.positions) == set(symbols)


def test_rerunning_the_same_decision_is_idempotent(tmp_path: Path) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = make_live_config(symbols, tmp_path)
    broker = SimBroker()
    broker.set_prices(closes)
    run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    positions_after_first = dict(broker.account().positions)

    result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == ALREADY_COMPLETED
    assert broker.account().positions.keys() == positions_after_first.keys()
    for symbol, position in broker.account().positions.items():
        assert position.quantity == positions_after_first[symbol].quantity
    did = journal.decision_id(config.strategy, as_of)
    assert len(journal.read_orders(did, data_directory=tmp_path)) == 2


def test_an_expired_decision_never_sends_orders_however_it_is_resumed(tmp_path: Path) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = make_live_config(symbols, tmp_path)
    broker = SimBroker()
    broker.set_prices(closes)
    run_cycle(config, broker, dry_run=True, as_of=as_of, data_dir=tmp_path)  # plan it
    did = journal.decision_id(config.strategy, as_of)
    journal.set_decision_status(did, journal.EXPIRED, data_directory=tmp_path)

    result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == ALREADY_EXPIRED
    assert broker.account().positions == {}
    assert journal.read_orders(did, data_directory=tmp_path).empty


def test_resume_after_a_simulated_crash_between_submit_and_fill(tmp_path: Path) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = make_live_config(symbols, tmp_path)
    broker = SimBroker()
    broker.set_prices(closes)

    planned = run_cycle(config, broker, dry_run=True, as_of=as_of, data_dir=tmp_path)
    assert planned.plan is not None and len(planned.plan.orders) == 2
    did = journal.decision_id(config.strategy, as_of)

    # Simulate a crash: the first order reached the broker and filled, but the process
    # died before the second order was even attempted.
    first = planned.plan.orders[0]
    journal.set_decision_status(did, journal.SUBMITTED, data_directory=tmp_path)
    filled = broker.place_order(first.symbol, first.side, first.quantity)
    journal.record_order(
        filled.id, did, first.symbol, first.side, first.quantity, first.price, filled.status,
        data_directory=tmp_path,
    )  # fmt: skip
    journal.record_fill(
        filled.id, did, first.symbol, first.side, filled.filled_quantity, first.price, 0.0,
        data_directory=tmp_path,
    )  # fmt: skip

    result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == COMPLETED
    orders = journal.read_orders(did, data_directory=tmp_path)
    assert len(orders) == 2  # the first order was never resubmitted
    assert (orders["symbol"] == first.symbol).sum() == 1

    account = broker.account()
    assert set(account.positions) == set(symbols)


def _slow_poll_config(symbols: list[str], tmp_path: Path, **overrides: object) -> LiveConfig:
    """A config whose poll loop times out almost immediately, for a fast unit test."""
    return make_live_config(
        symbols, tmp_path, order_poll_interval_seconds=0.01, order_poll_timeout_seconds=0.02,
        **overrides,
    )  # fmt: skip


def test_orders_still_open_after_the_poll_timeout_leave_the_decision_submitted(
    tmp_path: Path,
) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = _slow_poll_config(symbols, tmp_path)
    broker = SimBroker(fill_behavior="never")
    broker.set_prices(closes)

    result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == SUBMITTED
    assert broker.account().positions == {}
    assert len(broker.open_orders()) == 2


def test_dry_run_never_disturbs_an_already_submitted_decision(tmp_path: Path) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = _slow_poll_config(symbols, tmp_path)
    broker = SimBroker(fill_behavior="never")
    broker.set_prices(closes)
    run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    did = journal.decision_id(config.strategy, as_of)
    orders_before = journal.read_orders(did, data_directory=tmp_path)

    result = run_cycle(config, broker, dry_run=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == SUBMITTED  # not AWAITING_APPROVAL: this decision was already sent
    orders_after = journal.read_orders(did, data_directory=tmp_path)
    assert len(orders_after) == len(orders_before) == 2
    assert len(broker.open_orders()) == 2  # dry run touched neither order


def test_resuming_a_submitted_decision_completes_once_orders_resolve(tmp_path: Path) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = _slow_poll_config(symbols, tmp_path)
    broker = SimBroker(fill_behavior="never")
    broker.set_prices(closes)
    run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    did = journal.decision_id(config.strategy, as_of)
    assert len(journal.read_orders(did, data_directory=tmp_path)) == 2

    # Simulate the broker now reporting both orders filled (whatever happened on IB's
    # side between calls): resuming must pick these up, not resubmit new orders.
    for order in broker._orders.values():
        order.status = "filled"
        order.filled_quantity = order.quantity

    result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == COMPLETED
    orders = journal.read_orders(did, data_directory=tmp_path)
    assert len(orders) == 2  # never resubmitted
    assert set(orders["status"]) == {"filled"}


def test_kill_switch_blocks_submission_and_journals_an_alert(tmp_path: Path) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = make_live_config(symbols, tmp_path)
    (tmp_path / "KILL").touch()
    broker = SimBroker()
    broker.set_prices(closes)

    result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == FAILED
    assert broker.account().positions == {}
    alerts = journal.read_alerts(data_directory=tmp_path)
    assert alerts["message"].str.contains("kill switch").any()


def test_stale_prices_block_submission(tmp_path: Path) -> None:
    symbols, as_of, closes = _seeded_two_symbol_store(tmp_path)
    config = make_live_config(symbols, tmp_path)
    broker = SimBroker()
    broker.set_prices(closes, now=datetime(2000, 1, 1))

    result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
    assert result.status == FAILED
    assert "stale" in result.message
    assert broker.account().positions == {}
