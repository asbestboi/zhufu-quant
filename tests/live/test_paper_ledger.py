"""PaperLedgerBroker: fills day limit orders against real downloaded bars, journaled.

Fills at the open, a limit not reached, partial resolution across separate broker
instances (standing in for separate ``qr live run`` processes), the cost model, bit-for-
bit determinism, and a 3-month run matching the backtester are each their own test; the
rest exercise a thin slice directly against the journal, the same way a resumed order
actually gets looked up (no in-memory broker state survives between calls).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from qr import calendars
from qr.backtest.config import BacktestConfig, CostConfig
from qr.backtest.data import load_market_data
from qr.live import journal
from qr.live.cycle import COMPLETED, run_cycle
from qr.live.paper_ledger import PaperLedgerBroker
from qr.load import fingerprint

from ..bt_helpers import NO_COSTS, run
from .live_config import make_live_config
from .live_store import seed_store

STRATEGY = "test-ledger"
SYMBOL = "AAA"
DECISION_DATE = date(2023, 1, 31)


def _bar_frame(rows: list[tuple[date, float, float, float, float]]) -> pd.DataFrame:
    dates, opens, highs, lows, closes = zip(*rows, strict=False)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(list(dates)),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1_000_000.0,
        }
    )


def _sessions_after(decision_date: date, n: int) -> list[date]:
    window = calendars.sessions(
        "etf", decision_date + timedelta(days=1), decision_date + timedelta(days=30)
    )
    return [pd.Timestamp(ts).date() for ts in window[:n]]


def _seed_order(
    tmp_path: Path,
    *,
    side: str = "buy",
    quantity: float = 10.0,
    planned_price: float = 100.0,
    order_id: str = "PLTEST01",
    strategy: str = STRATEGY,
    decision_date: date = DECISION_DATE,
) -> tuple[str, str]:
    """Journal a decision and an open order for it, as ``_submit_and_track`` would have."""
    did = journal.decision_id(strategy, decision_date)
    if journal.find_decision(strategy, decision_date, data_directory=tmp_path) is None:
        journal.record_decision(strategy, decision_date, {SYMBOL: 1.0}, data_directory=tmp_path)
    journal.record_order(
        order_id, did, SYMBOL, side, quantity, planned_price, "open",
        submitted_at=datetime.combine(decision_date, datetime.min.time()),
        data_directory=tmp_path,
    )  # fmt: skip
    return did, order_id


def _broker(tmp_path: Path, **overrides: object) -> PaperLedgerBroker:
    kwargs: dict[str, object] = {
        "strategy": STRATEGY,
        "universe": [SYMBOL],
        "capital_limit": 100_000.0,
        "data_dir": tmp_path,
        "costs": CostConfig(**NO_COSTS),
    }
    kwargs.update(overrides)
    return PaperLedgerBroker(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- fills at the open


def test_a_buy_fills_at_the_next_sessions_open_when_the_limit_allows_it(tmp_path: Path) -> None:
    session = _sessions_after(DECISION_DATE, 1)[0]
    seed_store(tmp_path, {SYMBOL: _bar_frame([(session, 100.0, 101.0, 99.5, 100.5)])})
    _did, order_id = _seed_order(tmp_path, side="buy", quantity=10.0, planned_price=100.0)

    status = _broker(tmp_path).order_status(order_id)

    assert status.status == "filled"
    assert status.filled_quantity == 10.0
    fill = journal.find_fill_by_order(order_id, data_directory=tmp_path)
    assert fill is not None
    assert fill["price"] == 100.0  # NO_COSTS: fills exactly at the open, zero spread/slippage
    assert (
        fill["data_hash"]
        == fingerprint([SYMBOL], start=session, end=session, data_dir=tmp_path).hash
    )


def test_a_sell_fills_at_the_open_when_the_limit_allows_it(tmp_path: Path) -> None:
    session = _sessions_after(DECISION_DATE, 1)[0]
    seed_store(tmp_path, {SYMBOL: _bar_frame([(session, 100.0, 101.0, 99.0, 100.5)])})
    journal.record_fill(
        "PRIOR", journal.decision_id(STRATEGY, DECISION_DATE), SYMBOL, "buy", 10.0, 100.0, 0.0,
        filled_at=datetime.combine(DECISION_DATE, datetime.min.time()), data_directory=tmp_path,
    )  # fmt: skip
    _did, order_id = _seed_order(tmp_path, side="sell", quantity=10.0, planned_price=100.0)

    status = _broker(tmp_path).order_status(order_id)

    assert status.status == "filled"
    assert status.filled_quantity == 10.0
    fill = journal.find_fill_by_order(order_id, data_directory=tmp_path)
    assert fill is not None and fill["price"] == 100.0


# ---------------------------------------------------------------------- limit not reached


def test_an_order_that_never_crosses_its_limit_is_cancelled_after_the_window(
    tmp_path: Path,
) -> None:
    sessions = _sessions_after(DECISION_DATE, 3)
    # Gaps up and never trades back down to the (lower) buy limit, for every session in
    # the window.
    bars = _bar_frame([(s, 105.0, 106.0, 104.0, 105.5) for s in sessions])
    seed_store(tmp_path, {SYMBOL: bars})
    did, order_id = _seed_order(tmp_path, side="buy", quantity=10.0, planned_price=100.0)

    status = _broker(tmp_path, window_sessions=3).order_status(order_id)

    assert status.status == "cancelled"
    assert status.filled_quantity == 0.0
    assert journal.find_fill_by_order(order_id, data_directory=tmp_path) is None
    alerts = journal.read_alerts(data_directory=tmp_path)
    message = alerts.loc[alerts["decision_id"] == did, "message"].iloc[0]
    assert "unfilled after 3 sessions" in message and "not chased" in message


def test_a_limit_order_fills_passively_when_a_later_session_trades_through_it(
    tmp_path: Path,
) -> None:
    sessions = _sessions_after(DECISION_DATE, 2)
    # Session 1 gaps above the limit and stays there (no fill); session 2's low dips
    # below the limit (100.0 * 1.003 = 100.3), filling passively at the limit itself.
    bars = _bar_frame(
        [
            (sessions[0], 105.0, 106.0, 104.0, 105.0),
            (sessions[1], 104.0, 104.5, 99.0, 101.0),
        ]
    )
    seed_store(tmp_path, {SYMBOL: bars})
    _did, order_id = _seed_order(tmp_path, side="buy", quantity=10.0, planned_price=100.0)

    status = _broker(tmp_path).order_status(order_id)

    assert status.status == "filled"
    assert status.limit_price == 100.3
    fill = journal.find_fill_by_order(order_id, data_directory=tmp_path)
    assert fill is not None
    assert fill["price"] == 100.3  # passive fill, at the limit, no costs under NO_COSTS


# ---------------------------------------------------------------------- partial resolution


def test_an_order_stays_open_until_the_needed_session_is_downloaded(tmp_path: Path) -> None:
    sessions = _sessions_after(DECISION_DATE, 2)
    # The decision's own close is already ingested (it had to be, to plan the order at
    # all) -- only the sessions after it are still missing.
    seed_store(tmp_path, {SYMBOL: _bar_frame([(DECISION_DATE, 100.0, 100.5, 99.5, 100.0)])})
    _did, order_id = _seed_order(tmp_path, side="buy", quantity=10.0, planned_price=100.0)

    # The first candidate session has not arrived yet; the order stays open.
    status = _broker(tmp_path).order_status(order_id)
    assert status.status == "open"
    assert journal.find_fill_by_order(order_id, data_directory=tmp_path) is None

    # "Next run": session 1's bar has since been ingested (still gaps above the limit,
    # so it alone would not resolve anything), but session 2 is still missing.
    seed_store(tmp_path, {SYMBOL: _bar_frame([(sessions[0], 105.0, 106.0, 104.0, 105.0)])})
    status = _broker(tmp_path).order_status(order_id)
    assert status.status == "open"

    # A later run: session 2 has arrived and trades through the limit.
    seed_store(tmp_path, {SYMBOL: _bar_frame([(sessions[1], 104.0, 104.5, 99.0, 101.0)])})
    status = _broker(tmp_path).order_status(order_id)
    assert status.status == "filled"
    assert journal.find_fill_by_order(order_id, data_directory=tmp_path) is not None

    # Resuming again afterwards must not re-resolve or re-record the fill.
    status = _broker(tmp_path).order_status(order_id)
    assert status.status == "filled"
    assert status.filled_quantity == 10.0


# ---------------------------------------------------------------------- costs applied correctly


def test_an_at_open_fill_pays_full_costs_a_passive_fill_pays_commission_only(
    tmp_path: Path,
) -> None:
    costs = CostConfig(
        commission_model="per_share", commission_per_share=0.01, commission_min_per_order=0.0,
        spread_bps=10.0, slippage_bps=20.0,
    )  # fmt: skip

    # At the open: full spread + slippage push the fill price worse than the open.
    session = _sessions_after(DECISION_DATE, 1)[0]
    seed_store(tmp_path, {SYMBOL: _bar_frame([(session, 100.0, 101.0, 99.5, 100.5)])})
    _did, order_id = _seed_order(tmp_path, side="buy", quantity=10.0, planned_price=100.0)
    status = _broker(tmp_path, costs=costs).order_status(order_id)
    assert status.status == "filled"
    fill = journal.find_fill_by_order(order_id, data_directory=tmp_path)
    assert fill is not None
    expected_commission = 0.01 * 10.0
    expected_slippage_dollars = 100.0 * 10.0 * (10.0 + 20.0) / 1e4
    expected_price = 100.0 + expected_slippage_dollars / 10.0
    assert fill["commission"] == expected_commission
    assert fill["price"] == expected_price

    # Passive (filled through the day at the limit): commission only, no spread/slippage.
    sessions = _sessions_after(DECISION_DATE, 2)
    seed_store(
        tmp_path,
        {
            "BBB": _bar_frame(
                [
                    (sessions[0], 105.0, 106.0, 104.0, 105.0),
                    (sessions[1], 104.0, 104.5, 99.0, 101.0),
                ]
            )
        },
    )
    did2 = journal.decision_id(STRATEGY, DECISION_DATE)
    journal.record_order(
        "PLPASSIVE", did2, "BBB", "buy", 10.0, 100.0, "open",
        submitted_at=datetime.combine(DECISION_DATE, datetime.min.time()), data_directory=tmp_path,
    )  # fmt: skip
    broker = PaperLedgerBroker(
        strategy=STRATEGY, universe=[SYMBOL, "BBB"], capital_limit=100_000.0,
        data_dir=tmp_path, costs=costs,
    )  # fmt: skip
    status = broker.order_status("PLPASSIVE")
    assert status.status == "filled"
    fill = journal.find_fill_by_order("PLPASSIVE", data_directory=tmp_path)
    assert fill is not None
    assert fill["commission"] == 0.01 * 10.0
    assert fill["price"] == 100.3  # the limit itself: no spread or slippage on a passive fill


# ---------------------------------------------------------------------- determinism


def test_the_same_order_and_data_always_produce_the_same_fill(tmp_path: Path) -> None:
    session = _sessions_after(DECISION_DATE, 1)[0]
    seed_store(tmp_path, {SYMBOL: _bar_frame([(session, 100.0, 101.0, 99.5, 100.5)])})
    costs = CostConfig(
        commission_model="per_share", commission_per_share=0.01, spread_bps=5.0, slippage_bps=5.0
    )

    _, order_a = _seed_order(tmp_path, order_id="PLDET_A", planned_price=100.0)
    status_a = _broker(tmp_path, costs=costs).order_status(order_a)

    _, order_b = _seed_order(tmp_path, order_id="PLDET_B", planned_price=100.0)
    status_b = PaperLedgerBroker(  # a second, independent instance -- no shared state
        strategy=STRATEGY, universe=[SYMBOL], capital_limit=100_000.0,
        data_dir=tmp_path, costs=costs,
    ).order_status(order_b)  # fmt: skip

    assert status_a.status == status_b.status == "filled"
    fill_a = journal.find_fill_by_order(order_a, data_directory=tmp_path)
    fill_b = journal.find_fill_by_order(order_b, data_directory=tmp_path)
    assert fill_a is not None and fill_b is not None
    assert fill_a["price"] == fill_b["price"]
    assert fill_a["commission"] == fill_b["commission"]
    assert fill_a["data_hash"] == fill_b["data_hash"]
    assert (
        fill_a["data_hash"]
        == fingerprint([SYMBOL], start=session, end=session, data_dir=tmp_path).hash
    )


# ---------------------------------------------------------------------- no shorting


def test_a_sell_larger_than_the_position_is_clamped_not_a_short(tmp_path: Path) -> None:
    session = _sessions_after(DECISION_DATE, 1)[0]
    seed_store(tmp_path, {SYMBOL: _bar_frame([(session, 100.0, 101.0, 99.0, 100.5)])})
    journal.record_fill(
        "PRIOR", journal.decision_id(STRATEGY, DECISION_DATE), SYMBOL, "buy", 5.0, 100.0, 0.0,
        filled_at=datetime.combine(DECISION_DATE, datetime.min.time()), data_directory=tmp_path,
    )  # fmt: skip
    _did, order_id = _seed_order(tmp_path, side="sell", quantity=10.0, planned_price=100.0)

    status = _broker(tmp_path).order_status(order_id)

    assert status.status == "filled"
    assert status.filled_quantity == 5.0  # clamped to what was actually held


def test_a_sell_with_no_position_at_all_is_cancelled_not_a_short(tmp_path: Path) -> None:
    session = _sessions_after(DECISION_DATE, 1)[0]
    seed_store(tmp_path, {SYMBOL: _bar_frame([(session, 100.0, 101.0, 99.0, 100.5)])})
    _did, order_id = _seed_order(tmp_path, side="sell", quantity=10.0, planned_price=100.0)

    status = _broker(tmp_path).order_status(order_id)

    assert status.status == "cancelled"
    assert status.filled_quantity == 0.0


# ---------------------------------------------------------------------- 3-month parity


def _flat_bars(start: date, end: date, price: float) -> pd.DataFrame:
    dates = calendars.sessions("etf", start, end)
    return pd.DataFrame(
        {
            "date": dates, "open": price, "high": price, "low": price, "close": price,
            "volume": 1_000_000.0,
        }
    )  # fmt: skip


def _month_ends(start: date, end: date) -> list[date]:
    sessions = calendars.sessions("etf", start, end)
    by_month = sessions.to_series().dt.to_period("M")
    return [sessions[by_month == m][-1].date() for m in by_month.unique()]


def test_three_monthly_cycles_match_the_backtest_within_whole_shares(tmp_path: Path) -> None:
    symbols = ["AAA", "BBB", "CCC"]
    prices = {"AAA": 100.0, "BBB": 90.0, "CCC": 110.0}
    start, end = date(2022, 11, 1), date(2023, 4, 15)
    bars = {symbol: _flat_bars(start, end, price) for symbol, price in prices.items()}
    seed_store(tmp_path, bars)
    decision_dates = _month_ends(date(2023, 1, 1), date(2023, 3, 31))
    assert len(decision_dates) == 3

    params = {"lookback": 10, "skip": 2, "top_n": 2}
    config = make_live_config(
        symbols, tmp_path, strategy="xs_momentum", params=params, broker="paper_ledger",
        capital_limit=100_000.0, order_poll_interval_seconds=0.01, order_poll_timeout_seconds=1.0,
        order_window_minutes=5.0,
        # The simulated "today" is a 2023 decision date, honestly reported as this
        # quote's age (qr.live.broker.PriceQuote's own contract) -- the staleness check
        # is comparing it against the real wall clock, years later, so it must be
        # widened here for the test's time travel, not for anything a real run needs.
        max_price_age_days=100_000.0,
    )  # fmt: skip

    class _Clock:
        day = decision_dates[0]

        def today(self) -> date:
            return self.day

    clock = _Clock()
    broker = PaperLedgerBroker(
        strategy=config.strategy, universe=symbols, capital_limit=config.capital_limit,
        data_dir=tmp_path, today=clock.today, costs=CostConfig(**NO_COSTS),
    )  # fmt: skip
    for as_of in decision_dates:
        clock.day = as_of
        result = run_cycle(config, broker, approve=True, as_of=as_of, data_dir=tmp_path)
        assert result.status == COMPLETED, result.message

    live_positions = broker.account().positions
    assert live_positions

    data = load_market_data(symbols, decision_dates[-1], data_dir=tmp_path)
    backtest_config = BacktestConfig(
        name="reference", engine="event", strategy="xs_momentum", params=params,
        universe=symbols, start=start, end=decision_dates[-1], initial_cash=config.capital_limit,
        benchmark=None, rebalance="monthly", costs=CostConfig(**NO_COSTS), cost_stress=[],
    )  # fmt: skip
    reference = run(data, backtest_config)
    assert reference.final_state is not None
    reference_positions = reference.final_state.positions

    for symbol in symbols:
        live_qty = live_positions[symbol].quantity if symbol in live_positions else 0.0
        ref_qty = reference_positions[symbol].quantity if symbol in reference_positions else 0.0
        assert live_qty == ref_qty, f"{symbol}: live={live_qty} backtest={ref_qty}"
