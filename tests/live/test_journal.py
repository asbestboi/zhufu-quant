"""The journal: idempotency, resumability, and every table's round trip."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from qr.live import journal


def test_decision_id_is_deterministic() -> None:
    assert journal.decision_id("etf_inverse_vol_momentum", date(2024, 1, 31)) == (
        "etf_inverse_vol_momentum:2024-01-31"
    )


def test_find_decision_on_an_empty_journal_is_none(tmp_path: Path) -> None:
    assert journal.find_decision("s", date(2024, 1, 31), data_directory=tmp_path) is None


def test_record_and_find_a_decision(tmp_path: Path) -> None:
    did = journal.record_decision(
        "s", date(2024, 1, 31), {"AAA": 0.5, "BBB": 0.5}, equity=10_000.0, cash_buffer=0.01,
        data_directory=tmp_path,
    )  # fmt: skip
    found = journal.find_decision("s", date(2024, 1, 31), data_directory=tmp_path)
    assert found is not None
    assert found.decision_id == did
    assert found.status == journal.PLANNED
    assert found.target_weights == {"AAA": 0.5, "BBB": 0.5}
    assert found.equity == 10_000.0


def test_recording_the_same_decision_twice_is_an_error(tmp_path: Path) -> None:
    journal.record_decision("s", date(2024, 1, 31), {}, data_directory=tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        journal.record_decision("s", date(2024, 1, 31), {}, data_directory=tmp_path)


def test_this_is_exactly_what_makes_a_caller_idempotent(tmp_path: Path) -> None:
    """The documented pattern: check find_decision() before ever calling record_decision()."""
    strategy, when = "s", date(2024, 1, 31)
    if journal.find_decision(strategy, when, data_directory=tmp_path) is None:
        journal.record_decision(strategy, when, {"AAA": 1.0}, data_directory=tmp_path)
    # A second "run" of the same decision does nothing new.
    if journal.find_decision(strategy, when, data_directory=tmp_path) is None:
        journal.record_decision(strategy, when, {"AAA": 1.0}, data_directory=tmp_path)
    assert journal.find_decision(strategy, when, data_directory=tmp_path) is not None


def test_decision_status_transitions(tmp_path: Path) -> None:
    did = journal.record_decision("s", date(2024, 1, 31), {}, data_directory=tmp_path)
    journal.set_decision_status(did, journal.SUBMITTED, data_directory=tmp_path)
    found = journal.find_decision("s", date(2024, 1, 31), data_directory=tmp_path)
    assert found is not None and found.status == journal.SUBMITTED
    journal.set_decision_status(did, journal.COMPLETED, data_directory=tmp_path)
    found = journal.find_decision("s", date(2024, 1, 31), data_directory=tmp_path)
    assert found is not None and found.status == journal.COMPLETED


def test_orders_and_fills_round_trip(tmp_path: Path) -> None:
    did = journal.record_decision("s", date(2024, 1, 31), {"AAA": 1.0}, data_directory=tmp_path)
    journal.record_order("SIM1", did, "AAA", "buy", 10.0, 100.0, "filled", data_directory=tmp_path)
    journal.record_fill("SIM1", did, "AAA", "buy", 10.0, 100.0, 0.0, data_directory=tmp_path)

    orders = journal.read_orders(did, data_directory=tmp_path)
    assert len(orders) == 1 and orders.iloc[0]["order_id"] == "SIM1"
    fills = journal.read_fills(did, data_directory=tmp_path)
    assert len(fills) == 1 and fills.iloc[0]["quantity"] == 10.0


def test_recording_the_same_fill_twice_is_a_no_op(tmp_path: Path) -> None:
    did = journal.record_decision("s", date(2024, 1, 31), {}, data_directory=tmp_path)
    journal.record_order("SIM1", did, "AAA", "buy", 10.0, 100.0, "filled", data_directory=tmp_path)
    journal.record_fill("SIM1", did, "AAA", "buy", 10.0, 100.0, 0.0, data_directory=tmp_path)
    journal.record_fill("SIM1", did, "AAA", "buy", 10.0, 100.0, 0.0, data_directory=tmp_path)
    assert len(journal.read_fills(did, data_directory=tmp_path)) == 1


def test_resubmitting_an_order_updates_status_without_duplicating(tmp_path: Path) -> None:
    did = journal.record_decision("s", date(2024, 1, 31), {}, data_directory=tmp_path)
    journal.record_order("SIM1", did, "AAA", "buy", 10.0, 100.0, "open", data_directory=tmp_path)
    journal.record_order("SIM1", did, "AAA", "buy", 10.0, 100.0, "filled", data_directory=tmp_path)
    orders = journal.read_orders(did, data_directory=tmp_path)
    assert len(orders) == 1 and orders.iloc[0]["status"] == "filled"


def test_positions_snapshot_round_trip(tmp_path: Path) -> None:
    journal.record_positions("run1", {"AAA": (10.0, 1_000.0)}, data_directory=tmp_path)
    positions = journal.read_positions("run1", data_directory=tmp_path)
    assert len(positions) == 1 and positions.iloc[0]["market_value"] == 1_000.0


def test_alerts_round_trip(tmp_path: Path) -> None:
    journal.record_alert("something went wrong", severity="error", data_directory=tmp_path)
    alerts = journal.read_alerts(data_directory=tmp_path)
    assert len(alerts) == 1 and alerts.iloc[0]["message"] == "something went wrong"


def test_runs_round_trip(tmp_path: Path) -> None:
    journal.record_run("R1", "live.yaml", dry_run=False, catch_up=False, data_directory=tmp_path)
    journal.finish_run("R1", "ok", message="done", data_directory=tmp_path)
    runs = journal.read_runs(data_directory=tmp_path)
    assert len(runs) == 1
    assert runs.iloc[0]["status"] == "ok"
    assert runs.iloc[0]["message"] == "done"


def test_reads_on_a_fresh_data_dir_return_empty_frames(tmp_path: Path) -> None:
    assert journal.read_orders("nope", data_directory=tmp_path).empty
    assert journal.read_fills("nope", data_directory=tmp_path).empty
    assert journal.read_positions("nope", data_directory=tmp_path).empty
    assert journal.read_alerts(data_directory=tmp_path).empty
    assert journal.read_runs(data_directory=tmp_path).empty
