"""The monthly report: built from the journal, filtered to one calendar month."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from qr.live import journal
from qr.live.report import build_report, render_text


def test_report_is_empty_for_a_month_with_no_decisions(tmp_path: Path) -> None:
    report = build_report("xs_momentum", "2023-01", data_dir=tmp_path)
    assert report.decisions.empty
    assert "No decisions recorded" in render_text(report)


def test_report_shows_managed_capital_weights_orders_and_fills(tmp_path: Path) -> None:
    did = journal.record_decision(
        "xs_momentum", date(2023, 1, 31), {"AAA": 0.6, "BBB": 0.4}, equity=10_000.0,
        data_directory=tmp_path,
    )  # fmt: skip
    journal.record_order("O1", did, "AAA", "buy", 60.0, 100.0, "filled", data_directory=tmp_path)
    journal.record_fill("O1", did, "AAA", "buy", 60.0, 100.0, 0.0, data_directory=tmp_path)

    report = build_report("xs_momentum", "2023-01", data_dir=tmp_path)
    assert len(report.decisions) == 1
    text = render_text(report)
    assert "managed capital: $10,000.00" in text
    assert "target AAA: 60.00%" in text
    assert "order AAA buy 60" in text
    assert "fill AAA buy 60" in text


def test_report_excludes_decisions_outside_the_month(tmp_path: Path) -> None:
    journal.record_decision(
        "xs_momentum", date(2023, 2, 28), {"AAA": 1.0}, equity=10_000.0, data_directory=tmp_path
    )
    report = build_report("xs_momentum", "2023-01", data_dir=tmp_path)
    assert report.decisions.empty


def test_report_includes_alerts_for_its_own_decisions_only(tmp_path: Path) -> None:
    did = journal.record_decision(
        "xs_momentum", date(2023, 1, 31), {"AAA": 1.0}, equity=10_000.0, data_directory=tmp_path
    )
    other = journal.record_decision(
        "xs_momentum", date(2023, 2, 28), {"AAA": 1.0}, equity=10_000.0, data_directory=tmp_path
    )
    journal.record_alert("stale prices", did=did, data_directory=tmp_path)
    journal.record_alert("kill switch engaged", did=other, data_directory=tmp_path)

    report = build_report("xs_momentum", "2023-01", data_dir=tmp_path)
    assert len(report.alerts) == 1
    assert "stale prices" in render_text(report)
    assert "kill switch" not in render_text(report)
