"""The monthly paper-trading report: what a strategy's live cycle decided, sent, and
ended up holding in one calendar month, read straight from the journal.

Comparing this against what the backtester would have done over the same month (the
spec's "shadow" fidelity check) is Part 4's job, once real broker fills exist to compare
against; this only reports what actually happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qr.live import journal


@dataclass(frozen=True)
class MonthlyReport:
    """Everything journaled for ``strategy`` in one calendar month."""

    strategy: str
    month: str  # "YYYY-MM"
    decisions: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    alerts: pd.DataFrame


def build_report(strategy: str, month: str, data_dir: Path | str | None = None) -> MonthlyReport:
    """Everything journaled for ``strategy`` in calendar month ``month`` ("YYYY-MM")."""
    decisions = journal.read_decisions(strategy, data_directory=data_dir)
    if not decisions.empty:
        decisions = decisions[decisions["decision_date"].astype(str).str.startswith(month)]
    decision_ids = list(decisions["decision_id"])

    empty_orders = journal.read_orders("", data_directory=data_dir).iloc[0:0]
    empty_fills = journal.read_fills("", data_directory=data_dir).iloc[0:0]
    orders = (
        pd.concat(
            [journal.read_orders(did, data_directory=data_dir) for did in decision_ids],
            ignore_index=True,
        )
        if decision_ids
        else empty_orders
    )
    fills = (
        pd.concat(
            [journal.read_fills(did, data_directory=data_dir) for did in decision_ids],
            ignore_index=True,
        )
        if decision_ids
        else empty_fills
    )

    alerts = journal.read_alerts(data_directory=data_dir)
    if not alerts.empty:
        alerts = alerts[alerts["decision_id"].isin(decision_ids)]

    return MonthlyReport(
        strategy=strategy, month=month, decisions=decisions, orders=orders, fills=fills,
        alerts=alerts,
    )  # fmt: skip


def render_text(report: MonthlyReport) -> str:
    """A plain-text summary of a :class:`MonthlyReport`, for the terminal or a log file."""
    lines = [f"# {report.strategy} -- {report.month}", ""]
    if report.decisions.empty:
        lines.append("No decisions recorded this month.")
        return "\n".join(lines)

    for _, decision in report.decisions.iterrows():
        lines.append(f"## {decision['decision_id']}  [{decision['status']}]")
        # "equity" in the journal is the managed capital (LiveConfig.capital_limit) this
        # decision was sized against, not the paper account's total equity.
        capital = decision["equity"]
        if pd.notna(capital):
            lines.append(f"  managed capital: ${capital:,.2f}")
        weights = json.loads(decision["target_weights_json"])
        for symbol, weight in sorted(weights.items()):
            lines.append(f"  target {symbol}: {weight:.2%}")
        did = decision["decision_id"]
        did_orders = report.orders[report.orders["decision_id"] == did]
        for _, order in did_orders.iterrows():
            lines.append(
                f"  order {order['symbol']} {order['side']} {order['quantity']:g} "
                f"@ ~{order['planned_price']:.2f} [{order['status']}]"
            )
        did_fills = report.fills[report.fills["decision_id"] == did]
        for _, fill in did_fills.iterrows():
            lines.append(
                f"  fill {fill['symbol']} {fill['side']} {fill['quantity']:g} @ {fill['price']:.2f}"
            )
        lines.append("")

    if not report.alerts.empty:
        lines.append("## Alerts")
        for _, alert in report.alerts.iterrows():
            lines.append(f"  [{alert['severity']}] {alert['decision_id']}: {alert['message']}")

    return "\n".join(lines)
